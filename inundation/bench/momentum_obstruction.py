"""EA Test 3 benchmark: momentum conservation over a small obstruction.

Runs the UK Environment Agency "Benchmarking of 2D Hydraulic Modelling
Packages" Test 3 from the published May-2010 dataset
(``benchmark_assets/Test3 dataset 2010``): the georeferenced ASCII
DEM (``test3DEM.asc``, prismatic 1:200 slope with two depressions separated
by an obstruction) and the upstream inflow hydrograph (``Test3BC.csv``,
65.5 m^3/s plateau, 1310 m^3 total). Modelled area per the spec: x in
[0, 300] m by y in [0, 100] m, Manning n = 0.01 uniform, dry-bed initial
condition, run to t = 900 s. Gauges per the spec: Point 1 (150, 50) in the
first depression, Point 2 (250, 50) in the second.

The physics under test: whether the solver's momentum (inertia) terms carry
fast flow over a barrier it could never cross by water-surface gradient
alone. The inflow travels ~150 m down the 1:200 slope and arrives at the
first depression as a fast bore. By design the total inflow volume is *just
sufficient* to fill the first depression to the obstruction crest — an
inertia-free (diffusive-wave) model moves water strictly down surface
gradients, so at best it fills the depression and stops, and Point 2 stays
dry; only conserved momentum carries water over the crest, so any settled
pond at Point 2 is the momentum signature the published intercomparison
looks for.

One adaptation: this solver has no open-boundary inflow, so the hydrograph
is injected as near-boundary *volume* sources (a line of point sources
hugging the upstream wall). This is conservative for the discriminator — the
sources add mass with zero momentum, and all momentum is acquired on the
slope descent exactly as in the published setup (all other boundaries are
closed per the spec; this solver's walls are reflective).

Every backend run is paired with an untimed *still-water control*: the same
volume placed at rest in the first depression, filled to the crest (the
deepest lake any gradient-driven transport could build). A well-balanced
solver must keep the control's Point 2 dry — if it leaks over the crest from
rest, Point 2 ponding is a numerical well-balance artifact, not momentum,
and the case fails with an explicit reason.

There is no closed form (model-*intercomparison* benchmark); acceptance is
invariant/qualitative: mass balance against the injected hydrograph volume,
positivity / wetting-drying stability, a settled Point 1 pond, both pond
surfaces below the crest, and — the momentum signature — a measurable pond
at Point 2 past the obstruction.

Backends are configurable (`--backends`) across the retained Vulkan solver
implementations. The release is a pure initial condition with zero sources.
`--gif` renders untimed extra runs into a top-down depth-heatmap
animation and a longitudinal side-view (bed + water-surface profile)
animation; timed runs are never the frame source, so perf stays clean.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from inundation.bench.common import (
    build_channel_mesh,
    check_state_health,
    load_ascii_grid,
    nearest_cell_indices,
    resample_snapshots_uniform,
    save_depth_gif,
    save_faceted_line_plot,
    save_profile_gif,
    speed_from_momentum,
    volume_drift_rel,
    water_surface_elevation,
    write_mesh_parquet,
)
from inundation.tuning import MOMENTUM_SNAPSHOT_BACKENDS
from inundation.workflow import (
    PointSource,
    SimulationPhase,
    SWEWorkflow,
    WorkflowConfig,
    WorkflowResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Dataset files (May-2010 EA benchmark distribution, tracked in the repo).
DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parents[2] / "benchmark_assets" / "Test3 dataset 2010"
)
DEM_FILENAME = "test3DEM.asc"
BC_FILENAME = "Test3BC.csv"

# Modelled area per the spec: a perfect rectangle x in [0, 300] m by y in
# [0, 100] m (the DEM raster carries an apron beyond it, x in [-50, 322]).
# Mesh coordinates coincide with the DEM's georeference, so gauge locations
# are used as published. All boundaries are closed (reflective walls); the
# published inflow enters along the upstream (x = 0) edge.
CHANNEL_LEN_M = 300.0
CHANNEL_WIDTH_M = 100.0
NX_DEFAULT = 150  # dx = 2 m (native DEM resolution)
NY_DEFAULT = 50  # dy = 2 m

POINT1_X_M = 150.0  # first depression centre (published gauge: (150, 50))
CREST_X_M = 200.0  # obstruction crest between the depressions (z ~ 10.0)
POINT2_X_M = 250.0  # second depression centre (published gauge: (250, 50))

# Manning's n per the spec: 0.01 uniform.
MANNING_N = 0.01

# Inflow adaptation: the published hydrograph is an upstream *boundary*
# inflow; this solver has no open boundaries, so the discharge is injected
# as a line of near-boundary volume sources hugging the x = 0 wall (equal
# split, area-weighted within each circle). Volume sources carry no momentum
# vector — the flood wave acquires all momentum on the 1:200 slope descent,
# as in the published setup.
INFLOW_X_M = 2.0
INFLOW_RADIUS_M = 10.0
INFLOW_CENTERS_Y_M = (10.0, 30.0, 50.0, 70.0, 90.0)
# The hydrograph ramps are piecewise-linear with breakpoints on whole
# seconds, so 1 s piecewise-constant phases sampled at midpoints integrate
# the published curve exactly (total 1310 m^3).
INFLOW_PHASE_S = 1.0

T_END_S = 900.0  # spec: run to t = 15 min; integer -> float32-exact stop

DT_MAX_DEFAULT = 2.0
DT_INIT_DEFAULT = 1e-2
CFL_INTERVAL_DEFAULT = 10

# ── Correctness gates (invariant / qualitative; no closed-form reference) ───
# Closed domain (reflective walls) + a volume-conserving source schedule: the
# final volume must match the injected hydrograph volume to float32
# accumulation error. This is the anchor gate. The dam-break bench holds 1e-5
# over a 20 s frictionless run; this case runs 900 s (~10k steps) of
# friction + wetting/drying, so the gate is 5e-5 (same "round-off only"
# spirit, ~3x headroom over the release-variant's measured ~1.7e-5).
GATE_VOLUME_DRIFT_REL = 5e-5
# The first depression (Point 1) must catch and retain a settled pond.
GATE_POINT1_PONDED_M = 0.05
# Disconnected-reservoir signature: both pond *surfaces* must settle at least
# this far below the obstruction crest elevation. The two depressions are
# then hydraulically isolated (no continuous water body spans the crest), so
# any water in the second must have been carried *over* the crest by the
# bore. Dryness is asserted via water-surface elevation rather than the crest
# cell's raw depth, so a residual film there could not mask a connection (on
# the published dataset the crest in fact drains to exactly 0). The dataset
# fills the first depression to the brim by design (inflow 1310 m^3 vs
# ~1309 m^3 capacity below the crest), so Point 1 settles below the crest
# only by the volume that overtopped (~9 mm measured on gpu_resident_batch);
# the margin must sit below that design freeboard (5 mm ~= 1.8x headroom).
GATE_PONDS_BELOW_CREST_M = 0.005
# Momentum signature: Point 2 (second depression, past the dry crest) must
# hold at least this depth at t_end. An inertia-free model can at best fill
# the first depression flush with the crest and stop; only conserved
# momentum puts water here.
GATE_POINT2_MIN_DEPTH_M = 0.02
# Still-water control: the same volume placed *at rest* in the first
# depression, filled to the crest (the deepest lake any inertia-free,
# gradient-driven transport could build) must leave Point 2 essentially dry.
# If the control wets Point 2, water is crossing the crest without momentum
# — a numerical well-balance leak — and the inflow run's Point 2 pond proves
# nothing. This control is what makes the momentum gate meaningful.
GATE_CONTROL_POINT2_MAX_M = 0.005
# Repeat-to-repeat reproducibility (relative to peak depth). GPU atomicAdd flux
# scatter is not bit-reproducible; the device-side CFL reductions accumulate a
# run-to-run drift observed up to ~6e-4 over runs of this length on MoltenVK,
# so the gate carries ~3x headroom above the observed tail.
GATE_DETERMINISM_REL = 2e-3

DEFAULT_OUTPUT_ROOT = ".tmp/momentum-obstruction-bench"
# Default to the fully GPU-resident backend. Other retained Vulkan strategies
# remain selectable through --backends.
DEFAULT_BACKENDS = ("gpu_resident_batch",)
BackendImpl = Literal[
    "fixed_dt_batch_barrier",
    "gpu_resident_batch",
]
AVAILABLE_BACKENDS: tuple[BackendImpl, ...] = (
    "fixed_dt_batch_barrier",
    "gpu_resident_batch",
)
GIF_FRAMES_DEFAULT = 60
GIF_VMAX_M = 0.8


@dataclass(frozen=True)
class ObstructionSpec:
    """Geometry + dataset parameters for the momentum-obstruction case."""

    name: str
    nx: int
    ny: int
    channel_len_m: float
    channel_width_m: float
    manning_n: float
    t_end_s: float
    dt_max: float
    cfl_interval: int
    dataset_dir: Path
    cfl: float | None = None

    @property
    def dx_m(self) -> float:
        return self.channel_len_m / self.nx

    @property
    def dy_m(self) -> float:
        return self.channel_width_m / self.ny

    @property
    def dem_path(self) -> Path:
        return self.dataset_dir / DEM_FILENAME

    @property
    def bc_path(self) -> Path:
        return self.dataset_dir / BC_FILENAME

    @property
    def point1_xy(self) -> tuple[float, float]:
        return (POINT1_X_M, self.channel_width_m / 2.0)

    @property
    def point2_xy(self) -> tuple[float, float]:
        return (POINT2_X_M, self.channel_width_m / 2.0)


@dataclass(frozen=True)
class CaseMetrics:
    """Invariant/qualitative metrics + gate outcomes for one (case, backend)."""

    case: str
    backend: str
    volume_drift_rel: float
    min_depth_m: float
    crest_z_m: float
    point1_depth_m: float
    point1_wse_m: float
    crest_depth_m: float
    point2_depth_m: float
    point2_wse_m: float
    control_point2_depth_m: float
    left_ponded: bool
    ponds_disconnected: bool
    point2_risen: bool
    control_dry: bool
    h_final_finite: bool
    passed: bool
    fail_reasons: list[str] = field(default_factory=list[str])


@dataclass(frozen=True)
class PerfMetrics:
    """Timing summary for one (case, backend) group of timed repeats."""

    case: str
    backend: str
    repeats: int
    median_wall_s: float
    steps_total: int
    steps_per_s: float
    cell_steps_per_s: float
    deterministic: bool | None  # None when repeats < 2
    determinism_detail: str


def _build_spec(args: argparse.Namespace) -> ObstructionSpec:
    return ObstructionSpec(
        name="obstruction",
        nx=args.nx,
        ny=args.ny,
        channel_len_m=CHANNEL_LEN_M,
        channel_width_m=CHANNEL_WIDTH_M,
        manning_n=MANNING_N,
        t_end_s=args.t_end,
        dt_max=args.dt_max,
        cfl_interval=args.cfl_interval,
        cfl=args.cfl,
        dataset_dir=Path(args.dataset_dir),
    )


def _load_dem_profile(dem_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parse the ESRI ASCII DEM; return (cell-center x, bed z) of the profile.

    The published raster is prismatic (every row identical), which the
    harness relies on for the 1-D hypsometry and side-profile rendering —
    verified here rather than assumed.
    """
    grid = load_ascii_grid(dem_path)
    if not bool(np.isfinite(grid.z).all()):
        raise ValueError("DEM contains nodata cells")
    if not bool(np.all(grid.z == grid.z[0])):
        raise ValueError("DEM is not prismatic (rows differ)")
    return grid.x, grid.z[0].copy()


def _load_hydrograph(bc_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parse the inflow hydrograph CSV; return (time [s], discharge [m^3/s])."""
    data = np.loadtxt(bc_path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    t, q = data[:, 0], data[:, 1]
    if bool(np.any(np.diff(t) <= 0.0)):
        raise ValueError("hydrograph times must be strictly increasing")
    if bool(np.any(q < 0.0)) or q[-1] != 0.0:
        raise ValueError("hydrograph must be non-negative and end at zero inflow")
    return t, q


def _inflow_sources(discharge_m3s: float) -> list[PointSource]:
    """The upstream line inflow as equal-split near-boundary volume sources."""
    per_source = discharge_m3s / len(INFLOW_CENTERS_Y_M)
    return [PointSource(per_source, (INFLOW_X_M, y), INFLOW_RADIUS_M) for y in INFLOW_CENTERS_Y_M]


def _hydrograph_phases(spec: ObstructionSpec) -> list[SimulationPhase]:
    """Published hydrograph as 1 s piecewise-constant phases + settle tail.

    Midpoint sampling of the piecewise-linear curve is volume-exact because
    the CSV breakpoints land on whole seconds (no phase straddles a kink).
    """
    t, q = _load_hydrograph(spec.bc_path)
    active_end_s = float(t[int(np.flatnonzero(q > 0.0).max()) + 1])
    n_active = round(active_end_s / INFLOW_PHASE_S)
    phases: list[SimulationPhase] = []
    for k in range(n_active):
        t_mid = (k + 0.5) * INFLOW_PHASE_S
        q_k = float(np.interp(t_mid, t, q))
        sources = _inflow_sources(q_k) if q_k > 0.0 else []
        phases.append(SimulationPhase(duration_s=INFLOW_PHASE_S, sources=sources))
    settle_s = spec.t_end_s - n_active * INFLOW_PHASE_S
    if settle_s < 0.0:
        raise ValueError(f"t_end {spec.t_end_s} shorter than the hydrograph ({active_end_s} s)")
    phases.append(SimulationPhase(duration_s=settle_s, sources=[]))
    return phases


def _injected_volume(spec: ObstructionSpec) -> float:
    """Total hydrograph volume [m^3] as the phase schedule integrates it."""
    return float(
        sum(
            p.duration_s * sum(s.discharge_m3s for s in p.sources) for p in _hydrograph_phases(spec)
        )
    )


def _original_cell_centroids(
    spec: ObstructionSpec,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cell centers in generation (pre-Hilbert) order, matching write order."""
    n = spec.nx * spec.ny
    idx = np.arange(n)
    cx = (idx % spec.nx + 0.5) * spec.dx_m
    cy = (idx // spec.nx + 0.5) * spec.dy_m
    return cx.astype(np.float64), cy.astype(np.float64)


def _bed_original(spec: ObstructionSpec) -> NDArray[np.float64]:
    """Bed elevation at cell centers in original (generation) order."""
    dem_x, dem_z = _load_dem_profile(spec.dem_path)
    cx, _cy = _original_cell_centroids(spec)
    if cx.min() < dem_x[0] or cx.max() > dem_x[-1]:
        raise ValueError("mesh extends beyond the DEM coverage")
    return np.interp(cx, dem_x, dem_z)


def _crest_z(spec: ObstructionSpec) -> float:
    """Obstruction crest elevation: bed maximum between the two gauges."""
    cx, _cy = _original_cell_centroids(spec)
    zb = _bed_original(spec)
    between = (cx > POINT1_X_M) & (cx < POINT2_X_M)
    return float(zb[between].max())


def _ensure_mesh(spec: ObstructionSpec, output_dir: Path) -> Path:
    """Write the DEM-sampled mesh over the modelled area (idempotent)."""
    # "dem2010" = published May-2010 raster; bump if the sampling changes so
    # a stale cached parquet from an older bed is never silently reused.
    mesh_path = output_dir / f"obstruction-dem2010-{spec.nx}x{spec.ny}.parquet"
    if not mesh_path.exists():
        verts, quads = build_channel_mesh(
            spec.nx, spec.ny, spec.channel_len_m, spec.channel_width_m
        )
        zb = _bed_original(spec).astype(np.float32)
        write_mesh_parquet(mesh_path, verts, quads, zb)
        print(f"[mesh] DEM-sampled mesh written: {mesh_path} (N={spec.nx * spec.ny})")
    return mesh_path


def _initial_state(spec: ObstructionSpec) -> dict[str, object]:
    """Dry-bed initial condition (per the spec) in original cell order."""
    zeros = np.zeros(spec.nx * spec.ny, dtype=np.float32)
    return {"h": zeros.copy(), "hu": zeros.copy(), "hv": zeros.copy()}


def _control_wse(spec: ObstructionSpec) -> float:
    """Water-surface elevation holding the control volume at rest in Point 1.

    Solved by bisection on the depression's hypsometry (cells below the
    crest, left of the obstruction). The dataset sizes the inflow to *just*
    fill the depression (1310 m^3 vs ~1309 m^3 capacity), so the control
    volume is capped at the capacity and the surface tops out at the crest —
    the deepest lake any gradient-driven transport could build.
    """
    cx, _cy = _original_cell_centroids(spec)
    zb = _bed_original(spec)
    crest = _crest_z(spec)
    valley = (zb < crest) & (cx < CREST_X_M)
    cell_area = spec.dx_m * spec.dy_m

    def capacity(w: float) -> float:
        return float(np.clip(w - zb[valley], 0.0, None).sum() * cell_area)

    target = min(_injected_volume(spec), capacity(crest))
    lo, hi = float(zb[valley].min()), crest
    if capacity(hi) <= target:
        return crest
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if capacity(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _control_initial_state(spec: ObstructionSpec) -> dict[str, object]:
    """Inertia-free endpoint: the inflow volume already at rest in Point 1.

    This is the deepest lake any gradient-driven (diffusive-wave) transport
    of the inflow could ever build against the obstruction. Started at rest,
    a well-balanced solver must keep the crest and Point 2 dry; any water
    appearing past the crest is a numerical well-balance leak, not momentum.
    """
    cx, _cy = _original_cell_centroids(spec)
    zb = _bed_original(spec)
    wse = _control_wse(spec)
    crest = _crest_z(spec)
    valley = (zb < crest) & (cx < CREST_X_M)
    h0 = np.where(valley, np.clip(wse - zb, 0.0, None), 0.0).astype(np.float32)
    zeros = np.zeros(spec.nx * spec.ny, dtype=np.float32)
    return {"h": h0, "hu": zeros.copy(), "hv": zeros.copy()}


def _make_workflow(
    spec: ObstructionSpec,
    mesh_path: Path,
    backend: BackendImpl,
    *,
    output_interval_s: float,
    progress: bool,
    initial_state: dict[str, object] | None = None,
    capture_momentum_snapshots: bool = False,
) -> SWEWorkflow:
    config = WorkflowConfig(
        mesh_source=str(mesh_path),
        manning_n=spec.manning_n,
        output_interval_s=output_interval_s,
        dt_max=spec.dt_max,
        dt_init=DT_INIT_DEFAULT,
        cfl_interval=spec.cfl_interval,
        cfl=spec.cfl,
        progress=progress,
        capture_momentum_snapshots=capture_momentum_snapshots,
        initial_state_source=initial_state if initial_state is not None else _initial_state(spec),
        initial_state_order="original",
        solver_impl=backend,
    )
    workflow = SWEWorkflow(config)
    workflow.prepare()
    return workflow


def _run_once(
    spec: ObstructionSpec,
    mesh_path: Path,
    backend: BackendImpl,
    *,
    output_interval_s: float,
    progress: bool,
    initial_state: dict[str, object] | None = None,
    capture_momentum_snapshots: bool = False,
) -> tuple[SWEWorkflow, WorkflowResult]:
    workflow = _make_workflow(
        spec,
        mesh_path,
        backend,
        output_interval_s=output_interval_s,
        progress=progress,
        initial_state=initial_state,
        capture_momentum_snapshots=capture_momentum_snapshots,
    )
    phases = (
        _hydrograph_phases(spec)
        if initial_state is None
        else [SimulationPhase(duration_s=spec.t_end_s, sources=[])]
    )
    result = workflow.run(phases)
    return workflow, result


def _nearest_cell(workflow: SWEWorkflow, xy: tuple[float, float]) -> int:
    """Index of the solver cell whose centroid is nearest to ``xy``."""
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")
    return int(
        nearest_cell_indices(
            workflow.geom.world_centroid[:, 0],
            workflow.geom.world_centroid[:, 1],
            np.asarray([xy], dtype=np.float64),
        )[0]
    )


def _evaluate_case(
    spec: ObstructionSpec,
    backend: str,
    workflow: SWEWorkflow,
    result: WorkflowResult,
    control_point2_depth_m: float,
) -> CaseMetrics:
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")

    h_final = np.asarray(result.h_final, dtype=np.float32)
    zb = np.asarray(workflow.geom.zb, dtype=np.float32)
    finite, min_depth, fail_reasons = check_state_health(h_final, min_depth_tol=-1e-6)

    # Mass balance: dry start + volume-conserving sources, so the final
    # volume must equal the injected hydrograph volume (mirrors
    # bench/floodplain_depressions.py).
    injected = _injected_volume(spec)
    drift_rel = volume_drift_rel(result.volume_final_m3, injected)
    crest_z = _crest_z(spec)

    c1 = _nearest_cell(workflow, spec.point1_xy)
    cc = _nearest_cell(workflow, (CREST_X_M, spec.channel_width_m / 2.0))
    c2 = _nearest_cell(workflow, spec.point2_xy)
    point1_depth = float(h_final[c1])
    crest_depth = float(h_final[cc])
    point2_depth = float(h_final[c2])
    point1_wse = float(zb[c1] + h_final[c1])
    point2_wse = float(zb[c2] + h_final[c2])

    left_ponded = point1_depth >= GATE_POINT1_PONDED_M
    # Disconnected reservoirs: both pond surfaces settle below the crest.
    ponds_disconnected = max(point1_wse, point2_wse) <= crest_z - GATE_PONDS_BELOW_CREST_M
    point2_risen = point2_depth >= GATE_POINT2_MIN_DEPTH_M
    control_dry = control_point2_depth_m <= GATE_CONTROL_POINT2_MAX_M

    if drift_rel > GATE_VOLUME_DRIFT_REL:
        fail_reasons.append(f"volume_drift_rel:{drift_rel:.3e}>{GATE_VOLUME_DRIFT_REL:.0e}")
    if not left_ponded:
        fail_reasons.append(
            f"point1_depth:{point1_depth:.3f}<{GATE_POINT1_PONDED_M} (depression not ponded)"
        )
    if not ponds_disconnected:
        max_wse = max(point1_wse, point2_wse)
        fail_reasons.append(
            f"max_pond_wse:{max_wse:.3f}>{crest_z - GATE_PONDS_BELOW_CREST_M:.3f} "
            f"(ponds not disconnected below crest)"
        )
    if not point2_risen:
        fail_reasons.append(
            f"point2_depth:{point2_depth:.3f}<{GATE_POINT2_MIN_DEPTH_M} (no overtopping)"
        )
    if not control_dry:
        fail_reasons.append(
            f"control_point2_depth:{control_point2_depth_m:.3f}>{GATE_CONTROL_POINT2_MAX_M} "
            f"(well-balance leak: still water crosses the crest without momentum, "
            f"so point2_depth is not a momentum signature)"
        )

    return CaseMetrics(
        case=spec.name,
        backend=backend,
        volume_drift_rel=drift_rel,
        min_depth_m=min_depth,
        crest_z_m=crest_z,
        point1_depth_m=point1_depth,
        point1_wse_m=point1_wse,
        crest_depth_m=crest_depth,
        point2_depth_m=point2_depth,
        point2_wse_m=point2_wse,
        control_point2_depth_m=control_point2_depth_m,
        left_ponded=left_ponded,
        ponds_disconnected=ponds_disconnected,
        point2_risen=point2_risen,
        control_dry=control_dry,
        h_final_finite=finite,
        passed=not fail_reasons,
        fail_reasons=fail_reasons,
    )


def _safe_gif_frames(frames: int) -> int:
    """Nearest power of two >= 2, so t_end / frames is float32-exact."""
    n = max(2, frames)
    lo = 1 << (n.bit_length() - 1)
    hi = lo << 1
    return lo if (n - lo) <= (hi - n) else hi


def _run_group(
    spec: ObstructionSpec,
    mesh_path: Path,
    backend: BackendImpl,
    args: argparse.Namespace,
) -> tuple[CaseMetrics, PerfMetrics]:
    n_cells = spec.nx * spec.ny
    output_interval_s = args.output_interval_s or spec.t_end_s

    for w in range(args.warmup):
        print(f"[{spec.name}/{backend}] warmup {w + 1}/{args.warmup}")
        _run_once(
            spec,
            mesh_path,
            backend,
            output_interval_s=output_interval_s,
            progress=args.solver_progress,
        )

    walls: list[float] = []
    final_states: list[NDArray[np.float64]] = []
    last: tuple[SWEWorkflow, WorkflowResult] | None = None
    for r in range(args.repeats):
        print(f"[{spec.name}/{backend}] timed run {r + 1}/{args.repeats}")
        t0 = time.perf_counter()
        workflow, result = _run_once(
            spec,
            mesh_path,
            backend,
            output_interval_s=output_interval_s,
            progress=args.solver_progress,
        )
        walls.append(time.perf_counter() - t0)
        final_states.append(np.asarray(result.h_final, dtype=np.float64))
        last = (workflow, result)

    if last is None:
        raise RuntimeError("--repeats must be >= 1")
    workflow, result = last

    # Untimed still-water control (inertia-free endpoint at rest): must not
    # wet Point 2, or Point 2 ponding cannot be attributed to momentum.
    print(f"[{spec.name}/{backend}] still-water control run (untimed)")
    control_workflow, control_result = _run_once(
        spec,
        mesh_path,
        backend,
        output_interval_s=output_interval_s,
        progress=args.solver_progress,
        initial_state=_control_initial_state(spec),
    )
    control_h = np.asarray(control_result.h_final, dtype=np.float32)
    control_p2 = float(control_h[_nearest_cell(control_workflow, spec.point2_xy)])

    case_metrics = _evaluate_case(spec, backend, workflow, result, control_p2)

    deterministic: bool | None = None
    determinism_detail = "skipped (repeats=1)"
    if len(final_states) >= 2:
        scale = max(float(np.abs(final_states[0]).max()), 1e-12)
        max_rel_diff = max(
            float(np.abs(state - final_states[0]).max()) / scale for state in final_states[1:]
        )
        deterministic = max_rel_diff <= GATE_DETERMINISM_REL
        determinism_detail = (
            f"max rel diff {max_rel_diff:.2e} <= {GATE_DETERMINISM_REL:.0e}"
            if deterministic
            else f"max rel diff {max_rel_diff:.2e} > {GATE_DETERMINISM_REL:.0e}"
        )

    median_wall_s = statistics.median(walls)
    steps_total = int(result.steps_total)
    perf = PerfMetrics(
        case=spec.name,
        backend=backend,
        repeats=args.repeats,
        median_wall_s=median_wall_s,
        steps_total=steps_total,
        steps_per_s=steps_total / median_wall_s if median_wall_s > 0 else 0.0,
        cell_steps_per_s=(steps_total * n_cells) / median_wall_s if median_wall_s > 0 else 0.0,
        deterministic=deterministic,
        determinism_detail=determinism_detail,
    )

    if args.gif and backend == args.gif_backend:
        safe_frames = _safe_gif_frames(args.gif_frames)
        if safe_frames != args.gif_frames:
            print(f"[{spec.name}/{backend}] gif frames {args.gif_frames} -> {safe_frames}")
        print(f"[{spec.name}/{backend}] extra untimed gif run ({safe_frames} frames)")
        gif_workflow, gif_result = _run_once(
            spec,
            mesh_path,
            backend,
            output_interval_s=spec.t_end_s / safe_frames,
            progress=args.solver_progress,
            capture_momentum_snapshots=backend in MOMENTUM_SNAPSHOT_BACKENDS,
        )
        cells = np.asarray(
            [
                _nearest_cell(gif_workflow, spec.point1_xy),
                _nearest_cell(gif_workflow, spec.point2_xy),
            ],
            dtype=np.int64,
        )
        if gif_workflow.geom is None:
            raise RuntimeError("workflow must be prepared")
        save_faceted_line_plot(
            gif_result.snap_times,
            water_surface_elevation(gif_result.snapshots, gif_workflow.geom.zb, cells),
            ["1", "2"],
            Path(args.output_dir) / f"test3-water-levels-{backend}.png",
            ylabel="Water-surface elevation [m]",
            title="Test 3 obstruction water levels",
        )
        speed = speed_from_momentum(
            gif_result.hu_snapshots, gif_result.hv_snapshots, gif_result.snapshots, cells
        )
        if speed is not None:
            save_faceted_line_plot(
                gif_result.snap_times,
                speed,
                ["1", "2"],
                Path(args.output_dir) / f"test3-velocities-{backend}.png",
                ylabel="Speed [m/s]",
                title="Test 3 obstruction velocities",
            )
        else:
            print(
                f"[{spec.name}/{backend}] WARNING: omitted test3-velocities-{backend}.png; momentum snapshots unavailable"
            )
        gif_snapshots, gif_times = resample_snapshots_uniform(
            gif_result.snapshots, gif_result.snap_times, safe_frames
        )
        save_depth_gif(
            gif_workflow,
            gif_snapshots,
            gif_times,
            Path(args.output_dir) / f"{spec.name}-{backend}-2d.gif",
            vmax=GIF_VMAX_M,
            title_prefix=f"{spec.name} — ",
        )
        save_profile_gif(
            gif_workflow,
            gif_snapshots,
            gif_times,
            Path(args.output_dir) / f"{spec.name}-{backend}-side.gif",
            vmax=GIF_VMAX_M,
            title_prefix=f"{spec.name} — ",
        )

    return case_metrics, perf


def _print_report(all_metrics: list[CaseMetrics], all_perf: list[PerfMetrics]) -> None:
    print()
    print(
        f"gates: volume_drift<={GATE_VOLUME_DRIFT_REL:.0e} min_depth>=0 finite "
        f"point1_ponded>={GATE_POINT1_PONDED_M} ponds_below_crest>={GATE_PONDS_BELOW_CREST_M} "
        f"point2_depth>={GATE_POINT2_MIN_DEPTH_M} control_p2<={GATE_CONTROL_POINT2_MAX_M}"
    )
    print(
        f"{'case':12} {'backend':24} {'vol_drift':>9} {'min_h':>10} {'p1_depth':>9} "
        f"{'crest_h':>8} {'p2_depth':>9} {'ctrl_p2':>8} {'p1_pond':>8} {'disconn':>8} "
        f"{'p2_rise':>8} {'ctrl_dry':>8} {'pass':>5}"
    )
    for m in all_metrics:
        print(
            f"{m.case:12} {m.backend:24} {m.volume_drift_rel:9.3e} {m.min_depth_m:10.3e} "
            f"{m.point1_depth_m:9.3f} {m.crest_depth_m:8.3f} {m.point2_depth_m:9.3f} "
            f"{m.control_point2_depth_m:8.3f} {m.left_ponded!s:>8} {m.ponds_disconnected!s:>8} "
            f"{m.point2_risen!s:>8} {m.control_dry!s:>8} {m.passed!s:>5}"
        )
        for reason in m.fail_reasons:
            print(f"  !! {reason}")
        print(
            f"  Point 1 WSE {m.point1_wse_m:.3f} m, Point 2 WSE {m.point2_wse_m:.3f} m "
            f"(crest {m.crest_z_m:.3f} m)"
        )
    print()
    print(
        f"{'case':12} {'backend':24} {'wall_s':>8} {'steps':>8} {'steps/s':>10} "
        f"{'cell-steps/s':>13} {'determ':>7}"
    )
    for p in all_perf:
        det = "-" if p.deterministic is None else str(p.deterministic)
        print(
            f"{p.case:12} {p.backend:24} {p.median_wall_s:8.3f} {p.steps_total:8d} "
            f"{p.steps_per_s:10.1f} {p.cell_steps_per_s:13.3e} {det:>7}"
        )
        if p.deterministic is False:
            print(f"  !! {p.determinism_detail}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EA Test 3 momentum-conservation-over-an-obstruction benchmark"
    )
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help=f"CSV of solver_impl names {AVAILABLE_BACKENDS} (default: Vulkan reference)",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--nx", type=int, default=NX_DEFAULT)
    parser.add_argument("--ny", type=int, default=NY_DEFAULT)
    parser.add_argument("--dt-max", type=float, default=DT_MAX_DEFAULT)
    parser.add_argument("--cfl-interval", type=int, default=CFL_INTERVAL_DEFAULT)
    parser.add_argument("--cfl", type=float, default=None, help="CFL number")
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="directory holding test3DEM.asc and Test3BC.csv (published EA dataset)",
    )
    parser.add_argument("--t-end", type=float, default=T_END_S)
    parser.add_argument(
        "--output-interval-s",
        type=float,
        default=None,
        help="snapshot cadence for timed runs (default: t_end, i.e. one final snapshot)",
    )
    parser.add_argument("--repeats", type=int, default=2, help=">=2 enables determinism check")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--solver-progress", action="store_true")
    parser.add_argument("--gif", action="store_true", help="render untimed top-down 2D depth gif")
    parser.add_argument(
        "--gif-backend", default=None, help="backend for the gif run (default: first of --backends)"
    )
    parser.add_argument("--gif-frames", type=int, default=GIF_FRAMES_DEFAULT)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    raw_backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for backend_name in raw_backends:
        if backend_name not in AVAILABLE_BACKENDS:
            print(f"unknown backend {backend_name!r}; choose from {AVAILABLE_BACKENDS}")
            return 2
    backends: list[BackendImpl] = [cast("BackendImpl", b) for b in raw_backends]
    if args.gif_backend is None:
        args.gif_backend = backends[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = _build_spec(args)
    mesh_path = _ensure_mesh(spec, output_dir)
    t_bc, q_bc = _load_hydrograph(spec.bc_path)
    print(
        f"[inflow] published hydrograph {spec.bc_path.name}: peak {q_bc.max():.1f} m^3/s, "
        f"active {t_bc[int(np.flatnonzero(q_bc > 0.0).max()) + 1]:.0f} s, total "
        f"{_injected_volume(spec):.0f} m^3; t_end {spec.t_end_s:.0f} s"
    )

    all_metrics: list[CaseMetrics] = []
    all_perf: list[PerfMetrics] = []
    for backend in backends:
        metrics, perf = _run_group(spec, mesh_path, backend, args)
        all_metrics.append(metrics)
        all_perf.append(perf)

    _print_report(all_metrics, all_perf)

    summary = {
        "gates": {
            "volume_drift_rel": GATE_VOLUME_DRIFT_REL,
            "point1_ponded_m": GATE_POINT1_PONDED_M,
            "ponds_below_crest_m": GATE_PONDS_BELOW_CREST_M,
            "point2_min_depth_m": GATE_POINT2_MIN_DEPTH_M,
            "control_point2_max_m": GATE_CONTROL_POINT2_MAX_M,
            "determinism_rel": GATE_DETERMINISM_REL,
        },
        "injected_volume_m3": _injected_volume(spec),
        "correctness": [asdict(m) for m in all_metrics],
        "performance": [asdict(p) for p in all_perf],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[summary] {summary_path}")

    correctness_ok = all(m.passed for m in all_metrics)
    determinism_ok = all(p.deterministic is not False for p in all_perf)
    if correctness_ok and determinism_ok:
        print("RESULT: PASS")
        return 0
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
