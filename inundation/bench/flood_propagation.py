"""EA Test 4 benchmark: speed of flood propagation over an extended floodplain.

Runs the UK Environment Agency "Benchmarking of 2D Hydraulic Modelling
Packages" Test 4 from the published May-2010 dataset
(``benchmark_assets/Test4 dataset 2010``): the inflow hydrograph
(``Test4BC.csv``, peak 20 m^3/s over a ~5 h base, 285 000 m^3 total) and the
six published output points (``Test4output.csv``). The dataset ships **no
DEM** — per the spec the floodplain is horizontal at elevation 0 — so the bed
is flat by definition rather than by approximation. Per the spec: a
1000 m (E-W) x 2000 m (N-S) plain, 5 m model resolution (~80 000 nodes),
Manning n = 0.05 uniform, dry-bed initial condition, all boundaries closed,
inflow along a 20 m line at the middle of the western side, run to t = 5 h.

The physics under test is the *celerity* of the advancing flood front, plus
the transient depths and velocities at its leading edge — a wetting/drying
front-tracking problem rather than a ponding or shock-capture one.

Unlike the other EA cases this one admits a genuine reference. The inflow
enters on a closed wall, so by reflection the half-plane problem is identical
to a full-plane axisymmetric one at twice the discharge, and the published
gauges are laid out for exactly that: points 1-5 sit on the axis at radii
50-400 m from the source, and point 6 sits at 424.26 m on the 45° diagonal,
making it a grid-orientation isotropy probe. ``solve_radial_inflow``
(``bench/common.py``) integrates that 1-D problem on a fine grid with the
same friction law, and the harness scores front-arrival times, depths and
speeds against it.

The reference is only valid while the front is clear of the walls. All three
nearest boundaries are 1000 m from the source and the front reaches them at
~189 min, so profile comparisons are taken at ``--probe-time`` (default 2 h,
front at ~720 m) and the full 5 h run is scored on the invariants alone.

Backends are configurable (`--backends`) across the retained Vulkan solver
implementations. The inflow uses piecewise-constant phase source rates, so the
default `gpu_resident_batch` path remains device-resident. `--gif`
renders an *untimed* extra run into a top-down depth-heatmap animation; timed
runs are never the frame source, so perf stays clean.
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
    RadialInflowSolution,
    build_channel_mesh,
    check_state_health,
    l1_relative_error,
    max_relative_error,
    nearest_cell_indices,
    resample_snapshots_uniform,
    save_cross_section,
    save_depth_contours,
    save_depth_gif,
    save_faceted_line_plot,
    solve_radial_inflow,
    speed_from_momentum,
    volume_drift_rel,
    water_surface_elevation,
    write_mesh_parquet,
)
from inundation.tuning import GRAVITY_G, MOMENTUM_SNAPSHOT_BACKENDS
from inundation.workflow import (
    PointSource,
    SimulationPhase,
    SWEWorkflow,
    WorkflowConfig,
    WorkflowResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ── Case geometry / physics (published dataset; see _build_parser) ──────────
# Dataset files (May-2010 EA benchmark distribution, tracked in the repo).
DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parents[2] / "benchmark_assets" / "Test4 dataset 2010"
)
BC_FILENAME = "Test4BC.csv"
GAUGES_FILENAME = "Test4output.csv"

# Spec geometry: 1000 m west-east by 2000 m south-north, horizontal at z = 0.
# The gauge layout fixes the orientation — the published points run x = 50..400
# at y = 1000, which is the middle of the western side.
DOMAIN_X_M = 1000.0
DOMAIN_Y_M = 2000.0
NX_DEFAULT = 200  # dx = 5 m (spec: 5 m grid, ~80000 nodes)
NY_DEFAULT = 400

MANNING_N = 0.05
T_END_S = 5.0 * 3600.0  # spec: run to t = 5 h; integer -> float32-exact stop

# Inflow adaptation: the published boundary condition is a 20 m line at the
# middle of the western wall. This solver has no open boundaries, so the
# discharge enters as an equal split of volume sources on the first cell
# column over that stretch. SOURCE_EQUIV_RADIUS_M is the disc the reference
# spreads the same discharge over, so both see the same near-field.
INFLOW_LINE_Y_M = (990.0, 1010.0)
INFLOW_SOURCES = 4
INFLOW_RADIUS_M = 10.0
SOURCE_EQUIV_RADIUS_M = 10.0
SOURCE_XY_M = (0.0, 1000.0)
# Breakpoints land on whole minutes, so 60 s piecewise-constant phases sampled
# at midpoints integrate the published piecewise-linear curve exactly
# (spec: "linear interpolation should be used").
INFLOW_PHASE_S = 60.0
# Snapshot cadence for the scored run: matches the phase length, so the front
# arrival times are resolved to 60 s without extra downloads.
OUTPUT_INTERVAL_S = 60.0

# Profile comparison time. The front reaches the nearest wall (1000 m away in
# all three directions) at ~189 min, past which the radial reference no longer
# describes the 2D domain; 2 h puts the front at ~720 m, comfortably clear.
PROBE_TIME_S = 2.0 * 3600.0
RADIAL_VALID_UNTIL_S = 180.0 * 60.0

DT_MAX_DEFAULT = 2.0
DT_INIT_DEFAULT = 1e-2
CFL_INTERVAL_DEFAULT = 10
# Reference resolution (dr = 1 m). A wetting front converges at first order, so
# the reference's own arrival times still carry ~1.5% at this level (measured
# against a 2800-cell solve); depths are within ~0.3%. Both sit far inside the
# gates below.
REFERENCE_CELLS = 1400
WET_TOL_M = 0.01  # front definition, shared by solver and reference

# ── Correctness gates ───────────────────────────────────────────────────────
# Closed domain (reflective walls) started dry + a volume-conserving source
# schedule, so the final volume should equal the injected volume. It does not,
# and the residual is not one error but two of opposite sign whose balance
# flips with both mesh and stopping time (+2.07 m^3 at dx=10 m/t=2 h against
# -38.98 m^3 at dx=5 m/t=2 h). The positive term is the `max(h_new, 0.0)`
# positivity clamp in vulkan/shaders/update.py, which fires at the wet/dry
# margin and can only add mass; the negative term is not identified. Refinement
# makes the aggregate worse because both terms accumulate over steps and the
# CFL-bound step count grows as 1/dx. The gate therefore brackets the spec
# configuration only, and its smallness there is a cancellation rather than a
# bound — see docs/implementation/19-grid-convergence/results.md before
# refining or reading anything into this number.
GATE_VOLUME_DRIFT_REL = 1e-3
# Front celerity: the headline quantity of this test. Relative error of the
# arrival time at each published gauge against the axisymmetric reference.
# Measured 0.063 (the solver runs systematically ~5-6% fast); the reference
# carries ~1.5% of its own discretisation error and snapshots resolve 60 s.
# Effort 19 established this is discretisation error, converging at first order
# (0.208 -> 0.025 as dx goes 20 -> 2.5 m), not a bias in the scheme.
GATE_ARRIVAL_ERR_REL = 0.10
# Depth and speed agreement at the probe time, relative to the reference's own
# scale over the gauges used. Measured 0.026 and 0.013.
GATE_DEPTH_L1_REL = 0.05
GATE_SPEED_L1_REL = 0.05
# Point 6 sits at 45 degrees; a grid-aligned scheme that propagates faster
# along the axes than diagonally shows up as its error exceeding the axis
# gauges' by more than this (absolute, in metres of depth).
GATE_ISOTROPY_M = 0.02
# Repeat-to-repeat reproducibility (relative to peak depth).
GATE_DETERMINISM_REL = 1e-4

DEFAULT_OUTPUT_ROOT = ".tmp/flood-propagation-bench"
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
GIF_FRAMES_DEFAULT = 64
GIF_VMAX_M = 0.4


@dataclass(frozen=True)
class PropagationSpec:
    """Geometry + dataset parameters for the flood-propagation case."""

    name: str
    nx: int
    ny: int
    domain_x_m: float
    domain_y_m: float
    manning_n: float
    t_end_s: float
    probe_time_s: float
    dt_max: float
    cfl_interval: int
    dataset_dir: Path
    cfl: float | None = None

    @property
    def dx_m(self) -> float:
        return self.domain_x_m / self.nx

    @property
    def dy_m(self) -> float:
        return self.domain_y_m / self.ny

    @property
    def bc_path(self) -> Path:
        return self.dataset_dir / BC_FILENAME

    @property
    def gauges_path(self) -> Path:
        return self.dataset_dir / GAUGES_FILENAME

    @property
    def inflow_x_m(self) -> float:
        """First cell-column centre — as close to the wall as cells allow."""
        return 0.5 * self.dx_m

    @property
    def inflow_centers_y_m(self) -> NDArray[np.float64]:
        lo, hi = INFLOW_LINE_Y_M
        return lo + (np.arange(INFLOW_SOURCES, dtype=np.float64) + 0.5) * (hi - lo) / INFLOW_SOURCES


@dataclass(frozen=True)
class CaseMetrics:
    """Metrics + gate outcomes for one (case, backend) against the reference."""

    case: str
    backend: str
    volume_final_m3: float
    volume_injected_m3: float
    volume_drift_rel: float
    min_depth_m: float
    h_final_finite: bool
    gauge_radii_m: list[float]
    arrival_s: list[float]
    arrival_ref_s: list[float]
    max_arrival_err_rel: float
    probe_depth_m: list[float]
    probe_depth_ref_m: list[float]
    depth_l1_rel: float
    probe_speed_ms: list[float]
    probe_speed_ref_ms: list[float]
    speed_l1_rel: float
    isotropy_excess_m: float
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


def _build_spec(args: argparse.Namespace) -> PropagationSpec:
    return PropagationSpec(
        name="propagation",
        nx=args.nx,
        ny=args.ny,
        domain_x_m=DOMAIN_X_M,
        domain_y_m=DOMAIN_Y_M,
        manning_n=MANNING_N,
        t_end_s=args.t_end,
        probe_time_s=args.probe_time,
        dt_max=args.dt_max,
        cfl_interval=args.cfl_interval,
        cfl=args.cfl,
        dataset_dir=Path(args.dataset_dir),
    )


def _load_hydrograph(bc_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parse the inflow hydrograph CSV; return (time [s], discharge [m^3/s]).

    The published table is in minutes; times come back in seconds.
    """
    data = np.loadtxt(bc_path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    t, q = data[:, 0] * 60.0, data[:, 1]
    if bool(np.any(np.diff(t) <= 0.0)):
        raise ValueError("hydrograph times must be strictly increasing")
    if bool(np.any(q < 0.0)) or q[-1] != 0.0:
        raise ValueError("hydrograph must be non-negative and end at zero inflow")
    return t, q


def _load_gauges(gauges_path: Path) -> NDArray[np.float64]:
    """Published output-point coordinates, in file order (points 1..6)."""
    data = np.loadtxt(gauges_path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if not np.array_equal(data[:, 0], np.arange(1, data.shape[0] + 1, dtype=np.float64)):
        raise ValueError("output points must be numbered 1..N in file order")
    return data[:, 1:3].copy()


def _gauge_radii(gauges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance of each published output point from the inflow line's centre."""
    return np.hypot(gauges[:, 0] - SOURCE_XY_M[0], gauges[:, 1] - SOURCE_XY_M[1])


def _inflow_sources(spec: PropagationSpec, discharge_m3s: float) -> list[PointSource]:
    """The published inflow line as equal-split near-boundary volume sources."""
    per_source = discharge_m3s / INFLOW_SOURCES
    return [
        PointSource(per_source, (spec.inflow_x_m, float(y)), INFLOW_RADIUS_M)
        for y in spec.inflow_centers_y_m
    ]


def _hydrograph_phases(spec: PropagationSpec) -> list[SimulationPhase]:
    """Published hydrograph as 60 s piecewise-constant phases (+ settle tail).

    Midpoint sampling of the piecewise-linear curve is volume-exact because
    the CSV breakpoints land on whole minutes (no phase straddles a kink).
    """
    t, q = _load_hydrograph(spec.bc_path)
    active_end_s = min(float(t[int(np.flatnonzero(q > 0.0).max()) + 1]), spec.t_end_s)
    n_active = round(active_end_s / INFLOW_PHASE_S)
    phases: list[SimulationPhase] = []
    for k in range(n_active):
        t_mid = (k + 0.5) * INFLOW_PHASE_S
        q_k = float(np.interp(t_mid, t, q))
        sources = _inflow_sources(spec, q_k) if q_k > 0.0 else []
        phases.append(SimulationPhase(duration_s=INFLOW_PHASE_S, sources=sources))
    settle_s = spec.t_end_s - n_active * INFLOW_PHASE_S
    if settle_s > 0.0:
        phases.append(SimulationPhase(duration_s=settle_s, sources=[]))
    return phases


def _injected_volume(spec: PropagationSpec) -> float:
    """Total hydrograph volume [m^3] as the phase schedule integrates it."""
    return float(
        sum(
            p.duration_s * sum(s.discharge_m3s for s in p.sources) for p in _hydrograph_phases(spec)
        )
    )


def _reference(spec: PropagationSpec) -> RadialInflowSolution:
    """Axisymmetric reference at the probe time, with the wall-doubled inflow."""
    t, q = _load_hydrograph(spec.bc_path)
    # Source on a closed wall: reflection makes this the same as a full-plane
    # problem at twice the discharge.
    return solve_radial_inflow(
        [spec.probe_time_s],
        hydrograph_t_s=t,
        hydrograph_q_m3s=2.0 * q,
        source_radius_m=SOURCE_EQUIV_RADIUS_M,
        manning_n=spec.manning_n,
        g=GRAVITY_G,
        r_max=1.4 * min(spec.domain_x_m, spec.domain_y_m / 2.0),
        wet_tol_m=WET_TOL_M,
        n_cells=REFERENCE_CELLS,
    )


def _ensure_mesh(spec: PropagationSpec, output_dir: Path) -> Path:
    """Write the explicitly flat-bed mesh required by the benchmark spec."""
    mesh_path = output_dir / f"plain-{spec.nx}x{spec.ny}.parquet"
    if not mesh_path.exists():
        verts, quads = build_channel_mesh(spec.nx, spec.ny, spec.domain_x_m, spec.domain_y_m)
        zb = np.zeros(spec.nx * spec.ny, dtype=np.float32)
        write_mesh_parquet(mesh_path, verts, quads, zb)
        print(f"[mesh] flat plain mesh written: {mesh_path} (N={spec.nx * spec.ny})")
    return mesh_path


def initial_state(spec: PropagationSpec) -> dict[str, object]:
    """Dry-bed initial condition (per the spec) in original cell order.

    Consumed across modules: ``inundation.bringup.check_clamp_budget`` reuses
    the harness's exact dry-bed condition so its mass budget measures the same
    configuration the suite gates on.
    """
    zeros = np.zeros(spec.nx * spec.ny, dtype=np.float32)
    return {"h": zeros.copy(), "hu": zeros.copy(), "hv": zeros.copy()}


def _run_once(
    spec: PropagationSpec,
    mesh_path: Path,
    backend: BackendImpl,
    *,
    t_end_s: float,
    output_interval_s: float,
    progress: bool,
    capture_momentum_snapshots: bool = False,
) -> tuple[SWEWorkflow, WorkflowResult]:
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
        initial_state_order="original",
        solver_impl=backend,
    )
    workflow = SWEWorkflow(config)
    workflow.prepare()
    probe_spec = spec if t_end_s == spec.t_end_s else _with_t_end(spec, t_end_s)
    result = workflow.run(_hydrograph_phases(probe_spec))
    return workflow, result


def _with_t_end(spec: PropagationSpec, t_end_s: float) -> PropagationSpec:
    return PropagationSpec(
        name=spec.name,
        nx=spec.nx,
        ny=spec.ny,
        domain_x_m=spec.domain_x_m,
        domain_y_m=spec.domain_y_m,
        manning_n=spec.manning_n,
        t_end_s=t_end_s,
        probe_time_s=spec.probe_time_s,
        dt_max=spec.dt_max,
        cfl_interval=spec.cfl_interval,
        cfl=spec.cfl,
        dataset_dir=spec.dataset_dir,
    )


def _gauge_cells(workflow: SWEWorkflow, gauges: NDArray[np.float64]) -> NDArray[np.int64]:
    """Solver cell nearest each published output point."""
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")
    return nearest_cell_indices(workflow.geom.centroid[:, 0], workflow.geom.centroid[:, 1], gauges)


def _arrival_times(
    snap_times: list[float], snapshots: list[NDArray[np.float32]], cells: NDArray[np.int64]
) -> NDArray[np.float64]:
    """First snapshot time at which each gauge cell exceeds the wet threshold."""
    arrival = np.full(cells.shape[0], np.inf, dtype=np.float64)
    for t, snap in zip(snap_times, snapshots, strict=True):
        wet = np.asarray(snap, dtype=np.float32)[cells] > WET_TOL_M
        arrival[wet & ~np.isfinite(arrival)] = t
    return arrival


def _evaluate_case(
    spec: PropagationSpec,
    backend: str,
    workflow: SWEWorkflow,
    result: WorkflowResult,
    probe: tuple[SWEWorkflow, WorkflowResult],
) -> CaseMetrics:
    reference = _reference(spec)
    gauges = _load_gauges(spec.gauges_path)
    radii = _gauge_radii(gauges)

    h_final = np.asarray(result.h_final, dtype=np.float32)
    finite, min_depth, fail_reasons = check_state_health(h_final, min_depth_tol=-1e-6)

    injected = _injected_volume(spec)
    drift_rel = volume_drift_rel(result.volume_final_m3, injected)

    cells = _gauge_cells(workflow, gauges)
    arrival = _arrival_times(result.snap_times, result.snapshots, cells)
    arrival_ref = reference.arrival_at(radii)
    max_arrival_err = max_relative_error(arrival, arrival_ref)

    probe_workflow, probe_result = probe
    probe_cells = _gauge_cells(probe_workflow, gauges)
    h_probe = np.asarray(probe_result.h_final, dtype=np.float32)[probe_cells].astype(np.float64)
    hu = probe_result.hu_final
    hv = probe_result.hv_final
    if hu is None or hv is None:
        raise RuntimeError(f"backend {backend} does not expose momentum (download_hu/hv)")
    depth_all = np.asarray(probe_result.h_final, dtype=np.float64)
    speed_all = np.hypot(np.asarray(hu, np.float64), np.asarray(hv, np.float64)) / np.maximum(
        depth_all, 1e-9
    )
    speed_probe = speed_all[probe_cells]

    h_ref = reference.depth_at(0, radii)
    speed_ref = reference.speed_at(0, radii)
    # Point 1 sits only 5 source-radii out, where the line-vs-disc idealisation
    # of the inflow still shows; the profile norms use points 2-6.
    far = radii >= 100.0
    depth_l1 = l1_relative_error(h_probe[far], h_ref[far])
    speed_l1 = l1_relative_error(speed_probe[far], speed_ref[far])

    # Isotropy: point 6 is the only off-axis gauge, so compare its depth error
    # with the worst axis-gauge error at the same probe time.
    on_axis = far & (gauges[:, 1] == SOURCE_XY_M[1])
    off_axis = far & ~on_axis
    axis_err = (
        float(np.abs(h_probe[on_axis] - h_ref[on_axis]).max()) if bool(on_axis.any()) else 0.0
    )
    diag_err = (
        float(np.abs(h_probe[off_axis] - h_ref[off_axis]).max()) if bool(off_axis.any()) else 0.0
    )
    isotropy_excess = diag_err - axis_err

    if drift_rel > GATE_VOLUME_DRIFT_REL:
        fail_reasons.append(f"volume_drift_rel:{drift_rel:.3e}>{GATE_VOLUME_DRIFT_REL:.0e}")
    if not np.isfinite(arrival).all():
        dry = [int(k) + 1 for k, a in enumerate(arrival) if not np.isfinite(a)]
        fail_reasons.append(f"gauges_never_wetted:{dry}")
    if max_arrival_err > GATE_ARRIVAL_ERR_REL:
        fail_reasons.append(
            f"arrival_err_rel:{max_arrival_err:.3f}>{GATE_ARRIVAL_ERR_REL} (front celerity)"
        )
    if depth_l1 > GATE_DEPTH_L1_REL:
        fail_reasons.append(f"depth_l1_rel:{depth_l1:.3f}>{GATE_DEPTH_L1_REL}")
    if speed_l1 > GATE_SPEED_L1_REL:
        fail_reasons.append(f"speed_l1_rel:{speed_l1:.3f}>{GATE_SPEED_L1_REL}")
    if isotropy_excess > GATE_ISOTROPY_M:
        fail_reasons.append(
            f"isotropy_excess:{isotropy_excess:.3f}>{GATE_ISOTROPY_M} "
            f"(diagonal gauge tracks the reference worse than the axis gauges)"
        )

    return CaseMetrics(
        case=spec.name,
        backend=backend,
        volume_final_m3=float(result.volume_final_m3),
        volume_injected_m3=injected,
        volume_drift_rel=drift_rel,
        min_depth_m=min_depth,
        h_final_finite=finite,
        gauge_radii_m=[float(r) for r in radii],
        arrival_s=[float(a) for a in arrival],
        arrival_ref_s=[float(a) for a in arrival_ref],
        max_arrival_err_rel=max_arrival_err,
        probe_depth_m=[float(d) for d in h_probe],
        probe_depth_ref_m=[float(d) for d in h_ref],
        depth_l1_rel=depth_l1,
        probe_speed_ms=[float(s) for s in speed_probe],
        probe_speed_ref_ms=[float(s) for s in speed_ref],
        speed_l1_rel=speed_l1,
        isotropy_excess_m=isotropy_excess,
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
    spec: PropagationSpec,
    mesh_path: Path,
    backend: BackendImpl,
    args: argparse.Namespace,
) -> tuple[CaseMetrics, PerfMetrics]:
    n_cells = spec.nx * spec.ny

    for w in range(args.warmup):
        print(f"[{spec.name}/{backend}] warmup {w + 1}/{args.warmup}")
        _run_once(
            spec,
            mesh_path,
            backend,
            t_end_s=spec.t_end_s,
            output_interval_s=OUTPUT_INTERVAL_S,
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
            t_end_s=spec.t_end_s,
            output_interval_s=OUTPUT_INTERVAL_S,
            progress=args.solver_progress,
        )
        walls.append(time.perf_counter() - t0)
        final_states.append(np.asarray(result.h_final, dtype=np.float64))
        last = (workflow, result)

    if last is None:
        raise RuntimeError("--repeats must be >= 1")
    workflow, result = last

    # Untimed probe run stopping inside the radially symmetric window, where
    # the reference is valid and momentum can be compared against it.
    print(
        f"[{spec.name}/{backend}] untimed probe run to "
        f"{spec.probe_time_s / 60.0:.0f} min (radial window)"
    )
    probe = _run_once(
        spec,
        mesh_path,
        backend,
        t_end_s=spec.probe_time_s,
        output_interval_s=spec.probe_time_s,
        progress=args.solver_progress,
    )

    case_metrics = _evaluate_case(spec, backend, workflow, result, probe)

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
        print(f"[{spec.name}/{backend}] extra untimed gif run ({safe_frames} frames)")
        gif_workflow, gif_result = _run_once(
            spec,
            mesh_path,
            backend,
            t_end_s=spec.t_end_s,
            output_interval_s=spec.t_end_s / safe_frames,
            progress=args.solver_progress,
            capture_momentum_snapshots=backend in MOMENTUM_SNAPSHOT_BACKENDS,
        )
        # The solver snapshots every phase boundary too, so the 60 s hydrograph
        # phases would otherwise dominate the frame list.
        gif_snapshots, gif_times = resample_snapshots_uniform(
            gif_result.snapshots, gif_result.snap_times, safe_frames
        )
        gauges = _load_gauges(spec.gauges_path)
        cells = _gauge_cells(gif_workflow, gauges)
        selected = np.asarray([0, 2, 4, 5], dtype=np.int64)
        if gif_workflow.geom is None:
            raise RuntimeError("workflow must be prepared")
        wse = water_surface_elevation(gif_result.snapshots, gif_workflow.geom.zb, cells)
        save_faceted_line_plot(
            gif_result.snap_times,
            wse[:, selected],
            [str(i + 1) for i in selected],
            Path(args.output_dir) / f"test4-water-levels-{backend}.png",
            ylabel="Water level [m]",
            title="Test 4 flood-front water levels",
        )
        speed = speed_from_momentum(
            gif_result.hu_snapshots, gif_result.hv_snapshots, gif_result.snapshots, cells
        )
        if speed is not None:
            save_faceted_line_plot(
                gif_result.snap_times,
                speed[:, selected],
                [str(i + 1) for i in selected],
                Path(args.output_dir) / f"test4-velocities-{backend}.png",
                ylabel="Speed [m/s]",
                title="Test 4 flood-front velocities",
            )
        else:
            print(
                f"[{spec.name}/{backend}] WARNING: omitted test4-velocities-{backend}.png; momentum snapshots unavailable"
            )
        times = np.asarray(gif_result.snap_times, dtype=np.float64)
        picks = [int(np.argmin(abs(times - target))) for target in (3600.0, 10800.0)]
        grids = []
        for pick in picks:
            grid = np.zeros((spec.ny, spec.nx), dtype=np.float64)
            # geom is non-None: _run_once prepares the workflow before returning it
            # and the guard above re-checks, so re-testing per pick is dead code.
            cx = gif_workflow.geom.centroid[:, 0]
            cy = gif_workflow.geom.centroid[:, 1]
            col = np.clip((cx / (spec.domain_x_m / spec.nx)).astype(int), 0, spec.nx - 1)
            row = np.clip((cy / (spec.domain_y_m / spec.ny)).astype(int), 0, spec.ny - 1)
            grid[row, col] = gif_result.snapshots[pick]
            grids.append(grid)
        save_depth_contours(
            grids,
            [times[p] for p in picks],
            Path(args.output_dir) / f"test4-depth-contours-{backend}.png",
            extent=(0.0, spec.domain_x_m, 0.0, spec.domain_y_m),
        )
        line_order = np.argsort(gauges[:, 0])
        distance = gauges[line_order, 0] - gauges[line_order, 0].min()
        one_hour = picks[0]
        save_cross_section(
            distance,
            gif_result.snapshots[one_hour][cells[line_order]],
            Path(args.output_dir) / f"test4-depth-cross-section-{backend}.png",
            ylabel="Depth [m]",
            title="Test 4 depth cross-section at 1 hour",
        )
        if speed is not None:
            save_cross_section(
                distance,
                speed[one_hour, line_order],
                Path(args.output_dir) / f"test4-velocity-cross-section-{backend}.png",
                ylabel="Speed [m/s]",
                title="Test 4 velocity cross-section at 1 hour",
            )
        save_depth_gif(
            gif_workflow,
            gif_snapshots,
            gif_times,
            Path(args.output_dir) / f"{spec.name}-{backend}-2d.gif",
            vmax=GIF_VMAX_M,
            title_prefix=f"{spec.name} — ",
        )

    return case_metrics, perf


def _print_report(all_metrics: list[CaseMetrics], all_perf: list[PerfMetrics]) -> None:
    print()
    print(
        f"gates: volume_drift<={GATE_VOLUME_DRIFT_REL:.0e} min_depth>=0 finite "
        f"arrival_err<={GATE_ARRIVAL_ERR_REL} depth_l1<={GATE_DEPTH_L1_REL} "
        f"speed_l1<={GATE_SPEED_L1_REL} isotropy<={GATE_ISOTROPY_M}"
    )
    print(
        f"{'case':12} {'backend':24} {'vol_drift':>9} {'min_h':>10} {'arr_err':>8} "
        f"{'depth_l1':>9} {'speed_l1':>9} {'isotropy':>9} {'pass':>5}"
    )
    for m in all_metrics:
        print(
            f"{m.case:12} {m.backend:24} {m.volume_drift_rel:9.3e} {m.min_depth_m:10.3e} "
            f"{m.max_arrival_err_rel:8.3f} {m.depth_l1_rel:9.3f} {m.speed_l1_rel:9.3f} "
            f"{m.isotropy_excess_m:9.3f} {m.passed!s:>5}"
        )
        for reason in m.fail_reasons:
            print(f"  !! {reason}")
        radii = " ".join(f"{r:8.1f}" for r in m.gauge_radii_m)
        print(f"  gauge radius        [m]: {radii}")
        print("  front arrival     [min]: " + " ".join(f"{a / 60:8.2f}" for a in m.arrival_s))
        print("    reference       [min]: " + " ".join(f"{a / 60:8.2f}" for a in m.arrival_ref_s))
        print("  probe depth         [m]: " + " ".join(f"{d:8.3f}" for d in m.probe_depth_m))
        print("    reference         [m]: " + " ".join(f"{d:8.3f}" for d in m.probe_depth_ref_m))
        print("  probe speed      [m/s]: " + " ".join(f"{s:8.3f}" for s in m.probe_speed_ms))
        print("    reference      [m/s]: " + " ".join(f"{s:8.3f}" for s in m.probe_speed_ref_ms))
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
        description="EA Test 4 flood-propagation benchmark (published May-2010 dataset)"
    )
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help=f"CSV of solver_impl names {AVAILABLE_BACKENDS} (default: Vulkan reference)",
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--nx", type=int, default=NX_DEFAULT)
    parser.add_argument("--ny", type=int, default=NY_DEFAULT)
    parser.add_argument("--t-end", type=float, default=T_END_S)
    parser.add_argument(
        "--probe-time",
        type=float,
        default=PROBE_TIME_S,
        help=(
            "time at which depth/speed profiles are compared with the radial "
            f"reference; must stay under ~{RADIAL_VALID_UNTIL_S / 60:.0f} min, "
            "when the front reaches the walls"
        ),
    )
    parser.add_argument("--dt-max", type=float, default=DT_MAX_DEFAULT)
    parser.add_argument("--cfl-interval", type=int, default=CFL_INTERVAL_DEFAULT)
    parser.add_argument("--cfl", type=float, default=None, help="CFL number")
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
    if spec.probe_time_s > RADIAL_VALID_UNTIL_S:
        print(
            f"--probe-time {spec.probe_time_s / 60:.0f} min is past the "
            f"~{RADIAL_VALID_UNTIL_S / 60:.0f} min radial-symmetry window "
            f"(the front reaches the walls); the reference does not apply there"
        )
        return 2
    mesh_path = _ensure_mesh(spec, output_dir)
    gauges = _load_gauges(spec.gauges_path)
    print(
        f"[hydrograph] published inflow, injected volume = {_injected_volume(spec):.0f} m^3; "
        f"t_end {spec.t_end_s / 3600.0:.0f} h"
    )
    print(
        f"[reference] axisymmetric solve at {spec.probe_time_s / 60:.0f} min "
        f"for {gauges.shape[0]} gauges at radii "
        + ", ".join(f"{r:.0f}" for r in _gauge_radii(gauges))
        + " m"
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
            "arrival_err_rel": GATE_ARRIVAL_ERR_REL,
            "depth_l1_rel": GATE_DEPTH_L1_REL,
            "speed_l1_rel": GATE_SPEED_L1_REL,
            "isotropy_m": GATE_ISOTROPY_M,
            "determinism_rel": GATE_DETERMINISM_REL,
        },
        "injected_volume_m3": _injected_volume(spec),
        "probe_time_s": spec.probe_time_s,
        "correctness": [asdict(m) for m in all_metrics],
        "performance": [asdict(p) for p in all_perf],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[summary] {summary_path}")

    correctness_ok = all(m.passed for m in all_metrics)
    determinism_ok = all(p.deterministic is not False for p in all_perf)
    return 0 if (correctness_ok and determinism_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
