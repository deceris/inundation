"""EA Test 2 benchmark: filling of floodplain depressions.

Runs the UK Environment Agency "Benchmarking of 2D Hydraulic Modelling
Packages" Test 2 from the published May-2010 dataset
(``benchmark_assets/Test2 dataset 2010``): the georeferenced 2 m
ASCII DEM (``test2DEM.asc`` — a 2000 m x 2000 m "flattened egg box" of 16
~0.5 m depressions on a plane falling ~2 m along the NW→SE diagonal), the
inflow hydrograph (``Test2_BC.csv``, peak 20 m^3/s over an ~85 min base,
97 200 m^3 total) and the 16 published output points (``Test2output.csv``,
one at the centre of each depression). Per the spec: modelled area
x, y in [0, 2000] m (``Test2ActiveArea_region``), 20 m model resolution
(~10 000 nodes), Manning n = 0.03 uniform, dry-bed initial condition, run
to t = 48 h so the inundation settles to its final state.

The physics under test: inundation extent and final ponded depth under
*low-momentum* flow over complex topography. Water enters at the high (NW)
corner, runs downhill, fills the first depression to its sill, spills into
the next, and so on until the hydrograph volume is exhausted — the answer
|
That endpoint is *not* a simple fill-and-spill cascade. Conveying the 20 m^3/s
peak over a saddle needs ~0.24 m  of head (broad-crested weir, ~100 m crest)
while the competing saddles around a depression differ by only ~0.04 m, so
while the hydrograph runs the water spreads on a broad front and strands
itself across many depressions at once — which of them fill is genuinely
path-dependent. What is *not* path-dependent is the terrain: a settled pond
cannot stand above the sill it would spill over, and cannot exceed its
basin's storage capacity. ``depression_basin`` (``bench/common.py``) computes
both from the same bed the solver sees, and the gates are built on those
bounds plus the closed-domain mass balance.

One adaptation: this solver has no open-boundary inflow, so the hydrograph
is injected as near-boundary *volume* sources along the published inflow
line (``Test2BC_polyline``: the 100 m stretch of the western boundary
running south from the NW corner). Volume sources carry no momentum vector,
which is conservative for a case whose published emphasis is the settled
distribution rather than the peak wave.

Backends are configurable via ``--backends``; the retained Vulkan family runs
on macOS/MoltenVK. The inflow is applied as a piecewise-constant per-phase
``source_rate`` (not a ``source_fn``), so the default ``gpu_resident_batch``
backend stays on its GPU-resident path. ``--gif`` renders an *untimed* extra
run into a top-down depth-heatmap animation; timed runs are never the frame
source, so perf stays clean.
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
    block_average_grid,
    build_channel_mesh,
    check_state_health,
    depression_basin,
    load_ascii_grid,
    nearest_cell_indices,
    resample_snapshots_uniform,
    save_depth_gif,
    save_depth_png,
    save_faceted_line_plot,
    volume_drift_rel,
    water_surface_elevation,
    write_mesh_parquet,
)
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
    Path(__file__).resolve().parents[2] / "benchmark_assets" / "Test2 dataset 2010"
)
DEM_FILENAME = "test2DEM.asc"
BC_FILENAME = "Test2_BC.csv"
GAUGES_FILENAME = "Test2output.csv"

# Modelled area per Test2ActiveArea_region: a perfect 2000 m square with its
# SW corner on the DEM origin (the raster carries a 200 m apron beyond it).
# Mesh coordinates coincide with the DEM's georeference, so the published
# output-point locations are used as-is. All boundaries are closed per the
# spec; this solver's walls are reflective.
DOMAIN_L_M = 2000.0
NX_DEFAULT = 100  # dx = 20 m (spec: 20 m grid, ~10000 nodes)
NY_DEFAULT = 100

# EA Test 2 floodplain roughness per the spec: 0.03 uniform.
MANNING_N = 0.03

# Inflow adaptation: the published boundary condition runs along the western
# edge from the NW corner south to y = 1900 m (Test2BC_polyline). This solver
# has no open boundaries, so the discharge enters as an equal split of
# near-boundary volume sources on the first cell column over that stretch.
INFLOW_X_M = 10.0
INFLOW_Y_RANGE_M = (1900.0, 2000.0)
INFLOW_RADIUS_M = 20.0
INFLOW_SOURCES = 5
# The hydrograph breakpoints land on whole minutes, so 60 s piecewise-constant
# phases sampled at midpoints integrate the published piecewise-linear curve
# exactly (spec: "linear interpolation should be used").
INFLOW_PHASE_S = 60.0

T_END_S = 48.0 * 3600.0  # spec: run to t = 48 h; integer -> float32-exact stop

DT_MAX_DEFAULT = 5.0
DT_INIT_DEFAULT = 1e-2
CFL_INTERVAL_DEFAULT = 10

# ── Correctness gates ───────────────────────────────────────────────────────
# Closed domain (reflective walls) started dry + a volume-conserving source
# schedule: every injected cubic metre is retained, so the final volume must
# equal the injected volume to float32 accumulation error. This is the anchor
# gate; the run is ~4x longer than the EA Test 3 case (which holds 5e-5), so
# the bound carries proportionate headroom.
GATE_VOLUME_DRIFT_REL = 2e-4
# A gauge counts as ponded above this depth — below it is residual film, not
# inundation.
POND_LEVEL_M = 0.05
# A settled pond cannot stand above the sill it would spill over. The pool
# boundary is discretised, so equilibrium sits marginally above the discrete
# sill (a sub-cell effect: one 20 m cell spans ~0.026 m of the local bed
# gradient); this bound is comfortably inside that and still tight enough to
# catch water climbing out of its basin.
GATE_ABOVE_SILL_M = 0.03
# Basins filled to their sill must together hold no more than was injected —
# a conservation-derived bound on how much of the floodplain can fill.
GATE_FULL_BASIN_CAPACITY_REL = 1.0
# A basin counts as full when its surface is within this of its sill.
FULL_BASIN_TOL_M = 0.02
# Fraction of the injected volume that must have drained off the ridges into
# the depressions by t_end. The remainder is film in transit on the slopes.
GATE_PONDED_STORAGE_FRAC = 0.90
# The eastern column of depressions (output points 13-16) must stay dry: the
# injected volume is well below the storage capacity of the twelve
# depressions in the three western columns, so it cannot fill through to
# them. Anything appearing there is water stranded in transit.
GATE_FAR_COLUMN_DRY_M = POND_LEVEL_M
# Repeat-to-repeat reproducibility (relative to peak depth). GPU atomicAdd flux
# scatter is not bit-reproducible, but the settled state is an attractor: with
# the ponds at rest by t_end the drift damps out rather than accumulating over
# the ~2e5 float32 steps, measuring 1.8e-6 on MoltenVK (~50x headroom here).
GATE_DETERMINISM_REL = 1e-4

DEFAULT_OUTPUT_ROOT = ".tmp/floodplain-depressions-bench"
# gpu_resident_batch is the default Vulkan backend. The host-CFL-readback
# fixed_dt_batch_barrier does not develop the wave under MoltenVK, so it is
# opt-in via --backends where that environment supports it.
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
GIF_VMAX_M = 0.6


@dataclass(frozen=True)
class DepressionSpec:
    """Geometry + dataset parameters for the floodplain-depressions case."""

    name: str
    nx: int
    ny: int
    domain_l_m: float
    manning_n: float
    t_end_s: float
    dt_max: float
    cfl_interval: int
    dataset_dir: Path
    cfl: float | None = None

    @property
    def dx_m(self) -> float:
        return self.domain_l_m / self.nx

    @property
    def dy_m(self) -> float:
        return self.domain_l_m / self.ny

    @property
    def cell_area_m2(self) -> float:
        return self.dx_m * self.dy_m

    @property
    def dem_path(self) -> Path:
        return self.dataset_dir / DEM_FILENAME

    @property
    def bc_path(self) -> Path:
        return self.dataset_dir / BC_FILENAME

    @property
    def gauges_path(self) -> Path:
        return self.dataset_dir / GAUGES_FILENAME

    @property
    def inflow_centers_y_m(self) -> NDArray[np.float64]:
        """Evenly spaced source centres along the published inflow line."""
        lo, hi = INFLOW_Y_RANGE_M
        return lo + (np.arange(INFLOW_SOURCES, dtype=np.float64) + 0.5) * (hi - lo) / INFLOW_SOURCES

    @property
    def inflow_cell(self) -> tuple[int, int]:
        """(row, col) of the mesh cell at the centre of the inflow line."""
        y_mid = 0.5 * (INFLOW_Y_RANGE_M[0] + INFLOW_Y_RANGE_M[1])
        return (int(y_mid // self.dy_m), int(INFLOW_X_M // self.dx_m))


@dataclass(frozen=True)
class CaseMetrics:
    """Metrics + gate outcomes for one (case, backend) against the terrain bounds."""

    case: str
    backend: str
    volume_drift_rel: float
    min_depth_m: float
    ponded_count: int
    max_above_sill_m: float
    full_basin_count: int
    full_basin_capacity_rel: float
    ponded_storage_frac: float
    far_column_max_depth_m: float
    h_final_finite: bool
    point_depths_m: list[float]
    sill_depths_m: list[float]
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


def _build_spec(args: argparse.Namespace) -> DepressionSpec:
    return DepressionSpec(
        name="eggbox",
        nx=args.nx,
        ny=args.ny,
        domain_l_m=DOMAIN_L_M,
        manning_n=MANNING_N,
        t_end_s=args.t_end,
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
    """Published output-point coordinates, in file order (points 1..16)."""
    data = np.loadtxt(gauges_path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if not np.array_equal(data[:, 0], np.arange(1, data.shape[0] + 1, dtype=np.float64)):
        raise ValueError("output points must be numbered 1..N in file order")
    return data[:, 1:3].copy()


def _inflow_sources(spec: DepressionSpec, discharge_m3s: float) -> list[PointSource]:
    """The published inflow line as equal-split near-boundary volume sources."""
    per_source = discharge_m3s / INFLOW_SOURCES
    return [
        PointSource(per_source, (INFLOW_X_M, float(y)), INFLOW_RADIUS_M)
        for y in spec.inflow_centers_y_m
    ]


def _hydrograph_phases(spec: DepressionSpec) -> list[SimulationPhase]:
    """Published hydrograph as 60 s piecewise-constant phases + settle tail.

    Midpoint sampling of the piecewise-linear curve is volume-exact because
    the CSV breakpoints land on whole minutes (no phase straddles a kink).
    """
    t, q = _load_hydrograph(spec.bc_path)
    active_end_s = float(t[int(np.flatnonzero(q > 0.0).max()) + 1])
    n_active = round(active_end_s / INFLOW_PHASE_S)
    phases: list[SimulationPhase] = []
    for k in range(n_active):
        t_mid = (k + 0.5) * INFLOW_PHASE_S
        q_k = float(np.interp(t_mid, t, q))
        sources = _inflow_sources(spec, q_k) if q_k > 0.0 else []
        phases.append(SimulationPhase(duration_s=INFLOW_PHASE_S, sources=sources))
    settle_s = spec.t_end_s - n_active * INFLOW_PHASE_S
    if settle_s < 0.0:
        raise ValueError(f"t_end {spec.t_end_s} shorter than the hydrograph ({active_end_s} s)")
    phases.append(SimulationPhase(duration_s=settle_s, sources=[]))
    return phases


def _injected_volume(spec: DepressionSpec) -> float:
    """Total hydrograph volume [m^3] as the phase schedule integrates it."""
    return float(
        sum(
            p.duration_s * sum(s.discharge_m3s for s in p.sources) for p in _hydrograph_phases(spec)
        )
    )


def _bed_grid(spec: DepressionSpec) -> NDArray[np.float64]:
    """Cell-average bed elevation over the modelled area, shaped (ny, nx).

    Row-major with ``y`` ascending, so ``.ravel()`` is the mesh's original
    (pre-Hilbert) cell order.
    """
    grid = load_ascii_grid(spec.dem_path)
    return block_average_grid(
        grid,
        x_edges=np.linspace(0.0, spec.domain_l_m, spec.nx + 1),
        y_edges=np.linspace(0.0, spec.domain_l_m, spec.ny + 1),
    )


def _gauge_basins(
    spec: DepressionSpec, zb: NDArray[np.float64], gauges: NDArray[np.float64]
) -> list[tuple[float, NDArray[np.int64], float]]:
    """(sill elevation, pool cells, capacity) of the depression at each gauge."""
    return [
        depression_basin(
            zb, (int(gy // spec.dy_m), int(gx // spec.dx_m)), cell_area=spec.cell_area_m2
        )
        for gx, gy in gauges
    ]


def _ensure_mesh(spec: DepressionSpec, output_dir: Path) -> Path:
    """Write the DEM-sampled mesh over the modelled area (idempotent)."""
    # "dem2010" = published May-2010 raster; bump if the sampling changes so
    # a stale cached parquet from an older bed is never silently reused.
    mesh_path = output_dir / f"eggbox-dem2010-{spec.nx}x{spec.ny}.parquet"
    if not mesh_path.exists():
        verts, quads = build_channel_mesh(spec.nx, spec.ny, spec.domain_l_m, spec.domain_l_m)
        zb = _bed_grid(spec).ravel().astype(np.float32)
        write_mesh_parquet(mesh_path, verts, quads, zb)
        print(f"[mesh] DEM-sampled mesh written: {mesh_path} (N={spec.nx * spec.ny})")
    return mesh_path


def _initial_state(spec: DepressionSpec) -> dict[str, object]:
    """Dry-bed initial condition (per the spec) in original cell order."""
    zeros = np.zeros(spec.nx * spec.ny, dtype=np.float32)
    return {"h": zeros.copy(), "hu": zeros.copy(), "hv": zeros.copy()}


def _make_workflow(
    spec: DepressionSpec,
    mesh_path: Path,
    backend: BackendImpl,
    *,
    output_interval_s: float,
    progress: bool,
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
        initial_state_source=_initial_state(spec),
        initial_state_order="original",
        solver_impl=backend,
    )
    workflow = SWEWorkflow(config)
    workflow.prepare()
    return workflow


def _run_once(
    spec: DepressionSpec,
    mesh_path: Path,
    backend: BackendImpl,
    *,
    output_interval_s: float,
    progress: bool,
) -> tuple[SWEWorkflow, WorkflowResult]:
    workflow = _make_workflow(
        spec,
        mesh_path,
        backend,
        output_interval_s=output_interval_s,
        progress=progress,
    )
    result = workflow.run(_hydrograph_phases(spec))
    return workflow, result


def _gauge_cells(workflow: SWEWorkflow, gauges: NDArray[np.float64]) -> NDArray[np.int64]:
    """Solver cell nearest each published output point."""
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")
    return nearest_cell_indices(workflow.geom.centroid[:, 0], workflow.geom.centroid[:, 1], gauges)


def _depth_grid(
    spec: DepressionSpec, workflow: SWEWorkflow, h_final: NDArray[np.float32]
) -> NDArray[np.float64]:
    """Scatter solver-order depths back onto the (ny, nx) mesh grid."""
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")
    cx = workflow.geom.centroid[:, 0].astype(np.float64)
    cy = workflow.geom.centroid[:, 1].astype(np.float64)
    col = np.clip((cx / spec.dx_m).astype(np.int64), 0, spec.nx - 1)
    row = np.clip((cy / spec.dy_m).astype(np.int64), 0, spec.ny - 1)
    grid = np.zeros((spec.ny, spec.nx), dtype=np.float64)
    grid[row, col] = h_final.astype(np.float64)
    return grid


def _evaluate_case(
    spec: DepressionSpec,
    backend: str,
    workflow: SWEWorkflow,
    result: WorkflowResult,
) -> CaseMetrics:
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")

    h_final = np.asarray(result.h_final, dtype=np.float32)
    finite, min_depth, fail_reasons = check_state_health(h_final, min_depth_tol=-1e-6)

    injected = _injected_volume(spec)
    drift_rel = volume_drift_rel(result.volume_final_m3, injected)

    gauges = _load_gauges(spec.gauges_path)
    zb = _bed_grid(spec)
    basins = _gauge_basins(spec, zb, gauges)
    cells = _gauge_cells(workflow, gauges)
    point_depths = [float(h_final[i]) for i in cells]
    sill_depths = [
        sill - float(zb[int(gy // spec.dy_m), int(gx // spec.dx_m)])
        for (sill, _pool, _cap), (gx, gy) in zip(basins, gauges, strict=True)
    ]

    wet = [d > POND_LEVEL_M for d in point_depths]
    # A settled pond cannot stand above its sill, nor can a basin hold more
    # than its capacity — both are properties of the bed alone.
    max_above_sill = max(
        (d - s for d, s, w in zip(point_depths, sill_depths, wet, strict=True) if w),
        default=0.0,
    )
    full = [d > s - FULL_BASIN_TOL_M for d, s in zip(point_depths, sill_depths, strict=True)]
    full_capacity = sum(cap for (_sill, _pool, cap), f in zip(basins, full, strict=True) if f)
    full_basin_capacity_rel = full_capacity / injected if injected > 0.0 else float("inf")

    depth = _depth_grid(spec, workflow, h_final).reshape(-1)
    in_basin = np.zeros(depth.shape[0], dtype=np.bool_)
    for _sill, pool, _cap in basins:
        in_basin[pool] = True
    ponded_storage_frac = (
        float(depth[in_basin].sum()) * spec.cell_area_m2 / injected if injected > 0.0 else 0.0
    )

    far_column = gauges[:, 0] == gauges[:, 0].max()
    far_column_max = max(
        (d for d, far in zip(point_depths, far_column, strict=True) if far), default=0.0
    )

    if drift_rel > GATE_VOLUME_DRIFT_REL:
        fail_reasons.append(f"volume_drift_rel:{drift_rel:.3e}>{GATE_VOLUME_DRIFT_REL:.0e}")
    if max_above_sill > GATE_ABOVE_SILL_M:
        fail_reasons.append(
            f"pond_above_sill:{max_above_sill:.3f}>{GATE_ABOVE_SILL_M} "
            f"(water standing above the sill it would spill over)"
        )
    if full_basin_capacity_rel > GATE_FULL_BASIN_CAPACITY_REL:
        fail_reasons.append(
            f"full_basin_capacity_rel:{full_basin_capacity_rel:.3f}>"
            f"{GATE_FULL_BASIN_CAPACITY_REL} (more storage filled than was injected)"
        )
    if ponded_storage_frac < GATE_PONDED_STORAGE_FRAC:
        fail_reasons.append(
            f"ponded_storage_frac:{ponded_storage_frac:.3f}<{GATE_PONDED_STORAGE_FRAC} "
            f"(water still stranded on the ridges at t_end)"
        )
    if far_column_max > GATE_FAR_COLUMN_DRY_M:
        fail_reasons.append(f"far_column_depth:{far_column_max:.3f}>{GATE_FAR_COLUMN_DRY_M}")

    return CaseMetrics(
        case=spec.name,
        backend=backend,
        volume_drift_rel=drift_rel,
        min_depth_m=min_depth,
        ponded_count=sum(wet),
        max_above_sill_m=max_above_sill,
        full_basin_count=sum(full),
        full_basin_capacity_rel=full_basin_capacity_rel,
        ponded_storage_frac=ponded_storage_frac,
        far_column_max_depth_m=far_column_max,
        h_final_finite=finite,
        point_depths_m=point_depths,
        sill_depths_m=sill_depths,
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
    spec: DepressionSpec,
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

    case_metrics = _evaluate_case(spec, backend, workflow, result)

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
            print(
                f"[{spec.name}/{backend}] gif frames {args.gif_frames} -> "
                f"{safe_frames} (float32-safe interval)"
            )
        print(f"[{spec.name}/{backend}] extra untimed gif run ({safe_frames} frames)")
        gif_workflow, gif_result = _run_once(
            spec,
            mesh_path,
            backend,
            output_interval_s=spec.t_end_s / safe_frames,
            progress=args.solver_progress,
        )
        # The solver snapshots every phase boundary too, so the 91 one-minute
        # hydrograph phases would otherwise crowd the first 3% of the run.
        gif_snapshots, gif_times = resample_snapshots_uniform(
            gif_result.snapshots, gif_result.snap_times, safe_frames
        )
        gauges = _load_gauges(spec.gauges_path)
        cells = _gauge_cells(gif_workflow, gauges)
        if gif_workflow.geom is None:
            raise RuntimeError("workflow must be prepared")
        save_faceted_line_plot(
            gif_result.snap_times,
            water_surface_elevation(gif_result.snapshots, gif_workflow.geom.zb, cells),
            [str(i) for i in range(1, len(cells) + 1)],
            Path(args.output_dir) / f"test2-water-levels-{backend}.png",
            ylabel="Water-surface elevation [m]",
            title="Test 2 depression water levels",
        )
        save_depth_png(
            gif_workflow,
            gif_result.h_final,
            gif_result.snap_times[-1],
            Path(args.output_dir) / f"test2-final-inundation-{backend}.png",
        )
        save_depth_gif(
            gif_workflow,
            gif_snapshots,
            gif_times,
            Path(args.output_dir) / f"{spec.name}-{backend}-2d.gif",
            vmax=GIF_VMAX_M,
            title_prefix=f"{spec.name} depressions — ",
        )

    return case_metrics, perf


def _print_report(all_metrics: list[CaseMetrics], all_perf: list[PerfMetrics]) -> None:
    print()
    print(
        f"gates: volume_drift<={GATE_VOLUME_DRIFT_REL:.0e} min_depth>=0 finite "
        f"above_sill<={GATE_ABOVE_SILL_M} full_capacity<={GATE_FULL_BASIN_CAPACITY_REL} "
        f"ponded_storage>={GATE_PONDED_STORAGE_FRAC} far_column_dry"
    )
    print(
        f"{'case':10} {'backend':24} {'vol_drift':>9} {'min_h':>10} {'ponded':>7} "
        f"{'full':>5} {'abv_sill':>9} {'cap_rel':>8} {'storage':>8} {'far_h':>7} {'pass':>5}"
    )
    for m in all_metrics:
        print(
            f"{m.case:10} {m.backend:24} {m.volume_drift_rel:9.3e} {m.min_depth_m:10.3e} "
            f"{m.ponded_count:7d} {m.full_basin_count:5d} {m.max_above_sill_m:9.3f} "
            f"{m.full_basin_capacity_rel:8.3f} {m.ponded_storage_frac:8.3f} "
            f"{m.far_column_max_depth_m:7.3f} {m.passed!s:>5}"
        )
        for reason in m.fail_reasons:
            print(f"  !! {reason}")
        sim = " ".join(f"{d:.2f}" for d in m.point_depths_m)
        sills = " ".join(f"{d:.2f}" for d in m.sill_depths_m)
        print(f"  point depths (1..{len(m.point_depths_m)}) [m]: {sim}")
        print(f"  basin sill depths              [m]: {sills}")
    print()
    print(
        f"{'case':10} {'backend':24} {'wall_s':>8} {'steps':>8} {'steps/s':>10} "
        f"{'cell-steps/s':>13} {'determ':>7}"
    )
    for p in all_perf:
        det = "-" if p.deterministic is None else str(p.deterministic)
        print(
            f"{p.case:10} {p.backend:24} {p.median_wall_s:8.3f} {p.steps_total:8d} "
            f"{p.steps_per_s:10.1f} {p.cell_steps_per_s:13.3e} {det:>7}"
        )
        if p.deterministic is False:
            print(f"  !! {p.determinism_detail}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EA Test 2 floodplain-depressions benchmark (published May-2010 dataset)"
    )
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help=f"CSV of solver_impl names {AVAILABLE_BACKENDS} "
        f"(default: Vulkan reference, runnable without a second backend)",
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--nx", type=int, default=NX_DEFAULT)
    parser.add_argument("--ny", type=int, default=NY_DEFAULT)
    parser.add_argument("--t-end", type=float, default=T_END_S)
    parser.add_argument("--dt-max", type=float, default=DT_MAX_DEFAULT)
    parser.add_argument("--cfl-interval", type=int, default=CFL_INTERVAL_DEFAULT)
    parser.add_argument("--cfl", type=float, default=None, help="CFL number")
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
    gauges = _load_gauges(spec.gauges_path)
    capacity = sum(cap for _sill, _pool, cap in _gauge_basins(spec, _bed_grid(spec), gauges))
    print(
        f"[hydrograph] published inflow, injected volume = {_injected_volume(spec):.0f} m^3; "
        f"t_end {spec.t_end_s / 3600.0:.0f} h"
    )
    print(
        f"[terrain] {gauges.shape[0]} depressions hold {capacity:.0f} m^3 below their sills "
        f"({_injected_volume(spec) / capacity:.0%} of that is injected)"
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
            "above_sill_m": GATE_ABOVE_SILL_M,
            "full_basin_capacity_rel": GATE_FULL_BASIN_CAPACITY_REL,
            "ponded_storage_frac": GATE_PONDED_STORAGE_FRAC,
            "far_column_dry_m": GATE_FAR_COLUMN_DRY_M,
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
    return 0 if (correctness_ok and determinism_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
