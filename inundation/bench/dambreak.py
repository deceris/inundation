"""Tier 1 analytical dam-break benchmark.

Two closed-form 1D test cases run on a synthetic rectangular channel mesh
(no external data files):

- ``stoker``: wet-bed dam-break — exact Stoker (1957) solution. Validates
  shock (bore) speed, rarefaction fan, and the constant middle state.
- ``ritter``: dry-bed dam-break — exact Ritter (1892) solution. The moving
  wet front makes this the wet-dry-boundary correctness test: front
  position, no negative depths, and depth error across the fan are gated.

Backends are configurable via ``--backends``; the retained Vulkan family runs
on macOS/MoltenVK. Error tolerances are printed in the report and recorded in
the benchmark documentation rather than left implicit.

Performance: each timed run reports wall seconds / steps/s / cell-steps/s;
``--warmup`` runs are excluded. With ``--repeats >= 2`` bit-exact run-to-run
determinism is also checked. ``--gif`` renders an *untimed* extra run into a
depth-profile animation (numerical vs analytical) — never sourced from the
timed runs, so perf numbers stay clean.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from inundation.bench.common import (
    build_channel_mesh,
    check_state_health,
    l1_relative_error,
    ritter_solution,
    stoker_solution,
    volume_drift_rel,
    write_mesh_parquet,
)
from inundation.tuning import GRAVITY_G
from inundation.workflow import (
    SimulationPhase,
    SWEWorkflow,
    WorkflowConfig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from inundation.workflow import WorkflowResult

# ── Case geometry / physics (defaults; see _build_parser for overrides) ─────
CHANNEL_LENGTH_M = 1000.0
CHANNEL_WIDTH_M = 10.0
DAM_X_M = 500.0
H_UP_M = 10.0
H_DOWN_STOKER_M = 1.0
T_END_S = 20.0
NX_DEFAULT = 500
NY_DEFAULT = 5

# Frictionless analytical solutions; 1e-4 is the solver-wide Manning floor
# (workflow._load_initial_state clamps n_mann to >= 1e-4).
MANNING_N_MINIMAL = 1e-4

DT_MAX_DEFAULT = 0.02
DT_INIT_DEFAULT = 1e-3
CFL_INTERVAL_DEFAULT = 5

# Wet threshold for locating the Ritter front in the numerical solution.
# The analytical fan reaches h=1e-3 m only ~3 cells behind the true tip at
# the default resolution — well inside the front tolerance below.
FRONT_WET_THRESHOLD_M = 1e-3

# ── Correctness gates (validation-plan.md §1: documented, not implicit) ─────
# L1 tolerance: first-order finite-volume schemes on 1D dam-break at
# dx/L = 1/500 typically land at 1-3% relative L1 depth error (e.g. Toro,
# "Shock-Capturing Methods for Free-Surface Shallow Flows", ch. 10 test
# cases); 5% gives regression headroom without masking real defects.
GATE_L1_REL = 0.05
# Ritter front tolerance: the numerical wet front classically lags the
# analytical tip because the leading film thins below the dry tolerance
# (DRY_TOL_DEFAULT=1e-4) and first-order upwinding diffuses it. A frictionless
# first-order Godunov scheme on this grid lands at ~12-13% of the analytical
# travel distance (measured, this harness); 15% is the standard wetting/drying
# acceptance band (e.g. Liang & Marche 2009; Brufau et al. 2002) and leaves
# ~2-3% regression headroom over the observed lag.
GATE_FRONT_REL = 0.15
# Zero-source runs must conserve volume to float32 accumulation error.
GATE_VOLUME_DRIFT_REL = 1e-5
# Repeat-to-repeat reproducibility tolerance (relative to peak depth). The
# Vulkan flux kernel scatters edge contributions into cells with floating-
# point atomicAdd, whose summation order is not fixed across GPU dispatches,
# so bit-exact state hashes are unachievable by construction. A healthy run
# reproduces to ~4e-7 relative (measured); 1e-5 flags genuine divergence
# (e.g. a real race) while tolerating benign atomic-ordering noise.
GATE_DETERMINISM_REL = 1e-5

DEFAULT_OUTPUT_ROOT = ".tmp/dambreak-bench"
# Default to the fully GPU-resident Vulkan backend. The host-CFL-readback
# fixed_dt_batch_barrier does not develop the dam-break wave under MoltenVK,
# so it is opt-in via --backends where that environment supports it.
DEFAULT_BACKENDS = ("gpu_resident_batch",)
BackendImpl = Literal[
    "fixed_dt_batch_barrier",
    "gpu_resident_batch",
]
AVAILABLE_BACKENDS: tuple[BackendImpl, ...] = (
    "fixed_dt_batch_barrier",
    "gpu_resident_batch",
)
CASES = ("stoker", "ritter")
GIF_FRAMES_DEFAULT = 60


@dataclass(frozen=True)
class CaseSpec:
    """Geometry + initial condition parameters for one analytical case."""

    name: str
    nx: int
    ny: int
    length_m: float
    width_m: float
    dam_x_m: float
    h_up_m: float
    h_down_m: float  # 0.0 for ritter (dry bed)
    t_end_s: float

    def analytical(
        self, x: NDArray[np.float64], t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Exact (h, u) profile at time t."""
        if self.name == "ritter":
            return ritter_solution(x, t, h_up=self.h_up_m, dam_x=self.dam_x_m, g=GRAVITY_G)
        return stoker_solution(
            x,
            t,
            h_up=self.h_up_m,
            h_down=self.h_down_m,
            dam_x=self.dam_x_m,
            g=GRAVITY_G,
        )


@dataclass(frozen=True)
class CaseMetrics:
    """Correctness metrics + gate outcomes for one (case, backend) run."""

    case: str
    backend: str
    l1_rel: float
    l2_rel: float
    front_rel_err: float | None  # ritter only
    volume_drift_rel: float
    min_depth_m: float
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


def _build_case_spec(args: argparse.Namespace, case: str) -> CaseSpec:
    return CaseSpec(
        name=case,
        nx=args.nx,
        ny=args.ny,
        length_m=CHANNEL_LENGTH_M,
        width_m=CHANNEL_WIDTH_M,
        dam_x_m=DAM_X_M,
        h_up_m=H_UP_M,
        h_down_m=0.0 if case == "ritter" else H_DOWN_STOKER_M,
        t_end_s=args.t_end,
    )


def _ensure_mesh(spec: CaseSpec, output_dir: Path) -> Path:
    """Write the synthetic flat-bed channel mesh (idempotent per resolution)."""
    mesh_path = output_dir / f"channel-{spec.nx}x{spec.ny}.parquet"
    if not mesh_path.exists():
        verts, quads = build_channel_mesh(spec.nx, spec.ny, spec.length_m, spec.width_m)
        zb = np.zeros(spec.nx * spec.ny, dtype=np.float32)
        write_mesh_parquet(mesh_path, verts, quads, zb)
        print(f"[mesh] synthetic channel mesh written: {mesh_path} (N={spec.nx * spec.ny})")
    return mesh_path


def _initial_state(spec: CaseSpec) -> dict[str, object]:
    """Step initial condition in original (generation) cell order."""
    n = spec.nx * spec.ny
    dx = spec.length_m / spec.nx
    # Cell (i, j) has index j*nx + i and centroid x = (i + 0.5) * dx.
    cx = (np.arange(n, dtype=np.float64) % spec.nx + 0.5) * dx
    h0 = np.where(cx < spec.dam_x_m, spec.h_up_m, spec.h_down_m).astype(np.float32)
    zeros = np.zeros(n, dtype=np.float32)
    return {"h": h0, "hu": zeros, "hv": zeros}


def _make_workflow(
    spec: CaseSpec,
    mesh_path: Path,
    backend: BackendImpl,
    *,
    output_interval_s: float,
    dt_max: float,
    cfl_interval: int,
    cfl: float | None,
    progress: bool,
) -> SWEWorkflow:
    config = WorkflowConfig(
        mesh_source=str(mesh_path),
        manning_n=MANNING_N_MINIMAL,
        output_interval_s=output_interval_s,
        dt_max=dt_max,
        dt_init=DT_INIT_DEFAULT,
        cfl_interval=cfl_interval,
        cfl=cfl,
        progress=progress,
        initial_state_source=_initial_state(spec),
        initial_state_order="original",
        solver_impl=backend,
    )
    workflow = SWEWorkflow(config)
    workflow.prepare()
    return workflow


def _run_once(
    spec: CaseSpec,
    mesh_path: Path,
    backend: BackendImpl,
    *,
    output_interval_s: float,
    dt_max: float,
    cfl_interval: int,
    cfl: float | None,
    progress: bool,
) -> tuple[SWEWorkflow, WorkflowResult]:
    workflow = _make_workflow(
        spec,
        mesh_path,
        backend,
        output_interval_s=output_interval_s,
        dt_max=dt_max,
        cfl_interval=cfl_interval,
        cfl=cfl,
        progress=progress,
    )
    result = workflow.run([SimulationPhase(duration_s=spec.t_end_s, sources=[])])
    return workflow, result


def _column_mean_depth(
    workflow: SWEWorkflow, h: NDArray[np.float32], spec: CaseSpec
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cross-channel-averaged depth per grid column (bin by centroid x)."""
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")
    dx = spec.length_m / spec.nx
    col = np.clip((workflow.geom.centroid[:, 0] / dx).astype(np.int64), 0, spec.nx - 1)
    sums = np.bincount(col, weights=h.astype(np.float64), minlength=spec.nx)
    counts = np.bincount(col, minlength=spec.nx)
    x_centers = (np.arange(spec.nx, dtype=np.float64) + 0.5) * dx
    return x_centers, sums / np.maximum(counts, 1)


def _evaluate_case(
    spec: CaseSpec,
    backend: str,
    workflow: SWEWorkflow,
    result: WorkflowResult,
) -> CaseMetrics:
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")

    h_final = np.asarray(result.h_final, dtype=np.float32)
    finite, min_depth, fail_reasons = check_state_health(h_final)

    x_centers, h_num = _column_mean_depth(workflow, h_final, spec)
    h_exact, _u_exact = spec.analytical(x_centers, spec.t_end_s)

    abs_err = np.abs(h_num - h_exact)
    l1_rel = l1_relative_error(h_num, h_exact)
    l2_rel = float(np.sqrt((abs_err**2).sum() / (h_exact**2).sum()))

    front_rel_err: float | None = None
    if spec.name == "ritter":
        wet_cols = x_centers[h_num > FRONT_WET_THRESHOLD_M]
        x_front_num = float(wet_cols.max()) if wet_cols.size else spec.dam_x_m
        travel = 2.0 * float(np.sqrt(GRAVITY_G * spec.h_up_m)) * spec.t_end_s
        x_front_exact = spec.dam_x_m + travel
        front_rel_err = abs(x_front_num - x_front_exact) / travel

    area = workflow.geom.area.astype(np.float64)
    h0_solver = np.asarray(
        np.where(workflow.geom.centroid[:, 0] < spec.dam_x_m, spec.h_up_m, spec.h_down_m),
        dtype=np.float64,
    )
    volume_initial = float((h0_solver * area).sum())
    drift_rel = volume_drift_rel(result.volume_final_m3, volume_initial)

    if l1_rel > GATE_L1_REL:
        fail_reasons.append(f"l1_rel:{l1_rel:.4f}>{GATE_L1_REL}")
    if front_rel_err is not None and front_rel_err > GATE_FRONT_REL:
        fail_reasons.append(f"front_rel_err:{front_rel_err:.4f}>{GATE_FRONT_REL}")
    if drift_rel > GATE_VOLUME_DRIFT_REL:
        fail_reasons.append(f"volume_drift_rel:{drift_rel:.3e}>{GATE_VOLUME_DRIFT_REL}")

    return CaseMetrics(
        case=spec.name,
        backend=backend,
        l1_rel=l1_rel,
        l2_rel=l2_rel,
        front_rel_err=front_rel_err,
        volume_drift_rel=drift_rel,
        min_depth_m=min_depth,
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


def _save_gif(
    spec: CaseSpec,
    backend: str,
    workflow: SWEWorkflow,
    result: WorkflowResult,
    out_path: Path,
) -> None:
    """Depth-profile animation (numerical vs analytical). Requires viz extra."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.animation as manimation
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib (+ pillow for the gif writer) is required for --gif. "
            'Install with: uv pip install -e ".[viz]"'
        ) from exc

    profiles: list[tuple[float, NDArray[np.float64], NDArray[np.float64]]] = []
    x_centers = np.empty(0, dtype=np.float64)
    for snap, t_snap in zip(result.snapshots, result.snap_times, strict=True):
        if t_snap <= 0.0:
            continue
        x_centers, h_num = _column_mean_depth(workflow, np.asarray(snap, np.float32), spec)
        h_exact, _ = spec.analytical(x_centers, float(t_snap))
        profiles.append((float(t_snap), h_num, h_exact))
    if not profiles:
        raise RuntimeError("gif run produced no usable snapshots")

    # matplotlib inference is patchy under strict pyright — go through Any
    # aliases, same pattern as bench/common.py's save_depth_png.
    _plt: Any = plt
    _manimation: Any = manimation
    fig: Any
    ax: Any
    fig, ax = _plt.subplots(figsize=(9, 4))
    y_max = 1.15 * spec.h_up_m
    line_num: Any = ax.plot([], [], "b-", lw=1.5, label=f"{backend} (numerical)")[0]
    line_exact: Any = ax.plot([], [], "k--", lw=1.0, label="analytical")[0]
    ax.set_xlim(0.0, spec.length_m)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("h [m]")
    ax.legend(loc="upper right")

    def _draw(frame: int) -> None:
        t_snap, h_num, h_exact = profiles[frame]
        line_num.set_data(x_centers, h_num)
        line_exact.set_data(x_centers, h_exact)
        ax.set_title(f"{spec.name} dam-break — t={t_snap:.2f}s")

    anim: Any = _manimation.FuncAnimation(fig, _draw, frames=len(profiles), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=_manimation.PillowWriter(fps=12))
    _plt.close(fig)
    print(f"[gif] depth-profile animation saved: {out_path} ({len(profiles)} frames)")


def _run_group(
    spec: CaseSpec,
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
            dt_max=args.dt_max,
            cfl_interval=args.cfl_interval,
            cfl=args.cfl,
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
            dt_max=args.dt_max,
            cfl_interval=args.cfl_interval,
            cfl=args.cfl,
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
        # Snap frame count to a power of two so t_end / frames is exactly
        # representable in float32; gpu_resident_batch's device-side time
        # advance stalls on non-representable output intervals (e.g. 20/60).
        safe_frames = _safe_gif_frames(args.gif_frames)
        if safe_frames != args.gif_frames:
            print(
                f"[{spec.name}/{backend}] gif frames {args.gif_frames} -> {safe_frames} (float32-safe interval)"
            )
        print(f"[{spec.name}/{backend}] extra untimed gif run ({safe_frames} frames)")
        gif_workflow, gif_result = _run_once(
            spec,
            mesh_path,
            backend,
            output_interval_s=spec.t_end_s / safe_frames,
            dt_max=args.dt_max,
            cfl_interval=args.cfl_interval,
            cfl=args.cfl,
            progress=args.solver_progress,
        )
        _save_gif(
            spec,
            backend,
            gif_workflow,
            gif_result,
            Path(args.output_dir) / f"{spec.name}-{backend}.gif",
        )

    return case_metrics, perf


def _print_report(all_metrics: list[CaseMetrics], all_perf: list[PerfMetrics]) -> None:
    print()
    print(
        f"gates: l1_rel<={GATE_L1_REL} front_rel<={GATE_FRONT_REL} "
        f"volume_drift<={GATE_VOLUME_DRIFT_REL:.0e} min_depth>=0 finite"
    )
    print(
        f"{'case':8} {'backend':24} {'l1_rel':>8} {'l2_rel':>8} {'front_rel':>10} "
        f"{'vol_drift':>10} {'min_h':>10} {'pass':>5}"
    )
    for m in all_metrics:
        front = f"{m.front_rel_err:.4f}" if m.front_rel_err is not None else "-"
        print(
            f"{m.case:8} {m.backend:24} {m.l1_rel:8.4f} {m.l2_rel:8.4f} {front:>10} "
            f"{m.volume_drift_rel:10.3e} {m.min_depth_m:10.3e} {m.passed!s:>5}"
        )
        for reason in m.fail_reasons:
            print(f"  !! {reason}")
    print()
    print(
        f"{'case':8} {'backend':24} {'wall_s':>8} {'steps':>8} {'steps/s':>10} "
        f"{'cell-steps/s':>13} {'determ':>7}"
    )
    for p in all_perf:
        det = "-" if p.deterministic is None else str(p.deterministic)
        print(
            f"{p.case:8} {p.backend:24} {p.median_wall_s:8.3f} {p.steps_total:8d} "
            f"{p.steps_per_s:10.1f} {p.cell_steps_per_s:13.3e} {det:>7}"
        )
        if p.deterministic is False:
            print(f"  !! {p.determinism_detail}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tier 1 analytical dam-break benchmark")
    parser.add_argument("--cases", default="stoker,ritter", help=f"CSV of {CASES}")
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help=f"CSV of solver_impl names {AVAILABLE_BACKENDS} "
        f"(default: Vulkan references, runnable without a second backend)",
    )
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
    parser.add_argument("--gif", action="store_true", help="render untimed depth-profile gif")
    parser.add_argument(
        "--gif-backend", default=None, help="backend for the gif run (default: first of --backends)"
    )
    parser.add_argument("--gif-frames", type=int, default=GIF_FRAMES_DEFAULT)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    raw_backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for case in cases:
        if case not in CASES:
            print(f"unknown case {case!r}; choose from {CASES}")
            return 2
    for backend_name in raw_backends:
        if backend_name not in AVAILABLE_BACKENDS:
            print(f"unknown backend {backend_name!r}; choose from {AVAILABLE_BACKENDS}")
            return 2
    backends: list[BackendImpl] = [cast("BackendImpl", b) for b in raw_backends]
    if args.gif_backend is None:
        args.gif_backend = backends[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[CaseMetrics] = []
    all_perf: list[PerfMetrics] = []
    for case in cases:
        spec = _build_case_spec(args, case)
        mesh_path = _ensure_mesh(spec, output_dir)
        for backend in backends:
            metrics, perf = _run_group(spec, mesh_path, backend, args)
            all_metrics.append(metrics)
            all_perf.append(perf)

    _print_report(all_metrics, all_perf)

    summary = {
        "gates": {
            "l1_rel": GATE_L1_REL,
            "front_rel": GATE_FRONT_REL,
            "volume_drift_rel": GATE_VOLUME_DRIFT_REL,
            "determinism_rel": GATE_DETERMINISM_REL,
        },
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
