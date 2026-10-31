"""Tier 1 radially symmetric dam-break benchmark.

The circular dam-break: a cylindrical column of water (radius ``r_dam``,
depth ``h_in``) collapses over a flat frictionless bed into a shallower
still layer (``h_out``). Unlike the 1D channel dam-break, the outward bore
and inward rarefaction spread *radially*, so this case exercises 2D flux
directionality and grid-orientation isotropy — a circular front on a
Cartesian mesh reveals mesh-imprint/anisotropy bugs a channel-aligned
shock cannot.

No closed form exists. The reference is a fine-grid 1D axisymmetric
finite-volume solve (``bench/common.solve_radial_dambreak``); the coarse 2D
solver's depth is radially binned onto the same radii and gated against it.
Gates are explicit: depth L1/L2 against
the radial reference, bore-front radius error, front-radius isotropy across
angular sectors, volume drift, positivity, and run-to-run reproducibility.

Backends are configurable via ``--backends``; the retained Vulkan family runs
on macOS/MoltenVK. ``--gif`` renders an *untimed* extra run into a radial
depth-profile animation (numerical vs reference); timed runs are never the
frame source, so perf stays clean.
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
    save_depth_gif,
    solve_radial_dambreak,
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
DOMAIN_L_M = 40.0
CENTER_M = DOMAIN_L_M / 2.0
R_DAM_M = 2.5
H_IN_M = 2.5
H_OUT_M = 0.5
# 1.5 s is exactly representable in float32 (so gpu_resident_batch's device
# time-advance stops cleanly) and keeps the bore well clear of the walls.
T_END_S = 1.5
NX_DEFAULT = 200
NY_DEFAULT = 200

# Frictionless analytical reference; 1e-4 is the solver-wide Manning floor.
MANNING_N_MINIMAL = 1e-4

DT_MAX_DEFAULT = 0.05
DT_INIT_DEFAULT = 1e-3
CFL_INTERVAL_DEFAULT = 5

REF_N_CELLS = 2000
# Bore front = radius of the outward shock; 1.2*h_out sits above the
# undisturbed outer layer but below the (higher) post-bore plateau, so it
# locates the leading edge robustly in both reference and numerical fields.
FRONT_THRESHOLD_FACTOR = 1.2
N_SECTORS = 12

# ── Correctness gates (validation-plan.md §1: documented, not implicit) ─────
# Whole-domain relative L1 depth error vs the radial reference. A first-order
# FV scheme resolving a *circular* bore on a Cartesian grid carries the 1D
# dry/shock error plus grid-imprint (staircasing) of the front; the circular
# dam-break literature puts first-order L1 in the ~3-8% band (e.g. Toro,
# "Shock-Capturing Methods for Free-Surface Shallow Flows", circular
# dam-break test). 0.10 leaves regression headroom over the observed value.
GATE_L1_REL = 0.10
# Bore-front radius error vs the reference front (radius of the outward
# shock). First-order fronts lag slightly; 5% of the reference radius.
GATE_FRONT_REL = 0.05
# Front-radius isotropy: relative spread of the bore-front radius across
# N_SECTORS angular sectors. On a Cartesian grid the circular front is
# slightly faster along the axes than the diagonals (4-fold grid imprint);
# this gate is the core 2D directional-correctness check. Calibrated with
# headroom over the measured axis/diagonal spread.
GATE_ISOTROPY = 0.08
# Zero-source runs must conserve volume to float32 accumulation error.
GATE_VOLUME_DRIFT_REL = 1e-5
# Repeat-to-repeat reproducibility tolerance (relative to peak depth). The
# Vulkan flux kernel scatters edge contributions with floating-point
# atomicAdd whose summation order is not fixed across dispatches, so bit-exact
# hashes are unachievable; a healthy run reproduces to ~1e-6 relative.
GATE_DETERMINISM_REL = 1e-5

DEFAULT_OUTPUT_ROOT = ".tmp/radial-dambreak-bench"
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
GIF_FRAMES_DEFAULT = 60


@dataclass(frozen=True)
class RadialSpec:
    """Geometry + initial condition parameters for the circular dam-break."""

    name: str
    nx: int
    ny: int
    domain_l_m: float
    center_m: float
    r_dam_m: float
    h_in_m: float
    h_out_m: float
    t_end_s: float

    @property
    def dx_m(self) -> float:
        return self.domain_l_m / self.nx

    @property
    def r_max_compare_m(self) -> float:
        # Inscribed-circle radius: annuli at r < L/2 are fully inside the
        # square domain, so radial averaging is unbiased there.
        return self.center_m

    @property
    def n_radial_bins(self) -> int:
        return max(1, round(self.r_max_compare_m / self.dx_m))

    @property
    def front_threshold_m(self) -> float:
        return FRONT_THRESHOLD_FACTOR * self.h_out_m


@dataclass(frozen=True)
class Reference:
    """Cached radial reference profile + bore-front radius at ``t_end``."""

    r_centers: NDArray[np.float64]
    h_binned: NDArray[np.float64]
    front_r: float


@dataclass(frozen=True)
class CaseMetrics:
    """Correctness metrics + gate outcomes for one (case, backend) run."""

    case: str
    backend: str
    l1_rel: float
    l2_rel: float
    front_rel_err: float
    isotropy_spread: float
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


def _build_spec(args: argparse.Namespace) -> RadialSpec:
    return RadialSpec(
        name="circular",
        nx=args.nx,
        ny=args.ny,
        domain_l_m=DOMAIN_L_M,
        center_m=CENTER_M,
        r_dam_m=R_DAM_M,
        h_in_m=H_IN_M,
        h_out_m=H_OUT_M,
        t_end_s=args.t_end,
    )


def _ensure_mesh(spec: RadialSpec, output_dir: Path) -> Path:
    """Write the synthetic flat-bed square mesh (idempotent per resolution)."""
    mesh_path = output_dir / f"square-{spec.nx}x{spec.ny}.parquet"
    if not mesh_path.exists():
        verts, quads = build_channel_mesh(spec.nx, spec.ny, spec.domain_l_m, spec.domain_l_m)
        zb = np.zeros(spec.nx * spec.ny, dtype=np.float32)
        write_mesh_parquet(mesh_path, verts, quads, zb)
        print(f"[mesh] synthetic square mesh written: {mesh_path} (N={spec.nx * spec.ny})")
    return mesh_path


def _initial_state(spec: RadialSpec) -> dict[str, object]:
    """Circular column initial condition in original (generation) cell order."""
    n = spec.nx * spec.ny
    dx = spec.dx_m
    idx = np.arange(n)
    cx = (idx % spec.nx + 0.5) * dx
    cy = (idx // spec.nx + 0.5) * dx
    r = np.sqrt((cx - spec.center_m) ** 2 + (cy - spec.center_m) ** 2)
    h0 = np.where(r < spec.r_dam_m, spec.h_in_m, spec.h_out_m).astype(np.float32)
    zeros = np.zeros(n, dtype=np.float32)
    return {"h": h0, "hu": zeros, "hv": zeros}


def _make_workflow(
    spec: RadialSpec,
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
    spec: RadialSpec,
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


def _cell_radius(workflow: SWEWorkflow, spec: RadialSpec) -> NDArray[np.float64]:
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")
    cx = workflow.geom.world_centroid[:, 0] - spec.center_m
    cy = workflow.geom.world_centroid[:, 1] - spec.center_m
    return np.sqrt(cx * cx + cy * cy)


def _radial_profile(
    workflow: SWEWorkflow, h: NDArray[np.float32], spec: RadialSpec
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Azimuthally averaged depth per radial bin (bins fully inside domain)."""
    r = _cell_radius(workflow, spec)
    nb = spec.n_radial_bins
    dr_bin = spec.r_max_compare_m / nb
    idx = (r / dr_bin).astype(np.int64)
    inside = idx < nb
    idx_in = idx[inside]
    h_in = h.astype(np.float64)[inside]
    sums = np.bincount(idx_in, weights=h_in, minlength=nb)
    counts = np.bincount(idx_in, minlength=nb)
    r_centers = (np.arange(nb, dtype=np.float64) + 0.5) * dr_bin
    mean = sums / np.maximum(counts, 1)
    return r_centers, mean, counts.astype(np.float64)


def _front_radius(
    r_centers: NDArray[np.float64],
    depth: NDArray[np.float64],
    valid: NDArray[np.float64],
    threshold: float,
    fallback: float,
) -> float:
    wet = (depth > threshold) & (valid > 0)
    return float(r_centers[wet].max()) if bool(wet.any()) else fallback


def _sector_front_radii(
    workflow: SWEWorkflow, h: NDArray[np.float32], spec: RadialSpec
) -> NDArray[np.float64]:
    """Bore-front radius per angular sector (for the isotropy check)."""
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")
    cx = workflow.geom.world_centroid[:, 0] - spec.center_m
    cy = workflow.geom.world_centroid[:, 1] - spec.center_m
    r = np.sqrt(cx * cx + cy * cy)
    theta = np.arctan2(cy, cx)  # [-pi, pi)
    sector = ((theta + np.pi) / (2.0 * np.pi) * N_SECTORS).astype(np.int64) % N_SECTORS

    dr_bin = 2.0 * spec.dx_m
    nb = max(1, round(spec.r_max_compare_m / dr_bin))
    r_centers = (np.arange(nb, dtype=np.float64) + 0.5) * dr_bin
    hd = h.astype(np.float64)
    inside = r < spec.r_max_compare_m

    fronts = np.empty(N_SECTORS, dtype=np.float64)
    for s in range(N_SECTORS):
        sel = inside & (sector == s)
        idx = (r[sel] / dr_bin).astype(np.int64)
        idx = np.clip(idx, 0, nb - 1)
        sums = np.bincount(idx, weights=hd[sel], minlength=nb)
        counts = np.bincount(idx, minlength=nb)
        mean = sums / np.maximum(counts, 1)
        fronts[s] = _front_radius(
            r_centers, mean, counts.astype(np.float64), spec.front_threshold_m, spec.r_dam_m
        )
    return fronts


def _build_reference(spec: RadialSpec) -> Reference:
    r_fine, h_fine, _u = solve_radial_dambreak(
        spec.t_end_s,
        h_in=spec.h_in_m,
        h_out=spec.h_out_m,
        r_dam=spec.r_dam_m,
        g=GRAVITY_G,
        r_max=spec.r_max_compare_m,
        n_cells=REF_N_CELLS,
    )
    nb = spec.n_radial_bins
    dr_bin = spec.r_max_compare_m / nb
    r_centers = (np.arange(nb, dtype=np.float64) + 0.5) * dr_bin
    h_binned = np.interp(r_centers, r_fine, h_fine)
    wet = h_fine > spec.front_threshold_m
    front_r = float(r_fine[wet].max()) if bool(wet.any()) else spec.r_dam_m
    return Reference(r_centers=r_centers, h_binned=h_binned, front_r=front_r)


def _evaluate_case(
    spec: RadialSpec,
    backend: str,
    workflow: SWEWorkflow,
    result: WorkflowResult,
    reference: Reference,
) -> CaseMetrics:
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")

    h_final = np.asarray(result.h_final, dtype=np.float32)
    finite, min_depth, fail_reasons = check_state_health(h_final)

    r_centers, h_num, counts = _radial_profile(workflow, h_final, spec)
    valid = counts > 0
    h_ref = reference.h_binned

    abs_err = np.abs(h_num - h_ref)[valid]
    ref_valid = h_ref[valid]
    l1_rel = l1_relative_error(h_num[valid], ref_valid)
    l2_rel = float(np.sqrt((abs_err**2).sum() / (ref_valid**2).sum()))

    front_num = _front_radius(r_centers, h_num, counts, spec.front_threshold_m, spec.r_dam_m)
    front_rel_err = abs(front_num - reference.front_r) / reference.front_r

    sector_fronts = _sector_front_radii(workflow, h_final, spec)
    mean_front = float(sector_fronts.mean())
    isotropy_spread = (
        float(sector_fronts.max() - sector_fronts.min()) / mean_front if mean_front > 0 else 0.0
    )

    area = workflow.geom.area.astype(np.float64)
    r_solver = _cell_radius(workflow, spec)
    h0_solver = np.where(r_solver < spec.r_dam_m, spec.h_in_m, spec.h_out_m).astype(np.float64)
    volume_initial = float((h0_solver * area).sum())
    drift_rel = volume_drift_rel(result.volume_final_m3, volume_initial)

    if l1_rel > GATE_L1_REL:
        fail_reasons.append(f"l1_rel:{l1_rel:.4f}>{GATE_L1_REL}")
    if front_rel_err > GATE_FRONT_REL:
        fail_reasons.append(f"front_rel_err:{front_rel_err:.4f}>{GATE_FRONT_REL}")
    if isotropy_spread > GATE_ISOTROPY:
        fail_reasons.append(f"isotropy_spread:{isotropy_spread:.4f}>{GATE_ISOTROPY}")
    if drift_rel > GATE_VOLUME_DRIFT_REL:
        fail_reasons.append(f"volume_drift_rel:{drift_rel:.3e}>{GATE_VOLUME_DRIFT_REL}")

    return CaseMetrics(
        case=spec.name,
        backend=backend,
        l1_rel=l1_rel,
        l2_rel=l2_rel,
        front_rel_err=front_rel_err,
        isotropy_spread=isotropy_spread,
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
    spec: RadialSpec,
    backend: str,
    workflow: SWEWorkflow,
    result: WorkflowResult,
    out_path: Path,
) -> None:
    """Radial depth-profile animation (numerical vs reference). Requires viz extra."""
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
    r_centers = np.empty(0, dtype=np.float64)
    for snap, t_snap in zip(result.snapshots, result.snap_times, strict=True):
        if t_snap <= 0.0:
            continue
        r_centers, h_num, _counts = _radial_profile(workflow, np.asarray(snap, np.float32), spec)
        r_fine, h_fine, _u = solve_radial_dambreak(
            float(t_snap),
            h_in=spec.h_in_m,
            h_out=spec.h_out_m,
            r_dam=spec.r_dam_m,
            g=GRAVITY_G,
            r_max=spec.r_max_compare_m,
            n_cells=REF_N_CELLS,
        )
        h_ref = np.interp(r_centers, r_fine, h_fine)
        profiles.append((float(t_snap), h_num, h_ref))
    if not profiles:
        raise RuntimeError("gif run produced no usable snapshots")

    _plt: Any = plt
    _manimation: Any = manimation
    fig: Any
    ax: Any
    fig, ax = _plt.subplots(figsize=(9, 4))
    y_max = 1.15 * spec.h_in_m
    line_num: Any = ax.plot([], [], "b-", lw=1.5, label=f"{backend} (numerical)")[0]
    line_ref: Any = ax.plot([], [], "k--", lw=1.0, label="radial reference")[0]
    ax.set_xlim(0.0, spec.r_max_compare_m)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("r [m]")
    ax.set_ylabel("h [m]")
    ax.legend(loc="upper right")

    def _draw(frame: int) -> None:
        t_snap, h_num, h_ref = profiles[frame]
        line_num.set_data(r_centers, h_num)
        line_ref.set_data(r_centers, h_ref)
        ax.set_title(f"{spec.name} dam-break — t={t_snap:.2f}s")

    anim: Any = _manimation.FuncAnimation(fig, _draw, frames=len(profiles), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=_manimation.PillowWriter(fps=12))
    _plt.close(fig)
    print(f"[gif] radial depth-profile animation saved: {out_path} ({len(profiles)} frames)")


def _run_group(
    spec: RadialSpec,
    mesh_path: Path,
    backend: BackendImpl,
    reference: Reference,
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

    case_metrics = _evaluate_case(spec, backend, workflow, result, reference)

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

    if (args.gif or args.gif_2d) and backend == args.gif_backend:
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
            dt_max=args.dt_max,
            cfl_interval=args.cfl_interval,
            cfl=args.cfl,
            progress=args.solver_progress,
        )
        if args.gif:
            _save_gif(
                spec,
                backend,
                gif_workflow,
                gif_result,
                Path(args.output_dir) / f"{spec.name}-{backend}.gif",
            )
        if args.gif_2d:
            save_depth_gif(
                gif_workflow,
                gif_result.snapshots,
                gif_result.snap_times,
                Path(args.output_dir) / f"{spec.name}-{backend}-2d.gif",
                vmax=spec.h_in_m,
                title_prefix=f"{spec.name} dam-break — ",
            )

    return case_metrics, perf


def _print_report(all_metrics: list[CaseMetrics], all_perf: list[PerfMetrics]) -> None:
    print()
    print(
        f"gates: l1_rel<={GATE_L1_REL} front_rel<={GATE_FRONT_REL} "
        f"isotropy<={GATE_ISOTROPY} volume_drift<={GATE_VOLUME_DRIFT_REL:.0e} min_depth>=0 finite"
    )
    print(
        f"{'case':10} {'backend':24} {'l1_rel':>8} {'l2_rel':>8} {'front_rel':>10} "
        f"{'isotropy':>9} {'vol_drift':>10} {'min_h':>10} {'pass':>5}"
    )
    for m in all_metrics:
        print(
            f"{m.case:10} {m.backend:24} {m.l1_rel:8.4f} {m.l2_rel:8.4f} "
            f"{m.front_rel_err:10.4f} {m.isotropy_spread:9.4f} "
            f"{m.volume_drift_rel:10.3e} {m.min_depth_m:10.3e} {m.passed!s:>5}"
        )
        for reason in m.fail_reasons:
            print(f"  !! {reason}")
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
        description="Tier 1 radially symmetric (circular) dam-break benchmark"
    )
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help=f"CSV of solver_impl names {AVAILABLE_BACKENDS} "
        f"(default: Vulkan reference, runnable without a second backend)",
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
    parser.add_argument("--gif", action="store_true", help="render untimed radial-profile gif")
    parser.add_argument(
        "--gif-2d", action="store_true", help="render untimed top-down 2D depth-heatmap gif"
    )
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
    print(f"[reference] solving fine radial reference (n_cells={REF_N_CELLS}) at t={spec.t_end_s}s")
    reference = _build_reference(spec)
    print(f"[reference] bore front radius = {reference.front_r:.3f} m")

    all_metrics: list[CaseMetrics] = []
    all_perf: list[PerfMetrics] = []
    for backend in backends:
        metrics, perf = _run_group(spec, mesh_path, backend, reference, args)
        all_metrics.append(metrics)
        all_perf.append(perf)

    _print_report(all_metrics, all_perf)

    summary = {
        "gates": {
            "l1_rel": GATE_L1_REL,
            "front_rel": GATE_FRONT_REL,
            "isotropy": GATE_ISOTROPY,
            "volume_drift_rel": GATE_VOLUME_DRIFT_REL,
            "determinism_rel": GATE_DETERMINISM_REL,
        },
        "reference_front_r_m": reference.front_r,
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
