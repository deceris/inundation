"""Lake-at-rest (C-property) port-parity benchmark. See LAKE_AT_REST_TEST.md."""

from __future__ import annotations

import argparse
import hashlib
import platform
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from inundation.mesh.geometry import MeshGeometry, build_geometry, hilbert_reorder
from inundation.tuning import CFL_DEFAULT, DRY_TOL_DEFAULT, GRAVITY_G
from inundation.vulkan.fixed_dt_batch_barrier import SWESolverFixedDtBatchBarrier
from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch
from inundation.workflow import (
    _compile_solver_shaders,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from inundation.vulkan.solver import SWESolver

NX = NY = 128
G = GRAVITY_G
DRY_TOL = DRY_TOL_DEFAULT
CFL = CFL_DEFAULT
WORKGROUP_SIZE = 256
DT_MAX = 0.05
DT_INIT = 1e-2
CFL_INTERVAL = 10
T_END = 600.0
OUTPUT_INTERVAL_S = 60.0

VARIANT_ETA: dict[str, float] = {"A": 1.0, "B": 0.5}
FRICTION_N: dict[str, float] = {"minimal": 1e-4, "production": 0.035}
BACKENDS: dict[str, type[SWESolver]] = {
    "fixed_dt_batch_barrier": SWESolverFixedDtBatchBarrier,
    "gpu_resident_batch": SWESolverGpuResidentBatch,
}

ESCALATION_MAX_ABS_U = 0.01  # m/s, production-friction variants


@dataclass(frozen=True)
class RunResult:
    """Metrics for a single (variant, friction, backend) run."""

    variant: str
    friction: str
    backend: str
    max_abs_hu: float
    max_abs_u: float
    volume_drift: float
    h_rmse_vs_initial: float
    blew_up: bool = False
    blow_up_detail: str = ""
    state_hash: str = ""


def build_mesh() -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Regular quad grid on the unit square (verts, quads), matches doc §3.1."""
    xs = np.linspace(0.0, 1.0, NX + 1)
    ys = np.linspace(0.0, 1.0, NY + 1)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    verts = np.column_stack([xv.ravel(), yv.ravel()]).astype(np.float32)

    def vid(i: int, j: int) -> int:
        return j * (NX + 1) + i

    quads = np.array(
        [
            [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            for j in range(NY)
            for i in range(NX)
        ],
        dtype=np.int32,
    )
    return verts, quads


def build_bed(xs: NDArray[np.float32], ys: NDArray[np.float32]) -> NDArray[np.float32]:
    """Smooth Gaussian bump, crest 0.8 m, matches doc §3.2."""
    ccx, ccy = np.meshgrid(0.5 * (xs[:-1] + xs[1:]), 0.5 * (ys[:-1] + ys[1:]), indexing="xy")
    r2 = (ccx - 0.5) ** 2 + (ccy - 0.5) ** 2
    return (0.8 * np.exp(-50.0 * r2)).ravel().astype(np.float32)


def build_case(
    variant: str, friction: str, *, flat_bed: bool = False
) -> tuple[MeshGeometry, NDArray[np.float32], NDArray[np.float32]]:
    """Build (geom, h0, n0) for a given variant/friction combination.

    flat_bed=True builds zb=0 everywhere: a true trivial-rest control case,
    used to isolate whether divergence is a bed-slope/well-balance issue or
    a kernel defect independent of bathymetry (see LAKE_AT_REST_BASELINE_RESULTS.md).
    """
    xs = np.linspace(0.0, 1.0, NX + 1)
    ys = np.linspace(0.0, 1.0, NY + 1)
    verts, quads = build_mesh()
    zb = np.zeros((NX * NY,), dtype=np.float32) if flat_bed else build_bed(xs, ys)

    geom = build_geometry(verts, quads, zb_from_file=zb)
    geom, _perm = hilbert_reorder(geom, verbose=False)

    eta = VARIANT_ETA[variant]
    h0 = np.maximum(eta - geom.zb, 0.0).astype(np.float32)
    n0 = np.full(geom.N, FRICTION_N[friction], dtype=np.float32)
    return geom, h0, n0


def run_case(
    backend_name: str,
    variant: str,
    friction: str,
    *,
    t_end: float = T_END,
    output_interval_s: float = OUTPUT_INTERVAL_S,
    progress: bool = True,
    dt_init: float = DT_INIT,
    dt_max: float = DT_MAX,
    cfl_interval: int = CFL_INTERVAL,
    trace: bool = False,
    flat_bed: bool = False,
    hash_state: bool = False,
) -> RunResult:
    """Build the synthetic lake-at-rest case and run one solver backend on it."""
    solver_cls = BACKENDS[backend_name]
    geom, h0, n0 = build_case(variant, friction, flat_bed=flat_bed)
    spv = _compile_solver_shaders()
    zeros = np.zeros(geom.N, dtype=np.float32)

    print(f"[{backend_name}] variant={variant} friction={friction}: building solver...", flush=True)
    solver = solver_cls(
        geom,
        spv,
        h0,
        zeros,
        zeros,
        n0,
        g=G,
        dry_tol=DRY_TOL,
        cfl=CFL,
        workgroup_size=WORKGROUP_SIZE,
    )
    print(
        f"[{backend_name}] variant={variant} friction={friction}: running "
        f"(t_end={t_end}s dt_init={dt_init} dt_max={dt_max} cfl_interval={cfl_interval})...",
        flush=True,
    )

    if not trace:
        solver.run(  # pyright: ignore[reportUnknownMemberType]
            t_end=t_end,
            output_interval_s=output_interval_s,
            dt_max=dt_max,
            dt_init=dt_init,
            cfl_interval=cfl_interval,
            progress=progress,
            source_rate=None,
        )
    else:
        # Drive run() in resumed chunks of output_interval_s so we can download
        # hu/hv/h between chunks and see where/when the magnitude grows, instead
        # of only inspecting the final (possibly already-diverged) state.
        t_sim = 0.0
        resume = False
        while t_sim < t_end:
            chunk_end = min(t_sim + output_interval_s, t_end)
            phase_result = solver.run(  # pyright: ignore[reportUnknownMemberType]
                t_end=chunk_end,
                output_interval_s=chunk_end - t_sim,
                dt_max=dt_max,
                dt_init=dt_init,
                cfl_interval=cfl_interval,
                progress=progress,
                source_rate=None,
                resume=resume,
                t_start=t_sim,
            )
            # run() returns (snapshots, times) or (snapshots, times, hu, hv).
            if len(phase_result) == 4:
                _, snap_times, _, _ = phase_result
            else:
                _, snap_times = phase_result
            resume = True
            t_sim = snap_times[-1]
            hu_t = solver._download(solver.t_hu)  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType,reportUnknownArgumentType]
            hv_t = solver._download(solver.t_hv)  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType,reportUnknownArgumentType]
            max_hu_t = float(max(np.abs(hu_t).max(), np.abs(hv_t).max()))
            print(f"  [trace] t={t_sim:.4f}s max_abs_hu={max_hu_t:.3e}", flush=True)

    hu = solver._download(solver.t_hu)  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType,reportUnknownArgumentType]
    hv = solver._download(solver.t_hv)  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType,reportUnknownArgumentType]
    h = solver.download_h()

    max_abs_hu = float(max(np.abs(hu).max(), np.abs(hv).max()))

    wet = h > 10 * DRY_TOL
    if wet.any():
        u = np.abs(hu[wet]) / h[wet]
        v = np.abs(hv[wet]) / h[wet]
        max_abs_u = float(max(u.max(), v.max()))
    else:
        max_abs_u = 0.0

    area = geom.area
    v_initial = float((h0 * area).sum())
    v_final = float((h * area).sum())
    volume_drift = abs(v_final - v_initial) / v_initial

    h_rmse_vs_initial = float(np.sqrt(np.mean((h[wet] - h0[wet]) ** 2))) if wet.any() else 0.0

    state_hash = ""
    if hash_state:
        digest = hashlib.sha256()
        digest.update(h.tobytes())
        digest.update(hu.tobytes())
        digest.update(hv.tobytes())
        state_hash = digest.hexdigest()

    return RunResult(
        variant=variant,
        friction=friction,
        backend=backend_name,
        max_abs_hu=max_abs_hu,
        max_abs_u=max_abs_u,
        volume_drift=volume_drift,
        h_rmse_vs_initial=h_rmse_vs_initial,
        state_hash=state_hash,
    )


def print_hardware_info() -> None:
    """Print hardware/OS/driver info for the report (doc §6.4)."""
    print(f"# platform: {platform.platform()}")
    print(f"# processor: {platform.processor()}")
    print(f"# python: {platform.python_version()}")


def print_header() -> None:
    """Print the report table header (doc §6)."""
    print(
        f"{'variant':7} {'friction':9} {'backend':14} "
        f"{'max_abs_hu':>11} {'max_abs_u':>10} {'volume_drift':>13}"
    )


def print_row(r: RunResult) -> None:
    """Print one report table row, plus any blow-up/escalation flag (doc §6)."""
    print(
        f"{r.variant:7} {r.friction:9} {r.backend:14} "
        f"{r.max_abs_hu:11.3e} {r.max_abs_u:10.3e} {r.volume_drift:13.3e}",
        flush=True,
    )
    if r.blew_up:
        print(f"  !! BLOW-UP: {r.blow_up_detail}", flush=True)
    elif r.friction == "production" and r.max_abs_u > ESCALATION_MAX_ABS_U:
        print(
            f"  !! ESCALATION: max_abs_u={r.max_abs_u:.3e} m/s exceeds "
            f"{ESCALATION_MAX_ABS_U} m/s threshold for variant {r.variant} on {r.backend}",
            flush=True,
        )
    if r.state_hash:
        print(f"  state_hash={r.state_hash}", flush=True)


def run_case_safe(
    backend_name: str,
    variant: str,
    friction: str,
    *,
    t_end: float = T_END,
    output_interval_s: float = OUTPUT_INTERVAL_S,
    progress: bool = True,
    dt_init: float = DT_INIT,
    dt_max: float = DT_MAX,
    cfl_interval: int = CFL_INTERVAL,
    trace: bool = False,
    flat_bed: bool = False,
    hash_state: bool = False,
) -> RunResult:
    """Run a case, capturing a CFL blow-up as a result row instead of crashing."""
    try:
        return run_case(
            backend_name,
            variant,
            friction,
            t_end=t_end,
            output_interval_s=output_interval_s,
            progress=progress,
            dt_init=dt_init,
            dt_max=dt_max,
            cfl_interval=cfl_interval,
            trace=trace,
            flat_bed=flat_bed,
            hash_state=hash_state,
        )
    except RuntimeError as exc:
        if "CFL dt too small" not in str(exc):
            raise
        return RunResult(
            variant=variant,
            friction=friction,
            backend=backend_name,
            max_abs_hu=float("nan"),
            max_abs_u=float("nan"),
            volume_drift=float("nan"),
            h_rmse_vs_initial=float("nan"),
            blew_up=True,
            blow_up_detail=str(exc),
        )


def main() -> None:
    """CLI entry point for the lake-at-rest benchmark."""
    parser = argparse.ArgumentParser(description="Lake-at-rest port-parity benchmark")
    parser.add_argument(
        "--backend",
        choices=[*list(BACKENDS), "all"],
        default="fixed_dt_batch_barrier",
        help=(
            "solver backend (fixed_dt_batch_barrier is the barrier reference; the "
            "unbarriered fixed_dt_batch raced and refuses to run)"
        ),
    )
    parser.add_argument(
        "--variant",
        choices=[*list(VARIANT_ETA), "all"],
        default="all",
        help="A=fully wet, B=partial dry",
    )
    parser.add_argument(
        "--friction",
        choices=[*list(FRICTION_N), "all"],
        default="all",
        help="minimal=1e-4, production=0.035",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="disable per-batch progress logging from the solver run() loop",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=T_END,
        help=f"simulated end time in seconds (doc default {T_END}s; short values speed iteration)",
    )
    parser.add_argument(
        "--output-interval",
        type=float,
        default=OUTPUT_INTERVAL_S,
        help=f"snapshot interval in seconds (doc default {OUTPUT_INTERVAL_S}s)",
    )
    parser.add_argument(
        "--dt-init",
        type=float,
        default=DT_INIT,
        help=f"initial dt override for CFL-sensitivity diagnosis (doc default {DT_INIT}s)",
    )
    parser.add_argument(
        "--dt-max",
        type=float,
        default=DT_MAX,
        help=f"max dt override (doc default {DT_MAX}s)",
    )
    parser.add_argument(
        "--cfl-interval",
        type=int,
        default=CFL_INTERVAL,
        help=f"steps per CFL recompute override (doc default {CFL_INTERVAL})",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="download hu/hv every output-interval chunk and print max_abs_hu trace "
        "(diagnostic, slower due to resume overhead)",
    )
    parser.add_argument(
        "--flat-bed",
        action="store_true",
        help="zb=0 everywhere (true trivial-rest control case, ignores the Gaussian "
        "bump) — isolates bed-slope/well-balance defect from a kernel-independent one",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="print sha256 of final h/hu/hv arrays for cross-run determinism comparison",
    )
    args = parser.parse_args()

    backends = list(BACKENDS) if args.backend == "all" else [args.backend]
    variants = list(VARIANT_ETA) if args.variant == "all" else [args.variant]
    frictions = list(FRICTION_N) if args.friction == "all" else [args.friction]

    print_hardware_info()
    print(
        f"# dt_init={args.dt_init} dt_max={args.dt_max} cfl_interval={args.cfl_interval} "
        f"flat_bed={args.flat_bed}",
        flush=True,
    )
    print_header()
    results: list[RunResult] = []
    for backend in backends:
        for variant in variants:
            for friction in frictions:
                r = run_case_safe(
                    backend,
                    variant,
                    friction,
                    t_end=args.t_end,
                    output_interval_s=args.output_interval,
                    progress=not args.quiet,
                    dt_init=args.dt_init,
                    dt_max=args.dt_max,
                    cfl_interval=args.cfl_interval,
                    trace=args.trace,
                    flat_bed=args.flat_bed,
                    hash_state=args.hash,
                )
                print_row(r)
                results.append(r)


if __name__ == "__main__":
    main()
