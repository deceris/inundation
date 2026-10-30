"""Thacker's planar surface in a parabolic bowl: repeated wetting and drying.

A flat water surface tilts and rotates in a frictionless paraboloid, so the
shoreline wets and dries every period, and the exact solution is known at every
time (Thacker 1981, J. Fluid Mech. 107). The velocity is uniform over the wet
region, which is what makes the case sensitive to momentum in cells at the
shoreline: the front fixes of effort 24 can only show up on a case like this.

Run::

    .venv/bin/python -m inundation.bench.parabolic_bowl --nx 200 --periods 3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from inundation.tuning import CFL_DEFAULT, GRAVITY_G

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry

BOWL_RADIUS_M = 1.0
CENTRE_DEPTH_M = 0.1
SWING_M = 0.5
DOMAIN_M = 4.0
MANNING_FLOOR = 1e-4
DRY_TOL = 1e-4
WET_THRESHOLD_M = 1e-3
FILM_DEPTH_M = 1e-3
GATE_NX = 200
GATE_DEPTH_L1 = 0.18
GATE_FILM_SPEED_MS = 1.0
GATE_VOLUME_DRIFT = 1e-7


def omega() -> float:
    """Angular frequency of the sloshing mode [rad/s]."""
    return math.sqrt(2.0 * GRAVITY_G * CENTRE_DEPTH_M) / BOWL_RADIUS_M


def period_s() -> float:
    return 2.0 * math.pi / omega()


def bed(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    return -CENTRE_DEPTH_M * (1.0 - (x * x + y * y) / BOWL_RADIUS_M**2)


def exact(
    x: NDArray[np.float64], y: NDArray[np.float64], t: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Exact depth and velocity components; velocity is zero where dry."""
    w = omega()
    c, s = math.cos(w * t), math.sin(w * t)
    scale = SWING_M * CENTRE_DEPTH_M / BOWL_RADIUS_M**2
    surface = scale * (2.0 * x * c + 2.0 * y * s - SWING_M)
    h = np.maximum(surface - bed(x, y), 0.0)
    wet = h > 0.0
    return h, np.where(wet, -SWING_M * w * s, 0.0), np.where(wet, SWING_M * w * c, 0.0)


def build_bowl(nx: int) -> MeshGeometry:
    from inundation.mesh.geometry import build_geometry, hilbert_reorder

    edges = np.linspace(-DOMAIN_M / 2, DOMAIN_M / 2, nx + 1)
    xv, yv = np.meshgrid(edges, edges, indexing="xy")
    verts = np.column_stack([xv.ravel(), yv.ravel()]).astype(np.float32)
    i, j = np.meshgrid(np.arange(nx), np.arange(nx), indexing="xy")
    a = (j * (nx + 1) + i).ravel()
    faces = np.column_stack([a, a + 1, a + nx + 2, a + nx + 1]).astype(np.int32)
    centres = 0.5 * (edges[:-1] + edges[1:])
    cx, cy = np.meshgrid(centres, centres, indexing="xy")
    zb = bed(cx.ravel(), cy.ravel()).astype(np.float32)
    return hilbert_reorder(build_geometry(verts, faces, zb_from_file=zb), verbose=False)[0]


@dataclass(frozen=True)
class BowlMetrics:
    """Errors against the exact solution; lower is better except ``steps``."""

    nx: int
    cfl: float
    periods: int
    steps: int
    depth_l1_rel_mean: float
    depth_l1_rel_final: float
    wet_cells_wrong_mean: float
    speed_l1_rel_final: float
    film_speed_max_ms: float
    volume_drift_rel: float
    clamp_rel: float


def run_case(
    spv: dict[str, bytes], nx: int, cfl: float, periods: int, *, compensated: bool = True
) -> BowlMetrics:
    from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch

    geom = build_bowl(nx)
    x, y = geom.centroid[:, 0].astype(np.float64), geom.centroid[:, 1].astype(np.float64)
    area = geom.area.astype(np.float64)
    h0, u0, v0 = exact(x, y, 0.0)
    solver = SWESolverGpuResidentBatch(
        geom,
        spv,
        h0.astype(np.float32),
        (h0 * u0).astype(np.float32),
        (h0 * v0).astype(np.float32),
        np.full(geom.N, MANNING_FLOOR, np.float32),
        dry_tol=DRY_TOL,
        cfl=cfl,
    )
    solver.compensated = compensated
    solver.track_clamp = True
    t_end = periods * period_s()
    result = solver.run(
        t_end=t_end,
        output_interval_s=t_end / (8 * periods),
        dt_max=0.05,
        dt_init=1e-3,
        cfl_interval=1,
        progress=False,
    )
    snaps, times = result[0], result[1]
    errors: list[float] = []
    wrong: list[float] = []
    for snap, t in zip(snaps[1:], times[1:], strict=True):
        he = exact(x, y, t)[0]
        hs = np.asarray(snap, dtype=np.float64)
        errors.append(float((np.abs(hs - he) * area).sum() / (he * area).sum()))
        wrong.append(float(np.mean((hs > WET_THRESHOLD_M) != (he > WET_THRESHOLD_M))))

    h = solver.download_h().astype(np.float64)
    qx, qy = solver.download_hu().astype(np.float64), solver.download_hv().astype(np.float64)
    he, ue, ve = exact(x, y, float(times[-1]))
    wet = h > FILM_DEPTH_M
    us, vs = np.zeros_like(h), np.zeros_like(h)
    us[wet], vs[wet] = qx[wet] / h[wet], qy[wet] / h[wet]
    speed_err = (np.hypot(us - ue, vs - ve) * he * area).sum()
    speed_ref = (np.hypot(ue, ve) * he * area).sum()
    film = (h >= DRY_TOL) & (h < FILM_DEPTH_M)
    film_speed = np.hypot(qx[film], qy[film]) / h[film] if film.any() else np.zeros(1)
    precise = getattr(solver, "download_h_precise", solver.download_h)
    v0_total = float((h0.astype(np.float32).astype(np.float64) * area).sum())
    v1_total = float((np.asarray(precise(), dtype=np.float64) * area).sum())
    clamp = float(np.asarray(solver.download_clamp_mass(), dtype=np.float64).sum())
    return BowlMetrics(
        nx=nx,
        cfl=cfl,
        periods=periods,
        steps=int(solver.steps_total),
        depth_l1_rel_mean=float(np.mean(errors)),
        depth_l1_rel_final=errors[-1],
        wet_cells_wrong_mean=float(np.mean(wrong)),
        speed_l1_rel_final=float(speed_err / speed_ref),
        film_speed_max_ms=float(film_speed.max()),
        volume_drift_rel=abs(v1_total - v0_total) / v0_total,
        clamp_rel=clamp / v0_total,
    )


def evaluate(metrics: BowlMetrics) -> list[str]:
    """Reasons the case fails its gates; empty when it passes."""
    reasons: list[str] = []
    if metrics.nx == GATE_NX and metrics.depth_l1_rel_mean >= GATE_DEPTH_L1:
        reasons.append(f"depth_l1_rel_mean {metrics.depth_l1_rel_mean:.4f} >= {GATE_DEPTH_L1}")
    if metrics.film_speed_max_ms >= GATE_FILM_SPEED_MS:
        reasons.append(f"film_speed_max_ms {metrics.film_speed_max_ms:.2f} >= {GATE_FILM_SPEED_MS}")
    if metrics.clamp_rel != 0.0:
        reasons.append(f"clamp_rel {metrics.clamp_rel:.2e} != 0")
    if metrics.volume_drift_rel >= GATE_VOLUME_DRIFT:
        reasons.append(f"volume_drift_rel {metrics.volume_drift_rel:.2e} >= {GATE_VOLUME_DRIFT}")
    return reasons


def main() -> int:
    from inundation.workflow import (
        _compile_solver_shaders,  # pyright: ignore[reportPrivateUsage]
    )

    parser = argparse.ArgumentParser(description="Thacker planar surface in a parabolic bowl")
    parser.add_argument("--nx", type=int, default=GATE_NX)
    parser.add_argument("--periods", type=int, default=3)
    parser.add_argument("--cfl", type=float, default=None, help="CFL number")
    parser.add_argument("--output-dir", type=Path, default=Path(".tmp/parabolic-bowl"))
    parser.add_argument("--repeats", type=int, default=2, help=">=2 enables determinism checks")
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()

    spv = _compile_solver_shaders()
    cfl = CFL_DEFAULT if args.cfl is None else float(args.cfl)
    for _ in range(args.warmup):
        run_case(spv, args.nx, cfl, args.periods)
    walls: list[float] = []
    runs: list[BowlMetrics] = []
    for _ in range(max(1, args.repeats)):
        t0 = time.perf_counter()
        runs.append(run_case(spv, args.nx, cfl, args.periods))
        walls.append(time.perf_counter() - t0)

    metrics = runs[0]
    reasons = evaluate(metrics)
    deterministic = all(r == metrics for r in runs)
    if not deterministic:
        reasons.append("repeat runs differ")
    summary = {
        "gates": {
            "depth_l1_rel_mean_max": GATE_DEPTH_L1,
            "film_speed_max_ms_max": GATE_FILM_SPEED_MS,
            "volume_drift_rel_max": GATE_VOLUME_DRIFT,
            "clamp_rel_max": 0.0,
        },
        "correctness": [
            {
                "case": "thacker",
                "backend": "gpu_resident_batch",
                **asdict(metrics),
                "deterministic": deterministic,
                "fail_reasons": reasons,
                "passed": not reasons,
            }
        ],
        "performance": [
            {
                "case": "thacker",
                "backend": "gpu_resident_batch",
                "repeats": len(walls),
                "median_wall_s": statistics.median(walls),
                "steps_total": metrics.steps,
            }
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["correctness"], indent=2))
    print(f"RESULT: {'PASS' if not reasons else 'FAIL'}")
    return 1 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
