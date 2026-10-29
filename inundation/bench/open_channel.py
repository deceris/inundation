"""Steady flat-channel inflow/outflow through open boundaries.

Frictionless flat channel at rest at depth ``H``. The west edge prescribes an
inflow unit discharge ``Q`` and the east edge a fixed depth ``H``. The exact
steady state is uniform: ``h = H``, ``hu = Q``, ``hv = 0``. Gates:

- steady discharge: every cell's ``hu`` within ``Q_TOL_REL`` of ``Q``;
- steady depth: every cell within ``H_TOL_M`` of ``H``;
- mass balance: volume change equals the integrated boundary flux;
- no point sources: ``volume_injected_m3 == 0``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from inundation.bench.common import build_channel_mesh, write_mesh_parquet
from inundation.boundary import OpenBoundary
from inundation.workflow import SimulationPhase, SWEWorkflow, WorkflowConfig

LENGTH_M, WIDTH_M = 100.0, 10.0
NX, NY = 50, 5  # dx = 2 m
H_M = 1.0
Q_M2S = 0.5  # inflow unit discharge, u = 0.5 m/s, Froude ~0.16
T_END_S = 1200.0  # frictionless start-up transient rings ~1000 s before settling
MANNING_N = 1e-4  # frictionless: the uniform state is exact only without friction

Q_TOL_REL = 0.01
H_TOL_M = 0.005
MASS_TOL_REL = 1e-4  # of the stored volume

SOLVER_IMPLS = ("baseline", "gpu_resident_batch")


def run_case(solver_impl: str, workdir: Path) -> dict[str, float | bool | str]:
    """Run one backend to t_end and score it against the uniform steady state."""
    verts, quads = build_channel_mesh(NX, NY, LENGTH_M, WIDTH_M)
    mesh = workdir / "channel.parquet"
    write_mesh_parquet(mesh, verts, quads, np.zeros(len(quads), dtype=np.float32))
    wf = SWEWorkflow(
        WorkflowConfig(
            mesh_source=str(mesh),
            manning_n=MANNING_N,
            output_interval_s=T_END_S,
            dt_max=0.5,
            cfl_interval=1,
            progress=False,
            solver_impl=solver_impl,  # pyright: ignore[reportArgumentType]
            initial_state_source={
                "h": np.full(NX * NY, H_M, dtype=np.float32),
                "hu": np.zeros(NX * NY, dtype=np.float32),
                "hv": np.zeros(NX * NY, dtype=np.float32),
            },
        )
    )
    wf.prepare()
    geom = wf.geom
    if geom is None:
        raise RuntimeError("prepare() did not set geom")
    boundary = geom.edge_cellR == -1
    west = boundary & (geom.edge_nx < -0.5)
    east = boundary & (geom.edge_nx > 0.5)
    t0 = time.perf_counter()
    result = wf.run(
        [SimulationPhase(T_END_S, [])],
        open_boundaries=[
            OpenBoundary(west, "discharge", -Q_M2S),
            OpenBoundary(east, "depth", H_M),
        ],
    )
    wall = time.perf_counter() - t0

    h = result.h_final.astype(np.float64)
    hu = np.asarray(result.hu_final, dtype=np.float64)
    v0 = H_M * LENGTH_M * WIDTH_M
    mass_err = abs(result.volume_final_m3 - v0 + result.volume_boundary_out_m3) / v0
    q_err = float(np.abs(hu - Q_M2S).max() / Q_M2S)
    h_err = float(np.abs(h - H_M).max())
    passed = (
        q_err < Q_TOL_REL
        and h_err < H_TOL_M
        and mass_err < MASS_TOL_REL
        and result.volume_injected_m3 == 0.0
    )
    return {
        "solver_impl": solver_impl,
        "steps": result.steps_total,
        "wall_s": wall,
        "q_err_rel": q_err,
        "h_err_m": h_err,
        "mass_err_rel": mass_err,
        "boundary_out_m3": result.volume_boundary_out_m3,
        "passed": passed,
    }


def main() -> None:
    """CLI entry point; exits non-zero when any gate fails."""
    parser = argparse.ArgumentParser(description="Steady flat-channel open-boundary benchmark")
    parser.add_argument("--solver-impl", choices=[*SOLVER_IMPLS, "all"], default="all")
    args = parser.parse_args()
    impls = SOLVER_IMPLS if args.solver_impl == "all" else (args.solver_impl,)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for impl in impls:
            row = run_case(impl, Path(tmp))
            print(
                " ".join(
                    f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
                )
            )
            ok &= bool(row["passed"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
