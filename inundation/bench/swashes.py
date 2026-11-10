"""SWASHES analytic benchmarks (Delestre et al. 2013, doi:10.1002/fld.3741).

The reference is the SWASHES program itself (``swashes`` extra, a prebuilt
binary), asked for the solution at exactly this mesh's cell centres, so the
comparison is cell by cell with no interpolation. Each case reproduces the
SWASHES setup: bed, initial state, boundaries and Manning coefficient.

Steady cases (bumps, MacDonald) start from the SWASHES initial condition and
run until the flow settles; they are gated on depth and discharge error and on
mass balance against the open-boundary flux. Thacker's oscillations are
periodic: SWASHES reports them after a whole number of periods, so that same
state is both the initial condition and the reference. Lake at rest is gated
on staying at rest.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from inundation.bench.common import build_channel_mesh, l1_relative_error, write_mesh_parquet
from inundation.workflow import (
    Inflow,
    OpenBoundary,
    SimulationPhase,
    Stage,
    SWEWorkflow,
    Wall,
    WorkflowConfig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from inundation.workflow import BoundaryCondition, WorkflowResult

# "inflow": discharge per unit width into the mesh [m^2/s]; "depth": water depth
# beyond the edge over the boundary cell's bed [m].
BcKind = Literal["inflow", "depth"]
# Inflow splits its discharge by Manning conveyance; every case's inflow section
# is level, so any positive slope gives the same uniform split.
INFLOW_ENERGY_SLOPE = 1e-3

# SWASHES output columns.
COLS_1D = {"x": 0, "h": 1, "u": 2, "z": 3, "q": 4}
COLS_2D = {"x": 0, "y": 1, "h": 2, "u": 3, "v": 4, "z": 6}


@dataclass(frozen=True)
class Case:
    """One SWASHES solution reproduced on a regular quad channel."""

    name: str
    swashes: tuple[int, int, int, int]  # dimension type domain choice
    length_m: float
    nx: int
    ny: int  # 1D: cells across the (inert) channel width; 2D: SWASHES cells in y
    manning_n: float
    t_end_s: float  # steady cases; periodic cases run `periods` periods instead
    west: tuple[BcKind, float] | None = None  # None = reflective wall
    east: tuple[BcKind, float] | None = None
    eta0_m: float | None = None  # initial free surface at rest; None = periodic case
    periods: int = 0  # periodic cases: whole periods to run
    gates: tuple[float, float, float] = (0.0, 0.0, 0.0)  # depth L1, q max rel, mass

    @property
    def two_d(self) -> bool:
        """SWASHES dimension 2: the reference is a full 2D field."""
        return self.swashes[0] == 2

    @property
    def width_m(self) -> float:
        """Square domain in 2D; one cell size per row across a 1D channel."""
        return self.length_m if self.two_d else self.ny * self.length_m / self.nx


# Gates: measured on MoltenVK (docs/implementation/31-swashes/results.md) with
# about 2-3x headroom, on both backends. Mass is relative to throughput (see
# score): ~1e5 float32 steps leave 0.4-2e-4 of it in the steady cases. Steady
# frictionless bumps take a few hundred seconds to shed the start-up waves.
CASES: tuple[Case, ...] = (
    Case(
        "bump_subcritical",
        (1, 1, 1, 1),
        25.0,
        200,
        2,
        1e-4,
        300.0,
        ("inflow", 4.42),
        ("depth", 2.0),
        eta0_m=2.0,
        gates=(2e-3, 2e-3, 5e-4),
    ),
    Case(
        "bump_transcritical",
        (1, 1, 1, 2),
        25.0,
        200,
        2,
        1e-4,
        300.0,
        ("inflow", 1.53),
        # SWASHES: h_out = 0.66 "while subcritical"; the depth edge lets go once
        # the outflow turns supercritical.
        ("depth", 0.66),
        eta0_m=0.66,
        gates=(1e-2, 2e-3, 5e-4),
    ),
    Case(
        "bump_shock",
        (1, 1, 1, 3),
        25.0,
        200,
        2,
        1e-4,
        600.0,
        ("inflow", 0.18),
        ("depth", 0.33),
        eta0_m=0.33,
        gates=(1.5e-2, 1e-2, 5e-4),
    ),
    Case(
        "lake_emerged_bump",
        (1, 1, 1, 5),
        25.0,
        200,
        2,
        1e-4,
        100.0,
        eta0_m=0.1,
        gates=(1e-5, 1e-5, 1e-5),  # q gate: max |hu| [m^2/s], must stay at rest
    ),
    Case(
        "macdonald_long_subcritical",
        (1, 2, 1, 2),
        1000.0,
        200,
        2,
        0.033,
        6000.0,
        ("inflow", 2.0),
        ("depth", 0.748324),
        eta0_m=-1e9,  # SWASHES starts dry
        gates=(1.5e-2, 1.5e-2, 5e-4),
    ),
    Case(
        "macdonald_short_shock",
        (1, 2, 2, 2),
        100.0,
        200,
        2,
        0.0328,
        1500.0,
        ("inflow", 2.0),
        ("depth", 2.87871),
        eta0_m=2.87871,
        gates=(1e-2, 1e-2, 5e-4),
    ),
    Case("thacker_1d", (1, 4, 1, 1), 4.0, 400, 2, 1e-4, 0.0, periods=1, gates=(7e-2, 0.0, 1e-5)),
    Case(
        "thacker_2d_planar",
        (2, 1, 1, 2),
        4.0,
        100,
        100,
        1e-4,
        0.0,
        periods=1,
        # One period on a 100^2 grid: first-order smearing of the moving
        # shoreline; halves per doubling of resolution (results.md).
        gates=(3.5e-1, 0.0, 1e-5),
    ),
)
CASE_NAMES = tuple(c.name for c in CASES)

SWASHES_TIME_PREFIX = "# Time value:"
# SWASHES reports Thacker after a whole number of periods: 5 in 1D, 3 in 2D.
SWASHES_PERIODS = {1: 5, 2: 3}

SOLVER_IMPLS = ("baseline", "gpu_resident_batch")
DEFAULT_SOLVER_IMPLS = ("gpu_resident_batch",)


def swashes_executable() -> Path:
    """The prebuilt binary shipped by the ``swashes`` extra (as its wrapper finds it)."""
    try:
        files = importlib.metadata.files("swashes") or []
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            'SWASHES is required: uv pip install -e ".[swashes]" (or uv sync --all-extras)'
        ) from exc
    for f in files:
        if str(f).startswith("swashes/data/bin/swashes"):
            return Path(str(f.locate())).resolve(strict=True)
    raise RuntimeError("the swashes package ships no swashes/data/bin/swashes binary")


def swashes_output(args: tuple[int, ...]) -> str:
    """Raw stdout of ``swashes *args``."""
    exe = swashes_executable()
    out = subprocess.run(  # noqa: S603 - fixed binary, integer arguments
        [str(exe), *map(str, args)], capture_output=True, text=True, check=True
    )
    if "USE:" in out.stdout:
        raise ValueError(f"swashes rejected arguments {args}")
    return out.stdout


def parse_swashes(text: str) -> tuple[NDArray[np.float64], float | None]:
    """Numeric rows of a SWASHES output, and its ``Time value`` if it has one."""
    rows: list[list[float]] = []
    t: float | None = None
    for line in text.splitlines():
        if line.startswith(SWASHES_TIME_PREFIX):
            t = float(line[len(SWASHES_TIME_PREFIX) :].split()[0])
        elif line.strip() and not line.startswith("#"):
            rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=np.float64), t


@dataclass(frozen=True)
class Reference:
    """SWASHES fields in mesh (row-major, x fastest) cell order."""

    x: NDArray[np.float64]
    h: NDArray[np.float64]
    hu: NDArray[np.float64]
    hv: NDArray[np.float64]
    z: NDArray[np.float64]
    t_s: float | None


def reference(case: Case) -> Reference:
    """SWASHES solution at this case's cell centres, laid out like ``build_channel_mesh``."""
    if case.two_d:
        rows, t = parse_swashes(swashes_output((*case.swashes, case.nx, case.ny)))
        # SWASHES loops x outer, y inner; the mesh is y outer, x inner.
        grid = rows.reshape(case.nx, case.ny, -1).transpose(1, 0, 2).reshape(-1, rows.shape[1])
        c = COLS_2D
        h = grid[:, c["h"]]
        return Reference(
            grid[:, c["x"]], h, h * grid[:, c["u"]], h * grid[:, c["v"]], grid[:, c["z"]], t
        )
    rows, t = parse_swashes(swashes_output((*case.swashes, case.nx)))
    c = COLS_1D

    def tile(col: str) -> NDArray[np.float64]:  # same profile on every row
        return np.tile(rows[:, c[col]], case.ny)

    h = tile("h")
    return Reference(tile("x"), h, h * tile("u"), np.zeros_like(h), tile("z"), t)


def run_case(
    case: Case, solver_impl: str, workdir: Path, cfl: float | None = None
) -> dict[str, float | bool | str]:
    """Run one case on one backend and score it against SWASHES."""
    ref = reference(case)
    if case.eta0_m is not None:  # steady: start at rest at the SWASHES free surface
        t_end = case.t_end_s
        h0 = np.maximum(case.eta0_m - ref.z, 0.0)
        hu0 = np.zeros_like(h0)
        hv0 = np.zeros_like(h0)
    else:  # periodic: the reported state is also the state at t = 0, T, 2T, ...
        if ref.t_s is None:
            raise RuntimeError(f"{case.name}: SWASHES gave no output time")
        t_end = ref.t_s * case.periods / SWASHES_PERIODS[case.swashes[0]]
        h0, hu0, hv0 = ref.h, ref.hu, ref.hv

    verts, quads = build_channel_mesh(case.nx, case.ny, case.length_m, case.width_m)
    mesh = workdir / f"{case.name}.parquet"
    write_mesh_parquet(mesh, verts, quads, ref.z.astype(np.float32))
    dx = case.length_m / case.nx
    edges = {"west": 0.0, "east": case.length_m}
    specs = {"west": case.west, "east": case.east}
    open_sides = [side for side, spec in specs.items() if spec is not None]
    boundaries = [
        OpenBoundary(side, [(edges[side], 0.0), (edges[side], case.width_m)], 0.25 * dx)
        for side in open_sides
    ]
    # Bed under each end: the reference profile's first and last cells.
    beds = {"west": float(ref.z[0]), "east": float(ref.z[case.nx - 1])}

    def condition(side: str) -> BoundaryCondition:
        spec = specs[side]
        if spec is None:
            return Wall()
        kind, value = spec
        if kind == "inflow":
            return Inflow(value * case.width_m, INFLOW_ENERGY_SLOPE)
        return Stage(beds[side] + value)

    wf = SWEWorkflow(
        WorkflowConfig(
            mesh_source=str(mesh),
            manning_n=case.manning_n,
            output_interval_s=t_end / 10.0,
            dt_max=1.0,
            cfl_interval=1,
            progress=False,
            solver_impl=solver_impl,  # pyright: ignore[reportArgumentType]
            cfl=cfl,
            initial_state_source={
                "h": h0.astype(np.float32),
                "hu": hu0.astype(np.float32),
                "hv": hv0.astype(np.float32),
            },
            initial_state_order="original",
            open_boundaries=boundaries,
        )
    )
    wf.prepare()
    if wf.perm is None:
        raise RuntimeError("prepare() did not set perm")
    t0 = time.perf_counter()
    result = wf.run([SimulationPhase(t_end, [], {side: condition(side) for side in open_sides})])
    wall = time.perf_counter() - t0
    return score(case, ref, result, wf.perm, wall, solver_impl)


def score(
    case: Case,
    ref: Reference,
    result: WorkflowResult,
    perm: NDArray[np.int32],
    wall_s: float,
    solver_impl: str,
) -> dict[str, float | bool | str]:
    """Depth, discharge, steadiness and mass metrics in mesh order; gate them."""

    def original(a: NDArray[np.float32]) -> NDArray[np.float64]:
        out = np.empty(a.shape[0], dtype=np.float64)
        out[perm] = a  # perm[solver_index] = original_index
        return out

    h = original(result.h_final)
    hu = original(np.asarray(result.hu_final))
    area = (case.length_m / case.nx) * (case.width_m / case.ny)
    v0 = float((ref.h if case.eta0_m is None else np.maximum(case.eta0_m - ref.z, 0.0)).sum())
    v0 *= area
    # (out, in) through each stretch since the start of the run.
    net_out = sum(float(v[-1, 0] - v[-1, 1]) for v in result.boundary_volumes_m3.values())
    balance = result.volume_final_m3 - v0 + net_out
    # Relative to all the water the run held or moved: a steady case passes its
    # stored volume through many times, and the boundary volume is what is
    # being balanced (float32 state rounding scales with it).
    inflow = sum(float(v[-1, 1]) for v in result.boundary_volumes_m3.values())
    mass_err = abs(balance) / max(v0, result.volume_final_m3, inflow)
    depth_l1 = l1_relative_error(h, ref.h)
    q_scale = float(np.abs(ref.hu).max())
    if case.eta0_m is None:
        # Thacker: reported where the flow turns (u = 0), no discharge scale.
        q_err = 0.0
    elif q_scale == 0.0:  # lake at rest: absolute momentum
        q_err = float(np.abs(hu).max())
    else:  # L1, like depth: a max is set by the one smeared shock cell
        q_err = l1_relative_error(hu, ref.hu)
    # Change over the last output interval: how far from steady the run ended.
    snaps = result.snapshots
    settle = float(np.abs(snaps[-1] - snaps[-2]).max()) if len(snaps) > 1 else float("nan")
    g_depth, g_q, g_mass = case.gates
    passed = (
        bool(np.isfinite(h).all())
        and depth_l1 <= g_depth
        and q_err <= g_q
        and mass_err <= g_mass
        and result.volume_injected_m3 == 0.0
    )
    return {
        "case": case.name,
        "solver_impl": solver_impl,
        "steps": result.steps_total,
        "wall_s": wall_s,
        "depth_l1_rel": depth_l1,
        "depth_max_abs_m": float(np.abs(h - ref.h).max()),
        "q_l1_rel": q_err,
        "mass_err_rel": mass_err,
        "settle_m": settle,
        "passed": passed,
    }


def main() -> None:
    """CLI entry point; exits non-zero when any gate fails."""
    parser = argparse.ArgumentParser(description="SWASHES analytic benchmarks")
    parser.add_argument("--cases", default=",".join(CASE_NAMES), help=f"CSV of {CASE_NAMES}")
    parser.add_argument("--backends", default=",".join(DEFAULT_SOLVER_IMPLS))
    parser.add_argument("--output-dir", help="write summary.json here (bench report)")
    parser.add_argument("--cfl", type=float, default=None, help="CFL number")
    parser.add_argument(
        "--repeats", type=int, default=1, help=">=2 checks gpu_resident_batch repeats match"
    )
    parser.add_argument("--warmup", type=int, default=0, help="Runs of the first case to discard")
    args = parser.parse_args()
    names = args.cases.split(",")
    unknown = sorted(set(names) - set(CASE_NAMES))
    if unknown:
        parser.error(f"unknown cases: {unknown}")
    impls = args.backends.split(",")
    bad = sorted(set(impls) - set(SOLVER_IMPLS))
    if bad:
        parser.error(f"unsupported backends: {bad}")
    rows: list[dict[str, float | bool | str]] = []
    cases = [c for c in CASES if c.name in names]
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(args.warmup):
            run_case(cases[0], impls[0], Path(tmp), args.cfl)
        for case in cases:
            for impl in impls:
                runs = [
                    run_case(case, impl, Path(tmp), args.cfl) for _ in range(max(1, args.repeats))
                ]
                row = runs[0]
                # Baseline sums fluxes with float atomics, so only the resident
                # backend promises bit-identical repeats.
                if impl == "gpu_resident_batch" and len(runs) > 1:
                    metrics = [{k: v for k, v in r.items() if k != "wall_s"} for r in runs]
                    row["deterministic"] = all(m == metrics[0] for m in metrics)
                    row["passed"] = bool(row["passed"]) and bool(row["deterministic"])
                print(
                    " ".join(
                        f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items()
                    ),
                    flush=True,
                )
                rows.append(row)
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        gates = {
            f"{c.name}:{metric}": limit
            for c in CASES
            if c.name in names
            for metric, limit in zip(
                ("depth_l1_rel", "q_l1_rel", "mass_err_rel"), c.gates, strict=True
            )
        }
        summary = {"gates": gates, "correctness": rows}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sys.exit(0 if all(r["passed"] for r in rows) else 1)


if __name__ == "__main__":
    main()
