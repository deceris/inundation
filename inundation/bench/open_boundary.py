"""Open boundaries against analytical solutions, and water passed between basins.

Six cases, each with an independently measured volume balance:

- ``uniform``: inflow and normal-depth outflow on a sloped channel, against the
  Manning normal depth, on two grids.
- ``ritter``: a dry-bed dam break leaving through free outflow, against a longer
  channel that does not end at the cut.
- ``stoker``: the subcritical case, a wet-bed dam break leaving a channel. Reports how
  much each condition reflects.
- ``lake``: still water over a bumpy bed, held by its own level at both ends.
- ``backwater``: steady M1 and M2 profiles below a set level, against the integrated
  gradually varied flow equation.
- ``basins``: one channel run whole, then cut in two and run upstream first.

Run::

    .venv/bin/python -m inundation.bench.open_boundary
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from inundation.bench.common import (
    build_channel_mesh,
    ritter_solution,
    stoker_middle_depth,
    write_mesh_parquet,
)
from inundation.tuning import GRAVITY_G
from inundation.workflow import (
    FreeOutflow,
    Inflow,
    NormalDepth,
    OpenBoundary,
    SimulationPhase,
    Stage,
    SWEWorkflow,
    Wall,
    WorkflowConfig,
    WorkflowResult,
    inflow_phases,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    from inundation.workflow import BoundaryCondition

# Every case's balance closes to this fraction of the volume moved.
GATE_BALANCE = 1e-7
GATE_CLAMP = 1e-9

CHANNEL_WIDTH_M = 40.0
SLOPE = 1e-3
MANNING = 0.03
DISCHARGE_M3S = 10.0
FRICTIONLESS_MANNING = 1e-4
RUN_CELLS_ACROSS = 2

UNIFORM_LENGTH_M = 2000.0
UNIFORM_RUN_S = 4.0 * 3600.0
UNIFORM_GRIDS = (200, 400)
GATE_UNIFORM_DEPTH = 0.01
GATE_UNIFORM_ORDER = 0.9
GATE_STEADY_DISCHARGE = 1e-3

DAM_X_M = 500.0
DAM_CUT_X_M = 600.0
DAM_LONG_X_M = 2000.0
DAM_DX_M = 1.0
DAM_WIDTH_M = 2.0
DAM_RUN_S = 60.0
DAM_UP_M = 1.0
STOKER_DOWN_M = 0.3
GATE_RITTER_MATCH = 1e-3
GATE_RITTER_CONTROL = 0.1
GATE_STOKER_FREE = 0.05

LAKE_LENGTH_M = 1000.0
LAKE_CELLS = 200
LAKE_LEVEL_M = 12.0
LAKE_RUN_S = 300.0
GATE_LAKE_LEVEL_M = 1e-4
GATE_LAKE_SPEED_MS = 1e-3

BACKWATER_LENGTH_M = 3000.0
BACKWATER_CELLS = 600
BACKWATER_RUN_S = 6.0 * 3600.0
BACKWATER_DEPTHS_M = {"m1": 0.8, "m2": 0.25}
# First order, largest in the M2 drawdown beside the outlet: 2.7, 1.3 and 0.6 % on
# 300, 600 and 1200 cells.
GATE_BACKWATER = 0.015

WET_DEPTH_M = 0.01
BASIN_LENGTH_M = 4000.0
BASIN_CELLS = 400
BASIN_PHASE_S = 900.0
BASIN_DISCHARGES_M3S = (5.0, 10.0, 15.0, 20.0, 15.0, 10.0, 5.0, 5.0)
# The exchange holds discharge over each output window, and that sets the error at
# the cut: 2.4 % at 300 s, 0.7 % at 60 s.
BASIN_OUTPUT_S = 60.0
GATE_BASIN_VOLUME = 1e-6
GATE_BASIN_DEPTH = 0.01
GATE_BASIN_OUTFLOW = 0.02


@dataclass(frozen=True)
class Channel:
    """A straight channel from ``x0_m``, with its bed as a function of x."""

    length_m: float
    cells: int
    width_m: float
    bed: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    x0_m: float = 0.0
    across: int = RUN_CELLS_ACROSS

    @property
    def dx_m(self) -> float:
        return self.length_m / self.cells

    @property
    def centres(self) -> NDArray[np.float64]:
        """Cell centres along the channel, in original cell order."""
        x = self.x0_m + (np.arange(self.cells) + 0.5) * self.dx_m
        return np.tile(x, self.across)

    def end(self, x_m: float, name: str) -> OpenBoundary:
        return OpenBoundary(name, [(x_m, 0.0), (x_m, self.width_m)], 0.25 * self.dx_m)


@dataclass(frozen=True)
class Run:
    """A finished run, its depth by position along the channel."""

    result: WorkflowResult
    x: NDArray[np.float64]
    h: NDArray[np.float64]
    wall_s: float


def _write_mesh(channel: Channel, path: Path) -> None:
    verts, quads = build_channel_mesh(
        channel.cells, channel.across, channel.length_m, channel.width_m
    )
    world = verts.astype(np.float64)
    world[:, 0] += channel.x0_m
    write_mesh_parquet(path, world, quads, channel.bed(channel.centres).astype(np.float32))


def _run(
    channel: Channel,
    work_dir: Path,
    name: str,
    boundaries: Sequence[OpenBoundary],
    phases: Sequence[SimulationPhase],
    *,
    output_interval_s: float,
    manning_n: float = MANNING,
    initial_depth: NDArray[np.float64] | None = None,
    cfl: float | None = None,
) -> Run:
    work_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = work_dir / f"{name}.parquet"
    _write_mesh(channel, mesh_path)
    state: dict[str, object] | None = None
    if initial_depth is not None:
        zeros = np.zeros(initial_depth.size, dtype=np.float32)
        state = {"h": initial_depth.astype(np.float32), "hu": zeros, "hv": zeros}
    workflow = SWEWorkflow(
        WorkflowConfig(
            mesh_source=str(mesh_path),
            manning_n=manning_n,
            output_interval_s=output_interval_s,
            dt_max=5.0,
            cfl_interval=1,
            cfl=cfl,
            progress=False,
            track_clamp_mass=True,
            open_boundaries=boundaries,
            initial_state_source=state,
            initial_state_order="original",
        )
    )
    workflow.prepare()
    start = time.perf_counter()
    result = workflow.run(phases)
    wall_s = time.perf_counter() - start
    geom = workflow.geom
    if geom is None:
        raise RuntimeError("workflow lost its geometry")
    x = geom.world_centroid[:, 0]
    order = np.argsort(x, kind="stable")
    return Run(result, x[order], result.h_final[order].astype(np.float64), wall_s)


def _profile(run: Run) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Depth averaged across the channel, by position along it."""
    x, inverse = np.unique(np.round(run.x, 6), return_inverse=True)
    return x, np.bincount(inverse, weights=run.h) / np.bincount(inverse)


def _balance(run: Run, *, initial_m3: float = 0.0) -> dict[str, float]:
    """Volume in and out through the stretches, and how far they miss the final volume."""
    last = [volumes[-1] for volumes in run.result.boundary_volumes_m3.values()]
    out_m3 = float(sum(v[0] for v in last))
    in_m3 = float(sum(v[1] for v in last))
    moved = max(initial_m3 + in_m3, 1e-30)
    residual = initial_m3 + in_m3 - out_m3 - run.result.volume_final_m3
    return {
        "volume_in_m3": in_m3,
        "volume_out_m3": out_m3,
        "balance_rel": abs(residual) / moved,
        "clamp_rel": abs(run.result.clamp_mass_m3 or 0.0) / moved,
        "steps": run.result.steps_total,
        "wall_s": run.wall_s,
    }


def _balance_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row["balance_rel"] >= GATE_BALANCE:
        reasons.append(f"balance_rel {row['balance_rel']:.2e} >= {GATE_BALANCE}")
    if row["clamp_rel"] >= GATE_CLAMP:
        reasons.append(f"clamp_rel {row['clamp_rel']:.2e} >= {GATE_CLAMP}")
    return reasons


def _sloped(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 10.0 - SLOPE * x


def _flat(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.zeros_like(x)


def normal_depth(discharge_m3s: float, width_m: float, slope: float, manning: float) -> float:
    return float((discharge_m3s / width_m * manning / np.sqrt(slope)) ** 0.6)


def gradually_varied_profile(
    x: NDArray[np.float64], depth_at_end: float, unit_discharge: float
) -> NDArray[np.float64]:
    """Steady depth at ascending ``x``, integrated upstream from the channel's end.

    RK4 on ``dh/dx = (S - Sf) / (1 - Fr^2)``, in steps of at most 1 m.
    """

    def slope_of_depth(h: float) -> float:
        friction = MANNING**2 * unit_discharge**2 / h ** (10.0 / 3.0)
        froude2 = unit_discharge**2 / (GRAVITY_G * h**3)
        return (SLOPE - friction) / (1.0 - froude2)

    depths = np.empty(x.size)
    h = depth_at_end
    position = BACKWATER_LENGTH_M
    for k in range(x.size - 1, -1, -1):
        distance = position - float(x[k])
        steps = max(1, int(np.ceil(distance)))
        dx = -distance / steps
        for _ in range(steps):
            k1 = slope_of_depth(h)
            k2 = slope_of_depth(h + 0.5 * dx * k1)
            k3 = slope_of_depth(h + 0.5 * dx * k2)
            k4 = slope_of_depth(h + dx * k3)
            h += dx * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        depths[k] = h
        position = float(x[k])
    return depths


def case_uniform(work_dir: Path, cfl: float | None) -> list[dict[str, Any]]:
    target = normal_depth(DISCHARGE_M3S, CHANNEL_WIDTH_M, SLOPE, MANNING)
    rows: list[dict[str, Any]] = []
    for cells in UNIFORM_GRIDS:
        channel = Channel(UNIFORM_LENGTH_M, cells, CHANNEL_WIDTH_M, _sloped)
        boundaries = [channel.end(0.0, "up"), channel.end(UNIFORM_LENGTH_M, "down")]
        phase = SimulationPhase(
            UNIFORM_RUN_S,
            [],
            {"up": Inflow(DISCHARGE_M3S, SLOPE), "down": NormalDepth(SLOPE)},
        )
        run = _run(
            channel,
            work_dir,
            f"uniform-{cells}",
            boundaries,
            [phase],
            output_interval_s=UNIFORM_RUN_S / 8.0,
            cfl=cfl,
        )
        x, h = _profile(run)
        middle = (x > 0.4 * UNIFORM_LENGTH_M) & (x < 0.6 * UNIFORM_LENGTH_M)
        out = run.result.boundary_volumes_m3["down"][:, 0]
        last_window_s = UNIFORM_RUN_S / 8.0
        row: dict[str, Any] = {
            "case": f"uniform-{cells}",
            "normal_depth_m": target,
            "depth_error_rel": abs(float(h[middle].mean()) / target - 1.0),
            "discharge_error_rel": abs((out[-1] - out[-2]) / last_window_s / DISCHARGE_M3S - 1.0),
            "inflow_error_rel": abs(
                run.result.boundary_volumes_m3["up"][-1, 1] / (DISCHARGE_M3S * UNIFORM_RUN_S) - 1.0
            ),
            **_balance(run),
        }
        reasons = _balance_reasons(row)
        if row["depth_error_rel"] >= GATE_UNIFORM_DEPTH:
            reasons.append(f"depth_error_rel {row['depth_error_rel']:.4f}")
        if row["discharge_error_rel"] >= GATE_STEADY_DISCHARGE:
            reasons.append(f"discharge_error_rel {row['discharge_error_rel']:.2e}")
        if row["inflow_error_rel"] >= GATE_BALANCE:
            reasons.append(f"inflow_error_rel {row['inflow_error_rel']:.2e}")
        row["fail_reasons"] = reasons
        rows.append(row)
    order = float(np.log2(rows[0]["depth_error_rel"] / rows[1]["depth_error_rel"]))
    rows[1]["order"] = order
    if order < GATE_UNIFORM_ORDER:
        rows[1]["fail_reasons"].append(f"order {order:.2f} < {GATE_UNIFORM_ORDER}")
    return rows


def _dam_run(
    work_dir: Path,
    name: str,
    h_down: float,
    end_x_m: float,
    condition: BoundaryCondition,
    cfl: float | None,
) -> Run:
    cells = round(end_x_m / DAM_DX_M)
    channel = Channel(end_x_m, cells, DAM_WIDTH_M, _flat)
    depth = np.where(channel.centres < DAM_X_M, DAM_UP_M, h_down)
    return _run(
        channel,
        work_dir,
        name,
        [channel.end(end_x_m, "cut")],
        [SimulationPhase(DAM_RUN_S, [], {"cut": condition})],
        output_interval_s=DAM_RUN_S,
        manning_n=FRICTIONLESS_MANNING,
        initial_depth=depth,
        cfl=cfl,
    )


def _shared_difference(short: Run, long: Run) -> float:
    x, h = _profile(short)
    x_long, h_long = _profile(long)
    return float(np.abs(h - h_long[np.searchsorted(x_long, x - 1e-6)]).max())


def case_ritter(work_dir: Path, cfl: float | None) -> list[dict[str, Any]]:
    long = _dam_run(work_dir, "ritter-long", 0.0, DAM_LONG_X_M, Wall(), cfl)
    free = _dam_run(work_dir, "ritter-free", 0.0, DAM_CUT_X_M, FreeOutflow(), cfl)
    closed = _dam_run(work_dir, "ritter-wall", 0.0, DAM_CUT_X_M, Wall(), cfl)
    x, h = _profile(free)
    exact, _ = ritter_solution(x, DAM_RUN_S, h_up=DAM_UP_M, dam_x=DAM_X_M, g=GRAVITY_G)
    row: dict[str, Any] = {
        "case": "ritter",
        "difference_from_long_m": _shared_difference(free, long),
        "wall_difference_m": _shared_difference(closed, long),
        "l1_error_rel": float(np.abs(h - exact).mean() / exact.mean()),
        **_balance(free, initial_m3=DAM_UP_M * DAM_X_M * DAM_WIDTH_M),
    }
    reasons = _balance_reasons(row)
    if row["difference_from_long_m"] >= GATE_RITTER_MATCH:
        reasons.append(f"difference_from_long_m {row['difference_from_long_m']:.2e}")
    if row["wall_difference_m"] < GATE_RITTER_CONTROL:
        reasons.append(f"the wall control barely differs: {row['wall_difference_m']:.2e}")
    row["fail_reasons"] = reasons
    return [row]


def case_stoker(work_dir: Path, cfl: float | None) -> list[dict[str, Any]]:
    h_mid = stoker_middle_depth(DAM_UP_M, STOKER_DOWN_M, GRAVITY_G)
    speed_mid = 2.0 * (np.sqrt(GRAVITY_G * DAM_UP_M) - np.sqrt(GRAVITY_G * h_mid))
    long = _dam_run(work_dir, "stoker-long", STOKER_DOWN_M, DAM_LONG_X_M, Wall(), cfl)
    jump = h_mid - STOKER_DOWN_M
    # NormalDepth is told the slope that carries the star state's velocity.
    star_slope = (speed_mid * FRICTIONLESS_MANNING / h_mid ** (2.0 / 3.0)) ** 2
    conditions: dict[str, BoundaryCondition] = {
        "free": FreeOutflow(),
        "normal": NormalDepth(star_slope),
        "stage": Stage(h_mid),
    }
    rows: list[dict[str, Any]] = []
    initial = (DAM_UP_M * DAM_X_M + STOKER_DOWN_M * (DAM_CUT_X_M - DAM_X_M)) * DAM_WIDTH_M
    for label, condition in conditions.items():
        run = _dam_run(work_dir, f"stoker-{label}", STOKER_DOWN_M, DAM_CUT_X_M, condition, cfl)
        row: dict[str, Any] = {
            "case": f"stoker-{label}",
            "reflection_rel": _shared_difference(run, long) / jump,
            **_balance(run, initial_m3=initial),
        }
        reasons = _balance_reasons(row)
        if label == "free" and row["reflection_rel"] >= GATE_STOKER_FREE:
            reasons.append(f"reflection_rel {row['reflection_rel']:.3f}")
        row["fail_reasons"] = reasons
        rows.append(row)
    return rows


def _bumpy(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 10.0 + np.sin(2.0 * np.pi * x / 200.0) + 0.5 * np.sin(2.0 * np.pi * x / 73.0)


def case_lake(work_dir: Path, cfl: float | None) -> list[dict[str, Any]]:
    channel = Channel(LAKE_LENGTH_M, LAKE_CELLS, CHANNEL_WIDTH_M, _bumpy)
    depth = np.maximum(LAKE_LEVEL_M - channel.bed(channel.centres), 0.0)
    run = _run(
        channel,
        work_dir,
        "lake",
        [channel.end(0.0, "up"), channel.end(LAKE_LENGTH_M, "down")],
        [SimulationPhase(LAKE_RUN_S, [], {"up": Stage(LAKE_LEVEL_M), "down": Stage(LAKE_LEVEL_M)})],
        output_interval_s=LAKE_RUN_S,
        initial_depth=depth,
        cfl=cfl,
    )
    x, h = _profile(run)
    level = _bumpy(x) + h
    hu, hv = run.result.hu_final, run.result.hv_final
    speed = float(np.hypot(hu, hv).max() / h.min()) if hu is not None and hv is not None else 0.0
    row: dict[str, Any] = {
        "case": "lake",
        "level_error_m": float(np.abs(level - LAKE_LEVEL_M).max()),
        "speed_max_ms": speed,
        "steps": run.result.steps_total,
        "wall_s": run.wall_s,
    }
    reasons: list[str] = []
    if row["level_error_m"] >= GATE_LAKE_LEVEL_M:
        reasons.append(f"level_error_m {row['level_error_m']:.2e}")
    if row["speed_max_ms"] >= GATE_LAKE_SPEED_MS:
        reasons.append(f"speed_max_ms {row['speed_max_ms']:.2e}")
    row["fail_reasons"] = reasons
    return [row]


def case_backwater(work_dir: Path, cfl: float | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    channel = Channel(BACKWATER_LENGTH_M, BACKWATER_CELLS, CHANNEL_WIDTH_M, _sloped)
    level_bed = float(_sloped(np.array([BACKWATER_LENGTH_M]))[0])
    for label, end_depth in BACKWATER_DEPTHS_M.items():
        run = _run(
            channel,
            work_dir,
            f"backwater-{label}",
            [channel.end(0.0, "up"), channel.end(BACKWATER_LENGTH_M, "down")],
            [
                SimulationPhase(
                    BACKWATER_RUN_S,
                    [],
                    {"up": Inflow(DISCHARGE_M3S, SLOPE), "down": Stage(level_bed + end_depth)},
                )
            ],
            output_interval_s=BACKWATER_RUN_S / 6.0,
            cfl=cfl,
        )
        x, h = _profile(run)
        reference = gradually_varied_profile(x, end_depth, DISCHARGE_M3S / CHANNEL_WIDTH_M)
        out = run.result.boundary_volumes_m3["down"][:, 0]
        row: dict[str, Any] = {
            "case": f"backwater-{label}",
            "profile_error_rel": float((np.abs(h - reference) / reference)[1:-1].max()),
            "end_error_rel": float(abs(h[-1] - reference[-1]) / reference[-1]),
            "discharge_error_rel": abs(
                (out[-1] - out[-2]) / (BACKWATER_RUN_S / 6.0) / DISCHARGE_M3S - 1.0
            ),
            **_balance(run),
        }
        reasons = _balance_reasons(row)
        if row["profile_error_rel"] >= GATE_BACKWATER:
            reasons.append(f"profile_error_rel {row['profile_error_rel']:.4f}")
        if row["discharge_error_rel"] >= GATE_STEADY_DISCHARGE:
            reasons.append(f"discharge_error_rel {row['discharge_error_rel']:.2e}")
        row["fail_reasons"] = reasons
        rows.append(row)
    return rows


def case_basins(work_dir: Path, cfl: float | None) -> list[dict[str, Any]]:
    whole = Channel(BASIN_LENGTH_M, BASIN_CELLS, CHANNEL_WIDTH_M, _sloped)
    cut = 0.5 * BASIN_LENGTH_M
    upper = Channel(cut, BASIN_CELLS // 2, CHANNEL_WIDTH_M, _sloped)
    lower = Channel(cut, BASIN_CELLS // 2, CHANNEL_WIDTH_M, _sloped, x0_m=cut)

    def hydrograph(down: str) -> list[SimulationPhase]:
        return [
            SimulationPhase(BASIN_PHASE_S, [], {"up": Inflow(q, SLOPE), down: NormalDepth(SLOPE)})
            for q in BASIN_DISCHARGES_M3S
        ]

    reference = _run(
        whole,
        work_dir,
        "basins-whole",
        [whole.end(0.0, "up"), whole.end(BASIN_LENGTH_M, "down")],
        hydrograph("down"),
        output_interval_s=BASIN_OUTPUT_S,
        cfl=cfl,
    )
    first = _run(
        upper,
        work_dir,
        "basins-upper",
        [upper.end(0.0, "up"), upper.end(cut, "cut")],
        hydrograph("cut"),
        output_interval_s=BASIN_OUTPUT_S,
        cfl=cfl,
    )
    second = _run(
        lower,
        work_dir,
        "basins-lower",
        [lower.end(cut, "from_upper"), lower.end(BASIN_LENGTH_M, "down")],
        inflow_phases(first.result, "cut", "from_upper", SLOPE, {"down": NormalDepth(SLOPE)}),
        output_interval_s=BASIN_OUTPUT_S,
        cfl=cfl,
    )
    passed = float(first.result.boundary_volumes_m3["cut"][-1, 0])
    received = float(second.result.boundary_volumes_m3["from_upper"][-1, 1])
    x_lower, h_lower = _profile(second)
    x_whole, h_whole = _profile(reference)
    h_shared = h_whole[np.searchsorted(x_whole, x_lower - 1e-6)]
    wet = h_shared > WET_DEPTH_M
    out_whole = float(reference.result.boundary_volumes_m3["down"][-1, 0])
    out_lower = float(second.result.boundary_volumes_m3["down"][-1, 0])
    row: dict[str, Any] = {
        "case": "basins",
        "volume_passed_m3": passed,
        "volume_received_m3": received,
        "passed_error_rel": abs(received / passed - 1.0),
        "depth_difference_rel": float((np.abs(h_lower - h_shared)[wet] / h_shared[wet]).max()),
        "outflow_difference_rel": abs(out_lower / out_whole - 1.0),
        "upper_balance_rel": _balance(first)["balance_rel"],
        **_balance(second),
    }
    reasons = _balance_reasons(row)
    if row["upper_balance_rel"] >= GATE_BALANCE:
        reasons.append(f"upper_balance_rel {row['upper_balance_rel']:.2e}")
    if row["passed_error_rel"] >= GATE_BASIN_VOLUME:
        reasons.append(f"passed_error_rel {row['passed_error_rel']:.2e}")
    if row["depth_difference_rel"] >= GATE_BASIN_DEPTH:
        reasons.append(f"depth_difference_rel {row['depth_difference_rel']:.4f}")
    if row["outflow_difference_rel"] >= GATE_BASIN_OUTFLOW:
        reasons.append(f"outflow_difference_rel {row['outflow_difference_rel']:.4f}")
    row["fail_reasons"] = reasons
    return [row]


CASES: dict[str, Callable[[Path, float | None], list[dict[str, Any]]]] = {
    "uniform": case_uniform,
    "ritter": case_ritter,
    "stoker": case_stoker,
    "lake": case_lake,
    "backwater": case_backwater,
    "basins": case_basins,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Open boundaries against analytical solutions")
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    parser.add_argument("--cfl", type=float, default=None, help="CFL number")
    parser.add_argument("--output-dir", type=Path, default=Path(".tmp/open-boundary"))
    args = parser.parse_args()

    correctness: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    for name in args.case or list(CASES):
        rows = CASES[name](args.output_dir / "meshes", args.cfl)
        for row in rows:
            row["backend"] = "gpu_resident_batch"
            row["passed"] = not row["fail_reasons"]
            correctness.append(row)
            performance.append(
                {
                    "case": row["case"],
                    "backend": "gpu_resident_batch",
                    "repeats": 1,
                    "median_wall_s": statistics.median([row.get("wall_s", 0.0)]),
                    "steps_total": row.get("steps", 0),
                }
            )
    summary = {
        "gates": {
            "balance_rel_max": GATE_BALANCE,
            "clamp_rel_max": GATE_CLAMP,
            "uniform_depth_error_rel_max": GATE_UNIFORM_DEPTH,
            "uniform_order_min": GATE_UNIFORM_ORDER,
            "steady_discharge_error_rel_max": GATE_STEADY_DISCHARGE,
            "ritter_difference_m_max": GATE_RITTER_MATCH,
            "stoker_free_reflection_rel_max": GATE_STOKER_FREE,
            "lake_level_error_m_max": GATE_LAKE_LEVEL_M,
            "lake_speed_ms_max": GATE_LAKE_SPEED_MS,
            "backwater_profile_error_rel_max": GATE_BACKWATER,
            "basin_passed_error_rel_max": GATE_BASIN_VOLUME,
            "basin_depth_difference_rel_max": GATE_BASIN_DEPTH,
            "basin_outflow_difference_rel_max": GATE_BASIN_OUTFLOW,
        },
        "correctness": correctness,
        "performance": performance,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(correctness, indent=2))
    passed = all(row["passed"] for row in correctness)
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
