# pyright: reportPrivateUsage=false
"""Water passing both ways between basin meshes, against one mesh.

Every case runs one mesh whole, then cut along polygons by ``split_mesh`` and coupled:

- ``lake``: still water over a bumpy bed across a border.
- ``ritter``: a dam break through a border.
- ``backwater``: an M1 profile whose level has to reach the upper basin.
- ``reversal``: a source below the border, then one above it.
- ``crest``: water over a bed step at the border, up and down.
- ``confluence``: three basins, two flowing into one, cut along lines that meet.

Run::

    .venv/bin/python -m inundation.bench.basin_exchange
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyarrow.parquet as pq
import shapely

from inundation.bench.open_boundary import Channel, _bumpy, _flat, _metrics, _sloped, _write_mesh
from inundation.coupling import Basin, CoupledResult, run_coupled
from inundation.mesh.split import split_mesh
from inundation.workflow import (
    BoundaryCondition,
    Inflow,
    OpenBoundary,
    PointSource,
    SimulationPhase,
    Stage,
    SWEWorkflow,
    WorkflowConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry
    from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch

    Field = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]

LENGTH_M = 2000.0
WIDTH_M = 100.0
CELLS = 200
ACROSS = 8
CUT_M = 1000.0
SLOPE = 1e-3
MANNING = 0.03
WINDOW_S = 10.0

GATE_BALANCE = 1e-6
GATE_CLAMP = 0.0
GATE_LAKE_LEVEL_M = 1e-4
GATE_LAKE_SPEED_MS = 1e-3
GATE_DEPTH = 0.01


@dataclass(frozen=True)
class Forcing:
    """One phase: sources by basin, and each open stretch's condition."""

    duration_s: float
    sources: dict[int, Sequence[PointSource]] = field(default_factory=dict[int, Any])
    conditions: dict[str, BoundaryCondition] = field(default_factory=dict[str, BoundaryCondition])


@dataclass(frozen=True)
class Layout:
    """A mesh and how it is cut."""

    channel: Channel
    polygons: Sequence[shapely.Geometry]
    # Each open stretch and the basin it lies on.
    stretches: Sequence[tuple[OpenBoundary, int]] = ()
    depth: Field | None = None


def _config(
    mesh: Path, stretches: Sequence[OpenBoundary], depth: Field | None, cfl: float | None
) -> WorkflowConfig:
    table = pq.read_table(mesh, columns=["geometry"])
    cells = shapely.from_wkb(table["geometry"].to_numpy(zero_copy_only=False))
    state: dict[str, object] | None = None
    if depth is not None:
        at = shapely.get_coordinates(shapely.centroid(cells))
        h = depth(at[:, 0], at[:, 1]).astype(np.float32)
        zeros = np.zeros(h.size, dtype=np.float32)
        state = {"h": h, "hu": zeros, "hv": zeros}
    return WorkflowConfig(
        mesh_source=str(mesh),
        manning_n=MANNING,
        output_interval_s=WINDOW_S,
        dt_max=1.0,
        cfl_interval=1,
        progress=False,
        track_clamp_mass=True,
        open_boundaries=stretches,
        initial_state_source=state,
        initial_state_order="original",
        cfl=cfl,
    )


def _sorted_by_place(h: NDArray[np.float64], at: NDArray[np.float64]) -> NDArray[np.float64]:
    return h[np.lexsort((np.round(at[:, 1], 3), np.round(at[:, 0], 3)))]


def _compare(
    work_dir: Path,
    name: str,
    layout: Layout,
    forcing: Sequence[Forcing],
    cfl: float | None,
    window_s: float = WINDOW_S,
) -> tuple[NDArray[np.float64], CoupledResult, float]:
    """One mesh and its split: depths in the same order, the coupled run and its wall time."""
    case_dir = work_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    mesh = case_dir / "whole.parquet"
    _write_mesh(layout.channel, mesh)

    every = [stretch for stretch, _ in layout.stretches]
    workflow = SWEWorkflow(_config(mesh, every, layout.depth, cfl))
    workflow.prepare()
    whole = workflow.run(
        [
            SimulationPhase(
                part.duration_s,
                [source for sources in part.sources.values() for source in sources],
                part.conditions,
            )
            for part in forcing
        ]
    )
    geom = cast("MeshGeometry", workflow.geom)
    expected = _sorted_by_place(
        whole.h_final.astype(np.float64), geom.world_centroid.astype(np.float64)
    )

    split = split_mesh(mesh, layout.polygons, case_dir / "split")
    basins = []
    for k, path in enumerate(split.meshes):
        own = [stretch for stretch, basin in layout.stretches if basin == k]
        names = {stretch.name for stretch in own}
        phases = [
            SimulationPhase(
                part.duration_s,
                list(part.sources.get(k, ())),
                {key: value for key, value in part.conditions.items() if key in names},
            )
            for part in forcing
        ]
        basins.append(Basin(_config(path, own, layout.depth, cfl), phases))
    started = time.perf_counter()
    result = run_coupled(basins, split.links, window_s)
    return expected, result, time.perf_counter() - started


def _coupled_depth(result: CoupledResult) -> NDArray[np.float64]:
    h = np.concatenate([basin.h_final for basin in result.basins])
    at = np.concatenate(
        [
            cast("Any", basin.workflow.geom).world_centroid.astype(np.float64)
            for basin in result.basins
        ]
    )
    return _sorted_by_place(h, at)


def _row(
    case: str,
    result: CoupledResult,
    wall_s: float,
    **metrics: float,
) -> dict[str, Any]:
    water = result.volume_start_m3 + result.volume_in_m3
    clamp = sum(float(basin.clamp_mass_m3 or 0.0) for basin in result.basins)
    balance = abs(result.balance_gap_m3) / water
    row: dict[str, Any] = {
        "case": case,
        "balance_rel": balance,
        "clamp_m3": clamp,
        "exchanged_m3": result.exchanged_m3,
        "steps": sum(basin.steps_total for basin in result.basins),
        "wall_s": wall_s,
        **metrics,
        "fail_reasons": [],
    }
    if balance > GATE_BALANCE:
        row["fail_reasons"].append(f"balance {balance:.2e} above {GATE_BALANCE}")
    if clamp > GATE_CLAMP:
        row["fail_reasons"].append(f"clamp mass {clamp:.3e} m3")
    return row


def _depth_row(
    case: str, expected: NDArray[np.float64], result: CoupledResult, wall_s: float
) -> dict[str, Any]:
    difference = float(np.abs(_coupled_depth(result) - expected).max() / expected.max())
    row = _row(case, result, wall_s, depth_difference_rel=difference)
    if difference > GATE_DEPTH:
        row["fail_reasons"].append(f"depth difference {difference:.4f} above {GATE_DEPTH}")
    return row


def _x_below(x_m: float, y_m: float = WIDTH_M) -> shapely.Geometry:
    return shapely.box(-1.0, -1.0, x_m, y_m + 1.0)


def _x_above(x_m: float, y_m: float = WIDTH_M) -> shapely.Geometry:
    return shapely.box(x_m, -1.0, LENGTH_M + 1.0, y_m + 1.0)


def _channel(bed: Callable[[NDArray[np.float64]], NDArray[np.float64]]) -> Channel:
    return Channel(LENGTH_M, CELLS, WIDTH_M, bed, across=ACROSS)


def case_lake(work_dir: Path, cfl: float | None) -> list[dict[str, Any]]:
    level = 10.5
    channel = _channel(_bumpy)

    def depth(x: NDArray[np.float64], _: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(level - _bumpy(x), 0.0)

    layout = Layout(channel, [_x_below(CUT_M), _x_above(CUT_M)], depth=depth)
    _, result, wall_s = _compare(work_dir, "lake", layout, [Forcing(600.0)], cfl)
    errors, speeds = [], []
    for basin in result.basins:
        solver = cast("SWESolverGpuResidentBatch", basin.workflow.solver)
        geom = cast("Any", basin.workflow.geom)
        wet = basin.h_final > 0.01
        errors.append(np.abs(basin.h_final + geom.zb.astype(np.float64) - level)[wet].max())
        hu, hv = solver.download_hu(), solver.download_hv()
        speeds.append((np.hypot(hu, hv)[wet] / basin.h_final[wet]).max())
    error, speed = float(max(errors)), float(max(speeds))
    row = _row("lake", result, wall_s, level_error_m=error, speed_ms=speed)
    if error > GATE_LAKE_LEVEL_M:
        row["fail_reasons"].append(f"level error {error:.2e} m")
    if speed > GATE_LAKE_SPEED_MS:
        row["fail_reasons"].append(f"speed {speed:.2e} m/s")
    return [row]


def case_ritter(work_dir: Path, cfl: float | None) -> list[dict[str, Any]]:
    def depth(x: NDArray[np.float64], _: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(x < 800.0, 2.0, 0.0)

    layout = Layout(_channel(_flat), [_x_below(CUT_M), _x_above(CUT_M)], depth=depth)
    expected, result, wall_s = _compare(work_dir, "ritter", layout, [Forcing(90.0)], cfl)
    return [_depth_row("ritter", expected, result, wall_s)]


def case_backwater(work_dir: Path, cfl: float | None) -> list[dict[str, Any]]:
    channel = _channel(_sloped)
    up, down = channel.end(0.0, "up"), channel.end(LENGTH_M, "down")
    layout = Layout(channel, [_x_below(CUT_M), _x_above(CUT_M)], ((up, 0), (down, 1)))
    forcing = [Forcing(3600.0, conditions={"up": Inflow(20.0, SLOPE), "down": Stage(10.0)})]
    expected, result, wall_s = _compare(work_dir, "backwater", layout, forcing, cfl)
    return [_depth_row("backwater", expected, result, wall_s)]


def case_reversal(work_dir: Path, cfl: float | None) -> list[dict[str, Any]]:
    layout = Layout(_channel(_flat), [_x_below(CUT_M), _x_above(CUT_M)])
    forcing = [
        Forcing(600.0, {1: [PointSource(20.0, (1100.0, 50.0), 30.0)]}),
        Forcing(600.0, {0: [PointSource(20.0, (900.0, 50.0), 30.0)]}),
    ]
    expected, result, wall_s = _compare(work_dir, "reversal", layout, forcing, cfl)
    return [_depth_row("reversal", expected, result, wall_s)]


def case_crest(work_dir: Path, cfl: float | None) -> list[dict[str, Any]]:
    rows = []
    for name, first, second, left, right in (
        ("crest_up", 11.5, 10.0, 0.0, 1.6),
        ("crest_down", 10.0, 11.5, 1.6, 0.0),
        ("step_down", 11.5, 10.0, 0.3, 0.0),
    ):

        def bed(
            x: NDArray[np.float64], low: float = first, high: float = second
        ) -> NDArray[np.float64]:
            return np.where(x < CUT_M, low, high)

        def depth(
            x: NDArray[np.float64],
            _: NDArray[np.float64],
            low: float = left,
            high: float = right,
        ) -> NDArray[np.float64]:
            return np.where(x < CUT_M, low, high)

        layout = Layout(_channel(bed), [_x_below(CUT_M), _x_above(CUT_M)], depth=depth)
        expected, result, wall_s = _compare(work_dir, name, layout, [Forcing(300.0)], cfl)
        rows.append(_depth_row(name, expected, result, wall_s))
    return rows


def case_confluence(work_dir: Path, cfl: float | None) -> list[dict[str, Any]]:
    channel = Channel(600.0, 60, 400.0, _flat, across=40)
    polygons = [
        shapely.box(-1.0, -1.0, 300.0, 200.0),
        shapely.box(-1.0, 200.0, 300.0, 401.0),
        shapely.box(300.0, -1.0, 601.0, 401.0),
    ]
    forcing = [
        Forcing(
            900.0,
            {
                0: [PointSource(10.0, (100.0, 100.0), 30.0)],
                1: [PointSource(10.0, (100.0, 300.0), 30.0)],
            },
        )
    ]
    expected, result, wall_s = _compare(
        work_dir, "confluence", Layout(channel, polygons), forcing, cfl
    )
    return [_depth_row("confluence", expected, result, wall_s)]


CASES: dict[str, Callable[[Path, float | None], list[dict[str, Any]]]] = {
    "lake": case_lake,
    "ritter": case_ritter,
    "backwater": case_backwater,
    "reversal": case_reversal,
    "crest": case_crest,
    "confluence": case_confluence,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Water between basin meshes, against one mesh")
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    parser.add_argument("--cfl", type=float, default=None, help="CFL number")
    parser.add_argument("--output-dir", type=Path, default=Path(".tmp/basin-exchange"))
    parser.add_argument("--repeats", type=int, default=2, help=">=2 enables determinism checks")
    parser.add_argument("--warmup", type=int, default=0, help="Runs of the lake case to discard")
    args = parser.parse_args()

    work_dir = args.output_dir / "meshes"
    for _ in range(args.warmup):
        case_lake(work_dir, args.cfl)

    correctness: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    for name in args.case or list(CASES):
        repeats = [CASES[name](work_dir, args.cfl) for _ in range(max(1, args.repeats))]
        deterministic = all(_metrics(rows) == _metrics(repeats[0]) for rows in repeats)
        for row, *others in zip(*repeats, strict=True):
            row["backend"] = "gpu_resident_batch"
            row["deterministic"] = deterministic
            if not deterministic:
                row["fail_reasons"].append("repeat runs differ")
            row["passed"] = not row["fail_reasons"]
            correctness.append(row)
            performance.append(
                {
                    "case": row["case"],
                    "backend": "gpu_resident_batch",
                    "repeats": len(repeats),
                    "median_wall_s": statistics.median(r["wall_s"] for r in (row, *others)),
                    "steps_total": row["steps"],
                }
            )
    summary = {
        "gates": {
            "balance_rel_max": GATE_BALANCE,
            "clamp_m3_max": GATE_CLAMP,
            "lake_level_error_m_max": GATE_LAKE_LEVEL_M,
            "lake_speed_ms_max": GATE_LAKE_SPEED_MS,
            "depth_difference_rel_max": GATE_DEPTH,
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
