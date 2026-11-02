# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportOptionalMemberAccess=false, reportUnknownArgumentType=false
"""Two basins passing water both ways across a border (needs Vulkan).

A channel cut in two against the same channel in one piece. Each case is something one-way
exchange gets wrong, or the failure that sank an earlier scheme: see
``docs/implementation/30-two-way-coupling/plan.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from inundation.bench.open_boundary import Channel, _write_mesh
from inundation.coupling import Basin, CoupledResult, run_two_basins
from inundation.tuning import BORDER_TOLERANCE_M, OPEN_EXCHANGE
from inundation.workflow import (
    Inflow,
    NormalDepth,
    OpenBoundary,
    PointSource,
    SimulationPhase,
    Stage,
    SWEWorkflow,
    WorkflowConfig,
)

from ._fixtures import requires_vulkan

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch

pytestmark = requires_vulkan

LENGTH_M = 2000.0
WIDTH_M = 100.0
CELLS = 100
ACROSS = 4
CUT_M = 1000.0
SLOPE = 1e-3
MANNING = 0.03
WINDOW_S = 10.0
RUN_S = 600.0
BALANCE = 1e-6


def _sloped(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 10.0 - SLOPE * x


def _flat(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.full(x.shape, 10.0)


def _border() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    dy = WIDTH_M / ACROSS
    return [((CUT_M, k * dy), (CUT_M, (k + 1) * dy)) for k in range(ACROSS)]


def _config(
    tmp_path: Path,
    name: str,
    channel: Channel,
    boundaries: Sequence[OpenBoundary],
    depth: NDArray[np.float64] | None = None,
) -> WorkflowConfig:
    mesh = tmp_path / f"{name}.parquet"
    _write_mesh(channel, mesh)
    state: dict[str, object] | None = None
    if depth is not None:
        zeros = np.zeros(depth.size, dtype=np.float32)
        state = {"h": depth.astype(np.float32), "hu": zeros, "hv": zeros}
    return WorkflowConfig(
        mesh_source=str(mesh),
        manning_n=MANNING,
        output_interval_s=WINDOW_S,
        dt_max=1.0,
        cfl_interval=1,
        progress=False,
        track_clamp_mass=True,
        open_boundaries=boundaries,
        initial_state_source=state,
        initial_state_order="original",
    )


def _channels(
    bed: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> tuple[Channel, Channel, Channel]:
    half = CELLS // 2
    return (
        Channel(LENGTH_M, CELLS, WIDTH_M, bed, across=ACROSS),
        Channel(CUT_M, half, WIDTH_M, bed, across=ACROSS),
        Channel(CUT_M, half, WIDTH_M, bed, x0_m=CUT_M, across=ACROSS),
    )


def _whole_depth(config: WorkflowConfig, phases: Sequence[SimulationPhase]) -> NDArray[np.float64]:
    workflow = SWEWorkflow(config)
    workflow.prepare()
    result = workflow.run(phases)
    x = workflow.geom.world_centroid[:, 0].astype(np.float64)
    return _by_x(x, result.h_final.astype(np.float64))


def _by_x(x: NDArray[np.float64], h: NDArray[np.float64]) -> NDArray[np.float64]:
    """Depth along the channel, averaged across it."""
    columns = np.round(x / (LENGTH_M / CELLS)).astype(np.int64)
    return np.bincount(columns, h)[np.unique(columns)] / np.bincount(columns)[np.unique(columns)]


def _coupled_depth(result: CoupledResult) -> NDArray[np.float64]:
    xs, hs = [], []
    for basin in (result.first, result.second):
        xs.append(basin.workflow.geom.world_centroid[:, 0].astype(np.float64))
        hs.append(basin.h_final)
    return _by_x(np.concatenate(xs), np.concatenate(hs))


def _assert_clean(result: CoupledResult) -> None:
    assert abs(result.balance_gap_m3) <= BALANCE * (result.volume_start_m3 + result.volume_in_m3)
    assert result.first.clamp_mass_m3 == 0.0
    assert result.second.clamp_mass_m3 == 0.0


def test_a_lake_stays_at_rest_across_the_border(tmp_path: Path) -> None:
    _, upper, lower = _channels(_sloped)
    level = 10.5
    bases = [
        Basin(
            _config(tmp_path, name, channel, (), np.maximum(level - _sloped(channel.centres), 0.0)),
            [SimulationPhase(RUN_S, [])],
        )
        for name, channel in (("upper", upper), ("lower", lower))
    ]
    result = run_two_basins(bases[0], bases[1], _border(), WINDOW_S)
    for basin in (result.first, result.second):
        x = basin.workflow.geom.world_centroid[:, 0].astype(np.float64)
        assert np.abs(basin.h_final + _sloped(x) - level).max() < 1e-4
    assert result.exchanged_m3 < BALANCE * result.volume_stored_m3


def test_a_flood_crosses_the_border_as_in_one_channel(tmp_path: Path) -> None:
    whole, upper, lower = _channels(_sloped)
    up, down = whole.end(0.0, "up"), whole.end(LENGTH_M, "down")
    phases = [SimulationPhase(RUN_S, [], {"up": Inflow(30.0, SLOPE), "down": NormalDepth(SLOPE)})]
    expected = _whole_depth(_config(tmp_path, "whole", whole, [up, down]), phases)
    result = run_two_basins(
        Basin(
            _config(tmp_path, "upper", upper, [up]),
            [SimulationPhase(RUN_S, [], {"up": Inflow(30.0, SLOPE)})],
        ),
        Basin(
            _config(tmp_path, "lower", lower, [down]),
            [SimulationPhase(RUN_S, [], {"down": NormalDepth(SLOPE)})],
        ),
        _border(),
        WINDOW_S,
    )
    _assert_clean(result)
    wet = expected > 0.01
    assert np.abs(_coupled_depth(result) - expected)[wet].max() < 0.03 * expected.max()


def test_backwater_reaches_the_upper_basin(tmp_path: Path) -> None:
    """The failure of one-way exchange: a level below the cut must reach above it."""
    whole, upper, lower = _channels(_sloped)
    up, down = whole.end(0.0, "up"), whole.end(LENGTH_M, "down")
    run_s = 1800.0
    expected = _whole_depth(
        _config(tmp_path, "whole", whole, [up, down]),
        [SimulationPhase(run_s, [], {"up": Inflow(20.0, SLOPE), "down": Stage(10.0)})],
    )
    result = run_two_basins(
        Basin(
            _config(tmp_path, "upper", upper, [up]),
            [SimulationPhase(run_s, [], {"up": Inflow(20.0, SLOPE)})],
        ),
        Basin(
            _config(tmp_path, "lower", lower, [down]),
            [SimulationPhase(run_s, [], {"down": Stage(10.0)})],
        ),
        _border(),
        WINDOW_S,
    )
    _assert_clean(result)
    assert np.abs(_coupled_depth(result) - expected).max() < 0.03 * expected.max()


def test_water_flows_back_up_across_the_border(tmp_path: Path) -> None:
    """The failure of a signed flux: the lower basin gives water to the upper one."""
    whole, upper, lower = _channels(_flat)
    source = PointSource(20.0, (1100.0, 50.0), 30.0)
    expected = _whole_depth(
        _config(tmp_path, "whole", whole, ()), [SimulationPhase(RUN_S, [source])]
    )
    result = run_two_basins(
        Basin(_config(tmp_path, "upper", upper, ()), [SimulationPhase(RUN_S, [])]),
        Basin(_config(tmp_path, "lower", lower, ()), [SimulationPhase(RUN_S, [source])]),
        _border(),
        WINDOW_S,
    )
    _assert_clean(result)
    assert result.first.h_final.max() > 0.05
    assert np.abs(_coupled_depth(result) - expected).max() < 0.05 * expected.max()


def test_a_higher_dry_bed_passes_nothing(tmp_path: Path) -> None:
    def step(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(x < CUT_M, 10.0, 11.0)

    _, upper, lower = _channels(step)
    result = run_two_basins(
        Basin(
            _config(tmp_path, "upper", upper, (), np.full(upper.centres.size, 0.5)),
            [SimulationPhase(RUN_S, [])],
        ),
        Basin(_config(tmp_path, "lower", lower, ()), [SimulationPhase(RUN_S, [])]),
        _border(),
        WINDOW_S,
    )
    _assert_clean(result)
    assert result.second.h_final.max() == 0.0
    assert result.exchanged_m3 == 0.0


@pytest.mark.parametrize(
    ("first_bed", "second_bed", "first_depth", "second_depth"),
    [(11.5, 10.0, 0.0, 1.6), (10.0, 11.5, 1.6, 0.0), (11.5, 10.0, 0.3, 0.0)],
    ids=["up over a crest", "down over a crest", "down a step"],
)
def test_water_crosses_a_bed_step_as_in_one_channel(
    tmp_path: Path, first_bed: float, second_bed: float, first_depth: float, second_depth: float
) -> None:
    """Only the depth over the crest drives the flow, not the depth below it."""

    def bed(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(x < CUT_M, first_bed, second_bed)

    def full(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(x < CUT_M, first_depth, second_depth)

    whole, upper, lower = _channels(bed)
    run = [SimulationPhase(WINDOW_S, [], {}) for _ in range(30)]
    control = _whole_depth(_config(tmp_path, "whole", whole, (), full(whole.centres)), run)
    result = run_two_basins(
        Basin(_config(tmp_path, "upper", upper, (), full(upper.centres)), run),
        Basin(_config(tmp_path, "lower", lower, (), full(lower.centres)), run),
        _border(),
        WINDOW_S,
    )
    _assert_clean(result)
    assert np.abs(_coupled_depth(result) - control).max() < 0.02


def test_a_trickle_too_thin_to_wet_a_cell_still_arrives(tmp_path: Path) -> None:
    """Its ghost is below the dry tolerance, so only the live edge keeps the volume."""
    q = 1e-7
    _, upper, _ = _channels(_flat)
    segment = OpenBoundary("edge", list(_border()[0]), BORDER_TOLERANCE_M)
    workflow = SWEWorkflow(_config(tmp_path, "upper", upper, (segment,)))
    workflow.prepare()
    solver = cast("SWESolverGpuResidentBatch", workflow.solver)
    solver.set_open_conditions(
        np.array([OPEN_EXCHANGE], dtype=np.uint32),
        np.zeros(1, dtype=np.float32),
        np.array([q], dtype=np.float32),
    )
    solver.run(
        t_end=WINDOW_S,
        output_interval_s=WINDOW_S,
        dt_max=1.0,
        dt_init=0.01,
        cfl_interval=1,
        progress=False,
    )
    stored = float((solver.download_h_precise() * workflow.geom.area.astype(np.float64)).sum())
    assert stored == pytest.approx(q * WIDTH_M / ACROSS * WINDOW_S, rel=1e-3)


def test_repeat_runs_are_bit_identical(tmp_path: Path) -> None:
    _, upper, lower = _channels(_sloped)
    up = upper.end(0.0, "up")

    def run() -> bytes:
        result = run_two_basins(
            Basin(
                _config(tmp_path, "upper", upper, [up]),
                [SimulationPhase(120.0, [], {"up": Inflow(30.0, SLOPE)})],
            ),
            Basin(_config(tmp_path, "lower", lower, ()), [SimulationPhase(120.0, [])]),
            _border(),
            WINDOW_S,
        )
        return result.first.h_final.tobytes() + result.second.h_final.tobytes()

    assert run() == run()


def test_snapshots_land_on_output_times(tmp_path: Path) -> None:
    _, upper, lower = _channels(_sloped)
    up = upper.end(0.0, "up")
    result = run_two_basins(
        Basin(
            _config(tmp_path, "upper", upper, [up]),
            [SimulationPhase(60.0, [], {"up": Inflow(30.0, SLOPE)})],
        ),
        Basin(_config(tmp_path, "lower", lower, ()), [SimulationPhase(60.0, [])]),
        _border(),
        WINDOW_S,
        output_interval_s=30.0,
    )
    assert result.snap_times == [30.0, 60.0]
    assert len(result.first.snapshots) == len(result.second.snapshots) == 2


def test_a_window_that_does_not_divide_a_phase_raises(tmp_path: Path) -> None:
    _, upper, lower = _channels(_sloped)
    basins = [
        Basin(_config(tmp_path, name, channel, ()), [SimulationPhase(95.0, [])])
        for name, channel in (("upper", upper), ("lower", lower))
    ]
    with pytest.raises(ValueError, match=r"whole number of 10\.0 s windows"):
        run_two_basins(basins[0], basins[1], _border(), WINDOW_S)


def test_basins_of_different_lengths_raise(tmp_path: Path) -> None:
    _, upper, lower = _channels(_sloped)
    first = Basin(_config(tmp_path, "upper", upper, ()), [SimulationPhase(100.0, [])])
    second = Basin(_config(tmp_path, "lower", lower, ()), [SimulationPhase(200.0, [])])
    with pytest.raises(ValueError, match="must run for the same time"):
        run_two_basins(first, second, _border(), WINDOW_S)


def test_a_segment_off_one_mesh_raises(tmp_path: Path) -> None:
    _, upper, lower = _channels(_sloped)
    basins = [
        Basin(_config(tmp_path, name, channel, ()), [SimulationPhase(20.0, [])])
        for name, channel in (("upper", upper), ("lower", lower))
    ]
    with pytest.raises(ValueError, match="selects no boundary edges"):
        run_two_basins(basins[0], basins[1], [((500.0, 0.0), (500.0, 25.0))], WINDOW_S)
