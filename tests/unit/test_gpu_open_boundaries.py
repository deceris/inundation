# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownArgumentType=false
"""Open boundaries on the GPU-resident backend (needs Vulkan).

The analytical cases live in ``inundation.bench.open_boundary``. These pin what they
cannot show: the active set, repeatability, and how the tally lines up with a run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from inundation.bench.open_boundary import Channel, _write_mesh
from inundation.tuning import OPEN_INFLOW
from inundation.workflow import (
    FreeOutflow,
    Inflow,
    NormalDepth,
    SimulationPhase,
    Stage,
    SWEWorkflow,
    Wall,
    WorkflowConfig,
    WorkflowResult,
)

from ._fixtures import requires_vulkan

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

pytestmark = requires_vulkan

LENGTH_M = 3000.0
WIDTH_M = 200.0
CELLS = 300
ACROSS = 20
SLOPE = 1e-3
MANNING = 0.03
END_S = 60.0
DRAINING_DEPTH_M = 1.003e-4
DRAIN_S = 800.0


def _bed(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 10.0 - SLOPE * x


def _channel() -> Channel:
    return Channel(LENGTH_M, CELLS, WIDTH_M, _bed, across=ACROSS)


def _workflow(
    tmp_path: Path,
    *,
    open_boundaries: bool = True,
    initial_depth: float | NDArray[np.float64] | None = None,
    initial_velocity: float = 0.0,
    **config: Any,
) -> SWEWorkflow:
    channel = _channel()
    path = tmp_path / "channel.parquet"
    if not path.exists():
        _write_mesh(channel, path)
    state: dict[str, object] | None = None
    if initial_depth is not None:
        depth = np.broadcast_to(initial_depth, (CELLS * ACROSS,)).astype(np.float32)
        zeros = np.zeros(depth.size, dtype=np.float32)
        state = {"h": depth, "hu": initial_velocity * depth, "hv": zeros}
    workflow = SWEWorkflow(
        WorkflowConfig(
            mesh_source=str(path),
            manning_n=MANNING,
            output_interval_s=5.0,
            dt_max=1.0,
            cfl_interval=1,
            progress=False,
            track_clamp_mass=True,
            initial_state_source=state,
            initial_state_order="original",
            open_boundaries=(
                [channel.end(0.0, "up"), channel.end(LENGTH_M, "down")] if open_boundaries else ()
            ),
            **config,
        )
    )
    workflow.prepare()
    return workflow


def _bits(result: WorkflowResult) -> bytes:
    return result.h_final.tobytes() + b"".join(
        volumes.tobytes() for volumes in result.boundary_volumes_m3.values()
    )


def _phase(up: Any, down: Any, duration_s: float = END_S) -> SimulationPhase:
    return SimulationPhase(duration_s, [], {"up": up, "down": down})


@pytest.mark.parametrize("up", [Inflow(50.0, SLOPE), Stage(10.4)], ids=["inflow", "stage"])
def test_water_entering_a_dry_mesh_is_the_same_with_skipping_off(tmp_path: Path, up: Any) -> None:
    skipping = _workflow(tmp_path)
    assert skipping.solver is not None
    assert skipping.solver.n_blocks > 8
    on = skipping.run([_phase(up, NormalDepth(SLOPE))])
    inactive = int((skipping.solver.download_block_flags()[:, 1] == 0).sum())
    assert inactive > 0, "the active set must skip something for this test to mean anything"
    assert on.h_final.any()

    full = _workflow(tmp_path)
    full.solver.active_set = False
    off = full.run([_phase(up, NormalDepth(SLOPE))])
    assert _bits(on) == _bits(off)
    assert on.steps_total == off.steps_total


def test_repeat_runs_are_bit_identical(tmp_path: Path) -> None:
    phases = [_phase(Inflow(50.0, SLOPE), Stage(7.3))]
    assert _bits(_workflow(tmp_path).run(phases)) == _bits(_workflow(tmp_path).run(phases))


def test_volumes_line_up_with_the_snapshots(tmp_path: Path) -> None:
    result = _workflow(tmp_path).run(
        [_phase(Inflow(50.0, SLOPE), Wall(), 12.0), _phase(Inflow(20.0, SLOPE), Wall(), 8.0)]
    )
    volumes = result.boundary_volumes_m3["up"]
    assert volumes.shape == (len(result.snap_times), 2)
    assert (np.diff(volumes[:, 1]) >= 0.0).all()
    times = np.asarray(result.snap_times)
    for k in np.flatnonzero(times == 12.0):
        assert volumes[k, 1] == pytest.approx(50.0 * 12.0, rel=1e-6)
    assert volumes[-1, 1] == pytest.approx(50.0 * 12.0 + 20.0 * 8.0, rel=1e-6)


def test_a_trickle_too_thin_to_wet_a_cell_still_arrives(tmp_path: Path) -> None:
    discharge = 1e-3
    result = _workflow(tmp_path).run([_phase(Inflow(discharge, SLOPE), Wall())])
    assert float(result.h_final.max()) < 1e-4
    assert result.boundary_volumes_m3["up"][-1, 1] == pytest.approx(discharge * END_S, rel=1e-6)
    assert result.volume_final_m3 == pytest.approx(discharge * END_S, rel=1e-6)


def test_balance_closes_while_water_enters_and_leaves(tmp_path: Path) -> None:
    initial = 0.5 * LENGTH_M * WIDTH_M
    result = _workflow(tmp_path, initial_depth=0.5).run(
        [_phase(Inflow(50.0, SLOPE), NormalDepth(SLOPE), 120.0)]
    )
    went_in = result.boundary_volumes_m3["up"][-1, 1]
    went_out = result.boundary_volumes_m3["down"][-1, 0]
    assert went_out > 0.0
    balance = initial + went_in - went_out - result.volume_final_m3
    assert abs(balance) / initial < 1e-7


@pytest.mark.parametrize("down", [FreeOutflow(), NormalDepth(SLOPE)], ids=["free", "normal"])
def test_outflow_conditions_never_draw_water_in(tmp_path: Path, down: Any) -> None:
    result = _workflow(tmp_path, initial_depth=0.5).run([_phase(Wall(), down)])
    assert result.boundary_volumes_m3["down"][-1, 1] == 0.0


@pytest.mark.parametrize("down", [FreeOutflow(), NormalDepth(SLOPE)], ids=["free", "normal"])
def test_flow_running_away_from_an_outlet_draws_nothing_in(tmp_path: Path, down: Any) -> None:
    result = _workflow(tmp_path, initial_depth=0.5, initial_velocity=-0.5).run(
        [_phase(Wall(), down, 5.0)]
    )
    assert result.boundary_volumes_m3["down"][-1, 1] == 0.0


def test_an_outlet_that_draws_nothing_in_acts_as_a_wall(tmp_path: Path) -> None:
    # Water running away from the outlet faster than the normal-depth ghost
    # leaves: the Riemann flux points inward at every stage of this short run,
    # so the edge must be a wall for momentum as well as mass. FreeOutflow
    # cannot show this: away from the face its ghost already is the wall mirror.
    def run(down: Any) -> WorkflowResult:
        workflow = _workflow(tmp_path, initial_depth=0.5, initial_velocity=-2.0)
        return workflow.run([_phase(Wall(), down, 1.0)])

    closed, outlet = run(Wall()), run(NormalDepth(1e-5))
    assert outlet.hu_final is not None and closed.hu_final is not None
    assert np.array_equal(outlet.h_final, closed.h_final)
    assert np.array_equal(outlet.hu_final, closed.hu_final)


def test_a_dry_outflow_passes_nothing(tmp_path: Path) -> None:
    result = _workflow(tmp_path).run([_phase(Wall(), FreeOutflow())])
    assert not result.boundary_volumes_m3["down"].any()
    assert not result.h_final.any()


def test_the_tally_stops_when_an_outlet_runs_dry(tmp_path: Path) -> None:
    # The film damping stalls a real sheet above dry_tol, so start just over it:
    # the outlet cells go dry mid-run, with their last flux words nonzero.
    channel = _channel()
    depth = np.where(channel.centres > LENGTH_M - channel.dx_m, DRAINING_DEPTH_M, 0.0)
    initial = float(depth.sum()) * channel.dx_m * WIDTH_M / ACROSS
    result = _workflow(tmp_path, initial_depth=depth).run(
        [_phase(Wall(), NormalDepth(1.0), DRAIN_S)]
    )
    assert float(result.h_final.max()) < 1e-4
    went_out = result.boundary_volumes_m3["down"][-1, 0]
    assert went_out > 0.0
    assert abs(initial - went_out - result.volume_final_m3) / initial < 1e-5


def test_the_first_step_sees_the_water_waiting_beyond_a_stage(tmp_path: Path) -> None:
    # A 20 m ghost over a 1 cm cell: bounding dt by the cell alone takes one 1 s step.
    result = _workflow(tmp_path, initial_depth=0.01).run([_phase(Stage(30.0), Wall(), 1.0)])
    assert result.steps_total >= 3


def test_closing_a_stretch_releases_its_blocks(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    solver = cast("Any", workflow.solver)
    workflow.run([_phase(Inflow(50.0, SLOPE), Wall(), 5.0)])
    assert solver.download_block_flags()[:, 2].sum() > 0
    workflow.run([_phase(Wall(), Wall(), 5.0)])
    assert solver.download_block_flags()[:, 2].sum() == 0


def test_inflow_only_forces_blocks_while_it_flows(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    solver = cast("Any", workflow.solver)
    kinds = np.full(solver.n_open, OPEN_INFLOW, dtype=np.uint32)
    solver.set_open_conditions(kinds, np.zeros(solver.n_open, dtype=np.float32))
    assert solver.download_block_flags()[:, 2].sum() == 0


def test_a_wall_stretch_behaves_like_no_stretch(tmp_path: Path) -> None:
    closed = _workflow(tmp_path, initial_depth=0.5).run([_phase(Wall(), Wall())])
    plain = _workflow(tmp_path, open_boundaries=False, initial_depth=0.5).run(
        [SimulationPhase(END_S, [])]
    )
    np.testing.assert_allclose(closed.h_final, plain.h_final, atol=1e-5)
    assert closed.steps_total == plain.steps_total


def test_inflow_carries_momentum_into_the_mesh(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path, initial_depth=0.5)
    result = workflow.run([_phase(Inflow(200.0, SLOPE), Wall())])
    assert workflow.geom is not None and result.hu_final is not None
    inlet = workflow.geom.world_centroid[:, 0] < 2.0 * LENGTH_M / CELLS
    assert float(result.hu_final[inlet].mean()) > 0.05
