# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
"""Momentum snapshots on the GPU-resident backend (needs Vulkan).

``hu_snapshots`` and ``hv_snapshots`` must line up one-for-one with the depth
snapshots, hold the committed state at each output time, and leave the
simulation itself untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from inundation.tuning import ACTIVE_BLOCK_CELLS, CFL_HOST_DRIVEN, MOMENTUM_SNAPSHOT_BACKENDS
from inundation.workflow import PointSource, SimulationPhase, SWEWorkflow, WorkflowConfig

from ._fixtures import requires_vulkan
from ._valley import DX, MANNING, NX, NY, point_source, sheet, valley_grid

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry
    from inundation.workflow import WorkflowResult

pytestmark = requires_vulkan

OUTPUT_INTERVAL_S = 5.0
MESH_CELLS_PER_SIDE = 48
MESH_DX = 2.0
SOURCE_XY = (10.0, 48.0)
# Phases of 12 s and 8 s end between outputs, so each has a tail snapshot.
PHASE_DURATIONS_S = (12.0, 8.0)
# The last phase ends on an output time, or between two.
LAST_ON_OUTPUT_S = 10.0
LAST_BETWEEN_OUTPUTS_S = 13.0
# Each phase contributes its start: snapshots at both ends of a boundary.
PHASE_BOUNDARIES_S = (12.0, 20.0)
# Thick enough that the draining sheet falls below it inside the test's run.
DRAIN_DRY_TOL = 5e-3


def _write_sloped_mesh(path: Path) -> None:
    n = MESH_CELLS_PER_SIDE
    lines = [
        f"v {i * MESH_DX} {j * MESH_DX} {10.0 - 0.02 * i * MESH_DX}"
        for j in range(n + 1)
        for i in range(n + 1)
    ]

    def vid(i: int, j: int) -> int:
        return j * (n + 1) + i + 1

    lines += [
        f"f {vid(i, j)} {vid(i + 1, j)} {vid(i + 1, j + 1)} {vid(i, j + 1)}"
        for j in range(n)
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _run_workflow(
    mesh: Path, *, capture: bool, last_phase_s: float, solver_impl: Any = "gpu_resident_batch"
) -> WorkflowResult:
    workflow = SWEWorkflow(
        WorkflowConfig(
            mesh_source=str(mesh),
            manning_n=MANNING,
            output_interval_s=OUTPUT_INTERVAL_S,
            dt_max=1.0,
            cfl_interval=10,
            progress=False,
            capture_momentum_snapshots=capture,
            solver_impl=solver_impl,
        )
    )
    workflow.prepare()
    return workflow.run(
        [
            SimulationPhase(PHASE_DURATIONS_S[0], [PointSource(2.0, SOURCE_XY, 3.0)]),
            SimulationPhase(PHASE_DURATIONS_S[1], [PointSource(6.0, SOURCE_XY, 3.0)]),
            SimulationPhase(last_phase_s, []),
        ]
    )


@pytest.fixture(scope="module")
def mesh(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("momentum") / "sloped.obj"
    _write_sloped_mesh(path)
    return path


@pytest.fixture(scope="module")
def captured(mesh: Path) -> WorkflowResult:
    return _run_workflow(mesh, capture=True, last_phase_s=LAST_ON_OUTPUT_S)


@pytest.fixture(scope="module")
def captured_tail(mesh: Path) -> WorkflowResult:
    return _run_workflow(mesh, capture=True, last_phase_s=LAST_BETWEEN_OUTPUTS_S)


def _momentum(result: WorkflowResult) -> tuple[list[Any], list[Any]]:
    assert result.hu_snapshots is not None
    assert result.hv_snapshots is not None
    return result.hu_snapshots, result.hv_snapshots


class TestWorkflowContract:
    @pytest.mark.parametrize("fixture_name", ["captured", "captured_tail"])
    def test_lists_align_across_phases(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        result: WorkflowResult = request.getfixturevalue(fixture_name)
        hu, hv = _momentum(result)
        assert len(hu) == len(hv) == len(result.snapshots) == len(result.snap_times)
        assert max(float(np.abs(a).max()) for a in hu) > 0.0
        assert max(float(np.abs(a).max()) for a in hv) > 0.0

    def test_phase_boundary_duplicate_is_kept_for_momentum_too(
        self, captured: WorkflowResult
    ) -> None:
        hu, hv = _momentum(captured)
        times = np.asarray(captured.snap_times)
        for boundary in PHASE_BOUNDARIES_S:
            first, second = (int(k) for k in np.flatnonzero(times == boundary))
            assert second == first + 1
            np.testing.assert_array_equal(captured.snapshots[first], captured.snapshots[second])
            np.testing.assert_array_equal(hu[first], hu[second])
            np.testing.assert_array_equal(hv[first], hv[second])

    def test_initial_snapshot_is_the_initial_state(self, captured: WorkflowResult) -> None:
        hu, hv = _momentum(captured)
        assert captured.snap_times[0] == 0.0
        assert float(np.abs(hu[0]).max()) == 0.0
        assert float(np.abs(hv[0]).max()) == 0.0

    @pytest.mark.parametrize("fixture_name", ["captured", "captured_tail"])
    def test_last_snapshot_is_the_final_state(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        result: WorkflowResult = request.getfixturevalue(fixture_name)
        assert result.snap_times[-1] == sum(PHASE_DURATIONS_S) + (
            LAST_ON_OUTPUT_S if fixture_name == "captured" else LAST_BETWEEN_OUTPUTS_S
        )
        hu, hv = _momentum(result)
        assert result.hu_final is not None
        assert result.hv_final is not None
        np.testing.assert_array_equal(hu[-1], result.hu_final)
        np.testing.assert_array_equal(hv[-1], result.hv_final)

    def test_depth_is_bit_identical_with_the_flag_on_and_off(
        self, mesh: Path, captured_tail: WorkflowResult
    ) -> None:
        off = _run_workflow(mesh, capture=False, last_phase_s=LAST_BETWEEN_OUTPUTS_S)
        assert off.hu_snapshots is None
        assert off.hv_snapshots is None
        assert off.snap_times == captured_tail.snap_times
        assert off.steps_total == captured_tail.steps_total
        assert len(off.snapshots) == len(captured_tail.snapshots)
        for a, b in zip(
            [*off.snapshots, off.h_final],
            [*captured_tail.snapshots, captured_tail.h_final],
            strict=True,
        ):
            assert a.tobytes() == b.tobytes()

    @pytest.mark.parametrize("solver_impl", sorted(MOMENTUM_SNAPSHOT_BACKENDS))
    def test_every_listed_backend_delivers(self, mesh: Path, solver_impl: str) -> None:
        result = _run_workflow(
            mesh, capture=True, last_phase_s=LAST_ON_OUTPUT_S, solver_impl=solver_impl
        )
        hu, hv = _momentum(result)
        assert len(hu) == len(hv) == len(result.snapshots)


@pytest.fixture(scope="module")
def spv() -> dict[str, bytes]:
    from inundation.workflow import _compile_solver_shaders

    return _compile_solver_shaders()


@pytest.fixture(scope="module")
def grid() -> MeshGeometry:
    return valley_grid()


@pytest.fixture
def make_resident(grid: MeshGeometry, spv: dict[str, bytes]) -> Callable[..., Any]:
    from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch

    def _make(h0: NDArray[np.float32], **kwargs: Any) -> Any:
        zeros = np.zeros(grid.N, dtype=np.float32)
        return SWESolverGpuResidentBatch(
            grid, spv, h0, zeros, zeros, np.full(grid.N, MANNING, np.float32), **kwargs
        )

    return _make


def _run_solver(solver: Any, t_end: float, interval_s: float, **kwargs: Any) -> Any:
    return solver.run(
        t_end=t_end,
        output_interval_s=interval_s,
        dt_max=1.0,
        dt_init=0.01,
        cfl_interval=kwargs.pop("cfl_interval", 10),
        progress=False,
        **kwargs,
    )


class TestStateAtOutputTime:
    def test_snapshot_equals_a_run_stopped_at_that_time(
        self, grid: MeshGeometry, make_resident: Callable[..., Any]
    ) -> None:
        # A flag-off run to an output time takes the same steps, so with
        # deterministic sums its final state is the exact expected snapshot.
        # This fails if the download reads the Heun predictor (hu1, hv1).
        src = point_source(grid, 1.0, NX * DX * 0.2, NY * DX / 2)
        h0 = np.zeros(grid.N, dtype=np.float32)
        interval = 10.0
        solver = make_resident(h0)
        _, times, hu, hv = _run_solver(
            solver, 30.0, interval, source_rate=src, capture_momentum_snapshots=True
        )
        assert times == [0.0, 10.0, 20.0, 30.0]
        for k in (1, 2, 3):
            stopped = make_resident(h0)
            _run_solver(stopped, times[k], interval, source_rate=src)
            assert hu[k].tobytes() == stopped.download_hu().tobytes()
            assert hv[k].tobytes() == stopped.download_hv().tobytes()

    def test_dried_inactive_blocks_hold_zero_momentum(
        self, grid: MeshGeometry, make_resident: Callable[..., Any]
    ) -> None:
        # A sheet drains off the upstream slope, so blocks that carried flow go
        # dry and then inactive: their last momentum must not be frozen there.
        h0 = sheet(grid)
        on = make_resident(h0, dry_tol=DRAIN_DRY_TOL)
        off = make_resident(h0, dry_tol=DRAIN_DRY_TOL)
        off.active_set = False
        snaps, _, hu, hv = _run_solver(on, 3000.0, 500.0, capture_momentum_snapshots=True)
        snaps_off, _, hu_off, hv_off = _run_solver(
            off, 3000.0, 500.0, capture_momentum_snapshots=True
        )

        flows_in_block = np.zeros(on.n_blocks, dtype=bool)
        for h_k, hu_k, hv_k in zip(snaps, hu, hv, strict=True):
            dry = h_k < DRAIN_DRY_TOL
            assert float(np.abs(hu_k[dry]).max(initial=0.0)) == 0.0
            assert float(np.abs(hv_k[dry]).max(initial=0.0)) == 0.0
            moving = np.flatnonzero((hu_k != 0.0) | (hv_k != 0.0))
            flows_in_block[moving // ACTIVE_BLOCK_CELLS] = True

        inactive = on.download_block_flags()[:, 1] == 0
        assert int(np.count_nonzero(flows_in_block & inactive)) > 0
        # Compared by value: a skipped dry cell may keep a -0.0 (see test_gpu_active_set).
        for a, b in zip([*snaps, *hu, *hv], [*snaps_off, *hu_off, *hv_off], strict=True):
            np.testing.assert_array_equal(a, b)


# Relative L2 distance of (hu, hv) from the barrier reference, over the mesh.
# The two backends take different steps (the barrier reuses dt within a batch)
# and the barrier's depth is not compensated, so they agree only up to the
# scheme's truncation error. Measured on MoltenVK: 0.12-0.29 %.
BARRIER_REL_L2_MAX = 0.01


class TestAgainstBarrierBackend:
    def test_momentum_matches_the_host_driven_reference(
        self, grid: MeshGeometry, spv: dict[str, bytes], make_resident: Callable[..., Any]
    ) -> None:
        # The barrier backend has no snapshots of its own. Stopping it at each
        # output time reads its state at the same simulated time, which is what
        # its own output cadence would not: it snapshots on the step past a time.
        from inundation.vulkan.fixed_dt_batch_barrier import SWESolverFixedDtBatchBarrier

        src = point_source(grid, 1.0, NX * DX * 0.2, NY * DX / 2)
        h0 = np.zeros(grid.N, dtype=np.float32)
        interval = 20.0
        resident = make_resident(h0, cfl=CFL_HOST_DRIVEN)
        _, times, hu, hv = _run_solver(
            resident, 60.0, interval, source_rate=src, capture_momentum_snapshots=True
        )
        assert times == [0.0, 20.0, 40.0, 60.0]

        zeros = np.zeros(grid.N, dtype=np.float32)
        barrier = SWESolverFixedDtBatchBarrier(
            grid, spv, h0, zeros, zeros, np.full(grid.N, MANNING, np.float32), cfl=CFL_HOST_DRIVEN
        )
        for k in (1, 2, 3):
            _run_solver(
                barrier,
                times[k],
                interval,
                source_rate=src,
                cfl_interval=1,
                resume=k > 1,
                t_start=times[k - 1],
            )
            q_ref = np.concatenate([barrier.download_hu(), barrier.download_hv()]).astype(float)
            q = np.concatenate([hu[k], hv[k]]).astype(float)
            assert float(np.linalg.norm(q_ref)) > 0.0
            rel = float(np.linalg.norm(q - q_ref) / np.linalg.norm(q_ref))
            assert rel < BARRIER_REL_L2_MAX
