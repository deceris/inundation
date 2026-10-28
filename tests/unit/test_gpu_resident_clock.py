# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# (pytest.approx's stub is partially unresolved; see test_bench_dambreak_analytical.py)
"""GPU-resident integer clock, in-step CFL and record-once loop (needs Vulkan).

The kernel tests drive ``step_control`` directly with hand-set control blocks;
the solver tests run the whole strategy on a small closed basin and check the
properties the change exists for: injected volume equals Q times elapsed time
regardless of the absolute start time, outputs land exactly on their times,
and the submit size does not change the physics.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from inundation.tuning import SIM_TICKS_PER_SECOND

from ._fixtures import requires_vulkan

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry

pytestmark = requires_vulkan

DT_SENTINEL_BITS = struct.unpack("<I", struct.pack("<f", 1e30))[0]


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


@pytest.fixture(scope="module")
def spv() -> dict[str, bytes]:
    from inundation.workflow import _compile_solver_shaders

    return _compile_solver_shaders()


# ── step_control kernel ──────────────────────────────────────────────────────


class StepControlHarness:
    """Run ``step_control`` once against a hand-built control block."""

    def __init__(self, spv: dict[str, bytes]) -> None:
        import kp

        from inundation.vulkan.gpu_resident_batch import RESIDENT_CTRL_DTYPE

        self._kp = kp
        self._mgr = kp.Manager()
        self._spv = spv["step_control"]
        self._ctrl_dtype = RESIDENT_CTRL_DTYPE

    def run(
        self,
        mode: Any,
        *,
        slot0: float = 0.0,
        slot1: float = 1e30,
        **fields: float,
    ) -> tuple[np.void, NDArray[np.uint32]]:
        kp = self._kp
        from inundation.vulkan.gpu_resident_batch import _pc_step_control

        ctrl = np.zeros(1, dtype=self._ctrl_dtype)
        ctrl["dt_max_ticks"] = SIM_TICKS_PER_SECOND  # 1 s
        ctrl["dry_dt_ticks"] = SIM_TICKS_PER_SECOND // 100
        ctrl["cfl_safety"] = 0.8
        ctrl["stop_ticks"] = 10**9
        for name, value in fields.items():
            ctrl[name] = value
        dt = np.array([_f32_bits(slot0), _f32_bits(slot1), 0, 0], dtype=np.uint32)

        dummies = [self._mgr.tensor(np.zeros(1, dtype=np.float32)) for _ in range(17)]
        t_dt = self._mgr.tensor(dt.view(np.float32))
        t_scratch = self._mgr.tensor(np.zeros(1, dtype=np.float32))
        t_ctrl = self._mgr.tensor(ctrl.view(np.float32).copy())
        tensors = [*dummies, t_dt, t_scratch, t_ctrl]
        algo = self._mgr.algorithm(
            tensors,
            self._spv,
            workgroup=(1, 1, 1),
            spec_consts=[],
            push_consts=_pc_step_control(mode),
        )
        (
            self._mgr.sequence()
            .record(kp.OpTensorSyncDevice(tensors))
            .record(kp.OpAlgoDispatch(algo))
            .record(kp.OpTensorSyncLocal([t_dt, t_ctrl]))
            .eval()
        )
        out_ctrl = np.array(t_ctrl.data(), dtype=np.float32).view(self._ctrl_dtype)[0]
        out_dt = np.array(t_dt.data(), dtype=np.float32).view(np.uint32)
        return out_ctrl, out_dt


@pytest.fixture(scope="module")
def harness(spv: dict[str, bytes]) -> StepControlHarness:
    return StepControlHarness(spv)


def _shader_ticks(raw: float, safety: float = 0.8) -> int:
    """Ticks step_control derives from a CFL bound, in the shader's float32."""
    scaled = np.float32(raw) * np.float32(safety)
    return int(scaled * np.float32(SIM_TICKS_PER_SECOND))


class TestStepControl:
    def test_start_takes_the_prologue_bound_with_safety(self, harness: StepControlHarness) -> None:
        from inundation.vulkan.gpu_resident_batch import StepControlMode

        ctrl, dt = harness.run(StepControlMode.START, slot0=0.05, now_ticks=123)
        expected = _shader_ticks(0.05)
        assert ctrl["dt_ticks"] == expected
        assert ctrl["now_ticks"] == 123  # start does not advance
        assert ctrl["steps"] == 0
        assert dt[0] == _f32_bits(expected / SIM_TICKS_PER_SECOND)
        assert dt[1] == DT_SENTINEL_BITS

    def test_dt_is_exactly_the_tick_count(self, harness: StepControlHarness) -> None:
        from inundation.vulkan.gpu_resident_batch import StepControlMode

        ctrl, dt = harness.run(StepControlMode.START, slot0=0.0314)
        dt_f = struct.unpack("<f", struct.pack("<I", int(dt[0])))[0]
        assert dt_f * SIM_TICKS_PER_SECOND == int(ctrl["dt_ticks"])

    def test_step_advances_by_the_previous_ticks_and_counts(
        self, harness: StepControlHarness
    ) -> None:
        from inundation.vulkan.gpu_resident_batch import StepControlMode

        ctrl, _ = harness.run(
            StepControlMode.STEP, slot1=0.02, now_ticks=1000, dt_ticks=777, steps=5
        )
        assert ctrl["now_ticks"] == 1777
        assert ctrl["steps"] == 6
        assert ctrl["dt_ticks"] == _shader_ticks(0.02)
        assert ctrl["last_raw"] == np.float32(0.02)

    def test_noop_step_neither_advances_nor_counts_and_keeps_the_last_bound(
        self, harness: StepControlHarness
    ) -> None:
        from inundation.vulkan.gpu_resident_batch import StepControlMode

        ctrl, _ = harness.run(
            StepControlMode.STEP, now_ticks=1000, dt_ticks=0, steps=5, last_raw=0.03
        )
        assert ctrl["now_ticks"] == 1000
        assert ctrl["steps"] == 5
        assert ctrl["dt_ticks"] == _shader_ticks(0.03)

    def test_retarget_replans_from_the_last_bound(self, harness: StepControlHarness) -> None:
        from inundation.vulkan.gpu_resident_batch import StepControlMode

        ctrl, _ = harness.run(
            StepControlMode.RETARGET, slot1=0.5, now_ticks=1000, dt_ticks=0, last_raw=0.03
        )
        assert ctrl["now_ticks"] == 1000
        assert ctrl["dt_ticks"] == _shader_ticks(0.03)

    def test_capped_by_dt_max(self, harness: StepControlHarness) -> None:
        from inundation.vulkan.gpu_resident_batch import StepControlMode

        ctrl, _ = harness.run(StepControlMode.START, slot0=50.0, dt_max_ticks=4096)
        assert ctrl["dt_ticks"] == 4096

    def test_last_step_lands_exactly_on_the_stop(self, harness: StepControlHarness) -> None:
        from inundation.vulkan.gpu_resident_batch import StepControlMode

        ctrl, _ = harness.run(
            StepControlMode.START, slot0=0.05, now_ticks=10_000, stop_ticks=10_005
        )
        assert ctrl["dt_ticks"] == 5

    def test_window_done_gives_a_zero_dt(self, harness: StepControlHarness) -> None:
        from inundation.vulkan.gpu_resident_batch import StepControlMode

        ctrl, dt = harness.run(
            StepControlMode.STEP, slot1=0.05, now_ticks=9_000, dt_ticks=1_000, stop_ticks=10_000
        )
        assert ctrl["now_ticks"] == 10_000
        assert ctrl["dt_ticks"] == 0
        assert dt[0] == 0

    def test_all_dry_falls_back_to_the_dry_dt(self, harness: StepControlHarness) -> None:
        from inundation.vulkan.gpu_resident_batch import StepControlMode

        ctrl, _ = harness.run(StepControlMode.START, slot0=1e30, dry_dt_ticks=2048)
        assert ctrl["dt_ticks"] == 2048

    def test_degenerate_bound_stalls_instead_of_stepping(self, harness: StepControlHarness) -> None:
        from inundation.vulkan.gpu_resident_batch import StepControlMode

        ctrl, _ = harness.run(StepControlMode.START, slot0=1e-12)
        assert ctrl["dt_ticks"] == 0

    def test_clock_carries_past_32_bits(self, harness: StepControlHarness) -> None:
        from inundation.vulkan.gpu_resident_batch import StepControlMode

        start = 2**32 - 3
        ctrl, _ = harness.run(
            StepControlMode.STEP, slot1=0.05, now_ticks=start, dt_ticks=10, stop_ticks=2**40
        )
        assert ctrl["now_ticks"] == start + 10


# ── whole solver on a small closed basin ─────────────────────────────────────

GRID = 24
CELL_M = 1.0
N_CELLS = GRID * GRID
SOURCE_CELL = N_CELLS // 2 + GRID // 2


def _square_grid() -> MeshGeometry:
    from inundation.mesh.geometry import build_geometry

    xs, ys = np.meshgrid(np.arange(GRID + 1) * CELL_M, np.arange(GRID + 1) * CELL_M)
    verts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    row = GRID + 1
    faces = np.array(
        [
            [j * row + i, j * row + i + 1, (j + 1) * row + i + 1, (j + 1) * row + i]
            for j in range(GRID)
            for i in range(GRID)
        ],
        dtype=np.int32,
    )
    return build_geometry(verts, faces, zb_from_file=np.zeros(N_CELLS, dtype=np.float32))


@pytest.fixture(scope="module")
def grid() -> MeshGeometry:
    return _square_grid()


@pytest.fixture
def make_solver(grid: MeshGeometry, spv: dict[str, bytes]) -> Callable[..., Any]:
    from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch

    def _make(h0: NDArray[np.float32], steps_per_graph: int | None = None) -> Any:
        zeros = np.zeros(N_CELLS, dtype=np.float32)
        solver = SWESolverGpuResidentBatch(
            grid, spv, h0, zeros, zeros, np.full(N_CELLS, 0.03, dtype=np.float32)
        )
        if steps_per_graph is not None:
            solver.steps_per_graph = steps_per_graph
        return solver

    return _make


def _volume(solver: Any, grid: MeshGeometry) -> float:
    return float((solver.download_h().astype(np.float64) * grid.area).sum())


def _point_source(q: float) -> NDArray[np.float32]:
    src = np.zeros(N_CELLS, dtype=np.float32)
    src[SOURCE_CELL] = q
    return src


def _run(solver: Any, **kwargs: Any) -> Any:
    defaults: dict[str, Any] = {
        "output_interval_s": 10.0,
        "dt_max": 1.0,
        "dt_init": 0.01,
        "cfl_interval": 10,
        "progress": False,
    }
    return solver.run(**{**defaults, **kwargs})


class TestInjectedVolume:
    @pytest.mark.parametrize("t0", [0.0, 70_000.0])
    def test_injected_volume_matches_the_schedule_at_any_start_time(
        self, make_solver: Callable[..., Any], grid: MeshGeometry, t0: float
    ) -> None:
        # Before the integer clock this case was off by -1.45 % at t0 = 70 000 s
        # (docs/implementation/21-resident-clock-and-batch-loop/results.md).
        q, duration = 1.0, 30.0
        h0 = np.full(N_CELLS, 0.05, dtype=np.float32)
        solver = make_solver(h0)
        v0 = _volume(solver, grid)
        _run(solver, t_end=t0 + duration, source_rate=_point_source(q), resume=True, t_start=t0)
        gained = _volume(solver, grid) - v0
        assert gained == pytest.approx(q * duration, rel=1e-4)

    def test_chained_phases_inject_each_phase_volume(
        self, make_solver: Callable[..., Any], grid: MeshGeometry
    ) -> None:
        h0 = np.full(N_CELLS, 0.05, dtype=np.float32)
        solver = make_solver(h0)
        v0 = _volume(solver, grid)
        _run(solver, t_end=20.0, source_rate=_point_source(0.5))
        _run(solver, t_end=50.0, source_rate=_point_source(1.5), resume=True, t_start=20.0)
        assert _volume(solver, grid) - v0 == pytest.approx(0.5 * 20.0 + 1.5 * 30.0, rel=1e-4)

    def test_dry_start_fills_from_the_source(
        self, make_solver: Callable[..., Any], grid: MeshGeometry
    ) -> None:
        solver = make_solver(np.zeros(N_CELLS, dtype=np.float32))
        _run(solver, t_end=10.0, source_rate=_point_source(0.2))
        assert _volume(solver, grid) == pytest.approx(0.2 * 10.0, rel=1e-4)


def _hump() -> NDArray[np.float32]:
    grid = _square_grid()
    cx, cy = grid.centroid[:, 0], grid.centroid[:, 1]
    centre = GRID * CELL_M / 2
    return (0.5 + 0.2 * np.exp(-((cx - centre) ** 2 + (cy - centre) ** 2) / 10.0)).astype(
        np.float32
    )


class TestTimeAndSteps:
    def test_snapshots_land_exactly_on_output_times(self, make_solver: Callable[..., Any]) -> None:
        solver = make_solver(_hump())
        _, times = _run(solver, t_end=10.0, output_interval_s=2.5)
        assert times == [0.0, 2.5, 5.0, 7.5, 10.0]

    def test_resumed_outputs_are_relative_to_the_phase_start(
        self, make_solver: Callable[..., Any]
    ) -> None:
        solver = make_solver(_hump())
        _run(solver, t_end=3.0)
        _, times = _run(solver, t_end=8.0, output_interval_s=2.0, resume=True, t_start=3.0)
        assert times == [3.0, 5.0, 7.0, 8.0]

    def test_steps_total_counts_real_steps_only(self, make_solver: Callable[..., Any]) -> None:
        # Large submits overshoot each window with no-op steps; those must not
        # be counted.
        small = make_solver(_hump(), steps_per_graph=1)
        large = make_solver(_hump(), steps_per_graph=500)
        _run(small, t_end=5.0, output_interval_s=1.0)
        _run(large, t_end=5.0, output_interval_s=1.0)
        assert small.steps_total == large.steps_total
        assert small.steps_total > 0

    @pytest.mark.parametrize("steps_per_graph", [1, 7, 500])
    def test_submit_size_does_not_change_the_result(
        self, make_solver: Callable[..., Any], steps_per_graph: int
    ) -> None:
        reference = make_solver(_hump(), steps_per_graph=200)
        candidate = make_solver(_hump(), steps_per_graph=steps_per_graph)
        _run(reference, t_end=5.0)
        _run(candidate, t_end=5.0)
        # Same dt sequence; float atomics may reorder the flux sums.
        np.testing.assert_allclose(candidate.download_h(), reference.download_h(), atol=1e-5)

    def test_closed_basin_conserves_volume(
        self, make_solver: Callable[..., Any], grid: MeshGeometry
    ) -> None:
        solver = make_solver(_hump())
        v0 = _volume(solver, grid)
        _run(solver, t_end=20.0)
        assert _volume(solver, grid) == pytest.approx(v0, rel=1e-6)

    def test_rejects_dt_max_beyond_the_tick_range(self, make_solver: Callable[..., Any]) -> None:
        solver = make_solver(_hump())
        with pytest.raises(ValueError, match="dt"):
            _run(solver, t_end=1.0, dt_max=32.0)


class TestAgreesWithBarrierReference:
    def test_hump_matches_fixed_dt_batch_barrier(
        self, grid: MeshGeometry, spv: dict[str, bytes], make_solver: Callable[..., Any]
    ) -> None:
        from inundation.vulkan.fixed_dt_batch_barrier import SWESolverFixedDtBatchBarrier

        zeros = np.zeros(N_CELLS, dtype=np.float32)
        reference = SWESolverFixedDtBatchBarrier(
            grid, spv, _hump(), zeros, zeros, np.full(N_CELLS, 0.03, dtype=np.float32)
        )
        resident = make_solver(_hump())
        _run(reference, t_end=5.0, cfl_interval=1)
        _run(resident, t_end=5.0)
        # Different dt sequences (the reference re-plans on its own cadence), so
        # agreement is to discretization error, not bitwise.
        np.testing.assert_allclose(resident.download_h(), reference.download_h(), atol=2e-3)
