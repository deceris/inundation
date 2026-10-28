# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
"""GPU-resident deterministic accumulation (needs Vulkan).

With ``deterministic`` on, fluxes and CFL terms are summed per cell in a fixed
order, so repeated runs must be bit-identical. The gather must also never read
an edge value that was not written in the current stage, and its results must
match the float-atomic path to float32 noise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ._fixtures import requires_vulkan
from ._valley import DX, MANNING, NX, NY, dam, point_source, run, sheet, state, valley_grid

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry

pytestmark = requires_vulkan

ATOL_H = 1e-5


@pytest.fixture(scope="module")
def spv() -> dict[str, bytes]:
    from inundation.workflow import _compile_solver_shaders

    return _compile_solver_shaders()


@pytest.fixture(scope="module")
def grid() -> MeshGeometry:
    return valley_grid()


@pytest.fixture
def make_solver(grid: MeshGeometry, spv: dict[str, bytes]) -> Callable[..., Any]:
    from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch

    def _make(h0: NDArray[np.float32], *, deterministic: bool = True) -> Any:
        zeros = np.zeros(grid.N, dtype=np.float32)
        solver = SWESolverGpuResidentBatch(
            grid, spv, h0, zeros, zeros, np.full(grid.N, MANNING, np.float32)
        )
        solver.deterministic = deterministic
        return solver

    return _make


def _bits(arrays: list[NDArray[np.float32]]) -> list[NDArray[np.uint32]]:
    return [np.ascontiguousarray(a).view(np.uint32) for a in arrays]


def _assert_bit_identical(a: Any, b: Any, snaps_a: list[Any], snaps_b: list[Any]) -> None:
    assert a.steps_total == b.steps_total
    for x, y in zip(_bits([*snaps_a, *state(a)]), _bits([*snaps_b, *state(b)]), strict=True):
        np.testing.assert_array_equal(x, y)


def _cases(grid: MeshGeometry) -> dict[str, tuple[NDArray[np.float32], Any, float]]:
    """name -> (initial depth, source rate, simulated seconds)."""
    dry = np.zeros(grid.N, dtype=np.float32)
    return {
        "wetting": (dry, point_source(grid, 1.0, NX * DX * 0.2, NY * DX / 2), 200.0),
        "dam_break": (dam(grid), None, 60.0),
        "draining": (sheet(grid), None, 300.0),
    }


CASES = ["wetting", "dam_break", "draining"]


class TestRepeatable:
    @pytest.mark.parametrize("case", CASES)
    def test_repeat_runs_are_bit_identical(
        self, case: str, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        h0, src, t_end = _cases(grid)[case]
        a, b = make_solver(h0), make_solver(h0)
        snaps_a, snaps_b = run(a, t_end, source_rate=src), run(b, t_end, source_rate=src)
        _assert_bit_identical(a, b, snaps_a, snaps_b)

    def test_switching_back_reproduces_the_first_run(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        # The switch is read when a run is recorded: one solver can run the
        # atomic path in between and still reproduce its first result.
        h0 = dam(grid)
        solver = make_solver(h0)
        first = run(solver, 40.0)
        first_state = state(solver)
        solver.deterministic = False
        atomic = run(solver, 40.0)
        solver.deterministic = True
        again = run(solver, 40.0)
        for x, y in zip(
            _bits([*first, *first_state]), _bits([*again, *state(solver)]), strict=True
        ):
            np.testing.assert_array_equal(x, y)
        for x, y in zip(first, atomic, strict=True):
            np.testing.assert_allclose(x, y, rtol=0, atol=ATOL_H)


class TestMatchesAtomics:
    @pytest.mark.parametrize("case", CASES)
    def test_close_to_the_atomic_path(
        self, case: str, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        h0, src, t_end = _cases(grid)[case]
        det, atomic = make_solver(h0), make_solver(h0, deterministic=False)
        snaps_d = run(det, t_end, source_rate=src)
        snaps_a = run(atomic, t_end, source_rate=src)
        assert det.steps_total == atomic.steps_total
        for x, y in zip([*snaps_d, *state(det)], [*snaps_a, *state(atomic)], strict=True):
            np.testing.assert_allclose(x, y, rtol=0, atol=ATOL_H)


class TestEdgeBuffer:
    @pytest.mark.parametrize("case", CASES)
    def test_values_not_written_this_stage_are_never_read(
        self, case: str, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        # Fill the edge buffer with huge values. A dry or skipped edge keeps
        # them, so reading one would wreck the state or the CFL bound.
        h0, src, t_end = _cases(grid)[case]
        clean, poisoned = make_solver(h0), make_solver(h0)
        poisoned._upload(poisoned.t_edge_flux, np.full(6 * poisoned.E, 1e30, dtype=np.float32))
        snaps_c = run(clean, t_end, source_rate=src)
        snaps_p = run(poisoned, t_end, source_rate=src)
        _assert_bit_identical(clean, poisoned, snaps_c, snaps_p)

    def test_stale_predictor_state_does_not_admit_skipped_edges(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        # A skipped block keeps stale predictor depths. The gather must judge
        # its edges by the start-of-step depth, as the flux kernel does: wet
        # garbage in h1 must not admit an edge holding garbage flux.
        h0 = dam(grid)
        clean, poisoned = make_solver(h0), make_solver(h0)
        wet_garbage = np.full(grid.N, 0.5, dtype=np.float32)
        for tensor in (poisoned.t_h1, poisoned.t_hu1, poisoned.t_hv1):
            poisoned._upload(tensor, wet_garbage)
        poisoned._upload(poisoned.t_edge_flux, np.full(6 * poisoned.E, 1e30, dtype=np.float32))
        snaps_c, snaps_p = run(clean, 30.0), run(poisoned, 30.0)
        _assert_bit_identical(clean, poisoned, snaps_c, snaps_p)

    def test_marks_are_clear_after_a_run(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        import kp

        solver = make_solver(dam(grid))
        run(solver, 20.0)
        solver._mgr.sequence().record(kp.OpTensorSyncLocal([solver.t_touched])).eval()
        marks = np.asarray(solver.t_touched.data()).view(np.uint32)
        assert int(np.count_nonzero(marks)) == 0

    def test_live_cells_are_marked_by_the_flux_kernel(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        # Without the marks the update would sum nothing and the water would
        # never move: the dam must spread.
        h0 = dam(grid)
        solver = make_solver(h0)
        run(solver, 20.0)
        h = solver.download_h()
        assert int(np.count_nonzero(h > 1e-3)) > int(np.count_nonzero(h0 > 0.0))
