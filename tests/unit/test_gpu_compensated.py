# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
"""GPU-resident compensated depth (needs Vulkan).

With ``compensated`` on, each cell's depth is an exact float32 pair and every
update goes through an error-free transform, so the only non-conservative
operation left is the positivity clamp. On a closed domain the volume change
must then equal the clamp mass plus the injected volume to float64 rounding,
where the plain float32 update drifts by a measurable amount.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ._fixtures import requires_vulkan
from ._valley import DRY_TOL, DX, MANNING, NX, NY, dam, point_source, run, sheet, valley_grid

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry

pytestmark = requires_vulkan

# Relative to the final volume, over a few thousand steps on the valley. The
# compensated path is left with the float32 rounding of the per-cell flux sums
# (measured 3e-10 to 7e-10); the plain path loses whole increments below half
# an ulp of the depth (measured 9e-8 to 5e-6, fewer steps at a higher Courant
# number mean less drift). With MoltenVK fast math on the compensated path
# reads 1e-7 to 5e-7: the solver turns fast math off.
REL_COMPENSATED = 3e-9
REL_PLAIN_AT_LEAST = 3e-8


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

    def _make(h0: NDArray[np.float32], *, compensated: bool, deterministic: bool = True) -> Any:
        zeros = np.zeros(grid.N, dtype=np.float32)
        solver = SWESolverGpuResidentBatch(
            grid, spv, h0, zeros, zeros, np.full(grid.N, MANNING, np.float32), dry_tol=DRY_TOL
        )
        solver.compensated = compensated
        solver.deterministic = deterministic
        solver.track_clamp = True
        return solver

    return _make


def _cases(grid: MeshGeometry) -> dict[str, tuple[NDArray[np.float32], Any, float]]:
    """name -> (initial depth, source rate, simulated seconds)."""
    dry = np.zeros(grid.N, dtype=np.float32)
    return {
        "dam_break": (dam(grid), None, 600.0),
        "draining": (sheet(grid), None, 600.0),
        "wetting": (dry, point_source(grid, 1.0, NX * DX * 0.2, NY * DX / 2), 600.0),
    }


CASES = ["dam_break", "draining", "wetting"]


def _volume_error(solver: Any, grid: MeshGeometry, h0: Any, src: Any, t_end: float) -> float:
    """(final - initial - injected - clamp) / final volume, all in float64."""
    area = grid.area.astype(np.float64)
    v0 = float((h0.astype(np.float64) * area).sum())
    v1 = float((solver.download_h_precise() * area).sum())
    clamp = float(solver.download_clamp_mass().astype(np.float64).sum())
    injected = 0.0 if src is None else float(src.astype(np.float64).sum()) * t_end
    return (v1 - v0 - injected - clamp) / v1


class TestConservation:
    @pytest.mark.parametrize("deterministic", [True, False], ids=["gather", "atomic"])
    @pytest.mark.parametrize("case", CASES)
    def test_volume_change_is_exactly_the_clamp_and_the_sources(
        self, case: str, deterministic: bool, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        h0, src, t_end = _cases(grid)[case]
        solver = make_solver(h0, compensated=True, deterministic=deterministic)
        run(solver, t_end, source_rate=src)
        assert solver.steps_total > 1000
        assert abs(_volume_error(solver, grid, h0, src, t_end)) < REL_COMPENSATED

    @pytest.mark.parametrize("case", ["dam_break", "wetting"])
    def test_the_plain_update_drifts(
        self, case: str, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        # The gate above means something only if the plain path fails it.
        h0, src, t_end = _cases(grid)[case]
        solver = make_solver(h0, compensated=False)
        run(solver, t_end, source_rate=src)
        assert abs(_volume_error(solver, grid, h0, src, t_end)) > REL_PLAIN_AT_LEAST


class TestPositivity:
    def test_point_source_into_a_dry_cell_creates_no_clamp_mass(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        # The next dt is planned before the source adds its water; a strong
        # point source into a thin cell must still not drive the cell below
        # zero (the source kernel bounds dt by the cell's post-source state).
        src = point_source(grid, 2.0, NX * DX * 0.2, NY * DX / 2)
        solver = make_solver(np.zeros(grid.N, dtype=np.float32), compensated=True)
        solver.run(
            t_end=20.0,
            output_interval_s=20.0,
            dt_max=1.0,
            dt_init=0.01,
            cfl_interval=1,
            progress=False,
            source_rate=src,
        )
        clamp = float(solver.download_clamp_mass().astype(np.float64).sum())
        assert clamp == 0.0


class TestPair:
    def test_high_word_is_the_rounded_depth(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        # Renormalised pair: |lo| <= ulp(h)/2, so h alone is the float32
        # rounding of the exact depth and the flux kernels see a valid depth.
        solver = make_solver(dam(grid), compensated=True)
        run(solver, 60.0)
        h = solver.download_h()
        lo = solver.download_h_residual()
        assert (np.abs(lo) <= np.spacing(np.abs(h)) / 2).all()
        assert int(np.count_nonzero(lo)) > 0
        np.testing.assert_array_equal((h.astype(np.float64) + lo).astype(np.float32), h)

    def test_dry_cells_have_no_residual(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        solver = make_solver(sheet(grid), compensated=True)
        run(solver, 300.0)
        h = solver.download_h()
        lo = solver.download_h_residual()
        assert (lo[h == 0.0] == 0.0).all()
        assert (h.astype(np.float64) + lo >= 0.0).all()

    def test_reset_clears_the_residual(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        solver = make_solver(dam(grid), compensated=True)
        run(solver, 20.0)
        assert int(np.count_nonzero(solver.download_h_residual())) > 0
        solver._reset_to_initial()
        assert int(np.count_nonzero(solver.download_h_residual())) == 0

    def test_plain_runs_leave_the_residual_at_zero(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        solver = make_solver(dam(grid), compensated=False)
        run(solver, 20.0, source_rate=point_source(grid, 1.0, NX * DX * 0.2, NY * DX / 2))
        assert int(np.count_nonzero(solver.download_h_residual())) == 0


class TestAgreement:
    @pytest.mark.parametrize("case", CASES)
    def test_close_to_the_plain_update(
        self, case: str, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        # Same physics: the two paths differ by float32 rounding of the depth,
        # which the dry threshold can amplify in a few cells.
        h0, src, t_end = _cases(grid)[case]
        comp = make_solver(h0, compensated=True)
        plain = make_solver(h0, compensated=False)
        snaps_c = run(comp, t_end / 4, source_rate=src)
        snaps_p = run(plain, t_end / 4, source_rate=src)
        for a, b in zip(snaps_c, snaps_p, strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-4)
