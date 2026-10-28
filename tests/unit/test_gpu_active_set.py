# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
"""GPU-resident active set: skipping dry blocks must not change results (needs Vulkan).

Each case runs the same problem with ``active_set`` on and off. Off marks every
block active, so it is the full computation; on skips blocks outside the
two-ring of wet blocks. Every case runs on both accumulation paths: with the
deterministic gather, on and off must agree exactly; with float atomics the
summation order varies from run to run, so states agree to float32 noise. Step
counts must match exactly on both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ._fixtures import requires_vulkan
from ._valley import DRY_TOL, DX, MANNING, NX, NY, dam, point_source, run, sheet, state, valley_grid

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


@pytest.fixture(params=[True, False], ids=["gather", "atomic"])
def make_solver(
    request: pytest.FixtureRequest, grid: MeshGeometry, spv: dict[str, bytes]
) -> Callable[..., Any]:
    from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch

    deterministic = bool(request.param)

    def _make(h0: NDArray[np.float32], *, active_set: bool, **kwargs: Any) -> Any:
        zeros = np.zeros(grid.N, dtype=np.float32)
        solver = SWESolverGpuResidentBatch(
            grid, spv, h0, zeros, zeros, np.full(grid.N, MANNING, np.float32), **kwargs
        )
        solver.active_set = active_set
        solver.deterministic = deterministic
        return solver

    return _make


def _assert_same(on: Any, off: Any, snaps_on: list[Any], snaps_off: list[Any]) -> None:
    assert on.steps_total == off.steps_total
    assert len(snaps_on) == len(snaps_off)
    pairs = [*zip(snaps_on, snaps_off, strict=True), *zip(state(on), state(off), strict=True)]
    for a, b in pairs:
        if on.deterministic and off.deterministic:
            # Exact. Not bitwise: a skipped dry cell keeps a -0.0 momentum from
            # its initial state, which the full step rewrites as +0.0.
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=0, atol=ATOL_H)


def _active_fraction(solver: Any) -> float:
    return float(solver.download_block_flags()[:, 1].mean())


class TestSameResults:
    def test_wetting_front_from_a_source_on_a_dry_bed(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        src = point_source(grid, 1.0, NX * DX * 0.2, NY * DX / 2)
        h0 = np.zeros(grid.N, dtype=np.float32)
        on, off = make_solver(h0, active_set=True), make_solver(h0, active_set=False)
        snaps_on = run(on, 300.0, source_rate=src)
        snaps_off = run(off, 300.0, source_rate=src)
        _assert_same(on, off, snaps_on, snaps_off)
        # Skipping really happened: most of the mesh is still dry.
        assert _active_fraction(on) < 0.6
        assert _active_fraction(off) == 1.0

    def test_dam_break_onto_a_dry_slope(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        h0 = dam(grid)
        on, off = make_solver(h0, active_set=True), make_solver(h0, active_set=False)
        snaps_on, snaps_off = run(on, 60.0), run(off, 60.0)
        _assert_same(on, off, snaps_on, snaps_off)

    def test_draining_leaves_dry_cells_behind(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        # Upstream cells dry and keep half their momentum for a while.
        h0 = sheet(grid)
        on, off = make_solver(h0, active_set=True), make_solver(h0, active_set=False)
        snaps_on, snaps_off = run(on, 400.0), run(off, 400.0)
        _assert_same(on, off, snaps_on, snaps_off)

    def test_chained_runs_with_changing_sources(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        h0 = np.zeros(grid.N, dtype=np.float32)
        on, off = make_solver(h0, active_set=True), make_solver(h0, active_set=False)
        near = point_source(grid, 0.5, NX * DX * 0.2, NY * DX / 2)
        far = point_source(grid, 0.3, NX * DX * 0.9, NY * DX * 0.8) + near
        phases = [(100.0, near), (200.0, far), (260.0, np.zeros(grid.N, dtype=np.float32))]
        t = 0.0
        for solver in (on, off):
            t = 0.0
            for t_end, src in phases:
                solver.run(
                    t_end=t_end,
                    output_interval_s=50.0,
                    dt_max=1.0,
                    dt_init=0.01,
                    cfl_interval=10,
                    progress=False,
                    source_rate=src,
                    resume=t > 0.0,
                    t_start=t,
                )
                t = t_end
        _assert_same(on, off, [], [])


class TestSkippedBlocks:
    def test_stale_predictor_state_in_skipped_blocks_is_ignored(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        # Skipped blocks keep whatever predictor state they last had; a cell
        # that just dried can leave a wet one behind. Garbage everywhere is the
        # worst case: the full step overwrites it, the skipping step must not
        # read it.
        h0 = dam(grid)
        on, off = make_solver(h0, active_set=True), make_solver(h0, active_set=False)
        garbage = np.full(grid.N, 0.5, dtype=np.float32)
        for solver in (on, off):
            for tensor in (solver.t_h1, solver.t_hu1, solver.t_hv1):
                solver._upload(tensor, garbage)
        snaps_on, snaps_off = run(on, 30.0), run(off, 30.0)
        _assert_same(on, off, snaps_on, snaps_off)

    def test_dry_cells_keep_decaying_their_momentum(
        self, grid: MeshGeometry, spv: dict[str, bytes]
    ) -> None:
        # Dry cells with momentum, far from the water: the full step halves
        # their momentum every step, so their blocks must stay active until it
        # reaches zero rather than freeze it after the first step.
        from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch

        h0 = dam(grid)
        far = grid.centroid[:, 0] > NX * DX * 0.8
        hu0 = np.where(far, 0.1, 0.0).astype(np.float32)
        solvers: list[Any] = []
        for active_set in (True, False):
            solver = SWESolverGpuResidentBatch(
                grid, spv, h0, hu0, -hu0, np.full(grid.N, MANNING, np.float32)
            )
            solver.active_set = active_set
            run(solver, 2.0)
            solvers.append(solver)
        on, off = solvers
        assert on.steps_total > 3
        _assert_same(on, off, [], [])
        assert float(np.abs(on.download_hu()[far]).max()) < 0.1 / 2**3


class TestFlags:
    def test_every_block_with_water_is_active_after_arun(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        from inundation.tuning import ACTIVE_BLOCK_CELLS

        solver = make_solver(dam(grid), active_set=True)
        run(solver, 30.0)
        h, hu, hv = state(solver)
        live = (h >= DRY_TOL) | (hu != 0.0) | (hv != 0.0)
        live_blocks = np.unique(np.flatnonzero(live) // ACTIVE_BLOCK_CELLS)
        flags = solver.download_block_flags()
        assert flags[live_blocks, 1].all()
        assert flags[:, 1].sum() < flags.shape[0]

    def test_blocks_go_inactive_once_momentum_has_decayed(
        self, grid: MeshGeometry, spv: dict[str, bytes]
    ) -> None:
        # No water anywhere, momentum in one region: the full step halves it
        # until it underflows to zero, after which nothing is left to compute
        # and every mark must have been cleared.
        from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch

        zeros = np.zeros(grid.N, dtype=np.float32)
        hu0 = np.where(grid.centroid[:, 0] > NX * DX * 0.8, 0.1, 0.0).astype(np.float32)
        solver = SWESolverGpuResidentBatch(
            grid, spv, zeros, hu0, zeros, np.full(grid.N, MANNING, np.float32)
        )
        run(solver, 3.0)
        assert solver.steps_total > 200
        assert float(np.abs(solver.download_hu()).max()) == 0.0
        assert _active_fraction(solver) == 0.0

    def test_all_dry_mesh_has_no_active_block(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        solver = make_solver(np.zeros(grid.N, dtype=np.float32), active_set=True)
        run(solver, 5.0)
        assert _active_fraction(solver) == 0.0
        assert float(solver.download_h().max()) == 0.0


class TestSources:
    def test_source_in_a_dry_inactive_region_injects_its_volume(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        src = point_source(grid, 0.2, NX * DX * 0.9, NY * DX * 0.1)
        solver = make_solver(np.zeros(grid.N, dtype=np.float32), active_set=True)
        run(solver, 50.0, source_rate=src)
        volume = float((solver.download_h().astype(np.float64) * grid.area).sum())
        assert volume == pytest.approx(0.2 * 50.0, rel=1e-4)

    def test_all_zero_source_rate_is_a_plainrun(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        h0 = dam(grid)
        a, b = make_solver(h0, active_set=True), make_solver(h0, active_set=True)
        run(a, 20.0, source_rate=np.zeros(grid.N, dtype=np.float32))
        run(b, 20.0)
        _assert_same(a, b, [], [])


class TestRejects:
    def test_workgroup_size_other_than_the_block(
        self, grid: MeshGeometry, make_solver: Callable[..., Any]
    ) -> None:
        with pytest.raises(ValueError, match="workgroup_size"):
            make_solver(np.zeros(grid.N, dtype=np.float32), active_set=True, workgroup_size=128)

    def test_source_fn(self, grid: MeshGeometry, make_solver: Callable[..., Any]) -> None:
        solver = make_solver(np.zeros(grid.N, dtype=np.float32), active_set=True)
        with pytest.raises(ValueError, match="source_fn"):
            run(solver, 1.0, source_fn=lambda _t: np.zeros(grid.N, dtype=np.float32))  # pyright: ignore[reportUnknownLambdaType]
