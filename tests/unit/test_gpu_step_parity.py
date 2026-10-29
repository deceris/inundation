# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
"""GPU kernels against the float64 reference step (needs Vulkan).

A few fixed-dt Heun steps on the valley must match ``_cpu_step`` to float32
rounding on every backend. The reference states the intended physics, so a
case it covers goes red the moment a kernel disagrees with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from inundation.tuning import CFL_DEFAULT, GRAVITY_G

from ._cpu_step import State, cfl_dt, run_steps
from ._fixtures import requires_vulkan
from ._valley import DRY_TOL, DX, MANNING, NX, dam, sheet, valley_grid

if TYPE_CHECKING:
    from collections.abc import Callable

    from inundation.mesh.geometry import MeshGeometry

pytestmark = requires_vulkan

# A power of two: exact in ticks and in float32, so every backend takes the
# same steps.
DT = 2.0**-10
STEPS = 8
ATOL_H = 2e-6
ATOL_Q = 2e-5
BACKENDS = ["resident-gather", "resident-atomic", "barrier"]


@pytest.fixture(scope="module")
def spv() -> dict[str, bytes]:
    from inundation.workflow import _compile_solver_shaders

    return _compile_solver_shaders()


@pytest.fixture(scope="module")
def grid() -> MeshGeometry:
    return valley_grid()


def make_solver(backend: str, grid: MeshGeometry, spv: dict[str, bytes], s: State) -> Any:
    n0 = np.full(grid.N, MANNING, np.float32)
    h0, hu0, hv0 = (a.astype(np.float32) for a in (s.h, s.hu, s.hv))
    if backend == "barrier":
        from inundation.vulkan.fixed_dt_batch_barrier import SWESolverFixedDtBatchBarrier

        return SWESolverFixedDtBatchBarrier(grid, spv, h0, hu0, hv0, n0, dry_tol=DRY_TOL)
    from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch

    solver = SWESolverGpuResidentBatch(grid, spv, h0, hu0, hv0, n0, dry_tol=DRY_TOL)
    solver.deterministic = backend == "resident-gather"
    return solver


def step_gpu(solver: Any, steps: int) -> State:
    solver.run(
        t_end=steps * DT,
        output_interval_s=steps * DT,
        dt_max=DT,
        dt_init=DT,
        cfl_interval=steps,
        progress=False,
    )
    assert solver.steps_total == steps
    return State(
        solver.download_h().astype(np.float64),
        solver.download_hu().astype(np.float64),
        solver.download_hv().astype(np.float64),
    )


def wet_lake(grid: MeshGeometry) -> State:
    """Every cell wet, a gentle current: no front and no drying anywhere."""
    x = grid.centroid[:, 0].astype(np.float64)
    h = float(grid.zb.max()) + 0.5 - grid.zb.astype(np.float64)
    assert (h > 10 * DRY_TOL).all()
    hu = 0.1 * np.sin(2 * np.pi * x / (NX * DX)) * h
    return State(h, hu, np.zeros_like(h))


def dam_break(grid: MeshGeometry) -> State:
    h = dam(grid).astype(np.float64)
    return State(h, np.zeros_like(h), np.zeros_like(h))


def draining(grid: MeshGeometry) -> State:
    """A thin sheet running off the slope: cells behind it dry during the steps."""
    h = sheet(grid).astype(np.float64)
    hu = np.where(h > 0.0, 0.05, 0.0)
    return State(h, hu, np.zeros_like(h))


def dry_momentum(grid: MeshGeometry) -> State:
    """Dry cells that still carry momentum, as a cell that has just dried does."""
    h = dam(grid).astype(np.float64)
    far = grid.centroid[:, 0] > NX * DX * 0.8
    hu = np.where(far, 0.1, 0.0)
    return State(h, hu, -hu)


CASES: dict[str, Callable[[MeshGeometry], State]] = {
    "wet_lake": wet_lake,
    "dam_break": dam_break,
    "draining": draining,
    "dry_momentum": dry_momentum,
}


def assert_matches(
    got: State, want: State, *, atol_h: float = ATOL_H, atol_q: float = ATOL_Q
) -> None:
    for name, a, b, atol in (
        ("h", got.h, want.h, atol_h),
        ("hu", got.hu, want.hu, atol_q),
        ("hv", got.hv, want.hv, atol_q),
    ):
        worst = float(np.abs(a - b).max())
        assert worst <= atol, f"{name}: max |gpu - reference| = {worst:.3e} > {atol:.1e}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(CASES))
def test_steps_match_the_reference(
    case: str, backend: str, grid: MeshGeometry, spv: dict[str, bytes]
) -> None:
    s0 = CASES[case](grid)
    want = run_steps(
        grid,
        s0.h.astype(np.float32),
        s0.hu.astype(np.float32),
        s0.hv.astype(np.float32),
        np.full(grid.N, MANNING, np.float32),
        DT,
        STEPS,
        g=GRAVITY_G,
        dry_tol=DRY_TOL,
    )
    got = step_gpu(make_solver(backend, grid, spv, s0), STEPS)
    assert_matches(got, want)


@pytest.fixture(scope="module")
def high_grid() -> MeshGeometry:
    # Real terrain sits hundreds of metres above the datum, where float32 bed
    # elevations are spaced ~3e-5 m: a sizeable fraction of a thin film.
    return valley_grid(datum=300.0)


# Over 64 steps the datum-0 cases stay within these of the reference; the
# high-terrain cases must do no worse, so the bed datum cannot leak into the
# reconstructed depths.
LONG_STEPS = 64
LONG_ATOL_H = 5e-6
LONG_ATOL_Q = 6e-7


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", ["dam_break", "draining"])
def test_steps_on_high_terrain_match_the_reference(
    case: str, backend: str, high_grid: MeshGeometry, spv: dict[str, bytes]
) -> None:
    s0 = CASES[case](high_grid)
    want = run_steps(
        high_grid,
        s0.h.astype(np.float32),
        s0.hu.astype(np.float32),
        s0.hv.astype(np.float32),
        np.full(high_grid.N, MANNING, np.float32),
        DT,
        LONG_STEPS,
        g=GRAVITY_G,
        dry_tol=DRY_TOL,
    )
    got = step_gpu(make_solver(backend, high_grid, spv, s0), LONG_STEPS)
    assert_matches(got, want, atol_h=LONG_ATOL_H, atol_q=LONG_ATOL_Q)


@pytest.mark.parametrize("case", list(CASES))
def test_dry_cells_carry_no_momentum(case: str, grid: MeshGeometry, spv: dict[str, bytes]) -> None:
    got = step_gpu(make_solver("resident-gather", grid, spv, CASES[case](grid)), STEPS)
    dry = got.h < DRY_TOL
    assert float(np.abs(got.hu[dry]).max(initial=0.0)) == 0.0
    assert float(np.abs(got.hv[dry]).max(initial=0.0)) == 0.0


def prologue_bound(solver: Any) -> float:
    """The raw CFL bound the resident prologue plans the first step from."""
    solver._ctrl[...] = 0
    solver._sync_ctrl_to_device()
    solver._record_prologue().eval()
    solver._read_ctrl()
    return float(solver._ctrl["last_raw"][0])


@pytest.mark.parametrize("deterministic", [True, False], ids=["gather", "atomic"])
@pytest.mark.parametrize("case", list(CASES))
def test_cfl_bound_matches_the_reference(
    case: str, deterministic: bool, grid: MeshGeometry, spv: dict[str, bytes]
) -> None:
    s0 = CASES[case](grid)
    s32 = State(
        s0.h.astype(np.float32).astype(np.float64),
        s0.hu.astype(np.float32).astype(np.float64),
        s0.hv.astype(np.float32).astype(np.float64),
    )
    want = cfl_dt(grid, s32, g=GRAVITY_G, dry_tol=DRY_TOL, cfl=CFL_DEFAULT)
    solver = make_solver("resident-gather" if deterministic else "resident-atomic", grid, spv, s0)
    got = prologue_bound(solver)
    assert got == pytest.approx(want, rel=1e-4)
