# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
"""Backend selection: the default is the validated fast path, the racy one is closed."""

from __future__ import annotations

import numpy as np
import pytest

from inundation.tuning import CFL_DEFAULT, CFL_HOST_DRIVEN
from inundation.workflow import WorkflowConfig

from ._fixtures import requires_vulkan
from ._valley import MANNING, dam, run, valley_grid


def test_workflow_defaults_to_the_resident_backend() -> None:
    config = WorkflowConfig(
        mesh_source="mesh.gpkg", manning_n=0.03, output_interval_s=60.0, dt_max=1.0, cfl_interval=1
    )
    assert config.solver_impl == "gpu_resident_batch"


@requires_vulkan
def test_unbarriered_fixed_dt_batch_refuses_to_run() -> None:
    # It recorded dependent dispatches without compute barriers and raced
    # (docs/planning/decisions.md); the barrier subclass is the reference.
    from inundation.vulkan.fixed_dt_batch import SWESolverFixedDtBatch
    from inundation.workflow import _compile_solver_shaders

    grid = valley_grid()
    zeros = np.zeros(grid.N, dtype=np.float32)
    solver = SWESolverFixedDtBatch(
        grid,
        _compile_solver_shaders(),
        dam(grid),
        zeros,
        zeros,
        np.full(grid.N, MANNING, np.float32),
    )
    with pytest.raises(RuntimeError, match="barrier"):
        run(solver, 1.0)


@requires_vulkan
@pytest.mark.parametrize(
    "module",
    ["async_sync_window", "batched_submit", "device_cfl", "fixed_dt_batch_barrier"],
)
def test_host_backends_refuse_momentum_capture(module: str) -> None:
    import importlib

    from inundation.workflow import _compile_solver_shaders

    solvers = importlib.import_module(f"inundation.vulkan.{module}")
    cls = next(getattr(solvers, n) for n in solvers.__all__ if n.startswith("SWESolver"))
    grid = valley_grid()
    zeros = np.zeros(grid.N, dtype=np.float32)
    solver = cls(
        grid,
        _compile_solver_shaders(),
        dam(grid),
        zeros,
        zeros,
        np.full(grid.N, MANNING, np.float32),
    )
    with pytest.raises(ValueError, match="momentum"):
        run(solver, 1.0, capture_momentum_snapshots=True)


def test_host_driven_backends_keep_a_lower_cfl() -> None:
    # They reuse one dt for cfl_interval steps; the resident backend recomputes
    # the bound every step, so only it may run at CFL_DEFAULT.
    from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch
    from inundation.vulkan.solver import SWESolver

    assert SWESolver.default_cfl == CFL_HOST_DRIVEN
    assert SWESolverGpuResidentBatch.default_cfl == CFL_DEFAULT
    assert CFL_HOST_DRIVEN < CFL_DEFAULT


def test_workflow_leaves_the_cfl_to_the_backend() -> None:
    config = WorkflowConfig(
        mesh_source="mesh.gpkg", manning_n=0.03, output_interval_s=60.0, dt_max=1.0, cfl_interval=1
    )
    assert config.cfl is None
