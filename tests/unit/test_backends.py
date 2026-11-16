# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
"""Backend selection: the default is the validated fast path."""

from __future__ import annotations

from typing import Any

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
@pytest.mark.parametrize(
    ("open_edges", "kwargs", "match"),
    [((), {"capture_momentum_snapshots": True}, "momentum"), (None, {}, "open-boundary")],
    ids=["momentum", "open"],
)
def test_barrier_backend_refuses_what_it_does_not_record(
    open_edges: Any, kwargs: dict[str, Any], match: str
) -> None:
    from inundation.vulkan.fixed_dt_batch_barrier import SWESolverFixedDtBatchBarrier
    from inundation.workflow import _compile_solver_shaders

    grid = valley_grid()
    zeros = np.zeros(grid.N, dtype=np.float32)
    if open_edges is None:
        open_edges = [np.flatnonzero(grid.edge_cellR == -1)[:1]]
    solver = SWESolverFixedDtBatchBarrier(
        grid,
        _compile_solver_shaders(),
        dam(grid),
        zeros,
        zeros,
        np.full(grid.N, MANNING, np.float32),
        open_edges=open_edges,
    )
    with pytest.raises(ValueError, match=match):
        run(solver, 1.0, **kwargs)


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
