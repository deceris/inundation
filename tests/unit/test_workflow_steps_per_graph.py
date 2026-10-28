"""``WorkflowConfig.steps_per_graph`` and ``deterministic`` reach the GPU-resident backend only."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal, cast

import pytest

from inundation.mesh.cache import build_geometry_cache
from inundation.workflow import SWEWorkflow, WorkflowConfig

from ._fixtures import CapturingSolver, stub_solver_backends

if TYPE_CHECKING:
    from pathlib import Path

_QUAD_OBJ = """\
v 0 0 1
v 1 0 2
v 2 0 3
v 0 1 4
v 1 1 5
v 2 1 6
f 1 2 5 4
f 2 3 6 5
"""

RESIDENT_DEFAULT = 200


class ResidentStub(CapturingSolver):
    """Stand-in that, like the real resident backend, exposes the attributes."""

    steps_per_graph: int = RESIDENT_DEFAULT
    deterministic: bool = True


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    solver_impl: Literal["gpu_resident_batch", "fixed_dt_batch_barrier"],
    steps_per_graph: int | None,
    deterministic: bool | None = None,
) -> object:
    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text(_QUAD_OBJ, encoding="ascii")
    artifact = build_geometry_cache(mesh_path, tmp_path / "mesh.geometry.npz")
    stub_solver_backends(monkeypatch)
    monkeypatch.setattr(
        sys.modules["inundation.vulkan.gpu_resident_batch"],
        "SWESolverGpuResidentBatch",
        ResidentStub,
    )
    config = WorkflowConfig(
        mesh_source=None,
        geometry_cache_source=artifact,
        manning_n=0.05,
        output_interval_s=1.0,
        dt_max=0.1,
        cfl_interval=1,
        progress=False,
        solver_impl=solver_impl,
        steps_per_graph=steps_per_graph,
        deterministic=deterministic,
    )
    workflow = SWEWorkflow(config)
    workflow.prepare()
    return workflow.solver


def test_configured_value_reaches_the_resident_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    solver = cast("ResidentStub", _prepare(tmp_path, monkeypatch, "gpu_resident_batch", 50))
    assert solver.steps_per_graph == 50


def test_unset_keeps_the_backend_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    solver = cast("ResidentStub", _prepare(tmp_path, monkeypatch, "gpu_resident_batch", None))
    assert solver.steps_per_graph == RESIDENT_DEFAULT


def test_backends_without_graphs_are_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    solver = _prepare(tmp_path, monkeypatch, "fixed_dt_batch_barrier", 50)
    assert not hasattr(solver, "steps_per_graph")


@pytest.mark.parametrize("deterministic", [False, True])
def test_deterministic_reaches_the_resident_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deterministic: bool
) -> None:
    solver = cast(
        "ResidentStub",
        _prepare(tmp_path, monkeypatch, "gpu_resident_batch", None, deterministic=deterministic),
    )
    assert solver.deterministic is deterministic


def test_deterministic_unset_keeps_the_backend_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    solver = cast("ResidentStub", _prepare(tmp_path, monkeypatch, "gpu_resident_batch", None))
    assert solver.deterministic is True


def test_deterministic_leaves_other_backends_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    solver = _prepare(tmp_path, monkeypatch, "fixed_dt_batch_barrier", None, deterministic=False)
    assert not hasattr(solver, "deterministic")


def test_config_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="steps_per_graph"):
        WorkflowConfig(
            mesh_source="unused.obj",
            manning_n=0.05,
            output_interval_s=1.0,
            dt_max=0.1,
            cfl_interval=1,
            steps_per_graph=0,
        )
