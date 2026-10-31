"""``capture_momentum_snapshots`` fails loudly where a backend cannot honour it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from inundation.mesh.cache import build_geometry_cache
from inundation.tuning import MOMENTUM_SNAPSHOT_BACKENDS
from inundation.workflow import SimulationPhase, SWEWorkflow, WorkflowConfig

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

UNSUPPORTED = (
    "async_sync_window",
    "batched_submit",
    "device_cfl",
    "fixed_dt_batch",
    "fixed_dt_batch_barrier",
)


def _config(**overrides: Any) -> WorkflowConfig:
    return WorkflowConfig(
        mesh_source="unused.obj",
        manning_n=0.05,
        output_interval_s=1.0,
        dt_max=0.1,
        cfl_interval=1,
        **overrides,
    )


@pytest.mark.parametrize("solver_impl", UNSUPPORTED)
def test_requesting_momentum_on_an_unsupported_backend_names_it(solver_impl: str) -> None:
    with pytest.raises(ValueError, match=rf"capture_momentum_snapshots.*{solver_impl}"):
        _config(solver_impl=solver_impl, capture_momentum_snapshots=True)


@pytest.mark.parametrize("solver_impl", UNSUPPORTED)
def test_unsupported_backend_still_runs_without_the_flag(solver_impl: str) -> None:
    _config(solver_impl=solver_impl)


def test_default_backend_supports_momentum_snapshots() -> None:
    assert _config().solver_impl in MOMENTUM_SNAPSHOT_BACKENDS


def test_every_backend_is_classified() -> None:
    assert set(UNSUPPORTED).isdisjoint(MOMENTUM_SNAPSHOT_BACKENDS)
    assert set(UNSUPPORTED) | MOMENTUM_SNAPSHOT_BACKENDS == {
        "baseline",
        "gpu_resident_batch",
        *UNSUPPORTED,
    }


class _StubRunSolver(CapturingSolver):
    """Returns whatever the test put in ``result``, as ``run()`` would."""

    result: tuple[Any, ...] = ()

    def run(self, **_kwargs: object) -> tuple[Any, ...]:
        return self.result


def _prepared_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SWEWorkflow:
    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text(_QUAD_OBJ, encoding="ascii")
    artifact = build_geometry_cache(mesh_path, tmp_path / "mesh.geometry.npz")
    stub_solver_backends(monkeypatch)
    workflow = SWEWorkflow(
        WorkflowConfig(
            mesh_source=None,
            geometry_cache_source=artifact,
            manning_n=0.05,
            output_interval_s=1.0,
            dt_max=0.1,
            cfl_interval=1,
            progress=False,
            capture_momentum_snapshots=True,
        )
    )
    workflow.prepare()
    return workflow


def _phase() -> list[SimulationPhase]:
    return [SimulationPhase(1.0, [])]


def _stub(workflow: SWEWorkflow, result: tuple[Any, ...]) -> None:
    solver = cast("Any", workflow.solver)
    solver.__class__ = _StubRunSolver
    solver.result = result


def test_backend_returning_no_momentum_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = _prepared_workflow(tmp_path, monkeypatch)
    _stub(workflow, ([np.zeros(2, np.float32)], [0.0]))
    with pytest.raises(RuntimeError, match="gpu_resident_batch"):
        workflow.run(_phase())


def test_backend_returning_misaligned_momentum_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = _prepared_workflow(tmp_path, monkeypatch)
    depth = [np.zeros(2, np.float32), np.zeros(2, np.float32)]
    _stub(workflow, (depth, [0.0, 1.0], depth[:1], depth[:1]))
    with pytest.raises(RuntimeError, match="gpu_resident_batch"):
        workflow.run(_phase())
