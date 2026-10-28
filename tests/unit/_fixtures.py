"""Shared test constants and CPU-only solver stubs."""

from __future__ import annotations

import functools
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import NoReturn

    import numpy as np
    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry

TEST_MESH_SMALL_FOOTPRINT = 100
TEST_MESH_LARGE_FOOTPRINT = 10_000
TEST_GPU_FREE_BYTES = 1000

# The published EA benchmark inputs needed by the loader tests are tracked in
# benchmark_assets/. Tests still skip when a partial source distribution omits it.
EA_DATASET_ROOT = Path(__file__).resolve().parents[2] / "benchmark_assets"

requires_ea_dataset = pytest.mark.skipif(
    not EA_DATASET_ROOT.is_dir(),
    reason=f"EA benchmark dataset not present at {EA_DATASET_ROOT}; unpack it to run this test",
)


@functools.cache
def vulkan_available() -> bool:
    """True when kp can create a Vulkan device and glslang can compile shaders.

    Probed in a subprocess: without a portability-enabled driver
    ``kp.Manager()`` can hang or abort rather than raise.
    """
    if shutil.which("glslangValidator") is None:
        return False
    try:
        probe = subprocess.run(  # noqa: S603
            [sys.executable, "-c", "import kp; kp.Manager()"],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return probe.returncode == 0


requires_vulkan = pytest.mark.skipif(
    not vulkan_available(),
    reason="no Vulkan device (kp.Manager() failed) or no glslang compiler",
)

_SOLVER_CLASS_BY_MODULE = {
    "inundation.vulkan.async_sync_window": "SWESolverAsyncSyncWindow",
    "inundation.vulkan.baseline": "SWESolverBaseline",
    "inundation.vulkan.batched_submit": "SWESolverBatchedSubmit",
    "inundation.vulkan.device_cfl": "SWESolverDeviceCfl",
    "inundation.vulkan.fixed_dt_batch": "SWESolverFixedDtBatch",
    "inundation.vulkan.fixed_dt_batch_barrier": "SWESolverFixedDtBatchBarrier",
    "inundation.vulkan.gpu_resident_batch": "SWESolverGpuResidentBatch",
}


class CapturingSolver:
    """CPU-only stand-in for every Vulkan backend, used by ``prepare()`` tests.

    Keeps the geometry and roughness array ``prepare()`` handed to the solver so
    tests can assert on them without a GPU or a Vulkan driver.
    """

    def __init__(
        self,
        geom: MeshGeometry,
        _spv: dict[str, object],
        _h0: NDArray[np.float32],
        _hu0: NDArray[np.float32],
        _hv0: NDArray[np.float32],
        n0: NDArray[np.float32],
        **kwargs: object,
    ) -> None:
        self.geom = geom
        self.n0 = n0.copy()
        # The real backends take these as explicit keywords; recording them lets
        # tests pin prepare()'s solver wiring without a GPU.
        self.kwargs = kwargs


def _no_shaders() -> dict[str, object]:
    """Stand-in for ``_compile_solver_shaders`` that skips GLSL compilation."""
    return {}


def stub_solver_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Vulkan backends and shader compilation with CPU-only stubs.

    ``SWEWorkflow.prepare()`` then runs on any machine; the installed stub is
    ``CapturingSolver``.
    """
    from inundation import workflow as workflow_module

    for module_name, class_name in _SOLVER_CLASS_BY_MODULE.items():
        module = types.ModuleType(module_name)
        setattr(module, class_name, CapturingSolver)
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(workflow_module, "_compile_solver_shaders", _no_shaders)


def forbid_mesh_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any mesh-file load inside the workflow fail the test."""
    from inundation import workflow as workflow_module

    def _fail(_path: str) -> NoReturn:
        pytest.fail("this code path must not read a mesh file")

    monkeypatch.setattr(workflow_module, "load_mesh_file", _fail)
