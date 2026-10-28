"""GLSL compute shader compilation helpers for the SWE solver."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from .cfl import (
    CFL_ACCUM_GLSL,
    CFL_REDUCE_GLSL,
    CFL_RESOLVE_GLSL,
    DT_RESET_GLSL,
)
from .flux import FLUX_DTBUF_GLSL, FLUX_GLSL
from .resident import (
    BLOCK_DILATE_GLSL,
    BLOCK_MARK_GLSL,
    CFL_EDGE_GATHER_GLSL,
    CFL_GATHER_GLSL,
    FLUX_RESIDENT_GATHER_GLSL,
    FLUX_RESIDENT_GLSL,
    SOURCE_RESIDENT_GLSL,
    STEP_CONTROL_GLSL,
    UPDATE_RESIDENT_GATHER_GLSL,
    UPDATE_RESIDENT_GLSL,
)
from .source import SOURCE_DTBUF_GLSL, SOURCE_GLSL
from .update import UPDATE_DTBUF_GLSL, UPDATE_GLSL


def compile_glsl(glsl_src: str, label: str) -> bytes:
    """Compile GLSL compute shader source to SPIR-V bytes."""
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        raise RuntimeError("glslangValidator not found. Install with: brew install glslang")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, f"{label}.comp")
        spv_path = os.path.join(tmpdir, f"{label}.spv")

        with open(src_path, "w") as fh:
            fh.write(glsl_src)

        result = subprocess.run(  # noqa: S603
            [glslang, "--target-env", "vulkan1.0", "-V", "-o", spv_path, src_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Compiler failed for {label}:\n{result.stdout}\n{result.stderr}")

        with open(spv_path, "rb") as fh:
            return fh.read()


def compile_all() -> dict[str, bytes]:
    """Compile all SWE compute shaders and return label->SPIR-V map."""
    return {
        "flux": compile_glsl(FLUX_GLSL, "flux_kernel"),
        "flux_dtbuf": compile_glsl(FLUX_DTBUF_GLSL, "flux_dtbuf_kernel"),
        "update": compile_glsl(UPDATE_GLSL, "update_kernel"),
        "update_dtbuf": compile_glsl(UPDATE_DTBUF_GLSL, "update_dtbuf_kernel"),
        "cfl_accum": compile_glsl(CFL_ACCUM_GLSL, "cfl_accum_kernel"),
        "cfl_reduce": compile_glsl(CFL_REDUCE_GLSL, "cfl_reduce_kernel"),
        "cfl_resolve": compile_glsl(CFL_RESOLVE_GLSL, "cfl_resolve_kernel"),
        "dt_reset": compile_glsl(DT_RESET_GLSL, "dt_reset_kernel"),
        "source_dtbuf": compile_glsl(SOURCE_DTBUF_GLSL, "source_dtbuf_kernel"),
        "flux_resident": compile_glsl(FLUX_RESIDENT_GLSL, "flux_resident_kernel"),
        "update_resident": compile_glsl(UPDATE_RESIDENT_GLSL, "update_resident_kernel"),
        "source_resident": compile_glsl(SOURCE_RESIDENT_GLSL, "source_resident_kernel"),
        "step_control": compile_glsl(STEP_CONTROL_GLSL, "step_control_kernel"),
        "block_mark": compile_glsl(BLOCK_MARK_GLSL, "block_mark_kernel"),
        "block_dilate": compile_glsl(BLOCK_DILATE_GLSL, "block_dilate_kernel"),
        "flux_resident_gather": compile_glsl(
            FLUX_RESIDENT_GATHER_GLSL, "flux_resident_gather_kernel"
        ),
        "update_resident_gather": compile_glsl(
            UPDATE_RESIDENT_GATHER_GLSL, "update_resident_gather_kernel"
        ),
        "cfl_edge_gather": compile_glsl(CFL_EDGE_GATHER_GLSL, "cfl_edge_gather_kernel"),
        "cfl_gather": compile_glsl(CFL_GATHER_GLSL, "cfl_gather_kernel"),
    }


__all__ = [
    "BLOCK_DILATE_GLSL",
    "BLOCK_MARK_GLSL",
    "CFL_ACCUM_GLSL",
    "CFL_EDGE_GATHER_GLSL",
    "CFL_GATHER_GLSL",
    "CFL_REDUCE_GLSL",
    "CFL_RESOLVE_GLSL",
    "DT_RESET_GLSL",
    "FLUX_DTBUF_GLSL",
    "FLUX_GLSL",
    "FLUX_RESIDENT_GATHER_GLSL",
    "FLUX_RESIDENT_GLSL",
    "SOURCE_DTBUF_GLSL",
    "SOURCE_GLSL",
    "SOURCE_RESIDENT_GLSL",
    "STEP_CONTROL_GLSL",
    "UPDATE_DTBUF_GLSL",
    "UPDATE_GLSL",
    "UPDATE_RESIDENT_GATHER_GLSL",
    "UPDATE_RESIDENT_GLSL",
    "compile_all",
    "compile_glsl",
]
