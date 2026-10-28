"""Push-constant helpers must emit fields in the shader's declared struct order.

Vulkan push constants are a flat byte block: the Python helper's argument order
is irrelevant, only the order values are written in. Nothing in the type system
connects the two, so a helper can silently disagree with its GLSL struct and the
shader will read every field shifted.

That shipped. ``_pc_update`` emitted ``[nc, dt, stage, g, dry_tol, cfl]`` while
the shader declares ``{num_cells, dt, g, dry_tol, cfl_number, stage}``, so the
kernel read ``stage`` as ``g``, ``g`` as ``dry_tol`` (zeroing momentum in every
cell shallower than 9.81 m) and ``cfl`` as ``stage`` — which, being 0.45 < 0.5,
pinned the kernel in its predictor branch so the committed state was never
written at all. Every backend that routed through the helper was frozen at its
initial condition; the one backend that passed inline literals was fine. The
symptom was misattributed to MoltenVK for two efforts.

These tests parse the field order out of the GLSL source and check each helper
against it, so the shader stays the single source of truth.
"""

from __future__ import annotations

import re

import pytest

from inundation.vulkan.gpu_resident_batch import (
    StepControlMode,
    _pc_block_dilate,
    _pc_block_mark,
    _pc_cfl_edge_gather,
    _pc_cfl_gather,
    _pc_source_resident,
    _pc_step_control,
)

# pyright: reportPrivateUsage=false
from inundation.vulkan.shaders import (
    BLOCK_DILATE_GLSL,
    BLOCK_MARK_GLSL,
    CFL_ACCUM_GLSL,
    CFL_EDGE_GATHER_GLSL,
    CFL_GATHER_GLSL,
    CFL_REDUCE_GLSL,
    FLUX_DTBUF_GLSL,
    FLUX_GLSL,
    FLUX_RESIDENT_GATHER_GLSL,
    FLUX_RESIDENT_GLSL,
    SOURCE_RESIDENT_GLSL,
    STEP_CONTROL_GLSL,
    UPDATE_DTBUF_GLSL,
    UPDATE_GLSL,
    UPDATE_RESIDENT_GATHER_GLSL,
    UPDATE_RESIDENT_GLSL,
)
from inundation.vulkan.solver import (
    _pc_cfl_accum,
    _pc_cfl_reduce,
    _pc_flux,
    _pc_update,
)

# Distinct sentinel per semantic field, so a swap cannot coincidentally pass.
NUM_ELEMENTS = 11.0
DT = 22.0
G = 33.0
DRY_TOL = 44.0
CFL = 55.0
STAGE = 1.0  # must stay a valid stage: the shader branches on it
TRACK_CLAMP = 1.0

# GLSL field name -> the value the helper is expected to place there. The two
# count fields and the unused-dt alias are spelled differently across shaders
# but carry the same argument.
FIELD_VALUES = {
    "num_edges": NUM_ELEMENTS,
    "num_cells": NUM_ELEMENTS,
    "dt": DT,
    "dt_unused": DT,
    "g": G,
    "dry_tol": DRY_TOL,
    "cfl_number": CFL,
    "stage": STAGE,
    "track_clamp": TRACK_CLAMP,
}


def push_constant_fields(glsl_src: str) -> list[str]:
    """Field names of the ``push_constant`` block, in declaration order."""
    block = re.search(r"layout\(push_constant\)\s+uniform\s+\w+\s*\{(.*?)\}", glsl_src, re.DOTALL)
    if block is None:
        raise AssertionError("no push_constant block found")
    return re.findall(r"\bfloat\s+(\w+)\s*;", block.group(1))


def expected_layout(glsl_src: str) -> list[float]:
    return [FIELD_VALUES[name] for name in push_constant_fields(glsl_src)]


class TestPushConstantFieldParsing:
    def test_reads_declaration_order_not_alphabetical(self) -> None:
        assert push_constant_fields(UPDATE_GLSL) == [
            "num_cells",
            "dt",
            "g",
            "dry_tol",
            "cfl_number",
            "stage",
            "track_clamp",
        ]

    def test_rejects_source_without_a_push_constant_block(self) -> None:
        with pytest.raises(AssertionError, match="no push_constant block"):
            push_constant_fields("#version 450\nvoid main() {}")


class TestUpdatePushConstants:
    """The helper whose ordering bug froze four backends."""

    @pytest.mark.parametrize(
        "glsl", [UPDATE_GLSL, UPDATE_DTBUF_GLSL, UPDATE_RESIDENT_GLSL, UPDATE_RESIDENT_GATHER_GLSL]
    )
    def test_matches_the_shader_struct(self, glsl: str) -> None:
        emitted = _pc_update(int(NUM_ELEMENTS), DT, int(STAGE), G, DRY_TOL, CFL, track_clamp=True)
        assert emitted == expected_layout(glsl)

    def test_clamp_tracking_defaults_off(self) -> None:
        fields = push_constant_fields(UPDATE_GLSL)
        emitted = _pc_update(int(NUM_ELEMENTS), DT, 1, G, DRY_TOL, CFL)
        assert emitted[fields.index("track_clamp")] == 0.0

    def test_stage_lands_in_the_slot_the_shader_branches_on(self) -> None:
        # The regression proper: with stage in the wrong slot the shader read
        # cfl (0.45) as stage, took `stage < 0.5` and never left the predictor.
        fields = push_constant_fields(UPDATE_GLSL)
        emitted = _pc_update(int(NUM_ELEMENTS), DT, 1, G, DRY_TOL, 0.45)
        assert emitted[fields.index("stage")] == 1.0

    def test_both_stages_are_distinguishable(self) -> None:
        fields = push_constant_fields(UPDATE_GLSL)
        slot = fields.index("stage")
        predictor = _pc_update(int(NUM_ELEMENTS), DT, 0, G, DRY_TOL, CFL)
        corrector = _pc_update(int(NUM_ELEMENTS), DT, 1, G, DRY_TOL, CFL)
        assert predictor[slot] < 0.5
        assert corrector[slot] >= 0.5

    def test_dry_tol_is_not_shadowed_by_gravity(self) -> None:
        # The other half of the same bug: g landing in dry_tol zeroed momentum
        # in every cell shallower than 9.81 m.
        fields = push_constant_fields(UPDATE_GLSL)
        emitted = _pc_update(int(NUM_ELEMENTS), DT, 1, G, DRY_TOL, CFL)
        assert emitted[fields.index("dry_tol")] == DRY_TOL
        assert emitted[fields.index("g")] == G


class TestFluxPushConstants:
    @pytest.mark.parametrize(
        "glsl", [FLUX_GLSL, FLUX_DTBUF_GLSL, FLUX_RESIDENT_GLSL, FLUX_RESIDENT_GATHER_GLSL]
    )
    def test_matches_the_shader_struct(self, glsl: str) -> None:
        emitted = _pc_flux(int(NUM_ELEMENTS), DT, G, DRY_TOL, CFL, int(STAGE))
        assert emitted == expected_layout(glsl)


class TestCflPushConstants:
    def test_accum_matches_the_shader_struct(self) -> None:
        emitted = _pc_cfl_accum(int(NUM_ELEMENTS), G, DRY_TOL, CFL)
        expected = expected_layout(CFL_ACCUM_GLSL)
        # These kernels do not use dt; the helper pads the slot with zero.
        expected[push_constant_fields(CFL_ACCUM_GLSL).index("dt")] = 0.0
        assert emitted == expected

    def test_reduce_matches_the_shader_struct(self) -> None:
        emitted = _pc_cfl_reduce(int(NUM_ELEMENTS), G, DRY_TOL, CFL)
        expected = expected_layout(CFL_REDUCE_GLSL)
        expected[push_constant_fields(CFL_REDUCE_GLSL).index("dt")] = 0.0
        assert emitted == expected


class TestResidentPushConstants:
    @pytest.mark.parametrize("mode", list(StepControlMode))
    def test_step_control_mode_is_the_only_field(self, mode: StepControlMode) -> None:
        assert push_constant_fields(STEP_CONTROL_GLSL) == ["mode"]
        assert _pc_step_control(mode) == [float(mode.value)]

    def test_step_control_modes_match_the_shader_thresholds(self) -> None:
        # The shader branches on mode < 0.5 (start), < 1.5 (after a step), else
        # (new window); the enum values must sit inside those bands.
        assert StepControlMode.START.value < 0.5
        assert 0.5 < StepControlMode.STEP.value < 1.5
        assert StepControlMode.RETARGET.value > 1.5

    def test_source_matches_the_shader_struct(self) -> None:
        assert push_constant_fields(SOURCE_RESIDENT_GLSL) == ["num_sources", "dry_tol"]
        assert _pc_source_resident(int(NUM_ELEMENTS), DRY_TOL) == [NUM_ELEMENTS, DRY_TOL]

    @pytest.mark.parametrize("all_active", [False, True])
    def test_block_mark_matches_the_shader_struct(self, all_active: bool) -> None:
        assert push_constant_fields(BLOCK_MARK_GLSL) == [
            "num_cells",
            "num_blocks",
            "dry_tol",
            "all_active",
        ]
        n_blocks = 77
        assert _pc_block_mark(int(NUM_ELEMENTS), n_blocks, DRY_TOL, all_active) == [
            NUM_ELEMENTS,
            float(n_blocks),
            DRY_TOL,
            1.0 if all_active else 0.0,
        ]

    @pytest.mark.parametrize("force", [False, True])
    def test_block_dilate_matches_the_shader_struct(self, force: bool) -> None:
        assert push_constant_fields(BLOCK_DILATE_GLSL) == ["num_blocks", "force"]
        # The shader treats force > 0.5 as set.
        assert _pc_block_dilate(int(NUM_ELEMENTS), force) == [NUM_ELEMENTS, 1.0 if force else 0.0]

    def test_cfl_edge_gather_matches_the_shader_struct(self) -> None:
        assert _pc_cfl_edge_gather(int(NUM_ELEMENTS), G, DRY_TOL) == expected_layout(
            CFL_EDGE_GATHER_GLSL
        )

    def test_cfl_gather_matches_the_shader_struct(self) -> None:
        assert _pc_cfl_gather(int(NUM_ELEMENTS), CFL) == expected_layout(CFL_GATHER_GLSL)
