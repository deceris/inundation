# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false
"""The state beyond an open edge, kernel by kernel, against the formulas (needs Vulkan).

Runs ``open_ghost`` and ``boundary_mass_flux`` themselves on the device, because most
of what they get wrong barely shows in a steady depth.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from inundation.tuning import (
    GRAVITY_G,
    OPEN_FREE,
    OPEN_INFLOW,
    OPEN_NORMAL,
    OPEN_STAGE,
    OPEN_WALL,
)

# Importing the solver turns MoltenVK's fast math off, which has to happen before
# this file's first Vulkan instance or every later solver in the process fails.
from inundation.vulkan import solver as _solver  # noqa: F401  # pyright: ignore[reportUnusedImport]
from inundation.vulkan.shaders import compile_glsl
from inundation.vulkan.shaders.flux import HLLC_GLSL
from inundation.vulkan.shaders.resident import _OPEN_GHOST_GLSL

from ._fixtures import requires_vulkan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

pytestmark = requires_vulkan

DRY_TOL = 1e-4
CASE_WORDS = 6
RESULT_WORDS = 4

_PROBE_GLSL = (
    """\
#version 450

layout(local_size_x = 1) in;

layout(set=0, binding=0) buffer CASES  { float cases[];   };
layout(set=0, binding=1) buffer OPEN   { uint  open_u[];  };
layout(set=0, binding=2) buffer RESULT { float results[]; };

layout(push_constant) uniform PC {
    float g;
    float dry_tol;
} pc;

"""
    + HLLC_GLSL
    + _OPEN_GHOST_GLSL
    + """\
void main() {
    uint k = gl_GlobalInvocationID.x;
    int cR = -2 - int(k);
    uint o = 6u * k;
    float hG, huG, hvG;
    open_ghost(cR, cases[o], cases[o + 1u], cases[o + 2u], cases[o + 3u], cases[o + 4u],
               hG, huG, hvG);
    results[4u * k]      = hG;
    results[4u * k + 1u] = huG;
    results[4u * k + 2u] = hvG;
    results[4u * k + 3u] = boundary_mass_flux(cR, cases[o + 5u]);
}
"""
)


def _ghost(
    h: float,
    hu: float,
    hv: float,
    kind: int,
    value: float = 0.0,
    *,
    normal: tuple[float, float] = (1.0, 0.0),
    hllc_flux: float = 0.0,
) -> tuple[float, ...]:
    return _ghosts([(h, hu, hv, kind, value, normal, hllc_flux)])[0]


def _ghosts(
    cases: Sequence[tuple[float, float, float, int, float, tuple[float, float], float]],
) -> NDArray[np.float32]:
    import kp

    inputs = np.array(
        [[h, hu, hv, n[0], n[1], flux] for h, hu, hv, _, _, n, flux in cases], dtype=np.float32
    )
    words = np.zeros((len(cases), 2), dtype=np.uint32)
    words[:, 0] = [case[3] for case in cases]
    words[:, 1] = np.array([case[4] for case in cases], dtype=np.float32).view(np.uint32)
    manager = kp.Manager()
    tensors = [
        manager.tensor(inputs.ravel()),
        manager.tensor(words.ravel().view(np.float32).copy()),
        manager.tensor(np.zeros(RESULT_WORDS * len(cases), dtype=np.float32)),
    ]
    algorithm = manager.algorithm(
        tensors,
        compile_glsl(_PROBE_GLSL, "open_ghost_probe"),
        workgroup=(len(cases), 1, 1),
        push_consts=[GRAVITY_G, DRY_TOL],
    )
    sequence = manager.sequence()
    sequence.record(kp.OpTensorSyncDevice(tensors))
    sequence.record(kp.OpAlgoDispatch(algorithm))
    sequence.record(kp.OpTensorSyncLocal(tensors))
    sequence.eval()
    results: NDArray[np.float32] = np.array(tensors[2].data(), dtype=np.float32)
    return results.reshape(-1, RESULT_WORDS)


def _root(q: float, riemann: float) -> float:
    """``hG`` solving ``-q/hG + 2 sqrt(g hG) = riemann``, by bisection on sqrt(hG)."""
    root_g = math.sqrt(GRAVITY_G)
    low, high = 1e-9, 100.0
    for _ in range(200):
        s = 0.5 * (low + high)
        if 2.0 * root_g * s - q / (s * s) < riemann:
            low = s
        else:
            high = s
    return high * high


def test_free_outflow_copies_the_interior() -> None:
    assert _ghost(1.0, 0.5, 0.2, OPEN_FREE, hllc_flux=0.5) == pytest.approx(
        (1.0, 0.5, 0.2, 0.5), rel=1e-6
    )


def test_free_outflow_mirrors_flow_running_inward() -> None:
    assert _ghost(1.0, -0.5, 0.2, OPEN_FREE, hllc_flux=-0.5) == pytest.approx(
        (1.0, 0.5, 0.2, 0.0), rel=1e-6
    )


@pytest.mark.parametrize("kind", [OPEN_FREE, OPEN_NORMAL])
def test_outflow_conditions_never_take_a_negative_flux(kind: int) -> None:
    assert _ghost(1.0, 0.0, 0.0, kind, 1.0, hllc_flux=-0.25)[3] == 0.0


def test_normal_depth_uses_the_manning_velocity() -> None:
    value = math.sqrt(1e-3) / 0.03
    h = 0.8
    un = value * h ** (2.0 / 3.0)
    assert _ghost(h, 0.1, 0.05, OPEN_NORMAL, value) == pytest.approx(
        (h, h * un, 0.05, 0.0), rel=1e-5
    )


def test_a_stage_keeps_the_outgoing_characteristic() -> None:
    h, u, depth = 1.0, 0.5, 2.0
    un = u + 2.0 * (math.sqrt(GRAVITY_G * h) - math.sqrt(GRAVITY_G * depth))
    ghost = _ghost(h, h * u, 0.0, OPEN_STAGE, depth, hllc_flux=-0.7)
    assert ghost == pytest.approx((depth, depth * un, 0.0, -0.7), rel=1e-5)


def test_a_stage_cannot_reach_a_supercritical_outflow() -> None:
    h, u = 0.1, 3.0
    assert _ghost(h, h * u, 0.0, OPEN_STAGE, 2.0)[:3] == pytest.approx((h, h * u, 0.0), rel=1e-5)


def test_a_stage_over_a_dry_cell_enters_as_a_dam_break() -> None:
    depth = 0.5
    un = -2.0 * math.sqrt(GRAVITY_G * depth)
    assert _ghost(0.0, 0.0, 0.0, OPEN_STAGE, depth)[:2] == pytest.approx(
        (depth, depth * un), rel=1e-5
    )


def test_a_wall_kind_mirrors_and_passes_nothing() -> None:
    assert _ghost(1.0, 0.5, 0.2, OPEN_WALL, hllc_flux=0.3) == pytest.approx(
        (1.0, -0.5, 0.2, 0.0), rel=1e-6
    )


@pytest.mark.parametrize(
    ("h", "u", "q"),
    [(0.0, 0.0, 0.25), (0.05, 0.0, 2.0), (1.0, -0.2, 0.25), (3.0, 0.5, 5.0), (0.2, 0.0, 17.0)],
)
def test_inflow_solves_the_outgoing_characteristic(h: float, u: float, q: float) -> None:
    riemann = u + 2.0 * math.sqrt(GRAVITY_G * h)
    depth = _root(q, riemann)
    ghost = _ghost(h, h * u, 0.7 * h, OPEN_INFLOW, q, hllc_flux=9.0)
    assert ghost == pytest.approx((depth, -q, 0.0, -q), rel=1e-4, abs=1e-6)


def test_ghosts_follow_the_edge_normal() -> None:
    nx, ny = 0.6, 0.8
    h, q = 1.0, 0.5
    depth = _root(q, 2.0 * math.sqrt(GRAVITY_G * h))
    ghost = _ghost(h, 0.0, 0.0, OPEN_INFLOW, q, normal=(nx, ny))
    assert ghost[:3] == pytest.approx((depth, -q * nx, -q * ny), rel=1e-4)
