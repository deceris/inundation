# pyright: reportPrivateUsage=false
"""Repeated wetting and drying against Thacker's exact solution (needs Vulkan).

Momentum flowing into a cell barely above dry_tol used to reach 26 000 m/s and
stall the run on a CFL bound below one tick. The exact solution never exceeds
about 0.7 m/s, so film speeds have to stay of that order.
"""

from __future__ import annotations

import pytest

from inundation.bench.parabolic_bowl import run_case

from ._fixtures import requires_vulkan

pytestmark = requires_vulkan

# Measured at nx 100-400: 0.55-0.72 m/s. The exact peak is 0.70 m/s.
FILM_SPEED_MAX = 1.0
DEPTH_L1_AT_200 = 0.18


@pytest.fixture(scope="module")
def spv() -> dict[str, bytes]:
    from inundation.workflow import _compile_solver_shaders

    return _compile_solver_shaders()


@pytest.mark.parametrize(
    ("nx", "cfl"), [(150, 0.8), (150, 1.0), (200, 0.8), (200, 1.0), (250, 1.0)]
)
def test_thin_films_stay_stable_through_three_periods(
    nx: int, cfl: float, spv: dict[str, bytes]
) -> None:
    metrics = run_case(spv, nx, cfl, 3)
    assert metrics.film_speed_max_ms < FILM_SPEED_MAX
    assert metrics.clamp_rel == 0.0


def test_error_against_the_exact_solution(spv: dict[str, bytes]) -> None:
    assert run_case(spv, 200, 0.8, 3).depth_l1_rel_mean < DEPTH_L1_AT_200
