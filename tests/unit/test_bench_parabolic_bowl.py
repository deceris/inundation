"""The parabolic-bowl reference must solve the shallow-water equations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from inundation.bench.parabolic_bowl import (
    BOWL_RADIUS_M,
    CENTRE_DEPTH_M,
    SWING_M,
    bed,
    exact,
    omega,
    period_s,
)
from inundation.tuning import GRAVITY_G

D = 1e-4


def _points() -> tuple[np.ndarray, np.ndarray]:
    # Well inside the wet region at every phase of the swing.
    r = np.linspace(0.0, 0.2, 9)
    th = np.linspace(0.0, 2.0 * np.pi, 13)
    rr, tt = np.meshgrid(r, th)
    return (rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()


@pytest.mark.parametrize("t", [0.0, 0.37, 1.1, 2.9])
def test_continuity(t: float) -> None:
    x, y = _points()
    h, u, v = exact(x, y, t)
    h_t = (exact(x, y, t + D)[0] - exact(x, y, t - D)[0]) / (2 * D)
    qx = (exact(x + D, y, t)[0] * u - exact(x - D, y, t)[0] * u) / (2 * D)
    qy = (exact(x, y + D, t)[0] * v - exact(x, y - D, t)[0] * v) / (2 * D)
    assert (h > 0).all()
    np.testing.assert_allclose(h_t + qx + qy, 0.0, atol=1e-6)


@pytest.mark.parametrize("t", [0.0, 0.37, 1.1, 2.9])
def test_momentum(t: float) -> None:
    x, y = _points()
    u_t = (exact(x, y, t + D)[1] - exact(x, y, t - D)[1]) / (2 * D)
    v_t = (exact(x, y, t + D)[2] - exact(x, y, t - D)[2]) / (2 * D)

    def eta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return exact(a, b, t)[0] + bed(a, b)

    np.testing.assert_allclose(
        u_t + GRAVITY_G * (eta(x + D, y) - eta(x - D, y)) / (2 * D), 0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        v_t + GRAVITY_G * (eta(x, y + D) - eta(x, y - D)) / (2 * D), 0.0, atol=1e-6
    )


def test_the_solution_repeats_every_period() -> None:
    x, y = _points()
    np.testing.assert_allclose(exact(x, y, period_s())[0], exact(x, y, 0.0)[0], atol=1e-12)


def test_the_shoreline_moves_across_cells_that_wet_and_dry() -> None:
    # A point that is dry half the time is what makes this a wetting test.
    x, y = np.array([BOWL_RADIUS_M + SWING_M * 0.6]), np.array([0.0])
    depths = [float(exact(x, y, t)[0][0]) for t in np.linspace(0.0, period_s(), 41)]
    assert max(depths) > 0.0
    assert min(depths) == 0.0


def test_frequency_matches_the_bowl() -> None:
    assert omega() == pytest.approx(math.sqrt(2.0 * GRAVITY_G * CENTRE_DEPTH_M) / BOWL_RADIUS_M)
