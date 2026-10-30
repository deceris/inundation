"""HLLC wave-speed bounds: dry-bed estimates must not switch on abruptly.

A jump in the speeds at ``dry_tol`` lets float32 rounding flip a whole flux, and
a front then lands in a measurably different place from run to run.
"""

from __future__ import annotations

import numpy as np
import pytest

from ._cpu_step import hllc_speeds

G = 9.81
TOL = 1e-3
EPS = 1e-9


def speeds(h_l: float, u_l: float, h_r: float, u_r: float) -> tuple[float, float]:
    s_l, s_r = hllc_speeds(
        np.array([h_l]), np.array([u_l]), np.array([h_r]), np.array([u_r]), g=G, dry_tol=TOL
    )
    return float(s_l[0]), float(s_r[0])


@pytest.mark.parametrize("edge", [TOL, 2.0 * TOL])
@pytest.mark.parametrize("wet", [(0.05, 0.3), (0.5, -0.4), (0.01, 0.0)])
def test_speeds_are_continuous_as_a_side_dries(edge: float, wet: tuple[float, float]) -> None:
    h, u = wet
    below = speeds(h, u, edge * (1.0 - EPS), 0.0)
    above = speeds(h, u, edge * (1.0 + EPS), 0.0)
    assert below == pytest.approx(above, abs=1e-5)
    mirrored_below = speeds(edge * (1.0 - EPS), 0.0, h, u)
    mirrored_above = speeds(edge * (1.0 + EPS), 0.0, h, u)
    assert mirrored_below == pytest.approx(mirrored_above, abs=1e-5)


def test_a_dry_side_gets_the_dry_bed_speed() -> None:
    h, u = 0.5, 0.3
    c = float(np.sqrt(G * h))
    assert speeds(h, u, 0.0, 0.0) == pytest.approx((u - c, u + 2.0 * c))
    assert speeds(0.0, 0.0, h, -u) == pytest.approx((-u - 2.0 * c, -u + c))


def test_two_wet_sides_keep_the_roe_estimate() -> None:
    h_l, h_r = 0.5, 0.2
    s_l, s_r = speeds(h_l, 0.0, h_r, 0.0)
    c_roe = float(np.sqrt(G * 0.5 * (h_l + h_r)))
    assert s_l == pytest.approx(min(-float(np.sqrt(G * h_l)), -c_roe))
    assert s_r == pytest.approx(max(float(np.sqrt(G * h_r)), c_roe))
