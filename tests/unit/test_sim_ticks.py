"""Integer simulated-time ticks used by the GPU-resident clock.

The device clock counts 2^-20 s ticks in a uint64 and each step's dt is a whole
number of ticks, so the clock advances by exactly the dt the physics used.
These helpers are the host side of that contract: they must convert phase and
output times without drift and must only hand the device dt values whose
float32 form is exact.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from inundation.tuning import (
    SIM_DT_MAX_TICKS,
    SIM_TICKS_PER_SECOND,
    dt_to_ticks,
    output_window_stops,
    seconds_to_ticks,
    ticks_to_seconds,
)

DAY_S = 86_400.0


class TestSecondsToTicks:
    @pytest.mark.parametrize("seconds", [0.0, 0.5, 1.0, 600.0, 1800.0, 7200.0, DAY_S])
    def test_phase_and_output_times_are_exact(self, seconds: float) -> None:
        ticks = seconds_to_ticks(seconds)
        # All of these are binary fractions, so the product is exact.
        assert ticks == int(seconds * SIM_TICKS_PER_SECOND)
        assert ticks_to_seconds(ticks) == seconds

    def test_rounds_to_the_nearest_tick(self) -> None:
        ticks = seconds_to_ticks(0.1)
        assert ticks == round(0.1 * SIM_TICKS_PER_SECOND)
        assert abs(ticks_to_seconds(ticks) - 0.1) <= 0.5 / SIM_TICKS_PER_SECOND

    def test_summed_phases_land_on_the_total(self) -> None:
        # The workflow chains phases by adding durations; in ticks that sum is
        # exact, so a 24 h schedule of 12 x 600 s + 79 200 s ends on 86 400 s.
        durations = [600.0] * 12 + [79_200.0]
        assert sum(seconds_to_ticks(d) for d in durations) == seconds_to_ticks(DAY_S)

    @pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, -math.inf])
    def test_rejects_negative_and_non_finite(self, bad: float) -> None:
        with pytest.raises(ValueError, match="simulated time"):
            seconds_to_ticks(bad)

    def test_a_year_fits_in_uint64(self) -> None:
        assert seconds_to_ticks(365 * DAY_S) < 2**64


class TestDtToTicks:
    def test_rounds_down_so_the_cfl_bound_is_never_exceeded(self) -> None:
        dt = 0.0314
        ticks = dt_to_ticks(dt)
        assert ticks_to_seconds(ticks) <= dt
        assert dt - ticks_to_seconds(ticks) < 1.0 / SIM_TICKS_PER_SECOND

    def test_upper_limit_is_sixteen_seconds(self) -> None:
        assert dt_to_ticks(16.0) == SIM_DT_MAX_TICKS
        with pytest.raises(ValueError, match="dt"):
            dt_to_ticks(16.5)

    @pytest.mark.parametrize("bad", [0.0, -0.1, math.nan, math.inf])
    def test_rejects_non_positive_and_non_finite(self, bad: float) -> None:
        with pytest.raises(ValueError, match="dt"):
            dt_to_ticks(bad)

    def test_rejects_dt_below_one_tick(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            dt_to_ticks(0.5 / SIM_TICKS_PER_SECOND)

    def test_every_allowed_tick_count_is_exact_in_float32(self) -> None:
        # The shader computes dt = float(ticks) * 2^-20 in float32. Exhaustive
        # over [1, 2^24]: the integer and the scaling must both be exact.
        ticks = np.arange(1, SIM_DT_MAX_TICKS + 1, dtype=np.int64)
        as_f32 = ticks.astype(np.float32) * np.float32(1.0 / SIM_TICKS_PER_SECOND)
        assert np.array_equal(as_f32.astype(np.float64), ticks / SIM_TICKS_PER_SECOND)


class TestOutputWindowStops:
    def test_outputs_every_interval_then_the_end(self) -> None:
        stops = list(
            output_window_stops(0, 3600 * SIM_TICKS_PER_SECOND, 1800 * SIM_TICKS_PER_SECOND)
        )
        assert stops == [1800 * SIM_TICKS_PER_SECOND, 3600 * SIM_TICKS_PER_SECOND]

    def test_last_window_is_capped_at_the_end(self) -> None:
        stops = list(output_window_stops(0, 1000, 600))
        assert stops == [600, 1000]

    def test_interval_longer_than_the_run_gives_one_window(self) -> None:
        assert list(output_window_stops(0, 500, 600)) == [500]

    def test_windows_are_relative_to_the_run_start(self) -> None:
        # A resumed phase starting at 7200 s outputs 1800 s after its own start,
        # matching the float loop it replaces.
        start = seconds_to_ticks(7200.0)
        stops = list(
            output_window_stops(start, start + seconds_to_ticks(3600.0), seconds_to_ticks(1800.0))
        )
        assert stops == [start + seconds_to_ticks(1800.0), start + seconds_to_ticks(3600.0)]

    def test_empty_when_already_at_the_end(self) -> None:
        assert list(output_window_stops(1000, 1000, 600)) == []

    def test_rejects_non_positive_interval(self) -> None:
        with pytest.raises(ValueError, match="interval"):
            list(output_window_stops(0, 1000, 0))
