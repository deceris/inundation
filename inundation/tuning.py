"""Cross-module numerical + structural tuning constants and small helpers.

Pure-data module: imports only stdlib at runtime. Shared constants prevent
drift between geometry, workflow, Vulkan solver, and benchmarks.
"""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# ── Geometry (swe_geometry.py) ────────────────────────────────────────────────
POLYGON_AREA_EPSILON = 1e-30
POLYGON_CENTROID_EPSILON = 1e-20
EDGE_LENGTH_EPSILON = 1e-30
FACE_NDIM = 2

# ── Mesh geometry ─────────────────────────────────────────────────────────────
MIN_POLYGON_VERTICES = 3

# ── Physical and numerical solver defaults ────────────────────────────────────
GRAVITY_G = 9.81
DRY_TOL_DEFAULT = 1e-4
# dt_i = CFL * A_i / sum_j(max|S_j| * len_j) over the cell's edges, with the
# wave speeds the flux assumed. In this form the HLL positivity limit is 1.0;
# the five gate suites pass at 0.45, 0.6, 0.8 and 1.0 with flat accuracy
# metrics (docs/implementation/24-solver-accuracy-fixes/results.md). 0.8 keeps
# a margin below the limit on terrain the sweep did not cover.
CFL_DEFAULT = 0.8
# Backends that reuse one dt for cfl_interval steps: the bound goes stale between
# recomputes, so they keep the margin they were validated with.
CFL_HOST_DRIVEN = 0.45
# Below this many dry_tol the velocity is desingularised and film momentum decays
# at a rate, so a cell barely above dry_tol cannot reach an absurd speed.
FILM_DEPTHS = 4.0
REGIME_DRY = 2

# ── Vulkan solver ─────────────────────────────────────────────────────────────
CFL_EPSILON_MIN = 1e-10
CFL_SANITY_MAX = 1e10
SIMULATION_TIME_EPSILON = 1e-12

# GPU-resident simulated clock: a uint64 count of 2^-20 s ticks. Every step's dt
# is a whole number of ticks no larger than 2^24, so float(ticks) * 2^-20 is
# exact in float32 and the clock advances by exactly the dt the physics used.
SIM_TICKS_PER_SECOND = 1 << 20
SIM_DT_MAX_TICKS = 1 << 24

# Active-set block: this many consecutive (Hilbert-ordered) cells, one cell
# workgroup. The resident kernels hard-code it as local_size_x and ``>> 8``.
ACTIVE_BLOCK_CELLS = 256

# Backends whose run() returns hu and hv snapshots. The others accept the flag
# for interface compatibility but return depth only, so the workflow refuses it.
MOMENTUM_SNAPSHOT_BACKENDS = frozenset({"baseline", "gpu_resident_batch"})

# Open boundary condition of an edge; the ``OPEN_*`` constants in the shaders.
OPEN_WALL = 0
OPEN_FREE = 1
OPEN_NORMAL = 2
OPEN_STAGE = 3
OPEN_INFLOW = 4
OPEN_EXCHANGE = 5

# Two-way coupling (coupling.py); see docs/implementation/30-two-way-coupling/plan.md.
# A border edge's midpoint lies on its segment, so this only absorbs rounding.
BORDER_TOLERANCE_M = 1e-3
# Weight of a basin's latest level in what its neighbour sees over the next window. At
# 1.0 the cut discharge on Cielnica swings from -3 to 42 m^3/s around a steady 20.
BORDER_LEVEL_WEIGHT = 0.5


# ── Workflow (workflow.py) ────────────────────────────────────────────────────
MIN_NDIM_NPY_INPUT = 2
NPY_COLS_HUV = 3
NPY_COLS_WITH_MANNING = 4
REQUIRED_CLI_ARGS = 2

# ── Benchmark validation ──────────────────────────────────────────────────────
MIN_VALID_CELLS = 2
MIN_SNAPSHOTS = 10
FINAL_TIME_TOLERANCE_S = 5.0
MIN_DEPTH_THRESHOLD_M = 0.05
WET_CELL_THRESHOLD = 1e-6
MIN_WET_CELLS = 100
MIN_VOLUME_RATIO = 0.01
MIN_REPEATS_FOR_DETERMINISM = 2


# Cells below this area are rejected by build_geometry. The CFL kernels skip
# cells at or below it (``area[i] > 1e-10`` in the GLSL) so a sliver cannot
# pin dt to zero; a wet sliver that slipped through would instead divide by
# its area and blow the state up.
MIN_CELL_AREA = 1e-10


# ── Helpers ───────────────────────────────────────────────────────────────────


def uint_as_float_bits(n: int) -> float:
    """A 32-bit count as the float whose bit pattern it is.

    Kompute takes push constants as a list of floats. Passing a count as a
    float value is exact only up to 2^24 (16.7 M edges or cells); above that
    it rounds and the kernels silently skip the tail. The shaders declare
    counts as ``uint`` and read these bits directly.
    """
    if not 0 <= n < 2**32:
        raise ValueError(f"count must fit in 32 bits, got {n}")
    return float(struct.unpack("<f", struct.pack("<I", n))[0])


def compute_workgroups(
    n: int, e: int, work_group_size: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return (edge_workgroup, cell_workgroup) for Vulkan dispatch."""
    return (
        (math.ceil(e / work_group_size), 1, 1),
        (math.ceil(n / work_group_size), 1, 1),
    )


def seconds_to_ticks(seconds: float) -> int:
    """Simulated time in seconds -> nearest whole tick."""
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"simulated time must be finite and non-negative, got {seconds}")
    return round(seconds * SIM_TICKS_PER_SECOND)


def ticks_to_seconds(ticks: int) -> float:
    """Whole ticks -> simulated time in seconds (float64)."""
    return ticks / SIM_TICKS_PER_SECOND


def dt_to_ticks(dt: float) -> int:
    """Timestep in seconds -> whole ticks, rounded down so a bound is never exceeded."""
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    ticks = math.floor(dt * SIM_TICKS_PER_SECOND)
    if ticks < 1 or ticks > SIM_DT_MAX_TICKS:
        raise ValueError(
            f"dt must be between one tick and {SIM_DT_MAX_TICKS / SIM_TICKS_PER_SECOND} s, got {dt}"
        )
    return ticks


def output_window_stops(start_ticks: int, end_ticks: int, interval_ticks: int) -> Iterator[int]:
    """Stop tick of each output window: every interval after the start, then the end."""
    if interval_ticks <= 0:
        raise ValueError(f"output interval must be positive, got {interval_ticks} ticks")
    stop = start_ticks
    while stop < end_ticks:
        stop = min(stop + interval_ticks, end_ticks)
        yield stop


def compute_eta_seconds(wall_elapsed: float, t_sim: float, t_offset: float, t_end: float) -> float:
    """ETA at current progress; returns inf when t_sim hasn't advanced past t_offset."""
    if t_sim - t_offset <= SIMULATION_TIME_EPSILON:
        return float("inf")
    return (wall_elapsed / max(t_sim - t_offset, SIMULATION_TIME_EPSILON)) * max(t_end - t_sim, 0.0)
