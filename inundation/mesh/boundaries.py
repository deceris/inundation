"""Open boundary stretches: the edges a line selects, and how inflow splits over them.

A stretch is a line in world coordinates. It selects the boundary edges whose
midpoints lie within a tolerance of that line.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from .geometry import MeshGeometry

# Bisection steps on the conveyance level: enough for a float64 level.
_LEVEL_BISECTIONS = 80


def boundary_edge_midpoints(
    geom: MeshGeometry,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Boundary edge indices and their midpoints, relative to ``geom.origin``."""
    edges = np.flatnonzero(geom.edge_cellR == -1)
    cells = geom.edge_cellL[edges]
    sides = geom.edge_sideL[edges].astype(np.int64)
    first = geom.cell_vertex_ptr[cells].astype(np.int64)
    degree = geom.cell_vertex_ptr[cells + 1].astype(np.int64) - first
    a = geom.cell_vertices[first + sides].astype(np.float64)
    b = geom.cell_vertices[first + (sides + 1) % degree].astype(np.float64)
    return edges, 0.5 * (a + b)


def distance_to_line(points: NDArray[np.float64], line: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance from each point to a polyline."""
    best = np.full(points.shape[0], np.inf)
    for a, b in pairwise(line):
        along = b - a
        t = ((points - a) @ along) / max(float(along @ along), 1e-30)
        nearest = a + np.clip(t, 0.0, 1.0)[:, np.newaxis] * along
        best = np.minimum(best, np.hypot(*(points - nearest).T))
    return best


def select_boundary_edges(
    geom: MeshGeometry, line: Sequence[tuple[float, float]], tolerance_m: float
) -> NDArray[np.int64]:
    """Boundary edges whose midpoints lie within ``tolerance_m`` of a world-coordinate line."""
    if len(line) < 2:
        raise ValueError(f"a stretch needs at least 2 points, got {len(line)}")
    local = np.asarray(line, dtype=np.float64) - geom.origin
    edges, midpoints = boundary_edge_midpoints(geom)
    return edges[distance_to_line(midpoints, local) <= tolerance_m]


def conveyance_split(
    discharge_m3s: float,
    lengths: NDArray[np.float32],
    beds: NDArray[np.float32],
    manning: NDArray[np.float32],
    energy_slope: float,
) -> NDArray[np.float32]:
    """Discharge per unit width on each edge of a section carrying ``discharge_m3s``.

    Finds the level at which the section's Manning conveyance carries the
    discharge, so water enters along the low part of the section. The result is
    float32, for the kernels, and the largest edge takes the cast's rounding.
    """
    if discharge_m3s <= 0.0:
        return np.zeros(lengths.size, dtype=np.float32)
    length = lengths.astype(np.float64)
    bed = beds.astype(np.float64)
    width_rate = length * np.sqrt(energy_slope) / manning.astype(np.float64)

    def conveyance(level: float) -> NDArray[np.float64]:
        return width_rate * np.maximum(level - bed, 0.0) ** (5.0 / 3.0)

    low = float(bed.min())
    high = low + 1.0
    while conveyance(high).sum() < discharge_m3s:
        high = low + 2.0 * (high - low)
    for _ in range(_LEVEL_BISECTIONS):
        middle = 0.5 * (low + high)
        if conveyance(middle).sum() < discharge_m3s:
            low = middle
        else:
            high = middle
    carried = conveyance(high)
    per_width = (discharge_m3s * carried / carried.sum() / length).astype(np.float32)
    largest = int(np.argmax(carried))
    rounded = float((per_width.astype(np.float64) * length).sum())
    per_width[largest] += np.float32((discharge_m3s - rounded) / length[largest])
    return per_width
