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


def select_boundary_segments(
    geom: MeshGeometry, segments: NDArray[np.float64], tolerance_m: float
) -> NDArray[np.int64]:
    """The boundary edge at each world-coordinate segment, in order; one is missing: raises.

    Midpoints fall on a grid twice the tolerance, so a match is in one of nine squares.
    ``tolerance_m`` must be well below half the shortest boundary edge.
    """
    edges, midpoints = boundary_edge_midpoints(geom)
    wanted = 0.5 * (segments[:, :2] + segments[:, 2:]) - geom.origin
    size = 2.0 * tolerance_m
    corner = np.minimum(midpoints.min(axis=0), wanted.min(axis=0)) - 2.0 * size
    width = int(np.ceil((max(midpoints[:, 0].max(), wanted[:, 0].max()) - corner[0]) / size)) + 3

    def square(points: NDArray[np.float64]) -> NDArray[np.int64]:
        cell = np.floor((points - corner) / size).astype(np.int64)
        return cell[:, 0] * width + cell[:, 1]

    squares = square(midpoints)
    order = np.argsort(squares)
    squares = squares[order]
    found = np.full(wanted.shape[0], -1, dtype=np.int64)
    base = square(wanted)
    for shift in (dx * width + dy for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
        at = np.searchsorted(squares, base + shift)
        inside = at < squares.size
        hit = np.zeros(wanted.shape[0], dtype=bool)
        hit[inside] = squares[at[inside]] == (base + shift)[inside]
        candidate = order[at[hit]]
        near = np.hypot(*(midpoints[candidate] - wanted[hit]).T) <= tolerance_m
        found[np.flatnonzero(hit)[near]] = edges[candidate[near]]
    if (found < 0).any():
        raise ValueError(
            f"{int((found < 0).sum())} of {found.size} segments match no boundary edge "
            f"within {tolerance_m} m"
        )
    return found


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
