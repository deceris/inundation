"""Static geometry for linear reconstruction: where faces sit, and how the bed slopes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .geometry import MeshGeometry


def edge_midpoint_offsets(geom: MeshGeometry) -> NDArray[np.float64]:
    """``(E, 4)`` from each side's centroid to the edge midpoint: rLx, rLy, rRx, rRy.

    A boundary edge has no right side, so its right offset is zero.
    """
    cells = geom.edge_cellL
    sides = geom.edge_sideL.astype(np.int64)
    first = geom.cell_vertex_ptr[cells].astype(np.int64)
    degree = geom.cell_vertex_ptr[cells + 1].astype(np.int64) - first
    a = geom.cell_vertices[first + sides].astype(np.float64)
    b = geom.cell_vertices[first + (sides + 1) % degree].astype(np.float64)
    middle = 0.5 * (a + b)
    centroid = geom.centroid.astype(np.float64)
    offsets = np.zeros((geom.E, 4))
    offsets[:, :2] = middle - centroid[cells]
    interior = geom.edge_cellR >= 0
    offsets[interior, 2:] = middle[interior] - centroid[geom.edge_cellR[interior]]
    return offsets


def cell_bed_gradient(geom: MeshGeometry) -> NDArray[np.float64]:
    """``(N, 2)`` Green-Gauss gradient of the bed; a boundary edge counts as level."""
    interior = np.flatnonzero(geom.edge_cellR >= 0)
    left, right = geom.edge_cellL[interior], geom.edge_cellR[interior]
    bed = geom.zb.astype(np.float64)
    flux = 0.5 * (bed[right] - bed[left]) * geom.edge_len[interior].astype(np.float64)
    gradient = np.zeros((geom.N, 2))
    for axis, normal in enumerate((geom.edge_nx, geom.edge_ny)):
        along = flux * normal[interior].astype(np.float64)
        np.add.at(gradient[:, axis], left, along)
        np.add.at(gradient[:, axis], right, along)
    return gradient / geom.area.astype(np.float64)[:, np.newaxis]
