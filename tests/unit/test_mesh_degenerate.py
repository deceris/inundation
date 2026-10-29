"""build_geometry refuses cells the solver cannot step.

The CFL kernels skip cells with area at or below ``MIN_CELL_AREA`` so a sliver
cannot pin dt to zero. A wet sliver that got through would then divide the
flux sum by its area and turn the state to NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

from inundation.mesh.geometry import build_geometry
from inundation.tuning import MIN_CELL_AREA


def _two_triangles_and(extra_vertex: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], extra_vertex],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5]], dtype=np.int32)
    return verts, faces


def test_zero_area_cell_is_rejected() -> None:
    verts, faces = _two_triangles_and((1.5, 0.0))  # collinear
    with pytest.raises(ValueError, match="area"):
        build_geometry(verts, faces, zb_from_file=np.zeros(3, dtype=np.float32))


def test_sliver_below_the_cfl_floor_is_rejected() -> None:
    verts, faces = _two_triangles_and((1.5, 1e-12))
    with pytest.raises(ValueError, match="area"):
        build_geometry(verts, faces, zb_from_file=np.zeros(3, dtype=np.float32))


def test_small_but_steppable_cell_is_accepted() -> None:
    verts, faces = _two_triangles_and((1.5, 1e-6))
    geom = build_geometry(verts, faces, zb_from_file=np.zeros(3, dtype=np.float32))
    assert geom.N == 3
    assert float(geom.area.min()) > MIN_CELL_AREA
