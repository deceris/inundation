"""Per-edge bed steps: the face elevation minus each side's bed.

The kernels reconstruct the face depth as ``h - dz`` rather than
``h + zb - z_face``. With a float32 bed hundreds of metres above the datum the
second form loses the low bits of the surface elevation; the steps are computed
in float64 so they carry the exact difference of the two float32 beds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from inundation.mesh.geometry import MeshGeometry, edge_bed_steps

from ._valley import valley_grid

if TYPE_CHECKING:
    from numpy.typing import NDArray

DATUM = 300.0


def _expected(geom: MeshGeometry) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    zb = geom.zb.astype(np.float64)
    wall = geom.edge_cellR < 0
    zbL = zb[geom.edge_cellL]
    zbR = np.where(wall, zbL, zb[np.where(wall, 0, geom.edge_cellR)])
    z_face = np.maximum(zbL, zbR)
    return z_face - zbL, z_face - zbR


def test_steps_are_the_exact_float64_bed_difference() -> None:
    geom = valley_grid(datum=DATUM)
    dzL, dzR = _expected(geom)
    # Nearby float32 beds differ by a float32-representable amount, so the
    # float64 difference survives the cast unchanged.
    np.testing.assert_array_equal(geom.edge_dzL, dzL.astype(np.float32))
    np.testing.assert_array_equal(geom.edge_dzR, dzR.astype(np.float32))
    np.testing.assert_array_equal(geom.edge_dzL.astype(np.float64), dzL)
    np.testing.assert_array_equal(geom.edge_dzR.astype(np.float64), dzR)


def test_one_side_sits_on_the_face() -> None:
    geom = valley_grid(datum=DATUM)
    assert (geom.edge_dzL >= 0.0).all()
    assert (geom.edge_dzR >= 0.0).all()
    assert (np.minimum(geom.edge_dzL, geom.edge_dzR) == 0.0).all()


def test_boundary_edges_have_no_step() -> None:
    geom = valley_grid(datum=DATUM)
    wall = geom.edge_cellR < 0
    assert wall.any()
    assert (geom.edge_dzL[wall] == 0.0).all()
    assert (geom.edge_dzR[wall] == 0.0).all()


def test_steps_follow_the_edges_through_reordering() -> None:
    # valley_grid() already Hilbert-reorders; the steps must still describe the
    # reordered cells' beds rather than the build-order ones.
    geom = valley_grid(datum=DATUM)
    dzL, dzR = _expected(geom)
    np.testing.assert_array_equal(geom.edge_dzL.astype(np.float64), dzL)
    np.testing.assert_array_equal(geom.edge_dzR.astype(np.float64), dzR)


def test_helper_handles_a_bed_far_from_the_datum() -> None:
    zb = np.array([1000.0, 1000.0 + 2.0**-13, 999.5], dtype=np.float32)
    cell_l = np.array([0, 1, 2], dtype=np.int32)
    cell_r = np.array([1, 2, -1], dtype=np.int32)
    dzL, dzR = edge_bed_steps(zb, cell_l, cell_r)
    np.testing.assert_array_equal(dzL, np.array([2.0**-13, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(dzR, np.array([0.0, 0.5 + 2.0**-13, 0.0], dtype=np.float32))
