# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Mesh geometry does not depend on where the mesh sits on the map."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import shapely

from inundation.mesh.cache import (
    _GEOMETRY_ARRAY_FIELDS,
    _GEOMETRY_SCALAR_FIELDS,
    build_geometry_cache,
    load_geometry_cache,
    load_geometry_cache_artifact,
    save_geometry_cache,
)
from inundation.mesh.geometry import build_geometry, hilbert_reorder
from inundation.mesh.loader import load_mesh_file
from inundation.workflow import PointSource, SimulationPhase, _build_phase_source_rate

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry

# Float32 spacing here is 25 cm in x and 50 cm in y, coarser than the test cells.
UTM = (512_345.5, 5_512_345.25)


def _rotated_grid(
    nx: int, ny: int, side: float, *, angle: float = 0.4
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Square cells of side ``side``, rotated so nothing lands on a float32 grid."""
    x, y = np.meshgrid(np.arange(nx + 1.0), np.arange(ny + 1.0))
    c, s = np.cos(angle), np.sin(angle)
    verts = side * np.column_stack([c * x.ravel() - s * y.ravel(), s * x.ravel() + c * y.ravel()])
    a = (np.arange(ny)[:, None] * (nx + 1) + np.arange(nx)).ravel()
    faces = np.column_stack([a, a + 1, a + nx + 2, a + nx + 1]).astype(np.int32)
    return verts, faces


def _on_binary_grid(verts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Round to 1/1024 m, so that adding ``UTM`` is exact in float64."""
    return np.round(verts * 1024.0) / 1024.0


def _build(verts: NDArray[np.float64], faces: NDArray[np.int32]) -> MeshGeometry:
    return build_geometry(verts, faces, zb_from_file=np.zeros(faces.shape[0], dtype=np.float32))


def _assert_same_arrays(a: MeshGeometry, b: MeshGeometry, *, skip: tuple[str, ...] = ()) -> None:
    for name in _GEOMETRY_SCALAR_FIELDS:
        assert getattr(a, name) == getattr(b, name), name
    for name in _GEOMETRY_ARRAY_FIELDS:
        if name in skip or name == "manning_n":
            continue
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), name


@pytest.mark.parametrize("side", [1.0, 0.25, 0.05])
def test_small_cells_at_utm_coordinates_keep_their_geometry(side: float) -> None:
    verts, faces = _rotated_grid(12, 9, side)
    geom = _build(verts + UTM, faces)

    np.testing.assert_allclose(geom.area, side * side, rtol=1e-5)
    np.testing.assert_allclose(geom.edge_len, side, rtol=1e-5)
    interior = geom.edge_cellR >= 0
    centres = geom.centroid.astype(np.float64)
    between = centres[geom.edge_cellR[interior]] - centres[geom.edge_cellL[interior]]
    along = between[:, 0] * geom.edge_nx[interior] + between[:, 1] * geom.edge_ny[interior]
    np.testing.assert_allclose(along, side, rtol=1e-2)


def test_translation_changes_nothing_but_the_origin() -> None:
    verts, faces = _rotated_grid(30, 20, 0.5)
    verts = _on_binary_grid(verts)
    near, _ = hilbert_reorder(_build(verts, faces), verbose=False)
    far, _ = hilbert_reorder(_build(verts + UTM, faces), verbose=False)

    _assert_same_arrays(far, near, skip=("origin",))
    np.testing.assert_array_equal(far.origin, near.origin + UTM)


def test_origin_is_the_lower_left_corner_of_the_mesh() -> None:
    verts, faces = _rotated_grid(4, 3, 2.0)
    geom = _build(verts + UTM, faces)

    assert geom.origin.dtype == np.float64
    assert geom.origin.shape == (2,)
    np.testing.assert_array_equal(geom.origin, (verts + UTM).min(axis=0))
    assert geom.cell_vertices.min() == 0.0


def test_world_centroid_returns_to_the_mesh_file_coordinates() -> None:
    verts, faces = _rotated_grid(4, 3, 2.0)
    world = verts + UTM
    geom = _build(world, faces)

    assert geom.world_centroid.dtype == np.float64
    np.testing.assert_allclose(geom.world_centroid, world[faces].mean(axis=1), atol=1e-5, rtol=0)


def test_orientation_check_tolerates_float32_rounding_far_from_the_origin() -> None:
    # 5 mm cells 200 km from a far corner: their centroids round to a 1.6 cm grid.
    # Rounding keeps each component's sign, so only a sheared cell, whose step to
    # its neighbour is not along the normal, can come out backwards.
    verts, faces = _rotated_grid(40, 40, 0.005, angle=0.0)
    verts[:, 0] += 1.5 * verts[:, 1]
    verts += (200_000.0, 200_000.0)
    corner = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    n = verts.shape[0]
    far_cell = np.array([[n, n + 1, n + 3, n + 2]], dtype=np.int32)
    geom = _build(np.vstack([verts, corner]), np.vstack([faces, far_cell]))

    hilbert_reorder(geom, verbose=False)


@pytest.mark.parametrize("radius", [0.0, 0.9, 6.0], ids=["containing", "small", "wide"])
def test_source_in_world_coordinates_selects_the_same_cells(radius: float) -> None:
    verts, faces = _rotated_grid(20, 15, 2.0)
    verts = _on_binary_grid(verts)
    near = _build(verts, faces)
    far = _build(verts + UTM, faces)
    centre = (-3.3, 17.1)
    valid = np.ones(near.N, dtype=bool)

    def rate(geom: MeshGeometry, xy: tuple[float, float]) -> NDArray[np.float32]:
        phase = SimulationPhase(1.0, [PointSource(1.0, xy, radius)])
        return _build_phase_source_rate(geom, phase, valid)

    expected = rate(near, centre)
    assert (expected > 0.0).any()
    world = rate(far, (centre[0] + UTM[0], centre[1] + UTM[1]))
    assert world.tobytes() == expected.tobytes()


def test_a_source_given_in_local_coordinates_selects_nothing() -> None:
    verts, faces = _rotated_grid(6, 6, 2.0)
    geom = _build(verts + UTM, faces)
    local = (float(geom.centroid[0, 0]), float(geom.centroid[0, 1]))
    phase = SimulationPhase(1.0, [PointSource(1.0, local, 1.0)])
    assert not _build_phase_source_rate(geom, phase, np.ones(geom.N, dtype=bool)).any()


def _write_mesh(path: Path, verts: NDArray[np.float64], faces: NDArray[np.int32]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    wkb: Any = shapely.to_wkb(shapely.polygons(shapely.linearrings(verts[faces])))
    table = {
        "geometry": pa.array([bytes(w) for w in wkb], pa.binary()),
        "z_mean": [0.0] * faces.shape[0],
    }
    pq.write_table(pa.table(table), path)


def test_loader_keeps_float64_coordinates(tmp_path: Path) -> None:
    verts, faces = _rotated_grid(3, 2, 0.1)
    path = tmp_path / "mesh.parquet"
    _write_mesh(path, verts + UTM, faces)

    loaded = load_mesh_file(str(path))[0]
    assert loaded.dtype == np.float64
    np.testing.assert_array_equal(np.sort(loaded, axis=0), np.sort(verts + UTM, axis=0))


def test_obj_loader_keeps_float64_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "mesh.obj"
    path.write_text(
        "v 512345.125 5512345.0625 1\nv 512346.125 5512345.0625 1\n"
        "v 512345.125 5512346.0625 1\nf 1 2 3\n",
        encoding="ascii",
    )
    loaded = load_mesh_file(str(path))[0]
    assert loaded.dtype == np.float64
    assert loaded[0].tolist() == [512345.125, 5512345.0625]


def test_origin_survives_both_cache_layers(tmp_path: Path) -> None:
    verts, faces = _rotated_grid(6, 5, 3.0)
    path = tmp_path / "mesh.parquet"
    _write_mesh(path, verts + UTM, faces)

    artifact = build_geometry_cache(path, tmp_path / "geometry.npz")
    from_artifact, perm, _ = load_geometry_cache_artifact(artifact)
    np.testing.assert_array_equal(from_artifact.origin, (verts + UTM).min(axis=0))

    save_geometry_cache(tmp_path / "cache", "key", from_artifact, perm)
    cached = load_geometry_cache(tmp_path / "cache", "key")
    assert cached is not None
    assert cached[0].origin.dtype == np.float64
    np.testing.assert_array_equal(cached[0].origin, from_artifact.origin)
