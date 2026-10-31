# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The vectorised mesh build reproduces the pure-Python one bit for bit.

Cell numbering, edge numbering and each cell's edge order fix the gather order
of the deterministic solver, so a rounding-level difference here would change
every run. The cases cover what a numpy rewrite gets wrong first: rows of 8 or
more terms, whose ``np.sum`` is not a sequential sum; clockwise cells, whose
centroid and area are summed in different orders; and edges shared unusually.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import shapely
from hilbertcurve.hilbertcurve import HilbertCurve

from inundation.mesh.cache import _GEOMETRY_ARRAY_FIELDS, _GEOMETRY_SCALAR_FIELDS
from inundation.mesh.geometry import (
    build_geometry,
    hilbert_distances,
    hilbert_permutation,
    hilbert_reorder,
)
from inundation.mesh.loader import load_mesh_file

from . import _geometry_reference as reference

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry

# Nysa-like coordinates, then UTM-like ones, whose float32 rounding skews cells.
ORIGIN = (380_000.0, 290_000.0)
UTM_ORIGIN = (500_000.0, 5_500_000.0)


def at_origin(verts: NDArray[np.float32]) -> NDArray[np.float32]:
    """``verts`` moved to their origin, the only mesh the old build can match.

    The result is exact: both terms sit on the same float32 grid.
    """
    return (verts.astype(np.float64) - verts.min(axis=0)).astype(np.float32)


def _assert_identical(a: MeshGeometry, b: MeshGeometry) -> None:
    for name in _GEOMETRY_SCALAR_FIELDS:
        assert getattr(a, name) == getattr(b, name), name
    for name in _GEOMETRY_ARRAY_FIELDS:
        if name == "origin":
            continue
        x, y = getattr(a, name), getattr(b, name)
        if x is None or y is None:
            assert x is y, name
            continue
        assert x.dtype == y.dtype, name
        assert x.shape == y.shape, name
        assert x.tobytes() == y.tobytes(), name


def _polygon_mesh(
    seed: int,
    *,
    nx: int = 24,
    ny: int = 18,
    max_extra: int = 4,
    flip: float = 0.3,
    origin: tuple[float, float] = ORIGIN,
) -> tuple[
    list[list[tuple[float, float]]],
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    """Jittered grid whose shared sides carry extra points, so degrees run 4 to 20.

    Returns each cell's polygon (float64), then the welded vertices, flat faces
    and face offsets.
    """
    rng = np.random.default_rng(seed)
    corners = np.stack(np.meshgrid(np.arange(ny + 1.0), np.arange(nx + 1.0), indexing="ij"), -1)
    corners = corners[..., ::-1] * 7.0 + rng.uniform(-1.5, 1.5, corners.shape)

    def side_points(a: NDArray[np.float64], b: NDArray[np.float64]) -> list[tuple[float, float]]:
        n = int(rng.integers(0, max_extra + 1))
        return [tuple((a + (b - a) * (k + 1) / (n + 1)).tolist()) for k in range(n)]

    horizontal = {
        (j, i): side_points(corners[j, i], corners[j, i + 1])
        for j in range(ny + 1)
        for i in range(nx)
    }
    vertical = {
        (j, i): side_points(corners[j, i], corners[j + 1, i])
        for j in range(ny)
        for i in range(nx + 1)
    }

    polygons: list[list[tuple[float, float]]] = []
    for j in range(ny):
        for i in range(nx):
            ring = [
                tuple(corners[j, i].tolist()),
                *horizontal[j, i],
                tuple(corners[j, i + 1].tolist()),
                *vertical[j, i + 1],
                tuple(corners[j + 1, i + 1].tolist()),
                *reversed(horizontal[j + 1, i]),
                tuple(corners[j + 1, i].tolist()),
                *reversed(vertical[j, i]),
            ]
            ring = [(x + origin[0], y + origin[1]) for x, y in ring]
            start = int(rng.integers(0, len(ring)))
            ring = ring[start:] + ring[:start]
            if rng.uniform() < flip:
                ring.reverse()
            polygons.append(ring)

    ids: dict[tuple[float, float], int] = {}
    flat: list[int] = []
    offsets = [0]
    for ring in polygons:
        flat += [ids.setdefault(point, len(ids)) for point in ring]
        offsets.append(len(flat))
    return (
        polygons,
        np.array(list(ids), dtype=np.float32),
        np.array(flat, dtype=np.int32),
        np.array(offsets, dtype=np.int32),
    )


def _bed(rng: np.random.Generator, n: int) -> NDArray[np.float32]:
    return rng.uniform(100.0, 120.0, n).astype(np.float32)


@pytest.mark.parametrize("origin", [ORIGIN, UTM_ORIGIN], ids=["nysa", "utm"])
@pytest.mark.parametrize("seed", range(5))
def test_mixed_degree_mesh_is_bit_identical(seed: int, origin: tuple[float, float]) -> None:
    _, verts, faces, offsets = _polygon_mesh(seed, nx=60, ny=50, origin=origin)
    assert np.diff(offsets).max() >= 8
    zb = _bed(np.random.default_rng(seed), offsets.size - 1)
    manning = np.full(offsets.size - 1, 0.04, dtype=np.float32)
    args: dict[str, Any] = {
        "face_offsets": offsets,
        "zb_from_file": zb,
        "manning_n_from_file": manning,
    }
    old = reference.build_geometry(at_origin(verts), faces, **args)
    new = build_geometry(verts, faces, **args)
    _assert_identical(new, old)

    reordered_old, perm_old = reference.hilbert_reorder(old)
    reordered_new, perm_new = hilbert_reorder(new, verbose=False)
    assert perm_new.dtype == perm_old.dtype
    assert perm_new.tobytes() == perm_old.tobytes()
    _assert_identical(reordered_new, reordered_old)


def test_uniform_degree_mesh_is_bit_identical() -> None:
    rng = np.random.default_rng(7)
    xs, ys = np.meshgrid(np.arange(21.0), np.arange(11.0))
    verts = (np.column_stack([xs.ravel(), ys.ravel()]) * 3.0 + ORIGIN).astype(np.float32)

    def vid(i: int, j: int) -> int:
        return j * 21 + i

    faces = np.array(
        [
            [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            for j in range(10)
            for i in range(20)
        ],
        dtype=np.int32,
    )
    faces[::3] = faces[::3, ::-1]
    zb = _bed(rng, 200)
    _assert_identical(
        build_geometry(verts, faces, zb_from_file=zb),
        reference.build_geometry(at_origin(verts), faces, zb_from_file=zb),
    )


def test_edge_shared_by_three_cells_keeps_the_first_two() -> None:
    verts = np.array([[0, 0], [2, 0], [1, 1], [1, -1], [3, 3]], dtype=np.float32)
    faces = np.array([0, 1, 2, 1, 0, 3, 0, 1, 4, 2, 1, 4], dtype=np.int32)
    offsets = np.array([0, 3, 6, 9, 12], dtype=np.int32)
    zb = np.zeros(4, dtype=np.float32)
    _assert_identical(
        build_geometry(verts, faces, offsets, zb_from_file=zb),
        reference.build_geometry(at_origin(verts), faces, offsets, zb_from_file=zb),
    )


@pytest.mark.parametrize("hilbert_p", [1, 2, 3, 5, 10])
def test_hilbert_distances_match_the_library_on_every_grid_point(hilbert_p: int) -> None:
    side = 2**hilbert_p
    x, y = (a.ravel() for a in np.meshgrid(np.arange(side), np.arange(side)))
    points = np.column_stack([x, y]).tolist()
    expected = HilbertCurve(p=hilbert_p, n=2).distances_from_points(points)
    np.testing.assert_array_equal(
        hilbert_distances(x, y, hilbert_p), np.array(expected, dtype=np.int64)
    )


def test_hilbert_permutation_ties_keep_file_order() -> None:
    centroid = np.array([[0, 0], [9, 9], [0, 0], [0, 9], [0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(
        hilbert_permutation(centroid), reference.hilbert_permutation(centroid)
    )


def _old_weld(
    polygons: list[list[tuple[float, float]]],
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    ids: dict[tuple[float, float], int] = {}
    flat: list[int] = []
    offsets = [0]
    for ring in polygons:
        for point in ring:
            flat.append(ids.setdefault(point, len(ids)))
        offsets.append(len(flat))
    return (
        np.array(list(ids), dtype=np.float64),
        np.array(flat, dtype=np.int32),
        np.array(offsets, dtype=np.int32),
    )


def _write_parquet(
    path: Path, geometry: Sequence[bytes | None], zb: list[float], manning: list[float] | None
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns: dict[str, Any] = {"geometry": pa.array(geometry, pa.binary()), "z_mean": zb}
    if manning is not None:
        columns["manning_n"] = manning
    pq.write_table(pa.table(columns), path)


def test_parquet_load_numbers_vertices_as_first_met(tmp_path: Path) -> None:
    polygons = _polygon_mesh(3)[0]
    wkb = shapely.to_wkb(shapely.polygons([shapely.linearrings(ring) for ring in polygons]))
    # A null row is dropped along with its elevation.
    geometry: list[bytes | None] = [bytes(w) for w in wkb]
    geometry.insert(5, None)
    zb = [float(k) for k in range(len(geometry))]
    path = tmp_path / "mesh.parquet"
    _write_parquet(path, geometry, zb, [0.04] * len(geometry))

    verts, faces, offsets, zb_read, manning = load_mesh_file(str(path))
    expected = _old_weld(polygons)
    assert verts.tobytes() == expected[0].tobytes()
    assert faces.tobytes() == expected[1].tobytes()
    assert offsets.tobytes() == expected[2].tobytes()
    assert verts.dtype == np.float64
    assert faces.dtype == np.int32
    assert offsets.dtype == np.int32
    assert 5.0 not in zb_read.tolist()
    assert zb_read.size == len(polygons)
    assert manning is not None
    assert manning.size == len(polygons)


def test_gpkg_load_matches_parquet_load(tmp_path: Path) -> None:
    gpd = pytest.importorskip("geopandas")
    polygons = _polygon_mesh(4, nx=6, ny=5)[0]
    geoms = shapely.polygons([shapely.linearrings(ring) for ring in polygons])
    zb = [float(k) for k in range(len(polygons))]
    gpkg = tmp_path / "mesh.gpkg"
    gpd.GeoDataFrame({"z_mean": zb}, geometry=list(geoms)).to_file(gpkg)
    parquet = tmp_path / "mesh.parquet"
    _write_parquet(parquet, [bytes(w) for w in shapely.to_wkb(geoms)], zb, None)

    for a, b in zip(load_mesh_file(str(gpkg))[:4], load_mesh_file(str(parquet))[:4], strict=True):
        assert a.tobytes() == b.tobytes()


def test_load_rejects_a_ring_with_too_few_vertices(tmp_path: Path) -> None:
    empty = bytes(shapely.to_wkb(shapely.Polygon()))
    path = tmp_path / "mesh.parquet"
    _write_parquet(path, [empty], [1.0], None)
    with pytest.raises(ValueError, match="need at least 3"):
        load_mesh_file(str(path))


def test_load_rejects_a_non_polygon(tmp_path: Path) -> None:
    point = bytes(shapely.to_wkb(shapely.Point(0, 0)))
    path = tmp_path / "mesh.parquet"
    _write_parquet(path, [point], [1.0], None)
    with pytest.raises(ValueError, match="not polygons"):
        load_mesh_file(str(path))
