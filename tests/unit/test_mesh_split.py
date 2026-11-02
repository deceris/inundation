# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Cutting one mesh into basins: rows, shared edges and what the links save."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from inundation.mesh.split import cell_basins, load_split, split_mesh

if TYPE_CHECKING:
    from pathlib import Path

NX, NY = 6, 3


def _grid(path: Path) -> None:
    """A 6 by 3 grid of unit cells far from the origin, as a mesh file would be."""
    cells = [
        shapely.box(380_000.0 + i, 290_000.0 + j, 380_001.0 + i, 290_001.0 + j)
        for j in range(NY)
        for i in range(NX)
    ]
    table = {
        "geometry": pa.array([bytes(shapely.to_wkb(cell)) for cell in cells], pa.binary()),
        "z_mean": [float(k) for k in range(len(cells))],
        "extra": list(range(len(cells))),
    }
    pq.write_table(pa.table(table), path)


def _box(x0: float, x1: float, y0: float = 0.0, y1: float = NY) -> shapely.Geometry:
    return shapely.box(380_000.0 + x0, 290_000.0 + y0, 380_000.0 + x1, 290_000.0 + y1)


def test_each_cell_goes_to_the_polygon_holding_its_centre(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.parquet"
    _grid(mesh)
    basins = cell_basins(mesh, [_box(0.0, 2.0), _box(2.0, 6.0, 0.0, 2.0)])
    assert basins.reshape(NY, NX).tolist() == [
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, -1, -1, -1, -1],
    ]


def test_cells_outside_every_polygon_can_be_kept(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.parquet"
    _grid(mesh)
    basins = cell_basins(mesh, [_box(0.0, 2.0)], keep_rest=True)
    assert np.bincount(basins).tolist() == [6, 12]


def test_a_basin_keeps_its_rows_unchanged(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.parquet"
    _grid(mesh)
    split = split_mesh(mesh, [_box(0.0, 2.0), _box(2.0, 6.0)], tmp_path / "out")
    table = pq.read_table(split.meshes[0])
    assert table.column_names == ["geometry", "z_mean", "extra"]
    assert table["extra"].to_pylist() == [0, 1, 6, 7, 12, 13]


def test_links_hold_the_shared_edges_and_only_those(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.parquet"
    _grid(mesh)
    split = split_mesh(mesh, [_box(0.0, 2.0), _box(2.0, 6.0)], tmp_path / "out")
    assert len(split.links) == 1
    segments = split.links[0].segments
    assert segments.shape == (NY, 4)
    assert (segments[:, [0, 2]] == 380_002.0).all()
    assert sorted(segments[:, [1, 3]].min(axis=1).tolist()) == [290_000.0, 290_001.0, 290_002.0]


def test_a_polygon_with_no_cells_is_skipped_and_numbers_stay_compact(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.parquet"
    _grid(mesh)
    far = shapely.box(0.0, 0.0, 1.0, 1.0)
    split = split_mesh(mesh, [_box(0.0, 2.0), far, _box(2.0, 6.0)], tmp_path / "out")
    assert len(split.meshes) == 2
    assert [(link.first, link.second) for link in split.links] == [(0, 1)]


def test_a_split_reads_back_the_same(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.parquet"
    _grid(mesh)
    polygons = [_box(0.0, 2.0), _box(2.0, 4.0), _box(4.0, 6.0)]
    split = split_mesh(mesh, polygons, tmp_path / "out")
    again = load_split(tmp_path / "out")
    assert again.meshes == split.meshes
    assert [(link.first, link.second) for link in again.links] == [(0, 1), (1, 2)]
    for a, b in zip(again.links, split.links, strict=True):
        assert a.segments.tobytes() == b.segments.tobytes()
