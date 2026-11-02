"""Cut one mesh into basin meshes along catchment polygons.

A cell belongs to the polygon holding its centroid, so the cut follows cell edges and
both sides of a border share its vertices exactly. Cells outside every polygon are
dropped, or kept as one more basin. Each basin is the original file's rows, unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from .loader import load_mesh_file

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

_LINKS_FILE = "links.npz"
_BASIN_FILE = "basin-{}.parquet"


@dataclass(frozen=True)
class BasinLink:
    """The edges two basins share, as ``(K, 4)`` x0, y0, x1, y1 in world coordinates."""

    first: int
    second: int
    segments: NDArray[np.float64]


@dataclass(frozen=True)
class SplitMesh:
    meshes: list[Path]
    links: list[BasinLink]


def cell_basins(
    mesh_path: str | Path, polygons: Sequence[shapely.Geometry], *, keep_rest: bool = False
) -> NDArray[np.int64]:
    """Each cell's polygon, ``len(polygons)`` outside them all if ``keep_rest``, else -1."""
    table = pq.read_table(mesh_path, columns=["geometry"])
    cells = shapely.from_wkb(table["geometry"].to_numpy(zero_copy_only=False))
    centroids = shapely.centroid(cells)
    basins = np.full(cells.size, len(polygons) if keep_rest else -1, dtype=np.int64)
    found = shapely.STRtree(centroids).query(np.asarray(polygons), predicate="contains")
    # Reversed, so the first polygon wins where two overlap.
    basins[found[1][::-1]] = found[0][::-1]
    return basins


def _shared_edges(
    faces: NDArray[np.int32], offsets: NDArray[np.int32]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Vertices and the two cells of every edge held by two cells."""
    degree = np.diff(offsets).astype(np.int64)
    cell = np.repeat(np.arange(degree.size, dtype=np.int64), degree)
    start = faces.astype(np.int64)
    following = np.roll(start, -1)
    following[offsets[1:] - 1] = start[offsets[:-1]]
    low, high = np.minimum(start, following), np.maximum(start, following)
    order = np.lexsort((high, low))
    low, high, cell = low[order], high[order], cell[order]
    same = (low[1:] == low[:-1]) & (high[1:] == high[:-1])
    return low[:-1][same], high[:-1][same], cell[:-1][same], cell[1:][same]


def split_mesh(
    mesh_path: str | Path,
    polygons: Sequence[shapely.Geometry],
    out_dir: str | Path,
    *,
    keep_rest: bool = False,
) -> SplitMesh:
    """Write one parquet per basin that holds cells, and the links between them."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    basins = cell_basins(mesh_path, polygons, keep_rest=keep_rest)
    verts, faces, offsets, _, _ = load_mesh_file(str(mesh_path))
    table = pq.read_table(mesh_path)
    if table.num_rows != basins.size or offsets.size - 1 != basins.size:
        raise ValueError("the mesh has null rows, so cells do not match its file rows")

    present = np.unique(basins[basins >= 0])
    number = np.full(max(len(polygons) + 1, 1), -1, dtype=np.int64)
    number[present] = np.arange(present.size)
    meshes = []
    for k, basin in enumerate(present.tolist()):
        path = out / _BASIN_FILE.format(k)
        pq.write_table(table.filter(pa.array(basins == basin)), path)
        meshes.append(path)

    a, b, left, right = _shared_edges(faces, offsets)
    left_basin, right_basin = basins[left], basins[right]
    cut = (left_basin != right_basin) & (left_basin >= 0) & (right_basin >= 0)
    first = number[np.minimum(left_basin[cut], right_basin[cut])]
    second = number[np.maximum(left_basin[cut], right_basin[cut])]
    segments = np.column_stack([verts[a[cut]], verts[b[cut]]]).astype(np.float64)
    links = []
    for i, j in np.unique(np.column_stack([first, second]), axis=0).tolist():
        links.append(BasinLink(i, j, segments[(first == i) & (second == j)]))
    np.savez(
        out / _LINKS_FILE,
        first=np.array([link.first for link in links], dtype=np.int64),
        second=np.array([link.second for link in links], dtype=np.int64),
        counts=np.array([link.segments.shape[0] for link in links], dtype=np.int64),
        segments=np.concatenate([link.segments for link in links])
        if links
        else np.empty((0, 4), dtype=np.float64),
    )
    return SplitMesh(meshes, links)


def load_split(out_dir: str | Path) -> SplitMesh:
    """A split written by ``split_mesh``."""
    out = Path(out_dir)
    saved = np.load(out / _LINKS_FILE)
    chunks = np.split(saved["segments"], np.cumsum(saved["counts"])[:-1])
    links = [
        BasinLink(int(i), int(j), chunk)
        for i, j, chunk in zip(saved["first"], saved["second"], chunks, strict=True)
    ]
    meshes = sorted(out.glob(_BASIN_FILE.format("*")), key=lambda path: int(path.stem[6:]))
    return SplitMesh(meshes, links)
