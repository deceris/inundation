"""Mesh I/O helpers for the mesh-based SWE GPU solver.

Supports loading meshes (triangles, quads, or mixed polygons) from
GeoPackage (.gpkg), Shapefile (.shp), GeoParquet (.parquet / .geoparquet),
and Wavefront OBJ (.obj) formats.

Usage
-----
    verts, faces, face_offsets, zb_from_file, manning_n_from_file = load_mesh_file(
        "path/to/mesh.gpkg"
    )
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..tuning import MIN_POLYGON_VERTICES

# shapely.GeometryType.POLYGON
_POLYGON_TYPE_ID = 3


def _load_optional_manning_n(values: object) -> NDArray[np.float32]:
    """Return finite-positive per-cell Manning roughness from a source column."""
    try:
        manning_n = np.asarray(values, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("manning_n must contain finite positive values") from exc
    if not np.isfinite(manning_n).all() or not (manning_n > 0.0).all():
        raise ValueError("manning_n must contain finite positive values")
    return manning_n


def _weld_polygons(
    geoms: NDArray[Any],
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
    """Vertices and flat faces of polygons, numbering vertices as they are first met.

    Vertices weld on their exact float64 coordinates. Interior rings are ignored.
    """
    import shapely

    wrong_type = np.flatnonzero(shapely.get_type_id(geoms) != _POLYGON_TYPE_ID)
    if wrong_type.size:
        raise ValueError(
            f"{wrong_type.size} geometries are not polygons (first: row {int(wrong_type[0])})"
        )
    ring_xy, ring_of = shapely.get_coordinates(shapely.get_exterior_ring(geoms), return_index=True)
    # A closed ring repeats its first point last.
    n_cells = geoms.size
    degrees = np.maximum(np.bincount(ring_of, minlength=n_cells) - 1, 0)
    short = np.flatnonzero(degrees < MIN_POLYGON_VERTICES)
    if short.size:
        raise ValueError(f"Geometry has {degrees[short[0]]} vertices — need at least 3.")
    total = int(degrees.sum())
    if total >= 2**31:
        raise ValueError(f"mesh too large for int32 face offsets: {total} polygon vertices")
    offsets = np.zeros(n_cells + 1, dtype=np.int32)
    np.cumsum(degrees, out=offsets[1:])
    keep = np.ones(ring_xy.shape[0], dtype=bool)
    keep[offsets[1:].astype(np.int64) + np.arange(n_cells)] = False
    xy = ring_xy[keep]
    del ring_xy, ring_of, keep

    # lexsort is stable, so each run of equal points starts at its first use.
    order = np.lexsort((xy[:, 1], xy[:, 0]))
    ordered = xy[order]
    new_point = np.r_[True, (ordered[1:] != ordered[:-1]).any(axis=1)]
    del ordered
    first_use = order[new_point]
    id_by_first_use = np.empty(first_use.size, dtype=np.int32)
    by_use = np.argsort(first_use)
    id_by_first_use[by_use] = np.arange(first_use.size, dtype=np.int32)
    faces = np.empty(total, dtype=np.int32)
    faces[order] = id_by_first_use[np.cumsum(new_point) - 1]
    return xy[first_use[by_use]].astype(np.float32), faces, offsets


def load_mesh_file(
    path: str,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.float32] | None,
]:
    """Load a polygonal mesh with required bed elevations, plus optional roughness.

    Accepts triangles, quads, and arbitrary convex polygons.

    .gpkg / .shp / .parquet / .geoparquet :
                   each feature must be a simple polygon (convex recommended).
                   .parquet / .geoparquet use pyarrow + shapely only (no geopandas).
                   .gpkg / .shp require geopandas (pip install geopandas).
                   All GIS formats require a ``z_mean`` bed-elevation column;
                   the optional ``manning_n`` column must be finite and positive.
    .obj         : Wavefront OBJ; every ``v`` record must include Z. Face
                   vertex elevations are averaged into per-cell bed elevations.

    Returns
    -------
    verts               : (V, 2) float32   vertex (x, y) coordinates
    faces               : (sum_of_degrees,) int32  flat vertex indices for all faces
    face_offsets        : (N+1,) int32  face i has verts faces[face_offsets[i]:face_offsets[i+1]]
    zb_from_file        : (N,) float32  required per-cell bed elevations
    manning_n_from_file : (N,) float32 or None  per-cell Manning roughness attribute

    """
    import os as _os

    ext = _os.path.splitext(path)[1].lower()

    if ext in (".parquet", ".geoparquet"):
        # pyarrow + shapely only — no geopandas needed for the parquet path.
        # Both real GeoParquet (written with 'geo' metadata) and plain parquet
        # (e.g. produced by a DuckDB Query clip node) store geometry as a WKB
        # binary column, so reading it with pyarrow and decoding via
        # shapely.from_wkb works uniformly for both.
        import pyarrow.parquet as pq
        from shapely import from_wkb, is_missing

        table = pq.read_table(path)
        geometry_wkb = table.column("geometry").to_numpy(zero_copy_only=False)
        geoms = from_wkb(geometry_wkb)
        not_null = ~is_missing(geoms)

        if not not_null.any():
            raise ValueError(
                f"Mesh file '{path}' contains no valid polygon geometries "
                f"({table.num_rows} rows, {int(not_null.sum())} non-null geometries)."
            )
        verts, faces_flat, offsets = _weld_polygons(geoms[not_null])
        if "z_mean" not in table.column_names:
            raise ValueError(f"Mesh file '{path}' is missing required 'z_mean' bed elevations")
        zb_vals = table.column("z_mean").to_numpy(zero_copy_only=False)[not_null].astype(np.float32)
        if not np.isfinite(zb_vals).all():
            raise ValueError(f"Mesh file '{path}' contains non-finite 'z_mean' elevations")
        manning_n_vals = (
            _load_optional_manning_n(
                table.column("manning_n").to_numpy(zero_copy_only=False)[not_null]
            )
            if "manning_n" in table.column_names
            else None
        )

        return verts, faces_flat, offsets, zb_vals, manning_n_vals

    if ext in (".gpkg", ".shp"):
        import geopandas as gpd

        gdf = gpd.read_file(path)
        valid_geometry = gdf.geometry.notna()
        if not valid_geometry.any():
            n_rows = len(gdf)
            n_valid = valid_geometry.sum()
            raise ValueError(
                f"Mesh file '{path}' contains no valid polygon geometries "
                f"({n_rows} rows, {n_valid} non-null geometries)."
            )
        verts, faces_flat, offsets = _weld_polygons(np.asarray(gdf.geometry[valid_geometry]))
        if "z_mean" not in gdf.columns:
            raise ValueError(f"Mesh file '{path}' is missing required 'z_mean' bed elevations")
        zb_vals = gdf.loc[valid_geometry, "z_mean"].to_numpy(dtype=np.float32)
        if not np.isfinite(zb_vals).all():
            raise ValueError(f"Mesh file '{path}' contains non-finite 'z_mean' elevations")
        manning_n_vals = (
            _load_optional_manning_n(gdf.loc[valid_geometry, "manning_n"].to_numpy())
            if "manning_n" in gdf.columns
            else None
        )

        return verts, faces_flat, offsets, zb_vals, manning_n_vals

    if ext == ".obj":
        vert_list: list[list[float]] = []
        vert_z: list[float] = []
        faces_list: list[list[int]] = []
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts or parts[0].startswith("#"):
                    continue
                if parts[0] == "v":
                    if len(parts) < 4:
                        raise ValueError(
                            f"OBJ file '{path}' has a vertex without required Z elevation"
                        )
                    vert_list.append([float(parts[1]), float(parts[2])])
                    vert_z.append(float(parts[3]))
                elif parts[0] == "f":
                    idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    if len(idx) < MIN_POLYGON_VERTICES:
                        raise ValueError(f"OBJ face has {len(idx)} vertices — need at least 3.")
                    faces_list.append(idx)

        if not faces_list:
            raise ValueError(f"OBJ file '{path}' contains no face ('f') records.")
        vertex_z = np.asarray(vert_z, dtype=np.float32)
        if not np.isfinite(vertex_z).all():
            raise ValueError(f"OBJ file '{path}' contains non-finite vertex elevations")

        verts = np.array(vert_list, dtype=np.float32)
        faces_flat = np.concatenate([np.array(f, dtype=np.int32) for f in faces_list])
        offsets = np.zeros(len(faces_list) + 1, dtype=np.int32)
        for i, f in enumerate(faces_list):
            offsets[i + 1] = offsets[i] + len(f)
        zb_vals = np.asarray([vertex_z[f].mean() for f in faces_list], dtype=np.float32)

        return verts, faces_flat, offsets, zb_vals, None

    raise ValueError(
        f"Unsupported format: '{ext}'. Supported: .gpkg, .shp, .parquet, .geoparquet, .obj"
    )
