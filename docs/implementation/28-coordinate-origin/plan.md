# Coordinate Origin — Plan

**Status: done.** Second of the four solver items for the pavé meshes. This
one changes the mesh arrays, so it sets a new baseline.

## Goal

Make mesh geometry independent of where the mesh sits on the map, so that a small
cell at UTM coordinates is as accurate as the same cell near zero.

## Problem

`loader.py` casts vertices straight to `float32`. At Nysa's coordinates that puts
them on a 3.1 cm grid, and at a UTM northing of 5.5 M on a 50 cm grid, which is
coarser than the cells pavé can produce. Area, edge length and normals are all
computed from those rounded vertices. [Effort 25](../25-minimum-cell-size/results.md)
measured the area error.

## Design

- **Load.** `load_mesh_file` returns float64 vertices.
- **Build.** `build_geometry` takes the lower-left corner of the vertices as the
  origin and subtracts it in float64. Area, centroid, edge length and normals are
  computed from those unrounded local coordinates, and only the results are cast
  to `float32`.
- **Store.** `MeshGeometry.origin` holds the origin as `float64`, and the cache
  stores it. Every coordinate on `MeshGeometry` is local to it.
- **Convert.** Real-world coordinates stay at the edge of the solver. `PointSource`
  centres are in the mesh file's coordinates and `_build_phase_source_rate`
  subtracts the origin. `MeshGeometry.world_centroid` gives float64 centroids in
  the mesh file's coordinates, and benches read that. Plots draw from the loader's
  vertices, which stay in world coordinates.
- **Orientation.** Normals are oriented with the float64 centroids. The check after
  the Hilbert reorder only sees `float32` centroids, so it tolerates their rounding
  and no longer rejects a mesh over it.
- **Cache.** `GEOMETRY_CACHE_VERSION` goes to 6. The same bump drops `cell_edges`
  and `cell_neighbors`, which nothing reads.

The GPU buffers hold no coordinates, so no shader changes.

Lower-left is enough: with geometry computed in float64, the `float32` centroid and
vertices only feed source selection, Hilbert order and plots, and a centred origin
would gain one bit there.

MUSCL does not conflict with this. It will need centroid differences, which local
coordinates make more accurate.

## Verification

- Unit tests: a small-celled mesh at UTM coordinates, which fails before the change;
  translation changes nothing but the origin; sources in world coordinates select
  the same cells; the origin survives both cache layers.
- The old build stays the oracle: run on a mesh already at its origin, it must
  still match bit for bit.
- Gate suites before and after, by step count and metrics.
- Nysa 24 h on both meshes against 682 876 and 1 125 485 steps. Any change comes
  from the rounding of area, normals and Hilbert order, and is the new baseline.
- The mosaic: prepare and a short run with exact volume.
