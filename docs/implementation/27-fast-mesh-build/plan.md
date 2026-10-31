# Fast Mesh Build — Plan

**Status: done.** Changes how meshes are loaded and built, not what is built.

## Goal

Bring a first run on a new mesh from minutes to seconds, without changing a bit
of the result.

## Problem

`prepare()` spent 210 s on the 2.34 M cell Nysa mesh, nearly all of it in
per-cell and per-edge Python loops, which scaled to about 40 minutes on the
27.7 M cell mosaic.

## Design

The build stays one pass from polygons to arrays, in numpy:

- **Load.** Exterior rings come from `shapely.get_coordinates`. Vertices weld on
  their exact float64 coordinates with a stable `lexsort`, and take ids in
  first-seen order. The GeoPackage and Shapefile path shares this code. OBJ keeps
  its parser.
- **Cells.** Cells are processed one degree at a time. `np.bincount` would be
  simpler but sums sequentially, and `np.sum` does not from 8 terms up, so it
  would change the centroid of any 8-sided cell.
- **Edges.** Sides match by a stable sort on the sorted vertex pair, and edges
  take the order of their first side. A third side on one edge is ignored, as
  before.
- **CSR.** One integer key of cell and side orders each cell's edges by side,
  which is the per-cell gather order.
- **Hilbert order.** `hilbert_distances` is Skilling's transform over arrays.

The int32 CSR overflow check, the wrong-way-normal check and the sliver rejection
stay. The loader gains an int32 offset check, and a non-polygon row now raises a
`ValueError` instead of an `AttributeError`.

## Verification

The gate is bit-identity with the previous build on the unfloored and floored
Nysa meshes: every array in `_GEOMETRY_ARRAY_FIELDS`, `perm`, the scalars and the
loader's output, compared in full by dtype, shape and bytes. Any difference would
mean a new baseline.

The previous build stays in `tests/unit/_geometry_reference.py` as the oracle for
the unit tests, and `hilbertcurve` stays as the oracle for the Hilbert order.
