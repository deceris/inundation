# Fast Mesh Build — Results

**Status: done.** `GEOMETRY_CACHE_VERSION` stays at 5, and existing caches stay
valid.

## Gate

All 31 arrays are bit-identical to the build at `4c9fa8b` on both meshes.

| Mesh | Cells | Edges | Vertices | Result |
| --- | --- | --- | --- | --- |
| Unfloored Nysa (`full0`) | 2 343 783 | 6 124 484 | 3 780 702 | all identical |
| Floored Nysa (`final`) | 2 339 765 | 6 119 824 | 3 780 060 | all identical |

The 31 are the 23 arrays of `_GEOMETRY_ARRAY_FIELDS`, the 4 scalars, `perm` and
the loader's vertices, faces and offsets. Because the build is identical, step
counts cannot change, and no solver run was needed.

## Time and memory

Load plus build plus reorder, on Nysa.

| Stage | Old code | New code |
| --- | --- | --- |
| Load | 46–49 s | 10 s |
| Build | 126–149 s | 6 s |
| Reorder | 29–32 s | 3 s |
| Total | 201–227 s | 19–20 s |
| Peak memory footprint | 7.3 GB | 4.3 GB |

The ranges cover old-code runs before and after the new code, so the order of the
pair does not explain the gain. The two faster old-code runs overlapped unit tests
on the same machine, which makes the gain if anything larger than the low end.

## Mosaic

27 749 948 cells, 65 405 333 edges, 37 655 386 vertices, maximum degree 8.
`prepare()` takes 495 s and 514 s in two runs, against the projected 40 minutes.
The stages of the 495 s run:

| Stage | Time |
| --- | --- |
| Load | 222 s |
| Cell area and centroid | 15 s |
| Half-edge match | 77 s |
| Edge length and normal | 14 s |
| CSR adjacency | 18 s |
| Dense arrays | 3 s |
| Hilbert permutation | 28 s |
| Reorder | 55 s |
| Cache save | 10 s |
| The rest, including solver setup | 54 s |

Peak resident memory is 22 GiB. Peak footprint is 41.5 GB and includes the GPU
buffers. The geometry cache is 8.0 GB. A 10 s run from a point source then took
16 steps and held exact volume (50.000 m³ injected, 50.000 m³ in the domain). The
old build was not run on the mosaic, so its time is only a projection, and the
build has no bit-identity proof at that size.

## Rounding traps in the prototype

The prototype matched up to ordering and tolerance. Three of its choices would
have broken bit-identity or could have:

- `np.bincount` sums sequentially, unlike `np.sum` from 8 terms up, which changes
  centroids.
- `np.hypot` differs from `sqrt(dx*dx + dy*dy)` in the last float64 bit.
- A clockwise cell keeps its area sum but needs its centroid summed in reversed
  order.

## Tests

`test_mesh_build_equivalence.py` compares the new build with the old one on
mixed meshes of degree 4–20 with clockwise cells, at Nysa and UTM coordinates, and
on uniform and non-manifold meshes. The Hilbert distance is checked against the
library on every grid point up to order 10, which is the order the solver uses.

Of seven deliberate mutations, four are caught. The three that survive cannot
show in a small test. Two change the area summation order, and that order cannot
matter: products of float32 coordinates are exact in float64. The third, `hypot`,
is a last-bit float64 effect that rarely survives the float32 cast, so the mesh
gate is its only check.

## Dense arrays

`cell_edges` and `cell_neighbors` have no reader outside the cache. They cost
888 MB on the mosaic, and removing them changes the cache's fields, so they stay
until the version bump in the next effort.
