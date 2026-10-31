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
counts cannot change. The solver runs below confirm that.

## End-to-end runs

The 24 h Nysa run through `flood.py` at `d074793`, starting from no cache:

| Mesh | Steps | Prepare | `depth.parquet` |
| --- | --- | --- | --- |
| Floored Nysa | 682 876 | 25–26 s | 63 of 63 columns bit-identical |
| Unfloored Nysa | 1 125 485 | 25–26 s | 63 of 63 columns bit-identical |

Both step counts match the earlier runs exactly. All Nysa cells are already
counter-clockwise, so these runs do not reach the clockwise handling; only the
unit tests below cover it.

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
16 steps and held exact volume (50.000 m³ injected, 50.000 m³ in the domain).

### Mosaic bit-identity

Scaled from the slices below, the old build needs about 65–70 GB on the mosaic and
does not fit in the machine's 64 GB. It was never run whole, so the projected 40
minutes stays a projection.
Each stage was compared separately instead:

| Stage | Compared on | Result |
| --- | --- | --- |
| Load | full mosaic, old against new | 5 of 5 arrays identical |
| Reorder | full mosaic, both fed by the new build | 28 of 28 array hashes identical |
| Load, build, reorder | first 12 M cells, old against new | 5, 27, 28 arrays identical |
| Load, build, reorder | last 12 M cells, old against new | 5, 27, 28 arrays identical |

The two slices hold 86.5 % of the cells. The other 13.5 % sit between them and were
never in an old build, and neither was any edge joining the two slices. For those
cells, the new build on the full mosaic rests on these checks, which do not use
the old code:

- `perm` is a permutation, and elevation, Manning and cell degree follow it.
- Areas match a local-frame shoelace, with total area 2973.581 km² both ways.
- Every centroid lies inside its bounding box.
- Every CSR slot names its own cell, and every CSR neighbour is the other cell of
  that edge.
- The 130 786 953 CSR slots equal the edges seen from both sides, with 23 713
  boundary edges, and CSR degree equals polygon degree in every cell.
- Edges are sorted by lower cell, and every cell boundary closes: the largest
  `|Σ n L|` over perimeter is 6.3e-08, which is rounding.

The sort keys stay far inside int64: `V*(V+1)` is 1.42e15 against 9.2e18.

Stage times on the full mosaic in these runs:

| Stage | Old code | New code |
| --- | --- | --- |
| Load | 543 s | 177 s |
| Build | not run | 106 s |
| Reorder | 380 s | 71 s |

On a 12 M-cell slice the old build takes 619–626 s and 29 GiB, and the new one
33–34 s and 10 GiB.

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
