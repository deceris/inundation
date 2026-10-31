# Coordinate Origin — Results

**Status: done.** `GEOMETRY_CACHE_VERSION` is now 6, so every existing cache
rebuilds and every explicit artifact has to be rebuilt. This is a new baseline for
real meshes. Benchmark meshes keep their results, because they already sit at zero.

## What the old rounding cost

Old and new arrays on the floored Nysa mesh, cell by cell. Origin
(380 373, 288 204).

| Quantity | Change |
| --- | --- |
| Cell area | RMS 0.11 %, p99.9 0.51 %, largest 1.07 % |
| Total area | 1 040 904 603.000 → 1 040 904 603.002 m² |
| Edge length | largest 1.04 % |
| Centroid | RMS 1.4 cm, largest 4.2 cm |
| Hilbert order | 1 155 816 of 2 339 765 cells move |

The old rounding moved vertices but cells still shared them, which is why total
area hardly changes.

## Nysa 24 h

`flood.py` at `fa9a87e` with the standard Nysa arguments.

| Mesh | Steps before | Steps after | Injected = in domain |
| --- | --- | --- | --- |
| Floored | 682 876 | 685 289 (+0.35 %) | 3 739 500 m³ |
| Unfloored | 1 125 485 | 1 127 159 (+0.15 %) | 3 739 500 m³ |

Depth against the previous run, over cells wetter than 1 cm:

| Mesh | Wet cells | RMS | p99 | p99.9 | Largest | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| Floored `h_max` | 58 634 | 0.04 cm | 0.13 cm | 0.37 cm | 2.38 cm | 0.000 cm |
| Unfloored `h_max` | 58 936 | 0.04 cm | 0.13 cm | 0.39 cm | 2.45 cm | 0.000 cm |

Wall times were 508 s and 773 s. There is no back-to-back pair behind them, so they
say nothing about speed.

## Gate suites

All pass at `fa9a87e`, against the baseline at `4c9fa8b`. Mesh output did not change
between those commits. Step counts are identical in every suite.

| Suite | Steps | Other values |
| --- | --- | --- |
| Dam break | 1001, 1001 | identical |
| Radial dam break | 175 | identical |
| Flood propagation | 76 221 | identical |
| Momentum obstruction | 4130 | identical |
| Floodplain depressions | 112 550 | identical |
| Parabolic bowl | 4648 | see below |

The dam break, radial, propagation, momentum and depression meshes have their
lower-left corner at zero and vertices that fit `float32`, so their geometry is
unchanged by construction.

The bowl is centred on zero, so its origin is not zero. Its areas, normals and edge
order are bit-identical to before. Only the initial state moves, because it is
evaluated at centroids rounded differently.

| Bowl metric | Before | After |
| --- | --- | --- |
| Depth L1, mean | 0.1727009 | 0.1727009 |
| Depth L1, final | 0.3090160 | 0.3090159 |
| Speed L1, final | 0.3740187 | 0.3740187 |
| Volume drift | 1.6e-10 | 3.1e-9 |

The drift is noise, not a loss of conservation. Over seven CFL values from 0.76 to
0.84 it spans 1.6e-10 to 4.0e-9 on the old code and 5.0e-11 to 3.1e-9 on the new
code. The gate limit is 1e-7.

## Mosaic

Origin (303 670.12, 257 407.87). `prepare()` from no cache took 376 s, and a 10 s run
took 16 steps, as before, and held exact volume (50.000 m³ injected, 50.000 m³ in
the domain). Peak resident memory went from 22 to 28 GiB and peak footprint from
41.5 to 45.3 GB, because vertices and centroids stay in float64 during the build.
The geometry cache fell from 8.0 to 6.8 GB. The 376 s has no back-to-back pair with
the earlier 495–514 s.

## Tests

`test_mesh_origin.py` adds 14 tests, 11 of which failed before the change. They
cover 5 cm to 1 m cells at a UTM position, translation changing nothing but the
origin, world-coordinate sources selecting the same cells, and the origin through
the loaders and both cache layers.

Translation is the strongest of these. The GPU only receives geometry arrays, so
arrays that match bit for bit after a shift to UTM mean the run matches too.

Four deliberate mutations are caught: skipping the source conversion, losing the
origin in the reorder, orienting normals with `float32` centroids, and removing the
reorder check's tolerance. The two orientation mutations survived the first version
of the test, which needed smaller cells and then a sheared mesh: rounding keeps each
coordinate's sign, so on a square grid the `float32` step between neighbours cannot
point backwards.

The old build stays the oracle. Moved to its origin first, a mesh still matches it
bit for bit, which also shows the origin shift changed nothing else in the build.

## Not verified

No solver run has used cells below Nysa's 1.7 m. The 5 cm figure comes from geometry
tests, so stability, bed elevation in `float32` and thin films at that size stay
open, as listed in [effort 25](../25-minimum-cell-size/plan.md). The run time limit
there still applies.

The clockwise handling is still covered only by unit tests.

## Other changes

`cell_edges` and `cell_neighbors` are gone, which saves 888 MB on the mosaic.
