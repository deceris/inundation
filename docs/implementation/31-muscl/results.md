# Second-Order Reconstruction — Results

**Status: done, as an option.** `reconstruction="muscl"` on `WorkflowConfig` adds linear
reconstruction on `gpu_resident_batch`. `"first_order"` stays the default and uses the same
kernels. All figures are from MoltenVK on the local machine.

## Order

A smooth wave on a flat frictionless channel, against a 3200-cell run of the same scheme,
at CFL 0.4, L1 order between 100, 200, 400 and 800 cells:

| Scheme | Orders |
| --- | --- |
| First order | 0.86, 0.99, 1.16 |
| MUSCL | 2.02, 1.92, 2.05 |

Lake at rest over a bumpy bed is 6.5e-7 m against 6.2e-7 m, which is rounding.

## Analytical suites

At the default CFL 0.8, with MUSCL on by default in a throwaway copy of the package:

| Suite | Measure | First order | MUSCL |
| --- | --- | --- | --- |
| Stoker | L1 | 0.77 % | 0.15 % |
| Ritter | L1 | 0.86 % | 0.20 % |
| Ritter | Front error | 13.9 % | 13.4 % |
| Radial dam break | L1 | 5.1 % | 0.73 % |
| Radial dam break | Front error | 1.7 % | 0.64 % |
| Parabolic bowl | Depth L1, mean | 17.3 % | 3.1 % |
| Parabolic bowl | Speed L1, final | 37.4 % | 5.8 % |
| Parabolic bowl | Volume drift | 3.1e-9 | 1.4e-9 |

Floodplain, momentum obstruction, flood propagation, the dam breaks and the bowl pass all
their gates, clamp mass 0 where measured, deterministic. Flood propagation passes, and its
arrival times move later by up to 300 s of 3 660 s. The Ritter front barely moves because
the tip stays first order.

The bowl suite builds the solver directly, so a first run with the default flipped never
used MUSCL and showed the same numbers to every digit. The figures above are from a run
with `muscl=True` on the solver. The suites can only run MUSCL through such a throwaway
copy: no suite has a switch for it.

`open_boundary` misses 2 of its gates:

| Case | Measure | First order | MUSCL | Gate |
| --- | --- | --- | --- | --- |
| Uniform flow | Order | 1.17 | 0.78 | 0.9 |
| Stoker, free outflow | Reflection | 1.1 % | 5.9 % | 5 % |

Open edges stay first order. Uniform depth error itself falls, 0.38 to 0.32 % at 400
cells, so the order falls because the boundary's error now dominates. The sharper bore
reflects more from a first-order outlet. These gates still apply to the default, which
passes them, and are left as they are.

## The face depth limit

The first version created water: 29 % volume drift on the egg-box bed, at CFL 0.8. On a
steep bed a face can be far deeper than its cell, so a shallow cell is asked for more than
it holds, and the clamp makes up the rest. A face that differs from its cell's depth by
over 25 % now keeps the cell's own state. Drift on the egg box is 1.3e-9 against 4e-9
first order. Order and lake at rest are unchanged by the limit, and level stays consistent
across the edge when a side falls back. The value 25 % was chosen on the positivity bound
at CFL 0.8 and was not swept.

## Nysa 24 h

| | First order | MUSCL |
| --- | --- | --- |
| Steps | 685 277 | 711 685 |
| Wall time | 502 to 572 s | 822 s |
| In domain, injected 3 739 500 m³ | 3 739 500 m³ | 3 739 500 m³ |
| Wet area, `h` > 5 cm, at 24 h | 5.25 km² | 5.32 km² |
| Deepest at 24 h | 3.85 m | 3.79 m |

Depths differ by 5.5 cm RMS at 24 h, 1.02 m at most. There is no observation here, so this
shows that MUSCL is stable on real terrain and what it changes. It does not show that it
is closer to the truth. Wall time is a single run, with other jobs on the machine. The
cost is about 1.4 to 1.6 times per run on Nysa, and about 1.15 to 1.5 on the analytical
suites.

## Departures from the plan

- The face depth limit, which the plan did not foresee.
- A cell with a depth step above the limit is first order, so on steep shallow terrain
  MUSCL falls back. How much of Nysa does was not measured.
- The Green-Gauss gradient is exact on regular cells, not on every pavé cell. The accuracy
  on an irregular mesh was not measured separately.

## Mutation check

12 mutations of the reconstruction, and 8 are caught by the tests as first written. 3
survived and led to the dam-break test: without the limiters a 2 m dam break overshoots to
2.02 m, and gradients left over from the predictor triple the rise along it. It now
catches those. "Dry neighbour ignored" survives and changes no digit in these cases: the
25 % face limit already stops such a side. The guard stays, and nothing tests it.

The dry-neighbour rule is the only case where a guard has no test.

## Limits

- Walls, open edges and basin borders stay first order.
- Wetting fronts stay first order, by design.
- Basin coupling has not run with MUSCL.
- The default did not change, and changing it needs a new baseline.
- Measured on MoltenVK only.
