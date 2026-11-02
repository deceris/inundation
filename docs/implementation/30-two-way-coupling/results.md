# Two-Way Basin Coupling — Results

**Status: step 1 done, steps 2 and 3 in code, step 4 running.** Basins on separate meshes
pass water both ways across shared borders, through `run_coupled` in
`inundation/coupling.py`. `inundation/mesh/split.py` cuts one mesh along catchment
polygons. The figures below up to Limits were measured on `run_two_basins`, the two-basin
driver `run_coupled` replaced, and have not been rerun on it.

All figures are from MoltenVK on the local machine, on code that is not committed yet.
The base is `f609715`.

## Real catchment pairs

Each pair ran 8 h against the same area as one outline, with the forcing of
[effort 29](../29-open-boundaries/results.md#real-catchment-pairs), which is assumed.
The window is 60 s.

| | Cielnica | Jasionka |
| --- | --- | --- |
| Upper basin RMS at 8 h, one-way / two-way | 19 / 0.3 cm | 18 / under 0.05 cm |
| Lower basin RMS at 8 h, one-way / two-way | 1.6 / 0.2 cm | 2.0 / under 0.05 cm |
| Largest difference at 8 h, one-way / two-way | 0.98 m / 2.4 cm | 1.17 m / 0.1 cm |
| Largest difference at any time, 10 min sampling | 14.6 cm, at 20 min | 3.9 cm, at 1.8 h |
| Across the cut, control / two-way | 446 667 / 445 455 m³ | −676 / −672 m³ |
| Outflow volume, control / two-way | 376 979 / 373 555 m³ | 185 731 / 185 733 m³ |
| In transit at the end | 3 405 m³ | 5 m³ |
| Balance gap, share of volume in | 9.7e-9 | 8.7e-9 |
| Clamp mass | 0 | 0 |
| Steps, upper + lower | 115 765 + 143 152 | 40 137 + 121 198 |
| Control steps | 145 197 | 121 056 |
| Cell-steps against control | −8 % | −25 % |
| Wall time, two basins in sequence / control | 129 / 83 s | 98 / 72 s |

The real-mesh gate of the plan, RMS 1 cm or less and largest 10 cm or less at 8 h, is met
on both pairs.

Over all 48 snapshots, RMS peaks at 2.9 cm on Cielnica at 20 min and 0.3 cm on Jasionka at
1.8 h. At most 10 cells on Cielnica are off by more than 5 cm at any one time, and none on
Jasionka. The worst cell is often on a shallow wetting front, and can be 1.7 km from the
crossing. Those cells were not traced further.

Wall times are single runs, not back-to-back pairs, so they only show the size of the
cost. Two basins in sequence are slower than one mesh on both pairs, despite fewer
cell-steps. The reason was not measured.

GIFs of the final runs are in `~/projects/deceris/mphp/gifs/`, named `*-two-way.gif`.

## The bed step at a border

The first version of the driver left a 76 cm error on Jasionka at 1.7 h, and a 10 s window
did not change it, so it was not lag. A crest of 171.59 m sat in the upper basin over a
bed of 170.02 m in the lower one, with 8 cm of water above the crest. In one mesh the
reconstruction sees that 8 cm. A boundary edge had no bed step, so the exchange saw a
1.57 m ghost against a 1.65 m cell, and the upper cell filled about three times too fast
and spilled into a pond behind it.

Border faces now sit at the higher of the two beds, through `set_open_steps`. The shader
did not change. Jasionka's worst fell from 76 to 3.9 cm, and Cielnica's from 15.9 to
14.6 cm. A channel with a 1.5 m step at the border, 30 windows, now stays within 2 cm of
one mesh in all three cases: up over a crest, down over a crest and down a step. The up
case failed before the fix, at 2.1 cm against the 0.2 cm it reaches now.

## Churn

Before this step cut discharge on Cielnica swung from −3 to 42 m³/s against a near-steady
flow. Three options were tested on a prototype at 60 s:

| Delivery | Cut discharge late in the run | Largest at 8 h |
| --- | --- | --- |
| Source in the cell, level as read | −3 to 42 m³/s, std 11.46 | 7 cm |
| Source in the cell, smoothed level | 18 to 22 m³/s, std 0.96 | 6.1 cm |
| Through the edge, smoothed level | 16.9 to 22.8 m³/s, std 1.32 | 3.1 cm |

- **Delivered through the edge.** It carries momentum and keeps the error near the cut
  lower, at a slightly larger swing.
- **Smoothed level.** The level seen is half the new reading plus half the level seen
  before, `BORDER_LEVEL_WEIGHT`. That is an exponential average, not the mean of the last
  two readings.
- **Anticipating the level the neighbour will reach** made things worse: upper RMS 3 cm.
- **Netting each edge's two volumes** was dropped. It cannot conserve per edge.

Gross volume on each edge is still about twice the net: 883 937 m³ down and 438 482 m³
back on Cielnica, for a net 445 455 m³. The total across the cut is steady, so this is
two-way passage at the window scale, not a swing. It does not show up in depth.

## Window

Cielnica on the final code, 8 h:

| Window | Upper RMS at 8 h | Largest at 8 h | Worst at any time | Outflow | In transit | Wall |
| --- | --- | --- | --- | --- | --- | --- |
| 300 s | 0.5 cm | 2.2 cm | 41 cm | 361 440 m³ | 18 765 m³ | 101 s |
| 120 s | 0.3 cm | 4.0 cm | 24 cm | 371 038 m³ | 6 781 m³ | 109 s |
| 60 s | 0.3 cm | 2.4 cm | 14.6 cm | 373 555 m³ | 3 405 m³ | 129 s |
| 30 s | 0.3 cm | 2.3 cm | 11.6 cm | 374 755 m³ | 1 651 m³ | 141 s |

The end state barely depends on the window. The window sets how far a passing flood lags,
and the volume in transit grows with it. 60 s stays the default. It is what the gate uses,
and halving it costs 9 % of wall time for 3 cm of a brief worst case.

The ramp is cut into window-long steps, so volume in differs slightly between rows, 810 450
to 814 500 m³. The control used a 60 s ramp. That muddies the worst case at other windows:
Jasionka at 10 s shows 11 cm at 40 min where 60 s shows 3.9 cm at 1.8 h, in cells that are
behind on a front. That the ramp causes it is likely, not verified.

## Regression with no border

On the final code, Nysa 24 h gives 685 289 steps, and all 63 depth columns are
bit-identical to the base in both a1 and a2. Before the bed-step change a full a1 b1 b2 a2
sequence gave the same result in both orders. Wall times were 656, 609, 532 and 488 s, a
steady fall in run order, so no wall-time claim can be made either way.

All seven suites pass, and every step count and metric in their summaries matches effort
29 exactly. `just check` passes with 566 tests.

The second unfloored Nysa mesh, 1 127 159 steps, was not rerun.

## Mutation check

18 mutations of the kernel, solver and driver code. 15 are caught by the tests, one hangs
and two survive.

- **Hangs: the ledger is not reduced after delivery.** Inflow grows without limit and the
  test never ends. It was stopped at 400 s, and it counts as found, not as caught. The
  first run had no limit and sat for 9 hours, which is why this list has a limit now.
- **Survives: no smoothing.** No channel test churns, so nothing depends on it there. The
  evidence for smoothing is the Cielnica run under Churn. Step 3's suite should cover it.
- **Survives: negative depth allowed.** A negative ghost depth acts like zero, because
  the reconstruction clips it and `safe_vel` returns 0 at that depth. The guard stays, as
  it keeps the `sqrt` in the ghost defined.

"Exchange edge not kept live" survived at first. It matters when a trickle is too thin to
wet a cell, and without the guard that volume is lost. A test now covers it.

## Departures from the plan

- Volume per edge uses one single-edge stretch per border edge, with no tally change.
- Delivery is through the edge, not a source, and levels are smoothed. See Churn.
- Border faces take the real bed step, which the plan did not foresee.
- The balance report does not check that inward flow on an exchange edge is 0. A device
  probe test covers it directly, in `tests/unit/test_gpu_open_ghost.py`.
- The driver takes two basins. `run_coupled` over any number is step 3.

## Steps 2 and 3, in progress

Splitting takes about 23 s on the 2 339 765 cell Nysa mesh and keeps every cell. MPHP level
3 gives 7 basins and 11 228 border edges, level 4 gives 29 and 33 198. Border edges are now
found by an exact lookup, and a dry basin that nothing reaches is not run.

`inundation/bench/basin_exchange.py` runs every case whole and split. **4 of its 8 cases
miss the proposed depth gate of 1 %**, so it is not in the report yet:

| Case | Largest depth difference, share of the deepest | Gate |
| --- | --- | --- |
| Lake at rest | Level and speed at rounding | Pass |
| Backwater | 0.45 % | Pass |
| Up over a crest, down over a crest | 0.20 % each | Pass |
| Reversal | 1.1 % | Fail |
| Confluence of three | 3.9 % | Fail |
| Down a step | 5.3 % | Fail |
| Ritter | 11.9 % | Fail |

These are single worst cells at a 10 s window. Whether they are front lag or a defect has
not been traced, and the gate was not changed. Balance is within 1e-6, clamp mass is 0
and repeats are identical in all 8.

On Nysa the 24 h flood never reaches a border at level 3, so that split passed 0 m³ and
measures overhead only: 852 s against 572 s for one mesh, in a single run, with 33 % fewer
cell-steps. 843 s of it is inside the solver's run calls. The cause has not been measured.

## Limits

- Not run: more than two basins, a confluence of three catchments, water crossing the
  whole length of a border, supercritical flow across a border, separate GPUs.
- Whether splitting saves time is still open, and so far it does not.
- Clamp mass 0 is measured on these two pairs, not proved.
- One stretch per border edge may be slow on long borders. It is checked at step 3.
- The exchange is first order at a border.
- The forcing and Manning n are assumed. These runs compare against the same solver on
  one mesh, not against observations.
