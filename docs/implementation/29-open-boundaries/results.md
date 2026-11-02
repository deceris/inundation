# Open Boundaries — Results

**Status: done.** A run can name stretches of the mesh edge and give each one, per
phase, a wall, free outflow, normal depth, a stage or a discharge inflow. The volume
through each stretch is measured. One-way basin exchange works through
`inflow_phases`. Two-way coupling is the next step on the
[roadmap](../../planning/roadmap.md).

Code is `867a1c4`, plus the report fix below. All figures are from MoltenVK on the
local machine.

## Regression with no open boundaries

Nysa, 24 h, base `7958247` against new `867a1c4`, back to back in both orders:

| Run | Commit | Steps | Wall |
| --- | --- | --- | --- |
| a1 | base | 685 289 | 463.6 s |
| b1 | new | 685 289 | 509.1 s |
| b2 | new | 685 289 | 497.7 s |
| a2 | base | 685 289 | 514.8 s |

All 59 depth columns are bit-identical in both a1 against b1 and a2 against b2. The
unfloored mesh gives 1 127 159 steps, unchanged, and injected equals in domain.

No wall-time cost can be claimed either way. The two base runs alone differ by 11 %,
which is larger than the 3 % between the means.

The six existing suites pass. Step totals and every metric match effort 28 exactly.

## Analytical gates

The new `open_boundary` suite, 10 cases, two repeats each, bit-identical between
repeats. All pass. Every case that reports a balance closes it to 2.5e-9 or better,
and clamp mass is 0.

| Case | Measure | Result | Gate |
| --- | --- | --- | --- |
| Uniform flow, 200 cells | Depth against Manning | 0.85 % | 1 % |
| Uniform flow, 400 cells | Depth against Manning | 0.38 % | 1 % |
| Uniform flow | Order | 1.17 | 0.9 |
| Uniform flow | Discharge error | 8.7e-9 | 1e-3 |
| Ritter through a cut | Difference from a longer channel | 0.0 m | 1e-3 m |
| Ritter, wall control | Same difference with a wall | 0.65 m | 0.1 m at least |
| Stoker, free outflow | Reflection, share of the jump | 1.1 % | 5 % |
| Lake at rest, stage | Level error | 6.4e-7 m | 1e-4 m |
| Lake at rest, stage | Largest speed | 3.3e-6 m/s | 1e-3 m/s |
| M1 backwater | Profile error | 0.43 % | 1.5 % |
| M2 backwater | Profile error | 1.32 % | 1.5 % |
| Two basins | Volume passed against received | 1.6e-9 | 1e-6 |
| Two basins | Depth against the whole channel | 0.67 % | 1 % |
| Two basins | Outflow against the whole channel | 0.79 % | 2 % |

Stoker with normal depth reflects 52 % and with a stage 138 %. Those conditions impose
something the bore does not match, so a reflection is the correct answer. They are
reported, not gated.

M2 converges at first order, 2.68, 1.32 and 0.63 % on 300, 600 and 1200 cells. The
error peaks in the drawdown at the outlet and is 0.4 % over the first 2500 m.

Uniform depth error on 100 to 800 cells is 1.95, 0.85, 0.38 and 0.17 %.

### The exchange window sets basin accuracy

Discharge is held constant over each output window, so the error at the cut follows
the window, not the mesh:

| Window | Error at the cut | Beyond 100 m |
| --- | --- | --- |
| 300 s | 2.4 % | 0.9 % |
| 60 s | 0.67 % | 0.24 % |

Use short windows at a basin boundary while the flood passes. The gate runs at 60 s.

## Nysa with an outlet

The plan asked for 24 h. After 24 h the flood is still 4.8 km from the lowest point of
the rim, 139.16 m at (406 850, 324 963), so the run became 72 h. The stretch follows
the north-east rim, with `NormalDepth(1e-3)`, against the same run with that stretch
as a `Wall`.

| | Outlet open | Outlet walled |
| --- | --- | --- |
| Steps | 2 373 925 | 2 373 925 |
| Injected | 11 515 500 m³ | 11 515 500 m³ |
| Left through the stretch | 4 858 136 m³ | 0 m³ |
| In domain at the end | 6 657 361 m³ | 11 515 497 m³ |
| Injected − left − in domain | 2.9 m³ | 2.7 m³ |
| Wet area, `h_max` > 5 cm | 10.29 km² | 14.99 km² |
| Deepest | 3.86 m | 5.73 m |

- Water reaches the stretch at 30 h. The runs are bit-identical in every snapshot
  to 29 h, and the 30 h snapshot is the first that differs.
- At the end the open run is deeper than the walled one by 1.3 µm at most, which is
  rounding. `h_max` agrees to 0.7 µm in the same direction. Where they differ, it is
  lower or equal, by up to 4.07 m.
- Outflow settles at 34.3 m³/s by 60 h, against 45 m³/s coming in, so the valley is
  still filling. The cumulative volume never falls.
- The balance gap is about 2.5e-7 of injected in both runs, so it is not the outlet's.
  The outlet's share is the 0.2 m³ difference, 4.8e-8 of the volume through it. The
  rest was not traced.

Step counts match because the time step is set by cells near the sources.

This shows the outlet behaves. It does not validate it: there is no observed
hydrograph, and the slope of 1e-3 is an assumption.

## Mutation check

14 kernel mutations, 13 caught by the tests. Several tests exist because a mutation
survived at first: the device probe of the ghost state, the drain test, the prologue
CFL test and the away-flow test. "Outflow can draw in" is caught only by the probe.

The survivor reads inactive blocks in the tally. A drying cell nearly always writes
zero in its last corrector, so the guard matters only on rare fast drying. It stays as
cheap protection and has no test.

## Found by the verification

`report.py` passes `--repeats` and `--warmup` to every suite and the new bench took
neither, so it exited 2 after 0.2 s and `report` failed. `just check` had stayed green
because the test only checked registration. The bench now takes both: repeats enforce
determinism and warm-up discards runs of the lake case. A test now hands each suite's
real parser those arguments. This is why the suite has determinism at all.

## Departures from the plan

- `OPEN_WORDS` is 2, kind and value, not 3.
- The 3.5e-8 m³ tally resolution is a design figure and was not measured. Measured
  balances are the table above.
- Volume through an inflow is exact to 1e-7-level, not perfectly, because discharge is
  a float32 in the kernel.
- `GATE_BASIN_DEPTH` was tightened to 1 % and the gate runs at a 60 s window, once the
  window turned out to set the error.
- The real run is 72 h, for the reason above.

## Real catchment pairs

Two touching MPHP pairs on the Nysa DEM, on `f609715`, as `~/projects/deceris/mphp`
holds them. Each pair ran clipped to its union outline as the control, then as two
basins exchanging across their shared edge through `inflow_phases`, with a 60 s window.
Both ran 8 h with a 1 h ramp.

Direction comes from the Shreve field `shr` in the MPHP stream segments, not only
from the names: Cielnica goes 10 to 15 where Korzkiew (5) joins, and Jasionka (2) and
Młynówka Bielicka (4) make 6.

The forcing is assumed, about 0.2 m³/s per km² of the upstream area, and Manning n
is a uniform 0.04. Neither pair is a headwater: 142 km² drains into Cielnica's 3.7.

| | Cielnica | Jasionka |
| --- | --- | --- |
| Mesh verdict | OK | OK |
| Cells, rectangle / outline | 13 839 / 5 794 | 10 885 / 4 412 |
| Inflow | 20 m³/s inlet, 10 m³/s Korzkiew | 3 m³/s inlet, 10 m³/s Młynówka |
| Control steps | 145 197 | 121 056 |
| Two-basin steps, upper + lower | 115 759 + 144 724 | 32 575 + 121 646 |
| Control wall time | 83 s | 72 s |
| Largest balance gap, share of volume in | 2.8e-8 | 2.9e-8 |
| Clamp mass | 0 | 0 |
| Volume passed = volume received | yes | yes |
| Lower basin depth against control, RMS | 1.6 cm | 2.0 cm |
| Upper basin depth against control, RMS | 19 cm | 18 cm |
| Largest depth difference | 0.98 m | 1.17 m |
| Outflow at 7-8 h, control / two-basin | 27.0 / 27.2 m³/s | 10.0 / 10.9 m³/s |
| Across the cut over 8 h, control / two-basin | 446 667 / 455 779 m³ | −676 / 13 797 m³ |

Mass passes and the lower basin is right when the flow really does go downstream. The
error is backwater. A cut outflow at normal depth draws the upper basin down, because
the level below the cut cannot reach it. In Cielnica this fades out by 600 m.

Jasionka fails in two places. At the confluence the Młynówka backs up across the cut,
and the one-way run leaves that side dry. At the east end a pond at 171.5 m sits across
the shared edge, so almost nothing crosses in the control. The one-way run drains that
pond through the cut, and the conveyance split then puts the water back at the river,
500 m from where it left. A single discharge for a whole cut cannot say where the
water crossed.

MPHP catchments end at confluences, so a river crosses their shared edge at a junction
by construction. That is the worst place for one-way exchange, so two-way coupling is
needed for these meshes, not only an improvement.

The mesh limited but did not block either run. The time step is set by small patch
cells where water is deep: 18 m² and 1.3 m deep in Cielnica, 50 m² and 1.8 m deep in
Jasionka. That gives a mean of 0.20 and 0.24 s.

The outline is a stair-stepped wall away from the open stretches. In Cielnica 2.6 km of
it is wet at the end, 1.4 km deeper than 0.5 m, so a real flood would cross it there.

## Limits

- Boundaries are one-way between basins. Water cannot come back through a cut.
- `FreeOutflow` and `Stage` have not run on a real mesh.
- A basin that stays dry until its inflow arrives pays for it. It steps at `dt_init`,
  0.01 s, until first water, which cost about 330 000 steps in the lower basin of the
  two-basin case. Results are not harmed.
- `energy_slope` is a user input. A wrong value shifts the `NormalDepth` stage.
