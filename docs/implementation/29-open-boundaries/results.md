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

## Found by the review

An independent review of the branch found three defects, fixed in one commit on top of
`f609715`:

- **Outflow edges drew momentum in.** On `FreeOutflow` and `NormalDepth`, a flux that
  would draw water in had only its mass zeroed; the momentum flux of the outflow ghost
  stayed. Such an edge is now a wall for the whole flux. A new test, water leaving a
  `NormalDepth` outlet at 2 m/s, fails on the old kernel and matches a wall bit for bit
  on the new one.
- **Non-finite condition values were accepted.** `Stage(nan)` reached the kernels, and
  the run zero-filled the non-finite snapshot. Every condition value must now be finite.
- **Host backends ignored `capture_momentum_snapshots`** when called directly, returning
  depth only. They now raise, as the workflow already did.

All seven gate suites give the same summaries field for field before and after the fix,
wall times aside, and 551 unit tests pass. None of the gates has an outlet whose flow
turns back inward, so the fix changes none of their numbers.

## Departures from the plan

- `OPEN_WORDS` is 2, kind and value, not 3.
- The 3.5e-8 m³ tally resolution is a design figure and was not measured. Measured
  balances are the table above.
- Volume through an inflow is exact to 1e-7-level, not perfectly, because discharge is
  a float32 in the kernel.
- `GATE_BASIN_DEPTH` was tightened to 1 % and the gate runs at a 60 s window, once the
  window turned out to set the error.
- The real run is 72 h, for the reason above.

## Limits

- Boundaries are one-way between basins. Water cannot come back through a cut.
- `inflow_phases` has only run on channels. No real pair of Nysa basins has been cut
  and run, and only `NormalDepth` and `Wall` have run on the real mesh.
- A basin that stays dry until its inflow arrives pays for it. It steps at `dt_init`,
  0.01 s, until first water, which cost about 330 000 steps in the lower basin of the
  two-basin case. Results are not harmed.
- `energy_slope` is a user input. A wrong value shifts the `NormalDepth` stage.
