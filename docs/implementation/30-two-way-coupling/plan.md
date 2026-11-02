# Two-Way Basin Coupling — Plan

**Status: approved, step 1 done, see [results.md](results.md).** The figures under What
was tried come from a throwaway prototype on a patched copy of the package. Measured
results on the real code are in the results. Follows
[effort 29](../29-open-boundaries/results.md).

## Goal

Run a large area as separate catchment meshes, with water free to cross a shared border
in either direction. Effort 29 showed that one-way exchange fails at confluences, and
MPHP catchments end at confluences by construction.

One joined mesh is out: the work has to scale to a wider catchment, so coupling must
work across separate basin meshes.

## What was tried

Two touching MPHP pairs, 8 h, against the same area run as one outline. Each pair is one
pavé mesh cut in two along the catchment border, so border edges match exactly.

| Scheme | Window | Cielnica | Jasionka |
| --- | --- | --- | --- |
| Upper runs a stage, lower takes its measured signed flux | 60 s | Blows up, depths reach 2 km at 59 min | Not run |
| Same | 10 to 1 s | Stable to the time tested, clamp mass never reaches 0 | Not run |
| Higher side runs a stage, lower side receives | 60 s | Creates 375 149 m³ | Largest difference 1.6 cm |
| Same | 10 s | Creates 188 830 m³, outflow 22 % high | 0.1 cm |
| Both sides run an outflow-only level | 60 s | RMS 0.6 cm, largest 7 cm | RMS under 0.05 cm, largest 0.1 cm |

The first scheme takes water from cells that may not hold it, and a clamp at zero then
creates water. Clamp mass at 45 min was 162 390 m³ at 60 s and still 1 592 m³ at 1 s,
about 5 % of the water in by then. Windows of 1 s would take about 40 min against 83 s.

The second scheme's balance closed only because the created water was booked as
negative volume in transit. Its stage edges could draw water in.

## Scheme

At every shared edge, each basin sees the other's water level. Water can only leave.

1. At the start of a window, read each basin's level in its cell at every shared edge,
   and smooth it: half the new level plus half the level seen before.
2. Each basin runs the window with its shared edges set to the neighbour's smoothed level,
   as depth over its own bed. The face sits at the higher of the two beds.
3. An edge passes water out only where the basin's level is above the neighbour's. Where
   it is below, the edge is a wall, or an inflow of what it received.
4. The volume that left through each edge is measured. Next window it arrives through
   that edge in the neighbour, spread evenly over the window, carrying momentum.

Basins do not wait for each other inside a window, and each keeps its own time step.
Nothing is taken from a receiving basin, and every cubic metre is either in a basin or
on its way to one.

Clamp mass stayed 0 in every run of this scheme, so no basin passed more than its cell
held. That is measured on these two pairs, not proved.

## Changes

**Kernel.** A new condition, `OPEN_EXCHANGE`, not a change to `OPEN_STAGE`: lake at rest
depends on a stage passing water both ways. A third word per open edge carries the
discharge per unit width received. The ghost is a stage while the neighbour is lower,
and otherwise an inflow of what was received, or a wall. Without the wall the edge pushes
momentum into a nearly dry cell and the run stops: "CFL bound below one tick" at 13 min on
Cielnica. The mass flux is `max(Fh, 0) - received`.

**Bed step.** Border edges take the real step to the neighbour's bed through
`set_open_steps`, so the existing reconstruction sees only the depth over a crest. No
shader change.

**Tally.** Volume is needed per edge, not per stretch. Step 1 makes each border edge its
own stretch, as the prototype did, with no change to the tally. Whether that holds on
long borders is checked at step 3.

**Driver.** `run_two_basins` in `inundation/coupling.py` owns the window loop, the resume
calls, the volumes in transit and the balance. Each basin has its own `SWEWorkflow`, and
forcing stays per basin as `SimulationPhase`. Step 3 widens it to `run_coupled` over any
number of basins.

**Splitting.** A new step cuts one pavé mesh into basin meshes along the catchment
polygons and records the shared edges, so no matching by geometry is needed.

**Balance.** The result gives in, out, stored and in transit for the whole system.

`inflow_phases` stays. It suits a true outlet into a separate downstream run.

## Settled in step 1

Churn, momentum and the window are settled in [results.md](results.md). Delivery goes
through the edge, levels are smoothed, netting was dropped because it cannot conserve
per-edge volume, and the default window is 60 s.

## Not yet verified

- More than two basins, or two upstream basins into one. A confluence of three
  catchments meets at a point.
- Water crossing the whole length of a border. On Cielnica 54 of the 117 edges pass
  water, and Jasionka barely crosses.
- Fast flow across a border. The stage ghost ignores the level under supercritical
  outflow, so a level can only reach the upstream side when the flow is subcritical.
- Separate GPUs. Basins are independent inside a window, so the design allows it, but
  nothing has run that way and it is not in this effort.
- Whether splitting saves time. Two basins in sequence took longer than one mesh on both
  pairs. Cell-steps fall, 7 % and 27 % on effort 29's one-way runs.

## Gates

Existing behaviour:

- No-open-boundary Nysa stays bit-identical, 685 289 and 1 127 159 steps.
- The six analytical suites and `open_boundary` keep the step counts and metrics of
  effort 29.

New `basin_exchange` suite, against a single-mesh reference, each with repeats:

| Case | Measure | Gate |
| --- | --- | --- |
| Lake at rest across a border | Level error, largest speed | 1e-4 m, 1e-3 m/s |
| Ritter through a border | Depth against one channel | 1 % |
| M1 backwater through a border | Profile against one channel | 1.5 % |
| Flow reversing across a border | Depth against one channel | 1 % |
| Three basins, two into one | Depth against one mesh | 1 % |
| Every case | Balance, clamp mass, inward flow | 1e-6, 0, 0 |

Gate values are proposed and may be tightened once measured, as in effort 29.

Real meshes, all against one mesh over the same outline:

- Cielnica and Jasionka: RMS 1 cm or less at 8 h in both basins, largest 10 cm or less
  at 8 h, on the final code.
- The Nysa area split into catchment meshes against the single-mesh run, for accuracy
  and for wall time compared back to back in both orders.

Plus a mutation check of the new kernel code, as in effort 29, and `just check` green.

## Build order

Stop and report after each.

1. Done. `OPEN_EXCHANGE`, per-edge volumes and tests, with a two-basin driver. Churn
   settled by measurement.
2. The splitting step, from a pavé mesh and catchment polygons to basin meshes and links.
3. `run_coupled` over any number of basins, the new suite, and three basins.
4. The Nysa split, which answers whether this scales.

## Needed from the mesher

Nothing, if one pavé run covers the whole area and the basins are cut from it. Each pair
above came from a single `pave-m5` run over both catchments, cut by cell centroid along
the catchment polygons, so border edges match by construction. The cut follows cell
edges: on Cielnica and Jasionka it sits a median 6.3 and 3.6 m off the true border, and
at most 25 m.

If a wider area ever has to be meshed in separate pavé runs, both runs need the same
border line as a constraint, with matching vertices. That is not needed now.

## Other work

- **MUSCL (roadmap priority 2):** no conflict. The exchange stays first order at a
  border, which limits order locally. Reconstruction can be added at borders later.
- **Barriers:** on hold, as agreed. A barrier sitting on a catchment border would need a
  look when that design resumes.
- No sub-grid support, face hydraulic tables or 8-face cap are added.
