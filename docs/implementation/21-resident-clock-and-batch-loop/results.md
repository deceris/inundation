# Resident Clock and Batch Loop — Results

**Status: done** (2026-10-26).
Plan: [`plan.md`](plan.md).

## Baseline probe

Setup: 40 × 40 grid of 1 m quads, flat bed, closed walls, uniform initial
depth h0, Manning n 0.03, one point source of Q m³/s in a central cell,
600 s simulated, `gpu_resident_batch`, `cfl_interval=10`, MoltenVK on M1 Max.
The same run starts at t0 = 0 and at t0 = 70 000 s (`resume=True`). Error is
(volume gained − Q·600) / (Q·600).

| Q (m³/s) | h0 (m) | Error, t0 = 0 | Error, t0 = 70 000 s |
| --- | --- | --- | --- |
| 0.01 | 0.5 | −4.80 % | −1.40 % |
| 1.0 | 0.5 | −0.01 % | −0.67 % |
| 0.01 | 0.05 | −0.00 % | +1.57 % |
| 1.0 | 0.05 | −0.00 % | −1.45 % |

- **Clock.** At t0 = 0 three of four cases are exact to 0.01 %; at
  t0 = 70 000 s the same physics is off by −1.5 % to +1.6 %, either sign. Only
  the absolute time differs, so this is the float32 device clock.
- **Float32 state floor.** Q = 0.01 into h0 = 0.5 loses 4.8 % even at t0 = 0:
  per-step increments (~4e-4 m in the source cell, far less elsewhere) are
  small against the 6e-8 m spacing at 0.5 m. This is the float32 floor
  described in the architecture overview; it is out of scope here.
- Walls are not the cause: the same basin with no source and a sloshing hump
  keeps volume to −1.9e-7 over 600 s.

The Nysa 24 h run's −0.95 % is consistent with a mixture of both effects; the
full run is repeated after the change.

## After the change

### Probe sweep

Same setup as the baseline, with the integer tick clock and per-step CFL.

| Q (m³/s) | h0 (m) | Error, t0 = 0 | Error, t0 = 70 000 s |
| --- | --- | --- | --- |
| 0.01 | 0.5 | −4.81 % | −4.80 % |
| 1.0 | 0.5 | −0.01 % | −0.01 % |
| 0.01 | 0.05 | −0.00 % | −0.00 % |
| 1.0 | 0.05 | −0.01 % | −0.01 % |

The start time no longer matters: the two columns agree to 0.01 % or better.
What remains is the float32 state floor (first row), which is unchanged and
out of scope. The walls check keeps volume to 1.2e-7 for the hump and exactly
for the flat basin.

### Gates

All run natively (MoltenVK, M1 Max).

- Lake at rest, variants A/B × friction minimal/production:
  `gpu_resident_batch` and `fixed_dt_batch_barrier` pass with errors
  ~2e-5 to 7e-5 m, as before. `fixed_dt_batch` blows up in all four
  cases; it does the same on unmodified `HEAD`, which is the known
  missing-barrier race recorded in `decisions.md`.
- Benchmark report (`inundation.bench.report`): dam-break, radial
  dam-break, floodplain depressions, momentum obstruction and flood
  propagation all PASS. No baseline change was needed.
- Unit tests: 230 pass, including the new GPU tests for the step-control
  kernel and the resident solver.

### Cost per step

Measured against unmodified `HEAD` on the same machine, same session.

| Case | Cells | `HEAD` | This change |
| --- | --- | --- | --- |
| Flat grid, 1 h simulated, 3 reps | 47 089 | 0.60–1.13 ms, CPU 3.7–4.4 s | 0.22–0.24 ms, CPU 1.6–1.8 s |
| Nysa, 6 sources, 10 min probe | 2.35 M | 2.37 ms (18.8 s) | 1.96 ms (15.5 s) |

The fixed per-step overhead drops from roughly 0.4–0.9 ms to under 0.2 ms, and
host CPU time halves. On the full Nysa mesh the kernels dominate, so the gain
is 17 %; making those kernels cheaper is the active-set work. With the
geometry cache warm, `prepare()` takes about 5 s; the 250 s in the original
run was the cache build.

### Full Nysa run (24 h simulated)

Same command as the original run (six sources, Q max 30 m³/s, 2 h ramp,
22 h hold), output to `flood-21/`.

| Metric | Original run | This change |
| --- | --- | --- |
| Wall time | 4 720 s | 4 855 s |
| Steps | — | 2 744 249 |
| Cost per step | 1.71 ms | 1.77 ms |
| System time | 390 s | 411 s |
| Peak RSS | 9.1 GB | 5.5 GB |
| Injected volume | — | 3 739 500 m³ |
| Missing at the end | 0.95 % (35 545 m³) | 1.26 % (47 187 m³) |

- **Speed.** Over the full run there is no gain; it is slightly slower. The
  10-minute probe's 17 % did not carry over, and the original figure comes
  from an earlier session, so this is not a controlled comparison. At 2.35 M
  cells the step is kernel-bound.
- **Mass.** The clock now advances by exactly the dt the physics used, so the
  scheduled volume is what the sources inject. The deficit grew from 0.95 % to
  1.26 %, so it does not come from the clock. Candidates are the float32 state
  floor and the positivity clamp. In the original run the clock error, which
  can have either sign, probably masked part of it.

### Where the Nysa deficit builds up

Volume in each 30-minute snapshot of the 24 h run (float64 sum of h·area)
against the scheduled inflow:

| Simulated time | Wet cells (h > 5 cm) | Deficit |
| --- | --- | --- |
| 2 h | 3 497 | −0.009 % |
| 6 h | 14 146 | −0.087 % |
| 12 h | 28 245 | −0.345 % |
| 18 h | 39 795 | −0.685 % |
| 24 h | 53 783 | −1.262 % |

The loss is negligible while the reach fills and grows with the wet area: in
the last half hour 3.8 % of that window's inflow is lost. A separate 2 h run
with `track_clamp_mass` reports zero clamp volume against a −15 m³ residual,
so the positivity clamp is not the cause in the filling phase. The full 24 h
run repeated with `track_clamp_mass` reports zero clamp volume against a
−47 174 m³ residual (−1.261 %, the same as the untracked run), as effort 19
found on EA Test 4.

## Conclusion

The clock fix works: injected volume no longer depends on start time, and
per-step overhead on small meshes drops about threefold. The Nysa deficit
does not come from the clock or the clamp. It grows with the wet area, like
the float32 state floor in the first probe row, so it belongs to the float64
or compensated-state work (roadmap priority 1), which needs a float64
reference to confirm. On the full Nysa mesh the step is kernel-bound; the
next speedup is the active set.
