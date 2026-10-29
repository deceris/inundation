# Solver Accuracy Fixes — Results

**Status: in progress** (2026-10-29): the Courant sweep is running; its
table and the final `CFL_DEFAULT` are still to be written. Plan: [`plan.md`](plan.md).

All measurements native on M1 Max with MoltenVK 1.4.2. GPU unit tests run on
the 32 768-cell Hilbert-ordered valley (`tests/unit/_valley.py`).

## Reference parity

`tests/unit/_cpu_step.py` is a float64 mirror of one Heun step. Eight fixed
steps of 2⁻¹⁰ s on every backend (`gpu_resident_batch` gather and atomic,
`fixed_dt_batch_barrier`) match it to 2e-6 m in depth and 2e-5 in momentum on
a wet lake with a current, a dam break onto a dry slope, a draining sheet and
dry cells carrying momentum; the prologue CFL bound matches to 1e-4 relative.
Before the fixes the dam break and every CFL bound failed (the bound differed
by 50 % on the dam break, where the front edges now count 2c).

A dry cell's momentum is exactly zero after one step on every case; the
active-set tests that encoded the old halving now assert the clearing.

## Bed datum

Maximum difference from the reference after 64 steps, `gpu_resident_batch`:

| Bed datum | Case | Depth before | Momentum before | Depth after | Momentum after |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 m | dam break | 2.8e-6 | 2.4e-7 | 2.8e-6 | 2.4e-7 |
| 0 m | draining | 5.4e-8 | 9.2e-8 | 5.4e-8 | 9.2e-8 |
| 300 m | dam break | 2.6e-6 | 2.5e-6 | 2.6e-6 | 3.4e-7 |
| 300 m | draining | 8.9e-7 | 1.6e-6 | 5.9e-8 | 1.3e-7 |
| 3000 m | dam break | 1.8e-4 | 3.0e-5 | 3.2e-6 | 3.3e-7 |
| 3000 m | draining | 3.1e-6 | 6.9e-6 | 5.4e-8 | 1.2e-7 |

With per-edge bed steps the datum no longer shows. The parity test now runs
the dam break and the draining sheet at a 300 m datum for 64 steps and holds
them to the datum-0 tolerances (5e-6 m, 6e-7).

## Mass balance

Closed valley, `track_clamp` on, error = (final − initial − injected − clamp
mass) / final volume, float64 from `download_h_precise()`:

| Case | Steps | Plain update | Compensated, fast math on | Compensated, fast math off |
| --- | ---: | ---: | ---: | ---: |
| Dam break 600 s | 7 283 | −1.5e-6 | −8.2e-8 | +2.6e-10 |
| Draining sheet 600 s | 3 087 | −7.4e-7 | −2.8e-9 | (within noise) |
| Wetting from a source 600 s | 5 104 | +4.5e-6 | −5.1e-7 | +7.2e-10 |

The plain drift has one sign per case and grows with the step count; the
compensated residual is the float32 rounding of the per-cell flux sums. The
gather and atomic paths agree. The middle column is MoltenVK's default: its
fast math reassociates the two-sum, and the `precise` qualifier does not stop
it in the on-demand mode. Fast math off costs nothing measurable: 0.293 vs
0.297 ms per step on the 32 k valley, active set off.

Forcing a wall edge's mass flux to exactly zero took the draining sheet from
+2.2e-8 to −2.8e-9 (fast math on); the mirrored Riemann problem had left a
flux of a few ulp with the sign of the flow.

Unit test gates: compensated < 3e-9, plain > 1e-7, both on these three cases.

## Gates

Each step's commit was run through the five report suites (two repeats, one
warm-up) from a worktree snapshot; all passed at every step. Headline
metrics, `gpu_resident_batch`:

| Suite | Metric | Before | Fronts | Bed steps | Compensated | Robustness |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dam break (Ritter) | volume drift rel | 1.6e-7 | 1.6e-7 | 1.6e-7 | 2.1e-11 | 2.1e-11 |
| Dam break (Ritter) | L1 rel | 8.62e-3 | 8.63e-3 | 8.63e-3 | 8.63e-3 | 8.63e-3 |
| Radial dam break | volume drift rel | 3.5e-8 | 3.5e-8 | 3.5e-8 | 1.2e-11 | 1.2e-11 |
| Floodplain depressions | volume drift rel | 1.0e-4 | 1.0e-4 | 1.5e-4 | 2.2e-9 | 2.2e-9 |
| Floodplain depressions | steps | 215 469 | 213 323 | 213 328 | 213 321 | 213 321 |
| Momentum obstruction | volume drift rel | 9.3e-8 | 9.3e-8 | 9.3e-8 | 2.5e-8 | 2.5e-8 |
| Momentum obstruction | point 2 depth [m] | 0.04694 | 0.04714 | 0.04714 | 0.04714 | 0.04714 |
| Flood propagation | volume drift rel | 2.6e-4 | 2.6e-4 | 2.6e-4 | 2.1e-8 | 2.1e-8 |
| Flood propagation | arrival err rel | 6.26e-2 | 6.26e-2 | 6.26e-2 | 6.26e-2 | 6.26e-2 |

The physical metrics are unchanged within the gates' resolution; the mass
drift falls by four orders of magnitude with the compensated update. The
robustness column repeats the compensated one (no numerical change).

## Courant sweep

Running at the time of writing: full suite at CFL 0.45, 0.6, 0.8 and 1.0
(`gpu_resident_batch`, `cfl_safety` 1.0). 0.45 passed every suite; 0.6 had
passed dam break, radial dam break and floodplain depressions. Logs and
summaries under `.tmp/bench-report/sweep-cfl-*`.

## Unit tests

378 pass (299 before this effort).
