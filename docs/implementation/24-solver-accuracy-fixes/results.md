# Solver Accuracy Fixes — Results

**Status: done** (2026-10-29). Plan: [`plan.md`](plan.md).

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
it in the on-demand mode. See "Cost per step" for what this costs.

Forcing a wall edge's mass flux to exactly zero took the draining sheet from
+2.2e-8 to −2.8e-9 (fast math on); the mirrored Riemann problem had left a
flux of a few ulp with the sign of the flow.

Unit test gates: compensated < 3e-9, plain > 3e-8, both on these three cases.

## Gates

Each step's commit was run through the five report suites (two repeats, one
warm-up) from a worktree snapshot; all passed at every step. Headline
metrics, `gpu_resident_batch`:

| Suite | Metric | Before | Fronts | Bed steps | Compensated | Robustness | Final (CFL 0.8) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dam break (Ritter) | volume drift rel | 1.6e-7 | 1.6e-7 | 1.6e-7 | 2.1e-11 | 2.1e-11 | 1.7e-11 |
| Dam break (Ritter) | L1 rel | 8.62e-3 | 8.63e-3 | 8.63e-3 | 8.63e-3 | 8.63e-3 | 8.63e-03 |
| Radial dam break | volume drift rel | 3.5e-8 | 3.5e-8 | 3.5e-8 | 1.2e-11 | 1.2e-11 | 2.0e-11 |
| Floodplain depressions | volume drift rel | 1.0e-4 | 1.0e-4 | 1.5e-4 | 2.2e-9 | 2.2e-9 | 3.1e-09 |
| Floodplain depressions | steps | 215 469 | 213 323 | 213 328 | 213 321 | 213 321 | 112 550 |
| Momentum obstruction | volume drift rel | 9.3e-8 | 9.3e-8 | 9.3e-8 | 2.5e-8 | 2.5e-8 | 2.5e-08 |
| Momentum obstruction | point 2 depth [m] | 0.04694 | 0.04714 | 0.04714 | 0.04714 | 0.04714 | 0.04736 |
| Flood propagation | volume drift rel | 2.6e-4 | 2.6e-4 | 2.6e-4 | 2.1e-8 | 2.1e-8 | 2.3e-08 |
| Flood propagation | arrival err rel | 6.26e-2 | 6.26e-2 | 6.26e-2 | 6.26e-2 | 6.26e-2 | 5.88e-02 |

The physical metrics are unchanged within the gates' resolution; the mass
drift falls by four orders of magnitude with the compensated update. The
robustness column repeats the compensated one (no numerical change); the
final column is the branch head (CFL 0.8, source bound, film damping): flood
propagation 126 449 → 76 221 steps, floodplain depressions 213 321 → 112 550.

## Courant sweep

Full suite at CFL 0.45, 0.6, 0.8 and 1.0 (`gpu_resident_batch`,
`cfl_safety` 1.0, `--cfl` on the report runner). "Before" is the committed
state ahead of the sweep: CFL 0.45 times the old 0.8 safety factor, 0.36
effective. Every value passes every gate.

| Suite | Metric | Before (0.36) | 0.45 | 0.6 | 0.8 | 1.0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Radial dam break | steps | 388 | 311 | 233 | 175 | 140 |
| Radial dam break | L1 rel | 5.11e-2 | 5.11e-2 | 5.11e-2 | 5.11e-2 | 5.12e-2 |
| Floodplain depressions | steps | 213 321 | 176 664 | 140 004 | 112 516 | 96 025 |
| Floodplain depressions | ponded storage | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| Momentum obstruction | steps | 8 505 | 6 911 | 5 314 | 4 112 | 3 398 |
| Momentum obstruction | point 2 depth [m] | 0.0471 | 0.0472 | 0.0473 | 0.0474 | 0.0475 |
| Flood propagation | steps | 126 449 | 107 127 | 87 929 | 73 437 | 64 799 |
| Flood propagation | arrival err rel | 6.26e-2 | 6.26e-2 | 5.88e-2 | 5.88e-2 | 5.88e-2 |
| Flood propagation | depth L1 rel | 2.59e-2 | 2.56e-2 | 2.52e-2 | 2.46e-2 | 2.40e-2 |
| Flood propagation | speed L1 rel | 1.27e-2 | 1.30e-2 | 1.35e-2 | 1.42e-2 | 1.48e-2 |

The 1D dam breaks are capped by `dt_max` and take 1 001 steps at every
value. Volume drift stays below 3e-8 throughout, which also bounds any clamp
mass; the gate suites do not track the clamp directly.

`CFL_DEFAULT` is now 0.8: 1.9x fewer steps than before on the EA cases, with
a margin below the positivity limit of 1.0 for terrain the sweep did not
cover (the Nysa case is the first thing to run at the new default). 1.0 is
available through `WorkflowConfig.cfl`.

### Sources and the planned dt

The resident path plans the next dt from the corrector's input state, before
the source kernel adds its water. At 0.36 that lag was hidden by the margin;
at 0.6 and above a point source of 0.2 m³/s into a dry 1 m² cell with the dt
capped at 1 s drove the cell and its neighbours below zero, and the clamp
created 3.8 %, 12.7 % and 22 % of the injected volume at 0.6, 0.8 and 1.0.
The gate suites never clamped because their sources are spread over many
cells. The source kernel now bounds the next dt by each source cell's
post-source state (`(|u| + 2c) × perimeter`, dry-bed speed on every edge);
the clamp mass is zero again at 0.8. A first version took 2c on every edge
and cost flood propagation 113 k instead of 73 k steps (its line source
sits in wet water); the bound now follows each neighbour's state (2c only
towards a dry one, |u| + c of the faster side otherwise), at 76 k steps.

## Lake at rest

Final code, `CFL_DEFAULT` 0.8, variants A/B × friction minimal/production,
600 s:

| Backend | max abs u [m/s] | volume drift rel |
| --- | ---: | ---: |
| `fixed_dt_batch_barrier` | 1.1e-5 to 1.5e-5 | 1.3e-7 |
| `gpu_resident_batch` | 8.3e-6 to 1.3e-5 | 0 |

The figures above are at 600 s. A same-session comparison at 120 s, variant A
with minimal friction, `gpu_resident_batch`, shows where the gain comes from:

| Commit | max abs u [m/s] |
| --- | ---: |
| Original | 6.7e-5 |
| Front fixes | 7.9e-5 |
| Bed steps | 2.5e-5 |
| Compensated depth | 1.3e-5 |
| Final | 1.2e-5 |

All four variants improve by about 5×. Bed steps give about 3× and compensated
depth about 2×.

## Parabolic bowl

`inundation/bench/parabolic_bowl.py` runs Thacker's planar solution in a
paraboloid, which wets and dries the shoreline every period and has an exact
answer. It is frictionless, so nothing hides thin-film errors. Three periods,
12 runs across nx 100 to 400 at CFL 0.8 and 1.0:

| Variant | Crashes | Depth L1, nx 400 | Speed L1, nx 400 | Peak film speed |
| --- | ---: | ---: | ---: | ---: |
| Dry-bed speeds, no film damping | 5 of 12 | 0.0915 | 0.2019 | 2.75 m/s |
| No dry-bed speeds | 0 of 12 | 0.0916 | 0.2020 | 2.09 m/s |
| Dry-bed speeds with film damping | 0 of 12 | 0.0855 | 0.1785 | 0.72 m/s |
| Time-scaled film damping (final) | 0 of 24 | 0.0863 | 0.1818 | 0.72 m/s |

The original code, at nx 200 and 400, gives the same errors as the second row.
The dry-bed speeds alone therefore bought no accuracy and added a crash: momentum
flowing into a cell barely above `dry_tol` reached 26 000 m/s and took the CFL
bound below one tick. None of the five gate suites reach this case. With the
Kurganov-Petrova film damping the errors are about 6 % and 10 % lower in the
final form, a rate like friction rather than a fixed factor per stage. Results
now agree from CFL 0.2 to 1.0: depth L1 0.0864 to 0.0862 at nx 400.

## Review fixes

- `FILM_DEPTHS` lives in `tuning.py`, the one source for the kernels and the
  float64 reference.
- `CFL_DEFAULT` 0.8 applies to `gpu_resident_batch` only. Host-driven backends
  reuse one dt for several steps and keep `CFL_HOST_DRIVEN` 0.45;
  `WorkflowConfig.cfl` defaults to `None`.
- The parabolic bowl is a sixth gate suite with a `just benchmark-bowl` recipe.
- The fast-math setting moved to import time. A resident run checks the
  error-free arithmetic on the device first and raises if it is broken.
- Strict pyright passes natively against the venv and in the container, after
  fixing 75 errors added earlier in this effort. In the container, ruff, pyright
  and markdown lint pass, and pytest gives 295 passed and 118 skipped, the GPU
  tests. The image has no Node.js and pyright's download fails on arm64, so Node
  was installed in the running container for that check; `just typecheck` still
  fails on a fresh arm64 image.
- All 413 tests pass natively with the GPU. Five more deliberate breaks of these
  changes are all caught.

Final gates, commit `0a0ec77`: all six suites pass. Lake at rest at 120 s reads
8.4e-6 to 1.2e-5 m/s on `gpu_resident_batch` and 1.3e-5 to 2.2e-5 m/s on
`fixed_dt_batch_barrier`.

## Atomic path repeatability

The dry-bed speeds first switched on abruptly at `dry_tol`. Float32 rounding
could then flip a flux and shift a front: 9 of 16 atomic runs differed by
5.9e-5 m from the gather path, and the tolerance in two test files was raised to
1e-4 to get past it. That was wrong, and the explanation left in those comments
was unverified and turned out false. The speeds now blend continuously over one
to two `dry_tol`: 0 of 24 runs diverge. `ATOL_H` is back to 1e-5.

A rarer transient remains on the atomic path only: up to 3.5e-5 m in a few
cells, intermittently at CFL 0.7 to 0.85, gone by the next snapshot. Its trigger
is not identified. The atomic comparisons allow at most 16 cells up to 5e-5 m.
The default gather path stays bit-identical.

## Deliberate breaks

Each of 19 fixes was broken in a scratch worktree. 15 were caught at once. Four
survived: a source skipping the low word, the volume ignoring the low word, the
wall flux, and the wet-neighbour term of the source bound. New tests pin the
first three, and the fourth was redundant and is gone. The new point-source
positivity test fails without the source bound. Without the film damping in the
update 5 of 6 bowl tests fail; the flux side of the damping is pinned by the
reference parity tests.

## Cost per step

Interleaved runs on the 32 k fully wet valley, active set off, eight rounds. The
fastest run of each is the figure to trust; two-repeat suite wall times on this
machine varied by up to 2× between identical runs, so the wall times in the
sweep above do not measure cost.

| Configuration | ms per step |
| --- | ---: |
| Original code | 0.284 |
| Plain, fast math on | 0.288 |
| Plain, fast math off | 0.305 |
| Compensated, fast math on, `precise` | 0.564 |
| Compensated, fast math off, `precise` | 0.598 |

The `precise` qualifier was almost all of that. MoltenVK gains nothing from it,
so it is dropped on macOS and kept elsewhere. Final code, interleaved:

| Configuration | Steps | Fastest ms per step | Fastest wall |
| --- | ---: | ---: | ---: |
| Original, effective CFL 0.36 | 2 813 | 0.192 | 0.541 s |
| Final, CFL 0.36 | 2 812 | 0.213 | 0.599 s |
| Final, CFL 0.8 | 1 266 | 0.226 | 0.286 s |

The final code costs about 11 to 18 % more per step and is 1.9× faster for the
same simulated time. This is a fully wet mesh; partly dry meshes were not timed
this way.

## Nysa 24 h

2.35 M cells, six sources, 30 m³/s peak, active set and deterministic gather on,
commit `0a0ec77`, clamp tracking on. Single runs, M1 Max with MoltenVK:

| | Effort 23 | CFL 0.8 | CFL 0.4 |
| --- | ---: | ---: | ---: |
| Water in domain | 3 692 242 m³ | 3 739 500 m³ | 3 739 500 m³ |
| Deficit against 3 739 500 m³ injected | 1.26 % | 0.0000 % | 0.0000 % |
| Clamp mass | not recorded | 0 | 0 |
| Steps | 2 744 246 | 1 168 669 | 2 337 299 |
| Wall time | 1 527 s | 830 s | 2 334 s |
| Peak host memory | 7.8 GB | 7.6 GB | 7.4 GB |
| Wet area, h_max above 5 cm | 5.207 km² | 5.246 km² | 5.246 km² |

The deficit was float32 rounding, as suspected: it is gone with the compensated
depth, with no clamp contribution.

Depth difference against the CFL 0.4 run, cells wet in either run:

| At 24 h | CFL 0.8 | Effort 23 |
| --- | ---: | ---: |
| Mean | 0.009 cm | 1.28 cm |
| 99th percentile | 0.14 cm | 23 cm |
| Maximum | 1.5 cm | 75 cm |
| Cells differing by over 10 cm | 0 | 1 489 |
| Flood extent mismatch | 0.0003 km² | 0.19 km² |

CFL 0.8 is therefore safe on this terrain: the largest difference at any of the
four checked times is 2.1 cm. CFL 0.4 is a converged-time reference, not the true
flood, so the effort 23 column shows how much results moved, not its error
against reality. At 18 h the largest such move was 1.48 m.

## Grid convergence

Observed orders at the final kernels of the film-damping commit:

| Case | Metric | Effort 19 | Now |
| --- | --- | ---: | ---: |
| Stoker | L1 | 0.79 | 0.79 |
| Ritter | L1 | 0.65 | 0.65 |
| Radial | L1 | 0.85 | 0.85 |
| Flood propagation | Depth L1 | 1.12 | 1.14 |
| Flood propagation | Speed L1 | 1.51 | 1.51 |
| Flood propagation | Arrival | 0.99 | 1.06 |

Flood-propagation volume drift now reads 4.8e-8, 5.0e-8, 2.3e-8, 1.3e-8 from
coarse to fine. It no longer grows under refinement.

The Ritter wet-front error rose from 12.9 % to 13.9 % at the gate resolution,
because the damping slows the thin tip, and it does not fall under refinement
(11.2 %, 12.7 %, 13.9 %, 13.6 % coarse to fine). The depth L1 is unchanged.

These orders predate the time-scaled film damping and the `precise` change.

## Unit tests

413 pass (299 before this effort).
