# SWASHES Analytic Benchmarks - Results

**Status: done** (2026-11-10). Plan: [`plan.md`](plan.md).

Measured on Apple M1 Pro, MoltenVK, host `.venv`. Reference: SWASHES 1.05.00
(`swashes==1.5.0` extra). `just benchmark-swashes --backends
gpu_resident_batch,baseline`, one run each.

## Cases

| Case | SWASHES | Cells | Run | Depth L1 | Discharge L1 | Mass, resident | Mass, baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bump, subcritical | 1 1 1 1 | 200 × 2 | 300 s | 7.4-7.6e-4 | 5.4-6.6e-4 | 2.8e-8 | 1.5e-4 |
| Bump, transcritical | 1 1 1 2 | 200 × 2 | 300 s | 3.9e-3 | 5.6-6.5e-4 | 3.2e-9 | 1.3e-4 |
| Bump, with shock | 1 1 1 3 | 200 × 2 | 600 s | 4.9e-3 | 3.4-3.5e-3 | 1.9e-8 | 1.1e-4 |
| Lake at rest, emerged bump | 1 1 1 5 | 200 × 2 | 100 s | 1.5-1.7e-8 | 5.9-7.6e-9 m²/s (max) | 1.4e-8 | 1.4e-8 |
| MacDonald long, subcritical | 1 2 1 2 | 200 × 2 | 6000 s | 5.5-6.1e-3 | 4.8e-3 | 1.9e-9 | 1.0e-4 |
| MacDonald short, transition + shock | 1 2 2 2 | 200 × 2 | 1500 s | 3.0e-3 | 2.8e-3 | 4.5e-8 | 8.2e-6 |
| Thacker 1D | 1 4 1 1 | 400 × 2 | 1 period | 3.5e-2 | - | 1.6e-8 | 6.8e-8 |
| Thacker 2D planar | 2 1 1 2 | 100 × 100 | 1 period | 2.3e-1 | - | 4.9e-9 | 7.8e-8 |

Ranges span the two backends; depth and discharge agree between them to the
second or third digit, so the resident boundary kernels match an independent
implementation. All 16 runs pass their gates. `gpu_resident_batch` takes
0.5-25 s per case, 10-20× faster than `baseline`.

## Findings

- **No new boundary kind needed.** The transcritical outlet must stop imposing
  its depth once the outflow is supercritical. Effort 29's `Stage` already
  does (it takes the interior state when `u_n >= c`), and gives depth L1
  3.9e-3, against 0.62 with the depth imposed throughout (measured with the
  earlier fixed-depth ghost).
- **Mass balance against throughput.** Steady cases pass their stored volume
  through 10-30 times, so mass error is relative to the largest of stored,
  final and inflowed volume. On the compensated resident depth the residual is
  at most 4.5e-8. On `baseline` it is 1e-4: plain float32 depth over ~1e5
  steps, not the boundary, whose per-edge volume is a compensated pair.
- **Thacker 2D** is resolution-limited, not biased: depth L1 0.43, 0.24 and
  0.13 at 50², 100² and 200² after one period (first-order shoreline
  smearing, measured in the first SWASHES port). Gated at the default 100².
- **Discharge metric is L1.** Its max is set by the single smeared cell at a
  shock (0.25 in the shock bump) and says nothing about the scheme elsewhere.

## Open boundaries on `baseline`

- The host flux and CFL kernels now build the effort 29 ghost state, and an
  outflow edge whose flux would draw water in is redone against the wall
  ghost, as on the resident path. `ecR == -2 - k` marks open edge `k`.
- `tests/unit/test_gpu_open_boundaries.py` runs seven of its cases on both
  backends: volumes aligned with snapshots, a sub-`dry_tol` trickle, the
  balance with water entering and leaving (1e-6 on baseline against 1e-7
  resident; a closed run of that channel drifts 1.2e-7 on baseline), outflow
  never drawing water in, and an outlet that draws nothing in acting as a
  wall. Baseline sums with float atomics, so the last compares within 1e-6:
  two closed baseline runs already differ by 3e-8. Removing the wall redo
  from the host kernel fails it by 4.5e-3.
- The other host backends refuse open boundaries at config time and in
  `run()`, as they refuse momentum capture.
- `baseline` now counts `steps_total`; it reported 0 before.

## Tests

- `tests/unit/test_bench_swashes.py` (CPU, skips without the extra): output
  parsing, references on the mesh's cell centres for every case, the 2D x/y
  transpose against the paraboloid bed formula, bump discharge, and the
  transcritical Froude profile.
