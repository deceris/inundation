# Minimal Open-Boundary Forcing — Results

**Status: done** (2026-10-27). Plan: [`plan.md`](plan.md).

Measured on Apple M1 Pro, MoltenVK, host `.venv` (the dev container has no
Vulkan ICD).

## Steady flat-channel benchmark

`just benchmark-open-channel`: 100 m × 10 m frictionless flat channel, 2 m
cells, at rest at `H = 1 m`. West: inflow `q = 0.5 m²/s`. East: depth `1 m`.
Exact steady state is uniform `h = H`, `hu = q`. Run to 1200 s.

| Backend | Steps | max `\|hu−q\|/q` | max `\|h−H\|` [m] | mass error / stored volume |
| --- | --- | --- | --- | --- |
| `baseline` | — | 1.4e-3 | 2.5e-5 | 2.0e-6 |
| `gpu_resident_batch` | 22 690 | 1.4e-3 | 2.4e-5 | 2.5e-6 |

Gates: 1e-2, 5e-3 m, 1e-4. Both pass. `baseline` does not count steps
(pre-existing). The frictionless start-up transient rings for about 1000 s;
at 600 s `hu` is still 4 % off.

## Boundary-volume accumulator

A plain float32 per-edge sum lost about 0.8 m³ of the balance over 24 k steps,
because increments fell below half an ULP. Two alternatives were measured:

- Kahan summation does nothing under MoltenVK's default fast-math, which folds
  `(t − s) − y` to zero. Marking it `precise` restores it but made the whole
  flux shader about 50 % slower, walls-only runs included (interleaved timing:
  `gpu_resident_batch` 6.4–7.7 s → 10.3–12.2 s, `baseline` 21–24 s →
  32–34 s).
- Shipped: a two-level sum. A small `fine` partial is folded into `coarse`
  once it reaches 2⁻¹² of it. It needs no `precise`, and it cut the benchmark
  mass error from 2.8e-5 (`precise` Kahan) to 2.0–2.5e-6.

## Wall-only regression

- `tests/unit/test_open_boundaries.py`: an explicit all-wall table and the
  default produce bitwise-identical runs on both flux shader variants.
- Lake-at-rest (128², variant B, production friction, 5 s simulated), old and
  new code run back to back so background load hits both equally:
  `gpu_resident_batch` 7.70 s → 8.02 s, `baseline` 26.02 s → 26.02 s. Within
  this machine's run-to-run noise (load average about 5 from other apps;
  uninterleaved runs drifted 1.4–8×).
- Bitwise comparison against the pre-change code cannot be made: repeated
  pre-change runs of the same backend already differ (atomic float
  accumulation order). Max `|Δh|` stays 1.8e-7 and max `|hu|` about 6e-6
  before and after.

## Tests

`tests/unit/test_open_boundaries.py` has 22 tests. CPU-only validation:
interior edges, overlap, mask length, NaN/inf, negative depth, mask dtype,
frozen copy. GPU smoke tests on `baseline` and `gpu_resident_batch`: wall
parity, dry-channel inflow with no source kernel, prescribed depth, mass
balance, discharge sign, outflow cap with zero clamp mass. The GPU tests skip
without `kp` or a Vulkan device.

## Limit

The depth boundary is a fixed ghost state, not a characteristic condition:
outgoing waves partially reflect at it.
