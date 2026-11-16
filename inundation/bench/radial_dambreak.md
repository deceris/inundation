# Radial (circular) dam-break benchmark

Tier 1 correctness benchmark (`validation-plan.md` §1, Go/No-Go #1).
Harness: [radial_dambreak.py](radial_dambreak.py). Effort:
[docs/implementation/14-radial-dambreak/](../../../docs/implementation/14-radial-dambreak/).

## What it validates

A cylindrical column of water (radius `r_dam`, depth `h_in`) collapses over a
flat frictionless bed into a shallower still layer (`h_out`). The outward bore
and inward rarefaction spread _radially_, so this case exercises 2D flux
directionality and grid-orientation isotropy — a circular front on a Cartesian
mesh reveals mesh-imprint/anisotropy bugs a channel-aligned shock cannot.

No closed form exists. The reference is a fine-grid 1D axisymmetric
finite-volume solve (`bench/common.solve_radial_dambreak`, HLL + SSP-RK2,
r-weighted so the `r = 0` face flux vanishes). The coarse 2D solver's depth is
radially binned onto the same radii and gated against it.

## Setup

| Parameter            | Default                         | Meaning                    |
| -------------------- | ------------------------------- | -------------------------- |
| Domain               | 40 m × 40 m                     | square, center at (20, 20) |
| Dam radius           | 2.5 m                           | initial water column       |
| Depths               | `h_in` = 2.5 m, `h_out` = 0.5 m | inside/outside             |
| Grid                 | 200 × 200                       | `--nx` / `--ny`            |
| `t_end`              | 1.5 s (float32-exact)           | `--t-end`                  |
| Manning `n`          | 1e-4                            | frictionless floor         |
| `dt_max` / `dt_init` | 0.05 s / 1e-3 s                 | timestep bounds            |
| `cfl_interval`       | 5                               | steps per CFL recompute    |
| Reference            | 2000 cells                      | axisymmetric FV resolution |
| Sectors              | 12                              | angular bins for isotropy  |

## Gates

| Gate                    | Threshold | Notes                                      |
| ----------------------- | --------- | ------------------------------------------ |
| `GATE_L1_REL`           | 0.10      | relative L1 depth error vs reference       |
| `GATE_FRONT_REL`        | 0.05      | bore-front radius error                    |
| `GATE_ISOTROPY`         | 0.08      | front-radius spread across angular sectors |
| `GATE_VOLUME_DRIFT_REL` | 1e-5      | volume conservation (zero source)          |
| `GATE_DETERMINISM_REL`  | 1e-5      | run-to-run reproducibility                 |

The isotropy gate is the core 2D directional-correctness check: on a Cartesian
grid the circular front is slightly faster along the axes than the diagonals
(4-fold grid imprint). Determinism is tolerance-based, not bit-exact (see the
`atomicAdd` note in [dambreak.md](dambreak.md)); a healthy run reproduces to
~1e-6 relative.

## Backends

Same policy as the 1D case: default is `gpu_resident_batch` on MoltenVK.
The host-CFL-readback `fixed_dt_batch_barrier` does not develop the wave under
MoltenVK, so it is opt-in via `--backends` where supported. Available:
`fixed_dt_batch_barrier`, `gpu_resident_batch`.

## Running

```sh
# In the dev container:
just benchmark-radial-dambreak --repeats 3 --warmup 1 --gif --gif-2d

# On the macOS host (unsandboxed, patched kp in .venv):
.venv/bin/python -m inundation.bench.radial_dambreak --repeats 3 --warmup 1 --gif --gif-2d

# Explicit backend matrix:
just benchmark-radial-dambreak --backends gpu_resident_batch,fixed_dt_batch_barrier
```

`--repeats >= 2` enables the determinism check; `--warmup` runs are excluded
from timing. Both animations come from a single _untimed_ extra run, so perf
numbers stay clean.

### Key flags

- `--backends` — CSV of solver impls (see above).
- `--nx` / `--ny` / `--t-end` / `--dt-max` / `--cfl-interval` — case overrides.
- `--output-interval-s` — snapshot cadence for timed runs (default: one final
  snapshot).
- `--gif` — radial depth-profile animation (numerical vs reference).
- `--gif-2d` — top-down 2D depth-heatmap animation.
- `--gif-backend` / `--gif-frames` — animation controls.

## Artifacts

Written under `.tmp/radial-dambreak-bench/`:

- `circular-<backend>.gif` — radial profile vs reference.
- `circular-<backend>-2d.gif` — top-down depth heatmap.

## Measured result (MoltenVK, `gpu_resident_batch`)

PASS — L1 = 0.051, L2 = 0.089, front_rel = 0.017, isotropy = 0.000,
vol_drift = 3.8e-8. See
[results.md](../../../docs/implementation/14-radial-dambreak/results.md).
