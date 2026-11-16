# 1D analytical dam-break benchmark

Tier 1 correctness benchmark (`validation-plan.md` §1, Go/No-Go #1).
Harness: [dambreak.py](dambreak.py). Effort:
[docs/implementation/13-analytical-dambreak/](../../../docs/implementation/13-analytical-dambreak/).

## What it validates

Two closed-form 1D cases on a synthetic rectangular channel mesh (no external
data files):

- **`stoker`** — wet-bed dam-break, exact Stoker (1957) solution. Checks the
  shock (bore) speed, the rarefaction fan, and the constant middle state.
- **`ritter`** — dry-bed dam-break, exact Ritter (1892) solution. The moving
  wet front makes this the wet-dry-boundary test: front position, no negative
  depths, and depth error across the fan.

The numerical depth profile is compared against the analytical `(h, u)` at
`t_end`; tolerances are printed in the report rather than left implicit.

## Setup

| Parameter            | Default                                               | Meaning                 |
| -------------------- | ----------------------------------------------------- | ----------------------- |
| Channel              | 1000 m × 10 m                                         | length × width          |
| Dam position         | 500 m                                                 | initial discontinuity   |
| Depths               | `h_up` = 10 m, `h_down` = 1 m (stoker) / 0 m (ritter) | up/downstream           |
| Grid                 | 500 × 5                                               | `--nx` / `--ny`         |
| `t_end`              | 20 s (float32-exact)                                  | `--t-end`               |
| Manning `n`          | 1e-4                                                  | frictionless floor      |
| `dt_max` / `dt_init` | 0.02 s / 1e-3 s                                       | timestep bounds         |
| `cfl_interval`       | 5                                                     | steps per CFL recompute |

## Gates

| Gate                    | Threshold | Notes                             |
| ----------------------- | --------- | --------------------------------- |
| `GATE_L1_REL`           | 0.05      | relative L1 depth error           |
| `GATE_FRONT_REL`        | 0.15      | Ritter wet-front radius error     |
| `GATE_VOLUME_DRIFT_REL` | 1e-5      | volume conservation (zero source) |
| `GATE_DETERMINISM_REL`  | 1e-5      | run-to-run reproducibility        |

Determinism is tolerance-based, not bit-exact: the Vulkan flux kernel scatters
edge contributions with floating-point `atomicAdd`, whose summation order is
not fixed across dispatches. A healthy run reproduces to ~4e-7 relative.

## Backends

Default backend is `gpu_resident_batch` on MoltenVK (macOS host). The
host-CFL-readback `fixed_dt_batch_barrier` does not develop the wave under
MoltenVK, so it is opt-in via `--backends` where supported. Available:
`fixed_dt_batch_barrier`, `gpu_resident_batch`.

## Running

```sh
# In the dev container:
just benchmark-dambreak --repeats 3 --warmup 1 --gif

# On the macOS host (unsandboxed, patched kp in .venv):
.venv/bin/python -m inundation.bench.dambreak --repeats 3 --warmup 1 --gif

# Explicit backend matrix:
just benchmark-dambreak --backends gpu_resident_batch,fixed_dt_batch_barrier
```

`--repeats >= 2` enables the determinism check; `--warmup` runs are excluded
from timing. `--gif` renders an _untimed_ extra run into a depth-profile
animation (numerical vs analytical), so perf numbers stay clean.

### Key flags

- `--cases stoker,ritter` — which cases to run.
- `--backends` — CSV of solver impls (see above).
- `--nx` / `--ny` / `--t-end` / `--dt-max` / `--cfl-interval` — case overrides.
- `--output-interval-s` — snapshot cadence for timed runs (default: one final
  snapshot).
- `--gif` / `--gif-backend` / `--gif-frames` — animation controls.

## Measured result (MoltenVK, `gpu_resident_batch`)

PASS — Stoker L1 = 0.0077, Ritter L1 = 0.0086, front_rel = 0.129. See
[results.md](../../../docs/implementation/13-analytical-dambreak/results.md).
