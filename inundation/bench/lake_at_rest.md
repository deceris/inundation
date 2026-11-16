# Lake-at-rest (C-property) benchmark

Tier 1 well-balancedness benchmark. Harness:
[lake_at_rest.py](lake_at_rest.py). Effort:
[docs/implementation/00-lake-at-rest/](../../../docs/implementation/00-lake-at-rest/).

## What it validates

The C-property (well-balancedness): a still lake over variable bathymetry must
stay exactly at rest. A still free surface `η = h + zb = const` over a smooth
Gaussian bed bump should generate _no_ spurious momentum — the pressure and
bed-slope source terms must cancel. This is the port-parity check that the
Vulkan flux/source discretization is well-balanced.

Metrics per run: `max_abs_hu` (peak spurious discharge), `max_abs_u` (peak
spurious velocity), and `volume_drift` (mass conservation). A CFL blow-up is
captured as a flagged result row instead of crashing.

## Setup

| Parameter            | Default                              | Meaning                  |
| -------------------- | ------------------------------------ | ------------------------ |
| Mesh                 | 128 × 128 quad grid on unit square   | `NX` / `NY`              |
| Bed                  | Gaussian bump, crest 0.8 m at center | or `--flat-bed` (zb = 0) |
| `t_end`              | 600 s                                | `--t-end`                |
| `output_interval`    | 60 s                                 | `--output-interval`      |
| `dt_max` / `dt_init` | 0.05 s / 1e-2 s                      | `--dt-max` / `--dt-init` |
| `cfl_interval`       | 10                                   | `--cfl-interval`         |

Two initial-surface variants and two friction regimes are swept:

- **variant A** (`η = 1.0`) — fully wet; **variant B** (`η = 0.5`) —
  partial dry (surface intersects the bump crest).
- **friction** `minimal` = 1e-4, `production` = 0.035.

Production-friction variants that exceed `max_abs_u` = 0.01 m/s are flagged as
an escalation.

## Backends

Vulkan family only (this is a port-parity / well-balancedness check, not a
timed perf benchmark):

- `fixed_dt_batch_barrier` — production reference (default).
- `gpu_resident_batch` — fully GPU-resident.

## Running

```sh
# In the dev container:
just benchmark-lake --backend all --variant all --friction all

# On the macOS host (unsandboxed, patched kp in .venv):
.venv/bin/python -m inundation.bench.lake_at_rest --backend gpu_resident_batch

# Trivial-rest control (flat bed) + determinism hash:
just benchmark-lake --flat-bed --hash
```

### Key flags

- `--backend` — `fixed_dt_batch_barrier` (default), `gpu_resident_batch`, or
  `all`.
- `--variant` — `A`, `B`, or `all`.
- `--friction` — `minimal`, `production`, or `all`.
- `--flat-bed` — zb = 0 everywhere; isolates a bed-slope/well-balance defect
  from a kernel-independent one.
- `--trace` — download hu/hv every output-interval chunk and print a
  `max_abs_hu` trace (diagnostic, slower due to resume overhead).
- `--hash` — print sha256 of final h/hu/hv for cross-run determinism.
- `--t-end` / `--output-interval` / `--dt-init` / `--dt-max` /
  `--cfl-interval` — run overrides.
