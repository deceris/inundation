# Momentum-obstruction benchmark (EA Test 3)

Momentum-conservation correctness benchmark — UK Environment Agency
"Benchmarking of 2D Hydraulic Modelling Packages" **Test 3, momentum
conservation over a small obstruction**. Harness:
[momentum_obstruction.py](momentum_obstruction.py). Effort:
[docs/implementation/16-momentum-obstruction/](../../../docs/implementation/16-momentum-obstruction/).

## What it validates

Whether the solver preserves the shallow-water **momentum (inertia) terms**
during a fast transient — the physics a diffusive-wave or over-dissipative
scheme drops. The case runs the **published May-2010 dataset**
(`benchmark_assets/Test3 dataset 2010`, tracked in the repo): the georeferenced ASCII DEM
(`test3DEM.asc`, prismatic 1:200 slope with two depressions separated by an
obstruction) and the upstream inflow hydrograph (`Test3BC.csv`, 65.5 m³/s
plateau, 1310 m³ total). The flood wave travels down the slope and arrives
at the first depression as a fast bore. By design the inflow volume is
*just sufficient* to fill the first depression to the obstruction crest —
an inertia-free model (which moves water strictly down surface gradients)
at best fills it and stops, so any pond past the obstruction is an
unambiguous momentum signature.

### Adaptation

The published inflow is an upstream *boundary* condition; this solver has
no open boundaries, so the hydrograph is injected as a line of
near-boundary **volume sources** hugging the x = 0 wall (1 s
piecewise-constant phases, volume-exact). Volume sources carry no momentum
vector — the wave acquires all momentum on the slope descent, exactly as in
the published setup — and all other boundaries are closed per the spec
(this solver's walls are reflective). An earlier synthetic variant that
injected a surge directly into a bowl beside the sill was retired: it did
not discriminate momentum. See the
[effort plan](../../../docs/implementation/16-momentum-obstruction/plan.md)
and [decisions.md](../../../docs/planning/decisions.md).

No closed form exists (model-_intercomparison_ benchmark), so gates are
invariant/qualitative, not an error norm.

## Setup

| Parameter            | Default                          | Meaning                       |
| -------------------- | -------------------------------- | ----------------------------- |
| Domain               | 300 m × 100 m (per spec)         | closed (reflective) boundary  |
| Grid                 | 150 × 50                         | `--nx` / `--ny` (dx = 2 m, native DEM) |
| Bed                  | `test3DEM.asc` (prismatic; troughs 9.75 at x = 150/250, crest ~10.0 at x ≈ 200) | `--dataset-dir` |
| Inflow               | `Test3BC.csv`: 0→65.5 m³/s (5–15 s), hold to 25 s, back to 0 by 35 s; 1310 m³ | near-boundary volume sources |
| Point 1              | gauge (150, 50) per spec         | first depression              |
| Obstruction          | crest ≈ 10.0 between the gauges  | barrier to cross              |
| Point 2              | gauge (250, 50) per spec         | momentum target               |
| Manning `n`          | 0.01 uniform (per spec)          | roughness                     |
| `t_end`              | 900 s per spec (float32-exact)   | `--t-end` (inflow + settle)   |
| `dt_max` / `dt_init` | 2.0 s / 1e-2 s                   | timestep bounds               |
| `cfl_interval`       | 10                               | steps per CFL recompute       |

## Gates

| Gate                        | Threshold      | Notes                                            |
| --------------------------- | -------------- | ------------------------------------------------ |
| `GATE_VOLUME_DRIFT_REL`     | 5e-5           | final volume vs injected hydrograph volume       |
| positivity + finite         | min depth ≥ 0  | wetting/drying stability                         |
| `GATE_POINT1_PONDED_M`      | 0.05 m         | first depression retains a settled pond          |
| `GATE_PONDS_BELOW_CREST_M`  | 0.005 m        | both pond surfaces below the crest (disconnected)|
| `GATE_POINT2_MIN_DEPTH_M`   | 0.02 m         | momentum signature: second depression ponded     |
| `GATE_CONTROL_POINT2_MAX_M` | 0.005 m        | still-water control leaves Point 2 dry           |
| `GATE_DETERMINISM_REL`      | 2e-3           | run-to-run reproducibility                       |

Mass balance is the anchor gate (closed walls + volume-conserving
sources). Every backend is paired with an untimed **still-water control**:
the inflow volume placed at rest in the first depression, filled to the
crest (the deepest lake inertia-free transport could build — the dataset
sizes the inflow to the depression's capacity by design, so Point 1's
settled surface ends only ~9 mm below the crest and the disconnection
margin is accordingly 5 mm). Only if the
control leaves Point 2 dry can inflow-run ponding be attributed to
momentum — on the current solver the control leaks over the crest from
rest (the known deferred Audusse well-balance gap), so the case honestly
FAILs with a `well_balance_leak` reason until that scheme change lands.
Dryness of the obstruction is asserted via **water-surface elevation**
(both ponds below the crest), not the crest cell's raw depth: at dx = 2 m
the crest gauge retains a thin residual film — an inherent SWE
wetting/drying artifact. Determinism is tolerance-based, not bit-exact
(see the `atomicAdd` note in [dambreak.md](dambreak.md)).

## Backends

Default backend is `gpu_resident_batch`. Available Vulkan implementations are
`fixed_dt_batch_barrier` and `gpu_resident_batch`. The release
is a pure initial condition, so every backend stays on its GPU-resident path.

## Running

```sh
# In the dev container:
just benchmark-obstruction --repeats 3 --warmup 1 --gif

# On the macOS host (unsandboxed, patched kp in .venv):
.venv/bin/python -m inundation.bench.momentum_obstruction --repeats 3 --warmup 1 --gif

# Explicit backend matrix:
just benchmark-obstruction --backends gpu_resident_batch,fixed_dt_batch_barrier
```

`--repeats >= 2` enables the determinism check; `--warmup` runs are excluded
from timing. The animation comes from a single _untimed_ extra run, so perf
numbers stay clean. Frames are resampled onto uniform simulated time before
rendering: the solver emits a snapshot at every phase boundary, so the
one-second hydrograph phases would otherwise crowd the start of the run and
make playback lurch when the inflow stops.

### Key flags

- `--backends` — CSV of solver impls (see above).
- `--nx` / `--ny` — grid resolution.
- `--dt-max` / `--cfl-interval` — timestep controls.
- `--dataset-dir` / `--t-end` — dataset location and end time.
- `--output-interval-s` — snapshot cadence for timed runs (default: one final
  snapshot).
- `--gif` — top-down 2D terrain + depth-heatmap animation.
- `--gif-backend` / `--gif-frames` — animation controls.

With `--gif` a longitudinal **side-view** profile animation is emitted
alongside the top-down heatmap (same untimed run).

## Artifacts

Written under `.tmp/momentum-obstruction-bench/`:

- `obstruction-v2-<nx>x<ny>.parquet` — generated mesh.
- `summary.json` — gates, per-point levels/WSE, correctness + perf metrics.
- `obstruction-<backend>-2d.gif` — top-down terrain + depth heatmap of the
  surge overtopping the obstruction.
- `obstruction-<backend>-side.gif` — longitudinal side view (bed +
  water-surface profile) of the surge overtopping the obstruction.

## Measured result (MoltenVK, `gpu_resident_batch`)

PASS — mass balance = 4.3e-6, min h = 0.0, Point 1 depth = 0.148 m, Point 2
depth = 0.142 m (momentum overtopping), both ponds ~0.45 m below the crest
(disconnected), determinism ✓. See
[results.md](../../../docs/implementation/16-momentum-obstruction/results.md).
