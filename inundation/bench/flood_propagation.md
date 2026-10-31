# Flood-propagation benchmark (EA Test 4)

Flood-front celerity benchmark — UK Environment Agency "Benchmarking of 2D
Hydraulic Modelling Packages" **Test 4, speed of flood propagation over an
extended floodplain**, run from the published May-2010 dataset. Harness:
[flood_propagation.py](flood_propagation.py). Effort:
[docs/implementation/18-flood-propagation/](../../../docs/implementation/18-flood-propagation/).

## What it validates

A dry, horizontal 1000 m × 2000 m plain is flooded through a 20 m breach in
the middle of its western wall. The published objective is the **celerity of
the advancing front** plus the transient depths and velocities at its leading
edge — pure wetting/drying front tracking, with no topography to hide behind
and no shock to capture.

Unlike the other EA cases this one carries a genuine reference. The inflow
sits on a closed wall, so reflection makes the half-plane problem identical
to a full-plane axisymmetric one at twice the discharge, and the published
gauges are laid out for exactly that: points 1–5 on the axis at radii
50–400 m, and point 6 at 424.26 m on the 45° diagonal, which makes it a
grid-orientation isotropy probe. `bench/common.solve_radial_inflow` solves
that 1-D problem on a fine grid with the same friction law and reports depth,
speed and first-wetting time per radius.

## Dataset

`benchmark_assets/Test4 dataset 2010` (tracked in the repo; seven required EA Test 2/3/4 input files are committed,
`--dataset-dir` to override):

| File              | Use                                     |
| ----------------- | --------------------------------------- |
| `Test4BC.csv`     | inflow hydrograph, minutes vs cumecs    |
| `Test4output.csv` | 6 output points                         |

There is deliberately **no DEM** — the spec fixes the ground at elevation 0
throughout, so the flat bed is exact rather than an approximation, and this
case carries no well-balancedness exposure.

## Setup

| Parameter            | Default                | Meaning                                  |
| -------------------- | ---------------------- | ---------------------------------------- |
| Domain               | 1000 m (E–W) × 2000 m  | closed (reflective) walls, bed z = 0     |
| Grid                 | 200 × 400              | `--nx` / `--ny` (dx = 5 m, spec value)   |
| Manning `n`          | 0.05                   | spec value, uniform                      |
| Inflow               | 20 m line at (0, 1000) | peak 20 m³/s, ~5 h base, 285 000 m³      |
| Initial condition    | dry bed                | spec                                     |
| `t_end`              | 18 000 s (5 h)         | `--t-end`                                |
| Probe time           | 7 200 s (2 h)          | `--probe-time`, profile comparison       |
| `dt_max` / `dt_init` | 2.0 s / 1e-2 s         | timestep bounds                          |

The published inflow is a boundary condition; this solver has no open
boundaries, so it enters as four equal-split volume sources on the first cell
column along the published line. The hydrograph is applied as 60 s
piecewise-constant phases sampled at midpoints, which integrates the
published piecewise-linear curve exactly (every breakpoint is a whole minute)
and keeps the default backend on its GPU-resident path. It ends exactly at
`t_end`, so there is no settle tail — the run finishes on the recession limb.

### Two runs per backend

- **Scored run** to 5 h, snapshots every 60 s → mass balance, front-arrival
  times at the six gauges, determinism.
- **Untimed probe run** to `--probe-time` → its final state carries `h`, `hu`
  and `hv`, giving depth *and velocity* against the reference. The scored run
  keeps depth snapshots only, so velocity needs its own run.

### Validity window

The reference describes an unbounded plain. Every nearest wall is 1000 m from
the source and the front reaches them at ~189 min, so profile comparisons
must stay earlier than that; the harness refuses a `--probe-time` past
180 min. The 5 h run is still scored, on invariants and arrivals.

## Gates

| Gate                      | Threshold | Notes                                          |
| ------------------------- | --------- | ---------------------------------------------- |
| `GATE_VOLUME_DRIFT_REL`   | 5e-5      | final volume vs injected (closed, dry start)   |
| positivity + finite       | min h ≥ 0 | wetting/drying stability over dry bed          |
| `GATE_ARRIVAL_ERR_REL`    | 0.15      | front celerity at the six gauges — the headline |
| `GATE_DEPTH_L1_REL`       | 0.10      | depth profile at the probe time (points 2–6)   |
| `GATE_SPEED_L1_REL`       | 0.25      | speed profile at the probe time (points 2–6)   |
| `GATE_ISOTROPY_M`         | 0.02 m    | 45° gauge vs the on-axis gauges                |
| `GATE_DETERMINISM_REL`    | 1e-4      | run-to-run reproducibility                     |

Point 1 (r = 50 m) is excluded from the profile norms — it sits only five
source-radii out, where the line-vs-disc idealisation of the inflow still
shows — but is still gated on arrival time. The reference's own arrival times
carry ~1.5 % discretisation error at dr = 1 m, and snapshots resolve 60 s;
the arrival gate leaves room for both.

## Backends

Default backend is `gpu_resident_batch`. Other retained Vulkan implementations
are selectable with `--backends`. Velocity scoring uses momentum downloads.

## Running

```sh
# In the dev container:
just benchmark-propagation --repeats 3 --warmup 1 --gif

# On the macOS host (unsandboxed, patched kp in .venv):
.venv/bin/python -m inundation.bench.flood_propagation --repeats 3 --warmup 1 --gif

# Explicit backend matrix:
just benchmark-propagation --backends gpu_resident_batch,fixed_dt_batch_barrier
```

`--repeats >= 2` enables the determinism check; `--warmup` runs are excluded
from timing. The animation comes from a single _untimed_ extra run, so perf
numbers stay clean, and its frames are resampled onto uniform simulated time
(the solver snapshots every phase boundary, so the 60 s hydrograph phases
would otherwise dominate the frame list).

### Key flags

- `--backends` — CSV of solver impls (see above).
- `--dataset-dir` — published dataset location.
- `--nx` / `--ny` — grid resolution.
- `--t-end` / `--dt-max` / `--cfl-interval` — run length and timestep controls.
- `--probe-time` — when depth/speed profiles are compared with the reference.
- `--gif` — top-down 2D depth-heatmap animation.
- `--gif-backend` / `--gif-frames` — animation controls.

## Artifacts

Written under `.tmp/flood-propagation-bench/`:

- `plain-<nx>x<ny>.parquet` — flat-bed mesh.
- `summary.json` — gates, per-gauge arrivals/depths/speeds against the
  reference, correctness + perf metrics.
- `propagation-<backend>-2d.gif` — top-down depth heatmap of the advance.

## Measured result

See
[results.md](../../../docs/implementation/18-flood-propagation/results.md).
