# deceris-inundation

Flooding is expensive. A single event can shut down roads, damage property,
and delay construction. Flood risk is hard to judge from a terrain map
alone: water follows slopes, collects in low ground, and hits places that
look safe.

Deceris Inundation answers a simple question: "Where will the water go, and how deep will it get?".
A site, road, or district can be tested under different water
inputs, such as a swollen river or intense rainfall, and the outcomes
compared side by side.

It is built for decisions that need numbers. Planners check whether a
development stays dry, infrastructure teams see which roads stay passable,
and risk analysts put depth figures on specific locations. Running on the
graphics processor (GPU) keeps large areas and long events fast enough for
repeated scenario testing.

## How it works

In plain terms, the simulation divides the terrain into a grid of connected cells and advances the solution through time. At each step, it computes how
much water is present in each cell and how water flows between neighboring cells. The underlying physics is described by the Shallow Water Equations,
a widely used model for simulating the movement of a shallow layer of water over terrain. Fluxes between cells are computed using the HLLC solver, a well-established approximate Riemann solver for this type of flow.

## The runtime pipeline

`SWEWorkflow` has two stages: preparation and simulation.

1. **Prepare the mesh.** Read polygon cells and bed elevations from GeoPackage,
   Shapefile, Parquet/GeoParquet, or OBJ input.
2. **Build the geometry.** NumPy computes cell and edge geometry plus signed
   adjacency. The result can be Hilbert-reordered and cached for reuse, or
   loaded prebuilt from a `build_geometry_cache` artifact (no mesh file read).
3. **Prepare the GPU.** Compile the GLSL kernels to SPIR-V, allocate Vulkan
   buffers, and upload the initial state and mesh data.
4. **Run the simulation.** For each timestep, Vulkan computes HLLC edge fluxes,
   applies source terms, updates depth and momentum, and periodically estimates
   a stable CFL timestep.
5. **Return results.** The workflow records depth snapshots at the configured
   output interval and can optionally capture momentum snapshots.

`run()` advances one or more `SimulationPhase` objects and returns a
`WorkflowResult`. See `Backends` for solver choices.

## Prerequisites

- Docker Compose with the `dev` service (`just up`), plus `just`.
- For GPU runs: a host Vulkan driver and ICD.
- macOS Vulkan work: Homebrew, Xcode command-line tools
  (`xcode-select --install`), `uv`.

## Supported runtime

| Host | Path | Use |
| --- | --- | --- |
| Linux + NVIDIA | Docker image, or Apptainer with `--nv` passthrough | Production GPU runs |
| macOS | Native host via MoltenVK (see below) | Development and correctness checks |

The Docker image contains Python 3.12, all optional Python dependencies,
Vulkan development tools, and the Kompute binding. Source code is
bind-mounted into the running container.

## Development setup

Build the image once, start the persistent container, then run checks:

```sh
just image-build
just up
just check
```

Stop the container when finished:

```sh
just down
```

`just up` assumes the `deceris-inundation:dev` image already exists. Rebuild
it after changing `Dockerfile`, `pyproject.toml`, `uv.lock`,
`inundation/vulkan/patches/`, or
`inundation/scripts/build-kp-linux.sh`. Source changes under the
bind-mounted repository do not require an image rebuild.

## macOS development

Docker Desktop on macOS does not expose the Apple GPU inside the Linux
container, so use the native host environment for Vulkan work through
MoltenVK. Do not use the default Docker-backed `just benchmark-*` recipes
for GPU execution on macOS:

```sh
brew install uv cmake molten-vk vulkan-headers vulkan-loader vulkan-tools glslang
uv sync --all-extras
inundation/scripts/build-kp-macos.sh
export VK_ICD_FILENAMES="$(brew --prefix)/etc/vulkan/icd.d/MoltenVK_icd.json"
```

If `patch` is unavailable, install the Xcode command-line tools with
`xcode-select --install`. The Kompute build script validates a real
`kp.Manager()` round trip against MoltenVK and installs the patched binding in
`.venv`.

Run checks natively by bypassing the Docker prefix:

```sh
IN_CONTAINER=1 just lint
IN_CONTAINER=1 just typecheck
IN_CONTAINER=1 just test
```

Run GPU benchmarks either through the same `IN_CONTAINER=1` switch:

```sh
IN_CONTAINER=1 just benchmark-lake --backend gpu_resident_batch
```

or invoke the host interpreter directly:

```sh
.venv/bin/python -m inundation.bench.lake_at_rest \
  --backend gpu_resident_batch
```

The Apptainer recipe and `--nv` passthrough do not apply to macOS.

## Running the solver

### Install mesh readers

The base package supports OBJ meshes. Install the extras for GIS mesh input:

```sh
pip install "deceris-inundation[mesh-parquet,mesh-gis]"
```

Use `mesh-parquet` for `.parquet` and `.geoparquet` meshes. Use `mesh-gis` for
`.gpkg` and `.shp` meshes. GIS mesh features need at least three vertices and a
finite `z_mean` column containing per-cell bed elevations. OBJ files need a Z
coordinate on every vertex; vertex elevations are averaged per cell. Meshes
without elevation data are rejected.

GIS meshes may also carry an optional per-cell `manning_n` column (`.parquet`,
`.geoparquet`, `.gpkg`, `.shp`). Values must be finite and positive, and they
override `config.manning_n` cell by cell where present; `WorkflowConfig.manning_n`
is still required and must be positive — it is the fallback. OBJ meshes cannot
carry the column.

### Minimal API usage

```python
from inundation import PointSource, SimulationPhase, SWEWorkflow, WorkflowConfig

config = WorkflowConfig(
    mesh_source="mesh.gpkg",
    manning_n=0.035,
    output_interval_s=300.0,
    dt_max=0.05,
    cfl_interval=1,
)
workflow = SWEWorkflow(config)
workflow.prepare()  # load mesh, compile shaders, allocate Vulkan resources
result = workflow.run([
    SimulationPhase(
        duration_s=3600.0,
        sources=[PointSource(discharge_m3s=10.0, center_xy=(100.0, 200.0), radius_m=2.0)],
    ),
])
print(result.h_final, result.volume_final_m3)
```

Call `prepare()` before `run()`. Set `geometry_cache_dir` in
`WorkflowConfig` to reuse processed geometry between runs.

To run with no mesh file at all — for jobs that only receive a prebuilt
geometry artifact — build the artifact once and point `geometry_cache_source`
at it:

```python
from inundation.mesh import build_geometry_cache

artifact = build_geometry_cache("mesh.gpkg", "artifacts/mesh.geometry.npz")

workflow = SWEWorkflow(
    WorkflowConfig(
        mesh_source=None,  # the artifact is the geometry source
        geometry_cache_source=artifact,
        manning_n=0.035,
        output_interval_s=300.0,
        dt_max=0.05,
        cfl_interval=1,
    )
)
```

`build_geometry_cache` runs the full mesh preprocessing chain (load mesh →
`build_geometry` → Hilbert reorder), needs no GPU, and captures whatever the
mesh file carries, including a per-cell `manning_n` column. The artifact
records the cache format version, so one written by a different release is
rejected rather than silently consumed.

When `geometry_cache_source` is set it wins over `geometry_cache_dir` and
`mesh_source`, which are then never read, and `mesh_source` may be `None`. The
artifact's build-time reorder mode is authoritative: it must match
`WorkflowConfig.use_hilbert_reorder`, and a mismatch is rejected rather than
silently reordering cells. A foreign cache version raises, whereas the
directory cache simply misses after a version change. Cache-only workflows
cannot be plotted — the plotting helpers need the mesh file to reconstruct
cell polygons. See `inundation/workflow.py` for the complete configuration
surface.

## Input contracts

`PointSource.center_xy` must already use the mesh coordinate system. The
workflow does not inspect or reproject source CRS metadata. Source coordinates
are treated as planar Cartesian coordinates, and `radius_m` uses the mesh
coordinate units. Do not pass longitude/latitude coordinates when the radius
is intended to be measured in metres.

`discharge_m3s` is the source's total discharge. For each phase, the workflow
selects cells whose centroids lie within `radius_m` and also includes the cell
containing the source point. It distributes the discharge across the selected
cells in proportion to their areas.

If a source selects no valid cells, the workflow skips it without raising an
error. Validate source locations against the mesh in the calling application
before running. `volume_injected_m3` is calculated from the requested source
schedule, so it can include discharge from a source that selected no cells.

Open boundaries are passed to `SWEWorkflow.run(phases, open_boundaries=[...])`
as `inundation.OpenBoundary(edge_mask, kind, value)`. Every boundary edge not
selected stays a reflective wall. The masks have shape `(geom.E,)` and index
`workflow.geom` after `prepare()`, because the Hilbert reorder permutes edges.
Select edges with `geom.edge_cellR == -1` plus the outward normal
`edge_nx`/`edge_ny`, or with edge positions.

- `kind="discharge"`: `value` is the unit discharge along the **outward**
  normal [m²/s]. Negative values are inflow; each edge carries
  `value * edge_len` m³/s. Outflow is capped at what the boundary cell holds.
- `kind="depth"`: `value` is the exterior depth [m] above the adjacent bed.
  This is a simple fixed-depth ghost state, not a non-reflecting condition.

Values are constant for the whole run. `WorkflowResult.volume_boundary_out_m3`
reports the net volume that left through open boundaries (negative means net
inflow). `volume_injected_m3` still counts point sources only.

`dt_max` is the maximum numerical timestep, not the snapshot interval.
`output_interval_s` controls depth snapshot timing. `cfl_interval` controls how
many steps may reuse a CFL estimate: larger values reduce overhead but react
more slowly to changing flow. Choose these values for the mesh resolution,
velocities, source rates, and solver backend.

## State and ordering

`WorkflowConfig.initial_state_source` accepts a dictionary or `.npz`, `.npy`,
`.csv`, and `.parquet` files. CSV and Parquet state tables require `pandas`.
The accepted fields are:

- `h` or `wse` for water depth or water-surface elevation;
- `hu` and `hv`, or `u` and `v`, for momentum or velocity;
- optional `n_mann` for per-cell Manning roughness;
- optional integer `cell_id` for explicit row-to-cell alignment.

Every per-cell field must contain exactly one finite value per mesh cell.
`h` takes precedence when both `h` and `wse` are present. Negative depths are
clamped to zero, momentum is cleared in dry cells, and `n_mann` is floored at
`1e-4`. When the mesh (or its geometry artifact) carries a `manning_n` column,
that value replaces `config.manning_n` cell by cell; a loaded-state `n_mann`
takes precedence over both.

The default `initial_state_order` is `"solver"`. With Hilbert reordering,
`workflow.perm` satisfies:

```text
perm[solver_index] = original_mesh_index
```

Use `initial_state_order="original"` when input arrays follow mesh-file order.
To map final solver output back to mesh-file order for GIS export:

```python
import numpy as np

if workflow.perm is None:
    raise RuntimeError("workflow was not prepared")
h_original = np.empty_like(result.h_final)
h_original[workflow.perm] = result.h_final
```

State files contain hydraulic arrays, not schedule or clock state. Metadata such
as `t_end_s`, `dt_max`, and `cfl_interval` is ignored when loading a state. A
new `run()` starts its `snap_times` at zero; callers must manage absolute time
and hydrograph continuation.

## GIS application example

The following pattern matches a common GeoParquet-mesh and Shapefile-source
workflow. The application must verify CRS consistency before constructing
`PointSource` objects; the solver will not reproject them.

```python
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
from shapely import from_wkb

from inundation import PointSource, SimulationPhase, SWEWorkflow, WorkflowConfig

mesh_path = Path("mesh.parquet")
source_path = Path("sources.shp")
mesh_crs = "EPSG:2180"  # verify this against the mesh and source coordinates
source_gdf = gpd.read_file(source_path)
if source_gdf.empty or "inflow" not in source_gdf:
    raise ValueError("sources must contain Point geometries and an inflow column")

# If CRS metadata is wrong but coordinates are already in mesh units, correct
# the metadata only after verifying the stored coordinates. Do not call to_crs()
# unless the coordinates actually need transformation.
# source_gdf = source_gdf.set_crs(mesh_crs, allow_override=True)

sources = []
for index, row in source_gdf.iterrows():
    point = row.geometry
    if point is None or point.geom_type != "Point" or not np.isfinite(row.inflow):
        raise ValueError(f"invalid source row {index}")
    sources.append(
        PointSource(
            discharge_m3s=float(row.inflow),
            center_xy=(float(point.x), float(point.y)),
            radius_m=5.0,
        )
    )

workflow = SWEWorkflow(
    WorkflowConfig(
        mesh_source=str(mesh_path),
        manning_n=0.035,
        output_interval_s=1200.0,
        dt_max=2.0,
        cfl_interval=1000,
        geometry_cache_dir=".deceris-cache",
        initial_state_source="state.npz",  # omit for a dry initial state
        initial_state_order="solver",
        solver_impl="gpu_resident_batch",
    )
)
workflow.prepare()
result = workflow.run([
    SimulationPhase(duration_s=12 * 3600.0, sources=sources),
])

h_original = np.empty_like(result.h_final)
h_original[workflow.perm] = result.h_final  # see `State and ordering` for perm semantics

# Re-read the original mesh geometry for GIS export. The solver returns arrays,
# not GeoDataFrame geometry or CRS metadata.
table = pq.read_table(mesh_path, columns=["tri_id", "z_mean", "geometry"])
geometry = from_wkb(table.column("geometry").to_numpy(zero_copy_only=False))
bed = table.column("z_mean").to_numpy(zero_copy_only=False)
wet = h_original > workflow.config.dry_tol
gpd.GeoDataFrame(
    {
        "tri_id": table.column("tri_id").to_numpy(zero_copy_only=False)[wet],
        "depth_m": h_original[wet],
        "water_lvl": (bed[wet] + h_original[wet]),
    },
    geometry=geometry[wet],
    crs=mesh_crs,
).to_file("inundation-final.shp", index=False)
```

## Workflow results

`WorkflowResult` contains:

- `snapshots` and `snap_times`: depth arrays and their simulation times;
- `h_final`: final depth in solver order;
- `hu_final` and `hv_final`: final momentum in solver order, when exposed by
  the backend;
- `hu_snapshots` and `hv_snapshots`: optional momentum arrays aligned with
  `snapshots`;
- `volume_final_m3` and `volume_injected_m3`: final and scheduled source
  volumes;
- `wall_seconds` and `steps_total`: runtime measurements;
- `clamp_mass_m3`: optional volume added by the positivity clamp when
  `track_clamp_mass=True`.

## Backends

Select a backend with `WorkflowConfig.solver_impl`:

| Backend | Use |
| --- | --- |
| `baseline` | Default compatibility and debugging path. |
| `fixed_dt_batch_barrier` | Validated correctness reference with explicit dispatch barriers. |
| `device_cfl` | GPU-side CFL resolution with a host-driven timestep loop. |
| `gpu_resident_batch` | Preferred production candidate for constant per-phase sources after target-driver validation; minimizes host synchronization. |

`gpu_resident_batch` advances time on the device. Choose output intervals that
are well represented by the backend's float32 clock; validate snapshot cadence
on the target Vulkan driver. Use `fixed_dt_batch_barrier` for correctness
comparisons and `baseline` for debugging.

The remaining implementations are retained for benchmark and advanced
comparison use. Validate backend behavior on the target Vulkan driver before
using a performance-oriented path in production.

`hu_final` and `hv_final` are needed to write a complete warm-start state.
Momentum snapshots require `capture_momentum_snapshots=True` and backend
support.

## Benchmarks

Benchmark recipes run inside the development container (see runtime table
above). On macOS, use the native `IN_CONTAINER=1` workflow above instead.

```sh
just benchmark-lake --help
just benchmark-radial-dambreak --help
just benchmark-depressions --help
just benchmark-obstruction --help
just benchmark-propagation --help
just benchmark-report --help
```

The lake-at-rest benchmark is a quick correctness starting point:

```sh
just benchmark-lake --backend fixed_dt_batch_barrier --flat-bed --hash
```

Detailed benchmark contracts and flags are documented beside their harnesses:

- [lake-at-rest](inundation/bench/lake_at_rest.md)
- [analytical dam-break](inundation/bench/dambreak.md)
- [radial dam-break](inundation/bench/radial_dambreak.md)
- [floodplain depressions](inundation/bench/floodplain_depressions.md)
- [momentum obstruction](inundation/bench/momentum_obstruction.md)
- [flood propagation](inundation/bench/flood_propagation.md)

## Production Apptainer image

Build the SIF from the already-built Docker image and run a benchmark with
NVIDIA Vulkan passthrough:

```sh
just image-build
just apptainer-build
just apptainer-run -- python -m inundation.bench.lake_at_rest --help
```

The repository is bind-mounted at `/workspace`; code changes do not require a
new image. Rebuild the base image when its dependencies or Kompute build inputs
change.

## Project layout

- `inundation/` — solver package and workflow API
- `inundation/vulkan/` — Vulkan/Kompute implementations and shaders
- `inundation/mesh/` — mesh loading, geometry, and caching
- `inundation/bench/` — benchmark harnesses and reports
- `tests/` — unit tests
- `docs/architecture/overview.md` — solver architecture and invariants

Optional host-only dependencies can be installed with:

```sh
uv sync --all-extras
```

Linux host execution also needs the Kompute binding built by
`inundation/scripts/build-kp-linux.sh`; the Docker image performs that
step automatically.
