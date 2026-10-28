# Decisions

Settled engineering choices and their rationale. Open work belongs in
`roadmap.md`; measurements belong under `../implementation/`.

## Vulkan reference implementation

`fixed_dt_batch_barrier` is the validated barrier reference. The older
`fixed_dt_batch` path omitted a compute barrier between dependent dispatches,
causing a read-after-write race, rapid divergence, and nondeterminism.
`gpu_resident_batch` independently confirms the corrected results.

## Mesh representation

Flux and update stages retain CSR adjacency with packed signed edge
orientation. Fixed per-cell gather order supports deterministic accumulation;
int32 CSR offsets fail explicitly before overflow.

## Simulated-time precision

Long-horizon simulated time must use float64 or integer representation. Float32
resolution becomes too coarse for ordinary timesteps at production horizons.

## Momentum state

Initial depth and both momentum components are preserved across workflow
preparation and phase resets. Final momentum downloads are part of the retained
warm-start and velocity-scoring contract.

## Hydrostatic interface correction

The Audusse source correction is part of both Vulkan GLSL flux variants. Each
cell uses its own actual and reconstructed depth along its own outward normal.
This reduces lake-at-rest velocity from roughly 0.3–0.5 m/s to below 1e-4 m/s.

## Momentum-obstruction benchmark

The published Test 3 dataset is accepted only with a paired equal-volume
still-water control. After the hydrostatic correction, the release ponds
0.047 m beyond the obstruction while the control remains exactly dry. This
makes the far pond a momentum signature rather than numerical leakage.

## Float32 mass floor

Flood-propagation drift worsened under timestep and grid refinement and then
saturated. This is consistent with small `dt * dh` increments being absorbed
by float32 state, but remains a hypothesis until a float64 state comparison.
The 1e-3 mass gate applies only to the measured Test 4 configuration.

## Mesh-sourced Manning roughness

A mesh file may carry a per-cell `manning_n` column (GeoParquet, GeoPackage).
When present it defines roughness per cell; otherwise `config.manning_n` applies
uniformly. Precedence is initial-state `n_mann`, then the mesh column, then the
config value, so a restart keeps the roughness its saved state was calibrated
with. The column must be finite and positive — rejected rather than clamped —
and it follows its source cell through `hilbert_reorder` into solver order.
Absent roughness round-trips through the geometry cache as `None` (stored as an
empty array), so a reloaded geometry is indistinguishable from one built
without the column.
Evidence: [`20-mesh-cache-artifacts/`](../implementation/20-mesh-cache-artifacts/plan.md).

## Geometry cache artifacts: explicit path

A cache-only job gets its geometry from an explicit artifact path
(`WorkflowConfig.geometry_cache_source`), not from a manifest lookup or a
content hash of the built arrays: callers that preprocess meshes in an earlier
pipeline stage own naming and versioning in their own storage layout, and the
accepted tradeoff is that a mismatched artifact is the caller's problem. The
one hazard the caller cannot see is checked — the artifact records the cache
format version and refuses to load under a different `GEOMETRY_CACHE_VERSION`,
and its build-time reorder mode must match `WorkflowConfig.use_hilbert_reorder`
(a mismatch is rejected) so cell ordering can never disagree with the caller's
intent.
When set, the artifact takes precedence over `geometry_cache_dir` and
`mesh_source`, which are then never read; `mesh_source` may be `None`.
Evidence: [`20-mesh-cache-artifacts/`](../implementation/20-mesh-cache-artifacts/results.md).

## Resident simulated clock: integer ticks

`gpu_resident_batch` keeps simulated time on device as a `uint64_t` count of
2⁻²⁰ s ticks, and quantizes each step's dt down to whole ticks so the clock
advances by exactly the dt the physics used. A float64 clock was rejected:
Apple GPUs under MoltenVK have no `shaderFloat64`, so the reference platform
could not run it, and even float64 would leave the clock only approximately
equal to the summed dt. A compensated float32 pair was rejected because it
depends on the shader compiler not reassociating floating-point arithmetic.
Evidence: [`21-resident-clock-and-batch-loop/`](../implementation/21-resident-clock-and-batch-loop/plan.md).

## Nysa volume deficit: not the clock or the clamp

With the integer tick clock the Nysa 24 h run still loses 1.26 % of injected
volume, growing with the wet area, with zero clamp volume. The remaining
suspect is float32 state arithmetic; it is handled by the float64 or
compensated-state work, not by further clock changes. Evidence:
[`21-resident-clock-and-batch-loop/results.md`](../implementation/21-resident-clock-and-batch-loop/results.md).

## Active set: exact block skipping

`gpu_resident_batch` skips 256-cell blocks outside the two-ring of blocks that
hold water or momentum, recomputed every step. Two cell rings bound what one
Heun step can change, so results match the full step; there is no
approximate threshold to tune. Compacted lists were rejected because Kompute
has no indirect dispatch. Evidence:
[`22-active-set/results.md`](../implementation/22-active-set/results.md).

## Deterministic accumulation by default

`gpu_resident_batch` sums fluxes and CFL terms per cell in the fixed order of
the mesh CSR instead of with float `atomicAdd`, so repeated runs are
bit-identical on one device and driver. This restores the roadmap invariant
that kernel restructuring preserves per-cell gather order. It is the default
because it costs nothing: dry edges write nothing and cells with no live edge
skip the sum, so it is as fast as the atomics on a fully wet mesh and faster
on Nysa. The atomic path stays behind `deterministic = False` as a fallback.
Bit-identity across devices or drivers is not claimed. Evidence:
[`23-deterministic-accumulation/results.md`](../implementation/23-deterministic-accumulation/results.md).
