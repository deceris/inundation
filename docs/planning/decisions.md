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

## Dry cells carry no momentum

A cell whose Heun-averaged depth is below `dry_tol` ends the step with zero
momentum. Before, the corrector blended the start-of-step momentum into the
average while the stage result had zeroed it, so a drying cell kept half its
momentum, halved it for ~150 steps, and fed it back on rewetting. The
active-set wet test is now depth alone. Evidence:
[`24-solver-accuracy-fixes/`](../implementation/24-solver-accuracy-fixes/results.md).

## HLLC dry-bed wave speeds and a flux-consistent CFL bound

A face with a dry side uses Toro's dry-bed estimates (`u ∓ 2c`); the Roe
average gave about 0.7c there. The CFL bound is the largest wave speed the
flux assumed on each edge — the resident corrector takes it from the flux
itself, the prologue and host-driven CFL kernels reconstruct the face states
the same way — rather than `|u| + c` of the raw states, so it covers the
dry-bed speeds and is tighter on wet-wet faces. Evidence:
[`24-solver-accuracy-fixes/`](../implementation/24-solver-accuracy-fixes/results.md).

## Face depths from per-edge bed steps

The kernels reconstruct the face depth as `h - dz` with `dz = z_face - zb`
precomputed per edge in float64 from the float32 bed, instead of
`h + zb - z_face` in float32, which at a bed hundreds of metres above the
datum loses the low bits of the surface elevation. The device no longer reads
the bed; binding 3 carries the `(dzL, dzR)` pairs. Geometry cache format
version 5. Evidence:
[`24-solver-accuracy-fixes/`](../implementation/24-solver-accuracy-fixes/results.md).

## Compensated depth on the resident backend

`gpu_resident_batch` keeps each depth as an exact float32 pair `(h, h_lo)`
and adds every increment with Knuth's two-sum, so nothing below half an ulp
is rounded away; the flux kernels read `h` alone. This replaces the float64
state reference the roadmap called for: on a closed valley the volume change
net of clamp mass and sources is ~5e-10 relative over several thousand steps,
against ~1e-6 for the plain update, and the plain drift has one sign and
grows with the step count, as the Nysa deficit did. A float64 state was
rejected for the same reason as a float64 clock (no `shaderFloat64` on the
MoltenVK reference platform). The earlier worry that a compensated pair
depends on the shader compiler proved real and is handled: MoltenVK's default
fast math reassociates the two-sum away, so the solver sets
`MVK_CONFIG_FAST_MATH_ENABLED=0` before creating the instance (no measurable
cost). A wall edge passes exactly zero mass flux for the same reason. Default
on; `compensated = False` keeps the plain update. Evidence:
[`24-solver-accuracy-fixes/`](../implementation/24-solver-accuracy-fixes/results.md).

## Resident backend is the default; the unbarriered batch is closed

`WorkflowConfig.solver_impl` defaults to `gpu_resident_batch`, the backend
every gate runs on. `fixed_dt_batch` refuses to run; its barrier subclass
remains the host-driven reference. Element counts reach the shaders as
`uint` bit patterns (a float value is exact only to 2^24), and
`build_geometry` rejects cells at or below the CFL kernels' area floor.
Evidence:
[`24-solver-accuracy-fixes/`](../implementation/24-solver-accuracy-fixes/results.md).

## Courant number 0.8 on the resident backend

`gpu_resident_batch` recomputes the CFL bound every step from the wave
speeds the flux assumed, so the 0.8 safety factor it inherited from the
batch backends (which reuse one dt for several steps) is gone and the CFL
number is its only knob. In the solver's perimeter-sum form the HLL
positivity limit is 1.0. A sweep of the five gate suites passes at 0.45,
0.6, 0.8 and 1.0 with accuracy metrics flat within the gates' resolution and
step counts falling in proportion. `CFL_DEFAULT` is 0.8, 1.9x fewer steps
than the previous effective 0.36, keeping a margin below the limit for
terrain the sweep did not cover; 1.0 remains a configuration choice. The
sweep exposed that the next dt was planned before the sources added their
water; the source kernel now bounds it by each source cell's post-source
state, so a point source into a thin cell cannot drive it negative.
Evidence:
[`24-solver-accuracy-fixes/`](../implementation/24-solver-accuracy-fixes/results.md).

## Thin-film damping and continuous dry-bed speeds

The dry-bed wave speeds blend in over one to two `dry_tol`, and velocity below
four `dry_tol` uses the Kurganov-Petrova desingularised form, with the stored
momentum damped to match. Abrupt speeds let rounding shift a front between runs,
and undamped films reached 26 000 m/s and stalled the CFL bound on the parabolic
bowl, a case no gate reached. This replaces the plain dry-bed estimates of the
earlier entry. Removing the speeds also stopped the crashes but gave no accuracy
gain; damping improved bowl errors by about 7 to 12 %. The accepted cost is a
slower thin front tip: Ritter front error 12.9 % to 13.9 %. Evidence:
[`24-solver-accuracy-fixes/`](../implementation/24-solver-accuracy-fixes/results.md).

## Cost of compensated depth

The earlier claim that disabling MoltenVK fast math has no measurable cost was
wrong: interleaved runs show about 7 %, and the compensated update with the
`precise` qualifier cost about 100 %. `precise` is now dropped on macOS, where
fast math off already protects the arithmetic. It stays on other platforms, so
the cost there is unmeasured until the update runs on NVIDIA. Evidence:
[`24-solver-accuracy-fixes/`](../implementation/24-solver-accuracy-fixes/results.md).

## Nysa deficit resolved; CFL 0.8 confirmed on real terrain

The 1.26 % Nysa deficit was float32 rounding of depth increments: with the
compensated depth it reads 0.0000 % with no clamp contribution. Depths at CFL 0.8
stay within 2.1 cm of a CFL 0.4 run at every checked time, at about half the wall
time of effort 23, so 0.8 stays the default. Evidence:
[`24-solver-accuracy-fixes/`](../implementation/24-solver-accuracy-fixes/results.md).

## CFL default per backend, film damping as a rate

`CFL_DEFAULT` 0.8 applies only to `gpu_resident_batch`, which recomputes its
bound every step; backends that reuse one dt keep `CFL_HOST_DRIVEN` 0.45. Film
momentum decays at a rate, semi-implicit like friction, because a fixed factor
per stage made results depend on the tick size. This gives back a little of the
bowl gain and removes the dependence. Evidence:
[`24-solver-accuracy-fixes/`](../implementation/24-solver-accuracy-fixes/results.md).

## Momentum snapshots fail loudly

Requesting momentum on a backend that cannot provide it now raises, naming the
backend, instead of returning `None`. Support is an explicit list rather than a
signature check, because most backends accept the flag and ignore it. Benches
ask only where it is supported, so they keep working on the barrier backend.
Evidence:
[`26-momentum-snapshots/`](../implementation/26-momentum-snapshots/results.md).

## Mesh build stays bit-identical

The numpy mesh build reproduces the Python one exactly, so
`GEOMETRY_CACHE_VERSION` stays at 5 and no new baseline is needed. The one
exception to doing the simplest thing is that cells are summed per degree rather
than with `np.bincount`, whose rounding differs. The old build stays in the tests
as the oracle. `hilbertcurve` is now only used by tests. Evidence:
[`27-fast-mesh-build/`](../implementation/27-fast-mesh-build/results.md).
