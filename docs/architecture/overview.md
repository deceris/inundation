# SWE Solver Architecture

The solver advances the shallow-water equations on an unstructured mesh with
HLLC fluxes (Roe-average wave speeds on wet faces, Toro's dry-bed speeds on
wet-dry faces), Audusse hydrostatic reconstruction from per-edge bed steps,
semi-implicit Manning friction and SSP-RK2 integration. The CFL bound is the
largest wave speed the flux assumed on each edge. NumPy loads and preprocesses
geometry; Vulkan/Kompute executes the timestep kernels.

## Pipeline

1. Load mesh cells and build edge plus CSR adjacency.
2. Optionally Hilbert-reorder cells and cache the geometry, or load a prebuilt
   versioned artifact (no mesh file required).
3. Compile GLSL compute shaders to SPIR-V.
4. Dispatch flux, update, source, and CFL stages through a selected Vulkan
   synchronization strategy.
5. Download snapshots at time- or step-based cadence.

`gpu_resident_batch` is the default: simulated time, dt and the active set
stay on the device, flux sums are gathered in a fixed order, and the depth is
an exact float32 pair updated with error-free transforms so the volume balance
is exact up to the positivity clamp. `fixed_dt_batch_barrier` is the
host-driven barrier reference.

## Synchronization

Batched dispatches require compute barriers between a stage's writes and the
next stage's reads. The barrier implementation fixed the race present in the
unbarriered fixed-step batch and agrees with the independently barriered
GPU-resident implementation on the lake-at-rest matrix.

## Mesh and precision constraints

Geometry retains Hilbert ordering, content-keyed caching plus explicit
versioned artifacts, optional per-cell mesh-sourced Manning roughness, dense
and CSR adjacency, and packed signed edge orientation. CSR offsets are int32
and reject meshes whose slot count would overflow. Initial and warm-start
states preserve depth, both momentum components, and per-cell roughness.

See `../implementation/00-lake-at-rest/` for retained evidence.
