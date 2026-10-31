# Momentum Snapshots on the Resident Backend — Plan

**Status: done.** Changes the `gpu_resident_batch` run loop and the workflow's
handling of `capture_momentum_snapshots`. No kernel changes.

## Goal

Return `hu` and `hv` at every output time on the default backend, for
flow-direction maps and particle animation on Nysa.

## Problem

`SWESolverGpuResidentBatch.run()` accepted `capture_momentum_snapshots`, ignored
it and returned depth only. The workflow treated a 2-tuple as "no momentum", so
the flag gave empty results with no error. Its `inspect` check could never fire,
because every backend's `run()` has the parameter.

## Design

**Capture.** `run()` reads `hu` and `hv` wherever it records depth: the initial
snapshot, each output time, the non-finite exit and the final tail. Reads happen
between graphs, after the device has finished, so they add no sync point inside
a graph and cost nothing with the flag off.

**State read.** `t_hu` and `t_hv` hold the committed step. The corrector writes
them; `hu1` and `hv1` are the predictor, and no-op steps return before writing.

**Phase boundaries.** The depth path keeps a duplicate at each boundary, because
every phase contributes its start. Momentum keeps it too, so the lists stay
aligned one-for-one.

**Inactive blocks.** A block goes inactive only after a corrector pass found all
its cells dry, and that pass zeroes their momentum. Inactive cells therefore hold
exactly zero, not stale values.

**Failure.** `MOMENTUM_SNAPSHOT_BACKENDS` in `tuning.py` lists the backends that
really capture. `WorkflowConfig` raises a `ValueError` naming any other backend,
before a mesh is loaded. The workflow raises a `RuntimeError` if a backend still
returns no momentum or misaligned lists.

**Benches.** `momentum_obstruction` and `flood_propagation` request capture only
on a supported backend, which keeps their "velocities omitted" warning on the
barrier backend. `floodplain_depressions` never read momentum, so it stopped
asking.

## Out of scope

Edge fluxes, a direction helper, GIS export and streaming snapshots to disk.

## Verification

- Lists align across a multi-phase run, and the last snapshot matches the final
  state.
- Depth is bit-identical with the flag on and off.
- Each snapshot is bit-identical to a flag-off run stopped at that time.
- Momentum matches a barrier run stopped at each output time, within a measured
  tolerance: the backends take different steps, so an exact match is not valid.
- Dried inactive blocks hold zero momentum.
- Nysa 24 h with the flag on and off, timed per snapshot.
