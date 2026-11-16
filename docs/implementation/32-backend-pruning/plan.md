# Backend Pruning - Plan

**Status: done.** Results: [`results.md`](results.md).

## Goal

Measure every selectable `solver_impl` on one case and delete the ones that do
not produce correct results.

## Scope

In: a timing and correctness run of all seven backends on the radial
dam-break; deletion of `batched_submit`, `async_sync_window`, `device_cfl` and
the refusing `fixed_dt_batch`, with the shaders only they used.

Out: fixing the deleted backends. `gpu_resident_batch` already covers device
CFL and batched submission correctly, so a repaired copy would duplicate it.

## Design

- One-off driver over `inundation.bench.radial_dambreak` helpers: one warmup,
  median of three timed runs, solver-loop wall time
  (`WorkflowResult.wall_seconds`), the bench's own correctness gates.
- 200² and 400² meshes, each at the backend's default CFL and at 0.45 for all.
- Deleted: the three modules, `fixed_dt_batch.py` (the barrier class now
  derives from `SWESolver`), `FLUX_DTBUF_GLSL`, `UPDATE_DTBUF_GLSL`,
  `SOURCE_DTBUF_GLSL`, `CFL_RESOLVE_GLSL`, and the dtbuf branch of
  `SWESolver._build_source_algo`. `inundation_metal` compares the two
  remaining fast backends instead of measuring speedup against
  `batched_submit`.
