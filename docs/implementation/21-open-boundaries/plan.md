# Minimal Open-Boundary Forcing — Plan

**Status: done.** Results: [`results.md`](results.md).

## Goal

Prescribe constant normal discharge or constant depth on selected mesh
boundary edges; every other boundary edge stays a reflective wall.

## Scope

In: walls (default), prescribed outward unit discharge, prescribed depth,
static values, selection by caller-built edge masks.

Out: hydrographs, rating curves, tides, characteristic/radiation outflow,
GIS-named boundary segments, boundary labels in mesh files or geometry caches.

## Design

- `inundation.boundary.OpenBoundary(edge_mask, kind, value)`, frozen; the mask
  is copied read-only. `boundary_table(geom, specs)` validates shape,
  boundary-only selection, and overlap against the geometry.
- Masks index solver-order edges (after Hilbert reorder), so they are passed to
  `SWEWorkflow.run(phases, open_boundaries=...)` after `prepare()`, not to the
  frozen `WorkflowConfig`.
- One `(kind, value, volume_coarse, volume_fine)` float4 per edge on binding 19 of
  the flux and CFL-accumulate kernels only. It is read only inside `cR < 0`, so
  interior edges pay nothing.
- Discharge: mass flux imposed directly (`Fh = q_n`), not via a Riemann solve,
  so the boundary volume is exact. Outflow upwinds the interior and is capped
  at the cell's stage volume; inflow enters at `max(h_L, h_critical)`.
- Depth: fixed-depth ghost state with the interior velocity, through HLLC.
- The dry-edge early exit and the CFL wave speed both use the exterior state.
- Boundary volume accumulates per edge in the flux kernel (no atomics) and is
  exposed as `WorkflowResult.volume_boundary_out_m3`; `volume_injected_m3`
  still counts point sources only.
