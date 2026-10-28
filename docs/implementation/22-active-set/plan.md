# Active Set — Plan

**Status: done.** Changes the `gpu_resident_batch` execution strategy
only.

## Goal

Make the cost of a step scale with the wet part of the mesh rather than the
whole mesh, without changing results.

## Problem

After [effort 21](../21-resident-clock-and-batch-loop/results.md) the Nysa
24 h run (2.35 M cells) is kernel-bound at 1.77 ms per step, yet at most
2.3 % of cells are ever wet. Every step still runs the flux kernel over all
edges and the update kernel over all cells. Dry edges return early but still
read their cell indices and depths, and dry cells still read and write their
whole state.

## Design

**Blocks.** Cells are in Hilbert order, so each run of 256 consecutive cells
(one update workgroup) is a compact patch of mesh. Each block has two flags:
`wet` (holds a cell that is wet or still has momentum) and `active`.

**Exact halo.** A cell receives flux only through an edge with a wet side.
Within one Heun step the predictor can wet the first ring of neighbours of the
step's initially wet cells, and the corrector can then push flux one ring
further. Every nonzero flux and every changed cell therefore lies within two
cell rings of the initially wet cells. Each cell ring is contained in one ring
of blocks, so `active` = `wet` dilated over each block's two-ring of
neighbouring blocks covers every cell and edge the full step would change.
Skipping the rest is exact: a dry cell with no flux and zero momentum is left
unchanged by the update.

Two details keep it exact:

- A skipped cell's predictor state (`h1`, `hu1`, `hv1`) is stale. The
  corrector flux reads the start-of-step state instead for any cell in an
  inactive block; the full step would have computed exactly that state for it.
- The corrector averages momentum, so a cell that just dried keeps half of its
  old momentum, which the full solver halves every step. A block is therefore
  marked while any of its cells is wet or has nonzero momentum, not only while
  wet.

**Per-step rebuild without a full pass.** Wet cells are always inside active
blocks, so the corrector update can set `wet` for its block when the final
depth is at least dry_tol or the final momentum is nonzero. The source kernel
does the same for source cells. One small kernel over blocks then sets
`active` from `wet` over the two-ring, which the host precomputes. It runs
before `step_control` and skips no-op steps, so the flags survive steps with
dt = 0. The predictor update clears `wet` for its block; a marked block is
always active the next step, so every mark is cleared before it goes stale.
A first version used two one-ring passes; folding them into one dispatch
saved a dispatch and a barrier per step.

**Kernels.**

- Update: a workgroup returns at once if its block is not active.
- Flux: edges are sorted by the block of their left cell when the solver is
  built. Each edge workgroup stores the block range its edges belong to and
  returns at once if none of those blocks is active.
- Source: a compact list of (cell, rate) pairs instead of a pass over all
  cells.
- Prologue: a full mark pass (wet or nonzero momentum), then the dilation.

The CFL bound is unchanged: dry edges never contributed to it.

## Rejected alternatives

- **Compacted cell and edge lists.** Kompute has no indirect dispatch, so the
  grid size cannot follow the list length. Workgroups would still be launched
  for the whole mesh; per-block skipping gives the same saving more simply.
- **Rebuild every K steps with a wider halo.** It needs 2K rings of margin,
  and the per-step rebuild here costs only one pass over about 9 k blocks.

## Tasks

1. Block adjacency, edge sort and per-workgroup block ranges on the host.
2. Resident flux, update and source kernels with block skipping; mark and
   dilate kernels.
3. Record the dilation into the step; full mark in the prologue; a switch
   that turns skipping off, for testing.
4. Tests: host-side block helpers on CPU; on GPU, runs with skipping on and
   off agree (wetting front on a dry slope, dam break, sources, resume).
5. Gates: lake at rest and the five report suites.
6. Measure: 47 k grid, Nysa 10-minute probe and the full 24 h run.

## Out of scope

- Float32 state floor (roadmap priority 1).
- Other execution strategies.
