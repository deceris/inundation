# Deterministic Accumulation — Plan

**Status: implemented; final Nysa run pending.** Changes the
`gpu_resident_batch` execution strategy only.

## Goal

Make repeated `gpu_resident_batch` runs bit-identical on the same device and
driver, without making them slower.

## Problem

The resident flux kernel adds each edge's flux into its two cells with float
`atomicAdd`, and the corrector and the prologue accumulate the CFL bound the
same way. The GPU applies those additions in a different order on every run,
so the last bits of each sum differ. Over 2.7 M steps of unsteady flow the
differences grow: two identical 24 h Nysa runs differ by 0.07 mm on average
and by up to 1 cm in single cells
([effort 22](../22-active-set/results.md)). The roadmap invariant "kernel
restructuring preserves per-cell CSR gather order" did not hold for this
backend.

## Design

**Per-edge results, per-cell gather.** The flux kernel writes each edge's
contributions to an edge buffer instead of adding them into the cells: the
mass flux, the left and right momentum contributions (they differ by the
hydrostatic correction), and in the corrector the CFL wave-speed term. Six
floats per edge. The update kernel then sums each cell's edges in the fixed
order of the mesh's per-cell CSR (`cell_edge_ptr`, `cell_edge_idx_signed`,
remapped to the solver's block-sorted edge order). The prologue CFL bound is
computed the same way. Remaining atomics are order-independent: `atomicMin`
for dt and the block flag stores.

**Dry edges write nothing.** The flux kernel returns early on an edge whose
two sides are both dry in the stage's input state, as before. The gather
applies the same test, so it never reads a value that was not written this
stage. The test reads the neighbour's depth with the flux kernel's rule
(predictor state in an active block, start-of-step state otherwise), so both
kernels see the same bits.

**Idle cells skip the gather.** The flux kernel marks both cells of every
edge it writes (a store of the same value from every writer, so the race is
benign). The update kernel reads and clears its cell's mark; an unmarked cell
has no live edge and gathers nothing. In the dry margin of the active set this
saves most of the gather's reads.

**Skipped edges.** An edge whose left cell's block is inactive is not computed
at all. Such an edge is dry on both sides in both stages (the active set's
two-ring argument), so the gather's dry test excludes it.

**Switch.** `deterministic` on the solver (default on) and
`WorkflowConfig.deterministic` select the gather or the atomic kernels; both
are built, so the switch can change between runs. The atomic path is kept as
a fallback and as a comparison for tests.

## Prototype evidence

A prototype subclass measured on the 24 h Nysa run (2.35 M cells, active set
on), run back to back with the atomic path:

| Variant | Wall time |
| --- | --- |
| Atomics | 1 783 s |
| Gather, every edge written | 2 737–2 815 s |
| Gather, dry edges skipped | 2 218 s |
| Gather, dry edges and idle cells skipped | 1 601 s |

All gather variants gave bit-identical output to each other and across
repeats.

## Tasks

1. Host: remap the per-cell CSR to the block-sorted edge order.
2. Shaders: gather variants of the resident flux and update kernels;
   per-edge CFL and gather-reduce kernels for the prologue.
3. Solver: `deterministic` switch; edge, CSR and mark buffers; barriers.
4. Workflow: `WorkflowConfig.deterministic`.
5. Tests: repeat runs bit-identical; skipping on and off bit-identical; gather
   close to atomics; garbage in the edge buffer must not be read; marks are
   clear after a run; host remap on CPU.
6. Gates: lake at rest and the five report suites.
7. Measure: 47 k grid and the Nysa 24 h run.

## Out of scope

- Bit-identity across devices or drivers: compilers may contract or reorder
  arithmetic differently.
- The other execution strategies, which keep their atomics.
