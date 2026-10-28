# Deterministic Accumulation — Results

**Status: in progress** (2026-10-27). The final 24 h Nysa run with the
merged code is still to do; the Nysa figures below come from the prototype.
Plan: [`plan.md`](plan.md).

All measurements native on M1 Max with MoltenVK, `gpu_resident_batch`.

## Exactness

- Repeat runs are bit-identical: wetting front from a source, dam break, and
  a draining sheet on a 32 768-cell Hilbert-ordered valley, compared bit for
  bit in every snapshot and in the final depth and momentum.
- Skipping on and off now agree exactly, where with atomics they agreed only
  to 1e-5 m. The only bit difference is the sign of a zero: a skipped dry cell
  keeps a −0.0 momentum from its initial state, which the full step rewrites
  as +0.0.
- The gather path matches the atomic path to 1e-5 m with the same step count.
- Huge values planted in the edge buffer, alone or with wet garbage in the
  predictor state, leave the results bit-identical: the gather never reads an
  edge value that was not written in the current stage.
- One solver switched to atomics and back reproduces its first run bit for
  bit.

Mutation checks: each of these, introduced on its own, fails at least one
test:

- gathering every edge, without the dry-edge test;
- the flux kernel not marking its cells;
- the update not clearing the marks.

Three mutations survive and are equivalent:

- a wall edge judged by the cell's own depth instead of skipped;
- the CFL term read in the predictor, where it is unused;
- the neighbour test ignoring the inactive-block rule, because a marked dry
  cell sits next to water and all its neighbours are in active blocks.

## Gates

- Lake at rest, variants A/B × friction minimal/production: errors 2e-5 to
  7e-5 m, as before.
- Benchmark report (two repeats each): dam break, radial dam break, floodplain
  depressions, momentum obstruction and flood propagation all pass with no
  baseline change. Every determinism check now reports a maximum relative
  difference of 0, against 3.8e-7 to 3.1e-6 with atomics. Suite wall times are
  unchanged within noise (floodplain depressions 206 → 199 s, flood
  propagation 254 → 251 s).
- Unit tests: 298 pass.

## Cost per step

| Case | Cells | Atomics | Gather |
| --- | --- | --- | --- |
| Flat grid, fully wet, 1 h simulated | 47 089 | 0.304–0.308 ms | 0.300–0.306 ms |
| Valley, wetting and dam break, 600 s | 32 768 | 0.217–0.236 ms | 0.205–0.250 ms |
| Nysa 24 h, prototype, run back to back | 2.35 M | 1 783 s | 1 601 s |

The 47 k figures are from a slower session than effort 22's 0.25 ms; both
paths were measured together.

On Nysa the prototype went through three versions, all bit-identical to each
other and across repeats:

| Prototype | Nysa 24 h wall time |
| --- | --- |
| Every edge written, every cell gathers | 2 737–2 815 s |
| Dry edges write nothing | 2 218 s |
| Cells with no live edge skip the gather | 1 601 s |

Per 30-minute window, the gather path gains most mid-run (0.54 against
0.74 ms per step at 12 h simulated); at 24 h both cost about 0.75 ms. The
merged code differs from the prototype only in gathering each cell's edges in
the mesh CSR order instead of edge order.

## Nysa 24 h with the merged code

Not yet measured: the run was stopped at 59 % by host memory pressure, not by
the solver.
