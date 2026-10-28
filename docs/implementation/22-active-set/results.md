# Active Set — Results

**Status: done** (2026-10-27). Plan: [`plan.md`](plan.md).

All measurements native on M1 Max with MoltenVK, `gpu_resident_batch`.

## Exactness

Each GPU test runs the same problem with skipping on and off (off marks every
block active, which is the full step). Step counts match exactly; depth and
momentum agree within 1e-5, the run-to-run noise of float atomics. Cases:

- wetting front from a source on a dry slope;
- dam break onto a dry slope;
- a sheet draining off a slope;
- chained runs with changing sources;
- garbage in the predictor buffers, which skipped blocks must not read;
- dry cells carrying momentum, which must decay as in the full step.

Mutation checks: each of these, introduced on its own, fails at least one
test:

- reading the stale predictor state;
- marking blocks for wet cells only;
- dilating one ring instead of two;
- dropping the source mark;
- never clearing marks.

A CPU test checks the halo claim directly: on a Hilbert-ordered mesh, two cell
rings around random wet sets always fall inside the block two-ring.

## Gates

- Lake at rest, variants A/B × friction minimal/production: pass, errors
  2e-5 to 7e-5 m as before.
- Benchmark report: dam break, radial dam break, floodplain depressions,
  momentum obstruction and flood propagation all pass with no baseline
  change. Suite wall time fell for the partly dry cases: floodplain
  depressions 387 → 206 s, radial dam break 75 → 49 s.
- Unit tests: 264 pass.

## Cost per step

| Case | Cells | Effort 21 | This change |
| --- | --- | --- | --- |
| Flat grid, fully wet, 1 h simulated | 47 089 | 0.22–0.24 ms | 0.25–0.27 ms |
| Sloped grid, source, 15 % of blocks active | 240 000 | 0.256 ms (skipping off) | 0.238 ms |
| Nysa, 10-minute probe | 2.35 M | 1.96 ms (15.5 s) | 0.39–0.47 ms (3.1–3.7 s) |
| Nysa, full 24 h | 2.35 M | 1.77 ms (4 855 s) | 0.65 ms (1 789 s) |

- On Nysa the full run is 2.7 times faster: 30 minutes instead of 81 for
  24 h simulated. System time fell from 411 s to 275 s.
- A fully wet mesh pays about 0.02–0.03 ms per step for the extra dilation
  dispatch. Below about 250 k cells the step is dominated by dispatch
  overhead, so skipping gains little there.

## Nysa 24 h run against effort 21

| Metric | Effort 21 | This change |
| --- | --- | --- |
| Steps | 2 744 249 | 2 744 231 |
| Volume in domain | 3 692 313 m³ | 3 692 241 m³ |
| Wet cells (h_max > 5 cm) | 53 783 | 53 781 |
| Mean \|Δh_max\| over wet cells | — | 8.4e-5 m |
| 99th percentile \|Δh_max\| | — | 1.1 mm |
| Max \|Δh_max\| | — | 14 mm |

The runs differ only in summation order (float atomics, and edges now sorted
by block), and that difference grows over 2.7 M steps of unsteady flow. The
on/off tests above show the skipping itself is exact. The 1.26 % volume
deficit is unchanged; see effort 21.
