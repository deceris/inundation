# Momentum Snapshots on the Resident Backend — Results

**Status: done on MoltenVK.** Not run on NVIDIA.

## Backend support

Only `baseline` and `gpu_resident_batch` capture momentum.
`fixed_dt_batch_barrier`, `device_cfl`, `batched_submit` and `async_sync_window`
accept the flag and ignore it; `async_sync_window` delegates only with
`source_fn`, which the workflow never passes. `fixed_dt_batch` refuses to run. A
direct `solver.run()` call on those backends still ignores the flag: only the
workflow raises.

## Nysa 24 h

2.35 M cells, CFL 0.8, 69 snapshots, 1 168 669 steps.

| | Flag on | Flag off |
| --- | --- | --- |
| Wall time | 1 360 s | 1 387 s |
| Peak memory footprint | 6.4 GB | 4.6 GB |

Added time per snapshot is about 2.8 ms (median 1.47 ms for `hu`, 1.34 ms for
`hv`), or 0.2 s over the run. All 69 depth snapshots and the final `h`, `hu` and
`hv` are bit-identical between the two runs and to the flag-off result at
`0a0ec77`.

Both runs shared the GPU with another application, so these wall times cannot be
compared with the 830 s baseline; the 27 s gap between them is noise. The
expected memory increase is about 1.3 GB; the measured 1.8 GB includes copies
made by the measuring script.

## Comparison with the barrier backend

Relative L2 distance of (`hu`, `hv`) over the mesh, on the 32 k cell valley with
a point source, at 20, 40 and 60 s: 0.29 %, 0.12 %, 0.22 %. The test limit is
1 %.

## Tests

26 new tests. Before the fix, 10 of the 12 GPU tests failed. Depth bit-identity
passed because the flag did nothing, and `baseline` already delivered. Reading
`hu1`, dropping the initial snapshot and dropping the tail snapshot were each
caught.

## Gates

All six suites pass on `4c9fa8b`. Every correctness metric in their summaries is
identical to the previous run on `0a0ec77`; only wall-time fields differ.

## Known limitation

The final tail snapshot is kept only if `np.allclose` sees a depth change. With a
tail of 2⁻¹⁶ s the snapshot was dropped and the last momentum snapshot differed
from `hu_final` by 2e-7. Momentum stays aligned with `snap_times`, but the run's
last ticks have no snapshot. The behaviour comes from the depth path, and
changing it would change flag-off output.
