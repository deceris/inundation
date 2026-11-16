# Backend Pruning - Results

Radial dam-break (t_end 1.5 s, flat bed), MoltenVK on Apple M1 Pro. Median of
three timed runs after one warmup; wall time is the solver loop only.

## Correct backends

| Mesh | Backend | CFL | Steps | Wall s | Steps/s | Mcell-steps/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 400² | `gpu_resident_batch` | 0.80 | 357 | 0.33 | 1076 | 172 |
| 400² | `gpu_resident_batch` | 0.45 | 634 | 0.64 | 997 | 160 |
| 400² | `fixed_dt_batch_barrier` | 0.45 | 793 | 1.19 | 664 | 106 |
| 400² | `baseline` | 0.45 | 793 | 2.90 | 273 | 44 |
| 200² | `gpu_resident_batch` | 0.80 | 175 | 0.084 | 2078 | 83 |
| 200² | `gpu_resident_batch` | 0.45 | 311 | 0.141 | 2206 | 88 |
| 200² | `fixed_dt_batch_barrier` | 0.45 | 389 | 0.176 | 2211 | 88 |
| 200² | `baseline` | 0.45 | 389 | 0.636 | 612 | 24 |

All pass the radial gates (400²: L1 0.0284, front 0.011). Volume drift is
about 2e-11 resident and 3e-8 on the host-driven pair; only the resident
backend repeats bit for bit. At its default CFL the resident backend is
3.6–3.9× faster than the barrier backend, mostly through fewer steps; at equal
CFL the gap is 1.2× (200²) and 1.8× (400²).

## Deleted backends

| Backend | Outcome |
| --- | --- |
| `batched_submit` | L1 up to 1.6e10, volume drift up to 9.5e15, 10–371 s as dt collapsed |
| `device_cfl` | L1 up to 3e15, volume drift up to 4e15 |
| `async_sync_window` | `CFL dt too small` at every size |

Cause, from the code: each recorded dependent dispatches (CFL accumulate →
reduce → resolve, flux → update → flux → update) in one sequence without
`kp.OpComputeBarrier`, the race recorded for `fixed_dt_batch`. None updated
`steps_total`, so `WorkflowResult.steps_total` read 0. No test checked their
numerical output.

## Not re-measured

The removal touches no kernel the remaining backends dispatch, so the gate
suites were not re-run; the GPU unit tests cover the remaining backends.
