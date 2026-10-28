# Benchmark Ledger

| Item | Status | Headline result | Detail |
| --- | --- | --- | --- |
| Lake-at-rest Vulkan baseline | passing | Barrier race and hydrostatic imbalance corrected | [`00-lake-at-rest/`](00-lake-at-rest/results.md) |
| Stoker and Ritter dam-break | passing | Closed-form depth and front gates pass on MoltenVK | [`13-analytical-dambreak/`](13-analytical-dambreak/results.md) |
| Radial dam-break | passing | Axisymmetric profile and isotropy gates pass on MoltenVK | [`14-radial-dambreak/`](14-radial-dambreak/results.md) |
| Floodplain depressions | passing | Published terrain storage and reach gates pass | [`15-floodplain-depressions/`](15-floodplain-depressions/results.md) |
| Momentum obstruction | passing | Release overtops while still-water control remains dry | [`16-momentum-obstruction/`](16-momentum-obstruction/results.md) |
| Hydrostatic correction | done | Lake-at-rest currents reduced below 1e-4 m/s | [`17-audusse-well-balance-correction/`](17-audusse-well-balance-correction/results.md) |
| Flood propagation | passing | Arrival, depth, speed, and isotropy gates pass; mass drift is a single signed error and the clamp contributes zero | [`18-flood-propagation/`](18-flood-propagation/results.md) |
| Grid convergence | done | Expected profile order; front bias is discretization error; mass drift changes sign and the clamp contributes zero | [`19-grid-convergence/`](19-grid-convergence/results.md) |
| Mesh geometry cache artifacts | done | 4000-cell mesh: artifact reloads identical to a direct build; `prepare()` runs with the mesh file deleted (GPU allocation untested — no Vulkan device in the dev container) | [`20-mesh-cache-artifacts/`](20-mesh-cache-artifacts/results.md) |
| Resident clock and batch loop | done | Start time no longer affects injected volume (t0 = 0 and 70 000 s agree to 0.01 %); 47 k cells 0.6–1.1 → 0.23 ms per step; Nysa probe 2.37 → 1.96 ms, 24 h run kernel-bound (1.77 ms); Nysa deficit 1.26 % with zero clamp, attributed to float32 state; all gates pass | [`21-resident-clock-and-batch-loop/`](21-resident-clock-and-batch-loop/results.md) |
| Active set | done | Skips 256-cell blocks beyond two block rings of water, same results as the full step; Nysa 24 h 4 855 → 1 789 s (1.77 → 0.65 ms per step); all gates pass | [`22-active-set/`](22-active-set/results.md) |
| Deterministic accumulation | in progress | Per-edge writes and fixed-order per-cell sums replace float atomics: repeat runs and skipping on/off bit-identical, benchmark determinism diffs 0; all gates pass; Nysa 24 h prototype 1 601 s against 1 783 s for atomics | [`23-deterministic-accumulation/`](23-deterministic-accumulation/results.md) |
