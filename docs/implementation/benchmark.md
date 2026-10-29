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
| Open-boundary forcing | passing | Steady flat channel: `hu` within 1.4e-3 of the prescribed inflow, mass balance 2.5e-6; wall-only runs unchanged, no measurable slowdown | [`21-open-boundaries/`](21-open-boundaries/results.md) |
