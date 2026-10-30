# Solver Accuracy Fixes — Summary

**Status (2026-10-30):** ready for independent review, not yet ready for
production. Branch `feat/solver-accuracy-fixes`, not merged. Full evidence:
[`results.md`](results.md). Design: [`plan.md`](plan.md).

## Results

Nysa 24 h, real terrain, 2.35 M cells:

| Measure | Before | After |
| --- | ---: | ---: |
| Water deficit | 1.26 % | 0.0000 % |
| Wall time, single runs | 1 527 s | 830 s |
| Steps | 2 744 246 | 1 168 669 |
| Clamp mass | not recorded | 0 |
| Peak memory | 7.8 GB | 7.6 GB |

CFL 0.8 stays within 2.1 cm of a CFL 0.4 reference run.

Benchmarks:

- Volume drift: floodplain depressions 1.0e-4 to 4.0e-9, flood propagation
  2.6e-4 to 2.3e-8.
- Steps: 40–50 % fewer on the floodplain, flood propagation and momentum
  obstruction cases.
- Speed: 1.9× faster on a fully wet 32 k cell mesh for the same simulated time.
  Each step costs 11–18 % more.
- Lake at rest: velocity about 5× lower on the default backend.
- Parabolic bowl, a new test with an exact solution: errors 6–10 % lower.

## What changed

- Compensated depth keeps the tiny increments that were being rounded away.
- Dry cells stop carrying momentum.
- Wet-dry fronts use dry-bed wave speeds that blend in smoothly.
- Thin-film damping prevents unrealistic velocities in nearly dry cells.
- Bed height differences are precomputed per edge, so high terrain loses no
  precision.
- Default CFL on `gpu_resident_batch` rose from an effective 0.36 to 0.8.
  Host-driven backends stay at 0.45.
- Sources limit the next timestep, so a point source cannot drive a cell below
  zero.
- Large meshes no longer lose their last cells, sliver cells are rejected,
  `gpu_resident_batch` is the default, and `fixed_dt_batch` refuses to run.

## Verification

- 413 unit tests pass with the GPU.
- All six gate suites pass on MoltenVK, including the parabolic bowl.
- Fixes were deliberately broken to confirm that tests catch them.
- Strict pyright, ruff and markdown lint pass.

## Known costs

- Ritter dam-break front error rose from 12.9 % to 13.9 %.
- Flood propagation speed error rose from 1.27 % to 1.40 %. It passes its gate.
- Suite wall time is unchanged on flood propagation; those timings are noisy.
- A rare transient of up to 3.5e-5 m remains on the non-default atomic path. Its
  trigger is not identified.

## Next steps

1. Get an independent review of the code, especially the shader changes.
2. Run the tests and gates on the NVIDIA production GPU. `precise` stays on there
   and the fast-math setting does not apply, so its speed and compensation
   behaviour are unknown.
3. Add Node.js to the Docker image: `just typecheck` fails on a fresh ARM image.
4. Rerun grid convergence on the final code.
5. Check old flood maps. The old code differs from the new runs by up to 1.48 m
   in some cells, and the locations have not been reviewed.
