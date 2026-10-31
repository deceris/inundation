# Roadmap

## Current scope

Maintain the Vulkan/Kompute solver, its application workflow, mesh processing,
and reproducible analytical benchmarks.

## Current evidence

All retained analytical and EA gates pass with the MoltenVK reference. Grid
convergence shows expected profile order and confirms Test 4 front bias shrinks
with resolution. The float32 mass drift is understood and removed on the
resident backend: increments below half an ulp of the depth were rounded
away, and the compensated depth pair keeps them
([effort 24](../implementation/24-solver-accuracy-fixes/results.md)).

## Priorities

1. Run the unit and gate suites on the NVIDIA production path, where `precise`
   stays on and the fast-math setting does not apply.
2. Second-order spatial reconstruction (MUSCL), or forward Euler in place of
   Heun: the scheme is first order in space with a second-order integrator.
3. Optimize only after a measured Vulkan bottleneck identifies a useful target.

## After open boundaries

Two-way basin coupling follows
[effort 29](../implementation/29-open-boundaries/plan.md): basins in lockstep that pass
stage back upstream, so backwater can cross a cut. Effort 29 builds the one-way exchange.

## Invariants

- Repeated runs remain deterministic within each benchmark's stated tolerance.
- Kernel restructuring preserves per-cell CSR gather order.
- New execution strategies match existing physical invariants before benchmark
  results are accepted.
- Numerical-scheme changes require a new baseline rather than relaxed checks.
