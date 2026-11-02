# Second-Order Reconstruction — Plan

**Status: done, see [results.md](results.md).** Roadmap priority 2. The scheme is first order in space with a
second-order integrator. This adds linear reconstruction as an option on
`gpu_resident_batch`.

## Goal

Lower the error on smooth flow at the same mesh, without losing what the first-order
scheme holds: lake at rest, positivity, exact mass, determinism and active blocks.

## Design

`reconstruction` on `WorkflowConfig`, `"first_order"` by default. First-order runs use
the same kernels as now, so every baseline stays and they stay bit-identical. Changing the
default is a separate decision, because numerical-scheme changes need a new baseline.

**Gradients.** A new kernel runs before the flux in both Heun stages. For each cell it
takes the Green-Gauss gradient of water level, x velocity and y velocity from differences
to its neighbours, in the cell's fixed edge order. The level difference comes from
depths and bed steps, so the kernels still never read the bed.

**Limiter.** Barth-Jespersen per variable: no face value goes beyond the cell and its
neighbours. A cell next to dry ground, or close to dry itself, uses no gradient, which
is the first-order scheme at wetting fronts.

**Face states.** Level and velocity move from the centroid to the edge midpoint, using
a new per-edge offset. Bed height moves with a static per-cell bed gradient, so depth at
the face is level minus bed. Audusse reconstruction then runs as it does now.

**Bed source.** The correction becomes the Audusse-Bristeau form, the interface term from
the reconstructed depth plus a centred term from the bed slope in the cell. At rest the
level gradient is zero and the terms cancel exactly.

**Boundaries.** Walls and open edges stay first order, so the exchange at a basin border
is first order. A gradient treats them as a zero difference.

## Gates

- With `first_order`, Nysa stays bit-identical and every suite matches effort 30.
- With MUSCL, lake at rest to rounding on a bumpy bed.
- With MUSCL, measured order above 1.5 on a smooth case, against about 1 first order.
- With MUSCL, every analytical suite passes, clamp mass 0, deterministic.
- Nysa 24 h with MUSCL: stable, and the cost per step and CFL measured.
- A mutation check of the new kernel code, and `just check` green.

## Not in this effort

Changing the default, and second order at walls or borders.
