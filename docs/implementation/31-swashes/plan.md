# SWASHES Analytic Benchmarks - Plan

**Status: done.** Results: [`results.md`](results.md).

## Goal

Validate the solver, open boundaries included, against SWASHES (Delestre et al.
2013, [doi:10.1002/fld.3741](https://doi.org/10.1002/fld.3741)), a compilation of
shallow-water analytic solutions covering steady flow over a bed, friction,
hydraulic jumps, transcritical flow and moving shorelines. It is the only
independent analytic check of the effort 29 boundary conditions, and it runs on
two backends, so the second one checks the resident kernels at the boundary.

## Scope

In: steady flow over a bump (subcritical, transcritical, with a shock), lake at
rest with an emerged bump, MacDonald long channel subcritical and short channel
with a smooth transition and a shock (Manning), Thacker's planar oscillation in
1D and 2D. Open boundaries on `baseline`, so the suite runs on both backends.

Out: MacDonald with rain (depth imposed at both ends), pseudo-2D variable-width
channels, Thacker radial, Dressler and Sampson (non-Manning friction), bedload,
solute and spherical cases.

## Reference

The SWASHES program itself, from the `swashes` optional extra (PyPI
`swashes==1.5.0`, SWASHES 1.05.00, CeCILL-2.0). The wheels ship a prebuilt
binary for macOS and Linux, so nothing compiles; the sdist only downloads the
C++ source from a server that does not always answer. The harness asks for the
solution at exactly the mesh's cell centres (`swashes dim type domain choice nx
[ny]`), so the comparison is cell by cell with no interpolation and no ported
formulas.

## Design

- `inundation/bench/swashes.py`: one `Case` per SWASHES solution with its bed,
  initial state, boundaries and Manning coefficient, all taken from the binary's
  output header. 1D cases run on a 2-row channel.
- Boundaries use the effort 29 stretches: the west inlet is `Inflow` (SWASHES'
  unit discharge times the width; the section is level, so the conveyance split
  is uniform) and the east outlet is `Stage` at the bed plus SWASHES' outlet
  depth. `Stage` already lets go of the level once the outflow is supercritical,
  which is SWASHES' "imposed while subcritical" outlet, so no new kind is needed.
- Steady cases start at rest at the SWASHES free surface and run until the
  start-up waves leave; periodic (Thacker) cases start from the SWASHES state at
  a whole number of periods and run one period.
- Gates: depth L1, discharge L1, and mass balance against the measured stretch
  volumes, relative to the largest of stored, final and inflowed volume.
- Baseline: the host flux and CFL kernels build the same ghost state as the
  resident kernels (`OPEN_GHOST_GLSL`, moved to `shaders/flux.py`) and record
  each open edge's volume as a compensated pair, one writer per edge.
- CPU tests (`tests/unit/test_bench_swashes.py`) pin the reference layer: the
  binary's output lands on the mesh's cell centres in mesh order.
