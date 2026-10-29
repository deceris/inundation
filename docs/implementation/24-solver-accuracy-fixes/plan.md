# Solver Accuracy Fixes — Plan

**Status: done.** Changes the numerical scheme (new baseline), the geometry
format (cache version 5) and the `gpu_resident_batch` default settings.

## Goal

Close the correctness flaws found in a review of the kernels, each with a
test that fails on the old kernel, and make the mass balance of a run
exact up to the positivity clamp.

## Problems

1. **Dry cells kept half their momentum.** The Heun corrector averages the
   start-of-step momentum with the stage result; the stage zeroes momentum in
   a dry cell, so a cell that dried kept half of its momentum, halved it every
   step for ~150 steps, and fed it back into the flow when it wetted again.
   Fronts flip wet and dry constantly.
2. **Wet-dry wave speeds were underestimated.** The Roe-average HLLC estimate
   gives about 0.7c on the dry side of a face while the dry-bed rarefaction
   tip runs at 2c. The CFL bound used the same estimate.
3. **The hydrostatic reconstruction cancelled in absolute datum.** `h + zb -
   z_face` in float32 at a bed 300 m above the datum loses the low bits of
   the surface; on the valley test the momentum error against a float64
   reference was 10-30x the datum-0 value, and at 3000 m the depth error
   reached 1.4e-4 m.
4. **Mass drift.** `h - dt/A * flux` rounds away any increment below half an
   ulp of `h`; the sign is set by the flow regime, so it is a systematic
   drift that grows with the step count — the shape of the Nysa 1.26 %
   deficit and of the Test 4 drift that worsens with refinement.
5. **Robustness.** Element counts travelled as float32 values (exact only to
   2^24, ~11 M triangles); a sliver cell below the CFL kernels' area floor
   got no dt bound and would divide by its area; the workflow defaulted to the
   per-dispatch-synchronised baseline backend; the racy `fixed_dt_batch` was
   still selectable.
6. **Courant number.** The resident path multiplied the CFL number by a 0.8
   safety factor inherited from the batch backends, which reuse one dt for
   several steps; it recomputes the bound every step.

## Design

**Reference step.** `tests/unit/_cpu_step.py` is a float64 NumPy mirror of
one Heun step (reconstruction, HLLC, dry tests, friction, average). It states
the intended physics; GPU parity tests compare eight fixed-dt steps and the
prologue CFL bound on every backend against it, so each kernel change starts
red.

**Momentum.** The corrector zeroes momentum when the averaged depth is below
dry_tol. The active-set wet test reduces to "depth at least dry_tol".

**Wave speeds.** `hllc_speeds` switches to Toro's dry-bed estimates when one
reconstructed side is below dry_tol. The flux kernels return the largest wave
speed they assumed, and the resident corrector accumulates the CFL bound from
it; the prologue and host-driven CFL kernels reconstruct the face states the
same way (`wave_bound`). One shared GLSL snippet holds reconstruction, solver
and bound; the host-driven flux kernel no longer carries its own copy.

**Bed steps.** `build_geometry` computes per edge `dzL = z_face - zbL` and
`dzR` in float64 from the float32 bed; the kernels reconstruct `h - dz`. The
bed is no longer read on the device: binding 3 carries the `(dzL, dzR)`
pairs, one coalesced edge read instead of two random cell reads.

**Compensated depth.** On `gpu_resident_batch` the depth is an exact float32
pair `(h, h_lo)`, `|h_lo| <= ulp(h)/2`. Both stages, the Heun average and the
sources add through Knuth's two-sum. The flux kernels read `h` alone;
`download_h()` is unchanged and `download_h_precise()` returns `h + h_lo` in
float64, which the workflow uses for the final volume. Switch:
`compensated` (default on), `WorkflowConfig.compensated`.

Two findings on the way: a wall edge now passes exactly zero mass flux (the
mirrored Riemann problem gave zero only to float32 rounding, with the sign of
the flow), and MoltenVK compiles with Metal fast math by default and its
on-demand mode ignores the GLSL `precise` qualifier, which reassociates the
two-sum away. The solver sets `MVK_CONFIG_FAST_MATH_ENABLED=0` before the
first `kp.Manager()` unless the environment sets it.

**Robustness.** Counts are pushed as the float whose bits they are and the
shaders declare them `uint`; every inline push-constant list goes through a
helper so the struct-order test covers all backends. `build_geometry` rejects
cells with area at or below `MIN_CELL_AREA` (1e-10 m², the kernels' floor).
`WorkflowConfig.solver_impl` defaults to `gpu_resident_batch`.
`SWESolverFixedDtBatch.run()` raises; the barrier subclass is unchanged.

**Courant.** `cfl_safety` becomes a resident-backend attribute, default 1.0.
`CFL_DEFAULT` is set from a sweep of the five gate suites.

## Tasks

1. Reference step and parity tests. 2. Momentum and wave speeds. 3. Bed
steps and cache version 5. 4. Compensated depth, wall flux, fast math.
5. Robustness. 6. Courant sweep (`--cfl` on every harness and on the report
runner). 7. Gates after every step; lake at rest on the result.

## Out of scope

- Second-order spatial reconstruction (MUSCL) or forward Euler in place of
  Heun: a scheme change with its own baseline.
- Compensation on the host-driven backends.
- Bit-identity across devices or drivers.
