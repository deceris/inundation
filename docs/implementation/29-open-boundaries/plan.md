# Open Boundaries — Plan

**Status: design choices settled, build waiting for approval.** Nothing is built.
Third of the four solver items for the pavé meshes.

## Goal

Let water leave a mesh at a named stretch, and let it enter at one, with the volume
through each stretch measured. The approach is per catchment, so passing water from
one basin to the next is the main use: inflow matters as much as outflow.

## Problem

Every mesh edge with no right cell is a reflective wall, and a kernel marks it with
`ecR < 0`. Water reaching the rim piles up and reflects. Point sources are the only
way in. They add no momentum and need a point inside the mesh.

## Naming a stretch

A stretch is a line in world coordinates, set once for the run. It selects the
boundary edges whose midpoints lie within a tolerance of the line.

```python
WorkflowConfig(
    open_boundaries=[
        OpenBoundary("outlet", [(x0, y0), (x1, y1)], tolerance_m=5.0),
        OpenBoundary("from_upstream", [(x2, y2), (x3, y3)], tolerance_m=5.0),
    ],
)
SimulationPhase(3600.0, sources, boundaries={
    "outlet": NormalDepth(energy_slope=1e-3),
    "from_upstream": Inflow(discharge_m3s=12.0, energy_slope=1e-3),
})
```

- A stretch selecting no edges raises, unlike a point source, which is skipped. So
  does an edge selected twice, or a phase that leaves a stretch out.
- Conditions change per phase, as sources do. A hydrograph is a list of phases.
- The mesh, `MeshGeometry` and the cache do not change, and neither does the mesher.

## Conditions

Each condition builds a ghost state for the edge, and the existing HLLC flux, CFL
bound and momentum code run on it. `n` is the outward normal. Boundary edges already
have zero bed steps, so a ghost needs no reconstruction.

| Condition | Ghost depth | Ghost normal velocity | Mass flux |
| --- | --- | --- | --- |
| `Wall` | `hL` | `-unL` | forced to 0, as today |
| `FreeOutflow` | `hL` | `unL` | from HLLC, never inward |
| `NormalDepth(S)` | `hL` | `sqrt(S) * hL^(2/3) / n` | from HLLC, never inward |
| `Stage(eta)` | `max(eta - zb, 0)` | `unL + 2 (cL - cG)` | from HLLC, either way |
| `Inflow(Q, S)` | solved, see below | `-q / hG` | forced to `-q` |

- **`FreeOutflow`** passes supercritical flow exactly. On subcritical flow it only
  partly absorbs, so it suits a steep cut.
- **`NormalDepth`** is the local form of a rating curve, and the default for a
  subcritical outlet with a known slope. `n` is the cell's Manning value.
- **`Stage`** sets a downstream level. Supercritical outflow uses the interior state
  instead, because the level cannot reach the boundary.
- **`Inflow`** splits `Q` over the stretch by conveyance. The host finds the level at
  which `sum(len * h^(5/3) * sqrt(S) / n) = Q` over the edges, and gives each edge its
  `q`, once per phase. Water enters along the low part of the section and not over
  the whole line. Each stage then solves `hG` from the outgoing characteristic,
  `-q/hG + 2 sqrt(g hG) = unL + 2 cL`. The left side rises with `hG`, so there is
  always exactly one root, and four Newton steps run in fixed order. On a dry cell
  this gives `(q^2 / 4g)^(1/3)`, close to critical depth.

Forcing the mass flux for `Inflow` is the same move as the wall's `Fh = 0`: it makes
the volume exact and leaves momentum to the ghost state.

A rating curve over a whole stretch, `Q(stage)`, is left out. It needs a stretch-wide
stage every stage of every step, which is a new reduction on the device, and
`NormalDepth` covers the common case.

## Kernels

- `ecR` encodes the edge: `-1` is a wall as today, `-2 - k` is open edge `k`. Tests of
  `cR >= 0` keep working. Only the uploaded tensor changes, and the host's `-1` stays.
- Binding 19, unused by the flux kernels, holds `(kind, value, sqrt(S)/n, ...)` per open
  edge, uploaded at each phase. The flux kernel has 27 of the device's 31 bindings.
- One GLSL function builds the ghost. The flux kernel and the prologue CFL kernel both
  call it, so the bound sees what the flux sees.
- An open edge always writes its `ef` words, zeros if both sides are dry. The gather's
  wall test becomes `nb == -1`, and it reads open edges without repeating the dry test.
- Only `gpu_resident_batch` with `deterministic` on supports this. Any other backend
  raises at config time, as `capture_momentum_snapshots` does.

With no open boundaries every new path is unreachable or unrecorded, so results must
stay bit-identical.

## Volume accounting

A new kernel tallies each stretch after each flux stage. It runs one invocation per
stretch, so the sum is sequential and deterministic, and it adds `dt/2 * ef[edge]` to a
pair of float32 words with the compensated depth's two-sum. That gives about 48 bits,
3.5e-8 m³ at 1e7 m³. It is recorded only when a run has open boundaries.

The host reads the tally at each output window. `WorkflowResult` gains per-stretch
cumulative volumes aligned with `snap_times`, positive out of the mesh.

The balance is then a real check and not a definition:

```text
V_start + V_sources + V_in - V_out = V_final
```

Both sides are measured independently. It should close to the same 1e-9 order as
today's drift. `Inflow` also has a second check, the tally against `Q * T`.

A cell emptied through an open edge can hit the positivity clamp, which would break
the balance by the clamp's amount. `clamp_mass_m3` reports that, and the gates assert
it stays small.

## Active set

- **Outflow, `NormalDepth`.** Unchanged. A dry cell has no ghost water, so the two-ring
  argument holds.
- **`Inflow`, and `Stage` above the bed.** Water can appear in a dry block. A third
  block word, `BLK_FORCED`, is set by the host per phase for blocks with such an edge,
  and `BLOCK_DILATE` treats it as wet, so the block and its two-ring stay active.
- Edges sort by left cell's block, and a boundary edge has only a left cell, so its
  flux and its update always live in the same block.

## Passing water between basins

Run upstream first. Its outlet tally gives volume per output window, and a helper turns
that into `Inflow` phases for the next basin, with `Q = ΔV / Δt`:

```python
upstream = run(basin_a)
phases = inflow_phases(upstream, "outlet", into="from_upstream", energy_slope=1e-3)
```

- **Volume passes exactly.** The downstream basin receives the measured volume of each
  window, by construction.
- **The meshes need not match along the cut,** because only a total crosses and the
  receiving side redistributes it by conveyance.
- **Backwater does not cross upstream.** The cut needs a reach the downstream level does
  not control, or an overlap past that reach. This is the design's main limit.
- **The stretch must span the flood width at the cut.** A narrower one leaves the rest a
  wall, and water piles up beside it.

Two-way exchange, with basins in lockstep and stage passed back upstream, is the planned
next effort after this one, recorded in the [roadmap](../../planning/roadmap.md). Per-phase
conditions and per-window tallies are what it will use, so this design must not close that
off.

## Rejected

- **Sinks and sources in rim cells.** No momentum, and the wall still reflects.
- **Real ghost cells in the mesh.** Changes the mesh arrays and the cache for what one
  function does.
- **Inflow as an edge-weighted point source.** Exact and trivial, but with no momentum
  it mounds at the rim and spreads along the wall.

## Gates

A new `open_boundary` bench joins the report. Analytical cases:

1. **Uniform flow.** Sloped channel, `Inflow` upstream and `NormalDepth` downstream.
   Steady depth must equal the Manning normal depth `(q n / sqrt(S))^(3/5)`. Covers
   inflow, outflow and the balance together.
2. **Non-reflection, supercritical.** Ritter dam break with the channel cut downstream of
   the dam. `FreeOutflow` must match a longer channel's run cell by cell over the shared
   part, and the Ritter solution as before.
3. **Stoker wave leaving.** The subcritical case. Measures reflection against the analytic
   solution for `FreeOutflow`, `NormalDepth` and `Stage`, and reports it.
4. **Lake at rest.** Bumpy bed with `Stage` equal to the surface. No motion to rounding.
5. **Backwater.** Steady M1 and M2 profiles below a set `Stage`, against the integrated
   gradually varied flow equation.
6. **Two basins.** One sloped channel run whole, then cut in two and run in sequence. Volume
   must pass exactly, and the hydrograph at the far end must match the whole run.

Every case asserts the balance above to 1e-7, determinism, and `clamp_mass_m3`.

Regression: the six existing suites and 24 h Nysa stay bit-identical with no open
boundaries, at 685 289 and 1 127 159 steps.

Real run: 24 h Nysa with an outlet stretch, showing water leaving and the balance closed,
which is where a long flood run reflects today.

## Order of work

1. Encoding, tally, `FreeOutflow`, regression and cases 2 and 3.
2. `NormalDepth` and `Stage`, cases 4 and 5.
3. `Inflow` and `BLK_FORCED`, case 1.
4. `inflow_phases`, case 6, and the Nysa run.

## Needed from the mesher

Nothing is required. Two things would help per-catchment meshes:

- The rim as a line file, so a stretch can be picked without opening the mesh.
- A cut across the valley that is straight and roughly normal to the flow, wide enough for
  the flood. A slanted cut makes the ghost's normal direction wrong for the river.

## MUSCL

No conflict. The ghost is built from the left state, so a reconstructed face state fits the
same function, and the boundary cell can stay first order at an open edge.

## Open questions

- No prototype has run. The 48-bit tally, the four Newton steps and the reflection on
  subcritical flow are design figures, to be measured in the order above.
- `energy_slope` is a user input. A wrong value shifts the `NormalDepth` stage directly,
  and a chosen slope is a modelling decision the solver cannot check.
