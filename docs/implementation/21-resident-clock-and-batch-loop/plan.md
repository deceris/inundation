# Resident Clock and Batch Loop — Plan

**Status: done.** Changes the `gpu_resident_batch` execution strategy
only; the shared flux/update kernels used by the other strategies stay as they
are.

## Goal

Make `gpu_resident_batch` keep exact simulated time and remove its per-batch
host overhead, so long production runs are both mass-consistent with their
discharge schedule and limited by kernel time rather than by submission.

## Problem

Measured on the Nysa pavé run (2.35 M cells, 24 h simulated, M1 Max,
MoltenVK): 1.71 ms per step, 390 s of system time in a 4 720 s run, and 0.95 %
of the scheduled inflow volume missing at the end. On a 47 k-cell reach the
per-step cost was 0.82 ms, about 0.8 ms of it fixed overhead.

The code shows three causes:

1. **Float32 device clock.** `time_buf` is a float32 advanced by
   `time_buf += dt`. Above 65 536 s its spacing is 7.8 ms against a ~31 ms
   timestep, so each increment is rounded by up to ~12 %, systematically while
   dt is slowly varying. Sources inject with the true `dt_buf`, but phases stop
   on the rounded clock, so the injected volume differs from the scheduled
   volume. This also violates the architecture rule that long-horizon time is
   float64 or integer.
2. **Per-batch rebuild.** Every 10 steps the host creates a new Kompute
   sequence, re-records ~100 dispatches and barriers, submits, and blocks on
   the fence: ~275 k cycles for the Nysa run.
3. **Frozen batch timestep.** The CFL reduction runs once per batch and its dt
   is reused for every step in it, so batches cannot be lengthened safely.

Minor: `step += _BATCH_STEPS` miscounts when `cfl_interval > 10`.

## Evidence before the change

A 40 × 40 flat closed basin with one point source runs the same physics at
different absolute start times (`resume=True, t_start=t0`). See
[`results.md`](results.md#baseline-probe). Start-time dependence isolates the
clock; it also exposed a separate float32 state floor (deep water, tiny
increments) that this effort does not fix.

## Design

### Integer tick clock

- Simulated time on device is a `uint64_t` tick count
  (`GL_EXT_shader_explicit_arithmetic_types_int64`), tick = 2⁻²⁰ s
  (~0.95 µs). MoltenVK on M1 Max reports `shaderInt64 = true` and
  `shaderFloat64 = false`; a probe kernel carried past 2³² correctly through
  Kompute.
- The step timestep is quantized down to whole ticks:
  `dt = float(ticks) * 2^-20`. With `dt_max ≤ 16 s` the tick count fits in 24
  bits, so this float is exact, and the clock advances by exactly the dt the
  physics used.
- Stop time is an integer tick count too; the last step of a window is capped
  to the exact remaining ticks. Host time is `ticks · 2⁻²⁰` in float64.
- Quantization shortens dt by < 1 µs (< 0.004 % at 31 ms).

Alternatives rejected: a float64 clock (no `shaderFloat64` on Apple GPUs, and
even where available the clock would not equal the sum of the dt values the
physics used) and a compensated float32 pair (relies on the shader compiler
not reassociating floating-point arithmetic).

### Per-step CFL inside the step

- The corrector flux kernel (stage 1) adds the same per-edge
  `max(|uₙ|+c) · len` contribution that `cfl_accum` computes today, from the
  state it already loads, into `cfl_scratch`. Fully dry edges contribute zero
  in both, so the early exit is unchanged.
- The corrector update kernel reduces `cfl · A / λ` into a `dt_next` slot by
  `atomicMin` and zeroes its scratch entry.
- A single-thread `step_control` kernel advances the clock by the step's ticks,
  then sets the next step's ticks from `dt_next · 0.8`, `dt_max`, and the
  remaining ticks to the window's stop, and resets `dt_next`.
- Safety factor 0.8 and CFL 0.45 are unchanged; only the lag changes (half a
  step instead of up to ten steps). Relaxing them is a separate effort.

### Record once, submit many steps

- One sequence is recorded per `run()`: sync a small parameter tensor to the
  device, then `steps_per_submit` steps, then sync the clock back.
- Per-window values (stop ticks, `dt_max`, dry dt) live in that parameter
  tensor, so each submit re-runs the same recording.
- Steps after the window stop see `dt = 0`; every stage kernel returns
  immediately when `dt == 0`, so overshoot costs one dispatch each, not a
  sweep.
- The device counts steps with `dt > 0`; `steps_total` reads it instead of
  assuming a batch size.
- Steps per submit come from the existing, previously unused
  `WorkflowConfig.steps_per_graph` (default 200 when unset).

### Kernels

New resident-only shader variants (`flux_resident`, `update_resident`,
`source_resident`, `step_control`) built from the existing GLSL text, so
HLLC, the Audusse correction, friction and the clamp diagnostic stay a single
source. The `dtbuf` kernels shared by other strategies are not modified.

## Tasks

1. Clock tensor and `step_control` kernel; host-side tick conversion helpers
   in `tuning.py`.
2. Resident flux/update/source variants with in-step CFL and `dt == 0` early
   exit.
3. Record-once loop in `gpu_resident_batch.py`; wire `steps_per_graph`;
   device step counter.
4. Tests: tick conversion and capping on CPU; a GPU-marked test that the
   start-time sweep from the probe is within float32 state noise.
5. Gates: lake-at-rest, dam-break, radial dam-break, floodplain depressions,
   momentum obstruction and flood propagation pass against
   `fixed_dt_batch_barrier`; new baseline only if the per-step dt changes a
   measured value.
6. Measure: probe sweep, 47 k reach and Nysa 10-minute probe (ms per step,
   system time), then the full 24 h run (mass balance).

## Out of scope

- Float32 state floor (compensated or float64 state) — roadmap priority 1.
- Active set, CFL relaxation, local time stepping.
- Other execution strategies.
