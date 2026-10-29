"""Option #4 SWE solver with explicit compute barriers between batched dispatches.

Fixes a missing-synchronization hazard in ``SWESolverFixedDtBatch``: that class
chains many ``flux``/``update``/``cfl_accum``/``cfl_reduce`` dispatches into a
single Kompute sequence with one ``eval()`` per batch, but never inserts a
``kp.OpComputeBarrier`` between dispatches that write a storage buffer
(``dh``/``dhu``/``dhv`` via ``atomicAdd``) and the very next dispatch that reads
it. Without a barrier, Kompute/Vulkan does not guarantee the write is visible
to the next shader invocation, producing a race that reproduces even on a
flat-bed, still-water case (no scheme/bed-slope cause needed) and is
non-deterministic run to run. See ``LAKE_AT_REST_BASELINE_RESULTS.md`` §6.

``SWESolverGpuResidentBatch`` (Option #5) already demonstrates the correct
barrier placement for this dispatch chain; this class mirrors that pattern
while keeping Option #4's host-driven CFL readback and CPU-side early-exit
guard, so it is a minimal, behavior-preserving fix rather than a rewrite.
"""
# pyright: reportMissingImports=false,reportPrivateUsage=false,reportUnknownArgumentType=false,reportUnknownMemberType=false,reportUnknownVariableType=false,reportUnusedExpression=false

from __future__ import annotations

import math
import sys
import time as _time
from typing import TYPE_CHECKING

import kp
import numpy as np

from .fixed_dt_batch import SWESolverFixedDtBatch
from .solver import _pc_cfl_accum, _pc_cfl_reduce, _pc_flux, _pc_source, _pc_update

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

from ..tuning import (
    CFL_EPSILON_MIN,
    CFL_SANITY_MAX,
    SIMULATION_TIME_EPSILON,
    compute_eta_seconds,
)


class SWESolverFixedDtBatchBarrier(SWESolverFixedDtBatch):
    """Option #4 batching with explicit compute barriers between dispatches."""

    def _build_algorithms(self, spv: dict[str, bytes]) -> None:
        """Build Option #4 algorithms, then precompute the barrier ops they need."""
        super()._build_algorithms(spv)
        self._barrier_cfl = kp.OpComputeBarrier([self.t_cfl_scratch, self.t_dtbuf])
        self._barrier_flux = kp.OpComputeBarrier([self.t_dh, self.t_dhu, self.t_dhv])
        self._barrier_state = kp.OpComputeBarrier(
            [
                self.t_h,
                self.t_hu,
                self.t_hv,
                self.t_h1,
                self.t_hu1,
                self.t_hv1,
                self.t_dh,
                self.t_dhu,
                self.t_dhv,
            ]
        )

    def run(
        self,
        t_end: float,
        output_interval_s: float,
        dt_max: float,
        dt_init: float,
        cfl_interval: int,
        progress: bool = True,
        source_rate: NDArray[np.float32] | None = None,
        source_fn: Callable[[float], NDArray[np.float32]] | None = None,
        resume: bool = False,
        t_start: float = 0.0,
        capture_momentum_snapshots: bool = False,
        output_every_steps: int | None = None,
    ) -> (
        tuple[list[NDArray[np.float32]], list[float]]
        | tuple[
            list[NDArray[np.float32]],
            list[float],
            list[NDArray[np.float32]],
            list[NDArray[np.float32]],
        ]
    ):
        """Run Heun RK2 simulation with barriered fixed-dt steps per GPU submit.

        ``output_every_steps``, when set, switches snapshot cadence from
        time-based (``output_interval_s``, the default) to step-count-based
        — batches are no longer clamped to land on an output-time boundary,
        since there is no time boundary to land on; a snapshot is taken
        whenever the cumulative step counter (across this whole solver's
        lifetime, i.e. across phases, via ``self.steps_total``) crosses a
        multiple of ``output_every_steps``. This supports deterministic
        step-count cadence for Vulkan benchmarks.
        """
        N, E = self.N, self.E
        g, dt_ = self._g, self._dry_tol
        cfl = self._cfl

        if not resume:
            self._reset_to_initial()

        inv_area = (1.0 / self._area).astype(np.float32)
        use_gpu_source = False
        src_dh_per_sec: NDArray[np.float32] | None = None
        if source_rate is not None and source_fn is None:
            src_dh_per_sec = np.asarray(source_rate, dtype=np.float32) * inv_area
            if self._algo_source is None and hasattr(self, "_source_spv"):
                self._build_source_algo(src_dh_per_sec)
            elif self._algo_source is not None and hasattr(self, "t_source"):
                self._upload(self.t_source, src_dh_per_sec)
            if self._algo_source is not None:
                use_gpu_source = True

        t_offset = t_start if resume else 0.0
        if resume:
            snapshots: list[NDArray[np.float32]] = [self.download_h().copy()]
            snap_times: list[float] = [t_start]
        else:
            snapshots = [self._h0.copy()]
            snap_times = [0.0]

        t_sim = t_offset
        step = 0
        dt = dt_init
        next_output_time = t_offset + output_interval_s
        wall_start = _time.monotonic()
        last_log_wall = wall_start
        log_interval = 10.0
        cfl_safety = 0.8
        output_eps = 1e-7

        while t_sim < t_end:
            dt = min(dt, t_end - t_sim, dt_max)

            np.asarray(self.t_dtbuf.data())[0] = self._SENTINEL_F
            cfl_seq = self._mgr.sequence()
            cfl_seq.record(kp.OpTensorSyncDevice([self.t_dtbuf]))
            cfl_seq.record(kp.OpAlgoDispatch(self._algo_cfl_accum, _pc_cfl_accum(E, g, dt_, cfl)))
            cfl_seq.record(self._barrier_cfl)
            cfl_seq.record(kp.OpAlgoDispatch(self._algo_cfl_reduce, _pc_cfl_reduce(N, g, dt_, cfl)))
            cfl_seq.record(kp.OpTensorSyncLocal([self.t_dtbuf]))
            cfl_seq.eval()
            dt_cfl = float(np.array(self.t_dtbuf.data(), dtype=np.float32)[0])
            if dt_cfl < CFL_EPSILON_MIN:
                raise RuntimeError(f"CFL dt too small at step {step}: {dt_cfl:.3e}")
            if dt_cfl < CFL_SANITY_MAX:
                dt = min(dt_cfl * cfl_safety, dt_max)

            batch_steps = max(1, int(cfl_interval))
            batch_steps = min(batch_steps, math.ceil(max(t_end - t_sim, 0.0) / dt))
            if output_every_steps is not None:
                steps_so_far = self.steps_total + step
                steps_to_next_bucket = output_every_steps - (steps_so_far % output_every_steps)
                batch_steps = min(batch_steps, steps_to_next_bucket)
            else:
                batch_steps = min(
                    batch_steps,
                    math.ceil(max(next_output_time - t_sim, 0.0) / dt),
                )

            step_dts: list[float] = []
            step_t = t_sim
            for _ in range(batch_steps):
                if output_every_steps is not None:
                    step_dt = min(dt, t_end - step_t)
                else:
                    step_dt = min(dt, t_end - step_t, next_output_time - step_t)
                    if step_dt <= SIMULATION_TIME_EPSILON:
                        break
                step_dts.append(step_dt)
                step_t += step_dt
            if not step_dts:
                break

            step_seq = self._mgr.sequence()
            for step_dt in step_dts:
                step_seq.record(
                    kp.OpAlgoDispatch(self._algo_flux, _pc_flux(E, step_dt, g, dt_, cfl, 0))
                )
                step_seq.record(self._barrier_flux)
                step_seq.record(
                    kp.OpAlgoDispatch(
                        self._algo_update, _pc_update(N, step_dt, 0, g, dt_, cfl, self.track_clamp)
                    )
                )
                step_seq.record(self._barrier_state)
                step_seq.record(
                    kp.OpAlgoDispatch(self._algo_flux, _pc_flux(E, step_dt, g, dt_, cfl, 1))
                )
                step_seq.record(self._barrier_flux)
                step_seq.record(
                    kp.OpAlgoDispatch(
                        self._algo_update, _pc_update(N, step_dt, 1, g, dt_, cfl, self.track_clamp)
                    )
                )
                step_seq.record(self._barrier_state)
                if use_gpu_source:
                    step_seq.record(
                        kp.OpAlgoDispatch(self._require_algo_source(), _pc_source(N, step_dt))
                    )
                    step_seq.record(self._barrier_state)
            step_seq.eval()

            if not use_gpu_source and (source_fn is not None or src_dh_per_sec is not None):
                raise RuntimeError(
                    "fixed_dt_batch_barrier requires GPU source kernel for batched source terms"
                )

            steps_before_batch = self.steps_total + step
            t_sim += sum(step_dts)
            if output_every_steps is None and abs(t_sim - next_output_time) <= output_eps:
                t_sim = next_output_time
            if abs(t_sim - t_end) <= output_eps:
                t_sim = t_end
            step += len(step_dts)

            now = _time.monotonic()
            if progress and now - last_log_wall >= log_interval:
                wall_elapsed = now - wall_start
                pct = t_sim / t_end * 100.0 if t_end > 0 else 0.0
                eta = compute_eta_seconds(wall_elapsed, t_sim, t_offset, t_end)
                eta_str = f"{eta:.1f}s" if math.isfinite(eta) else "inf"
                sys.stdout.write(
                    f"[solver][batch {step}] sim={t_sim:.2f}/{t_end:.2f}s "
                    f"({pct:.1f}%) dt={dt:.4f} eta~{eta_str}\n"
                )
                sys.stdout.flush()
                last_log_wall = now

            if output_every_steps is not None:
                steps_after_batch = self.steps_total + step
                before_bucket = steps_before_batch // output_every_steps
                after_bucket = steps_after_batch // output_every_steps
                should_snap = before_bucket != after_bucket or t_sim >= t_end
            else:
                should_snap = t_sim + output_eps >= next_output_time
            if should_snap:
                h_snap = self.download_h()
                if not np.isfinite(h_snap).all():
                    snapshots.append(np.nan_to_num(h_snap, nan=0.0, posinf=0.0).copy())
                    snap_times.append(t_sim)
                    break
                if progress:
                    wall_elapsed = _time.monotonic() - wall_start
                    snap_idx = len(snapshots)
                    sys.stdout.write(
                        f"[solver] snapshot {snap_idx} captured at t={t_sim:.2f}s "
                        f"(wall={wall_elapsed:.1f}s)\n"
                    )
                    sys.stdout.flush()
                snapshots.append(h_snap.copy())
                snap_times.append(t_sim)
                if output_every_steps is None:
                    next_output_time += output_interval_s

        h_final = self.download_h()
        if not snapshots or not np.allclose(snapshots[-1], h_final):
            snapshots.append(h_final.copy())
            snap_times.append(t_sim)

        self.steps_total += step
        return snapshots, snap_times


__all__ = ["SWESolverFixedDtBatchBarrier"]
