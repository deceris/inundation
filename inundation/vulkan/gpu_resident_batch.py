"""Option #5 SWE solver: GPU-resident steps with an integer clock.

Each ``run()`` records one command sequence of ``steps_per_graph`` steps and
re-submits it until every output window is done. Every step picks its own dt on
device: the corrector flux accumulates the CFL bound, the corrector update
reduces it, and ``step_control`` turns it into whole 2^-20 s ticks capped by
``dt_max`` and by the window's stop time. Simulated time is a uint64 tick count,
so the clock advances by exactly the dt the physics used and every output lands
exactly on its time. Steps submitted after a window's stop have dt = 0 and
return immediately. See ``shaders/resident.py`` for the device side.

Active set: flux and update skip every 256-cell block that is more than two
block rings from a wet block (``docs/implementation/22-active-set/plan.md``).
Results are the same as with skipping off (``active_set = False``).

Deterministic accumulation (``deterministic``, on by default): fluxes and CFL
terms are written per edge and summed per cell in a fixed order instead of by
float ``atomicAdd``, so repeated runs are bit-identical on one device and
driver (``docs/implementation/23-deterministic-accumulation/plan.md``).
"""
# pyright: reportMissingImports=false,reportPrivateUsage=false,reportUnknownArgumentType=false,reportUnknownMemberType=false,reportUnknownVariableType=false,reportUnusedExpression=false

from __future__ import annotations

import dataclasses
import enum
import math
import sys
import time as _time
from typing import TYPE_CHECKING

import kp
import numpy as np

from ..mesh.blocks import (
    block_adjacency,
    block_two_ring,
    edge_order_by_block,
    edge_workgroup_block_ranges,
    num_blocks,
    remap_cell_edges,
)
from ..tuning import (
    ACTIVE_BLOCK_CELLS,
    compute_workgroups,
    dt_to_ticks,
    output_window_stops,
    seconds_to_ticks,
    ticks_to_seconds,
)
from .solver import SWESolver, _pc_cfl_accum, _pc_cfl_reduce, _pc_flux, _pc_update

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ..mesh.geometry import MeshGeometry

# Host mirror of the CTRL block in STEP_CONTROL_GLSL (std430, 48 bytes).
RESIDENT_CTRL_DTYPE = np.dtype(
    [
        ("now_ticks", "<u8"),
        ("stop_ticks", "<u8"),
        ("steps", "<u8"),
        ("dt_ticks", "<u4"),
        ("dt_max_ticks", "<u4"),
        ("dry_dt_ticks", "<u4"),
        ("last_raw", "<f4"),
        ("cfl_safety", "<f4"),
        ("_pad", "<u4"),
    ]
)

# Applied to every CFL bound, as the per-batch resolve did before.
CFL_SAFETY = 0.8
STEPS_PER_GRAPH_DEFAULT = 200
_DT_BINDING = 17


class StepControlMode(enum.Enum):
    """What ``step_control`` does; the shader branches on the float value."""

    START = 0.0  # plan the first step from the prologue CFL reduction
    STEP = 1.0  # advance the clock past the step just taken, plan the next
    RETARGET = 2.0  # a new window's stop was set: re-plan without advancing


def _pc_step_control(mode: StepControlMode) -> list[float]:
    return [float(mode.value)]


def _pc_source_resident(num_sources: int, dry_tol: float) -> list[float]:
    return [float(num_sources), dry_tol]


def _pc_block_mark(nc: int, n_blocks: int, dry_tol: float, all_active: bool) -> list[float]:
    return [float(nc), float(n_blocks), dry_tol, 1.0 if all_active else 0.0]


def _pc_block_dilate(n_blocks: int, force: bool) -> list[float]:
    return [float(n_blocks), 1.0 if force else 0.0]


def _pc_cfl_edge_gather(ne: int, g: float, dry_tol: float) -> list[float]:
    return [float(ne), g, dry_tol]


def _pc_cfl_gather(nc: int, cfl: float) -> list[float]:
    return [float(nc), cfl]


# Flag words per block in the active-set buffer: wet, active.
_BLOCK_FLAG_WORDS = 2
_DILATE_WORKGROUP = 64
# Per-edge words of the gather path's edge buffer (EF_WORDS in the shaders).
_EDGE_FLUX_WORDS = 6


def _f32_bits(words: NDArray[np.uint32]) -> NDArray[np.float32]:
    """uint32 words as a float32 array, for Kompute's float tensors."""
    return np.ascontiguousarray(words, dtype=np.uint32).view(np.float32)


class SWESolverGpuResidentBatch(SWESolver):
    """Record once, submit many steps; dt and time stay on the GPU."""

    steps_per_graph: int = STEPS_PER_GRAPH_DEFAULT
    # Skip blocks far from water. Off only to check that skipping is exact.
    active_set: bool = True
    # Fixed-order sums instead of float atomics: bit-identical repeat runs.
    deterministic: bool = True

    def _build_tensors(
        self,
        geom: MeshGeometry,
        h0: NDArray[np.float32],
        hu0: NDArray[np.float32],
        hv0: NDArray[np.float32],
        n0: NDArray[np.float32],
    ) -> None:
        if self._work_group_size != ACTIVE_BLOCK_CELLS:
            raise ValueError(
                f"gpu_resident_batch needs workgroup_size={ACTIVE_BLOCK_CELLS}, "
                f"got {self._work_group_size}"
            )
        # Edges sorted by the block of their left cell, so each edge workgroup
        # touches few blocks. Only the solver's copy is reordered.
        order = edge_order_by_block(geom.edge_cellL, ACTIVE_BLOCK_CELLS)
        sorted_geom = dataclasses.replace(
            geom,
            edge_len=geom.edge_len[order],
            edge_nx=geom.edge_nx[order],
            edge_ny=geom.edge_ny[order],
            edge_cellL=geom.edge_cellL[order],
            edge_cellR=geom.edge_cellR[order],
        )
        super()._build_tensors(sorted_geom, h0, hu0, hv0, n0)
        # dt slot 0: this step's dt; slot 1: next step's CFL bound (atomicMin).
        if self._all_tensors[_DT_BINDING] is not self.t_dtbuf:
            raise RuntimeError("base tensor layout changed: dt is no longer binding 17")
        dt_words = np.array([0, self._SENTINEL_UINT, 0, 0], dtype=np.uint32)
        self.t_dtbuf = self._mgr.tensor(dt_words.view(np.float32))
        self._all_tensors[_DT_BINDING] = self.t_dtbuf
        self._update_tensors = [*self._all_tensors, self.t_clamp_mass]

        self._ctrl = np.zeros(1, dtype=RESIDENT_CTRL_DTYPE)
        self.t_ctrl = self._mgr.tensor(self._ctrl.view(np.float32).copy())
        self._ctrl_tensors = [*self._all_tensors, self.t_ctrl]

        self.n_blocks = num_blocks(self.N, ACTIVE_BLOCK_CELLS)
        ranges = edge_workgroup_block_ranges(
            sorted_geom.edge_cellL, self._work_group_size, ACTIVE_BLOCK_CELLS
        )
        bptr, bidx = block_two_ring(
            *block_adjacency(
                sorted_geom.edge_cellL, sorted_geom.edge_cellR, self.N, ACTIVE_BLOCK_CELLS
            )
        )
        self.t_blocks = self._mgr.tensor(
            _f32_bits(np.zeros(_BLOCK_FLAG_WORDS * self.n_blocks, dtype=np.uint32))
        )
        self.t_edge_ranges = self._mgr.tensor(_f32_bits(ranges.ravel()))
        self.t_block_ptr = self._mgr.tensor(_f32_bits(bptr))
        # Kompute rejects empty tensors: pad a mesh with one block.
        self.t_block_idx = self._mgr.tensor(
            _f32_bits(bidx if bidx.size else np.zeros(1, dtype=np.uint32))
        )
        self._active_extra = [self.t_blocks, self.t_edge_ranges, self.t_block_ptr, self.t_block_idx]

        # Gather path: per-edge results, per-cell CSR in the sorted edge order,
        # and the per-cell marks of live edges.
        self.t_edge_flux = self._mgr.tensor(np.zeros(_EDGE_FLUX_WORDS * self.E, dtype=np.float32))
        self.t_cell_edge_ptr = self._mgr.tensor(_f32_bits(geom.cell_edge_ptr.astype(np.uint32)))
        self.t_cell_edge_idx = self._mgr.tensor(
            _f32_bits(remap_cell_edges(geom.cell_edge_idx_signed, order))
        )
        self.t_touched = self._mgr.tensor(_f32_bits(np.zeros(self.N, dtype=np.uint32)))
        self._gather_extra = [
            self.t_edge_flux,
            self.t_cell_edge_ptr,
            self.t_cell_edge_idx,
            self.t_touched,
        ]
        # Bindings 0-18, clamp mass at 19, active set at 20-23, gather at 24-27.
        self._active_tensors = [*self._update_tensors, *self._active_extra, *self._gather_extra]
        self._mgr.sequence().record(
            kp.OpTensorSyncDevice(
                [self.t_dtbuf, self.t_ctrl, *self._active_extra, *self._gather_extra]
            )
        ).eval()

    def _build_algorithms(self, spv: dict[str, bytes]) -> None:
        N, E = self.N, self.E
        wg_e, wg_c = compute_workgroups(N, E, self._work_group_size)
        g, dry_tol = self._g, self._dry_tol

        # Keyed by ``deterministic``: gather kernels (True) or atomics (False).
        self._algo_flux_stage = {
            gather: [
                self._mgr.algorithm(
                    self._active_tensors,
                    spv["flux_resident_gather" if gather else "flux_resident"],
                    workgroup=wg_e,
                    spec_consts=[],
                    push_consts=_pc_flux(E, 0.0, g, dry_tol, self._cfl, stage),
                )
                for stage in (0, 1)
            ]
            for gather in (False, True)
        }
        self._update_spv = {
            False: spv["update_resident"],
            True: spv["update_resident_gather"],
        }
        self._wg_c = wg_c
        self._build_update_algos()
        self._build_cfl_algos(spv, N, E, wg_e, wg_c, g, dry_tol)
        self._algo_cfl_edge_gather = self._mgr.algorithm(
            self._active_tensors,
            spv["cfl_edge_gather"],
            workgroup=wg_e,
            spec_consts=[],
            push_consts=_pc_cfl_edge_gather(E, g, dry_tol),
        )
        self._algo_cfl_gather = self._mgr.algorithm(
            self._active_tensors,
            spv["cfl_gather"],
            workgroup=wg_c,
            spec_consts=[],
            push_consts=_pc_cfl_gather(N, self._cfl),
        )
        self._algo_dt_reset = self._mgr.algorithm(
            self._all_tensors,
            spv["dt_reset"],
            workgroup=(1, 1, 1),
            spec_consts=[],
            push_consts=[self._SENTINEL_F, 0.0, 0.0, 0.0],
        )
        self._algo_step_control = {
            mode: self._mgr.algorithm(
                self._ctrl_tensors,
                spv["step_control"],
                workgroup=(1, 1, 1),
                spec_consts=[],
                push_consts=_pc_step_control(mode),
            )
            for mode in StepControlMode
        }

        n_blk = self.n_blocks
        wg_b = (math.ceil(n_blk / _DILATE_WORKGROUP), 1, 1)
        self._algo_block_mark = self._mgr.algorithm(
            self._active_tensors,
            spv["block_mark"],
            workgroup=wg_b,
            spec_consts=[],
            push_consts=_pc_block_mark(N, n_blk, dry_tol, all_active=False),
        )
        self._algo_block_dilate = self._mgr.algorithm(
            self._active_tensors,
            spv["block_dilate"],
            workgroup=wg_b,
            spec_consts=[],
            push_consts=_pc_block_dilate(n_blk, force=False),
        )
        self._barrier_blocks = kp.OpComputeBarrier([self.t_blocks])

        self._barrier_dt = kp.OpComputeBarrier([self.t_dtbuf, self.t_ctrl])
        self._barrier_cfl = kp.OpComputeBarrier(
            [self.t_cfl_scratch, self.t_edge_flux, self.t_dtbuf]
        )
        self._barrier_flux = kp.OpComputeBarrier(
            [
                self.t_dh,
                self.t_dhu,
                self.t_dhv,
                self.t_cfl_scratch,
                self.t_edge_flux,
                self.t_touched,
            ]
        )
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
                self.t_cfl_scratch,
                self.t_edge_flux,
                self.t_touched,
                self.t_dtbuf,
                self.t_blocks,
            ]
        )

        self._source_list_spv = spv["source_resident"]
        self._algo_source_list: kp.Algorithm | None = None
        self._source_count = 0

    def _build_update_algos(self) -> None:
        # Rebuilt when track_clamp changes: it is baked into the push constants.
        self._update_track_clamp = self.track_clamp
        self._algo_update_stage = {
            gather: [
                self._mgr.algorithm(
                    self._active_tensors,
                    spv,
                    workgroup=self._wg_c,
                    spec_consts=[],
                    push_consts=_pc_update(
                        self.N, 0.0, stage, self._g, self._dry_tol, self._cfl, self.track_clamp
                    ),
                )
                for stage in (0, 1)
            ]
            for gather, spv in self._update_spv.items()
        }

    def download_block_flags(self) -> NDArray[np.uint32]:
        """``(n_blocks, 2)`` active-set flags: wet, active."""
        self._mgr.sequence().record(kp.OpTensorSyncLocal([self.t_blocks])).eval()
        words = np.asarray(self.t_blocks.data()).view(np.uint32)
        return words.reshape(self.n_blocks, _BLOCK_FLAG_WORDS).copy()

    def _sync_ctrl_to_device(self) -> None:
        np.copyto(np.asarray(self.t_ctrl.data()), self._ctrl.view(np.float32))

    def _read_ctrl(self) -> None:
        self._ctrl[...] = np.asarray(self.t_ctrl.data()).view(RESIDENT_CTRL_DTYPE)

    def _record_step(self, seq: kp.Sequence, use_source: bool) -> None:  # pyright: ignore[reportAttributeAccessIssue]
        flux = self._algo_flux_stage[self.deterministic]
        update = self._algo_update_stage[self.deterministic]
        for stage in (0, 1):
            seq.record(kp.OpAlgoDispatch(flux[stage]))
            seq.record(self._barrier_flux)
            seq.record(kp.OpAlgoDispatch(update[stage]))
            seq.record(self._barrier_state)
        if use_source:
            seq.record(
                kp.OpAlgoDispatch(
                    self._require_source_list(),
                    _pc_source_resident(self._source_count, self._dry_tol),
                )
            )
            seq.record(self._barrier_state)
        if self.active_set:
            self._record_dilation(seq, force=False)
        seq.record(kp.OpAlgoDispatch(self._algo_step_control[StepControlMode.STEP]))
        seq.record(self._barrier_dt)

    def _record_dilation(self, seq: kp.Sequence, force: bool) -> None:  # pyright: ignore[reportAttributeAccessIssue]
        seq.record(
            kp.OpAlgoDispatch(self._algo_block_dilate, _pc_block_dilate(self.n_blocks, force))
        )
        seq.record(self._barrier_blocks)

    def _record_block_setup(self, seq: kp.Sequence) -> None:  # pyright: ignore[reportAttributeAccessIssue]
        """Active-set flags for the current state (all blocks when skipping is off)."""
        mark = _pc_block_mark(self.N, self.n_blocks, self._dry_tol, all_active=not self.active_set)
        seq.record(kp.OpAlgoDispatch(self._algo_block_mark, mark))
        seq.record(self._barrier_blocks)
        if self.active_set:
            self._record_dilation(seq, force=True)

    def _set_sources(self, src_dh_per_sec: NDArray[np.float32]) -> None:
        """Upload the nonzero sources as (cell, dh/dt) pairs; rebuild if the count changed."""
        cells = np.flatnonzero(src_dh_per_sec).astype(np.uint32)
        packed = np.empty(2 * max(cells.size, 1), dtype=np.uint32)
        packed[:] = 0
        packed[0 : 2 * cells.size : 2] = cells
        packed[1 : 2 * cells.size : 2] = src_dh_per_sec[cells].astype(np.float32).view(np.uint32)
        if self._algo_source_list is None or cells.size != self._source_count:
            self.t_source_list = self._mgr.tensor(_f32_bits(packed))
            tensors = [*self._all_tensors, self.t_source_list, *self._active_extra]
            self._algo_source_list = self._mgr.algorithm(
                tensors,
                self._source_list_spv,
                workgroup=(max(1, math.ceil(cells.size / self._work_group_size)), 1, 1),
                spec_consts=[],
                push_consts=_pc_source_resident(cells.size, self._dry_tol),
            )
        else:
            np.copyto(np.asarray(self.t_source_list.data()), _f32_bits(packed))
        self._mgr.sequence().record(kp.OpTensorSyncDevice([self.t_source_list])).eval()
        self._source_count = int(cells.size)

    def _require_source_list(self) -> kp.Algorithm:  # pyright: ignore[reportAttributeAccessIssue]
        if self._algo_source_list is None:
            raise RuntimeError("source list not built (call _set_sources first)")
        return self._algo_source_list

    def _record_prologue(self) -> kp.Sequence:  # pyright: ignore[reportAttributeAccessIssue]
        """CFL bound of the current state, then plan the first step."""
        N, E, g, dry_tol = self.N, self.E, self._g, self._dry_tol
        seq = self._mgr.sequence()
        seq.record(kp.OpTensorSyncDevice([self.t_ctrl]))
        self._record_block_setup(seq)
        seq.record(kp.OpAlgoDispatch(self._algo_dt_reset, [self._SENTINEL_F, 0.0, 0.0, 0.0]))
        seq.record(self._barrier_dt)
        if self.deterministic:
            seq.record(kp.OpAlgoDispatch(self._algo_cfl_edge_gather))
            seq.record(self._barrier_cfl)
            seq.record(kp.OpAlgoDispatch(self._algo_cfl_gather))
        else:
            seq.record(
                kp.OpAlgoDispatch(self._algo_cfl_accum, _pc_cfl_accum(E, g, dry_tol, self._cfl))
            )
            seq.record(self._barrier_cfl)
            seq.record(
                kp.OpAlgoDispatch(self._algo_cfl_reduce, _pc_cfl_reduce(N, g, dry_tol, self._cfl))
            )
        seq.record(self._barrier_cfl)
        seq.record(kp.OpAlgoDispatch(self._algo_step_control[StepControlMode.START]))
        seq.record(self._barrier_dt)
        seq.record(kp.OpTensorSyncLocal([self.t_ctrl]))
        return seq

    def _record_retarget(self) -> kp.Sequence:  # pyright: ignore[reportAttributeAccessIssue]
        seq = self._mgr.sequence()
        seq.record(kp.OpTensorSyncDevice([self.t_ctrl]))
        seq.record(kp.OpAlgoDispatch(self._algo_step_control[StepControlMode.RETARGET]))
        seq.record(self._barrier_dt)
        seq.record(kp.OpTensorSyncLocal([self.t_ctrl]))
        return seq

    def _record_steps(self, n_steps: int, use_source: bool) -> kp.Sequence:  # pyright: ignore[reportAttributeAccessIssue]
        seq = self._mgr.sequence()
        seq.record(kp.OpTensorSyncDevice([self.t_ctrl]))
        for _ in range(n_steps):
            self._record_step(seq, use_source)
        seq.record(kp.OpTensorSyncLocal([self.t_ctrl]))
        return seq

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
        """Advance to ``t_end``; snapshot every ``output_interval_s`` from the run's start.

        ``cfl_interval`` is accepted for interface compatibility and ignored:
        the CFL bound is recomputed every step.
        """
        if source_fn is not None:
            raise ValueError(
                "gpu_resident_batch takes a constant source_rate per run, not source_fn"
            )
        if self.steps_per_graph < 1:
            raise ValueError(f"steps_per_graph must be positive, got {self.steps_per_graph}")

        dt_max_ticks = dt_to_ticks(dt_max)
        dry_dt_ticks = dt_to_ticks(min(dt_init, dt_max))
        start_ticks = seconds_to_ticks(t_start if resume else 0.0)
        end_ticks = seconds_to_ticks(t_end)
        interval_ticks = seconds_to_ticks(output_interval_s)

        if not resume:
            self._reset_to_initial()
        if self.track_clamp != self._update_track_clamp:
            self._build_update_algos()

        use_source = False
        if source_rate is not None:
            src_dh_per_sec = np.asarray(source_rate, dtype=np.float32) * (1.0 / self._area).astype(
                np.float32
            )
            self._set_sources(src_dh_per_sec)
            use_source = self._source_count > 0

        snapshots: list[NDArray[np.float32]] = [
            self.download_h().copy() if resume else self._h0.copy()
        ]
        snap_times: list[float] = [ticks_to_seconds(start_ticks)]

        ctrl = self._ctrl
        ctrl["now_ticks"] = start_ticks
        ctrl["steps"] = 0
        ctrl["dt_ticks"] = 0
        ctrl["dt_max_ticks"] = dt_max_ticks
        ctrl["dry_dt_ticks"] = dry_dt_ticks
        ctrl["cfl_safety"] = CFL_SAFETY

        steps_seq = self._record_steps(self.steps_per_graph, use_source)
        retarget_seq = self._record_retarget()
        planned = False

        wall_start = _time.monotonic()
        last_log_wall = wall_start
        log_interval = 10.0
        finite = True

        for stop_ticks in output_window_stops(start_ticks, end_ticks, interval_ticks):
            ctrl["stop_ticks"] = stop_ticks
            self._sync_ctrl_to_device()
            if planned:
                retarget_seq.eval()
            else:
                self._record_prologue().eval()
                planned = True
            self._read_ctrl()

            while int(ctrl["now_ticks"][0]) < stop_ticks:
                before = int(ctrl["now_ticks"][0])
                steps_seq.eval()
                self._read_ctrl()
                now_ticks = int(ctrl["now_ticks"][0])
                if now_ticks == before:
                    raise RuntimeError(
                        f"GPU-resident run made no time progress at "
                        f"t={ticks_to_seconds(now_ticks):.6f}s (CFL bound below one tick)"
                    )
                now = _time.monotonic()
                if progress and now - last_log_wall >= log_interval:
                    self._log_progress(now - wall_start, now_ticks, start_ticks, end_ticks)
                    last_log_wall = now

            h_snap = self.download_h()
            if not np.isfinite(h_snap).all():
                snapshots.append(np.nan_to_num(h_snap, nan=0.0, posinf=0.0).copy())
                snap_times.append(ticks_to_seconds(stop_ticks))
                finite = False
                break
            # The final window may end short of a full interval: that is the end
            # state, appended below, not an output time.
            if (stop_ticks - start_ticks) % interval_ticks == 0:
                if progress:
                    sys.stdout.write(
                        f"[solver] snapshot {len(snapshots)} captured at "
                        f"t={ticks_to_seconds(stop_ticks):.2f}s "
                        f"(wall={_time.monotonic() - wall_start:.1f}s)\n"
                    )
                    sys.stdout.flush()
                snapshots.append(h_snap.copy())
                snap_times.append(ticks_to_seconds(stop_ticks))

        self.steps_total += int(ctrl["steps"][0])

        if finite:
            h_final = self.download_h()
            if not np.allclose(snapshots[-1], h_final):
                snapshots.append(h_final.copy())
                snap_times.append(ticks_to_seconds(int(ctrl["now_ticks"][0])))

        return snapshots, snap_times

    def _log_progress(
        self, wall_elapsed: float, now_ticks: int, start_ticks: int, end_ticks: int
    ) -> None:
        t_sim, t_end = ticks_to_seconds(now_ticks), ticks_to_seconds(end_ticks)
        done = now_ticks - start_ticks
        eta = wall_elapsed / done * (end_ticks - now_ticks) if done > 0 else float("inf")
        eta_str = f"{eta:.1f}s" if math.isfinite(eta) else "inf"
        pct = t_sim / t_end * 100.0 if t_end > 0 else 0.0
        sys.stdout.write(
            f"[solver][gpu-resident {int(self._ctrl['steps'][0])}] sim={t_sim:.2f}/{t_end:.2f}s "
            f"({pct:.1f}%) eta~{eta_str}\n"
        )
        sys.stdout.flush()


__all__ = ["RESIDENT_CTRL_DTYPE", "SWESolverGpuResidentBatch", "StepControlMode"]
