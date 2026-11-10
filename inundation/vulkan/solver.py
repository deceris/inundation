"""Vulkan/kompute GPU solver for the mesh-based Shallow Water Equations.

Manages GPU buffer allocation, shader pipeline setup, and the Heun RK2
time-integration loop.

Typical usage
-------------
    from swe_gpu import SWESolver

    solver = SWESolver(geom, spv, h0, hu0, hv0, n0)
    snapshots, snap_times = solver.run(t_end=120.0, output_interval_s=30.0)
    h_final = solver.download_h()
"""

from __future__ import annotations

import os
import struct
import sys
import time as _time
from typing import TYPE_CHECKING

import numpy as np

from ..tuning import (
    CFL_EPSILON_MIN,
    CFL_HOST_DRIVEN,
    CFL_SANITY_MAX,
    DRY_TOL_DEFAULT,
    GRAVITY_G,
    SIMULATION_TIME_EPSILON,
    compute_workgroups,
    uint_as_float_bits,
)

# MoltenVK compiles shaders with Metal fast math unless told otherwise, and that
# reassociates the error-free transforms of the compensated depth away, with or
# without `precise`. It reads this once, when the first Vulkan instance is made,
# so it is set at import. A value already in the environment wins, and a Manager
# made before this import keeps the default: gpu_resident_batch checks the
# arithmetic on the device before a run. No effect on other drivers; costs about
# 7 % per step on MoltenVK.
os.environ.setdefault("MVK_CONFIG_FAST_MATH_ENABLED", "0")

try:
    import kp
except ImportError as exc:
    raise ImportError("kompute (kp) is required.  Install with:  pip install kp") from exc

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from ..mesh.geometry import MeshGeometry

# Words per open edge: condition, value (OPEN_WORDS in the shaders).
_OPEN_WORDS = 2
# Words per open edge of the flux kernels' volume: out (hi, lo), in (hi, lo).
_OPEN_VOLUME_WORDS = 4

# ─────────────────────────────────────────────────────────────────────────────
# Push-constant helpers
# ─────────────────────────────────────────────────────────────────────────────


# Counts travel as the float whose bits they are (the shaders declare them
# uint): a float *value* is exact only up to 2^24 elements.


def _pc_flux(ne: int, dt: float, g: float, dry_tol: float, cfl: float, stage: int) -> list[float]:
    return [uint_as_float_bits(ne), float(dt), float(g), float(dry_tol), float(cfl), float(stage)]


def _pc_update(
    nc: int, dt: float, stage: int, g: float, dry_tol: float, cfl: float, track_clamp: bool = False
) -> list[float]:
    # Emission order must match the GLSL push-constant struct
    # {num_cells, dt, g, dry_tol, cfl_number, stage, track_clamp}, which is not
    # the argument order — `stage` is declared second-to-last in the shader.
    return [
        uint_as_float_bits(nc),
        float(dt),
        float(g),
        float(dry_tol),
        float(cfl),
        float(stage),
        1.0 if track_clamp else 0.0,
    ]


def _pc_cfl_accum(ne: int, g: float, dry_tol: float, cfl: float) -> list[float]:
    return [uint_as_float_bits(ne), 0.0, float(g), float(dry_tol), float(cfl)]


def _pc_cfl_reduce(nc: int, g: float, dry_tol: float, cfl: float) -> list[float]:
    return [uint_as_float_bits(nc), 0.0, float(g), float(dry_tol), float(cfl)]


def _pc_source(nc: int, dt: float) -> list[float]:
    """Host-driven source kernels: {num_cells, dt} (dt unused by the dtbuf variant)."""
    return [uint_as_float_bits(nc), float(dt)]


def refuse_unrecorded(backend: str, capture_momentum_snapshots: bool, n_open: int) -> None:
    """Raise for a run() that records no momentum or boundary volume, instead of dropping it."""
    if capture_momentum_snapshots:
        raise ValueError(f"{backend} does not capture momentum snapshots")
    if n_open:
        raise ValueError(f"{backend} does not record open-boundary volumes")


# ─────────────────────────────────────────────────────────────────────────────
# SWESolver
# ─────────────────────────────────────────────────────────────────────────────


class SWESolver:
    """GPU-accelerated SWE solver (Heun RK2, HLLC, edge-centric FVM).

    Parameters
    ----------
    geom : MeshGeometry
        Hilbert-reordered geometry from ``swe_geometry.hilbert_reorder``.
    spv : dict[str, bytes]
        SPIR-V binaries keyed ``"flux"``, ``"update"``, ``"cfl_accum"``,
        ``"cfl_reduce"``.  Produced by ``swe_shaders.compile_all()``.
    h0, hu0, hv0 : (N,) float32
        Initial cell state arrays (depth and momentum components).
    n0 : (N,) float32
        Manning roughness coefficient per cell.
    g : float
        Gravitational acceleration [m/s²].  Default 9.81.
    dry_tol : float
        Depth below which a cell is considered dry.  Default 1e-4.
    cfl : float
        CFL number.  Default ``default_cfl`` of the backend.
    workgroup_size : int
        GPU workgroup size.  Default 256.
    open_edges : sequence of (k,) int64
        Each open stretch's edges, as indices into ``geom``. Every other
        boundary edge is a wall; ``set_open_conditions`` sets the open ones.

    """

    # CFL number when none is given; see CFL_HOST_DRIVEN.
    default_cfl: float = CFL_HOST_DRIVEN

    def __init__(
        self,
        geom: MeshGeometry,
        spv: dict[str, bytes],
        h0: NDArray[np.float32],
        hu0: NDArray[np.float32],
        hv0: NDArray[np.float32],
        n0: NDArray[np.float32],
        *,
        g: float = GRAVITY_G,
        dry_tol: float = DRY_TOL_DEFAULT,
        cfl: float | None = None,
        workgroup_size: int = 256,
        open_edges: Sequence[NDArray[np.int64]] = (),
    ) -> None:
        self.geom = geom
        self._open_stretches = [np.asarray(edges, dtype=np.int64) for edges in open_edges]
        # (stretches, 2) volume out of and into the mesh at every snapshot of a run.
        self.open_volume_snapshots: list[NDArray[np.float64]] = []
        self.N = geom.N
        self.E = geom.E
        self._g = float(g)
        self._dry_tol = float(dry_tol)
        self._cfl = float(self.default_cfl if cfl is None else cfl)
        self._work_group_size = workgroup_size
        # Opt-in: costs one extra buffer write per cell per RK stage.
        self.track_clamp = False

        self._h0 = h0.astype(np.float32)
        self._hu0 = hu0.astype(np.float32)
        self._hv0 = hv0.astype(np.float32)
        self._area = geom.area

        # CFL sentinel
        self._SENTINEL_UINT = struct.unpack("I", struct.pack("f", 1e30))[0]
        self._SENTINEL_F = struct.unpack("f", struct.pack("I", self._SENTINEL_UINT))[0]

        self._mgr = kp.Manager()
        self._build_tensors(geom, h0, hu0, hv0, n0)
        self._build_algorithms(spv)
        self._eft_probe_spv = spv["eft_probe"]
        self._eft_checked = False

        # Cumulative RK2-step counter across all `run()` calls on this
        # instance (i.e. across all phases of one workflow run) — surfaced
        # via `WorkflowResult.steps_total` for benchmark timing (steps/s,
        # ms/step), not used by the physics itself.
        self.steps_total = 0

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _as_f32_1d(arr: NDArray[np.float32]) -> NDArray[np.float32]:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != 1:
            a = np.ravel(a)
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)
        return a

    def _gpu_tensor(self, arr: NDArray[np.float32]) -> kp.Tensor:  # pyright: ignore[reportAttributeAccessIssue] — kp stubs incomplete
        return self._mgr.tensor(self._as_f32_1d(arr))

    def _set_tensor_host_data(self, tensor: kp.Tensor, arr: NDArray[np.float32]) -> None:  # pyright: ignore[reportAttributeAccessIssue]
        src = self._as_f32_1d(arr)
        dst = np.asarray(tensor.data())
        if dst.size != src.size:
            raise ValueError(f"Tensor size mismatch: dst={dst.size}, src={src.size}")
        np.copyto(dst, src)

    def _build_tensors(
        self,
        geom: MeshGeometry,
        h0: NDArray[np.float32],
        hu0: NDArray[np.float32],
        hv0: NDArray[np.float32],
        n0: NDArray[np.float32],
    ) -> None:
        g = self._gpu_tensor
        self.t_h = g(h0)
        self.t_hu = g(hu0)
        self.t_hv = g(hv0)
        # Binding 3: per-edge (dzL, dzR) pairs. The kernels never read the bed
        # itself; they reconstruct face depths from these exact small steps.
        self.t_edz = g(np.column_stack([geom.edge_dzL, geom.edge_dzR]).ravel())
        self.t_area = g(geom.area)
        self.t_elen = g(geom.edge_len)
        self.t_enx = g(geom.edge_nx)
        self.t_eny = g(geom.edge_ny)
        # Open edge k is ``ecR == -2 - k``, in stretch order; ``-1`` stays a wall.
        open_edges = np.concatenate([np.empty(0, dtype=np.int64), *self._open_stretches])
        self.n_open = int(open_edges.size)
        self._open_cells = geom.edge_cellL[open_edges]
        cell_r = geom.edge_cellR.copy()
        cell_r[open_edges] = -2 - np.arange(self.n_open, dtype=cell_r.dtype)
        self.t_ecL = g(geom.edge_cellL.view(np.float32))
        self.t_ecR = g(cell_r.view(np.float32))
        self.t_dh = g(np.zeros(self.N, dtype=np.float32))
        self.t_dhu = g(np.zeros(self.N, dtype=np.float32))
        self.t_dhv = g(np.zeros(self.N, dtype=np.float32))
        self.t_h1 = g(np.zeros(self.N, dtype=np.float32))
        self.t_hu1 = g(np.zeros(self.N, dtype=np.float32))
        self.t_hv1 = g(np.zeros(self.N, dtype=np.float32))
        self.t_nmann = g(n0)
        self.t_dtbuf = g(np.array([self._SENTINEL_F], dtype=np.float32))
        # Dedicated CFL scratch buffer (binding 18) — decouples CFL from flux dh[]
        self.t_cfl_scratch = g(np.zeros(self.N, dtype=np.float32))
        # Per-cell positivity-clamp mass, only written when the diagnostic is
        # enabled. Per-cell rather than one global accumulator so each cell's
        # history stays small enough not to absorb its own increments.
        self.t_clamp_mass = g(np.zeros(self.N, dtype=np.float32))

        self._all_tensors = [
            self.t_h,
            self.t_hu,
            self.t_hv,
            self.t_edz,
            self.t_area,
            self.t_elen,
            self.t_enx,
            self.t_eny,
            self.t_ecL,
            self.t_ecR,
            self.t_dh,
            self.t_dhu,
            self.t_dhv,
            self.t_h1,
            self.t_hu1,
            self.t_hv1,
            self.t_nmann,
            self.t_dtbuf,
            self.t_cfl_scratch,
        ]
        # Binding 19 is claimed per-shader, not globally: the update kernels see
        # clamp_mass there, while the time-advance kernel sees its own time_buf.
        # Appending to _all_tensors instead would silently renumber whichever
        # tensor a subclass appends for its own binding 19.
        self._update_tensors = [*self._all_tensors, self.t_clamp_mass]
        # Binding 19 of the flux and CFL kernels: (condition, value) words per open
        # edge. Binding 20 of the flux kernels: each open edge's volume out (hi, lo)
        # and in (hi, lo). Kompute rejects empty tensors: pad a mesh with no open edges.
        n_words = max(self.n_open, 1)
        self.t_open = g(np.zeros(_OPEN_WORDS * n_words, dtype=np.float32))
        self.t_open_volume = g(np.zeros(_OPEN_VOLUME_WORDS * n_words, dtype=np.float32))
        self._flux_tensors = [*self._all_tensors, self.t_open, self.t_open_volume]
        self._mgr.sequence().record(
            kp.OpTensorSyncDevice([*self._update_tensors, self.t_open, self.t_open_volume])
        ).eval()

    def _build_algorithms(self, spv: dict[str, bytes]) -> None:
        N, E = self.N, self.E
        wg_e, wg_c = compute_workgroups(N, E, self._work_group_size)
        g, dt = self._g, self._dry_tol

        self._algo_flux = self._mgr.algorithm(
            self._flux_tensors,
            spv["flux"],
            workgroup=wg_e,
            spec_consts=[],
            push_consts=_pc_flux(E, 0.0, g, dt, self._cfl, 0),
        )
        self._algo_update = self._mgr.algorithm(
            self._update_tensors,
            spv["update"],
            workgroup=wg_c,
            spec_consts=[],
            push_consts=_pc_update(N, 0.0, 0, g, dt, self._cfl),
        )
        self._build_cfl_algos(spv, N, E, wg_e, wg_c, g, dt)
        self._wg_e = wg_e
        self._wg_c = wg_c

        # Source kernel (optional — only built if SPV provided)
        self._algo_source = None
        if "source" in spv:
            self._source_spv = spv["source"]

    def _build_cfl_algos(
        self,
        spv: dict[str, bytes],
        N: int,
        E: int,
        wg_e: tuple[int, int, int],
        wg_c: tuple[int, int, int],
        g: float,
        dt: float,
    ) -> None:
        """Build the CFL accumulation + reduction algorithms (shared by all variants)."""
        self._algo_cfl_accum = self._mgr.algorithm(
            [*self._all_tensors, self.t_open],
            spv["cfl_accum"],
            workgroup=wg_e,
            spec_consts=[],
            push_consts=_pc_cfl_accum(E, g, dt, self._cfl),
        )
        self._algo_cfl_reduce = self._mgr.algorithm(
            self._all_tensors,
            spv["cfl_reduce"],
            workgroup=wg_c,
            spec_consts=[],
            push_consts=_pc_cfl_reduce(N, g, dt, self._cfl),
        )

    def _build_source_algo(self, src_dh_per_sec: NDArray[np.float32]) -> None:
        """Build the GPU source kernel.

        Handles both the base pattern (``_source_spv``, ``[t_h, t_source]``) and
        the dtbuf subclass pattern (``_source_dtbuf_spv``,
        ``[*_all_tensors, t_source_dtbuf]``) — subclasses that set
        ``_source_dtbuf_spv`` in ``_build_algorithms`` get the dtbuf variant
        automatically.
        """
        N = self.N
        _, wg_c = compute_workgroups(N, self.E, self._work_group_size)
        has_dt = hasattr(self, "_source_dtbuf_spv")
        source_spv = getattr(self, "_source_dtbuf_spv" if has_dt else "_source_spv", None)
        if source_spv is None:
            msg = "No source SPV found — set _source_spv or _source_dtbuf_spv in _build_algorithms"
            raise RuntimeError(msg)
        source_tensor = self._mgr.tensor(self._as_f32_1d(src_dh_per_sec))
        if has_dt:
            self.t_source_dtbuf = source_tensor
            self._source_tensors = [*self._all_tensors, source_tensor]
        else:
            self.t_source = source_tensor
            self._source_tensors = [self.t_h, source_tensor]
        self._mgr.sequence().record(kp.OpTensorSyncDevice([source_tensor])).eval()
        self._algo_source = self._mgr.algorithm(
            self._source_tensors,
            source_spv,
            workgroup=wg_c,
            spec_consts=[],
            push_consts=_pc_source(N, 0.0),
        )

    def _upload(self, tensor: kp.Tensor, arr: NDArray[np.float32]) -> None:  # pyright: ignore[reportAttributeAccessIssue]
        self._set_tensor_host_data(tensor, arr)
        self._mgr.sequence().record(kp.OpTensorSyncDevice([tensor])).eval()

    def _download(self, tensor: kp.Tensor) -> NDArray[np.float32]:  # pyright: ignore[reportAttributeAccessIssue]
        self._mgr.sequence().record(kp.OpTensorSyncLocal([tensor])).eval()
        return np.array(tensor.data(), dtype=np.float32)

    def _read_dt(self) -> float:
        self._mgr.sequence().record(kp.OpTensorSyncLocal([self.t_dtbuf])).eval()
        return float(np.array(self.t_dtbuf.data(), dtype=np.float32)[0])

    def _require_algo_source(self) -> kp.Algorithm:
        if self._algo_source is None:
            raise RuntimeError("source algo not built (call _build_source_algo first)")
        return self._algo_source

    # ── public API ────────────────────────────────────────────────────────────

    def reset(
        self, h0: NDArray[np.float32], hu0: NDArray[np.float32], hv0: NDArray[np.float32]
    ) -> None:
        """Re-upload initial conditions without rebuilding GPU resources."""
        self._h0 = self._as_f32_1d(h0).copy()
        self._hu0 = self._as_f32_1d(hu0).copy()
        self._hv0 = self._as_f32_1d(hv0).copy()
        zeros = np.zeros(self.N, dtype=np.float32)

        self._set_tensor_host_data(self.t_h, h0)
        self._set_tensor_host_data(self.t_hu, hu0)
        self._set_tensor_host_data(self.t_hv, hv0)
        self._set_tensor_host_data(self.t_dh, zeros)
        self._set_tensor_host_data(self.t_dhu, zeros)
        self._set_tensor_host_data(self.t_dhv, zeros)
        self._set_tensor_host_data(
            self.t_open_volume, np.zeros(self.t_open_volume.size(), dtype=np.float32)
        )

        self._mgr.sequence().record(
            kp.OpTensorSyncDevice(
                [
                    self.t_h,
                    self.t_hu,
                    self.t_hv,
                    self.t_dh,
                    self.t_dhu,
                    self.t_dhv,
                    self.t_open_volume,
                ]
            )
        ).eval()

    def _reset_to_initial(self) -> None:
        """Restore the initial condition, momentum included, for a fresh run."""
        self.reset(self._h0, self._hu0, self._hv0)

    def set_open_conditions(self, kinds: NDArray[np.uint32], values: NDArray[np.float32]) -> None:
        """Set every open edge's condition (``OPEN_*``), in stretch order, before a run."""
        if kinds.size != self.n_open or values.size != self.n_open:
            raise ValueError(
                f"expected {self.n_open} open edges, got {kinds.size} kinds and "
                f"{values.size} values"
            )
        words = np.empty(_OPEN_WORDS * self.n_open, dtype=np.uint32)
        words[0::2] = kinds
        words[1::2] = values.astype(np.float32).view(np.uint32)
        self._upload(self.t_open, words.view(np.float32))

    def download_open_volumes(self) -> NDArray[np.float64]:
        """``(stretches, 2)`` volume [m^3] out of and into the mesh since the start."""
        words = self._download(self.t_open_volume).astype(np.float64)[
            : _OPEN_VOLUME_WORDS * self.n_open
        ]
        pairs = words.reshape(-1, _OPEN_VOLUME_WORDS)
        per_edge = np.column_stack([pairs[:, 0] + pairs[:, 1], pairs[:, 2] + pairs[:, 3]])
        starts = np.r_[0, np.cumsum([edges.size for edges in self._open_stretches])[:-1]]
        return np.add.reduceat(per_edge, starts.astype(np.int64), axis=0)

    def _check_error_free_arithmetic(self) -> None:
        """Refuse to run if the compiler optimised the compensated sums away."""
        if self._eft_checked:
            return
        words = np.array([1.0, 1e-9, 0.0], dtype=np.float32)
        tensor = self._mgr.tensor(words.copy())
        algo = self._mgr.algorithm([tensor], self._eft_probe_spv, workgroup=(1, 1, 1))
        seq = self._mgr.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor]))
        seq.record(kp.OpAlgoDispatch(algo))
        seq.record(kp.OpTensorSyncLocal([tensor]))
        seq.eval()
        if float(np.asarray(tensor.data())[2]) != float(words[1]):
            raise RuntimeError(
                "error-free arithmetic is broken on this device, so compensated depth would "
                "leak mass: set MVK_CONFIG_FAST_MATH_ENABLED=0 before any Vulkan instance is "
                "created, or set compensated = False"
            )
        self._eft_checked = True

    def reset_clamp_mass(self) -> None:
        """Zero the positivity-clamp accumulator."""
        self._upload(self.t_clamp_mass, np.zeros(self.N, dtype=np.float32))

    def download_clamp_mass(self) -> NDArray[np.float32]:
        """Per-cell volume [m^3] created by the positivity clamp since the last reset.

        Zero unless ``track_clamp`` was set before the run. Summing this gives
        the clamp's exact contribution to the volume drift, which is otherwise
        only observable tangled up with the opposing arithmetic loss.
        """
        return self._download(self.t_clamp_mass)

    def download_h(self) -> NDArray[np.float32]:
        """Download the current water-depth array from the GPU."""
        return self._download(self.t_h)

    def download_hu(self) -> NDArray[np.float32]:
        """Download the current x-momentum (h*u) array from the GPU.

        ``t_hu`` holds committed state, not a Heun intermediate: the corrector
        stage writes the averaged result back into ``h``/``hu``/``hv`` while
        ``h1``/``hu1``/``hv1`` carry the predictor (see ``shaders/update.py``).
        """
        return self._download(self.t_hu)

    def download_hv(self) -> NDArray[np.float32]:
        """Download the current y-momentum (h*v) array from the GPU. See ``download_hu``."""
        return self._download(self.t_hv)

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
        """Run the Heun RK2 simulation.

        Parameters
        ----------
        t_end : float
            Simulation end time [s].
        output_interval_s : float
            Simulation-time interval between depth snapshots [s].
            For example, ``300.0`` captures a snapshot every 5 minutes.
        dt_max : float
            Hard upper bound on the time step.
        dt_init : float
            Initial time step before the first CFL estimate.
        progress : bool
            Print step / time information periodically.
        source_rate : (N,) float32, optional
            Constant volumetric source rate per cell [m³/s].
            Positive = inflow, negative = outflow (extraction).
            Applied as  h += dt * source_rate / area  each step.
        source_fn : callable(t_sim: float) -> (N,) float32, optional
            Time-varying source function.  Called each step with the current
            simulation time, must return a (N,) array of volumetric rates [m³/s].
            If provided, overrides ``source_rate``.
        resume : bool, optional
            If True, skip the initial state reset and continue from the GPU
            state left by the previous run() call.  The simulation time starts
            at ``t_start`` instead of 0.
        t_start : float, optional
            Simulation time offset used when ``resume=True``.  Has no effect
            when ``resume=False``.
        cfl_interval : int, optional
            Recompute the CFL time step every *n* steps.  Set to 1 for maximum
            safety (recompute every step); higher values reduce overhead.

        Returns
        -------
        snapshots : list of (N,) float32 arrays
            Water depth at each recorded time.
        snap_times : list of float
            Simulation time of each snapshot.

        """
        N, E = self.N, self.E
        g, dt_ = self._g, self._dry_tol
        cfl = self._cfl

        if not resume:
            self._reset_to_initial()
            self.open_volume_snapshots = []
        if self.n_open:
            self._check_error_free_arithmetic()

        def record_volumes() -> None:
            if self.n_open:
                self.open_volume_snapshots.append(self.download_open_volumes())

        # Precompute inverse area for source application
        inv_area = (1.0 / self._area).astype(np.float32)

        # ── Source setup ──────────────────────────────────────────────────────
        # Prefer GPU-side source kernel; fall back to CPU if SPV not provided
        _use_gpu_source = False
        _src_dh_per_sec = None

        if source_rate is not None and source_fn is None:
            _src_dh_per_sec = np.asarray(source_rate, dtype=np.float32) * inv_area
            if self._algo_source is None and hasattr(self, "_source_spv"):
                self._build_source_algo(_src_dh_per_sec)
            elif self._algo_source is not None and hasattr(self, "t_source"):
                # Update the source tensor with the new rates (e.g. when resuming a phase)
                self._upload(self.t_source, _src_dh_per_sec)
            if self._algo_source is not None:
                _use_gpu_source = True

        _t_offset = t_start if resume else 0.0
        if resume:
            snapshots = [self.download_h().copy()]
            snap_times = [t_start]
        else:
            snapshots = [self._h0.copy()]
            snap_times = [0.0]
        hu_snapshots = [self.download_hu().copy()] if capture_momentum_snapshots else []
        hv_snapshots = [self.download_hv().copy()] if capture_momentum_snapshots else []
        record_volumes()
        t_sim = _t_offset
        step = 0
        dt = dt_init
        next_output_time = _t_offset + output_interval_s
        _wall_start = _time.monotonic()
        _last_log_wall = _wall_start
        _LOG_INTERVAL = 10.0  # print progress every 10s wall-clock

        # CFL recomputation interval — reuse dt between updates with safety margin.
        # Set to 1 to recompute every step (safest; required when source rates are
        # large or vary across phases, as the wave speed can change rapidly).
        CFL_SAFETY = 0.8  # use 80% of CFL dt to account for wave speed changes

        while t_sim < t_end:
            dt = min(dt, t_end - t_sim, dt_max)

            # ── CFL estimate (only every cfl_interval steps) ──────────────────
            if step % cfl_interval == 0:
                np.asarray(self.t_dtbuf.data())[0] = self._SENTINEL_F
                self._mgr.sequence().record(kp.OpTensorSyncDevice([self.t_dtbuf])).eval()
                self._mgr.sequence().record(
                    kp.OpAlgoDispatch(self._algo_cfl_accum, _pc_cfl_accum(E, g, dt_, cfl))
                ).eval()

                self._mgr.sequence().record(
                    kp.OpAlgoDispatch(self._algo_cfl_reduce, _pc_cfl_reduce(N, g, dt_, cfl))
                ).eval()
                dt_cfl = self._read_dt()

                if dt_cfl < CFL_EPSILON_MIN:
                    if progress:
                        sys.stdout.write(
                            f"[solver][step {step}] CFL dt too small ({dt_cfl:.3e}), stopping\n"
                        )
                        sys.stdout.flush()
                    break
                if dt_cfl < CFL_SANITY_MAX:
                    # Real CFL estimate — update dt
                    dt = min(dt_cfl * CFL_SAFETY, dt_max)
                # else: all-dry domain returned sentinel; keep current dt (dt_init)
            dt = min(dt, t_end - t_sim)

            # ── Heun stage 0 — predictor ──────────────────────────────────────
            self._mgr.sequence().record(
                kp.OpAlgoDispatch(self._algo_flux, _pc_flux(E, dt, g, dt_, cfl, 0))
            ).eval()
            self._mgr.sequence().record(
                kp.OpAlgoDispatch(
                    self._algo_update, _pc_update(N, dt, 0, g, dt_, cfl, self.track_clamp)
                )
            ).eval()

            # ── Heun stage 1 — corrector ──────────────────────────────────────
            self._mgr.sequence().record(
                kp.OpAlgoDispatch(self._algo_flux, _pc_flux(E, dt, g, dt_, cfl, 1))
            ).eval()
            self._mgr.sequence().record(
                kp.OpAlgoDispatch(
                    self._algo_update, _pc_update(N, dt, 1, g, dt_, cfl, self.track_clamp)
                )
            ).eval()

            # ── Source term (GPU kernel — no CPU round-trip) ──────────────────
            if _use_gpu_source:
                self._mgr.sequence().record(
                    kp.OpAlgoDispatch(self._require_algo_source(), _pc_source(N, dt))
                ).eval()
            elif not _use_gpu_source:
                if source_fn is not None:
                    src = np.asarray(source_fn(t_sim), dtype=np.float32)
                    dh_src = dt * src * inv_area
                    h_cur = self._download(self.t_h)
                    h_cur += dh_src
                    np.maximum(h_cur, 0.0, out=h_cur)
                    self._upload(self.t_h, h_cur)
                elif _src_dh_per_sec is not None:
                    dh_src = dt * _src_dh_per_sec
                    h_cur = self._download(self.t_h)
                    h_cur += dh_src
                    np.maximum(h_cur, 0.0, out=h_cur)
                    self._upload(self.t_h, h_cur)

            t_sim += dt
            step += 1

            # ── Periodic wall-clock progress (every 10 s, always) ────────────
            _now = _time.monotonic()
            if _now - _last_log_wall >= _LOG_INTERVAL:
                _wall_elapsed = _now - _wall_start
                _pct = t_sim / t_end * 100.0 if t_end > 0 else 0.0
                _last_log_wall = _now
                if progress:
                    _eta = (
                        (_wall_elapsed / t_sim) * max(t_end - t_sim, 0.0)
                        if t_sim > SIMULATION_TIME_EPSILON
                        else float("inf")
                    )
                    _eta_str = f"{_eta:.1f}s" if np.isfinite(_eta) else "inf"
                    sys.stdout.write(
                        f"[solver][step {step}] sim={t_sim:.2f}/{t_end:.2f}s "
                        f"({_pct:.1f}%) dt={dt:.4f} eta~{_eta_str}\n"
                    )
                    sys.stdout.flush()

            if t_sim >= next_output_time:
                h_snap = self.download_h()
                hu_snap = self.download_hu().copy() if capture_momentum_snapshots else None
                hv_snap = self.download_hv().copy() if capture_momentum_snapshots else None
                if not np.isfinite(h_snap).all():
                    snapshots.append(np.nan_to_num(h_snap, nan=0.0, posinf=0.0).copy())
                    snap_times.append(t_sim)
                    if hu_snap is not None and hv_snap is not None:
                        hu_snapshots.append(hu_snap)
                        hv_snapshots.append(hv_snap)
                    record_volumes()
                    break
                if progress:
                    _wall_elapsed = _time.monotonic() - _wall_start
                    _snap_idx = len(snapshots)
                    sys.stdout.write(
                        f"[solver] snapshot {_snap_idx} captured at t={t_sim:.2f}s "
                        f"(wall={_wall_elapsed:.1f}s)\n"
                    )
                    sys.stdout.flush()
                snapshots.append(h_snap.copy())
                snap_times.append(t_sim)
                if hu_snap is not None and hv_snap is not None:
                    hu_snapshots.append(hu_snap)
                    hv_snapshots.append(hv_snap)
                record_volumes()
                next_output_time += output_interval_s

        self.steps_total += step
        h_final = self.download_h()
        if not snapshots or not np.allclose(snapshots[-1], h_final):
            snapshots.append(h_final.copy())
            snap_times.append(t_sim)
            if capture_momentum_snapshots:
                hu_snapshots.append(self.download_hu().copy())
                hv_snapshots.append(self.download_hv().copy())
            record_volumes()

        if capture_momentum_snapshots:
            return snapshots, snap_times, hu_snapshots, hv_snapshots
        return snapshots, snap_times
