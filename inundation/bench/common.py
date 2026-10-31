"""Shared data, plotting, fingerprint, and invariant helpers for benchmarks."""

from __future__ import annotations

import hashlib
import heapq
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..mesh.loader import load_mesh_file
from ..tuning import (
    FINAL_TIME_TOLERANCE_S,
    MIN_DEPTH_THRESHOLD_M,
    MIN_REPEATS_FOR_DETERMINISM,
    MIN_SNAPSHOTS,
    MIN_VALID_CELLS,
    MIN_VOLUME_RATIO,
    MIN_WET_CELLS,
    WET_CELL_THRESHOLD,
)
from ..workflow import (
    PointSource,
    SimulationPhase,
    SWEWorkflow,
    WorkflowResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


# Deterministic two-phase threat-run schedule.
_PHASE1_FLOW_RATES: tuple[float, float] = (300.0, 306.0)
_PHASE2_FLOW_RATES: tuple[float, float] = (320.0, 314.5)


@dataclass(frozen=True)
class RunFingerprint:
    """Stable fingerprint + invariants for one simulation run.

    ``steps_total`` is populated for both backends — ``WorkflowResult.steps_total``
    is sourced from ``getattr(self.solver, "steps_total", 0)`` and the Vulkan
    ``SWESolver`` base class maintains the counter (see ``swe_gpu.py``).
    """

    run_index: int
    solver_impl: str
    snap_count: int
    snap_times: list[float]
    snap_times_hash: str
    snapshot_hashes: list[str]
    h_final_hash: str
    volume_final_bits: int
    volume_injected_bits: int
    max_depth: float
    wet_cells_final: int
    wall_seconds: float
    steps_total: int
    invariant_ok: bool
    invariant_reason: str


def hash_array(arr: NDArray[np.floating[Any]]) -> str:
    a = np.ascontiguousarray(arr)
    data = a.view(np.uint8).tobytes()
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def float64_bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def validate_run_invariants(
    snap_times: NDArray[np.floating[Any]],
    snapshots: Sequence[NDArray[np.floating[Any]]],
    h_final: NDArray[np.floating[Any]],
    volume_final: float,
    volume_injected: float,
    expected_final_time_s: float,
    *,
    min_snapshots: int = MIN_SNAPSHOTS,
) -> tuple[bool, str]:
    """Validate production-like correctness invariants for one run.

    The ``min_snapshots`` kwarg lets shorter warm-start half-runs use a lower
    bar without weakening the full-duration determinism/parity checks.
    """
    if len(snapshots) < min_snapshots:
        return False, f"too_few_snapshots:{len(snapshots)}"
    if not np.all(np.diff(snap_times) >= 0.0):
        return False, "non_monotonic_snap_times"
    if abs(snap_times[-1] - expected_final_time_s) > FINAL_TIME_TOLERANCE_S:
        return False, f"unexpected_final_time:{snap_times[-1]:.3f}"
    if not np.isfinite(h_final).all():
        return False, "h_final_non_finite"
    max_depth = float(np.max(h_final))
    if max_depth <= MIN_DEPTH_THRESHOLD_M:
        return False, f"max_depth_too_small:{max_depth:.6f}"
    wet_cells_final = int(np.count_nonzero(h_final > WET_CELL_THRESHOLD))
    if wet_cells_final < MIN_WET_CELLS:
        return False, f"too_few_wet_cells:{wet_cells_final}"
    if volume_injected <= 0.0:
        return False, f"invalid_volume_injected:{volume_injected:.3f}"
    vol_ratio = volume_final / volume_injected
    if not np.isfinite(vol_ratio):
        return False, "volume_ratio_non_finite"
    if vol_ratio < MIN_VOLUME_RATIO:
        return False, f"volume_ratio_too_small:{vol_ratio:.6f}"
    return True, "ok"


# ── Gate primitives (pure: arrays and scalars only, no solver objects) ──────
# The harnesses' _evaluate_case functions take a live SWEWorkflow, so they can
# only run behind a GPU and cannot be unit-tested. These are the pieces that
# every harness repeats; keeping them free of solver objects is what makes the
# scoring logic testable on a clean checkout.


def nearest_cell_indices(
    cx: NDArray[np.floating[Any]],
    cy: NDArray[np.floating[Any]],
    points: NDArray[np.floating[Any]],
) -> NDArray[np.int64]:
    """Index of the cell centroid nearest each ``(x, y)`` in ``points``."""
    cx64 = np.asarray(cx, dtype=np.float64)
    cy64 = np.asarray(cy, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if cx64.shape != cy64.shape:
        raise ValueError(f"cx and cy must match, got {cx64.shape} and {cy64.shape}")
    if cx64.size == 0:
        raise ValueError("no cell centroids given")
    nearest = np.empty(pts.shape[0], dtype=np.int64)
    for k, (px, py) in enumerate(pts):
        nearest[k] = int(np.argmin((cx64 - px) ** 2 + (cy64 - py) ** 2))
    return nearest


def volume_drift_rel(volume_final_m3: float, volume_reference_m3: float) -> float:
    """Relative volume drift; ``inf`` when the reference is non-positive.

    The guard matters: a case whose reference volume is zero (nothing injected,
    nothing initially present) must fail its mass gate rather than divide by
    zero and report a spurious pass.
    """
    if not volume_reference_m3 > 0.0:
        return float("inf")
    return abs(volume_final_m3 - volume_reference_m3) / volume_reference_m3


def check_state_health(
    h_final: NDArray[np.floating[Any]], *, min_depth_tol: float = 0.0
) -> tuple[bool, float, list[str]]:
    """Finiteness + positivity of a final depth field.

    Returns ``(finite, min_depth, fail_reasons)``. ``min_depth_tol`` is the
    most negative depth tolerated: 0.0 for the frictionless analytical cases,
    a small negative for the wetting/drying ones where the clamp can leave
    round-off below zero.
    """
    h = np.asarray(h_final)
    finite = bool(np.isfinite(h).all())
    min_depth = float(h.min()) if h.size else 0.0
    reasons: list[str] = []
    if not finite:
        reasons.append("h_final_non_finite")
    if min_depth < min_depth_tol:
        reasons.append(f"negative_depth:{min_depth:.3e}")
    return finite, min_depth, reasons


def l1_relative_error(
    actual: NDArray[np.floating[Any]], reference: NDArray[np.floating[Any]]
) -> float:
    """``sum|actual - reference| / sum|reference|``; ``inf`` if the reference is zero."""
    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {r.shape}")
    denom = float(np.abs(r).sum())
    if denom <= 0.0:
        return float("inf")
    return float(np.abs(a - r).sum() / denom)


def max_relative_error(
    actual: NDArray[np.floating[Any]], reference: NDArray[np.floating[Any]]
) -> float:
    """Largest relative error where both values are finite and the reference positive.

    Entries the simulation never reached (``inf``) are skipped rather than
    poisoning the reduction; ``inf`` is returned only when nothing is
    comparable at all.
    """
    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {r.shape}")
    usable = np.isfinite(a) & np.isfinite(r) & (r > 0.0)
    if not bool(usable.any()):
        return float("inf")
    return float((np.abs(a[usable] - r[usable]) / r[usable]).max())


@dataclass(frozen=True)
class ConvergenceFit:
    """Observed order of accuracy of an error metric under mesh refinement."""

    order: float
    pairwise_orders: list[float]
    monotone: bool


def observed_order(dx: Sequence[float], error: Sequence[float]) -> ConvergenceFit:
    """Fit ``error ~ dx**p`` over a refinement sequence given coarse-to-fine.

    ``order`` is the least-squares slope of ``log(error)`` against ``log(dx)``;
    ``pairwise_orders`` is the same ratio taken between consecutive levels, which
    is what exposes a curve that starts at the scheme's order and then flattens.
    ``monotone`` is the load-bearing one: truncation error must fall under
    refinement, so a level where it does not is evidence of a non-discretisation
    error floor regardless of what the fitted slope says.
    """
    h = np.asarray(dx, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    if h.shape != e.shape:
        raise ValueError(f"shape mismatch: {h.shape} vs {e.shape}")
    if h.size < 2:
        raise ValueError("need at least two refinement levels")
    if not bool(np.all(np.diff(h) < 0.0)):
        raise ValueError("dx must be strictly decreasing (coarse to fine)")
    if not bool(np.all(np.isfinite(e) & (e > 0.0))):
        raise ValueError("errors must be finite and positive to fit an order")
    log_h, log_e = np.log(h), np.log(e)
    pairwise = np.diff(log_e) / np.diff(log_h)
    slope = float(np.polyfit(log_h, log_e, 1)[0])
    return ConvergenceFit(
        order=slope,
        pairwise_orders=[float(p) for p in pairwise],
        monotone=bool(np.all(np.diff(e) < 0.0)),
    )


def build_fingerprint(
    run_index: int,
    solver_impl: str,
    result: WorkflowResult,
    expected_final_time_s: float,
    *,
    min_snapshots: int = MIN_SNAPSHOTS,
) -> RunFingerprint:
    """Build a RunFingerprint from a completed WorkflowResult."""
    snap_times = np.asarray(result.snap_times, dtype=np.float64)
    snapshots = [np.asarray(s, dtype=np.float32) for s in result.snapshots]
    h_final = np.asarray(result.h_final, dtype=np.float32)
    invariant_ok, invariant_reason = validate_run_invariants(
        snap_times=snap_times,
        snapshots=snapshots,
        h_final=h_final,
        volume_final=float(result.volume_final_m3),
        volume_injected=float(result.volume_injected_m3),
        expected_final_time_s=expected_final_time_s,
        min_snapshots=min_snapshots,
    )
    return RunFingerprint(
        run_index=run_index,
        solver_impl=solver_impl,
        snap_count=len(snapshots),
        snap_times=[float(t) for t in snap_times],
        snap_times_hash=hash_array(snap_times),
        snapshot_hashes=[hash_array(s) for s in snapshots],
        h_final_hash=hash_array(h_final),
        volume_final_bits=float64_bits(result.volume_final_m3),
        volume_injected_bits=float64_bits(result.volume_injected_m3),
        max_depth=float(np.max(h_final)) if h_final.size else 0.0,
        wet_cells_final=int(np.count_nonzero(h_final > WET_CELL_THRESHOLD)),
        wall_seconds=float(result.wall_seconds),
        steps_total=int(result.steps_total),
        invariant_ok=invariant_ok,
        invariant_reason=invariant_reason,
    )


def _mesh_vertices_for_plot(
    workflow: SWEWorkflow,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """Raw (verts, faces, face_offsets) for plotting, re-reading the mesh if needed.

    ``prepare()`` only populates these on the mesh-file path — a geometry-cache
    hit stores geom+perm alone — so the mesh is re-read here for plotting. A
    cache-only workflow has no mesh file to fall back on.
    """
    verts = workflow.verts
    faces_flat = workflow.faces_flat
    face_offsets = workflow.face_offsets
    if verts is not None and faces_flat is not None and face_offsets is not None:
        return verts, faces_flat, face_offsets

    mesh_source = workflow.config.mesh_source
    if mesh_source is None:
        raise RuntimeError(
            "plotting needs the original mesh file, but this workflow was prepared "
            "from a geometry cache artifact (config.mesh_source is None)"
        )
    sys.stdout.write(f"[plot] re-loading mesh for plotting: {mesh_source}\n")
    sys.stdout.flush()
    verts, faces_flat, face_offsets, _zb, _manning_n = load_mesh_file(mesh_source)
    return verts, faces_flat, face_offsets


def save_depth_png(
    workflow: SWEWorkflow,
    h_final: NDArray[np.float32],
    t_final_s: float,
    out_path: Path,
) -> None:
    """Save a water-depth PNG. Requires matplotlib (--viz extra)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import LinearSegmentedColormap, Normalize
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for --plot[/-dir]. Install with: "
            'uv pip install -e ".[viz]" (or pip install -r requirements.txt)'
        ) from exc

    if workflow.geom is None or workflow.perm is None:
        raise RuntimeError("Workflow must be prepared before plotting depth")

    verts, faces_flat, face_offsets = _mesh_vertices_for_plot(workflow)

    perm = workflow.perm
    n_cells = workflow.geom.area.shape[0]

    polygons_reordered: list[NDArray[np.float64]] = []
    for new_i in range(n_cells):
        old_i = int(perm[new_i])
        idx = faces_flat[face_offsets[old_i] : face_offsets[old_i + 1]]
        polygons_reordered.append(verts[idx])

    water_cmap_offset = 0.28
    water_base: NDArray[np.floating[Any]] = cast(
        "NDArray[np.floating[Any]]",
        cast("Any", plt.cm.Blues)(np.linspace(water_cmap_offset, 1.0, 256)),
    )
    water_cmap = LinearSegmentedColormap.from_list("BluesOffset", water_base)

    _plt: Any = plt
    fig_ax: tuple[Figure, Axes] = cast("tuple[Figure, Axes]", _plt.subplots(1, 1, figsize=(13, 5)))
    fig, ax = fig_ax
    pc = PolyCollection(
        polygons_reordered,
        array=h_final,
        cmap=water_cmap,
        edgecolors="face",
        linewidths=0.1,
        norm=Normalize(vmin=0.0, vmax=4.0),
    )
    ax.add_collection(pc)
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    cast("Any", fig).colorbar(pc, ax=ax, label="h [m]")
    ax.set_aspect("equal")
    cast("Any", ax).set_title(f"Water depth at t={t_final_s:.1f}s")
    cast("Any", ax).set_xlabel("x")
    cast("Any", ax).set_ylabel("y")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    cast("Any", fig).savefig(out_path, dpi=300)
    plt.close(fig)
    sys.stdout.write(f"[plot] water-depth PNG saved: {out_path}\n")
    sys.stdout.flush()


def _reordered_polygons(
    workflow: SWEWorkflow,
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Build per-cell polygons (solver order) + vertex array for plotting."""
    if workflow.geom is None or workflow.perm is None:
        raise RuntimeError("Workflow must be prepared before plotting depth")
    verts, faces_flat, face_offsets = _mesh_vertices_for_plot(workflow)
    perm = workflow.perm
    n_cells = workflow.geom.area.shape[0]
    polygons: list[NDArray[np.float64]] = []
    for new_i in range(n_cells):
        old_i = int(perm[new_i])
        idx = faces_flat[face_offsets[old_i] : face_offsets[old_i + 1]]
        polygons.append(verts[idx])
    return polygons, verts


def resample_snapshots_uniform(
    snapshots: Sequence[NDArray[np.floating[Any]]],
    snap_times: Sequence[float],
    n_frames: int,
) -> tuple[list[NDArray[np.floating[Any]]], list[float]]:
    """Pick the snapshot nearest each of ``n_frames`` evenly spaced sim times.

    Solvers emit a snapshot at every phase boundary as well as on the output
    interval, so a run whose hydrograph is chopped into many short phases
    returns frames bunched into the inflow window — an animation that races
    once the phases end. Resampling on simulated time restores constant
    playback speed; repeated frames where nothing new was captured are
    intentional.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    times = np.asarray(snap_times, dtype=np.float64)
    if times.shape[0] != len(snapshots):
        raise ValueError("snapshots and snap_times must have equal length")
    targets = np.linspace(float(times[0]), float(times[-1]), n_frames)
    right = np.clip(np.searchsorted(times, targets), 1, times.shape[0] - 1)
    left = right - 1
    nearest = np.where(targets - times[left] <= times[right] - targets, left, right)
    return [snapshots[int(i)] for i in nearest], [float(t) for t in targets]


def save_depth_gif(
    workflow: SWEWorkflow,
    snapshots: Sequence[NDArray[np.floating[Any]]],
    snap_times: Sequence[float],
    out_path: Path,
    *,
    vmax: float,
    title_prefix: str = "",
    fps: int = 12,
) -> None:
    """Save a top-down water-depth animation (gif). Requires matplotlib + pillow (viz extra)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.animation as manimation
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import LinearSegmentedColormap, Normalize
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib (+ pillow for the gif writer) is required for the 2D gif. "
            'Install with: uv pip install -e ".[viz]"'
        ) from exc

    frames = [
        (float(t), np.asarray(s, np.float32))
        for s, t in zip(snapshots, snap_times, strict=True)
        if t > 0.0
    ]
    if not frames:
        raise RuntimeError("gif run produced no usable snapshots")

    polygons, verts = _reordered_polygons(workflow)
    if workflow.geom is None:
        raise RuntimeError("Workflow must be prepared before plotting depth")
    zb = np.asarray(workflow.geom.zb, np.float32)

    # Terrain background: green gradient by bed elevation (low -> high).
    terrain_cmap = LinearSegmentedColormap.from_list(
        "TerrainGreens",
        ["#14532d", "#3f6212", "#65a30d", "#a3b18a", "#dcedc8"],
    )
    z_lo = float(zb.min())
    z_hi = float(zb.max())
    if z_hi - z_lo < 1e-6:  # flat bed -> uniform terrain color
        z_hi = z_lo + 1.0
    terrain_norm = Normalize(vmin=z_lo, vmax=z_hi)

    # Water overlay: Blues, transparent where dry so the terrain shows through.
    water_base: NDArray[np.floating[Any]] = cast(
        "NDArray[np.floating[Any]]",
        cast("Any", plt.cm.Blues)(np.linspace(0.28, 1.0, 256)),
    )
    water_cmap = LinearSegmentedColormap.from_list("BluesOffset", water_base)
    water_norm = Normalize(vmin=0.0, vmax=vmax)
    dry_thresh = max(1e-4, 1e-3 * vmax)

    def _water_rgba(h: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        rgba = np.asarray(cast("Any", water_cmap)(water_norm(h)), np.float64)
        rgba[np.asarray(h) <= dry_thresh, 3] = 0.0
        return rgba

    _plt: Any = plt
    _manim: Any = manimation
    fig: Any
    ax: Any
    fig, ax = _plt.subplots(figsize=(6, 5))
    terrain_pc: Any = PolyCollection(
        polygons,
        array=zb,
        cmap=terrain_cmap,
        edgecolors="face",
        linewidths=0.0,
        norm=terrain_norm,
    )
    ax.add_collection(terrain_pc)
    pc: Any = PolyCollection(
        polygons,
        facecolors=_water_rgba(frames[0][1]),
        edgecolors="face",
        linewidths=0.0,
    )
    ax.add_collection(pc)
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    water_sm: Any = _plt.cm.ScalarMappable(norm=water_norm, cmap=water_cmap)
    water_sm.set_array(np.empty(0, np.float32))
    fig.colorbar(terrain_pc, ax=ax, label="bed z [m]")
    fig.colorbar(water_sm, ax=ax, label="h [m]")

    def _draw(frame: int) -> None:
        t_snap, h = frames[frame]
        pc.set_facecolors(_water_rgba(h))
        ax.set_title(f"{title_prefix}t={t_snap:.2f}s")

    anim: Any = _manim.FuncAnimation(fig, _draw, frames=len(frames), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=_manim.PillowWriter(fps=fps))
    _plt.close(fig)
    sys.stdout.write(f"[gif] top-down depth animation saved: {out_path} ({len(frames)} frames)\n")
    sys.stdout.flush()


def save_profile_gif(
    workflow: SWEWorkflow,
    snapshots: Sequence[NDArray[np.floating[Any]]],
    snap_times: Sequence[float],
    out_path: Path,
    *,
    vmax: float,
    title_prefix: str = "",
    fps: int = 12,
) -> None:
    """Save a longitudinal side-view (bed + water surface vs x) animation (gif).

    Cells are aggregated into along-channel (x) bins and averaged across the
    width, so this is meaningful for prismatic (y-independent) beds such as
    the momentum-obstruction case. Requires matplotlib + pillow (viz extra).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.animation as manimation
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib (+ pillow for the gif writer) is required for the profile gif. "
            'Install with: uv pip install -e ".[viz]"'
        ) from exc

    if workflow.geom is None:
        raise RuntimeError("Workflow must be prepared before plotting a profile")

    frames = [
        (float(t), np.asarray(s, np.float32))
        for s, t in zip(snapshots, snap_times, strict=True)
        if t > 0.0
    ]
    if not frames:
        raise RuntimeError("gif run produced no usable snapshots")

    cx = workflow.geom.world_centroid[:, 0]
    zb = np.asarray(workflow.geom.zb, np.float64)

    # Aggregate into along-channel bins (average across the width per x).
    xs, inv = np.unique(np.round(cx, 6), return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    zb_prof = np.bincount(inv, weights=zb) / counts
    order = np.argsort(xs)
    xs = xs[order]
    zb_prof = zb_prof[order]

    def _profile(h: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        h_prof = np.bincount(inv, weights=np.asarray(h, np.float64)) / counts
        return h_prof[order]

    dry_thresh = max(1e-4, 1e-3 * vmax)
    wse_max = max(float((zb_prof + _profile(h)).max()) for _t, h in frames)
    z_lo = float(zb_prof.min())
    z_hi = max(float(zb_prof.max()), wse_max)
    pad = 0.08 * max(z_hi - z_lo, 1.0)

    _plt: Any = plt
    _manim: Any = manimation
    fig: Any
    ax: Any
    fig, ax = _plt.subplots(figsize=(8, 4))
    ax.fill_between(xs, z_lo - pad, zb_prof, color="#6b4f2a", zorder=1)
    ax.plot(xs, zb_prof, color="#3f2d16", linewidth=1.2, zorder=3)
    water_artists: list[Any] = []

    def _draw(frame: int) -> None:
        t_snap, h = frames[frame]
        for art in water_artists:
            art.remove()
        water_artists.clear()
        prof = _profile(h)
        wse = zb_prof + prof
        wet = prof > dry_thresh
        fill = ax.fill_between(xs, zb_prof, wse, where=wet, color="#2563eb", alpha=0.75, zorder=2)
        water_artists.append(fill)
        (line,) = ax.plot(
            np.where(wet, xs, np.nan),
            np.where(wet, wse, np.nan),
            color="#1e3a8a",
            linewidth=1.0,
            zorder=4,
        )
        water_artists.append(line)
        ax.set_title(f"{title_prefix}t={t_snap:.2f}s (side view)")

    ax.set_xlim(float(xs.min()), float(xs.max()))
    ax.set_ylim(z_lo - pad, z_hi + pad)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("elevation [m]")
    _draw(0)

    anim: Any = _manim.FuncAnimation(fig, _draw, frames=len(frames), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=_manim.PillowWriter(fps=fps))
    _plt.close(fig)
    sys.stdout.write(
        f"[gif] side-view profile animation saved: {out_path} ({len(frames)} frames)\n"
    )
    sys.stdout.flush()


def water_surface_elevation(
    snapshots: Sequence[NDArray[np.floating[Any]]],
    bed: NDArray[np.floating[Any]],
    cells: NDArray[np.integer[Any]],
) -> NDArray[np.float64]:
    """Return WSE [m] for ordered gauge cells across snapshot times."""
    h = np.asarray(snapshots, dtype=np.float64)[:, np.asarray(cells, dtype=np.int64)]
    return h + np.asarray(bed, dtype=np.float64)[np.asarray(cells, dtype=np.int64)]


def speed_from_momentum(
    hu_snapshots: Sequence[NDArray[np.floating[Any]]] | None,
    hv_snapshots: Sequence[NDArray[np.floating[Any]]] | None,
    depth_snapshots: Sequence[NDArray[np.floating[Any]]],
    cells: NDArray[np.integer[Any]] | None = None,
    *,
    dry_floor: float = 1e-9,
) -> NDArray[np.float64] | None:
    """Return speed [m/s], or None when momentum snapshots are unavailable."""
    if hu_snapshots is None or hv_snapshots is None:
        return None
    hu = np.asarray(hu_snapshots, dtype=np.float64)
    hv = np.asarray(hv_snapshots, dtype=np.float64)
    h = np.asarray(depth_snapshots, dtype=np.float64)
    speed = np.hypot(hu, hv) / np.maximum(h, dry_floor)
    return speed if cells is None else speed[:, np.asarray(cells, dtype=np.int64)]


def _plot_imports() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for --plot[/-dir]. Install with: "
            'uv pip install -e ".[viz]" (or pip install -r requirements.txt)'
        ) from exc
    return plt, plt


def save_line_plot(
    times_s: Sequence[float],
    series: Sequence[NDArray[np.floating[Any]]],
    labels: Sequence[str],
    out_path: Path,
    *,
    ylabel: str,
    title: str,
    annotation: str | None = None,
) -> None:
    """Write a labeled multi-series line plot."""
    plt, _ = _plot_imports()
    fig, ax = plt.subplots(figsize=(9, 5))
    for values, label in zip(series, labels, strict=True):
        ax.plot(np.asarray(times_s) / 3600.0, values, label=label)
    if annotation:
        ax.axhline(float(annotation), color="black", linestyle="--", label="crest")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_faceted_line_plot(
    times_s: Sequence[float],
    values: NDArray[np.floating[Any]],
    labels: Sequence[str],
    out_path: Path,
    *,
    ylabel: str,
    title: str,
) -> None:
    """Write one small-multiple axis per ordered point."""
    plt, _ = _plot_imports()
    data = np.asarray(values, dtype=np.float64)
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, max(3, 2.0 * len(labels))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, row, label in zip(axes, data.T, labels, strict=True):
        ax.plot(np.asarray(times_s) / 3600.0, row)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Point {label}", loc="left")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [hours]")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_depth_contours(
    grids: Sequence[NDArray[np.floating[Any]]],
    times_s: Sequence[float],
    out_path: Path,
    *,
    extent: tuple[float, float, float, float],
    level: float = 0.15,
) -> None:
    """Write depth maps with the requested wet-depth contour level."""
    plt, _ = _plot_imports()
    fig, axes = plt.subplots(1, len(grids), figsize=(6 * len(grids), 5), squeeze=False)
    for ax, grid, time_s in zip(axes[0], grids, times_s, strict=True):
        image = ax.imshow(grid, origin="lower", extent=extent)
        ax.contour(grid, levels=[level], origin="lower", extent=extent, colors="black")
        ax.set_title(f"t={time_s / 3600.0:g} h")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(image, ax=ax, label="Depth [m]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_cross_section(
    distance_m: NDArray[np.floating[Any]],
    values: NDArray[np.floating[Any]],
    out_path: Path,
    *,
    ylabel: str,
    title: str,
) -> None:
    """Write a longitudinal cross-section."""
    plt, _ = _plot_imports()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(distance_m, values)
    ax.set_xlabel("Distance west to east [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_pipeline_like_phases(
    workflow: SWEWorkflow,
    phase_duration_s: float,
) -> list[SimulationPhase]:
    """Return a deterministic two-phase schedule resembling production threat runs.

    Centers and radii derive from the mesh geometry.
    """
    if workflow.geom is None:
        raise RuntimeError("Workflow must be prepared before building benchmark phases")
    valid = workflow.geom.area > workflow.config.area_tol
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < MIN_VALID_CELLS:
        raise RuntimeError("Benchmark mesh must contain at least two valid cells")
    world = workflow.geom.world_centroid
    sorted_indices = valid_indices[np.argsort(world[valid_indices, 0])]
    source_a = int(sorted_indices[sorted_indices.size // 3])
    source_b = int(sorted_indices[(sorted_indices.size * 2) // 3])
    radius_a = max(1.0, float(np.sqrt(workflow.geom.area[source_a] / np.pi)))
    radius_b = max(1.0, float(np.sqrt(workflow.geom.area[source_b] / np.pi)))
    center_a = (float(world[source_a, 0]), float(world[source_a, 1]))
    center_b = (float(world[source_b, 0]), float(world[source_b, 1]))

    def _phase(flow_rates: tuple[float, float]) -> SimulationPhase:
        return SimulationPhase(
            duration_s=phase_duration_s,
            sources=[
                PointSource(flow_rates[0], center_a, radius_a),
                PointSource(flow_rates[1], center_b, radius_b),
            ],
        )

    return [_phase(_PHASE1_FLOW_RATES), _phase(_PHASE2_FLOW_RATES)]


def compare_fingerprints_exact(a: RunFingerprint, b: RunFingerprint) -> tuple[bool, str]:
    """Bit-exact comparison — only valid for repeats of the *same* solver_impl."""
    if a.snap_count != b.snap_count:
        return False, f"snap_count mismatch: {a.snap_count} != {b.snap_count}"
    if a.snap_times_hash != b.snap_times_hash:
        return False, "snap_times_hash mismatch"
    if a.snapshot_hashes != b.snapshot_hashes:
        return False, "snapshot_hashes mismatch"
    if a.h_final_hash != b.h_final_hash:
        return False, "h_final_hash mismatch"
    if a.volume_final_bits != b.volume_final_bits:
        return False, "volume_final_bits mismatch"
    if a.volume_injected_bits != b.volume_injected_bits:
        return False, "volume_injected_bits mismatch"
    return True, "ok"


def assert_deterministic(runs: list[RunFingerprint]) -> tuple[bool | None, str]:
    if len(runs) < MIN_REPEATS_FOR_DETERMINISM:
        return None, "skipped (repeats=1)"
    reference = runs[0]
    for index, run in enumerate(runs[1:], start=1):
        ok, reason = compare_fingerprints_exact(reference, run)
        if not ok:
            return False, f"run0 vs run{index}: {reason}"
    return True, "ok"


def all_invariants_ok(runs: list[RunFingerprint]) -> tuple[bool, str]:
    for run in runs:
        if not run.invariant_ok:
            return False, f"run{run.run_index}:{run.invariant_reason}"
    return True, "ok"


def write_fingerprint_json(fingerprint: RunFingerprint, out_path: Path) -> None:
    out_path.write_text(json.dumps(asdict(fingerprint), indent=2), encoding="utf-8")


def median_wall(runs: list[RunFingerprint]) -> float:
    return statistics.median(fp.wall_seconds for fp in runs)


# ── Analytical dam-break test cases (validation-plan.md §1 Tier 1) ──────────


def build_channel_mesh(
    nx: int, ny: int, length: float, width: float
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Regular quad grid on a [0, length] x [0, width] channel (verts, quads)."""
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, width, ny + 1)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    verts = np.column_stack([xv.ravel(), yv.ravel()]).astype(np.float32)

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    quads = np.array(
        [
            [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            for j in range(ny)
            for i in range(nx)
        ],
        dtype=np.int32,
    )
    return verts, quads


def write_mesh_parquet(
    out_path: Path,
    verts: NDArray[np.float32],
    faces: NDArray[np.int32],
    zb: NDArray[np.float32],
) -> None:
    """Write a polygon mesh + per-cell ``z_mean`` bed as a parquet mesh file.

    Produces the same WKB-geometry + ``z_mean`` layout ``load_mesh_file``
    reads back. Callers provide zeros for flat-bed cases or explicit terrain
    elevations for terrain cases. Requires the ``mesh-parquet`` extra.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import shapely
        from shapely import Polygon
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow + shapely are required to write a synthetic benchmark mesh. "
            'Install with: uv pip install -e ".[mesh-parquet]"'
        ) from exc

    if faces.ndim != 2:
        raise ValueError(f"faces must be (N, degree), got shape {faces.shape}")
    if zb.shape[0] != faces.shape[0]:
        raise ValueError(f"zb has {zb.shape[0]} rows, expected {faces.shape[0]}")

    # shapely + pyarrow inference is patchy under strict pyright (see the
    # executionEnvironments note in pyproject.toml) — go through Any aliases
    # (shapely.to_wkb's overloads are partially unknown even at import).
    _pq: Any = cast("Any", pq)
    _to_wkb: Any = cast("Any", shapely).to_wkb
    wkb_list: list[bytes] = [cast("bytes", _to_wkb(Polygon(verts[face]))) for face in faces]
    _pa: Any = pa
    table: Any = _pa.table(
        {
            "geometry": _pa.array(wkb_list, type=_pa.binary()),
            "z_mean": _pa.array(zb.astype(np.float64), type=_pa.float64()),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _pq.write_table(table, out_path)


def ritter_solution(
    x: NDArray[np.floating[Any]],
    t: float,
    *,
    h_up: float,
    dam_x: float,
    g: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exact Ritter (1892) dry-bed dam-break depth/velocity profile at time t.

    Instantaneous dam removal at ``dam_x`` over a flat frictionless bed,
    upstream depth ``h_up``, downstream dry. Wet front advances at
    ``2*sqrt(g*h_up)``.
    """
    if t <= 0.0:
        raise ValueError(f"t must be positive, got {t}")
    xi = (np.asarray(x, dtype=np.float64) - dam_x) / t
    c0 = float(np.sqrt(g * h_up))

    h = np.zeros_like(xi)
    u = np.zeros_like(xi)

    upstream = xi <= -c0
    h[upstream] = h_up

    fan = (~upstream) & (xi < 2.0 * c0)
    h[fan] = (2.0 * c0 - xi[fan]) ** 2 / (9.0 * g)
    u[fan] = 2.0 * (xi[fan] + c0) / 3.0
    return h, u


def stoker_middle_depth(h_up: float, h_down: float, g: float) -> float:
    """Constant-state depth h_m of the Stoker (1957) wet-bed dam-break.

    Root of the rarefaction/shock matching condition, solved by bisection
    on (h_down, h_up) — no scipy dependency.
    """
    if not 0.0 < h_down < h_up:
        raise ValueError(f"need 0 < h_down < h_up, got h_down={h_down}, h_up={h_up}")
    c0 = float(np.sqrt(g * h_up))

    def f(hm: float) -> float:
        # u_m from the rarefaction (left) minus u_m from the shock jump (right).
        cm = float(np.sqrt(g * hm))
        u_rarefaction = 2.0 * (c0 - cm)
        u_shock = (hm - h_down) * float(np.sqrt(g * (hm + h_down) / (2.0 * hm * h_down)))
        return u_rarefaction - u_shock

    lo, hi = h_down, h_up
    if f(lo) <= 0.0 or f(hi) >= 0.0:
        raise ValueError(f"bisection bracket invalid for h_up={h_up}, h_down={h_down}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def stoker_solution(
    x: NDArray[np.floating[Any]],
    t: float,
    *,
    h_up: float,
    h_down: float,
    dam_x: float,
    g: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exact Stoker (1957) wet-bed dam-break depth/velocity profile at time t.

    Four regions: undisturbed upstream, rarefaction fan, constant state
    (h_m, u_m), and undisturbed downstream behind the bore front.
    """
    if t <= 0.0:
        raise ValueError(f"t must be positive, got {t}")
    xi = (np.asarray(x, dtype=np.float64) - dam_x) / t
    c0 = float(np.sqrt(g * h_up))

    hm = stoker_middle_depth(h_up, h_down, g)
    cm = float(np.sqrt(g * hm))
    um = 2.0 * (c0 - cm)
    shock_speed = um * hm / (hm - h_down)

    h = np.full_like(xi, h_down)
    u = np.zeros_like(xi)

    upstream = xi <= -c0
    h[upstream] = h_up

    fan = (~upstream) & (xi < um - cm)
    h[fan] = (2.0 * c0 - xi[fan]) ** 2 / (9.0 * g)
    u[fan] = 2.0 * (xi[fan] + c0) / 3.0

    plateau = (xi >= um - cm) & (xi < shock_speed)
    h[plateau] = hm
    u[plateau] = um
    return h, u


def _radial_hll_flux(
    hL: NDArray[np.float64],
    mL: NDArray[np.float64],
    hR: NDArray[np.float64],
    mR: NDArray[np.float64],
    g: float,
    dry_tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """HLL mass/momentum flux for the planar SWE part of the radial system."""
    uL = np.where(hL > dry_tol, mL / np.maximum(hL, dry_tol), 0.0)
    uR = np.where(hR > dry_tol, mR / np.maximum(hR, dry_tol), 0.0)
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))

    fh_l, fh_r = mL, mR
    fm_l = mL * uL + 0.5 * g * hL * hL
    fm_r = mR * uR + 0.5 * g * hR * hR

    s_l = np.minimum(uL - cL, uR - cR)
    s_r = np.maximum(uL + cL, uR + cR)
    denom = np.where(s_r - s_l != 0.0, s_r - s_l, 1.0)
    fh_hll = (s_r * fh_l - s_l * fh_r + s_l * s_r * (hR - hL)) / denom
    fm_hll = (s_r * fm_l - s_l * fm_r + s_l * s_r * (mR - mL)) / denom

    fh = np.where(s_l >= 0.0, fh_l, np.where(s_r <= 0.0, fh_r, fh_hll))
    fm = np.where(s_l >= 0.0, fm_l, np.where(s_r <= 0.0, fm_r, fm_hll))
    return fh, fm


def solve_radial_dambreak(
    t: float,
    *,
    h_in: float,
    h_out: float,
    r_dam: float,
    g: float,
    r_max: float,
    n_cells: int = 2000,
    cfl: float = 0.9,
    dry_tol: float = 1e-8,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fine-grid 1D radial finite-volume reference for the circular dam-break.

    Solves the axisymmetric shallow-water equations (no closed form exists)

        d(r h)/dt   + d(r h u)/dr            = 0
        d(r h u)/dt + d(r (h u^2 + g h^2/2))/dr = g h^2 / 2

    with a first-order HLL flux and SSP-RK2 (Heun) time stepping on
    ``n_cells`` cells over ``[0, r_max]``. The ``r``-weighting makes the
    inner (r=0) face flux vanish by construction (radial symmetry) and
    conserves annular mass to round-off. Returns ``(r_centers, h, u)`` at
    time ``t``; use it as the quasi-exact reference for a much coarser 2D
    solver whose depth is radially binned onto the same radii.
    """
    if t <= 0.0:
        raise ValueError(f"t must be positive, got {t}")
    if not 0.0 < r_dam < r_max:
        raise ValueError(f"need 0 < r_dam < r_max, got r_dam={r_dam}, r_max={r_max}")

    dr = r_max / n_cells
    r_c = (np.arange(n_cells, dtype=np.float64) + 0.5) * dr
    r_f = np.arange(n_cells + 1, dtype=np.float64) * dr

    h = np.where(r_c < r_dam, h_in, h_out).astype(np.float64)
    m = np.zeros(n_cells, dtype=np.float64)

    def rhs(
        hs: NDArray[np.float64], ms: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        us = np.where(hs > dry_tol, ms / np.maximum(hs, dry_tol), 0.0)
        fh_int, fm_int = _radial_hll_flux(hs[:-1], ms[:-1], hs[1:], ms[1:], g, dry_tol)
        fh_face = np.empty(n_cells + 1, dtype=np.float64)
        fm_face = np.empty(n_cells + 1, dtype=np.float64)
        fh_face[1:-1] = fh_int
        fm_face[1:-1] = fm_int
        # Inner face at r=0: r-weight is 0, so mass contribution vanishes and
        # radial symmetry (u=0) is enforced automatically.
        fh_face[0] = 0.0
        fm_face[0] = 0.5 * g * hs[0] * hs[0]
        # Outer face: transmissive (the wave never reaches r_max here).
        fh_face[-1] = ms[-1]
        fm_face[-1] = ms[-1] * us[-1] + 0.5 * g * hs[-1] * hs[-1]

        rf_h = r_f * fh_face
        rf_m = r_f * fm_face
        dh = -(rf_h[1:] - rf_h[:-1]) / (dr * r_c)
        dm = (-(rf_m[1:] - rf_m[:-1]) / dr + 0.5 * g * hs * hs) / r_c
        return dh, dm

    elapsed = 0.0
    max_steps = 1_000_000
    for _ in range(max_steps):
        if elapsed >= t:
            break
        u = np.where(h > dry_tol, m / np.maximum(h, dry_tol), 0.0)
        wave = np.abs(u) + np.sqrt(g * np.maximum(h, 0.0))
        max_wave = float(wave.max())
        dt = cfl * dr / max_wave if max_wave > 0.0 else t - elapsed
        dt = min(dt, t - elapsed)

        dh1, dm1 = rhs(h, m)
        h1 = np.maximum(h + dt * dh1, 0.0)
        m1 = np.where(h1 > dry_tol, m + dt * dm1, 0.0)
        dh2, dm2 = rhs(h1, m1)
        h = np.maximum(h + 0.5 * dt * (dh1 + dh2), 0.0)
        m = np.where(h > dry_tol, m + 0.5 * dt * (dm1 + dm2), 0.0)
        elapsed += dt

    u_final = np.where(h > dry_tol, m / np.maximum(h, dry_tol), 0.0)
    return r_c, h, u_final


def radial_dambreak_reference(
    r_eval: NDArray[np.floating[Any]],
    t: float,
    *,
    h_in: float,
    h_out: float,
    r_dam: float,
    g: float,
    r_max: float,
    n_cells: int = 2000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reference (h, u) of the circular dam-break sampled at radii ``r_eval``.

    Thin wrapper over :func:`solve_radial_dambreak` that linearly
    interpolates the fine radial grid onto the coarse evaluation radii.
    """
    r_c, h, u = solve_radial_dambreak(
        t, h_in=h_in, h_out=h_out, r_dam=r_dam, g=g, r_max=r_max, n_cells=n_cells
    )
    r_eval64 = np.asarray(r_eval, dtype=np.float64)
    return np.interp(r_eval64, r_c, h), np.interp(r_eval64, r_c, u)


# ── Published EA dataset rasters + terrain-derived depression storage ───────


@dataclass(frozen=True)
class AsciiGrid:
    """An ESRI ASCII raster with both axes ascending.

    ``z[j, i]`` is the elevation of the cell centred on ``(x[i], y[j])``. The
    file's north→south row order is flipped on load so ``y`` ascends like
    ``x``, matching the benchmark meshes' cell ordering (row-major, y up).
    NODATA cells are NaN.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    cellsize: float


def load_ascii_grid(path: Path) -> AsciiGrid:
    """Parse a 6-header-line ESRI ASCII raster (the EA benchmark DEM format)."""
    header: dict[str, float] = {}
    with path.open(encoding="ascii") as f:
        for _ in range(6):
            key, value = f.readline().split()
            header[key.lower()] = float(value)
        rows = np.loadtxt(f, dtype=np.float64)
    ncols, nrows = int(header["ncols"]), int(header["nrows"])
    if rows.shape != (nrows, ncols):
        raise ValueError(f"raster shape {rows.shape} != header ({nrows}, {ncols})")
    cell = header["cellsize"]
    z = np.flipud(rows).copy()
    z[z == header["nodata_value"]] = np.nan
    return AsciiGrid(
        x=header["xllcorner"] + (np.arange(ncols, dtype=np.float64) + 0.5) * cell,
        y=header["yllcorner"] + (np.arange(nrows, dtype=np.float64) + 0.5) * cell,
        z=z,
        cellsize=cell,
    )


def block_average_grid(
    grid: AsciiGrid,
    *,
    x_edges: NDArray[np.float64],
    y_edges: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Average a raster onto the coarser mesh cells defined by the edge arrays.

    Cell-average (rather than point-sampled) bed elevation is the
    finite-volume-consistent way to take a fine published DEM to the coarser
    modelling resolution the EA specs prescribe. Raster cells outside the
    mesh extent (the DEM apron) are ignored; every mesh cell must receive at
    least one raster cell, and NODATA inside the mesh extent is an error.
    """
    nx = x_edges.shape[0] - 1
    ny = y_edges.shape[0] - 1
    if nx < 1 or ny < 1:
        raise ValueError("x_edges and y_edges must each have at least two entries")
    xi = np.searchsorted(x_edges, grid.x, side="right") - 1
    yj = np.searchsorted(y_edges, grid.y, side="right") - 1
    keep_x = (xi >= 0) & (xi < nx)
    keep_y = (yj >= 0) & (yj < ny)
    z = grid.z[np.ix_(keep_y, keep_x)]
    if not bool(np.isfinite(z).all()):
        raise ValueError("raster has NODATA cells inside the mesh extent")
    flat = (yj[keep_y][:, None] * nx + xi[keep_x][None, :]).ravel()
    count = np.bincount(flat, minlength=nx * ny)
    if bool(np.any(count == 0)):
        raise ValueError("mesh extends beyond the raster coverage (empty mesh cells)")
    total = np.bincount(flat, weights=z.ravel(), minlength=nx * ny)
    return (total / count).reshape(ny, nx)


_NEIGHBOUR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _descend(w: NDArray[np.float64], flat: int) -> int:
    """Walk from ``flat`` to a local minimum of surface ``w`` (4-connected)."""
    ny, nx = w.shape
    while True:
        j, i = divmod(flat, nx)
        best, best_w = flat, w[j, i]
        for dj, di in _NEIGHBOUR_OFFSETS:
            nj, ni = j + dj, i + di
            if 0 <= nj < ny and 0 <= ni < nx and w[nj, ni] < best_w:
                best, best_w = nj * nx + ni, w[nj, ni]
        if best == flat:
            return flat
        flat = best


def _flood_to_sill(
    w: NDArray[np.float64], seed: int
) -> tuple[float, NDArray[np.int64], int | None]:
    """Rise the level from ``seed`` until the pool finds a lower outlet.

    Returns ``(sill level, pool cell indices, outlet index)``; the outlet is
    ``None`` when the whole domain floods without one.
    """
    ny, nx = w.shape
    visited = np.zeros(ny * nx, dtype=np.bool_)
    frontier: list[tuple[float, int]] = [(float(w.flat[seed]), seed)]
    visited[seed] = True
    pool: list[int] = []
    level = float(w.flat[seed])
    while frontier:
        wc, flat = heapq.heappop(frontier)
        if wc < level:
            return level, np.array(pool, dtype=np.int64), flat
        level = wc
        pool.append(flat)
        j, i = divmod(flat, nx)
        for dj, di in _NEIGHBOUR_OFFSETS:
            nj, ni = j + dj, i + di
            if 0 <= nj < ny and 0 <= ni < nx:
                nflat = nj * nx + ni
                if not visited[nflat]:
                    visited[nflat] = True
                    heapq.heappush(frontier, (float(w[nj, ni]), nflat))
    return level, np.array(pool, dtype=np.int64), None


def _pool_capacity(surface: NDArray[np.float64], cell_area: float, level: float) -> float:
    """Volume needed to raise the pool cells ``surface`` to ``level``."""
    return float(np.clip(level - surface, 0.0, None).sum()) * cell_area


def depression_basin(
    zb: NDArray[np.float64], cell: tuple[int, int], *, cell_area: float
) -> tuple[float, NDArray[np.int64], float]:
    """The depression draining ``cell``: (sill elevation, pool cells, capacity).

    Walks downhill from ``cell`` to the basin floor, then raises a water level
    until the pool finds a lower outlet — the sill it would spill over. Water
    at rest cannot stand above that sill, and the basin cannot hold more than
    the returned capacity, so both are terrain-derived bounds a settled
    shallow-water solution must respect.

    ``cell`` is a ``(row, col)`` index into the ``(ny, nx)`` bed. A basin with
    no outlet (the whole domain) comes back with the domain maximum as its
    sill.
    """
    if cell_area <= 0.0:
        raise ValueError(f"cell_area must be positive, got {cell_area}")
    ny, nx = zb.shape
    flat = int(cell[0]) * nx + int(cell[1])
    if not 0 <= flat < ny * nx:
        raise ValueError(f"cell {cell} is outside the ({ny}, {nx}) bed")
    level, pool, _outlet = _flood_to_sill(zb, _descend(zb, flat))
    return level, pool, _pool_capacity(zb.reshape(-1)[pool], cell_area, level)


# ── EA Test 4: radial spread of a source inflow over a flat frictional plain ─


@dataclass(frozen=True)
class RadialInflowSolution:
    """Axisymmetric reference for a source-fed flood spreading over a plain."""

    r_m: NDArray[np.float64]  # (n_cells,) cell centres
    times_s: NDArray[np.float64]  # (n_times,) requested sample times
    h_m: NDArray[np.float64]  # (n_times, n_cells)
    speed_ms: NDArray[np.float64]  # (n_times, n_cells) |u|
    arrival_s: NDArray[np.float64]  # (n_cells,) first wet time, inf if never

    def depth_at(self, time_index: int, radii: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return np.interp(np.asarray(radii, dtype=np.float64), self.r_m, self.h_m[time_index])

    def speed_at(self, time_index: int, radii: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return np.interp(np.asarray(radii, dtype=np.float64), self.r_m, self.speed_ms[time_index])

    def arrival_at(self, radii: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return np.interp(np.asarray(radii, dtype=np.float64), self.r_m, self.arrival_s)


def solve_radial_inflow(
    sample_times_s: Sequence[float],
    *,
    hydrograph_t_s: NDArray[np.float64],
    hydrograph_q_m3s: NDArray[np.float64],
    source_radius_m: float,
    manning_n: float,
    g: float,
    r_max: float,
    wet_tol_m: float = 0.01,
    n_cells: int = 1400,
    cfl: float = 0.9,
    dt_max: float = 2.0,
    dry_tol: float = 1e-8,
) -> RadialInflowSolution:
    """Fine-grid 1D axisymmetric reference for a hydrograph poured onto a plain.

    Solves the axisymmetric shallow-water equations with Manning friction

        d(r h)/dt   + d(r h u)/dr               = r S(t)
        d(r h u)/dt + d(r (h u^2 + g h^2/2))/dr = g h^2 / 2 - r g n^2 u|u| h^(-1/3)

    over a flat bed, using the same first-order HLL flux and SSP-RK2 stepping
    as :func:`solve_radial_dambreak` with the friction applied implicitly. The
    inflow ``S`` is spread uniformly over a disc of ``source_radius_m``.

    For an inflow entering along a *closed wall*, reflection makes the
    half-plane problem identical to this full-plane one with twice the
    discharge — pass the doubled hydrograph. Valid only while the front stays
    clear of every domain boundary; past that the 2D solution sees walls this
    reference does not.

    ``arrival_s`` records when each radius first exceeds ``wet_tol_m``, which
    is what a flood-propagation-speed benchmark actually scores.
    """
    times = np.asarray(sample_times_s, dtype=np.float64)
    if times.size == 0 or bool(np.any(np.diff(times) <= 0.0)):
        raise ValueError("sample_times_s must be non-empty and strictly increasing")
    if float(times[0]) <= 0.0:
        raise ValueError(f"sample times must be positive, got {times[0]}")
    if source_radius_m <= 0.0 or source_radius_m >= r_max:
        raise ValueError(f"need 0 < source_radius_m < r_max, got {source_radius_m}, {r_max}")

    dr = r_max / n_cells
    r_c = (np.arange(n_cells, dtype=np.float64) + 0.5) * dr
    r_f = np.arange(n_cells + 1, dtype=np.float64) * dr
    in_source = r_c < source_radius_m
    source_area = float(np.pi * source_radius_m**2)

    h = np.zeros(n_cells, dtype=np.float64)
    m = np.zeros(n_cells, dtype=np.float64)
    arrival = np.full(n_cells, np.inf, dtype=np.float64)
    h_out = np.zeros((times.size, n_cells), dtype=np.float64)
    speed_out = np.zeros((times.size, n_cells), dtype=np.float64)

    def rhs(
        hs: NDArray[np.float64], ms: NDArray[np.float64], t_mid: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        us = np.where(hs > dry_tol, ms / np.maximum(hs, dry_tol), 0.0)
        fh_int, fm_int = _radial_hll_flux(hs[:-1], ms[:-1], hs[1:], ms[1:], g, dry_tol)
        fh_face = np.empty(n_cells + 1, dtype=np.float64)
        fm_face = np.empty(n_cells + 1, dtype=np.float64)
        fh_face[1:-1] = fh_int
        fm_face[1:-1] = fm_int
        fh_face[0] = 0.0
        fm_face[0] = 0.5 * g * hs[0] * hs[0]
        fh_face[-1] = ms[-1]
        fm_face[-1] = ms[-1] * us[-1] + 0.5 * g * hs[-1] * hs[-1]

        rf_h = r_f * fh_face
        rf_m = r_f * fm_face
        dh = -(rf_h[1:] - rf_h[:-1]) / (dr * r_c)
        dm = (-(rf_m[1:] - rf_m[:-1]) / dr + 0.5 * g * hs * hs) / r_c
        q = float(np.interp(t_mid, hydrograph_t_s, hydrograph_q_m3s))
        return dh + np.where(in_source, q / source_area, 0.0), dm

    t = 0.0
    t_end = float(times[-1])
    next_sample = 0
    max_steps = 10_000_000
    for _ in range(max_steps):
        if t >= t_end:
            break
        u = np.where(h > dry_tol, m / np.maximum(h, dry_tol), 0.0)
        wave = float((np.abs(u) + np.sqrt(g * np.maximum(h, 0.0))).max())
        dt = min(cfl * dr / wave if wave > 0.0 else dt_max, dt_max, t_end - t)
        if next_sample < times.size:
            dt = min(dt, float(times[next_sample]) - t)

        dh1, dm1 = rhs(h, m, t + 0.5 * dt)
        h1 = np.maximum(h + dt * dh1, 0.0)
        m1 = np.where(h1 > dry_tol, m + dt * dm1, 0.0)
        dh2, dm2 = rhs(h1, m1, t + 0.5 * dt)
        h = np.maximum(h + 0.5 * dt * (dh1 + dh2), 0.0)
        m = np.where(h > dry_tol, m + 0.5 * dt * (dm1 + dm2), 0.0)

        # Manning friction, implicit so it cannot reverse the flow.
        u = np.where(h > dry_tol, m / np.maximum(h, dry_tol), 0.0)
        friction = 1.0 + dt * g * manning_n**2 * np.abs(u) / np.maximum(h, dry_tol) ** (4.0 / 3.0)
        m = np.where(h > dry_tol, m / friction, 0.0)
        t += dt

        newly_wet = (h > wet_tol_m) & ~np.isfinite(arrival)
        arrival[newly_wet] = t
        while next_sample < times.size and t >= float(times[next_sample]) - 1e-9:
            h_out[next_sample] = h
            speed_out[next_sample] = np.abs(np.where(h > dry_tol, m / np.maximum(h, dry_tol), 0.0))
            next_sample += 1
    else:
        raise RuntimeError(f"radial inflow solve did not reach {t_end} s in {max_steps} steps")

    return RadialInflowSolution(
        r_m=r_c, times_s=times, h_m=h_out, speed_ms=speed_out, arrival_s=arrival
    )


# ── EA Test 3: sloping channel, obstruction between two depressions ─────────


def sloping_obstruction_bed(
    cx: NDArray[np.floating[Any]],
    *,
    control_x: Sequence[float],
    control_z: Sequence[float],
    smoothing_m: float = 0.0,
) -> NDArray[np.float32]:
    """Prismatic (y-independent) bed for the EA Test 3 momentum case.

    The bed elevation is the linear interpolation of the ``(control_x,
    control_z)`` control points along ``x`` and is uniform across ``y``. The
    momentum-obstruction harness traces the published Test 3 long profile: a
    reservoir shelf, a uniform approach slope, two cosine depressions
    (Points 1 and 2) separated by a rounded hump (the obstruction), and a
    rise back up to the right boundary. ``control_x`` must
    be strictly increasing and span the mesh's ``x`` range so no cell falls
    outside the interpolation.

    ``smoothing_m`` (Gaussian sigma, in x-units) rounds the piecewise-linear
    slope breaks: the profile is evaluated on a dense internal grid, convolved
    with a truncated Gaussian (edge-padded so the flat end segments are
    preserved), then resampled at ``cx``. ``0.0`` (default) is the exact
    piecewise-linear bed; wide flat regions stay flat in their interior and
    only the corners round.
    """
    xs = np.asarray(control_x, dtype=np.float64)
    zs = np.asarray(control_z, dtype=np.float64)
    if xs.ndim != 1 or xs.shape != zs.shape:
        raise ValueError("control_x and control_z must be 1-D arrays of equal length")
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError("control_x must be strictly increasing")
    cx64 = np.asarray(cx, dtype=np.float64)
    if smoothing_m <= 0.0:
        return np.interp(cx64, xs, zs).astype(np.float32)

    x_lo = float(xs[0])
    x_hi = float(xs[-1])
    step = smoothing_m / 8.0
    n_dense = max(int(np.ceil((x_hi - x_lo) / step)) + 1, 2)
    dense_x: NDArray[np.float64] = np.linspace(x_lo, x_hi, n_dense)
    dense_z: NDArray[np.float64] = np.interp(dense_x, xs, zs)
    sigma_cells = smoothing_m / float(dense_x[1] - dense_x[0])
    radius = int(np.ceil(3.0 * sigma_cells))
    offsets: NDArray[np.float64] = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel: NDArray[np.float64] = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    kernel = kernel / float(kernel.sum())
    padded: NDArray[np.float64] = np.pad(dense_z, radius, mode="edge")
    smoothed: NDArray[np.float64] = np.convolve(padded, kernel, mode="valid")
    return np.interp(cx64, dense_x, smoothed).astype(np.float32)
