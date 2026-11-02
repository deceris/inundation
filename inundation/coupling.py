"""Basins that pass water both ways across shared borders.

Each basin runs on its own mesh and its own time step. At every shared edge a basin
sees its neighbour's level and can only pass water out; what it passed reaches the
neighbour through that edge over the next window. A basin never gives up water it does
not hold, because nothing is taken from it. Border edges must match on both meshes,
which they do when both are cut from one mesh by ``inundation.mesh.split``. See
``docs/implementation/30-two-way-coupling/plan.md``.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from inundation.tuning import BORDER_LEVEL_WEIGHT, OPEN_EXCHANGE, OPEN_INFLOW, OPEN_STAGE
from inundation.workflow import (
    SimulationPhase,
    SWEWorkflow,
    WorkflowConfig,
    _build_phase_open_conditions,  # pyright: ignore[reportPrivateUsage]
    _build_phase_source_rate,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry
    from inundation.mesh.split import BasinLink
    from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch


@dataclass(frozen=True)
class Basin:
    """One mesh and its forcing.

    config: the run, with only this basin's own stretches in ``open_boundaries``.
    phases: its forcing, cut into whole windows, naming only those stretches.
    """

    config: WorkflowConfig
    phases: Sequence[SimulationPhase]


@dataclass
class BasinResult:
    """What one basin did. Arrays are in solver order, as ``workflow.perm`` maps them."""

    workflow: SWEWorkflow
    h_final: NDArray[np.float64]
    snapshots: list[NDArray[np.float32]]
    # Cumulative (out, in) [m^3] through each of the basin's own stretches, by name.
    boundary_volumes_m3: dict[str, NDArray[np.float64]]
    # Volume [m^3] that left through each border edge.
    passed_m3: NDArray[np.float64]
    steps_total: int
    windows_run: int
    clamp_mass_m3: float | None


@dataclass
class CoupledResult:
    """Every basin, and the volumes of them all together."""

    basins: list[BasinResult]
    snap_times: list[float]
    volume_start_m3: float
    volume_in_m3: float
    volume_out_m3: float
    volume_stored_m3: float
    # Left one basin and not yet delivered to another.
    volume_in_transit_m3: float
    wall_seconds: float
    # Of wall_seconds, the time inside the solver's run.
    solver_seconds: float

    @property
    def balance_gap_m3(self) -> float:
        return (
            self.volume_start_m3
            + self.volume_in_m3
            - self.volume_out_m3
            - self.volume_stored_m3
            - self.volume_in_transit_m3
        )

    @property
    def exchanged_m3(self) -> float:
        """Gross volume across all borders, both ways."""
        return float(sum(basin.passed_m3.sum() for basin in self.basins))


def _window_phases(phases: Sequence[SimulationPhase], window_s: float) -> list[SimulationPhase]:
    """The phase of each window."""
    windows: list[SimulationPhase] = []
    for phase in phases:
        count = phase.duration_s / window_s
        if phase.duration_s <= 0.0 or abs(count - round(count)) > 1e-9:
            raise ValueError(
                f"phase of {phase.duration_s} s is not a whole number of {window_s} s windows"
            )
        windows += [phase] * round(count)
    return windows


class _Side:
    """One basin's solver and its view of its borders, one entry per border edge."""

    def __init__(self, basin: Basin, segments: NDArray[np.float64], window_s: float) -> None:
        own = list(basin.config.open_boundaries)
        self.own = own
        self.workflow = SWEWorkflow(dataclasses.replace(basin.config, border_segments=segments))
        self.windows = _window_phases(basin.phases, window_s)
        self.window_s = window_s
        self.workflow.prepare()
        self.solver = cast("SWESolverGpuResidentBatch", self.workflow.solver)
        self.geom = cast("MeshGeometry", self.workflow.geom)
        stretches = self.workflow._open_stretches  # pyright: ignore[reportPrivateUsage]
        self.own_stretches = stretches[: len(own)]
        border = stretches[len(own) :]
        edges = np.concatenate(border) if border else np.empty(0, dtype=np.int64)
        self.cells = self.geom.edge_cellL[edges].astype(np.int64)
        self.bed = self.geom.zb[self.cells].astype(np.float64)
        self.lengths = self.geom.edge_len[edges].astype(np.float64)
        self.neighbour_level = np.zeros(edges.size)
        self.neighbour_bed = np.zeros(edges.size)
        self.waiting = np.zeros(edges.size)
        self.net_out = np.zeros(edges.size)
        self.passed = np.zeros(edges.size)
        self.snapshots: list[NDArray[np.float32]] = []
        self.valid_cell = self.geom.area > self.workflow.config.area_tol
        self.volume_start = self.stored()
        self.ran = False
        self.fresh = True
        self._read_depth()
        self.seen = self.level.copy()
        self.windows_run = 0
        self.solver_seconds = 0.0

    def set_faces(self) -> None:
        """Set each border face at the higher of the two beds."""
        if self.solver.n_open == 0:
            return
        steps = np.zeros(self.solver.n_open, dtype=np.float32)
        steps[steps.size - self.bed.size :] = np.maximum(self.neighbour_bed - self.bed, 0.0)
        self.solver.set_open_steps(steps)

    def stored(self) -> float:
        area = self.geom.area.astype(np.float64)
        return float((self.solver.download_h_precise() * area).sum())

    def _read_depth(self) -> None:
        h = self.solver.download_h()
        if not np.isfinite(h).all():
            raise RuntimeError("a basin's depth went non-finite")
        self.dry = bool(h.max() <= self.workflow.config.dry_tol)
        self.level = h[self.cells].astype(np.float64) + self.bed
        self.fresh = False

    def read_level(self) -> None:
        """Update the level the neighbours see, smoothed over the windows."""
        if self.fresh:
            self._read_depth()
        self.seen = BORDER_LEVEL_WEIGHT * self.level + (1.0 - BORDER_LEVEL_WEIGHT) * self.seen

    def run_window(self, index: int) -> NDArray[np.float64]:
        """Run one window and return the volume that left through each border edge.

        A dry basin that nothing reaches this window is not run.
        """
        phase = self.windows[index]
        config = self.workflow.config
        if self.own:
            kinds, values = _build_phase_open_conditions(
                self.geom,
                self.workflow._manning,  # pyright: ignore[reportPrivateUsage]
                self.own,
                self.own_stretches,
                phase,
            )
        else:
            kinds = np.empty(0, dtype=np.uint32)
            values = np.empty(0, dtype=np.float32)
        brings_water = bool(
            phase.sources or (np.isin(kinds, (OPEN_INFLOW, OPEN_STAGE)) & (values > 0.0)).any()
        )
        # Delivered as float32, so the float64 ledger keeps what the rounding left.
        received = (np.maximum(self.waiting, 0.0) / self.window_s / self.lengths).astype(np.float32)
        if self.dry and not brings_water and not (received > 0.0).any():
            return np.zeros(self.bed.size)

        delivered = received.astype(np.float64) * self.lengths * self.window_s
        self.waiting -= delivered
        if self.solver.n_open:
            depth = np.maximum(self.neighbour_level - self.bed, 0.0)
            self.solver.set_open_conditions(
                np.concatenate([kinds, np.full(self.bed.size, OPEN_EXCHANGE, dtype=np.uint32)]),
                np.concatenate([values, depth.astype(np.float32)]),
                np.concatenate([np.zeros(kinds.size, dtype=np.float32), received]),
            )
        t_start = index * self.window_s
        started = time.perf_counter()
        self.solver.run(
            t_end=t_start + self.window_s,
            output_interval_s=self.window_s,
            dt_max=config.dt_max,
            dt_init=config.dt_init,
            progress=False,
            source_rate=_build_phase_source_rate(self.geom, phase, self.valid_cell),
            resume=self.ran,
            t_start=t_start,
            cfl_interval=config.cfl_interval,
        )
        self.solver_seconds += time.perf_counter() - started
        self.ran = True
        self.fresh = True
        self.windows_run += 1
        if not self.solver.n_open:
            return np.zeros(0)
        self.solver.open_volume_snapshots.clear()
        volumes = self.solver.download_open_volumes()[len(self.own) :]
        net_out = volumes[:, 0] - volumes[:, 1]
        left = net_out - self.net_out + delivered
        self.net_out = net_out
        self.passed += left
        return left

    def result(self) -> BasinResult:
        volumes = self.solver.download_open_volumes() if self.solver.n_open else np.zeros((0, 2))
        return BasinResult(
            workflow=self.workflow,
            h_final=self.solver.download_h_precise(),
            snapshots=self.snapshots,
            boundary_volumes_m3={own.name: volumes[k] for k, own in enumerate(self.own)},
            passed_m3=self.passed,
            steps_total=int(getattr(self.solver, "steps_total", 0)),
            windows_run=self.windows_run,
            clamp_mass_m3=(
                float(np.asarray(self.solver.download_clamp_mass(), dtype=np.float64).sum())
                if self.workflow.config.track_clamp_mass
                else None
            ),
        )


def run_coupled(
    basins: Sequence[Basin],
    links: Sequence[BasinLink],
    window_s: float,
    *,
    output_interval_s: float | None = None,
) -> CoupledResult:
    """Run basins side by side, exchanging over ``links`` every ``window_s``.

    A snapshot is kept every ``output_interval_s``, a multiple of the window.
    """
    if output_interval_s is None:
        output_interval_s = window_s
    every = round(output_interval_s / window_s)
    if every < 1 or abs(every * window_s - output_interval_s) > 1e-9:
        raise ValueError(f"output interval {output_interval_s} s is not a multiple of the window")

    # A link's edges sit at the same place in both basins' lists: (side, slice) twice.
    parts: list[list[NDArray[np.float64]]] = [[] for _ in basins]
    used = [0] * len(basins)
    spans: list[tuple[int, slice, int, slice]] = []
    for link in links:
        size = link.segments.shape[0]
        ends: list[slice] = []
        for side in (link.first, link.second):
            ends.append(slice(used[side], used[side] + size))
            used[side] += size
            parts[side].append(link.segments)
        spans.append((link.first, ends[0], link.second, ends[1]))
    sides = [
        _Side(
            basin,
            np.concatenate(part) if part else np.empty((0, 4), dtype=np.float64),
            window_s,
        )
        for basin, part in zip(basins, parts, strict=True)
    ]
    count = len(sides[0].windows)
    if any(len(side.windows) != count for side in sides):
        raise ValueError("all basins must run for the same time")
    for first, at_first, second, at_second in spans:
        sides[first].neighbour_bed[at_first] = sides[second].bed[at_second]
        sides[second].neighbour_bed[at_second] = sides[first].bed[at_first]
    for side in sides:
        side.set_faces()

    snap_times: list[float] = []
    wall_start = time.perf_counter()
    for index in range(count):
        for side in sides:
            side.read_level()
        for first, at_first, second, at_second in spans:
            sides[first].neighbour_level[at_first] = sides[second].seen[at_second]
            sides[second].neighbour_level[at_second] = sides[first].seen[at_first]
        left = [side.run_window(index) for side in sides]
        for first, at_first, second, at_second in spans:
            sides[second].waiting[at_second] += left[first][at_first]
            sides[first].waiting[at_first] += left[second][at_second]
        if (index + 1) % every == 0:
            snap_times.append((index + 1) * window_s)
            for side in sides:
                side.snapshots.append(side.solver.download_h())
    wall_seconds = time.perf_counter() - wall_start

    results = [side.result() for side in sides]
    volume_in = volume_out = 0.0
    for result, basin in zip(results, basins, strict=True):
        volume_in += sum(
            float(phase.duration_s) * sum(source.discharge_m3s for source in phase.sources)
            for phase in basin.phases
        )
        for volumes in result.boundary_volumes_m3.values():
            volume_out += float(volumes[0])
            volume_in += float(volumes[1])
    return CoupledResult(
        basins=results,
        snap_times=snap_times,
        volume_start_m3=sum(side.volume_start for side in sides),
        volume_in_m3=volume_in,
        volume_out_m3=volume_out,
        volume_stored_m3=sum(side.stored() for side in sides),
        volume_in_transit_m3=float(sum(side.waiting.sum() for side in sides)),
        wall_seconds=wall_seconds,
        solver_seconds=sum(side.solver_seconds for side in sides),
    )
