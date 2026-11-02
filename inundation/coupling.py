"""Basins that pass water both ways across shared borders.

Each basin runs on its own mesh and its own time step. At every shared edge a basin
sees its neighbour's level and can only pass water out; what it passed reaches the
neighbour through that edge over the next window. A basin never gives up water it does
not hold, because nothing is taken from it. Border edges must match on both meshes,
which they do when both are cut from one mesh by ``inundation.mesh.split``. See
``docs/implementation/30-two-way-coupling/plan.md``.

A step costs about the same however few cells a basin has, so run in sequence, basins
are slower than one mesh. With ``processes`` each basin runs in its own process, which
shares the GPU better than threads can: ``kp`` holds the interpreter lock while it runs.
Basins do not touch each other inside a window, so the result does not change.
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

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
    from collections.abc import Callable, Sequence
    from multiprocessing.connection import Connection

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
    """What one basin did. Arrays are in solver order, ``perm`` maps it to the file's."""

    perm: NDArray[np.int64]
    world_centroid: NDArray[np.float64]
    bed: NDArray[np.float64]
    h_final: NDArray[np.float64]
    hu_final: NDArray[np.float32]
    hv_final: NDArray[np.float32]
    snapshots: list[NDArray[np.float32]]
    # Cumulative (out, in) [m^3] through each of the basin's own stretches, by name.
    boundary_volumes_m3: dict[str, NDArray[np.float64]]
    # Volume [m^3] that left through each border edge.
    passed_m3: NDArray[np.float64]
    volume_start_m3: float
    volume_stored_m3: float
    # Left a neighbour and not yet delivered to this basin.
    volume_waiting_m3: float
    steps_total: int
    windows_run: int
    solver_seconds: float
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
    # The time inside the solver's runs, summed over basins.
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
        self.waiting = np.zeros(edges.size)
        self.net_out = np.zeros(edges.size)
        self.passed = np.zeros(edges.size)
        self.snapshots: list[NDArray[np.float32]] = []
        self.valid_cell = self.geom.area > self.workflow.config.area_tol
        self.volume_start = self.stored()
        self.rate: NDArray[np.float32] | None = None
        self.rate_sources: Sequence[Any] | None = None
        self.ran = False
        self.fresh = True
        self._read_depth()
        self.seen = self.level.copy()
        self.windows_run = 0
        self.solver_seconds = 0.0

    def facts(self) -> tuple[NDArray[np.float64], int]:
        return self.bed, len(self.windows)

    def set_faces(self, neighbour_bed: NDArray[np.float64]) -> None:
        """Set each border face at the higher of the two beds."""
        if self.solver.n_open == 0:
            return
        steps = np.zeros(self.solver.n_open, dtype=np.float32)
        steps[steps.size - self.bed.size :] = np.maximum(neighbour_bed - self.bed, 0.0)
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

    def read_level(self) -> NDArray[np.float64]:
        """Update the level the neighbours see, smoothed over the windows, and return it."""
        if self.fresh:
            self._read_depth()
        self.seen = BORDER_LEVEL_WEIGHT * self.level + (1.0 - BORDER_LEVEL_WEIGHT) * self.seen
        return self.seen

    def source_rate(self, phase: SimulationPhase) -> NDArray[np.float32] | None:
        """The phase's sources per cell, kept while the sources stay the same."""
        if phase.sources != self.rate_sources:
            self.rate = _build_phase_source_rate(self.geom, phase, self.valid_cell)
            self.rate_sources = phase.sources
        return self.rate

    def snapshot(self) -> None:
        self.snapshots.append(self.solver.download_h())

    def run_window(
        self, index: int, neighbour_level: NDArray[np.float64], arriving: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Run one window and return the volume that left through each border edge.

        A dry basin that nothing reaches this window is not run.
        """
        self.waiting += arriving
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
            depth = np.maximum(neighbour_level - self.bed, 0.0)
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
            source_rate=self.source_rate(phase),
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
            perm=np.asarray(self.workflow.perm, dtype=np.int64),
            world_centroid=self.geom.world_centroid.astype(np.float64),
            bed=self.geom.zb.astype(np.float64),
            h_final=self.solver.download_h_precise(),
            hu_final=self.solver.download_hu(),
            hv_final=self.solver.download_hv(),
            snapshots=self.snapshots,
            boundary_volumes_m3={own.name: volumes[k] for k, own in enumerate(self.own)},
            passed_m3=self.passed,
            volume_start_m3=self.volume_start,
            volume_stored_m3=self.stored(),
            volume_waiting_m3=float(self.waiting.sum()),
            steps_total=int(getattr(self.solver, "steps_total", 0)),
            windows_run=self.windows_run,
            solver_seconds=self.solver_seconds,
            clamp_mass_m3=(
                float(np.asarray(self.solver.download_clamp_mass(), dtype=np.float64).sum())
                if self.workflow.config.track_clamp_mass
                else None
            ),
        )


class _Here:
    """A basin in this process."""

    def __init__(self, basin: Basin, segments: NDArray[np.float64], window_s: float) -> None:
        self.side = _Side(basin, segments, window_s)
        self.answer: Any = None

    def post(self, name: str, *args: Any) -> None:
        self.answer = getattr(self.side, name)(*args)

    def take(self) -> Any:
        return self.answer

    def close(self) -> None:
        pass


def _serve(
    connection: Connection, basin: Basin, segments: NDArray[np.float64], window_s: float
) -> None:
    """A basin's process: run what the parent asks until it says stop."""
    try:
        side = _Side(basin, segments, window_s)
        connection.send(None)
        while True:
            name, args = connection.recv()
            if name is None:
                return
            connection.send(getattr(side, name)(*args))
    except Exception as error:  # noqa: BLE001  # handed to the parent, which raises it
        connection.send(error)


class _There:
    """A basin in its own process."""

    def __init__(self, basin: Basin, segments: NDArray[np.float64], window_s: float) -> None:
        context = multiprocessing.get_context("spawn")
        self.connection, theirs = context.Pipe()
        self.process = context.Process(
            target=_serve, args=(theirs, basin, segments, window_s), daemon=True
        )
        self.process.start()
        # Only the child's copy may hold its end open, or a dead child never reads as EOF.
        theirs.close()

    def ready(self) -> None:
        self.take()

    def post(self, name: str, *args: Any) -> None:
        self.connection.send((name, args))

    def take(self) -> Any:
        try:
            answer = self.connection.recv()
        except EOFError as error:
            raise RuntimeError(
                f"a basin's process died with exit code {self.process.exitcode}"
            ) from error
        if isinstance(answer, Exception):
            raise answer
        return answer

    def close(self) -> None:
        if self.process.is_alive():
            self.connection.send((None, ()))
        self.process.join()


def run_coupled(
    basins: Sequence[Basin],
    links: Sequence[BasinLink],
    window_s: float,
    *,
    output_interval_s: float | None = None,
    processes: bool = False,
    on_window: Callable[[int], None] | None = None,
) -> CoupledResult:
    """Run basins side by side, exchanging over ``links`` every ``window_s``.

    A snapshot is kept every ``output_interval_s``, a multiple of the window. With
    ``processes``, each basin runs in its own process. ``on_window`` gets the index of
    each finished window, so a long run can show it is alive.
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
    segments = [
        np.concatenate(part) if part else np.empty((0, 4), dtype=np.float64) for part in parts
    ]

    def ask(name: str, *per_side: Sequence[Any]) -> list[Any]:
        for k, side in enumerate(sides):
            side.post(name, *(argument[k] for argument in per_side))
        return [side.take() for side in sides]

    sides: list[_Here | _There]
    if processes:
        remote = [
            _There(basin, part, window_s) for basin, part in zip(basins, segments, strict=True)
        ]
        for side in remote:
            side.ready()
        sides = list(remote)
    else:
        sides = [_Here(basin, part, window_s) for basin, part in zip(basins, segments, strict=True)]

    try:
        facts = ask("facts")
        beds = [fact[0] for fact in facts]
        count = facts[0][1]
        if any(fact[1] != count for fact in facts):
            raise ValueError("all basins must run for the same time")

        def across(values: Sequence[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
            """For each basin, what its neighbours hold at each of its border edges."""
            out = [np.zeros(bed.size) for bed in beds]
            for first, at_first, second, at_second in spans:
                out[first][at_first] = values[second][at_second]
                out[second][at_second] = values[first][at_first]
            return out

        ask("set_faces", across(beds))
        snap_times: list[float] = []
        arriving = [np.zeros(bed.size) for bed in beds]
        wall_start = time.perf_counter()
        for index in range(count):
            seen = ask("read_level")
            left = ask("run_window", [index] * len(sides), across(seen), arriving)
            arriving = across(left)
            if (index + 1) % every == 0:
                snap_times.append((index + 1) * window_s)
                ask("snapshot")
            if on_window is not None:
                on_window(index)
        wall_seconds = time.perf_counter() - wall_start
        results: list[BasinResult] = ask("result")
    finally:
        for side in sides:
            side.close()

    volume_in = volume_out = 0.0
    for result, basin in zip(results, basins, strict=True):
        volume_in += sum(
            float(phase.duration_s) * sum(source.discharge_m3s for source in phase.sources)
            for phase in basin.phases
        )
        for volumes in result.boundary_volumes_m3.values():
            volume_out += float(volumes[0])
            volume_in += float(volumes[1])
    # What left in the last window has not reached its neighbour's ledger yet.
    undelivered = float(sum(values.sum() for values in arriving))
    return CoupledResult(
        basins=results,
        snap_times=snap_times,
        volume_start_m3=sum(result.volume_start_m3 for result in results),
        volume_in_m3=volume_in,
        volume_out_m3=volume_out,
        volume_stored_m3=sum(result.volume_stored_m3 for result in results),
        volume_in_transit_m3=undelivered + sum(result.volume_waiting_m3 for result in results),
        wall_seconds=wall_seconds,
        solver_seconds=sum(result.solver_seconds for result in results),
    )
