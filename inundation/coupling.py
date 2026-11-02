"""Two basins that pass water both ways across a shared border.

Each basin runs on its own mesh and its own time step. At every shared edge a basin
sees its neighbour's level and can only pass water out; what it passed reaches the
neighbour through that edge over the next window. A basin never gives up water it does
not hold, because nothing is taken from it. Border edges must match on both meshes,
which they do when both are cut from one mesh. See
``docs/implementation/30-two-way-coupling/plan.md``.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from inundation.tuning import BORDER_LEVEL_WEIGHT, BORDER_TOLERANCE_M, OPEN_EXCHANGE
from inundation.workflow import (
    OpenBoundary,
    SimulationPhase,
    SWEWorkflow,
    Wall,
    WorkflowConfig,
    _build_phase_open_conditions,  # pyright: ignore[reportPrivateUsage]
    _build_phase_source_rate,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry
    from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch

    Point = tuple[float, float]


_BORDER_NAME = "border-{}"


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
    clamp_mass_m3: float | None


@dataclass
class CoupledResult:
    """Both basins, and the volumes of the two together."""

    first: BasinResult
    second: BasinResult
    snap_times: list[float]
    volume_start_m3: float
    volume_in_m3: float
    volume_out_m3: float
    volume_stored_m3: float
    # Left one basin and not yet delivered to the other.
    volume_in_transit_m3: float
    wall_seconds: float

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
        """Gross volume across the border, both ways."""
        return float(self.first.passed_m3.sum() + self.second.passed_m3.sum())


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
    """One basin's solver and its view of the border, edge for edge with the other side."""

    def __init__(self, basin: Basin, border: Sequence[OpenBoundary], window_s: float) -> None:
        own = list(basin.config.open_boundaries)
        self.own_names = [boundary.name for boundary in own]
        self.workflow = SWEWorkflow(
            dataclasses.replace(basin.config, open_boundaries=[*own, *border])
        )
        self.windows = _window_phases(basin.phases, window_s)
        self.workflow.prepare()
        self.solver = cast("SWESolverGpuResidentBatch", self.workflow.solver)
        self.geom = cast("MeshGeometry", self.workflow.geom)
        stretches = self.workflow._open_stretches  # pyright: ignore[reportPrivateUsage]
        sizes = [edges.size for edges in stretches]
        self.border_start = sum(sizes[: len(own)])
        edges = np.concatenate(stretches[len(own) :])
        if edges.size != len(border):
            raise ValueError(f"{len(border)} border segments select {edges.size} edges")
        self.cells = self.geom.edge_cellL[edges].astype(np.int64)
        self.bed = self.geom.zb[self.cells].astype(np.float64)
        self.lengths = self.geom.edge_len[edges].astype(np.float64)
        self.border_names = {boundary.name: Wall() for boundary in border}
        self.seen: NDArray[np.float64] | None = None
        self.waiting = np.zeros(edges.size)
        self.net_out = np.zeros(edges.size)
        self.passed = np.zeros(edges.size)
        self.snapshots: list[NDArray[np.float32]] = []
        self.valid_cell = self.geom.area > self.workflow.config.area_tol
        self.volume_start = self.stored()

    def face_neighbour(self, neighbour: _Side) -> None:
        """Set each border face at the higher of the two beds."""
        steps = np.zeros(self.solver.n_open, dtype=np.float32)
        steps[self.border_start :] = np.maximum(neighbour.bed - self.bed, 0.0)
        self.solver.set_open_steps(steps)

    def stored(self) -> float:
        area = self.geom.area.astype(np.float64)
        return float((self.solver.download_h_precise() * area).sum())

    def read_level(self) -> None:
        """Update the level the neighbour sees, smoothed over the windows."""
        h = self.solver.download_h()
        if not np.isfinite(h).all():
            raise RuntimeError("a basin's depth went non-finite")
        level = h[self.cells].astype(np.float64) + self.bed
        self.seen = (
            level
            if self.seen is None
            else BORDER_LEVEL_WEIGHT * level + (1.0 - BORDER_LEVEL_WEIGHT) * self.seen
        )

    def run_window(
        self, index: int, window_s: float, neighbour_level: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Run one window and return the volume that left through each border edge."""
        phase = self.windows[index]
        config = self.workflow.config
        kinds, values = _build_phase_open_conditions(
            self.geom,
            self.workflow._manning,  # pyright: ignore[reportPrivateUsage]
            config.open_boundaries,
            self.workflow._open_stretches,  # pyright: ignore[reportPrivateUsage]
            dataclasses.replace(phase, boundaries={**phase.boundaries, **self.border_names}),
        )
        at = np.arange(self.cells.size) + self.border_start
        kinds[at] = OPEN_EXCHANGE
        values[at] = np.maximum(neighbour_level - self.bed, 0.0)
        # Delivered as float32, so the float64 ledger keeps what the rounding left.
        received = np.zeros(values.size, dtype=np.float32)
        received[at] = np.maximum(self.waiting, 0.0) / window_s / self.lengths
        delivered = received[at].astype(np.float64) * self.lengths * window_s
        self.waiting -= delivered
        self.solver.set_open_conditions(kinds, values, received)
        t_start = index * window_s
        self.solver.run(
            t_end=t_start + window_s,
            output_interval_s=window_s,
            dt_max=config.dt_max,
            dt_init=config.dt_init,
            progress=False,
            source_rate=_build_phase_source_rate(self.geom, phase, self.valid_cell),
            resume=index > 0,
            t_start=t_start,
            cfl_interval=config.cfl_interval,
        )
        self.solver.open_volume_snapshots.clear()
        volumes = self.solver.download_open_volumes()[len(self.own_names) :]
        net_out = volumes[:, 0] - volumes[:, 1]
        left = net_out - self.net_out + delivered
        self.net_out = net_out
        self.passed += left
        return left

    def result(self) -> BasinResult:
        volumes = self.solver.download_open_volumes()
        return BasinResult(
            workflow=self.workflow,
            h_final=self.solver.download_h_precise(),
            snapshots=self.snapshots,
            boundary_volumes_m3={name: volumes[k] for k, name in enumerate(self.own_names)},
            passed_m3=self.passed,
            steps_total=int(getattr(self.solver, "steps_total", 0)),
            clamp_mass_m3=(
                float(np.asarray(self.solver.download_clamp_mass(), dtype=np.float64).sum())
                if self.workflow.config.track_clamp_mass
                else None
            ),
        )


def run_two_basins(
    first: Basin,
    second: Basin,
    border: Sequence[tuple[Point, Point]],
    window_s: float,
    *,
    output_interval_s: float | None = None,
) -> CoupledResult:
    """Run two basins side by side, exchanging over ``border`` every ``window_s``.

    ``border`` holds the shared edges as vertex pairs in mesh-file coordinates. A
    snapshot is kept every ``output_interval_s``, a multiple of the window.
    """
    if output_interval_s is None:
        output_interval_s = window_s
    every = round(output_interval_s / window_s)
    if every < 1 or abs(every * window_s - output_interval_s) > 1e-9:
        raise ValueError(f"output interval {output_interval_s} s is not a multiple of the window")
    stretches = [
        OpenBoundary(_BORDER_NAME.format(k), [a, b], BORDER_TOLERANCE_M)
        for k, (a, b) in enumerate(border)
    ]
    a = _Side(first, stretches, window_s)
    b = _Side(second, stretches, window_s)
    if len(a.windows) != len(b.windows):
        raise ValueError("both basins must run for the same time")
    a.face_neighbour(b)
    b.face_neighbour(a)

    snap_times: list[float] = []
    wall_start = time.perf_counter()
    for index in range(len(a.windows)):
        a.read_level()
        b.read_level()
        to_b = a.run_window(index, window_s, cast("NDArray[np.float64]", b.seen))
        to_a = b.run_window(index, window_s, cast("NDArray[np.float64]", a.seen))
        b.waiting += to_b
        a.waiting += to_a
        if (index + 1) % every == 0:
            snap_times.append((index + 1) * window_s)
            a.snapshots.append(a.solver.download_h())
            b.snapshots.append(b.solver.download_h())
    wall_seconds = time.perf_counter() - wall_start

    results = (a.result(), b.result())
    volume_in = volume_out = 0.0
    for result, basin in zip(results, (first, second), strict=True):
        volume_in += sum(
            float(phase.duration_s) * sum(source.discharge_m3s for source in phase.sources)
            for phase in basin.phases
        )
        for volumes in result.boundary_volumes_m3.values():
            volume_out += float(volumes[0])
            volume_in += float(volumes[1])
    return CoupledResult(
        first=results[0],
        second=results[1],
        snap_times=snap_times,
        volume_start_m3=a.volume_start + b.volume_start,
        volume_in_m3=volume_in,
        volume_out_m3=volume_out,
        volume_stored_m3=a.stored() + b.stored(),
        volume_in_transit_m3=float(a.waiting.sum() + b.waiting.sum()),
        wall_seconds=wall_seconds,
    )
