# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportOptionalMemberAccess=false, reportUnknownArgumentType=false
"""Linear reconstruction (needs Vulkan): order, rest, positivity and repeat runs.

See ``docs/implementation/31-muscl/plan.md``. The first-order kernels are the same ones
as before, so what these pin is the option itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from inundation.bench.common import build_channel_mesh
from inundation.bench.open_boundary import Channel, _bumpy, _flat, _write_mesh
from inundation.mesh.geometry import build_geometry
from inundation.mesh.reconstruction import cell_bed_gradient, edge_midpoint_offsets
from inundation.workflow import SimulationPhase, SWEWorkflow, WorkflowConfig

from ._fixtures import requires_vulkan

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

    Bed = Callable[[NDArray[np.float64]], NDArray[np.float64]]

pytestmark = requires_vulkan

LENGTH_M = 1000.0
RUN_S = 40.0
LEVEL_M = 11.2


def _run(
    tmp_path: Path,
    cells: int,
    reconstruction: str,
    bed: Bed = _flat,
    depth: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    *,
    cfl: float | None = 0.4,
    name: str = "mesh",
) -> tuple[NDArray[np.float64], float]:
    """Depth along the channel, in order, and the clamp mass."""
    channel = Channel(LENGTH_M, cells, 2.0 * LENGTH_M / cells, bed, across=2)
    mesh = tmp_path / f"{name}-{cells}-{reconstruction}.parquet"
    _write_mesh(channel, mesh)
    x = channel.centres
    h0 = 1.0 + 0.1 * np.exp(-(((x - 500.0) / 60.0) ** 2)) if depth is None else depth(x)
    zeros = np.zeros(x.size, dtype=np.float32)
    workflow = SWEWorkflow(
        WorkflowConfig(
            mesh_source=str(mesh),
            manning_n=1e-6,
            output_interval_s=RUN_S,
            dt_max=1.0,
            cfl_interval=1,
            progress=False,
            track_clamp_mass=True,
            initial_state_source={"h": h0.astype(np.float32), "hu": zeros, "hv": zeros},
            initial_state_order="original",
            reconstruction=reconstruction,
            cfl=cfl,
        )
    )
    workflow.prepare()
    result = workflow.run([SimulationPhase(RUN_S, [])])
    order = np.argsort(workflow.geom.world_centroid[:, 0], kind="stable")
    h = result.h_final.astype(np.float64)[order]
    clamp = float(np.asarray(workflow.solver.download_clamp_mass(), dtype=np.float64).sum())
    return 0.5 * (h[0::2] + h[1::2]), clamp


@pytest.mark.parametrize(
    ("reconstruction", "low", "high"), [("muscl", 1.7, 2.3), ("first_order", 0.7, 1.3)]
)
def test_a_smooth_wave_converges_at_the_scheme_s_order(
    tmp_path: Path, reconstruction: str, low: float, high: float
) -> None:
    reference = _run(tmp_path, 1600, reconstruction)[0]
    errors = []
    for cells in (100, 200, 400):
        h = _run(tmp_path, cells, reconstruction)[0]
        errors.append(np.abs(h - reference.reshape(cells, -1).mean(axis=1)).mean())
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert low < orders.min()
    assert orders.max() < high


def test_a_lake_stays_at_rest_over_a_bumpy_bed(tmp_path: Path) -> None:
    def depth(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(LEVEL_M - _bumpy(x), 0.0)

    h, clamp = _run(tmp_path, 200, "muscl", _bumpy, depth)
    x = (np.arange(200) + 0.5) * LENGTH_M / 200
    assert np.abs(h + _bumpy(x) - LEVEL_M)[h > 0.01].max() < 1e-5
    assert clamp == 0.0


def test_a_shallow_film_on_a_steep_bed_creates_no_water(tmp_path: Path) -> None:
    """A face far deeper than its cell would ask the cell for more than it holds."""

    def steep(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 10.0 + 2.0 * np.sin(2.0 * np.pi * x / 100.0)

    def film(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(x.shape, 0.05)

    h, clamp = _run(tmp_path, 200, "muscl", steep, film, cfl=None)
    assert clamp == 0.0
    assert h.mean() == pytest.approx(0.05, rel=1e-4)


@pytest.mark.parametrize(("name", "below"), [("wet", 1.0), ("dry", 0.0)])
def test_a_dam_break_stays_within_its_starting_depths(
    tmp_path: Path, name: str, below: float
) -> None:
    """The limiter's job: a shock must not ring. Without it this overshoots by 2 cm."""

    def dam(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(x < 500.0, 2.0, below)

    h, clamp = _run(tmp_path, 400, "muscl", _flat, dam, cfl=None, name=name)
    assert h.max() <= 2.0 + 1e-5
    assert h.min() >= below - 1e-5
    # Depth falls along a dam break; gradients left over from the predictor make it rise.
    assert np.diff(h).max() < 2e-3
    assert clamp == 0.0


def test_repeat_runs_are_bit_identical(tmp_path: Path) -> None:
    runs = [_run(tmp_path, 200, "muscl", name=name)[0] for name in ("first", "second")]
    assert runs[0].tobytes() == runs[1].tobytes()


def test_muscl_needs_the_resident_backend() -> None:
    with pytest.raises(ValueError, match="muscl needs"):
        WorkflowConfig(
            mesh_source="x",
            manning_n=0.03,
            output_interval_s=1.0,
            dt_max=1.0,
            cfl_interval=1,
            solver_impl="baseline",
            reconstruction="muscl",
        )


def test_an_unknown_reconstruction_raises() -> None:
    with pytest.raises(ValueError, match="unsupported reconstruction"):
        WorkflowConfig(
            mesh_source="x",
            manning_n=0.03,
            output_interval_s=1.0,
            dt_max=1.0,
            cfl_interval=1,
            reconstruction="weno",
        )


def test_offsets_reach_the_edge_midpoint_from_both_sides() -> None:
    verts, quads = build_channel_mesh(4, 2, 40.0, 20.0)
    geom = build_geometry(verts, quads, zb_from_file=np.zeros(quads.shape[0], dtype=np.float32))
    offsets = edge_midpoint_offsets(geom)
    centroid = geom.centroid.astype(np.float64)
    interior = geom.edge_cellR >= 0
    left = centroid[geom.edge_cellL[interior]] + offsets[interior, :2]
    right = centroid[geom.edge_cellR[interior]] + offsets[interior, 2:]
    np.testing.assert_allclose(left, right, atol=1e-5)
    np.testing.assert_array_equal(offsets[~interior, 2:], 0.0)


def test_the_bed_gradient_is_exact_on_a_plane_inside_the_mesh() -> None:
    verts, quads = build_channel_mesh(10, 6, 100.0, 60.0)
    geom = build_geometry(verts, quads, zb_from_file=np.zeros(quads.shape[0], dtype=np.float32))
    at = geom.centroid.astype(np.float64)
    geom.zb[:] = (0.02 * at[:, 0] - 0.01 * at[:, 1]).astype(np.float32)
    gradient = cell_bed_gradient(geom)
    x, y = at[:, 0] - at[:, 0].min(), at[:, 1] - at[:, 1].min()
    inside = (x > 5.0) & (x < x.max() - 5.0) & (y > 5.0) & (y < y.max() - 5.0)
    np.testing.assert_allclose(gradient[inside], [[0.02, -0.01]] * int(inside.sum()), atol=1e-4)
