# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Which edges a stretch selects, how inflow splits, and what the workflow refuses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from inundation.bench.open_boundary import gradually_varied_profile, normal_depth
from inundation.mesh.boundaries import (
    boundary_edge_midpoints,
    conveyance_split,
    distance_to_line,
    select_boundary_edges,
)
from inundation.mesh.geometry import build_geometry, hilbert_reorder
from inundation.workflow import (
    FreeOutflow,
    Inflow,
    NormalDepth,
    OpenBoundary,
    PointSource,
    SimulationPhase,
    Stage,
    Wall,
    WorkflowConfig,
    WorkflowResult,
    _build_phase_open_conditions,
    _open_stretch_edges,
    inflow_phases,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry

UTM = (512_345.5, 5_512_345.25)
NX, NY = 6, 4
DX = 2.0


def _grid(offset: tuple[float, float] = UTM) -> MeshGeometry:
    x, y = np.meshgrid(np.arange(NX + 1.0), np.arange(NY + 1.0))
    verts = DX * np.column_stack([x.ravel(), y.ravel()]) + offset
    a = (np.arange(NY)[:, None] * (NX + 1) + np.arange(NX)).ravel()
    faces = np.column_stack([a, a + 1, a + NX + 2, a + NX + 1]).astype(np.int32)
    zb = np.arange(NX * NY, dtype=np.float32) * 0.1
    return hilbert_reorder(build_geometry(verts, faces, zb_from_file=zb), verbose=False)[0]


def _east_line() -> list[tuple[float, float]]:
    x = UTM[0] + NX * DX
    return [(x, UTM[1]), (x, UTM[1] + NY * DX)]


def _config(**overrides: Any) -> WorkflowConfig:
    defaults: dict[str, Any] = {
        "mesh_source": "unused.obj",
        "manning_n": 0.03,
        "output_interval_s": 1.0,
        "dt_max": 0.1,
        "cfl_interval": 1,
    }
    return WorkflowConfig(**{**defaults, **overrides})


def test_a_line_in_world_coordinates_selects_its_edges() -> None:
    geom = _grid()
    edges = select_boundary_edges(geom, _east_line(), 0.5)
    assert edges.size == NY
    assert (geom.edge_cellR[edges] == -1).all()
    assert (geom.edge_len[edges] == DX).all()
    east = geom.centroid[geom.edge_cellL[edges], 0]
    assert (east == east.max()).all()


def test_a_line_follows_each_of_its_segments() -> None:
    geom = _grid()
    x, y = UTM[0] + NX * DX, UTM[1] + NY * DX
    edges = select_boundary_edges(geom, [(x, UTM[1]), (x, y), (UTM[0], y)], 0.5)
    assert edges.size == NX + NY


def test_a_line_away_from_the_rim_selects_nothing() -> None:
    geom = _grid()
    x = UTM[0] + 0.5 * NX * DX
    assert select_boundary_edges(geom, [(x, UTM[1] + 1.0), (x, UTM[1] + 3.0)], 0.5).size == 0


def test_a_stretch_needs_two_points() -> None:
    with pytest.raises(ValueError, match="at least 2 points"):
        select_boundary_edges(_grid(), [(0.0, 0.0)], 1.0)


def test_boundary_normals_point_out_of_the_mesh() -> None:
    geom = _grid()
    edges, midpoints = boundary_edge_midpoints(geom)
    outward = midpoints - geom.centroid[geom.edge_cellL[edges]].astype(np.float64)
    dots = outward[:, 0] * geom.edge_nx[edges] + outward[:, 1] * geom.edge_ny[edges]
    assert (dots > 0.0).all()


def test_distance_to_line_stops_at_the_segment_end() -> None:
    line = np.array([[0.0, 0.0], [10.0, 0.0]])
    points = np.array([[5.0, 3.0], [14.0, 3.0], [-3.0, -4.0]])
    np.testing.assert_allclose(distance_to_line(points, line), [3.0, 5.0, 5.0])


def _section() -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    lengths = np.array([5.0, 4.0, 6.0, 3.0], dtype=np.float32)
    beds = np.array([12.0, 10.0, 10.5, 30.0], dtype=np.float32)
    manning = np.array([0.03, 0.04, 0.03, 0.03], dtype=np.float32)
    return lengths, beds, manning


def test_inflow_splits_to_exactly_the_discharge() -> None:
    lengths, beds, manning = _section()
    q = conveyance_split(7.3, lengths, beds, manning, 1e-3)
    assert float((q.astype(np.float64) * lengths.astype(np.float64)).sum()) == pytest.approx(
        7.3, rel=2e-7
    )


def test_inflow_enters_along_the_low_part_of_the_section() -> None:
    lengths, beds, manning = _section()
    q = conveyance_split(7.3, lengths, beds, manning, 1e-3)
    assert q[3] == 0.0
    assert q[1] > 0.0
    assert q[2] > 0.0


def test_inflow_follows_manning_on_a_flat_section() -> None:
    lengths = np.full(3, 2.0, dtype=np.float32)
    beds = np.full(3, 5.0, dtype=np.float32)
    manning = np.array([0.02, 0.04, 0.08], dtype=np.float32)
    q = conveyance_split(6.0, lengths, beds, manning, 1e-3).astype(np.float64)
    np.testing.assert_allclose(q[0] / q, [1.0, 2.0, 4.0], rtol=1e-5)


def test_no_inflow_splits_to_zero() -> None:
    lengths, beds, manning = _section()
    assert not conveyance_split(0.0, lengths, beds, manning, 1e-3).any()


@pytest.mark.parametrize("backend", ["baseline", "device_cfl", "fixed_dt_batch"])
def test_other_backends_refuse_open_boundaries(backend: str) -> None:
    with pytest.raises(ValueError, match=rf"open_boundaries.*{backend}"):
        _config(solver_impl=backend, open_boundaries=[OpenBoundary("a", _east_line(), 1.0)])


def test_open_boundaries_need_deterministic_accumulation() -> None:
    with pytest.raises(ValueError, match="deterministic"):
        _config(deterministic=False, open_boundaries=[OpenBoundary("a", _east_line(), 1.0)])


def test_stretch_names_must_differ() -> None:
    line = _east_line()
    with pytest.raises(ValueError, match="names repeat"):
        _config(open_boundaries=[OpenBoundary("a", line, 1.0), OpenBoundary("a", line, 1.0)])


def test_a_stretch_selecting_nothing_raises() -> None:
    far = [(0.0, 0.0), (1.0, 0.0)]
    with pytest.raises(ValueError, match="'lost' selects no boundary edges"):
        _open_stretch_edges(_grid(), [OpenBoundary("lost", far, 1.0)])


def test_an_edge_selected_twice_raises() -> None:
    line = _east_line()
    with pytest.raises(ValueError, match="same edge twice"):
        _open_stretch_edges(_grid(), [OpenBoundary("a", line, 1.0), OpenBoundary("b", line, 1.0)])


def _conditions(**boundaries: Any) -> tuple[NDArray[np.uint32], NDArray[np.float32]]:
    geom = _grid()
    stretch = OpenBoundary("a", _east_line(), 1.0)
    edges = _open_stretch_edges(geom, [stretch])
    manning = np.full(geom.N, 0.04, dtype=np.float32)
    phase = SimulationPhase(1.0, [], boundaries)
    return _build_phase_open_conditions(geom, manning, [stretch], edges, phase)


def test_a_phase_must_set_every_stretch() -> None:
    with pytest.raises(ValueError, match="must name every open boundary"):
        _conditions()


def test_a_phase_must_not_name_an_unknown_stretch() -> None:
    with pytest.raises(ValueError, match="must name every open boundary"):
        _conditions(a=Wall(), b=Wall())


@pytest.mark.parametrize("condition", [NormalDepth(0.0), Inflow(1.0, 0.0), Inflow(-1.0, 1e-3)])
def test_bad_condition_values_raise(condition: Any) -> None:
    with pytest.raises(ValueError, match="open boundary 'a'"):
        _conditions(a=condition)


def test_condition_values_reach_the_kernels() -> None:
    kinds, values = _conditions(a=NormalDepth(1e-2))
    assert (kinds == 2).all()
    np.testing.assert_allclose(values, 0.1 / 0.04)
    assert (_conditions(a=FreeOutflow())[0] == 1).all()
    assert (_conditions(a=Wall())[0] == 0).all()


def test_a_stage_gives_depth_over_each_cell_bed_and_never_below_it() -> None:
    geom = _grid()
    cells = geom.edge_cellL[select_boundary_edges(geom, _east_line(), 1.0)]
    level = float(geom.zb[cells].mean())
    values = _conditions(a=Stage(level))[1]
    expected = np.maximum(level - geom.zb[cells].astype(np.float64), 0.0)
    np.testing.assert_allclose(values, expected, atol=1e-6)
    assert (values == 0.0).any()


def _upstream() -> WorkflowResult:
    # Three phases of 10 s with output every 5 s; each phase repeats its start.
    times = [0.0, 5.0, 10.0, 10.0, 15.0, 20.0, 20.0, 25.0, 30.0]
    volumes = np.zeros((len(times), 2))
    volumes[:, 0] = [0.0, 10.0, 30.0, 30.0, 30.0, 40.0, 40.0, 40.0, 100.0]
    empty = np.zeros(1, dtype=np.float32)
    return WorkflowResult(
        snapshots=[],
        snap_times=times,
        h_final=empty,
        volume_final_m3=0.0,
        volume_injected_m3=0.0,
        wall_seconds=0.0,
        steps_total=0,
        boundary_volumes_m3={"outlet": volumes},
    )


def test_inflow_phases_pass_the_volume_of_every_window() -> None:
    phases = inflow_phases(_upstream(), "outlet", "from_a", 1e-3)
    assert [phase.duration_s for phase in phases] == [5.0] * 6
    rates = [phase.boundaries["from_a"].discharge_m3s for phase in phases]  # pyright: ignore[reportAttributeAccessIssue]
    assert rates == [2.0, 4.0, 0.0, 2.0, 0.0, 12.0]
    assert sum(q * phase.duration_s for q, phase in zip(rates, phases, strict=True)) == 100.0


def test_inflow_phases_carry_the_rest_of_the_downstream_run() -> None:
    source = PointSource(1.0, (0.0, 0.0), 1.0)
    phases = inflow_phases(_upstream(), "outlet", "from_a", 1e-3, {"down": FreeOutflow()}, [source])
    assert set(phases[0].boundaries) == {"from_a", "down"}
    assert phases[0].sources == [source]


def test_the_backwater_reference_holds_normal_depth() -> None:
    target = normal_depth(10.0, 40.0, 1e-3, 0.03)
    x = np.linspace(100.0, 2900.0, 8)
    np.testing.assert_allclose(gradually_varied_profile(x, target, 0.25), target, rtol=1e-9)


def test_the_backwater_reference_rises_towards_a_high_level() -> None:
    x = np.linspace(100.0, 2900.0, 8)
    depths = gradually_varied_profile(x, 0.8, 0.25)
    assert (np.diff(depths) > 0.0).all()
