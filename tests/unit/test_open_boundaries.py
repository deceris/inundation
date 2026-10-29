"""Open-boundary forcing: spec validation (CPU) and solver smoke tests (GPU)."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pytest

from inundation.bench.common import build_channel_mesh
from inundation.boundary import OpenBoundary, boundary_table
from inundation.mesh.geometry import build_geometry, hilbert_reorder

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry
    from inundation.vulkan.solver import SWESolver

NX, NY, LENGTH, WIDTH = 20, 4, 20.0, 4.0  # 1 m cells
N_FRICTIONLESS = 1e-4


def _channel() -> MeshGeometry:
    verts, quads = build_channel_mesh(NX, NY, LENGTH, WIDTH)
    geom = build_geometry(verts, quads, zb_from_file=np.zeros(NX * NY, dtype=np.float32))
    return hilbert_reorder(geom, verbose=False)[0]


def _side(geom: MeshGeometry, nx: float) -> NDArray[np.bool_]:
    """Boundary edges whose outward normal has x-component ``nx`` (-1 west, +1 east)."""
    return (geom.edge_cellR == -1) & (np.abs(geom.edge_nx - nx) < 0.5)


# ── Pure validation ──────────────────────────────────────────────────────────


class TestValidation:
    def test_unselected_boundary_edges_stay_walls(self) -> None:
        geom = _channel()
        table = boundary_table(geom, [OpenBoundary(_side(geom, -1.0), "discharge", -0.5)])
        kind = table.reshape(-1, 2)[:, 0]
        assert (kind[_side(geom, -1.0)] == 1).all()
        assert (kind[~_side(geom, -1.0)] == 0).all()

    def test_rejects_interior_edges(self) -> None:
        geom = _channel()
        with pytest.raises(ValueError, match="interior"):
            boundary_table(geom, [OpenBoundary(geom.edge_cellR != -1, "depth", 1.0)])

    def test_rejects_overlap(self) -> None:
        geom = _channel()
        west = _side(geom, -1.0)
        with pytest.raises(ValueError, match="overlap"):
            boundary_table(
                geom, [OpenBoundary(west, "depth", 1.0), OpenBoundary(west, "discharge", -1.0)]
            )

    def test_rejects_wrong_length(self) -> None:
        geom = _channel()
        with pytest.raises(ValueError, match="shape"):
            boundary_table(geom, [OpenBoundary(np.zeros(geom.E - 1, dtype=bool), "depth", 1.0)])

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_rejects_non_finite(self, value: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            OpenBoundary(np.zeros(3, dtype=bool), "discharge", value)

    def test_rejects_negative_depth(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            OpenBoundary(np.zeros(3, dtype=bool), "depth", -0.1)

    def test_rejects_non_bool_mask(self) -> None:
        with pytest.raises(ValueError, match="boolean"):
            OpenBoundary(np.zeros(3, dtype=np.int32), "depth", 1.0)  # pyright: ignore[reportArgumentType]

    def test_mask_is_frozen_copy(self) -> None:
        mask = np.zeros(3, dtype=bool)
        bc = OpenBoundary(mask, "depth", 1.0)
        mask[0] = True
        assert not bc.edge_mask[0]
        assert not bc.edge_mask.flags.writeable


# ── Solver smoke tests (need kp + a Vulkan device) ───────────────────────────


@cache
def _spv() -> dict[str, bytes] | None:
    try:
        import kp

        kp.Manager()
        from inundation.workflow import (
            _compile_solver_shaders,  # pyright: ignore[reportPrivateUsage]
        )

        return _compile_solver_shaders()
    except Exception:
        return None


requires_gpu = pytest.mark.skipif(
    _spv() is None, reason="needs kp, glslangValidator, and a Vulkan device"
)

BACKENDS = ["baseline", "gpu_resident_batch"]  # one per flux shader variant


def _solver(backend: str, geom: MeshGeometry, h0: float) -> SWESolver:
    from inundation.vulkan.baseline import SWESolverBaseline
    from inundation.vulkan.gpu_resident_batch import SWESolverGpuResidentBatch

    cls = {"baseline": SWESolverBaseline, "gpu_resident_batch": SWESolverGpuResidentBatch}[backend]
    spv = _spv()
    assert spv is not None
    z = np.zeros(geom.N, dtype=np.float32)
    n0 = np.full(geom.N, N_FRICTIONLESS, dtype=np.float32)
    return cls(geom, spv, np.full(geom.N, h0, dtype=np.float32), z, z, n0)


def _run(s: SWESolver, t_end: float) -> None:
    s.run(  # pyright: ignore[reportUnknownMemberType]
        t_end=t_end,
        output_interval_s=t_end,
        dt_max=0.05,
        dt_init=1e-3,
        cfl_interval=1,
        progress=False,
    )


def _volume(s: SWESolver, geom: MeshGeometry) -> float:
    return float((s.download_h().astype(np.float64) * geom.area).sum())


@requires_gpu
@pytest.mark.parametrize("backend", BACKENDS)
class TestSolver:
    def test_walls_only_matches_no_open_boundaries(self, backend: str) -> None:
        """An explicit all-wall table and the default are the same run, bitwise."""
        geom = _channel()
        a = _solver(backend, geom, 0.5)
        b = _solver(backend, geom, 0.5)
        b.set_open_boundaries([])
        _run(a, 2.0)
        _run(b, 2.0)
        assert np.array_equal(a.download_h(), b.download_h())
        assert np.allclose(a.download_h(), 0.5, atol=1e-6)  # lake at rest
        assert b.boundary_volume_out_m3() == 0.0

    def test_inflow_fills_dry_channel_without_sources(self, backend: str) -> None:
        geom = _channel()
        s = _solver(backend, geom, 0.0)
        q = -0.2  # m^2/s, inflow
        s.set_open_boundaries([OpenBoundary(_side(geom, -1.0), "discharge", q)])
        _run(s, 5.0)  # no source_rate passed: the source kernel never runs
        expected_in = -q * WIDTH * 5.0
        assert abs(_volume(s, geom) - expected_in) < 1e-3 * expected_in
        assert abs(s.boundary_volume_out_m3() + expected_in) < 1e-3 * expected_in
        h = s.download_h()
        assert (h[geom.centroid[:, 0] < 1.0] > 0.0).all()
        assert float((s.download_hu() * geom.area).sum()) > 0.0  # flowing in +x

    def test_prescribed_depth_drives_boundary_cell(self, backend: str) -> None:
        geom = _channel()
        s = _solver(backend, geom, 0.5)
        s.set_open_boundaries([OpenBoundary(_side(geom, 1.0), "depth", 1.0)])
        _run(s, 2.0)
        h = s.download_h()
        east = h[geom.centroid[:, 0] > LENGTH - 1.0]
        assert (east > 0.6).all()
        assert _volume(s, geom) > 0.5 * LENGTH * WIDTH

    def test_mass_balance_inflow_outflow(self, backend: str) -> None:
        geom = _channel()
        s = _solver(backend, geom, 0.5)
        s.set_open_boundaries(
            [
                OpenBoundary(_side(geom, -1.0), "discharge", -0.3),
                OpenBoundary(_side(geom, 1.0), "depth", 0.5),
            ]
        )
        v0 = _volume(s, geom)
        _run(s, 10.0)
        dv = _volume(s, geom) - v0
        assert abs(dv + s.boundary_volume_out_m3()) < 1e-4 * v0

    def test_discharge_sign_sets_direction(self, backend: str) -> None:
        geom = _channel()
        inflow = _solver(backend, geom, 0.5)
        outflow = _solver(backend, geom, 0.5)
        west = _side(geom, -1.0)
        inflow.set_open_boundaries([OpenBoundary(west, "discharge", -0.1)])
        outflow.set_open_boundaries([OpenBoundary(west, "discharge", 0.1)])
        _run(inflow, 1.0)
        _run(outflow, 1.0)
        assert float(inflow.download_hu().sum()) > 0.0
        assert float(outflow.download_hu().sum()) < 0.0
        assert inflow.boundary_volume_out_m3() < 0.0 < outflow.boundary_volume_out_m3()

    def test_outflow_never_overdraws(self, backend: str) -> None:
        """Outflow larger than the cell holds is capped: no clamp-created mass."""
        geom = _channel()
        s = _solver(backend, geom, 0.01)
        s.track_clamp = True
        s.set_open_boundaries([OpenBoundary(_side(geom, -1.0), "discharge", 5.0)])
        v0 = _volume(s, geom)
        _run(s, 2.0)
        assert float(s.download_clamp_mass().sum()) < 1e-6
        assert abs(_volume(s, geom) - v0 + s.boundary_volume_out_m3()) < 1e-5
        assert s.boundary_volume_out_m3() < v0  # cannot drain more than was there
