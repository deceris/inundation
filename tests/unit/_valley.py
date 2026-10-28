"""Sloped-valley test problem for the GPU-resident solver tests.

A Hilbert-ordered quad grid (128 active-set blocks) whose bed falls in +x, with
a shallow valley along the centre line. Large enough that most blocks are
skipped early on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry

NX, NY, DX = 256, 128, 2.0
MANNING = 0.03
DRY_TOL = 1e-3


def valley_grid() -> MeshGeometry:
    from inundation.mesh.geometry import build_geometry, hilbert_reorder

    xs, ys = np.meshgrid(np.arange(NX + 1) * DX, np.arange(NY + 1) * DX)
    verts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)

    def vid(i: int, j: int) -> int:
        return j * (NX + 1) + i

    faces = np.array(
        [
            [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            for j in range(NY)
            for i in range(NX)
        ],
        dtype=np.int32,
    )
    cx = (np.arange(NX * NY) % NX + 0.5) * DX
    cy = (np.arange(NX * NY) // NX + 0.5) * DX
    zb = (-0.002 * cx - 0.3 * np.exp(-(((cy - NY * DX / 2) / (0.2 * NY * DX)) ** 2))).astype(
        np.float32
    )
    geom = build_geometry(verts, faces, zb_from_file=zb)
    return hilbert_reorder(geom, verbose=False)[0]


def cell_near(grid: MeshGeometry, x: float, y: float) -> int:
    d = (grid.centroid[:, 0] - x) ** 2 + (grid.centroid[:, 1] - y) ** 2
    return int(np.argmin(d))


def point_source(grid: MeshGeometry, q: float, x: float, y: float) -> NDArray[np.float32]:
    src = np.zeros(grid.N, dtype=np.float32)
    src[cell_near(grid, x, y)] = q
    return src


def dam(grid: MeshGeometry) -> NDArray[np.float32]:
    """A 1 m column of water in the upstream quarter of the valley."""
    x, y = grid.centroid[:, 0], grid.centroid[:, 1]
    inside = (x < NX * DX * 0.25) & (np.abs(y - NY * DX / 2) < NY * DX * 0.15)
    return np.where(inside, 1.0, 0.0).astype(np.float32)


def sheet(grid: MeshGeometry) -> NDArray[np.float32]:
    """A thin sheet on the upstream slope: it runs off and dries cells behind it."""
    return np.where(grid.centroid[:, 0] < NX * DX * 0.3, 0.02, 0.0).astype(np.float32)


def run(solver: Any, t_end: float, **kwargs: Any) -> list[NDArray[np.float32]]:
    snaps, _ = solver.run(
        t_end=t_end,
        output_interval_s=kwargs.pop("output_interval_s", t_end / 4),
        dt_max=1.0,
        dt_init=0.01,
        cfl_interval=10,
        progress=False,
        **kwargs,
    )
    return snaps


def state(solver: Any) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    return solver.download_h(), solver.download_hu(), solver.download_hv()
