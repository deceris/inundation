"""Static open-boundary forcing on mesh boundary edges.

Boundary edges are the ones with ``geom.edge_cellR == -1``; their normal
``(edge_nx, edge_ny)`` points out of the domain. Edges not selected by any
``OpenBoundary`` stay reflective walls.

Masks index edges in *solver* order — the geometry after Hilbert reordering,
i.e. ``SWEWorkflow.geom`` once ``prepare()`` has run — because reordering
permutes edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from .mesh.geometry import MeshGeometry

# Shader-side boundary kind codes (flux + CFL kernels). 0 = reflective wall.
BC_WALL = 0
BC_DISCHARGE = 1
BC_DEPTH = 2
_KIND_CODES = {"discharge": BC_DISCHARGE, "depth": BC_DEPTH}


@dataclass(frozen=True)
class OpenBoundary:
    """Constant forcing on a set of boundary edges.

    ``kind="discharge"``: ``value`` is the normal unit discharge
    ``q_n = hu*nx + hv*ny`` [m^2/s] along the *outward* normal, so a negative
    value is inflow into the domain and a positive value is outflow. Each edge
    carries ``q_n * edge_len`` [m^3/s]. The mass flux is imposed exactly
    (outflow is capped at what the boundary cell holds).

    ``kind="depth"``: ``value`` is the exterior water depth [m], measured above
    the adjacent cell's bed. A simple fixed-depth ghost state with the interior
    velocity — not a characteristic (non-reflecting) condition.
    """

    edge_mask: NDArray[np.bool_]
    kind: Literal["discharge", "depth"]
    value: float

    def __post_init__(self) -> None:
        """Reject bad kinds and values; freeze a private copy of the mask."""
        if self.kind not in _KIND_CODES:
            raise ValueError(f"unsupported open-boundary kind: {self.kind!r}")
        if not math.isfinite(self.value):
            raise ValueError(f"open-boundary value must be finite, got {self.value}")
        if self.kind == "depth" and self.value < 0.0:
            raise ValueError(f"prescribed depth must be non-negative, got {self.value}")
        mask = np.asarray(self.edge_mask)
        if mask.dtype != np.bool_ or mask.ndim != 1:
            raise ValueError("edge_mask must be a 1-D boolean array")
        mask = mask.copy()
        mask.setflags(write=False)
        object.__setattr__(self, "edge_mask", mask)


def boundary_table(
    geom: MeshGeometry, open_boundaries: Sequence[OpenBoundary]
) -> NDArray[np.float32]:
    """Validate ``open_boundaries`` against ``geom``; return the (2E,) shader table.

    Interleaved ``[kind_0, value_0, kind_1, value_1, ...]`` per edge; edges not
    selected are walls (kind 0, value 0).
    """
    table = np.zeros((geom.E, 2), dtype=np.float32)
    taken = np.zeros(geom.E, dtype=np.bool_)
    for i, bc in enumerate(open_boundaries):
        mask = bc.edge_mask
        if mask.shape != (geom.E,):
            raise ValueError(f"open boundary {i}: edge_mask shape {mask.shape} != ({geom.E},)")
        if (geom.edge_cellR[mask] != -1).any():
            raise ValueError(f"open boundary {i}: selects interior edges")
        if (taken & mask).any():
            raise ValueError(f"open boundary {i}: overlaps an earlier open boundary")
        taken |= mask
        table[mask] = (_KIND_CODES[bc.kind], bc.value)
    return table.ravel()


__all__ = ["BC_DEPTH", "BC_DISCHARGE", "BC_WALL", "OpenBoundary", "boundary_table"]
