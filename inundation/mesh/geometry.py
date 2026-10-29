"""Mesh geometry and memory-locality preprocessing for the SWE GPU solver.

Supports general polygonal cells (triangles, quads, mixed).
Given vertices and cell connectivity this module computes:
- Cell centroids, areas and bed elevations
- Edge normals, lengths, and left/right cell indices
- CSR adjacency (variable degree per cell)
- Hilbert-curve cell reordering for GPU cache locality

Typical usage
-------------
    from swe_geometry import build_geometry, hilbert_reorder

    geom = build_geometry(verts, faces, face_offsets, zb_from_file=zb)
"""

import sys
import time
from dataclasses import dataclass

import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from numpy.typing import NDArray

from ..tuning import (
    EDGE_LENGTH_EPSILON,
    FACE_NDIM,
    POLYGON_AREA_EPSILON,
    POLYGON_CENTROID_EPSILON,
)

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MeshGeometry:
    """All precomputed geometry arrays needed by the GPU solver.

    Cell arrays (N entries each)
    ----------------------------
    centroid   : (N, 2) float32  - cell centroids
    area       : (N,)   float32  - cell area (shoelace formula)
    zb         : (N,)   float32  - static bed elevation
    manning_n  : (N,)   float32 or None - optional source-cell Manning roughness
    cell_bbox  : (N, 4) float32  - per-cell xmin, xmax, ymin, ymax

    Polygon vertices
    ----------------
    cell_vertices   : (sum_of_degrees, 2) float32 - flattened cell vertices
    cell_vertex_ptr : (N+1,) int32 - row pointers into cell_vertices

    Edge arrays (E entries each)
    ----------------------------
    edge_len   : (E,)   float32  - edge length
    edge_nx   : (E,)   float32  - outward normal x (from left cell)
    edge_ny   : (E,)   float32  - outward normal y
    edge_dzL  : (E,)   float32  - face elevation minus the left cell's bed (>= 0)
    edge_dzR  : (E,)   float32  - face elevation minus the right cell's bed
        (>= 0; both are zero on a boundary edge). The kernels reconstruct the
        face depth as ``h - dz``: computed in float64 from the float32 bed, the
        small step is exact where ``h + zb - z_face`` would lose the low bits
        of a surface elevation hundreds of metres above the datum.
    edge_cellL : (E,)   int32    - left cell index
    edge_cellR : (E,)   int32    - right cell index (-1 = boundary)

    Per-cell adjacency (CSR, variable degree)
    ------------------------------------------
    cell_edge_ptr : (N+1,) int32  - row pointers into cell_edge_idx
    cell_edge_idx : (sum_of_degrees,) int32  - edge indices per cell
    cell_nbr_idx  : (sum_of_degrees,) int32  - neighbour cell per edge slot (-1 = boundary)
    cell_edge_side   : (sum_of_degrees,) int32  - 1 if the gathering cell is
        the edge's right (R) cell for this slot, 0 if it is the left (L) cell
    cell_edge_idx_signed : (sum_of_degrees,) int32  - packed
        ``(edge_idx << 1) | is_right``, precomputed so gather operations can
        determine orientation without another edge-cell lookup

    Dense arrays (padded to max_degree, for backward compat)
    ---------------------------------------------------------
    cell_edges     : (N, max_degree) int32
    cell_neighbors : (N, max_degree) int32

    Mesh counts
    -----------
    N : number of cells
    E : number of edges
    V : number of vertices
    max_degree : maximum number of edges on any cell
    """

    centroid: NDArray[np.float32]
    area: NDArray[np.float32]
    zb: NDArray[np.float32]
    cell_vertices: NDArray[np.float32]
    cell_vertex_ptr: NDArray[np.int32]
    cell_bbox: NDArray[np.float32]
    edge_len: NDArray[np.float32]
    edge_nx: NDArray[np.float32]
    edge_ny: NDArray[np.float32]
    edge_dzL: NDArray[np.float32]
    edge_dzR: NDArray[np.float32]
    edge_cellL: NDArray[np.int32]
    edge_cellR: NDArray[np.int32]
    edge_sideL: NDArray[np.int32]
    edge_sideR: NDArray[np.int32]
    cell_edges: NDArray[np.int32]
    cell_neighbors: NDArray[np.int32]
    cell_edge_ptr: NDArray[np.int32]
    cell_edge_idx: NDArray[np.int32]
    cell_nbr_idx: NDArray[np.int32]
    cell_edge_side: NDArray[np.int32]
    cell_edge_idx_signed: NDArray[np.int32]
    N: int
    E: int
    V: int
    max_degree: int = 3
    manning_n: NDArray[np.float32] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Polygon geometry helpers
# ─────────────────────────────────────────────────────────────────────────────


def _shoelace_area(poly_verts: NDArray[np.float32]) -> float:
    """Signed area of a simple polygon via the shoelace formula (float64)."""
    x = poly_verts[:, 0].astype(np.float64)
    y = poly_verts[:, 1].astype(np.float64)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _polygon_centroid(poly_verts: NDArray[np.float32], signed_area: float) -> NDArray[np.float32]:
    """Centroid of a simple polygon. Falls back to vertex mean for degenerate cells."""
    if abs(signed_area) < POLYGON_AREA_EPSILON:
        # Degenerate (collinear) polygon — use arithmetic mean of vertices
        return poly_verts.mean(axis=0).astype(np.float32)
    x = poly_verts[:, 0].astype(np.float64)
    y = poly_verts[:, 1].astype(np.float64)
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    cx = float(np.sum((x + x1) * cross)) / (6.0 * signed_area)
    cy = float(np.sum((y + y1) * cross)) / (6.0 * signed_area)
    return np.array([cx, cy], dtype=np.float32)


def edge_bed_steps(
    zb: NDArray[np.float32], edge_cellL: NDArray[np.int32], edge_cellR: NDArray[np.int32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Face elevation minus each side's bed, ``(dzL, dzR)``, both ``>= 0``.

    Computed in float64 from the float32 bed, so the result carries the exact
    bed step even when the bed itself sits far above the datum. A boundary
    edge mirrors its left cell: both steps are zero.
    """
    zb64 = zb.astype(np.float64)
    zbL = zb64[edge_cellL]
    zbR = np.where(edge_cellR >= 0, zb64[np.maximum(edge_cellR, 0)], zbL)
    z_face = np.maximum(zbL, zbR)
    return (z_face - zbL).astype(np.float32), (z_face - zbR).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry precomputation
# ─────────────────────────────────────────────────────────────────────────────


def build_geometry(
    verts: NDArray[np.float32],
    faces: NDArray[np.int32],
    face_offsets: NDArray[np.int32] | None = None,
    *,
    zb_from_file: NDArray[np.float32],
    manning_n_from_file: NDArray[np.float32] | None = None,
    progress: bool = False,
) -> MeshGeometry:
    """Precompute all geometric quantities from vertex/face arrays.

    Parameters
    ----------
    verts : (V, 2) float32
        Vertex (x, y) coordinates.
    faces : (N, 3) int32 **or** flat (sum_of_degrees,) int32
        If 2-D with shape (N, d): legacy triangle/polygon input (uniform degree).
        If 1-D: flat vertex indices for all faces (requires face_offsets).
    face_offsets : (N+1,) int32, optional
        Row pointers: face i uses verts faces[face_offsets[i]:face_offsets[i+1]].
        Required when faces is 1-D.
    zb_from_file : (N,) float32
        Required per-cell bed elevation.
    manning_n_from_file : (N,) float32, optional
        Per-cell Manning roughness. Values must be finite and positive.
    progress : bool, optional
        Print coarse per-stage timing to stdout. The edge-enumeration and CSR
        stages are Python loops over every cell and edge, so progress remains
        visible for large production meshes.

    Returns
    -------
    MeshGeometry

    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32, copy=True)  # always copy to avoid mutating caller

    def _log(msg: str) -> None:
        if progress:
            sys.stdout.write(f"[build_geometry] {msg}\n")
            sys.stdout.flush()

    # ── Normalise input to flat + offsets format ──────────────────────────────
    if faces.ndim == FACE_NDIM:
        shape0, shape1 = faces.shape
        n = int(shape0)
        d = int(shape1)
        face_offsets = np.arange(0, (n + 1) * d, d, dtype=np.int32)
        faces = faces.ravel()
    else:
        if face_offsets is None:
            raise ValueError("face_offsets is required when faces is a 1-D array")
        face_offsets = np.asarray(face_offsets, dtype=np.int32)
        n = len(face_offsets) - 1

    V = len(verts)
    _log(f"starting: N={n} cells, V={V} vertices")

    # ── Cell geometry (area, centroid) ────────────────────────────────────────
    _t_stage = time.perf_counter()
    degrees = np.diff(face_offsets)
    max_degree = int(degrees.max())
    min_degree = int(degrees.min())

    # Fast vectorised path for uniform-degree meshes (triangles or quads)
    if min_degree == max_degree:
        d = int(max_degree)
        # Reshape faces into (n, d) for vectorised ops
        face_matrix = faces.reshape(n, d)
        # Gather vertex coordinates: (n, d, 2)
        poly_verts = verts[face_matrix]

        # Vectorised shoelace area (works for any polygon degree)
        x = poly_verts[:, :, 0].astype(np.float64)
        y = poly_verts[:, :, 1].astype(np.float64)
        x_next = np.roll(x, -1, axis=1)
        y_next = np.roll(y, -1, axis=1)
        signed_area = 0.5 * np.sum(x * y_next - x_next * y, axis=1)

        # Fix CW-oriented cells: reverse vertex order
        cw_mask = signed_area < 0
        if cw_mask.any():
            face_matrix[cw_mask] = face_matrix[cw_mask, ::-1]
            # Update flat faces array
            faces[:] = face_matrix.ravel()
            # Recompute with corrected orientation
            poly_verts = verts[face_matrix]
            x = poly_verts[:, :, 0].astype(np.float64)
            y = poly_verts[:, :, 1].astype(np.float64)
            x_next = np.roll(x, -1, axis=1)
            y_next = np.roll(y, -1, axis=1)
            signed_area = 0.5 * np.sum(x * y_next - x_next * y, axis=1)

        area = np.maximum(signed_area, 1e-30).astype(np.float32)

        # Vectorised centroid
        cross = x * y_next - x_next * y
        cx = np.sum((x + x_next) * cross, axis=1) / (6.0 * signed_area)
        cy = np.sum((y + y_next) * cross, axis=1) / (6.0 * signed_area)

        # Handle degenerate cells (zero area → use vertex mean)
        degen = np.abs(signed_area) < POLYGON_CENTROID_EPSILON
        if degen.any():
            cx[degen] = poly_verts[degen, :, 0].mean(axis=1)
            cy[degen] = poly_verts[degen, :, 1].mean(axis=1)

        centroid = np.column_stack([cx, cy]).astype(np.float32)
    else:
        # Fallback: mixed-degree mesh — per-cell Python loop
        centroid = np.empty((n, 2), dtype=np.float32)
        area = np.empty(n, dtype=np.float32)

        for ci in range(n):
            s, e = int(face_offsets[ci]), int(face_offsets[ci + 1])
            cell_verts = verts[faces[s:e]]
            sa = _shoelace_area(cell_verts)
            if sa < 0:
                faces[s:e] = faces[s:e][::-1]
                sa = -sa
                cell_verts = cell_verts[::-1]
            area[ci] = np.float32(max(sa, 1e-30))
            centroid[ci] = _polygon_centroid(cell_verts, sa)
    cell_vertex_ptr = np.asarray(face_offsets, dtype=np.int32).copy()
    cell_vertices = verts[faces].astype(np.float32, copy=True)
    cell_bbox = np.column_stack(
        [
            np.minimum.reduceat(cell_vertices[:, 0], cell_vertex_ptr[:-1]),
            np.maximum.reduceat(cell_vertices[:, 0], cell_vertex_ptr[:-1]),
            np.minimum.reduceat(cell_vertices[:, 1], cell_vertex_ptr[:-1]),
            np.maximum.reduceat(cell_vertices[:, 1], cell_vertex_ptr[:-1]),
        ]
    ).astype(np.float32)

    # ── Bed elevation and optional roughness ──────────────────────────────────
    _log(f"cell area/centroid done in {time.perf_counter() - _t_stage:.1f}s")
    zb = np.asarray(zb_from_file, dtype=np.float32)
    if zb.shape != (n,):
        raise ValueError(f"zb_from_file has shape {zb.shape}, expected ({n},)")
    if not np.isfinite(zb).all():
        raise ValueError("zb_from_file must contain only finite values")

    if manning_n_from_file is None:
        manning_n = None
    else:
        manning_n = np.asarray(manning_n_from_file, dtype=np.float32).reshape(-1)
        if manning_n.size != n:
            raise ValueError(f"manning_n: expected {n} values, got {manning_n.size}")
        if not np.isfinite(manning_n).all() or not (manning_n > 0.0).all():
            raise ValueError("manning_n must contain finite positive values")
    # ── Edge enumeration ──────────────────────────────────────────────────────
    _t_stage = time.perf_counter()
    half_edge_map: dict[tuple[int, int], list[int]] = {}

    for ci in range(n):
        s, e = int(face_offsets[ci]), int(face_offsets[ci + 1])
        d = e - s
        cell_face = faces[s:e]
        for local_side in range(d):
            va = int(cell_face[local_side])
            vb = int(cell_face[(local_side + 1) % d])
            key = (min(va, vb), max(va, vb))
            if key not in half_edge_map:
                half_edge_map[key] = [ci, -1, local_side, -1, va, vb]
            else:
                entry = half_edge_map[key]
                if entry[1] == -1:
                    entry[1] = ci
                    entry[3] = local_side

    edge_records = list(half_edge_map.values())
    E = len(edge_records)
    _log(f"half-edge map built in {time.perf_counter() - _t_stage:.1f}s (E={E} edges)")

    _t_stage = time.perf_counter()
    edge_cellL = np.empty(E, dtype=np.int32)
    edge_cellR = np.empty(E, dtype=np.int32)
    edge_sideL = np.empty(E, dtype=np.int32)
    edge_sideR = np.empty(E, dtype=np.int32)
    edge_len = np.empty(E, dtype=np.float32)
    edge_nx = np.empty(E, dtype=np.float32)
    edge_ny = np.empty(E, dtype=np.float32)

    for eid, rec in enumerate(edge_records):
        cL, cR, sL, sR, va_idx, vb_idx = rec
        pa = verts[va_idx].astype(np.float64)
        pb = verts[vb_idx].astype(np.float64)
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        length = float(np.sqrt(dx * dx + dy * dy))

        if length < EDGE_LENGTH_EPSILON:
            nx, ny = 0.0, 0.0
        else:
            nx = dy / length
            ny = -dx / length

        if cR != -1:
            cL_c = centroid[cR].astype(np.float64) - centroid[cL].astype(np.float64)
            dot = cL_c[0] * nx + cL_c[1] * ny
            if dot < 0:
                nx, ny = -nx, -ny
            elif dot == 0.0:
                mid = 0.5 * (pa + pb)
                alt = (mid[0] - float(centroid[cL][0])) * nx + (
                    mid[1] - float(centroid[cL][1])
                ) * ny
                if alt < 0:
                    nx, ny = -nx, -ny

        edge_cellL[eid] = cL
        edge_cellR[eid] = cR
        edge_sideL[eid] = sL
        edge_sideR[eid] = sR
        edge_len[eid] = np.float32(length)
        edge_nx[eid] = np.float32(nx)
        edge_ny[eid] = np.float32(ny)

    # Sanity: all interior normals point from L toward R  (>= 0 tolerates
    # degenerate cells with coincident centroids)
    _log(f"per-edge length/normal done in {time.perf_counter() - _t_stage:.1f}s")
    imask = edge_cellR != -1
    dL = centroid[edge_cellR[imask]] - centroid[edge_cellL[imask]]
    dots = dL[:, 0] * edge_nx[imask] + dL[:, 1] * edge_ny[imask]
    if not (dots >= 0).all():
        raise ValueError("Some interior edge normals point the wrong way!")
    edge_dzL, edge_dzR = edge_bed_steps(zb, edge_cellL, edge_cellR)

    # ── CSR adjacency (primary representation) ────────────────────────────────
    _t_stage = time.perf_counter()
    cell_edge_lists: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]
    cell_nbr_lists: list[list[tuple[int, int]]] = [[] for _ in range(n)]

    for eid in range(E):
        cL = int(edge_cellL[eid])
        cR = int(edge_cellR[eid])
        sL = int(edge_sideL[eid])

        cell_edge_lists[cL].append((sL, eid, 0))
        cell_nbr_lists[cL].append((sL, cR))
        if cR != -1:
            sR = int(edge_sideR[eid])
            cell_edge_lists[cR].append((sR, eid, 1))
            cell_nbr_lists[cR].append((sR, cL))

    # Sort by local side and flatten
    cell_edge_ptr = np.zeros(n + 1, dtype=np.int32)
    for ci in range(n):
        cell_edge_lists[ci].sort(key=lambda x: x[0])
        cell_nbr_lists[ci].sort(key=lambda x: x[0])
        cell_edge_ptr[ci + 1] = cell_edge_ptr[ci] + len(cell_edge_lists[ci])

    total_slots = int(cell_edge_ptr[n])
    if total_slots >= 2**31 or np.any(np.diff(cell_edge_ptr.astype(np.int64)) < 0):
        # int32 CSR offsets overflow past this point; fail loudly instead of
        # wrapping cell_edge_ptr into a corrupt CSR.
        msg = f"mesh too large for int32 CSR offsets: total_slots={total_slots}"
        raise ValueError(msg)
    cell_edge_idx = np.empty(total_slots, dtype=np.int32)
    cell_nbr_idx = np.empty(total_slots, dtype=np.int32)
    cell_edge_side = np.empty(total_slots, dtype=np.int32)

    for ci in range(n):
        s = int(cell_edge_ptr[ci])
        for j, (_, eid, is_right) in enumerate(cell_edge_lists[ci]):
            cell_edge_idx[s + j] = eid
            cell_edge_side[s + j] = is_right
        for j, (_, nbr) in enumerate(cell_nbr_lists[ci]):
            cell_nbr_idx[s + j] = nbr

    cell_edge_idx_signed = (cell_edge_idx.astype(np.int32) << 1) | cell_edge_side

    # ── Dense arrays (padded to max_degree, backward compat) ──────────────────
    _log(f"CSR adjacency built in {time.perf_counter() - _t_stage:.1f}s (max_degree={max_degree})")
    _t_stage = time.perf_counter()
    cell_edges = np.full((n, max_degree), -1, dtype=np.int32)
    cell_neighbors = np.full((n, max_degree), -1, dtype=np.int32)
    for ci in range(n):
        s, e = int(cell_edge_ptr[ci]), int(cell_edge_ptr[ci + 1])
        d = e - s
        cell_edges[ci, :d] = cell_edge_idx[s:e]
        cell_neighbors[ci, :d] = cell_nbr_idx[s:e]
    _log(f"dense arrays built in {time.perf_counter() - _t_stage:.1f}s — build_geometry done")

    return MeshGeometry(
        centroid=centroid,
        area=area,
        zb=zb,
        manning_n=manning_n,
        cell_vertices=cell_vertices,
        cell_vertex_ptr=cell_vertex_ptr,
        cell_bbox=cell_bbox,
        edge_len=edge_len,
        edge_nx=edge_nx,
        edge_ny=edge_ny,
        edge_dzL=edge_dzL,
        edge_dzR=edge_dzR,
        edge_cellL=edge_cellL,
        edge_cellR=edge_cellR,
        edge_sideL=edge_sideL,
        edge_sideR=edge_sideR,
        cell_edges=cell_edges,
        cell_neighbors=cell_neighbors,
        cell_edge_ptr=cell_edge_ptr,
        cell_edge_idx=cell_edge_idx,
        cell_nbr_idx=cell_nbr_idx,
        cell_edge_side=cell_edge_side,
        cell_edge_idx_signed=cell_edge_idx_signed,
        N=n,
        E=E,
        V=V,
        max_degree=max_degree,
    )


def hilbert_permutation(centroid: NDArray[np.float32], hilbert_p: int = 10) -> NDArray[np.int32]:
    """Return ``perm[new_i] = old_i`` sorted by Hilbert-curve distance."""
    n_cells = centroid.shape[0]
    grid_size = 2**hilbert_p - 1

    hc = HilbertCurve(p=hilbert_p, n=2)

    cx = centroid[:, 0].astype(np.float64)
    cy = centroid[:, 1].astype(np.float64)
    c_min = np.array([cx.min(), cy.min()])
    c_span = np.array([cx.max() - cx.min(), cy.max() - cy.min()])
    c_span[c_span == 0] = 1.0

    cx_int = ((cx - c_min[0]) / c_span[0] * grid_size).astype(int)
    cy_int = ((cy - c_min[1]) / c_span[1] * grid_size).astype(int)
    points_2d = [[int(cx_int[i]), int(cy_int[i])] for i in range(n_cells)]
    hilbert_d = np.array(hc.distances_from_points(points_2d), dtype=np.int64)
    return np.argsort(hilbert_d, kind="stable").astype(np.int32)


def hilbert_reorder(
    geom: MeshGeometry,
    hilbert_p: int = 10,
    verbose: bool = True,
) -> tuple[MeshGeometry, NDArray[np.int32]]:
    """Reorder cells by Hilbert-curve index for GPU memory locality.

    Parameters
    ----------
    geom : MeshGeometry
        Pre-computed geometry (output of :func:`build_geometry`).
    hilbert_p : int
        Hilbert curve order; grid resolution = 2**hilbert_p per axis.
    verbose : bool
        Print timing and verification messages.

    Returns
    -------
    geom_reordered : MeshGeometry
        Same structure with all cell/edge arrays reordered.
    perm : (N,) int32
        Permutation array: ``perm[new_i] = old_i``.

    """
    N = geom.N
    _t_stage = time.perf_counter()

    def _log(msg: str) -> None:
        if verbose:
            sys.stdout.write(f"[hilbert_reorder] {msg}\n")
            sys.stdout.flush()

    _log(f"starting: N={N} cells")
    perm = hilbert_permutation(geom.centroid, hilbert_p=hilbert_p)
    _log(f"hilbert_permutation done in {time.perf_counter() - _t_stage:.1f}s")

    inv_perm = np.empty(N, dtype=np.int32)
    inv_perm[perm] = np.arange(N, dtype=np.int32)

    # Reorder cell arrays
    _t_stage = time.perf_counter()
    centroid = geom.centroid[perm]
    area = geom.area[perm]
    zb = geom.zb[perm]
    manning_n = None if geom.manning_n is None else geom.manning_n[perm]
    cell_bbox = geom.cell_bbox[perm]

    old_vertex_degrees = np.diff(geom.cell_vertex_ptr)
    new_vertex_ptr = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(old_vertex_degrees[perm], out=new_vertex_ptr[1:])
    cell_vertices = np.empty_like(geom.cell_vertices)
    for new_i in range(N):
        old_i = int(perm[new_i])
        old_s = int(geom.cell_vertex_ptr[old_i])
        old_e = int(geom.cell_vertex_ptr[old_i + 1])
        new_s = int(new_vertex_ptr[new_i])
        cell_vertices[new_s : new_s + old_e - old_s] = geom.cell_vertices[old_s:old_e]

    # Re-index edge cell references
    edge_cellL_new = inv_perm[geom.edge_cellL].astype(np.int32)
    edge_cellR_new = geom.edge_cellR.copy()
    bdy = geom.edge_cellR != -1
    edge_cellR_new[bdy] = inv_perm[geom.edge_cellR[bdy]]

    # Rebuild CSR in new order
    old_degrees = np.diff(geom.cell_edge_ptr)
    new_degrees = old_degrees[perm]
    cell_edge_ptr_new = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(new_degrees, out=cell_edge_ptr_new[1:])

    total_slots = int(cell_edge_ptr_new[N])
    cell_edge_idx_new = np.empty(total_slots, dtype=np.int32)
    cell_nbr_idx_new = np.empty(total_slots, dtype=np.int32)
    cell_edge_side_new = np.empty(total_slots, dtype=np.int32)

    for new_i in range(N):
        old_i = int(perm[new_i])
        old_s = int(geom.cell_edge_ptr[old_i])
        old_e = int(geom.cell_edge_ptr[old_i + 1])
        new_s = int(cell_edge_ptr_new[new_i])
        d = old_e - old_s

        cell_edge_idx_new[new_s : new_s + d] = geom.cell_edge_idx[old_s:old_e]
        # is_right is a property of the (cell, edge) pair at this CSR slot,
        # independent of cell/edge renumbering — carry it through unchanged.
        cell_edge_side_new[new_s : new_s + d] = geom.cell_edge_side[old_s:old_e]

        old_nbrs = geom.cell_nbr_idx[old_s:old_e]
        new_nbrs = np.where(old_nbrs == -1, -1, inv_perm[old_nbrs])
        cell_nbr_idx_new[new_s : new_s + d] = new_nbrs

    # Dense arrays (padded)
    max_degree = geom.max_degree
    cell_edges_new = np.full((N, max_degree), -1, dtype=np.int32)
    cell_neighbors_new = np.full((N, max_degree), -1, dtype=np.int32)
    for ci in range(N):
        s = int(cell_edge_ptr_new[ci])
        e = int(cell_edge_ptr_new[ci + 1])
        d = e - s
        cell_edges_new[ci, :d] = cell_edge_idx_new[s:e]
        cell_neighbors_new[ci, :d] = cell_nbr_idx_new[s:e]

    # Verify normal orientation (>= 0 tolerates degenerate zero-area cells)
    imask = edge_cellR_new != -1
    dL2 = centroid[edge_cellR_new[imask]] - centroid[edge_cellL_new[imask]]
    dots2 = dL2[:, 0] * geom.edge_nx[imask] + dL2[:, 1] * geom.edge_ny[imask]
    if not (dots2 >= 0).all():
        raise ValueError("Normal orientation broken after Hilbert reordering!")
    _log(f"CSR/dense rebuild + orientation check done in {time.perf_counter() - _t_stage:.1f}s")

    # ── Sort edges by min(cellL, cellR) for GPU memory locality ───────────────
    # Threads in the same warp process consecutive edges; if those edges reference
    # cells that are nearby in memory (Hilbert-ordered), L2 cache hit rates improve.
    _t_stage = time.perf_counter()
    E = geom.E
    edge_sort_key = np.where(
        edge_cellR_new >= 0,
        np.minimum(edge_cellL_new, edge_cellR_new),
        edge_cellL_new,
    )
    edge_perm = np.argsort(edge_sort_key, kind="stable").astype(np.int32)

    edge_len_sorted = geom.edge_len[edge_perm]
    edge_nx_sorted = geom.edge_nx[edge_perm]
    edge_ny_sorted = geom.edge_ny[edge_perm]
    edge_dzL_sorted = geom.edge_dzL[edge_perm]
    edge_dzR_sorted = geom.edge_dzR[edge_perm]
    edge_cellL_sorted = edge_cellL_new[edge_perm]
    edge_cellR_sorted = edge_cellR_new[edge_perm]
    edge_sideL_sorted = geom.edge_sideL[edge_perm]
    edge_sideR_sorted = geom.edge_sideR[edge_perm]

    # Rebuild edge inverse permutation for CSR edge indices
    edge_inv_perm = np.empty(E, dtype=np.int32)
    edge_inv_perm[edge_perm] = np.arange(E, dtype=np.int32)

    # Update CSR and dense edge indices to use new edge numbering
    cell_edge_idx_sorted = edge_inv_perm[cell_edge_idx_new]
    cell_edge_idx_signed_sorted = (cell_edge_idx_sorted.astype(np.int32) << 1) | cell_edge_side_new
    cell_edges_sorted = np.where(
        cell_edges_new >= 0,
        edge_inv_perm[np.clip(cell_edges_new, 0, E - 1)],
        -1,
    )
    # Restore -1 entries that were clipped
    cell_edges_sorted[cell_edges_new == -1] = -1

    _log(
        f"edge sort by locality done in {time.perf_counter() - _t_stage:.1f}s — "
        "hilbert_reorder done"
    )

    reordered = MeshGeometry(
        centroid=centroid,
        area=area,
        zb=zb,
        manning_n=manning_n,
        cell_vertices=cell_vertices,
        cell_vertex_ptr=new_vertex_ptr,
        cell_bbox=cell_bbox,
        edge_len=edge_len_sorted,
        edge_nx=edge_nx_sorted,
        edge_ny=edge_ny_sorted,
        edge_dzL=edge_dzL_sorted,
        edge_dzR=edge_dzR_sorted,
        edge_cellL=edge_cellL_sorted,
        edge_cellR=edge_cellR_sorted,
        edge_sideL=edge_sideL_sorted,
        edge_sideR=edge_sideR_sorted,
        cell_edges=cell_edges_sorted,
        cell_neighbors=cell_neighbors_new,
        cell_edge_ptr=cell_edge_ptr_new,
        cell_edge_idx=cell_edge_idx_sorted,
        cell_nbr_idx=cell_nbr_idx_new,
        cell_edge_side=cell_edge_side_new,
        cell_edge_idx_signed=cell_edge_idx_signed_sorted,
        N=N,
        E=E,
        V=geom.V,
        max_degree=max_degree,
    )
    return reordered, perm
