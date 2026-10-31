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
from numpy.typing import NDArray

from ..tuning import (
    EDGE_LENGTH_EPSILON,
    FACE_NDIM,
    MIN_CELL_AREA,
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


def _mixed_degree_cells(
    verts: NDArray[np.float32],
    faces: NDArray[np.int32],
    face_offsets: NDArray[np.int32],
    degrees: NDArray[np.int32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Area and centroid per cell, orienting clockwise cells in ``faces`` in place.

    Cells are taken one degree at a time because ``np.sum`` over a row is not a
    sequential sum from 8 terms up, so ``np.bincount`` would round differently
    and change the mesh. A clockwise cell's area keeps its original summation
    order and its centroid uses the reversed one.
    """
    area = np.empty(degrees.size, dtype=np.float32)
    centroid = np.empty((degrees.size, 2), dtype=np.float32)
    for degree in np.unique(degrees).tolist():
        cells = np.flatnonzero(degrees == degree)
        slots = face_offsets[cells, np.newaxis].astype(np.int64) + np.arange(degree)
        cell_faces = faces[slots]
        xy = verts[cell_faces].astype(np.float64)
        x, y = xy[:, :, 0], xy[:, :, 1]
        cross = x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y
        signed_area = 0.5 * np.sum(cross, axis=1)

        clockwise = signed_area < 0
        if clockwise.any():
            cell_faces[clockwise] = cell_faces[clockwise, ::-1]
            faces[slots] = cell_faces
            xy = verts[cell_faces].astype(np.float64)
            x, y = xy[:, :, 0], xy[:, :, 1]
            cross = x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y
            signed_area = np.abs(signed_area)
        area[cells] = np.maximum(signed_area, 1e-30)

        # A zero-area cell divides by zero here and is refused by the sliver
        # check right after.
        with np.errstate(divide="ignore", invalid="ignore"):
            six_area = 6.0 * signed_area
            centroid[cells, 0] = np.sum((x + np.roll(x, -1, axis=1)) * cross, axis=1) / six_area
            centroid[cells, 1] = np.sum((y + np.roll(y, -1, axis=1)) * cross, axis=1) / six_area
    return area, centroid


def _match_half_edges(
    faces: NDArray[np.int32], face_offsets: NDArray[np.int32], n_verts: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Pair up the cells' sides, numbering edges by the first side that reaches them.

    Returns ``(first, second, next_slot)``. ``first`` and ``second`` hold each
    edge's two sides as positions in ``faces``, ``second`` being -1 on a
    boundary; a third side on one edge is ignored. ``next_slot`` gives the
    position of the following vertex of each side's cell.
    """
    offsets = face_offsets.astype(np.int64)
    next_slot = np.arange(1, faces.size + 1, dtype=np.int64)
    next_slot[offsets[1:] - 1] = offsets[:-1]

    va = faces.astype(np.int64)
    vb = va[next_slot]
    key = np.minimum(va, vb) * (n_verts + 1) + np.maximum(va, vb)
    del va, vb
    order = np.argsort(key, kind="stable")
    key = key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    del key

    first = order[starts]
    has_second = np.diff(np.r_[starts, order.size]) >= 2
    second = np.full(starts.size, -1, dtype=np.int64)
    second[has_second] = order[starts[has_second] + 1]

    by_first = np.argsort(first)
    return first[by_first], second[by_first], next_slot


def _edge_lengths_and_normals(
    pa: NDArray[np.float64],
    pb: NDArray[np.float64],
    centroid: NDArray[np.float32],
    cell_l: NDArray[np.int32],
    cell_r: NDArray[np.int32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Edge length and unit normal, pointing from the left to the right cell."""
    dx = pb[:, 0] - pa[:, 0]
    dy = pb[:, 1] - pa[:, 1]
    length = np.sqrt(dx * dx + dy * dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        nx = dy / length
        ny = -dx / length
    point = length < EDGE_LENGTH_EPSILON
    nx[point] = 0.0
    ny[point] = 0.0

    interior = np.flatnonzero(cell_r != -1)
    c = centroid.astype(np.float64)
    c_l = c[cell_l[interior]]
    between = c[cell_r[interior]] - c_l
    inx, iny = nx[interior], ny[interior]
    dot = between[:, 0] * inx + between[:, 1] * iny
    flip = dot < 0
    # Coincident centroids: fall back on the edge midpoint.
    tied = np.flatnonzero(dot == 0.0)
    if tied.size:
        mid = 0.5 * (pa[interior[tied]] + pb[interior[tied]])
        off = (mid[:, 0] - c_l[tied, 0]) * inx[tied] + (mid[:, 1] - c_l[tied, 1]) * iny[tied]
        flip[tied] = off < 0
    flipped = interior[flip]
    nx[flipped] = -nx[flipped]
    ny[flipped] = -ny[flipped]
    return length.astype(np.float32), nx.astype(np.float32), ny.astype(np.float32)


def _dense_from_csr(
    values: NDArray[np.int32], ptr: NDArray[np.int32], max_degree: int
) -> NDArray[np.int32]:
    """``(N, max_degree)`` array of a CSR column, padded with -1."""
    n = ptr.size - 1
    rows = np.repeat(np.arange(n), np.diff(ptr))
    columns = np.arange(values.size) - ptr[rows].astype(np.int64)
    dense = np.full((n, max_degree), -1, dtype=np.int32)
    dense[rows, columns] = values
    return dense


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
        Print coarse per-stage timing to stdout.

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
        area, centroid = _mixed_degree_cells(verts, faces, face_offsets, degrees)
    # A sliver below the CFL kernels' area floor gets no dt bound, so the first
    # drop of water in it divides by its area and the run ends in NaN. Refuse
    # the mesh instead.
    slivers = np.flatnonzero(area <= MIN_CELL_AREA)
    if slivers.size:
        raise ValueError(
            f"{slivers.size} cell(s) have area <= {MIN_CELL_AREA} m^2 "
            f"(first: {slivers[:5].tolist()}); merge or remove degenerate cells"
        )
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
    first, second, next_slot = _match_half_edges(faces, face_offsets, V)
    E = first.size
    _log(f"half-edges matched in {time.perf_counter() - _t_stage:.1f}s (E={E} edges)")

    _t_stage = time.perf_counter()
    slot_cell = np.repeat(np.arange(n, dtype=np.int32), degrees)
    slot_side = (np.arange(faces.size) - face_offsets[slot_cell].astype(np.int64)).astype(np.int32)
    interior = np.flatnonzero(second >= 0)
    edge_cellL = slot_cell[first]
    edge_cellR = np.full(E, -1, dtype=np.int32)
    edge_cellR[interior] = slot_cell[second[interior]]
    edge_sideL = slot_side[first]
    edge_sideR = np.full(E, -1, dtype=np.int32)
    edge_sideR[interior] = slot_side[second[interior]]
    edge_len, edge_nx, edge_ny = _edge_lengths_and_normals(
        verts[faces[first]].astype(np.float64),
        verts[faces[next_slot[first]]].astype(np.float64),
        centroid,
        edge_cellL,
        edge_cellR,
    )
    del first, second, next_slot, slot_cell, slot_side

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
    slot_cells = np.r_[edge_cellL, edge_cellR[interior]]
    total_slots = slot_cells.size
    if total_slots >= 2**31:
        # int32 CSR offsets overflow past this point; fail loudly instead of
        # wrapping cell_edge_ptr into a corrupt CSR.
        msg = f"mesh too large for int32 CSR offsets: total_slots={total_slots}"
        raise ValueError(msg)
    # Within a cell, in the order of its sides: the per-cell gather order.
    slot_sides = np.r_[edge_sideL, edge_sideR[interior]]
    order = np.argsort(slot_cells.astype(np.int64) * max_degree + slot_sides)
    del slot_sides
    cell_edge_ptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(np.bincount(slot_cells, minlength=n), out=cell_edge_ptr[1:])
    del slot_cells
    cell_edge_idx = np.r_[np.arange(E, dtype=np.int32), interior.astype(np.int32)][order]
    cell_nbr_idx = np.r_[edge_cellR, edge_cellL[interior]][order]
    cell_edge_side = np.r_[np.zeros(E, dtype=np.int32), np.ones(interior.size, dtype=np.int32)][
        order
    ]
    del order

    cell_edge_idx_signed = (cell_edge_idx.astype(np.int32) << 1) | cell_edge_side

    # ── Dense arrays (padded to max_degree, backward compat) ──────────────────
    _log(f"CSR adjacency built in {time.perf_counter() - _t_stage:.1f}s (max_degree={max_degree})")
    _t_stage = time.perf_counter()
    cell_edges = _dense_from_csr(cell_edge_idx, cell_edge_ptr, max_degree)
    cell_neighbors = _dense_from_csr(cell_nbr_idx, cell_edge_ptr, max_degree)
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


def hilbert_distances(
    x: NDArray[np.int64], y: NDArray[np.int64], hilbert_p: int
) -> NDArray[np.int64]:
    """Distance along a Hilbert curve of order ``hilbert_p`` for each ``(x, y)``.

    Skilling's transform, the same curve as ``hilbertcurve.HilbertCurve(p, 2)``.
    """
    x, y = x.astype(np.int64), y.astype(np.int64)
    top = 1 << (hilbert_p - 1)

    q = top
    while q > 1:
        low = q - 1
        x = np.where(x & q, x ^ low, x)
        y_set = (y & q) != 0
        swap = (x ^ y) & low
        x, y = np.where(y_set, x ^ low, x ^ swap), np.where(y_set, y, y ^ swap)
        q >>= 1

    y = y ^ x
    gray = np.zeros_like(y)
    q = top
    while q > 1:
        gray ^= np.where(y & q, q - 1, 0)
        q >>= 1
    x, y = x ^ gray, y ^ gray

    distance = np.zeros_like(x)
    for bit in range(hilbert_p):
        distance |= ((x >> bit) & 1) << (2 * bit + 1)
        distance |= ((y >> bit) & 1) << (2 * bit)
    return distance


def hilbert_permutation(centroid: NDArray[np.float32], hilbert_p: int = 10) -> NDArray[np.int32]:
    """Return ``perm[new_i] = old_i`` sorted by Hilbert-curve distance."""
    grid_size = 2**hilbert_p - 1

    cx = centroid[:, 0].astype(np.float64)
    cy = centroid[:, 1].astype(np.float64)
    c_min = np.array([cx.min(), cy.min()])
    c_span = np.array([cx.max() - cx.min(), cy.max() - cy.min()])
    c_span[c_span == 0] = 1.0

    cx_int = ((cx - c_min[0]) / c_span[0] * grid_size).astype(np.int64)
    cy_int = ((cy - c_min[1]) / c_span[1] * grid_size).astype(np.int64)
    distance = hilbert_distances(cx_int, cy_int, hilbert_p)
    return np.argsort(distance, kind="stable").astype(np.int32)


def _reordered_slots(
    ptr: NDArray[np.int32], perm: NDArray[np.int32]
) -> tuple[NDArray[np.int32], NDArray[np.int64]]:
    """New row pointers of a CSR column after ``perm``, and the old slot of each new one."""
    degrees = np.diff(ptr)[perm]
    new_ptr = np.zeros(ptr.size, dtype=np.int32)
    np.cumsum(degrees, out=new_ptr[1:])
    shift = ptr[perm].astype(np.int64) - new_ptr[:-1]
    return new_ptr, np.repeat(shift, degrees) + np.arange(int(new_ptr[-1]))


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

    new_vertex_ptr, vertex_slots = _reordered_slots(geom.cell_vertex_ptr, perm)
    cell_vertices = geom.cell_vertices[vertex_slots]
    del vertex_slots

    # Re-index edge cell references
    edge_cellL_new = inv_perm[geom.edge_cellL].astype(np.int32)
    edge_cellR_new = geom.edge_cellR.copy()
    bdy = geom.edge_cellR != -1
    edge_cellR_new[bdy] = inv_perm[geom.edge_cellR[bdy]]

    # Rebuild CSR in new order
    cell_edge_ptr_new, edge_slots = _reordered_slots(geom.cell_edge_ptr, perm)
    cell_edge_idx_new = geom.cell_edge_idx[edge_slots]
    # is_right is a property of the (cell, edge) pair at this CSR slot,
    # independent of cell/edge renumbering — carry it through unchanged.
    cell_edge_side_new = geom.cell_edge_side[edge_slots]
    old_nbrs = geom.cell_nbr_idx[edge_slots]
    cell_nbr_idx_new = np.where(old_nbrs == -1, -1, inv_perm[old_nbrs])
    del edge_slots, old_nbrs

    # Dense arrays (padded)
    max_degree = geom.max_degree
    cell_edges_new = _dense_from_csr(cell_edge_idx_new, cell_edge_ptr_new, max_degree)
    cell_neighbors_new = _dense_from_csr(cell_nbr_idx_new, cell_edge_ptr_new, max_degree)

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
