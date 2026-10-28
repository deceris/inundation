"""Cell blocks for the GPU-resident active set.

A block is ``block_cells`` consecutive cells. With Hilbert-ordered cells a block
is a compact patch of mesh, so skipping whole dry blocks skips most of the dry
mesh. These helpers build, on the host, the block topology the device kernels
read: which blocks share an edge, and which blocks each edge workgroup touches.
``remap_cell_edges`` carries the per-cell edge lists through the edge sort.
``dilate_blocks`` is the reference for the device dilation kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def num_blocks(n_cells: int, block_cells: int) -> int:
    """Number of blocks covering ``n_cells`` cells; the last may be partial."""
    if block_cells < 1:
        raise ValueError(f"block_cells must be positive, got {block_cells}")
    return -(-n_cells // block_cells)


def edge_order_by_block(edge_cell_l: NDArray[np.int32], block_cells: int) -> NDArray[np.int64]:
    """Stable permutation of edges sorted by the block of their left cell."""
    return np.argsort(edge_cell_l // block_cells, kind="stable")


def remap_cell_edges(
    cell_edge_idx_signed: NDArray[np.int32], order: NDArray[np.int64]
) -> NDArray[np.uint32]:
    """Per-cell CSR entries ``(edge << 1) | is_right`` after edges are permuted.

    ``order[new] = old`` (see ``edge_order_by_block``). The per-cell order of
    the entries is kept; only the edge ids change.
    """
    new_id = np.empty(order.size, dtype=np.int64)
    new_id[order] = np.arange(order.size, dtype=np.int64)
    signed = np.asarray(cell_edge_idx_signed, dtype=np.int64)
    return ((new_id[signed >> 1] << 1) | (signed & 1)).astype(np.uint32)


def edge_workgroup_block_ranges(
    sorted_cell_l: NDArray[np.int32], workgroup: int, block_cells: int
) -> NDArray[np.uint32]:
    """``(n_workgroups, 2)`` first and last left-cell block of each edge workgroup.

    ``sorted_cell_l`` must be ordered by block (see ``edge_order_by_block``), so
    every block between the two bounds holds a left cell of the workgroup.
    """
    blocks = np.asarray(sorted_cell_l, dtype=np.int64) // block_cells
    if blocks.size and np.any(np.diff(blocks) < 0):
        raise ValueError("edges must be sorted by the block of their left cell")
    starts = np.arange(0, blocks.size, workgroup)
    ends = np.minimum(starts + workgroup, blocks.size) - 1
    return np.column_stack([blocks[starts], blocks[ends]]).astype(np.uint32)


def block_adjacency(
    edge_cell_l: NDArray[np.int32],
    edge_cell_r: NDArray[np.int32],
    n_cells: int,
    block_cells: int,
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """CSR ``(ptr, idx)`` of the other blocks each block shares an edge with.

    Boundary edges (``edge_cell_r < 0``) and edges inside one block add nothing.
    Neighbours of each block are sorted and unique.
    """
    n_blk = num_blocks(n_cells, block_cells)
    interior = edge_cell_r >= 0
    bl = np.asarray(edge_cell_l[interior], dtype=np.int64) // block_cells
    br = np.asarray(edge_cell_r[interior], dtype=np.int64) // block_cells
    cross = bl != br
    pairs = np.unique(
        np.concatenate(
            [np.column_stack([bl[cross], br[cross]]), np.column_stack([br[cross], bl[cross]])]
        ),
        axis=0,
    )
    if pairs.size == 0:
        pairs = np.zeros((0, 2), dtype=np.int64)
    counts = np.bincount(pairs[:, 0], minlength=n_blk)
    ptr = np.zeros(n_blk + 1, dtype=np.uint32)
    np.cumsum(counts, out=ptr[1:])
    return ptr, pairs[:, 1].astype(np.uint32)


def block_two_ring(
    ptr: NDArray[np.uint32], idx: NDArray[np.uint32]
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """CSR of the blocks within two adjacency steps of each block, itself excluded.

    One dilation over this graph equals two dilations over ``(ptr, idx)``.
    """
    n_blk = ptr.size - 1
    deg = np.diff(ptr).astype(np.int64)
    src1 = np.repeat(np.arange(n_blk), deg)
    dst1 = idx.astype(np.int64)
    # Neighbours of neighbours: expand every (b, n) pair by n's neighbour list.
    deg_n = deg[dst1]
    src2 = np.repeat(src1, deg_n)
    starts = ptr[dst1].astype(np.int64)
    offsets = np.arange(int(deg_n.sum())) - np.repeat(np.cumsum(deg_n) - deg_n, deg_n)
    dst2 = idx[np.repeat(starts, deg_n) + offsets].astype(np.int64)
    src = np.concatenate([src1, src2])
    dst = np.concatenate([dst1, dst2])
    keep = src != dst
    pairs = np.unique(np.column_stack([src[keep], dst[keep]]), axis=0)
    if pairs.size == 0:
        pairs = np.zeros((0, 2), dtype=np.int64)
    out_ptr = np.zeros(n_blk + 1, dtype=np.uint32)
    np.cumsum(np.bincount(pairs[:, 0], minlength=n_blk), out=out_ptr[1:])
    return out_ptr, pairs[:, 1].astype(np.uint32)


def dilate_blocks(
    flags: NDArray[np.bool_], ptr: NDArray[np.uint32], idx: NDArray[np.uint32]
) -> NDArray[np.bool_]:
    """One ring of dilation: a block is set if it or any neighbour is set."""
    out = flags.copy()
    src = np.repeat(np.arange(flags.size), np.diff(ptr).astype(np.int64))
    np.logical_or.at(out, src, flags[idx])
    return out


__all__ = [
    "block_adjacency",
    "block_two_ring",
    "dilate_blocks",
    "edge_order_by_block",
    "edge_workgroup_block_ranges",
    "num_blocks",
    "remap_cell_edges",
]
