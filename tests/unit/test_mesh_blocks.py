"""Block topology for the GPU-resident active set.

The device skips every block outside the two-ring of marked blocks. That is
exact only if every cell within two cell rings of a wet cell lies in that
block two-ring, and if each edge workgroup's block range covers its edges.
These tests check the host helpers that build that topology, and the halo
claim itself on a Hilbert-ordered mesh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from inundation.mesh.blocks import (
    block_adjacency,
    block_two_ring,
    dilate_blocks,
    edge_order_by_block,
    edge_workgroup_block_ranges,
    num_blocks,
    remap_cell_edges,
)
from inundation.mesh.geometry import MeshGeometry, build_geometry, hilbert_reorder
from inundation.tuning import ACTIVE_BLOCK_CELLS

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _grid(nx: int, ny: int) -> MeshGeometry:
    xs, ys = np.meshgrid(np.arange(nx + 1, dtype=np.float32), np.arange(ny + 1, dtype=np.float32))
    verts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    faces = np.array(
        [
            [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            for j in range(ny)
            for i in range(nx)
        ],
        dtype=np.int32,
    )
    geom = build_geometry(verts, faces, zb_from_file=np.zeros(nx * ny, dtype=np.float32))
    return hilbert_reorder(geom, verbose=False)[0]


@pytest.fixture(scope="module")
def grid() -> MeshGeometry:
    return _grid(64, 48)


def _cell_dilate(wet: NDArray[np.bool_], geom: MeshGeometry) -> NDArray[np.bool_]:
    """One ring of cell dilation through interior edges."""
    out = wet.copy()
    interior = geom.edge_cellR >= 0
    left, right = geom.edge_cellL[interior], geom.edge_cellR[interior]
    out[left] |= wet[right]
    out[right] |= wet[left]
    return out


class TestNumBlocks:
    @pytest.mark.parametrize(("cells", "expected"), [(0, 0), (1, 1), (256, 1), (257, 2), (512, 2)])
    def test_rounds_up(self, cells: int, expected: int) -> None:
        assert num_blocks(cells, 256) == expected

    def test_rejects_non_positive_block_size(self) -> None:
        with pytest.raises(ValueError, match="block_cells"):
            num_blocks(10, 0)


class TestEdgeOrder:
    def test_sorts_by_left_block_and_keeps_order_within_a_block(self) -> None:
        cell_l = np.array([5, 1, 9, 0, 6, 2], dtype=np.int32)
        order = edge_order_by_block(cell_l, 4)
        assert (cell_l[order] // 4).tolist() == [0, 0, 0, 1, 1, 2]
        # Stable: within block 0 the original order 1, 0, 2 is kept.
        assert cell_l[order][:3].tolist() == [1, 0, 2]

    def test_is_a_permutation(self, grid: MeshGeometry) -> None:
        order = edge_order_by_block(grid.edge_cellL, ACTIVE_BLOCK_CELLS)
        assert np.array_equal(np.sort(order), np.arange(grid.E))


class TestRemapCellEdges:
    def test_follows_the_permutation_and_keeps_the_side_bit(self) -> None:
        # Old edges 0, 1, 2 move to positions 2, 0, 1.
        order = np.array([1, 2, 0], dtype=np.int64)
        signed = np.array([0 << 1 | 1, 1 << 1, 2 << 1 | 1], dtype=np.int32)
        assert remap_cell_edges(signed, order).tolist() == [2 << 1 | 1, 0 << 1, 1 << 1 | 1]

    def test_every_entry_names_an_edge_of_its_cell_after_sorting(self, grid: MeshGeometry) -> None:
        order = edge_order_by_block(grid.edge_cellL, ACTIVE_BLOCK_CELLS)
        remapped = remap_cell_edges(grid.cell_edge_idx_signed, order).astype(np.int64)
        cell = np.repeat(np.arange(grid.N), np.diff(grid.cell_edge_ptr))
        edge, right = remapped >> 1, (remapped & 1).astype(bool)
        side = np.where(right, grid.edge_cellR[order][edge], grid.edge_cellL[order][edge])
        assert np.array_equal(side, cell)
        # Every interior edge appears twice (once per side), every wall once.
        expected = 2 * grid.E - int(np.count_nonzero(grid.edge_cellR < 0))
        assert remapped.size == expected
        assert np.unique(remapped).size == expected

    def test_keeps_each_cells_order(self, grid: MeshGeometry) -> None:
        order = edge_order_by_block(grid.edge_cellL, ACTIVE_BLOCK_CELLS)
        remapped = remap_cell_edges(grid.cell_edge_idx_signed, order).astype(np.int64)
        old = order[remapped >> 1]
        assert np.array_equal(old, grid.cell_edge_idx_signed.astype(np.int64) >> 1)


class TestEdgeWorkgroupRanges:
    def test_first_and_last_block_per_workgroup(self) -> None:
        sorted_l = np.array([0, 1, 4, 5, 5, 9, 12], dtype=np.int32)
        ranges = edge_workgroup_block_ranges(sorted_l, workgroup=3, block_cells=4)
        assert ranges.tolist() == [[0, 1], [1, 2], [3, 3]]

    def test_every_edge_block_is_inside_its_workgroup_range(self, grid: MeshGeometry) -> None:
        order = edge_order_by_block(grid.edge_cellL, ACTIVE_BLOCK_CELLS)
        sorted_l = grid.edge_cellL[order]
        ranges = edge_workgroup_block_ranges(sorted_l, 256, ACTIVE_BLOCK_CELLS)
        wg = np.arange(sorted_l.size) // 256
        blocks = sorted_l // ACTIVE_BLOCK_CELLS
        assert np.all(ranges[wg, 0] <= blocks)
        assert np.all(blocks <= ranges[wg, 1])

    def test_rejects_unsorted_edges(self) -> None:
        with pytest.raises(ValueError, match="sorted"):
            edge_workgroup_block_ranges(np.array([8, 0], dtype=np.int32), 2, 4)


class TestBlockAdjacency:
    def test_chain(self) -> None:
        # Cells 0..7 in a line, two cells per block: blocks 0-1-2-3.
        cell_l = np.array([0, 1, 2, 3, 4, 5, 6, 0], dtype=np.int32)
        cell_r = np.array([1, 2, 3, 4, 5, 6, 7, -1], dtype=np.int32)
        ptr, idx = block_adjacency(cell_l, cell_r, 8, 2)
        neighbours = [idx[ptr[b] : ptr[b + 1]].tolist() for b in range(4)]
        assert neighbours == [[1], [0, 2], [1, 3], [2]]

    def test_symmetric_unique_and_without_self(self, grid: MeshGeometry) -> None:
        ptr, idx = block_adjacency(grid.edge_cellL, grid.edge_cellR, grid.N, ACTIVE_BLOCK_CELLS)
        n_blk = num_blocks(grid.N, ACTIVE_BLOCK_CELLS)
        pairs = {(b, int(n)) for b in range(n_blk) for n in idx[ptr[b] : ptr[b + 1]]}
        assert len(pairs) == idx.size
        assert all((n, b) in pairs for b, n in pairs)
        assert all(b != n for b, n in pairs)

    def test_single_block_has_no_neighbours(self) -> None:
        ptr, idx = block_adjacency(
            np.array([0, 0], dtype=np.int32), np.array([1, -1], dtype=np.int32), 2, 256
        )
        assert ptr.tolist() == [0, 0]
        assert idx.size == 0


class TestTwoRing:
    def test_chain(self) -> None:
        ptr = np.array([0, 1, 3, 5, 6], dtype=np.uint32)
        idx = np.array([1, 0, 2, 1, 3, 2], dtype=np.uint32)
        p2, i2 = block_two_ring(ptr, idx)
        assert [i2[p2[b] : p2[b + 1]].tolist() for b in range(4)] == [
            [1, 2],
            [0, 2, 3],
            [0, 1, 3],
            [1, 2],
        ]

    def test_one_dilation_equals_two_on_a_mesh(self, grid: MeshGeometry) -> None:
        ptr, idx = block_adjacency(grid.edge_cellL, grid.edge_cellR, grid.N, ACTIVE_BLOCK_CELLS)
        p2, i2 = block_two_ring(ptr, idx)
        rng = np.random.default_rng(7)
        for _ in range(20):
            flags = rng.random(ptr.size - 1) < 0.05
            assert np.array_equal(
                dilate_blocks(flags, p2, i2),
                dilate_blocks(dilate_blocks(flags, ptr, idx), ptr, idx),
            )


class TestHalo:
    def test_two_cell_rings_stay_inside_the_block_two_ring(self, grid: MeshGeometry) -> None:
        ptr, idx = block_two_ring(
            *block_adjacency(grid.edge_cellL, grid.edge_cellR, grid.N, ACTIVE_BLOCK_CELLS)
        )
        cell_block = np.arange(grid.N) // ACTIVE_BLOCK_CELLS
        rng = np.random.default_rng(3)
        for frac in (0.0005, 0.005, 0.05):
            wet = rng.random(grid.N) < frac
            marked = np.zeros(ptr.size - 1, dtype=bool)
            marked[cell_block[wet]] = True
            active = dilate_blocks(marked, ptr, idx)
            reach = _cell_dilate(_cell_dilate(wet, grid), grid)
            assert active[cell_block[reach]].all()
