"""SWASHES reference layer (CPU only; needs the ``swashes`` extra).

Checks the binary is called as the harness assumes and its output lands on the
mesh's cell centres in mesh order. The solver comparisons live in
``inundation/bench/swashes.py`` (manual, GPU-required).
"""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

import numpy as np
import pytest

from inundation.bench.common import build_channel_mesh
from inundation.bench.swashes import CASES, Case, parse_swashes, reference

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _has_swashes() -> bool:
    try:
        importlib.metadata.distribution("swashes")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _has_swashes(), reason="needs the swashes extra")

G = 9.81


def _case(name: str) -> Case:
    return next(c for c in CASES if c.name == name)


def _centroids(case: Case) -> NDArray[np.float32]:
    verts, quads = build_channel_mesh(case.nx, case.ny, case.length_m, case.width_m)
    return verts[quads].mean(axis=1)


def test_parse_reads_rows_and_time() -> None:
    text = "# Time value: 1.5 seconds\n# header\n 1 2 3\n\n 4 5 6\n"
    rows, t = parse_swashes(text)
    assert t == 1.5
    assert rows.tolist() == [[1, 2, 3], [4, 5, 6]]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_reference_sits_on_mesh_cell_centres(case: Case) -> None:
    ref = reference(case)
    c = _centroids(case)
    assert ref.h.shape == (case.nx * case.ny,)
    assert np.allclose(ref.x, c[:, 0], atol=1e-4 * case.length_m)
    assert (ref.h >= 0.0).all()


def test_2d_reference_matches_its_own_bed_formula() -> None:
    # z = h0 (r^2 / a^2 - 1), h0 = 0.1, a = 1, centred on (2, 2): checks the
    # x-outer -> y-outer transpose, which a symmetric field would not catch.
    case = _case("thacker_2d_planar")
    ref = reference(case)
    c = _centroids(case)
    r2 = (c[:, 0] - 2.0) ** 2 + (c[:, 1] - 2.0) ** 2
    assert np.allclose(ref.z, 0.1 * (r2 - 1.0), atol=1e-5)
    # At whole periods the planar surface is at its x-extreme: u = 0, v != 0.
    assert np.allclose(ref.hu, 0.0, atol=1e-6) and np.abs(ref.hv).max() > 0.05


@pytest.mark.parametrize("name", ["bump_subcritical", "bump_transcritical", "bump_shock"])
def test_bump_reference_carries_the_inflow_discharge(name: str) -> None:
    case = _case(name)
    assert case.west is not None
    ref = reference(case)
    assert np.allclose(ref.hu, case.west[1], rtol=1e-5)


def test_transcritical_bump_turns_supercritical_at_the_crest() -> None:
    ref = reference(_case("bump_transcritical"))
    froude = (ref.hu / ref.h) / np.sqrt(G * ref.h)
    upstream, downstream = ref.x < 9.0, ref.x > 11.0
    assert (froude[upstream] < 1.0).all()
    assert (froude[downstream] > 1.0).all()  # why the east edge is a free outflow
