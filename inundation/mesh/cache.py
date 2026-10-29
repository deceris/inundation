"""Disk cache for preprocessed ``MeshGeometry``.

``build_geometry`` and ``hilbert_reorder`` are Python-loop heavy. The cache
avoids rebuilding identical geometry for repeated benchmark runs.

Two layers are provided:

* key-addressed directory cache (``geometry_cache_dir``) — the key covers mesh
  bytes, cache format version, and reorder mode. Best-effort by design: any
  problem is a cache miss and the mesh is rebuilt from scratch.
* explicit artifact files (``build_geometry_cache`` /
  ``load_geometry_cache_artifact``) — self-contained geometry plus provenance,
  for pipelines that preprocess the mesh in an earlier stage: the artifact is
  built once, stored anywhere, and later reused with no mesh file present.

Run ``python -m inundation.mesh.cache`` for the standalone self-check.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .geometry import MeshGeometry, build_geometry, hilbert_reorder
from .loader import load_mesh_file

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Bump whenever build_geometry()/hilbert_reorder() output arrays change
# shape, dtype, or semantics, so old on-disk caches are rejected instead of
# silently served with stale/incompatible contents.
GEOMETRY_CACHE_VERSION = 5

# Field order matters only for readability; np.savez uses keyword storage.
_GEOMETRY_ARRAY_FIELDS = (
    "centroid",
    "area",
    "zb",
    "manning_n",
    "cell_vertices",
    "cell_vertex_ptr",
    "cell_bbox",
    "edge_len",
    "edge_nx",
    "edge_ny",
    "edge_dzL",
    "edge_dzR",
    "edge_cellL",
    "edge_cellR",
    "edge_sideL",
    "edge_sideR",
    "cell_edges",
    "cell_neighbors",
    "cell_edge_ptr",
    "cell_edge_idx",
    "cell_nbr_idx",
    "cell_edge_side",
    "cell_edge_idx_signed",
)
_GEOMETRY_SCALAR_FIELDS = ("N", "E", "V", "max_degree")

# Explicit artifacts additionally carry a JSON provenance blob.
_META_FIELD = "meta"


@dataclass(frozen=True)
class GeometryCacheMeta:
    """Provenance recorded inside an explicit geometry-cache artifact."""

    version: int
    """``GEOMETRY_CACHE_VERSION`` of the code that wrote the artifact."""

    mesh_source: str
    """Mesh file the geometry was built from (provenance only)."""

    use_hilbert_reorder: bool
    """Whether the artifact's cell order is the Hilbert permutation."""


def geometry_cache_key(mesh_path: str | Path, *, use_hilbert_reorder: bool) -> str:
    """Return a stable cache key derived from mesh file content and configuration."""
    path = Path(mesh_path)
    hasher = hashlib.blake2b(digest_size=32)
    hasher.update(f"v{GEOMETRY_CACHE_VERSION}|hilbert={use_hilbert_reorder}|".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def geometry_cache_path(cache_dir: str | os.PathLike[str], cache_key: str) -> Path:
    """Return the on-disk path for a given cache key under ``cache_dir``."""
    return Path(cache_dir) / f"geometry_{cache_key}.npz"


def build_geometry_cache(
    mesh_source: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    use_hilbert_reorder: bool = True,
    progress: bool = False,
) -> Path:
    """Preprocess ``mesh_source`` into a standalone geometry-cache artifact.

    Runs the same load → ``build_geometry`` → ``hilbert_reorder`` chain as
    ``SWEWorkflow.prepare()``, needs no GPU, and reads the mesh exactly once.
    The resulting file can be stored and later consumed by
    ``WorkflowConfig.geometry_cache_source``, which never reads the mesh file.

    Parameters
    ----------
    mesh_source : path to the mesh file (.gpkg, .shp, .parquet, .geoparquet, .obj)
    out_path    : destination file (any name; ``.npz`` by convention)
    use_hilbert_reorder : whether cells are stored in Hilbert order
    progress    : print coarse per-stage timing to stdout

    Returns
    -------
    Path of the written artifact.

    """
    mesh_path = os.fspath(mesh_source)
    (
        verts,
        faces_flat,
        face_offsets,
        zb_from_file,
        manning_n_from_file,
    ) = load_mesh_file(mesh_path)
    geom = build_geometry(
        verts,
        faces_flat,
        face_offsets=face_offsets,
        zb_from_file=zb_from_file,
        manning_n_from_file=manning_n_from_file,
        progress=progress,
    )

    perm = np.arange(geom.N, dtype=np.int32)
    if use_hilbert_reorder:
        geom, perm = hilbert_reorder(geom, verbose=progress)

    meta = GeometryCacheMeta(
        version=GEOMETRY_CACHE_VERSION,
        mesh_source=mesh_path,
        use_hilbert_reorder=use_hilbert_reorder,
    )
    return _write_geometry_npz(Path(out_path), geom, perm, meta)


def load_geometry_cache_artifact(
    path: str | os.PathLike[str],
) -> tuple[MeshGeometry, NDArray[np.int32], GeometryCacheMeta]:
    """Load an artifact written by :func:`build_geometry_cache`.

    Unlike :func:`load_geometry_cache` this raises instead of falling back:
    the artifact *is* the geometry source, so silently ignoring a missing or
    stale file would run the solver on the wrong mesh.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"geometry cache artifact not found: {artifact_path}")

    try:
        with np.load(artifact_path) as npz:
            payload = {name: npz[name] for name in npz.files}
    # zipfile.BadZipFile is raised for a truncated/torn .npz and is not an
    # OSError/ValueError subclass, so it must be listed here explicitly.
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"unreadable geometry cache artifact {artifact_path}: {exc}") from exc

    required = (*_GEOMETRY_ARRAY_FIELDS, *_GEOMETRY_SCALAR_FIELDS, "perm", _META_FIELD)
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"geometry cache artifact {artifact_path} is missing fields: {missing}")

    meta = _decode_meta(payload[_META_FIELD])
    if meta.version != GEOMETRY_CACHE_VERSION:
        raise ValueError(
            f"geometry cache artifact {artifact_path} was written with cache version "
            f"{meta.version}, this build expects {GEOMETRY_CACHE_VERSION} — rebuild it "
            f"with build_geometry_cache()"
        )
    return _geometry_from_payload(payload), payload["perm"], meta


def save_geometry_cache(
    cache_dir: str | os.PathLike[str],
    cache_key: str,
    geom: MeshGeometry,
    perm: NDArray[np.int32],
) -> Path:
    """Persist ``geom`` + ``perm`` to ``<cache_dir>/geometry_<cache_key>.npz``."""
    return _write_geometry_npz(geometry_cache_path(cache_dir, cache_key), geom, perm)


def load_geometry_cache(
    cache_dir: str | os.PathLike[str], cache_key: str
) -> tuple[MeshGeometry, NDArray[np.int32]] | None:
    """Load a previously cached ``(MeshGeometry, perm)`` pair, or ``None`` on any miss.

    Any structural problem (missing file, missing field, unreadable archive)
    is treated as a cache miss — always falls back to rebuilding from scratch
    rather than raising, since the cache is a pure performance optimization.
    """
    path = geometry_cache_path(cache_dir, cache_key)
    if not path.exists():
        return None

    try:
        with np.load(path) as npz:
            payload = {name: npz[name] for name in npz.files}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None

    required = (*_GEOMETRY_ARRAY_FIELDS, *_GEOMETRY_SCALAR_FIELDS, "perm")
    if any(name not in payload for name in required):
        return None

    return _geometry_from_payload(payload), payload["perm"]


def _write_geometry_npz(
    path: Path,
    geom: MeshGeometry,
    perm: NDArray[np.int32],
    meta: GeometryCacheMeta | None = None,
) -> Path:
    """Write ``geom`` + ``perm`` (+ optional ``meta``) to ``path`` atomically."""
    # ``manning_n`` is the one optional array field; None round-trips as empty.
    payload: dict[str, Any] = {
        name: (
            np.empty(0, dtype=np.float32)
            if name == "manning_n" and geom.manning_n is None
            else getattr(geom, name)
        )
        for name in _GEOMETRY_ARRAY_FIELDS
    }
    payload.update({name: np.int64(getattr(geom, name)) for name in _GEOMETRY_SCALAR_FIELDS})
    payload["perm"] = perm
    if meta is not None:
        payload[_META_FIELD] = np.asarray(
            json.dumps(
                {
                    "version": meta.version,
                    "mesh_source": meta.mesh_source,
                    "use_hilbert_reorder": meta.use_hilbert_reorder,
                }
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then rename so interruption cannot leave a truncated
    # cache. Passing an open handle avoids np.savez appending another suffix.
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez(f, **payload)
    tmp_path.replace(path)
    return path


def _geometry_from_payload(payload: dict[str, Any]) -> MeshGeometry:
    """Rebuild a ``MeshGeometry`` from materialized npz contents."""
    arrays: dict[str, Any] = {name: payload[name] for name in _GEOMETRY_ARRAY_FIELDS}
    arrays["manning_n"] = None if arrays["manning_n"].size == 0 else arrays["manning_n"]
    return MeshGeometry(
        **arrays,
        N=int(payload["N"]),
        E=int(payload["E"]),
        V=int(payload["V"]),
        max_degree=int(payload["max_degree"]),
    )


def _decode_meta(raw: NDArray[Any]) -> GeometryCacheMeta:
    """Decode the JSON provenance blob stored in an artifact."""
    try:
        data = json.loads(str(raw.item()))
        return GeometryCacheMeta(
            version=int(data["version"]),
            mesh_source=str(data["mesh_source"]),
            use_hilbert_reorder=bool(data["use_hilbert_reorder"]),
        )
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid geometry cache metadata: {raw!r}") from exc


def _self_check() -> None:
    """Build a tiny synthetic quad mesh, cache it, reload it, and diff arrays."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 2x2 grid of unit quads -> 4 cells, shared vertices. Built directly
        # from arrays (not via a mesh file / load_mesh_file) since this is a
        # synthetic geometry-cache round-trip check, not a mesh-loader check.
        verts = np.array(
            [[x, y] for y in range(3) for x in range(3)],
            dtype=np.float32,
        )

        def quad(ix: int, iy: int) -> list[int]:
            return [
                iy * 3 + ix,
                iy * 3 + ix + 1,
                (iy + 1) * 3 + ix + 1,
                (iy + 1) * 3 + ix,
            ]

        faces = np.array([quad(0, 0), quad(1, 0), quad(0, 1), quad(1, 1)], dtype=np.int32)
        zb = np.zeros(len(faces), dtype=np.float32)

        geom = build_geometry(verts, faces, zb_from_file=zb)
        geom, perm = hilbert_reorder(geom, verbose=False)

        cache_dir = tmp_dir / "geom-cache"
        cache_key = "self-check-key"
        save_geometry_cache(cache_dir, cache_key, geom, perm)
        loaded = load_geometry_cache(cache_dir, cache_key)
        if loaded is None:
            raise AssertionError("cache load returned None right after save")
        loaded_geom, loaded_perm = loaded

        for name in _GEOMETRY_ARRAY_FIELDS:
            np.testing.assert_array_equal(
                getattr(loaded_geom, name), getattr(geom, name), err_msg=f"field mismatch: {name}"
            )
        for name in _GEOMETRY_SCALAR_FIELDS:
            if getattr(loaded_geom, name) != getattr(geom, name):
                raise AssertionError(f"scalar mismatch: {name}")
        np.testing.assert_array_equal(loaded_perm, perm)

        # A cache-key mismatch (e.g. different mesh content or version) must miss.
        if load_geometry_cache(cache_dir, "wrong-key") is not None:
            raise AssertionError("cache lookup with a mismatched key unexpectedly hit")

        # Explicit artifact round-trip, including provenance and version check.
        artifact = tmp_dir / "artifact.npz"
        meta = GeometryCacheMeta(
            version=GEOMETRY_CACHE_VERSION,
            mesh_source="synthetic",
            use_hilbert_reorder=True,
        )
        _write_geometry_npz(artifact, geom, perm, meta)
        artifact_geom, artifact_perm, artifact_meta = load_geometry_cache_artifact(artifact)
        np.testing.assert_array_equal(artifact_geom.zb, geom.zb)
        np.testing.assert_array_equal(artifact_perm, perm)
        if artifact_meta != meta:
            raise AssertionError(f"meta mismatch: {artifact_meta} != {meta}")

        stale = tmp_dir / "stale.npz"
        _write_geometry_npz(
            stale,
            geom,
            perm,
            GeometryCacheMeta(
                version=GEOMETRY_CACHE_VERSION + 1,
                mesh_source="synthetic",
                use_hilbert_reorder=True,
            ),
        )
        try:
            load_geometry_cache_artifact(stale)
        except ValueError:
            pass
        else:
            raise AssertionError("stale artifact version was accepted")

        print("geometry cache self-check: OK")


if __name__ == "__main__":
    _self_check()
