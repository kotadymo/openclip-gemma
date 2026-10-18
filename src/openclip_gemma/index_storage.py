"""Safe NumPy index archive helpers shared by image and video search."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile

_HASH_BLOCK_SIZE = 4096


class IndexArchiveError(RuntimeError):
    """Raised when a NumPy index archive cannot be safely read or written."""


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(array), _HASH_BLOCK_SIZE):
        block = np.asarray(
            array[start : start + _HASH_BLOCK_SIZE],
            dtype="<f4",
            order="C",
        )
        digest.update(block.tobytes())
    return digest.hexdigest()


def _save_numpy_archive_atomic(
    path: Path,
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> Path:
    if "metadata" in arrays or not arrays:
        raise ValueError("An index archive needs data arrays and a reserved metadata entry.")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_bytes = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.stem}-",
            suffix=".npz",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez(
                temporary_file,
                metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
                **arrays,
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        raise IndexArchiveError(f"Could not write index archive {path}: {error}") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


@contextmanager
def _open_numpy_archive(
    path: Path,
    *,
    expected_arrays: frozenset[str],
) -> Iterator[tuple[dict[str, Any], NpzFile]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_arrays | {"metadata"}:
                raise IndexArchiveError(
                    f"Index archive has unexpected entries: {path}"
                )
            metadata_array = archive["metadata"]
            if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
                raise IndexArchiveError(
                    f"Index archive metadata has an invalid representation: {path}"
                )
            try:
                metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise IndexArchiveError(
                    f"Index archive metadata is not UTF-8 JSON: {path}"
                ) from error
            if not isinstance(metadata, dict):
                raise IndexArchiveError(
                    f"Index archive metadata must be a JSON object: {path}"
                )
            yield metadata, archive
    except IndexArchiveError:
        raise
    except FileNotFoundError as error:
        raise IndexArchiveError(f"Index archive not found: {path}") from error
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as error:
        raise IndexArchiveError(f"Could not read index archive {path}: {error}") from error
