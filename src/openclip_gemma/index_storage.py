"""Safe NumPy index archive helpers shared by image and video search."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile

_HASH_BLOCK_SIZE = 4096
_MAX_ARCHIVE_BYTES = 512 * 1024**2
_MAX_EXPANDED_BYTES = 512 * 1024**2
_MAX_METADATA_BYTES = 64 * 1024**2
_MAX_NPY_HEADER_BYTES = 1024**2


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
    if len(metadata_bytes) > _MAX_METADATA_BYTES:
        raise IndexArchiveError(
            f"Index metadata exceeds the {_MAX_METADATA_BYTES}-byte safety limit."
        )
    if any(not isinstance(array, np.ndarray) or array.dtype.hasobject for array in arrays.values()):
        raise IndexArchiveError("Index data must use non-object NumPy arrays.")
    expanded_bytes = len(metadata_bytes) + sum(array.nbytes for array in arrays.values())
    expanded_bytes += _MAX_NPY_HEADER_BYTES * (len(arrays) + 1)
    if expanded_bytes > _MAX_EXPANDED_BYTES:
        raise IndexArchiveError(
            f"Index arrays exceed the {_MAX_EXPANDED_BYTES}-byte expanded-size limit."
        )
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


def _preflight_numpy_archive(
    path: Path,
    *,
    expected_arrays: frozenset[str],
) -> None:
    try:
        archive_size = path.stat().st_size
        if archive_size > _MAX_ARCHIVE_BYTES:
            raise IndexArchiveError(
                f"Index archive exceeds the {_MAX_ARCHIVE_BYTES}-byte size limit: {path}"
            )
        expected_members = {f"{name}.npy" for name in expected_arrays | {"metadata"}}
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if (
                len(members) != len(expected_members)
                or {member.filename for member in members} != expected_members
            ):
                raise IndexArchiveError(
                    f"Index archive has unexpected or duplicate members: {path}"
                )
            expanded_size = sum(member.file_size for member in members)
            if expanded_size > _MAX_EXPANDED_BYTES:
                raise IndexArchiveError(
                    "Index archive expands beyond the "
                    f"{_MAX_EXPANDED_BYTES}-byte safety limit: {path}"
                )

            for member in members:
                if member.file_size > _MAX_EXPANDED_BYTES:
                    raise IndexArchiveError(
                        f"Index archive member exceeds the safety limit: {member.filename}"
                    )
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(
                            stream,
                            max_header_size=_MAX_NPY_HEADER_BYTES,
                        )
                    elif version in ((2, 0), (3, 0)):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(
                            stream,
                            max_header_size=_MAX_NPY_HEADER_BYTES,
                        )
                    else:
                        raise IndexArchiveError(
                            f"Unsupported NumPy array format in {member.filename}."
                        )
                    if dtype.hasobject:
                        raise IndexArchiveError(
                            f"Object arrays are not permitted in {member.filename}."
                        )
                    if member.filename == "metadata.npy" and (
                        dtype != np.dtype(np.uint8)
                        or len(shape) != 1
                        or member.file_size > _MAX_METADATA_BYTES + _MAX_NPY_HEADER_BYTES
                    ):
                        raise IndexArchiveError(
                            f"Index metadata member exceeds its representation limits: {path}"
                        )
                    expected_size = stream.tell() + math.prod(shape) * dtype.itemsize
                    if expected_size != member.file_size:
                        raise IndexArchiveError(
                            f"NumPy member payload size does not match its header: "
                            f"{member.filename}"
                        )
    except IndexArchiveError:
        raise
    except (
        OSError,
        ValueError,
        EOFError,
        OverflowError,
        struct.error,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as error:
        raise IndexArchiveError(f"Could not safely inspect index archive {path}: {error}") from error


@contextmanager
def _open_numpy_archive(
    path: Path,
    *,
    expected_arrays: frozenset[str],
) -> Iterator[tuple[dict[str, Any], NpzFile]]:
    try:
        _preflight_numpy_archive(path, expected_arrays=expected_arrays)
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
