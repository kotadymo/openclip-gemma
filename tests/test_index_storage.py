from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from openclip_gemma.index_storage import (
    IndexArchiveError,
    _open_numpy_archive,
    _save_numpy_archive_atomic,
)


def test_numpy_archive_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    _save_numpy_archive_atomic(
        path,
        {"kind": "test", "format_version": 1},
        {"embeddings": embeddings},
    )

    with _open_numpy_archive(path, expected_arrays=frozenset({"embeddings"})) as (
        metadata,
        archive,
    ):
        assert metadata == {"kind": "test", "format_version": 1}
        np.testing.assert_array_equal(archive["embeddings"], embeddings)


def test_rejects_npy_header_with_unbacked_huge_shape_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "malformed.npz"
    metadata_buffer = io.BytesIO()
    np.save(
        metadata_buffer,
        np.frombuffer(b"{}", dtype=np.uint8),
        allow_pickle=False,
    )

    malicious_array = io.BytesIO()
    header = repr(
        {
            "descr": np.dtype(np.float32).str,
            "fortran_order": False,
            "shape": (1_000_000_000,),
        }
    ).encode("latin1")
    padding = 64 - ((10 + len(header) + 1) % 64)
    header += b" " * (padding - 1) + b"\n"
    malicious_array.write(b"\x93NUMPY\x01\x00")
    malicious_array.write(len(header).to_bytes(2, "little"))
    malicious_array.write(header)
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("metadata.npy", metadata_buffer.getvalue())
        archive.writestr("embeddings.npy", malicious_array.getvalue())

    with (
        pytest.raises(IndexArchiveError, match="payload size does not match"),
        _open_numpy_archive(path, expected_arrays=frozenset({"embeddings"})),
    ):
        pytest.fail("The invalid archive should be rejected before opening.")


def test_rejects_archive_above_configured_file_size_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "index.npz"
    _save_numpy_archive_atomic(
        path,
        {"kind": "test"},
        {"embeddings": np.ones((1, 2), dtype=np.float32)},
    )
    monkeypatch.setattr("openclip_gemma.index_storage._MAX_ARCHIVE_BYTES", 1)

    with (
        pytest.raises(IndexArchiveError, match="size limit"),
        _open_numpy_archive(path, expected_arrays=frozenset({"embeddings"})),
    ):
        pytest.fail("The oversized archive should be rejected.")
