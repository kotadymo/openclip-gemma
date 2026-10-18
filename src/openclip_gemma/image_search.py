"""Internal image indexing and text-to-image search helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from openclip_gemma.batching import ImagePath, iter_image_embeddings
from openclip_gemma.constants import MODEL_ID, MODEL_REVISION
from openclip_gemma.index_storage import (
    IndexArchiveError,
    _array_sha256,
    _open_numpy_archive,
    _save_numpy_archive_atomic,
)
from openclip_gemma.tokenizer import get_tokenizer

_INDEX_FILENAME = "index.npz"
_INDEX_FORMAT_VERSION = 1
_EMBEDDING_DIMENSION = 768
_PREPROCESSING_ID = "embeddinggemma-2-official-image-preprocess-v1"
_SUPPORTED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_SCORE_BLOCK_SIZE = 4096

__all__ = [
    "ImageIndexCompatibilityError",
    "ImageIndexError",
    "ImageIndexUpdate",
    "ImageSearchResult",
    "build_or_update_image_index",
    "search_images",
]

type _EmbeddingMatrix = NDArray[np.float32]
type _ImagePreprocess = Callable[[Image.Image], torch.Tensor]


class ImageIndexError(RuntimeError):
    """Raised when an image index cannot be built, read, or searched safely."""


class ImageIndexCompatibilityError(ImageIndexError):
    """Raised when an index was created for incompatible model settings."""


class _ImageSearchModel(Protocol):
    def parameters(self) -> Iterator[torch.nn.Parameter]: ...

    def encode_image(self, images: torch.Tensor) -> torch.Tensor: ...

    def encode_text(
        self,
        tokens: torch.Tensor,
        *,
        normalize: bool = False,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class _ImageRecord:
    relative_path: str
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class _ImageIndex:
    image_root: Path
    recursive: bool
    records: tuple[_ImageRecord, ...]
    embeddings: _EmbeddingMatrix


@dataclass(frozen=True)
class ImageIndexUpdate:
    """Summary of an index build or incremental update."""

    index_path: Path
    added: int
    updated: int
    removed: int
    unchanged: int


@dataclass(frozen=True)
class ImageSearchResult:
    """One image path and its unscaled cosine similarity score."""

    path: Path
    score: float


def _path_key(path: str) -> str:
    return os.path.normcase(path.replace("/", os.sep))


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")


def _scan_image_files(
    images_dir: str | os.PathLike[str],
    *,
    recursive: bool,
) -> tuple[Path, tuple[_ImageRecord, ...]]:
    if not isinstance(recursive, bool):
        raise TypeError("recursive must be a bool.")
    try:
        root = Path(images_dir).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Image directory does not exist: {images_dir}") from error
    if not root.is_dir():
        raise ValueError(f"Image directory is not a directory: {root}")

    candidates = root.rglob("*") if recursive else root.iterdir()
    records_by_key: dict[str, _ImageRecord] = {}
    try:
        for path in candidates:
            if path.is_symlink() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            if not path.is_file():
                continue
            file_stat = path.stat()
            relative_path = path.relative_to(root).as_posix()
            key = _path_key(relative_path)
            records_by_key.setdefault(
                key,
                _ImageRecord(
                    relative_path=relative_path,
                    size_bytes=file_stat.st_size,
                    mtime_ns=file_stat.st_mtime_ns,
                ),
            )
    except OSError as error:
        raise ImageIndexError(f"Could not scan image directory {root}: {error}") from error

    records = tuple(
        sorted(
            records_by_key.values(),
            key=lambda record: (_path_key(record.relative_path), record.relative_path),
        )
    )
    return root, records


def _model_settings(model: _ImageSearchModel) -> tuple[torch.device, str]:
    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise ImageIndexError("The model has no parameters to determine its settings.") from error

    precision_by_dtype = {
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
    }
    precision = precision_by_dtype.get(parameter.dtype)
    if precision is None:
        raise ImageIndexError(
            "Image search indexes support models using FP32 or BF16 parameters; "
            f"received {parameter.dtype}."
        )
    return parameter.device, precision


def _record_signature(record: _ImageRecord) -> tuple[int, int]:
    return record.size_bytes, record.mtime_ns


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ImageIndexError("Index metadata contains an invalid image path.")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or (len(path.parts[0]) == 2 and path.parts[0][1] == ":")
        or path.as_posix() != value
    ):
        raise ImageIndexError("Index metadata contains a non-relative image path.")
    return value


def _parse_record(value: object) -> _ImageRecord:
    if not isinstance(value, dict):
        raise ImageIndexError("Index metadata contains an invalid image record.")
    path = _validate_relative_path(value.get("path"))
    size_bytes = value.get("size_bytes")
    mtime_ns = value.get("mtime_ns")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
        or isinstance(mtime_ns, bool)
        or not isinstance(mtime_ns, int)
    ):
        raise ImageIndexError("Index metadata contains invalid file information.")
    return _ImageRecord(path, size_bytes, mtime_ns)


def _metadata_for_index(index: _ImageIndex, precision: str) -> dict[str, object]:
    return {
        "format_version": _INDEX_FORMAT_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_precision": precision,
        "embedding_dimension": _EMBEDDING_DIMENSION,
        "preprocessing_id": _PREPROCESSING_ID,
        "storage_dtype": "float32",
        "image_root": str(index.image_root),
        "recursive": index.recursive,
        "entry_count": len(index.records),
        "records": [
            {
                "path": record.relative_path,
                "size_bytes": record.size_bytes,
                "mtime_ns": record.mtime_ns,
            }
            for record in index.records
        ],
        "embedding_sha256": _array_sha256(index.embeddings),
    }


def _save_index(index_dir: Path, index: _ImageIndex, precision: str) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / _INDEX_FILENAME
    embeddings = np.ascontiguousarray(index.embeddings, dtype=np.float32)
    if embeddings.shape != (len(index.records), _EMBEDDING_DIMENSION):
        raise ImageIndexError("Refusing to save an index with inconsistent dimensions.")
    if not np.isfinite(embeddings).all():
        raise ImageIndexError("Refusing to save non-finite image embeddings.")

    try:
        return _save_numpy_archive_atomic(
            index_path,
            _metadata_for_index(index, precision),
            {"embeddings": embeddings},
        )
    except IndexArchiveError as error:
        raise ImageIndexError(str(error)) from error


def _load_index(
    index_dir: str | os.PathLike[str],
    *,
    expected_root: Path | None = None,
    expected_precision: str | None = None,
) -> _ImageIndex:
    index_path = Path(index_dir).expanduser() / _INDEX_FILENAME
    try:
        with _open_numpy_archive(
            index_path,
            expected_arrays=frozenset({"embeddings"}),
        ) as (metadata, archive):
            records, image_root, recursive = _validate_metadata(
                metadata,
                expected_root=expected_root,
                expected_precision=expected_precision,
            )
            embeddings = archive["embeddings"]
    except IndexArchiveError as error:
        if not index_path.exists():
            raise ImageIndexError(f"Image index not found: {index_path}") from error
        raise ImageIndexError(f"Could not read image index {index_path}: {error}") from error
    except ImageIndexError:
        raise

    expected_shape = (len(records), _EMBEDDING_DIMENSION)
    if embeddings.dtype != np.float32 or embeddings.shape != expected_shape:
        raise ImageIndexError(
            "Image index embeddings have an invalid dtype or shape: "
            f"expected float32{expected_shape}, received {embeddings.dtype}{embeddings.shape}."
        )
    if metadata["entry_count"] != len(records):
        raise ImageIndexError("Image index entry count does not match its metadata.")
    if not np.isfinite(embeddings).all():
        raise ImageIndexError("Image index contains non-finite embeddings.")
    if _array_sha256(embeddings) != metadata["embedding_sha256"]:
        raise ImageIndexError("Image index embedding checksum does not match its metadata.")
    return _ImageIndex(image_root, recursive, records, embeddings)


def _validate_metadata(
    metadata: dict[str, object],
    *,
    expected_root: Path | None,
    expected_precision: str | None,
) -> tuple[tuple[_ImageRecord, ...], Path, bool]:
    version = metadata.get("format_version")
    if type(version) is not int or version != _INDEX_FORMAT_VERSION:
        raise ImageIndexCompatibilityError(
            f"Unsupported image index format version: {version!r}."
        )
    compatibility_fields = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "embedding_dimension": _EMBEDDING_DIMENSION,
        "preprocessing_id": _PREPROCESSING_ID,
        "storage_dtype": "float32",
    }
    for name, expected in compatibility_fields.items():
        if metadata.get(name) != expected:
            raise ImageIndexCompatibilityError(
                f"Image index {name} does not match the current model configuration."
            )

    precision = metadata.get("model_precision")
    if not isinstance(precision, str) or precision not in {"fp32", "bf16"}:
        raise ImageIndexError("Image index metadata contains an invalid model precision.")
    if expected_precision is not None and precision != expected_precision:
        raise ImageIndexCompatibilityError(
            "Image index model precision does not match the loaded model; "
            "rebuild the index using the same precision."
        )

    root_value = metadata.get("image_root")
    if not isinstance(root_value, str) or not root_value:
        raise ImageIndexError("Image index metadata contains an invalid image root.")
    image_root = Path(root_value)
    if expected_root is not None and os.path.normcase(str(image_root)) != os.path.normcase(
        str(expected_root)
    ):
        raise ImageIndexCompatibilityError(
            "Image index was created for a different image directory."
        )

    recursive = metadata.get("recursive")
    if not isinstance(recursive, bool):
        raise ImageIndexError("Image index metadata contains an invalid recursion setting.")
    records_value = metadata.get("records")
    if not isinstance(records_value, list):
        raise ImageIndexError("Image index metadata contains an invalid record list.")
    records = tuple(_parse_record(record) for record in records_value)
    keys = [_path_key(record.relative_path) for record in records]
    if len(set(keys)) != len(keys):
        raise ImageIndexError("Image index metadata contains duplicate image paths.")
    if records != tuple(
        sorted(records, key=lambda record: (_path_key(record.relative_path), record.relative_path))
    ):
        raise ImageIndexError("Image index records are not in canonical path order.")

    entry_count = metadata.get("entry_count")
    if (
        isinstance(entry_count, bool)
        or not isinstance(entry_count, int)
        or entry_count < 0
        or entry_count != len(records)
    ):
        raise ImageIndexError("Image index metadata contains an invalid entry count.")
    checksum = metadata.get("embedding_sha256")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise ImageIndexError("Image index metadata contains an invalid embedding checksum.")
    return records, image_root, recursive


def _file_signature(path: Path) -> tuple[int, int]:
    try:
        file_stat = path.stat()
    except OSError as error:
        raise ImageIndexError(f"Could not inspect image file {path}: {error}") from error
    return file_stat.st_size, file_stat.st_mtime_ns


def _build_or_update_image_index(
    model: _ImageSearchModel,
    preprocess: _ImagePreprocess,
    images_dir: str | os.PathLike[str],
    index_dir: str | os.PathLike[str],
    *,
    batch_size: int = 8,
    recursive: bool = False,
) -> ImageIndexUpdate:
    """Create or incrementally update a safe, local image index."""
    _validate_batch_size(batch_size)
    root, records = _scan_image_files(images_dir, recursive=recursive)
    output_dir = Path(index_dir).expanduser()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Index output path is not a directory: {output_dir}")
    index_path = output_dir / _INDEX_FILENAME
    device, precision = _model_settings(model)
    del device

    previous: _ImageIndex | None = None
    if index_path.exists():
        if not index_path.is_file():
            raise ImageIndexError(f"Image index path is not a file: {index_path}")
        previous = _load_index(
            output_dir,
            expected_root=root,
            expected_precision=precision,
        )
    elif not records:
        raise ValueError(f"No supported images were found in {root}.")

    previous_rows = (
        {
            _path_key(record.relative_path): row
            for row, record in enumerate(previous.records)
        }
        if previous is not None
        else {}
    )
    current_keys = {_path_key(record.relative_path) for record in records}
    removed = (
        sum(_path_key(record.relative_path) not in current_keys for record in previous.records)
        if previous is not None
        else 0
    )

    embeddings = np.empty((len(records), _EMBEDDING_DIMENSION), dtype=np.float32)
    changed_records: list[_ImageRecord] = []
    changed_rows: dict[str, int] = {}
    added = 0
    updated = 0
    unchanged = 0
    for row, record in enumerate(records):
        key = _path_key(record.relative_path)
        previous_row = previous_rows.get(key)
        if previous_row is None:
            added += 1
        elif _record_signature(previous.records[previous_row]) == _record_signature(record):
            embeddings[row] = previous.embeddings[previous_row]
            unchanged += 1
            continue
        else:
            updated += 1
        changed_records.append(record)
        changed_rows[key] = row

    if (
        previous is not None
        and previous.recursive == recursive
        and not changed_records
        and removed == 0
    ):
        return ImageIndexUpdate(index_path, added, updated, removed, unchanged)

    if changed_records:
        image_paths: Sequence[ImagePath] = [
            root / PurePosixPath(record.relative_path) for record in changed_records
        ]
        encoded_count = 0
        for path, embedding in iter_image_embeddings(
            model,
            preprocess,
            image_paths,
            batch_size=batch_size,
        ):
            key = _path_key(path.relative_to(root).as_posix())
            row = changed_rows.get(key)
            if row is None:
                raise ImageIndexError(f"Image encoder returned an unexpected path: {path}")
            vector = embedding.detach().to(device="cpu", dtype=torch.float32).numpy()
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 0:
                raise ImageIndexError(f"Image encoder returned an invalid embedding for {path}.")
            embeddings[row] = vector / norm
            encoded_count += 1
        if encoded_count != len(changed_records):
            raise ImageIndexError(
                "Image encoder returned an unexpected number of embeddings: "
                f"expected {len(changed_records)}, received {encoded_count}."
            )

    for record in records:
        path = root / PurePosixPath(record.relative_path)
        if _file_signature(path) != _record_signature(record):
            raise ImageIndexError(
                f"Image changed during indexing: {path}. Run the index command again."
            )

    index = _ImageIndex(root, recursive, records, embeddings)
    saved_path = _save_index(output_dir, index, precision)
    return ImageIndexUpdate(saved_path, added, updated, removed, unchanged)


def _search_index(
    model: _ImageSearchModel,
    index_dir: str | os.PathLike[str],
    query: str,
    *,
    top_k: int = 5,
) -> list[ImageSearchResult]:
    """Search an index using the official EmbeddingGemma SearchQuery prompt."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be a positive integer.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    if not query.strip():
        raise ValueError("query must contain non-whitespace text.")

    device, precision = _model_settings(model)
    index = _load_index(index_dir, expected_precision=precision)
    if not index.records:
        raise ValueError("The image index is empty; run the index command first.")

    tokenizer = get_tokenizer(prompt_name="SearchQuery")
    tokens = tokenizer(query).to(device)
    with torch.inference_mode():
        query_features = model.encode_text(tokens, normalize=True)
    expected_shape = (1, _EMBEDDING_DIMENSION)
    if not isinstance(query_features, torch.Tensor) or tuple(query_features.shape) != expected_shape:
        actual_shape = (
            tuple(query_features.shape)
            if isinstance(query_features, torch.Tensor)
            else type(query_features).__name__
        )
        raise ImageIndexError(
            f"Text encoder returned an invalid query embedding: expected "
            f"{expected_shape}, received {actual_shape}."
        )
    query_vector = query_features.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
    query_norm = float(np.linalg.norm(query_vector))
    if not np.isfinite(query_vector).all() or not np.isfinite(query_norm) or query_norm <= 0:
        raise ImageIndexError("Text encoder returned an invalid query embedding.")
    query_vector = query_vector / query_norm

    limit = min(top_k, len(index.records))
    best: list[tuple[float, int]] = []

    def sort_key(item: tuple[float, int]) -> tuple[float, str, str]:
        relative_path = index.records[item[1]].relative_path
        return -item[0], relative_path.casefold(), relative_path

    for start in range(0, len(index.records), _SCORE_BLOCK_SIZE):
        stop = min(start + _SCORE_BLOCK_SIZE, len(index.records))
        scores = index.embeddings[start:stop] @ query_vector
        if not np.isfinite(scores).all():
            raise ImageIndexError("Image index produced non-finite similarity scores.")
        candidates = [
            (float(score), row + start)
            for row, score in enumerate(scores)
        ]
        best = sorted((*best, *candidates), key=sort_key)[:limit]

    return [
        ImageSearchResult(
            path=index.image_root / PurePosixPath(index.records[row].relative_path),
            score=score,
        )
        for score, row in best
    ]


def build_or_update_image_index(
    model: torch.nn.Module,
    preprocess: _ImagePreprocess,
    images_dir: str | os.PathLike[str],
    index_dir: str | os.PathLike[str],
    *,
    batch_size: int = 8,
    recursive: bool = False,
) -> ImageIndexUpdate:
    """Build or incrementally update a local image embedding index.

    The model's current device and precision are used. A corrupt image or
    incompatible index raises an error and leaves the previous index intact.
    """
    return _build_or_update_image_index(
        model,
        preprocess,
        images_dir,
        index_dir,
        batch_size=batch_size,
        recursive=recursive,
    )


def search_images(
    model: torch.nn.Module,
    index_dir: str | os.PathLike[str],
    query: str,
    *,
    top_k: int = 5,
) -> list[ImageSearchResult]:
    """Search an image index with a Japanese or English text query.

    Results are ordered by descending cosine similarity. Scores are not
    probabilities; equal scores are ordered by relative image path.
    """
    return _search_index(model, index_dir, query, top_k=top_k)
