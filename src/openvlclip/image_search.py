"""Internal image indexing and text-to-image search helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from openvlclip._dimensions import (
    DEFAULT_EMBEDDING_DIMENSION,
    EMBEDDING_STRATEGY,
    validate_positive_embedding_dimension,
)
from openvlclip.backends.base import validate_model_embedding_dimension
from openvlclip.batching import ImagePath, iter_image_embeddings
from openvlclip.constants import (
    BACKEND_ID,
    EMBEDDING_NORMALIZATION,
    IMAGE_PREPROCESSING_ID,
    MODEL_ID,
    MODEL_REVISION,
)
from openvlclip.embedding_space import (
    EmbeddingSpace,
    EmbeddingSpaceMetadata,
    has_embedding_space_metadata,
)
from openvlclip.index_storage import (
    IndexArchiveError,
    _array_sha256,
    _open_numpy_archive,
    _save_numpy_archive_atomic,
)
from openvlclip.registry import get_default_registry
from openvlclip.tokenizer import OFFICIAL_PROMPTS, get_tokenizer

_INDEX_FILENAME = "index.npz"
_INDEX_FORMAT_VERSION = 2
_LEGACY_INDEX_FORMAT_VERSION = 1
_PREPROCESSING_ID = IMAGE_PREPROCESSING_ID
_SUPPORTED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_SCORE_BLOCK_SIZE = 4096

__all__ = [
    "ImageIndexCompatibilityError",
    "ImageIndexError",
    "ImageIndexUpdate",
    "ImageSearchResult",
    "build_or_update_image_index",
    "convert_image_index",
    "search_images",
]

type _EmbeddingMatrix = NDArray[np.float32]
type _ImagePreprocess = Callable[[Image.Image], torch.Tensor]


class ImageIndexError(RuntimeError):
    """Raised when an image index cannot be built, read, or searched safely."""


class ImageIndexCompatibilityError(ImageIndexError):
    """Raised when an index was created for incompatible model settings."""


class _ImageSearchModel(Protocol):
    embedding_space_metadata: EmbeddingSpaceMetadata

    def parameters(self) -> Iterator[torch.nn.Parameter]: ...

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int: ...

    def encode_image(
        self,
        images: torch.Tensor,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor: ...

    def encode_text(
        self,
        tokens: torch.Tensor,
        *,
        normalize: bool = False,
        truncate_dim: int | None = None,
    ) -> torch.Tensor: ...


class _QueryTokenizer(Protocol):
    def __call__(self, text: str) -> torch.Tensor: ...


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
    precision: str
    embedding_space: EmbeddingSpace


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
        raise ImageIndexError(
            f"Could not scan image directory {root}: {error}"
        ) from error

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
        raise ImageIndexError(
            "The model has no parameters to determine its settings."
        ) from error

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


def _model_embedding_space(
    model: _ImageSearchModel,
    *,
    embedding_dimension: int,
    precision: str,
    sampling_method: str | None = None,
) -> EmbeddingSpace:
    metadata = getattr(model, "embedding_space_metadata", None)
    if not isinstance(metadata, EmbeddingSpaceMetadata):
        raise ImageIndexCompatibilityError(
            "The model does not expose registered embedding-space metadata."
        )
    try:
        embedding_dimension = validate_model_embedding_dimension(
            model,
            embedding_dimension,
            name="embedding_dimension",
        )
        return metadata.for_index(
            embedding_dimension=embedding_dimension,
            model_precision=precision,
            sampling_method=sampling_method,
        )
    except (TypeError, ValueError) as error:
        raise ImageIndexCompatibilityError(
            f"The model exposes invalid embedding-space metadata: {error}"
        ) from error


def _get_search_tokenizer(
    model: _ImageSearchModel,
    prompt_name: str,
    fallback: Callable[..., _QueryTokenizer],
) -> _QueryTokenizer:
    backend_tokenizer = getattr(model, "_create_tokenizer", None)
    if callable(backend_tokenizer):
        return backend_tokenizer(prompt_name=prompt_name)

    metadata = getattr(model, "embedding_space_metadata", None)
    if (
        isinstance(metadata, EmbeddingSpaceMetadata)
        and metadata.backend_id == BACKEND_ID
        and metadata.model_id == MODEL_ID
        and metadata.model_revision == MODEL_REVISION
        and metadata.query_prompt_name == prompt_name
        and metadata.query_prompt == OFFICIAL_PROMPTS.get(prompt_name)
    ):
        return fallback(prompt_name=prompt_name)
    raise ImageIndexCompatibilityError(
        "The model backend does not provide a tokenizer for the index query prompt."
    )


def _embedding_space_from_metadata(
    metadata: dict[str, object],
    *,
    embedding_dimension: int,
    precision: str,
) -> EmbeddingSpace:
    if has_embedding_space_metadata(metadata):
        try:
            embedding_space = EmbeddingSpace.from_metadata(metadata)
        except (TypeError, ValueError) as error:
            raise ImageIndexCompatibilityError(
                f"Image index embedding-space metadata is invalid: {error}"
            ) from error
        if embedding_space.embedding_strategy != EMBEDDING_STRATEGY:
            raise ImageIndexCompatibilityError(
                "Image index embedding_strategy is missing or unsupported."
            )
        return embedding_space

    compatibility_fields = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preprocessing_id": _PREPROCESSING_ID,
        "storage_dtype": "float32",
    }
    for name, expected in compatibility_fields.items():
        if metadata.get(name) != expected:
            raise ImageIndexCompatibilityError(
                f"Image index {name} does not match the current model configuration."
            )
    strategy = metadata.get("embedding_strategy", EMBEDDING_STRATEGY)
    if strategy != EMBEDDING_STRATEGY:
        raise ImageIndexCompatibilityError(
            "Image index embedding_strategy is missing or unsupported."
        )
    return EmbeddingSpace(
        backend_id=BACKEND_ID,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        embedding_dimension=embedding_dimension,
        embedding_strategy=EMBEDDING_STRATEGY,
        normalization=EMBEDDING_NORMALIZATION,
        preprocessing_id=_PREPROCESSING_ID,
        index_prompt=None,
        query_prompt_name="SearchQuery",
        query_prompt=OFFICIAL_PROMPTS["SearchQuery"],
        model_precision=precision,
    )


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
    if index.embedding_space.model_precision != precision:
        raise ImageIndexError(
            "Image index precision does not match its embedding space."
        )
    return {
        "format_version": _INDEX_FORMAT_VERSION,
        **index.embedding_space.to_metadata(),
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
    if embeddings.ndim != 2 or embeddings.shape[0] != len(index.records):
        raise ImageIndexError("Refusing to save an index with inconsistent dimensions.")
    try:
        validate_positive_embedding_dimension(
            embeddings.shape[1],
            name="embedding_dimension",
        )
    except (TypeError, ValueError) as error:
        raise ImageIndexError(
            "Refusing to save an index with an unsupported embedding dimension."
        ) from error
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
    expected_embedding_dimension: int | None = None,
    expected_embedding_space: EmbeddingSpace | None = None,
) -> _ImageIndex:
    index_path = Path(index_dir).expanduser() / _INDEX_FILENAME
    try:
        with _open_numpy_archive(
            index_path,
            expected_arrays=frozenset({"embeddings"}),
        ) as (metadata, archive):
            (
                records,
                image_root,
                recursive,
                embedding_dimension,
                embedding_space,
            ) = _validate_metadata(
                metadata,
                expected_root=expected_root,
                expected_precision=expected_precision,
            )
            embeddings = archive["embeddings"]
    except IndexArchiveError as error:
        if not index_path.exists():
            raise ImageIndexError(f"Image index not found: {index_path}") from error
        raise ImageIndexError(
            f"Could not read image index {index_path}: {error}"
        ) from error
    except ImageIndexError:
        raise

    if expected_embedding_dimension is not None:
        expected_embedding_dimension = validate_positive_embedding_dimension(
            expected_embedding_dimension,
            name="embedding_dim",
        )
        if embedding_dimension != expected_embedding_dimension:
            raise ImageIndexCompatibilityError(
                "Image index embedding dimension does not match the requested "
                f"dimension {expected_embedding_dimension}; convert it or use a new "
                "index directory."
            )
    if (
        expected_embedding_space is not None
        and embedding_space != expected_embedding_space
    ):
        raise ImageIndexCompatibilityError(
            "Image index embedding space does not match the requested model, "
            "revision, preprocessing, prompt, or precision."
        )
    expected_shape = (len(records), embedding_dimension)
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
        raise ImageIndexError(
            "Image index embedding checksum does not match its metadata."
        )
    return _ImageIndex(
        image_root,
        recursive,
        records,
        embeddings,
        embedding_space.model_precision,
        embedding_space,
    )


def _validate_metadata(
    metadata: dict[str, object],
    *,
    expected_root: Path | None,
    expected_precision: str | None,
) -> tuple[tuple[_ImageRecord, ...], Path, bool, int, EmbeddingSpace]:
    version = metadata.get("format_version")
    if type(version) is not int or version not in {
        _LEGACY_INDEX_FORMAT_VERSION,
        _INDEX_FORMAT_VERSION,
    }:
        raise ImageIndexCompatibilityError(
            f"Unsupported image index format version: {version!r}."
        )
    if (
        version == _INDEX_FORMAT_VERSION
        and metadata.get("embedding_strategy") != EMBEDDING_STRATEGY
    ):
        raise ImageIndexCompatibilityError(
            "Image index embedding_strategy is missing or unsupported."
        )
    try:
        embedding_dimension = validate_positive_embedding_dimension(
            metadata.get("embedding_dimension"),
            name="embedding_dimension",
        )
    except (TypeError, ValueError) as error:
        raise ImageIndexCompatibilityError(
            "Image index contains an unsupported embedding dimension."
        ) from error
    if (
        version == _LEGACY_INDEX_FORMAT_VERSION
        and embedding_dimension != DEFAULT_EMBEDDING_DIMENSION
    ):
        raise ImageIndexCompatibilityError(
            "Legacy image indexes must use the native 768-dimensional embeddings."
        )

    precision = metadata.get("model_precision")
    if not isinstance(precision, str) or precision not in {"fp32", "bf16"}:
        raise ImageIndexError(
            "Image index metadata contains an invalid model precision."
        )
    if expected_precision is not None and precision != expected_precision:
        raise ImageIndexCompatibilityError(
            "Image index model precision does not match the loaded model; "
            "rebuild the index using the same precision."
        )

    embedding_space = _embedding_space_from_metadata(
        metadata,
        embedding_dimension=embedding_dimension,
        precision=precision,
    )
    if (
        embedding_space.embedding_dimension != embedding_dimension
        or embedding_space.model_precision != precision
    ):
        raise ImageIndexCompatibilityError(
            "Image index embedding-space metadata conflicts with its legacy fields."
        )

    root_value = metadata.get("image_root")
    if not isinstance(root_value, str) or not root_value:
        raise ImageIndexError("Image index metadata contains an invalid image root.")
    image_root = Path(root_value)
    if expected_root is not None and os.path.normcase(
        str(image_root)
    ) != os.path.normcase(str(expected_root)):
        raise ImageIndexCompatibilityError(
            "Image index was created for a different image directory."
        )

    recursive = metadata.get("recursive")
    if not isinstance(recursive, bool):
        raise ImageIndexError(
            "Image index metadata contains an invalid recursion setting."
        )
    records_value = metadata.get("records")
    if not isinstance(records_value, list):
        raise ImageIndexError("Image index metadata contains an invalid record list.")
    records = tuple(_parse_record(record) for record in records_value)
    keys = [_path_key(record.relative_path) for record in records]
    if len(set(keys)) != len(keys):
        raise ImageIndexError("Image index metadata contains duplicate image paths.")
    if records != tuple(
        sorted(
            records,
            key=lambda record: (_path_key(record.relative_path), record.relative_path),
        )
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
        raise ImageIndexError(
            "Image index metadata contains an invalid embedding checksum."
        )
    return records, image_root, recursive, embedding_dimension, embedding_space


def _file_signature(path: Path) -> tuple[int, int]:
    try:
        file_stat = path.stat()
    except OSError as error:
        raise ImageIndexError(
            f"Could not inspect image file {path}: {error}"
        ) from error
    return file_stat.st_size, file_stat.st_mtime_ns


def _build_or_update_image_index(
    model: _ImageSearchModel,
    preprocess: _ImagePreprocess,
    images_dir: str | os.PathLike[str],
    index_dir: str | os.PathLike[str],
    *,
    batch_size: int = 8,
    recursive: bool = False,
    embedding_dim: int = DEFAULT_EMBEDDING_DIMENSION,
) -> ImageIndexUpdate:
    """Create or incrementally update a safe, local image index."""
    _validate_batch_size(batch_size)
    embedding_dim = validate_model_embedding_dimension(
        model,
        embedding_dim,
        name="embedding_dim",
    )
    root, records = _scan_image_files(images_dir, recursive=recursive)
    output_dir = Path(index_dir).expanduser()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Index output path is not a directory: {output_dir}")
    index_path = output_dir / _INDEX_FILENAME
    device, precision = _model_settings(model)
    del device
    embedding_space = _model_embedding_space(
        model,
        embedding_dimension=embedding_dim,
        precision=precision,
    )

    previous: _ImageIndex | None = None
    if index_path.exists():
        if not index_path.is_file():
            raise ImageIndexError(f"Image index path is not a file: {index_path}")
        previous = _load_index(
            output_dir,
            expected_root=root,
            expected_precision=precision,
            expected_embedding_dimension=embedding_dim,
            expected_embedding_space=embedding_space,
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
        sum(
            _path_key(record.relative_path) not in current_keys
            for record in previous.records
        )
        if previous is not None
        else 0
    )

    embeddings = np.empty((len(records), embedding_dim), dtype=np.float32)
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
        elif _record_signature(previous.records[previous_row]) == _record_signature(
            record
        ):
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
            embedding_dim=embedding_dim,
        ):
            key = _path_key(path.relative_to(root).as_posix())
            row = changed_rows.get(key)
            if row is None:
                raise ImageIndexError(
                    f"Image encoder returned an unexpected path: {path}"
                )
            vector = embedding.detach().to(device="cpu", dtype=torch.float32).numpy()
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 0:
                raise ImageIndexError(
                    f"Image encoder returned an invalid embedding for {path}."
                )
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

    index = _ImageIndex(
        root,
        recursive,
        records,
        embeddings,
        precision,
        embedding_space,
    )
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
    expected_embedding_space = _model_embedding_space(
        model,
        embedding_dimension=index.embeddings.shape[1],
        precision=precision,
    )
    if index.embedding_space != expected_embedding_space:
        raise ImageIndexCompatibilityError(
            "Image index embedding space does not match the search model, revision, "
            "preprocessing, prompt, or precision."
        )

    prompt_name = index.embedding_space.query_prompt_name
    if prompt_name is None:
        raise ImageIndexCompatibilityError(
            "Image index does not define a compatible text query prompt."
        )
    tokenizer = _get_search_tokenizer(model, prompt_name, get_tokenizer)
    tokens = tokenizer(query).to(device)
    with torch.inference_mode():
        embedding_dimension = index.embeddings.shape[1]
        if embedding_dimension == DEFAULT_EMBEDDING_DIMENSION:
            query_features = model.encode_text(tokens, normalize=True)
        else:
            query_features = model.encode_text(
                tokens,
                normalize=True,
                truncate_dim=embedding_dimension,
            )
    expected_shape = (1, embedding_dimension)
    if (
        not isinstance(query_features, torch.Tensor)
        or tuple(query_features.shape) != expected_shape
    ):
        actual_shape = (
            tuple(query_features.shape)
            if isinstance(query_features, torch.Tensor)
            else type(query_features).__name__
        )
        raise ImageIndexError(
            f"Text encoder returned an invalid query embedding: expected "
            f"{expected_shape}, received {actual_shape}."
        )
    query_vector = (
        query_features.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
    )
    query_norm = float(np.linalg.norm(query_vector))
    if (
        not np.isfinite(query_vector).all()
        or not np.isfinite(query_norm)
        or query_norm <= 0
    ):
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
        candidates = [(float(score), row + start) for row, score in enumerate(scores)]
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
    embedding_dim: int = DEFAULT_EMBEDDING_DIMENSION,
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
        embedding_dim=embedding_dim,
    )


def convert_image_index(
    index_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    embedding_dim: int,
) -> Path:
    """Convert an image index to a supported dimension without model inference."""
    embedding_dim = validate_positive_embedding_dimension(
        embedding_dim,
        name="embedding_dim",
    )
    index = _load_index(index_dir)
    try:
        backend = get_default_registry().resolve_backend_for_embedding_space(
            backend_id=index.embedding_space.backend_id,
            model_id=index.embedding_space.model_id,
            revision=index.embedding_space.model_revision,
        )
    except ValueError as error:
        raise ImageIndexCompatibilityError(
            "Image index conversion requires a registered backend matching its "
            "embedding-space identity."
        ) from error
    if (
        index.embedding_space.embedding_strategy
        != backend.embedding_space_metadata.embedding_strategy
    ):
        raise ImageIndexCompatibilityError(
            "Image index conversion is not supported for this embedding strategy."
        )
    source_dimension = index.embeddings.shape[1]
    try:
        validate_model_embedding_dimension(
            backend,
            source_dimension,
            name="source_embedding_dimension",
        )
    except (TypeError, ValueError) as error:
        raise ImageIndexCompatibilityError(
            "Image index source dimension is not supported by its registered backend."
        ) from error
    embedding_dim = validate_model_embedding_dimension(
        backend,
        embedding_dim,
        name="embedding_dim",
    )
    source_dir = Path(index_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise ValueError(f"Image index conversion output already exists: {destination}")
    if destination.resolve() == source_dir:
        raise ValueError("Image index conversion output must differ from its source.")

    if embedding_dim > source_dimension:
        raise ValueError(
            f"Cannot expand an image index from {source_dimension} to "
            f"{embedding_dim} dimensions."
        )
    truncated = np.ascontiguousarray(
        index.embeddings[:, :embedding_dim], dtype=np.float32
    )
    if len(index.records):
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ImageIndexError(
                "Image index contains an embedding with zero or non-finite norm "
                "after truncation."
            )
        truncated = np.ascontiguousarray(truncated / norms, dtype=np.float32)
    converted = _ImageIndex(
        index.image_root,
        index.recursive,
        index.records,
        truncated,
        index.precision,
        replace(index.embedding_space, embedding_dimension=embedding_dim),
    )
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(
            f"Image index conversion output already exists: {destination}"
        ) from error
    return _save_index(destination, converted, index.precision)


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
