"""Internal video-frame indexing and text search helpers."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from openvlclip._ranking import select_top_k
from openvlclip._dimensions import (
    DEFAULT_EMBEDDING_DIMENSION,
    EMBEDDING_STRATEGY,
    validate_positive_embedding_dimension,
)
from openvlclip.backends.base import (
    ProcessedImage,
    validate_model_embedding_dimension,
)
from openvlclip.batching import _iter_preprocessed_embeddings
from openvlclip.constants import (
    BACKEND_ID,
    EMBEDDING_NORMALIZATION,
    IMAGE_PREPROCESSING_ID,
    MODEL_ID,
    MODEL_REVISION,
)
from openvlclip.embedding_space import (
    EmbeddingSpace,
    embedding_spaces_compatible,
    has_embedding_space_metadata,
)
from openvlclip.image_search import (
    _PREPROCESSING_ID,
    ImageIndexCompatibilityError,
    ImageIndexError,
    _encode_search_query,
    _file_content_sha256,
    _ImageSearchModel,
    _model_embedding_space,
    _model_settings,
    _path_key,
    _policy_json,
    _validate_batch_size,
)
from openvlclip.index_storage import (
    IndexArchiveError,
    _array_sha256,
    _open_numpy_archive,
    _save_numpy_archive_atomic,
)
from openvlclip.registry import get_default_registry
from openvlclip.tokenizer import OFFICIAL_PROMPTS

_VIDEO_INDEX_FILENAME = "video-index.npz"
_VIDEO_INDEX_KIND = "openclip-gemma.video-index"
_VIDEO_INDEX_FORMAT_VERSION = 3
_LEGACY_VIDEO_INDEX_FORMAT_VERSION = 2
_NATIVE_VIDEO_INDEX_FILENAME = "native-video-index.npz"
_NATIVE_VIDEO_INDEX_KIND = "openvlclip.native-video-index"
_NATIVE_VIDEO_INDEX_FORMAT_VERSION = 1
_SAMPLING_METHOD = "first-presentation-frame-at-or-after-playback-origin-target-v2"
_DEFAULT_NATIVE_VIDEO_FPS = 1.0
_DEFAULT_NATIVE_VIDEO_MAX_FRAMES = 64
_NO_NATIVE_VIDEO_MAX_FRAMES = object()
_SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"})

type _EmbeddingMatrix = NDArray[np.float32]
type _ImagePreprocess = Callable[[Image.Image], ProcessedImage]

__all__ = [
    "NativeVideoIndexUpdate",
    "NativeVideoSearchResult",
    "VideoIndexCompatibilityError",
    "VideoIndexUpdate",
    "VideoSearchError",
    "VideoSearchResult",
    "build_or_update_native_video_index",
    "build_or_update_video_index",
    "convert_video_index",
    "search_native_video",
    "search_video",
]


class VideoSearchError(RuntimeError):
    """Raised when a video cannot be decoded, indexed, loaded, or searched."""


class VideoIndexCompatibilityError(VideoSearchError):
    """Raised when a saved video index does not match requested settings."""


@dataclass(frozen=True)
class _VideoRecord:
    relative_path: str
    size_bytes: int
    mtime_ns: int
    content_sha256: str | None = None


@dataclass(frozen=True)
class _FrameRecord:
    video_path: str
    timestamp_seconds: float
    pts: int
    time_base_numerator: int
    time_base_denominator: int
    timestamp_origin_numerator: int
    timestamp_origin_denominator: int


@dataclass(frozen=True)
class _SampledFrame:
    record: _FrameRecord
    image: Image.Image


@dataclass(frozen=True)
class _VideoIndex:
    video_root: Path
    recursive: bool
    interval_seconds: float
    videos: tuple[_VideoRecord, ...]
    frames: tuple[_FrameRecord, ...]
    embeddings: _EmbeddingMatrix
    precision: str
    embedding_space: EmbeddingSpace


@dataclass(frozen=True)
class VideoIndexUpdate:
    """Summary of video files added, updated, removed, or reused."""

    index_path: Path
    added_videos: int
    updated_videos: int
    removed_videos: int
    unchanged_videos: int
    frame_count: int


@dataclass(frozen=True)
class VideoSearchResult:
    """A matching frame with its playback-relative time and cosine similarity."""

    rank: int
    video_path: Path
    timestamp_seconds: float
    score: float


@dataclass(frozen=True)
class NativeVideoIndexUpdate:
    """Summary of videos added, updated, removed, or reused in a native index."""

    index_path: Path
    added_videos: int
    updated_videos: int
    removed_videos: int
    unchanged_videos: int


@dataclass(frozen=True)
class NativeVideoSearchResult:
    """A matching video and its cosine similarity, without a frame timestamp."""

    rank: int
    video_path: Path
    score: float


@dataclass(frozen=True)
class _NativeVideoIndex:
    video_root: Path
    recursive: bool
    fps: float
    max_frames: int
    videos: tuple[_VideoRecord, ...]
    embeddings: _EmbeddingMatrix
    precision: str
    embedding_space: EmbeddingSpace


def _load_av() -> Any:
    try:
        import av
    except ImportError as error:
        raise VideoSearchError(
            "Video indexing requires the optional PyAV dependency. "
            "Install it with `pip install openvlclip[video]`."
        ) from error
    return av


def _validate_interval(interval: float) -> tuple[float, Fraction]:
    if isinstance(interval, bool) or not isinstance(interval, (int, float)):
        raise TypeError("interval must be a finite positive number of seconds.")
    interval_seconds = float(interval)
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval must be a finite positive number of seconds.")
    return interval_seconds, Fraction(str(interval_seconds))


def _video_content_sha256(path: Path) -> str:
    try:
        return _file_content_sha256(path)
    except ImageIndexError as error:
        raise VideoSearchError(f"Could not hash video file {path}.") from error


def _scan_video_files(
    videos_dir: str | os.PathLike[str],
    *,
    recursive: bool,
    verify_contents: bool = False,
) -> tuple[Path, tuple[_VideoRecord, ...]]:
    if not isinstance(recursive, bool):
        raise TypeError("recursive must be a bool.")
    if not isinstance(verify_contents, bool):
        raise TypeError("verify_contents must be a bool.")
    try:
        root = Path(videos_dir).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Video directory does not exist: {videos_dir}") from error
    if not root.is_dir():
        raise ValueError(f"Video path is not a directory: {root}")

    candidates = root.rglob("*") if recursive else root.iterdir()
    records_by_key: dict[str, _VideoRecord] = {}
    try:
        for path in candidates:
            if (
                path.is_symlink()
                or path.suffix.lower() not in _SUPPORTED_VIDEO_SUFFIXES
            ):
                continue
            if not path.is_file():
                continue
            file_stat = path.stat()
            relative_path = path.relative_to(root).as_posix()
            key = _path_key(relative_path)
            records_by_key.setdefault(
                key,
                _VideoRecord(
                    relative_path,
                    file_stat.st_size,
                    file_stat.st_mtime_ns,
                    _video_content_sha256(path) if verify_contents else None,
                ),
            )
    except OSError as error:
        raise VideoSearchError(
            f"Could not scan video directory {root}: {error}"
        ) from error

    records = tuple(
        sorted(
            records_by_key.values(),
            key=lambda record: (_path_key(record.relative_path), record.relative_path),
        )
    )
    return root, records


def _fraction_time_base(value: Any, *, video_path: Path, frame_index: int) -> Fraction:
    if not value:
        raise VideoSearchError(
            f"Video frame {frame_index} in {video_path} has no usable PTS time base."
        )
    try:
        time_base = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise VideoSearchError(
            f"Video frame {frame_index} in {video_path} has an invalid PTS time base."
        ) from error
    if time_base <= 0:
        raise VideoSearchError(
            f"Video frame {frame_index} in {video_path} has a non-positive PTS time base."
        )
    return time_base


def _playback_time_origin(
    container: Any, av: Any, video_stream: Any
) -> Fraction | None:
    container_start = getattr(container, "start_time", None)
    if container_start is not None:
        av_time_base = getattr(av, "time_base", None)
        if (
            isinstance(container_start, bool)
            or not isinstance(container_start, int)
            or isinstance(av_time_base, bool)
            or not isinstance(av_time_base, int)
            or av_time_base <= 0
        ):
            raise VideoSearchError(
                "Video container has an invalid playback start time."
            )
        return Fraction(container_start, av_time_base)

    stream_start = getattr(video_stream, "start_time", None)
    stream_time_base = getattr(video_stream, "time_base", None)
    if stream_start is None:
        return None
    if isinstance(stream_start, bool) or not isinstance(stream_start, int):
        raise VideoSearchError("Video stream has an invalid start time.")
    if not stream_time_base:
        raise VideoSearchError("Video stream start time has no usable time base.")
    try:
        return Fraction(stream_start) * Fraction(stream_time_base)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise VideoSearchError(
            "Video stream start time has an invalid time base."
        ) from error


def _iter_sampled_frames(
    video_path: Path,
    interval: Fraction,
) -> Iterator[_SampledFrame]:
    """Decode one video lazily and sample using actual presentation timestamps."""
    av = _load_av()
    try:
        container = av.open(str(video_path), mode="r")
    except (OSError, ValueError, av.error.FFmpegError) as error:
        raise VideoSearchError(f"Could not open video {video_path}: {error}") from error

    decoded_count = 0
    sampled_count = 0
    try:
        video_streams = container.streams.video
        if not video_streams:
            raise VideoSearchError(f"Video contains no video stream: {video_path}")
        video_stream = video_streams[0]
        time_origin = _playback_time_origin(container, av, video_stream)
        next_target = Fraction(0, 1)
        previous_timestamp: Fraction | None = None

        try:
            for frame_index, frame in enumerate(container.decode(video=0)):
                decoded_count += 1
                if frame.pts is None or isinstance(frame.pts, bool):
                    raise VideoSearchError(
                        f"Video frame {frame_index} in {video_path} has no presentation timestamp."
                    )
                frame_time_base = frame.time_base
                if not frame_time_base:
                    frame_time_base = video_stream.time_base
                time_base = _fraction_time_base(
                    frame_time_base,
                    video_path=video_path,
                    frame_index=frame_index,
                )
                presentation_timestamp = Fraction(frame.pts) * time_base
                if (
                    previous_timestamp is not None
                    and presentation_timestamp < previous_timestamp
                ):
                    raise VideoSearchError(
                        f"Video presentation timestamps are not monotonic in {video_path} "
                        f"at frame {frame_index}."
                    )
                previous_timestamp = presentation_timestamp

                if time_origin is None:
                    time_origin = presentation_timestamp
                timestamp = presentation_timestamp - time_origin
                if timestamp < 0:
                    continue
                if timestamp < next_target:
                    continue

                try:
                    decoded_image = frame.to_image()
                    try:
                        image = decoded_image.convert("RGB")
                    finally:
                        decoded_image.close()
                except (OSError, ValueError, av.error.FFmpegError) as error:
                    raise VideoSearchError(
                        f"Could not convert video frame {frame_index} from "
                        f"{video_path} to RGB: {error}"
                    ) from error

                yield _SampledFrame(
                    _FrameRecord(
                        video_path="",
                        timestamp_seconds=float(timestamp),
                        pts=frame.pts,
                        time_base_numerator=time_base.numerator,
                        time_base_denominator=time_base.denominator,
                        timestamp_origin_numerator=time_origin.numerator,
                        timestamp_origin_denominator=time_origin.denominator,
                    ),
                    image,
                )
                sampled_count += 1
                if next_target <= timestamp:
                    steps = (timestamp - next_target) // interval + 1
                    next_target += steps * interval
        except VideoSearchError:
            raise
        except (OSError, ValueError, av.error.FFmpegError) as error:
            raise VideoSearchError(
                f"Could not decode video {video_path}: {error}"
            ) from error

        if decoded_count == 0 or sampled_count == 0:
            raise VideoSearchError(
                f"No video frames could be sampled from {video_path}."
            )
    finally:
        container.close()


def _iter_frame_tensors(
    video_path: Path,
    relative_path: str,
    interval: Fraction,
    preprocess: _ImagePreprocess,
) -> Iterator[tuple[_FrameRecord, ProcessedImage]]:
    for sampled_frame in _iter_sampled_frames(video_path, interval):
        try:
            processed_image = preprocess(sampled_frame.image)
            if processed_image is sampled_frame.image:
                processed_image = sampled_frame.image.copy()
        finally:
            sampled_frame.image.close()
        frame_record = _FrameRecord(
            video_path=relative_path,
            timestamp_seconds=sampled_frame.record.timestamp_seconds,
            pts=sampled_frame.record.pts,
            time_base_numerator=sampled_frame.record.time_base_numerator,
            time_base_denominator=sampled_frame.record.time_base_denominator,
            timestamp_origin_numerator=sampled_frame.record.timestamp_origin_numerator,
            timestamp_origin_denominator=sampled_frame.record.timestamp_origin_denominator,
        )
        yield frame_record, processed_image


def _frame_sort_key(record: _FrameRecord) -> tuple[str, float, str, int]:
    return (
        _path_key(record.video_path),
        record.timestamp_seconds,
        record.video_path,
        record.pts,
    )


def _video_embedding_space(
    model: _ImageSearchModel,
    *,
    embedding_dimension: int,
    precision: str,
    interval_seconds: float,
    query_instruction: str | None = None,
) -> EmbeddingSpace:
    try:
        return _model_embedding_space(
            model,
            embedding_dimension=embedding_dimension,
            precision=precision,
            sampling_method=_SAMPLING_METHOD,
            input_type="sampled-frame",
            query_instruction=query_instruction,
            processing_parameters=_video_processing_parameters(interval_seconds),
        )
    except ImageIndexCompatibilityError as error:
        raise VideoIndexCompatibilityError(str(error)) from error


def _video_processing_parameters(interval_seconds: float) -> str:
    return json.dumps(
        {"interval_seconds": interval_seconds},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _embedding_space_from_video_metadata(
    metadata: dict[str, Any],
    *,
    embedding_dimension: int,
    precision: str,
) -> EmbeddingSpace:
    if has_embedding_space_metadata(metadata):
        backend_id = metadata.get("backend_id")
        model_id = metadata.get("model_id")
        model_revision = metadata.get("model_revision")
        if not all(isinstance(value, str) for value in (backend_id, model_id, model_revision)):
            raise VideoIndexCompatibilityError(
                "Video index embedding-space identity is invalid."
            )
        try:
            backend = get_default_registry().resolve_backend_for_embedding_space(
                backend_id=backend_id,
                model_id=model_id,
                revision=model_revision,
            )
        except ValueError as error:
            raise VideoIndexCompatibilityError(
                "Video index model identity is not registered."
            ) from error
        expected_metadata = backend.embedding_space_metadata
        if metadata.get("embedding_strategy") != expected_metadata.embedding_strategy:
            raise VideoIndexCompatibilityError(
                "Video index embedding_strategy is missing or unsupported."
            )
        try:
            embedding_space = EmbeddingSpace.from_metadata(metadata)
        except (TypeError, ValueError) as error:
            raise VideoIndexCompatibilityError(
                f"Video index embedding-space metadata is invalid: {error}"
            ) from error
        expected_space = expected_metadata.for_index(
            embedding_dimension=embedding_dimension,
            model_precision=precision,
            storage_dtype=embedding_space.storage_dtype,
            sampling_method=_SAMPLING_METHOD,
        )
        embedding_space_core = replace(
            embedding_space,
            schema_version=1,
            input_type=None,
            index_policy=None,
            query_policy=None,
            processing_parameters=None,
        )
        if embedding_space_core != expected_space:
            raise VideoIndexCompatibilityError(
                "Video index embedding-space metadata does not match its registered "
                "model configuration."
            )
        if embedding_space.sampling_method != _SAMPLING_METHOD:
            raise VideoIndexCompatibilityError(
                "Video index sampling_method does not match the current configuration."
            )
        if embedding_space.schema_version == 2:
            capabilities = getattr(backend, "capabilities", {})
            instruction_aware = (
                isinstance(capabilities, Mapping)
                and capabilities.get("instruction_aware") is True
            )
            expected_index_policy = (
                _policy_json(
                    {
                        "input_type": "sampled-frame",
                        "instruction": "backend-default",
                    }
                )
                if instruction_aware
                else None
            )
            if (
                embedding_space.input_type != "sampled-frame"
                or embedding_space.index_policy != expected_index_policy
                or (
                    instruction_aware
                    and embedding_space.query_policy is None
                )
                or (
                    not instruction_aware
                    and embedding_space.query_policy is not None
                )
            ):
                raise VideoIndexCompatibilityError(
                    "Video index input or instruction policy is unsupported."
                )
        return embedding_space

    for name, expected in (
        ("model_id", MODEL_ID),
        ("model_revision", MODEL_REVISION),
        ("preprocessing_id", _PREPROCESSING_ID),
        ("storage_dtype", "float32"),
        ("sampling_method", _SAMPLING_METHOD),
    ):
        if metadata.get(name) != expected:
            raise VideoIndexCompatibilityError(
                f"Video index {name} does not match the current configuration."
            )
    strategy = metadata.get("embedding_strategy", EMBEDDING_STRATEGY)
    if strategy != EMBEDDING_STRATEGY:
        raise VideoIndexCompatibilityError(
            "Video index embedding_strategy is missing or unsupported."
        )
    return EmbeddingSpace(
        backend_id=BACKEND_ID,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        embedding_dimension=embedding_dimension,
        embedding_strategy=EMBEDDING_STRATEGY,
        normalization=EMBEDDING_NORMALIZATION,
        preprocessing_id=IMAGE_PREPROCESSING_ID,
        index_prompt=None,
        query_prompt_name="SearchQuery",
        query_prompt=OFFICIAL_PROMPTS["SearchQuery"],
        model_precision=precision,
        sampling_method=_SAMPLING_METHOD,
    )


def _metadata_for_index(index: _VideoIndex, precision: str) -> dict[str, object]:
    if index.embedding_space.model_precision != precision:
        raise VideoSearchError(
            "Video index precision does not match its embedding space."
        )
    return {
        "kind": _VIDEO_INDEX_KIND,
        "format_version": _VIDEO_INDEX_FORMAT_VERSION,
        **index.embedding_space.to_metadata(),
        "video_root": str(index.video_root),
        "recursive": index.recursive,
        "interval_seconds": index.interval_seconds,
        "sampling_method": _SAMPLING_METHOD,
        "video_count": len(index.videos),
        "frame_count": len(index.frames),
        "videos": [_video_record_metadata(video) for video in index.videos],
        "frames": [
            {
                "video_path": frame.video_path,
                "timestamp_seconds": frame.timestamp_seconds,
                "pts": frame.pts,
                "time_base_numerator": frame.time_base_numerator,
                "time_base_denominator": frame.time_base_denominator,
                "timestamp_origin_numerator": frame.timestamp_origin_numerator,
                "timestamp_origin_denominator": frame.timestamp_origin_denominator,
            }
            for frame in index.frames
        ],
        "embedding_sha256": _array_sha256(index.embeddings),
    }


def _save_video_index(
    index_dir: Path,
    index: _VideoIndex,
    precision: str,
) -> Path:
    index_path = index_dir / _VIDEO_INDEX_FILENAME
    if index.embeddings.ndim != 2 or index.embeddings.shape[0] != len(index.frames):
        raise VideoSearchError(
            "Refusing to save a video index with inconsistent dimensions."
        )
    try:
        validate_positive_embedding_dimension(
            index.embeddings.shape[1],
            name="embedding_dimension",
        )
    except (TypeError, ValueError) as error:
        raise VideoSearchError(
            "Refusing to save a video index with an unsupported embedding dimension."
        ) from error
    if index.embeddings.dtype != np.float32 or not np.isfinite(index.embeddings).all():
        raise VideoSearchError("Refusing to save invalid video frame embeddings.")
    try:
        return _save_numpy_archive_atomic(
            index_path,
            _metadata_for_index(index, precision),
            {"embeddings": index.embeddings},
        )
    except IndexArchiveError as error:
        raise VideoSearchError(str(error)) from error


def _parse_video_record(value: object) -> _VideoRecord:
    if not isinstance(value, dict):
        raise VideoSearchError("Video index metadata contains an invalid video record.")
    relative_path = _validate_video_relative_path(value.get("path"))
    size_bytes = value.get("size_bytes")
    mtime_ns = value.get("mtime_ns")
    content_sha256 = value.get("content_sha256")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
        or isinstance(mtime_ns, bool)
        or not isinstance(mtime_ns, int)
    ):
        raise VideoSearchError(
            "Video index metadata contains invalid video file information."
        )
    if content_sha256 is not None and (
        not isinstance(content_sha256, str)
        or len(content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in content_sha256)
    ):
        raise VideoSearchError(
            "Video index metadata contains an invalid video content SHA-256."
        )
    if PurePosixPath(relative_path).suffix.lower() not in _SUPPORTED_VIDEO_SUFFIXES:
        raise VideoSearchError(
            "Video index metadata contains an unsupported video extension."
        )
    return _VideoRecord(relative_path, size_bytes, mtime_ns, content_sha256)


def _video_record_metadata(record: _VideoRecord) -> dict[str, object]:
    metadata: dict[str, object] = {
        "path": record.relative_path,
        "size_bytes": record.size_bytes,
        "mtime_ns": record.mtime_ns,
    }
    if record.content_sha256 is not None:
        metadata["content_sha256"] = record.content_sha256
    return metadata


def _validate_video_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise VideoSearchError("Video index metadata contains an invalid video path.")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or (len(path.parts[0]) == 2 and path.parts[0][1] == ":")
        or path.as_posix() != value
    ):
        raise VideoSearchError(
            "Video index metadata contains a non-relative video path."
        )
    return value


def _parse_frame_record(value: object, video_paths: set[str]) -> _FrameRecord:
    if not isinstance(value, dict):
        raise VideoSearchError("Video index metadata contains an invalid frame record.")
    video_path = _validate_video_relative_path(value.get("video_path"))
    if video_path not in video_paths:
        raise VideoSearchError("Video frame metadata references an unknown video.")
    timestamp = value.get("timestamp_seconds")
    pts = value.get("pts")
    time_base_numerator = value.get("time_base_numerator")
    time_base_denominator = value.get("time_base_denominator")
    timestamp_origin_numerator = value.get("timestamp_origin_numerator")
    timestamp_origin_denominator = value.get("timestamp_origin_denominator")
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (float, int))
        or not math.isfinite(float(timestamp))
        or float(timestamp) < 0
        or isinstance(pts, bool)
        or not isinstance(pts, int)
        or isinstance(time_base_numerator, bool)
        or not isinstance(time_base_numerator, int)
        or time_base_numerator <= 0
        or isinstance(time_base_denominator, bool)
        or not isinstance(time_base_denominator, int)
        or time_base_denominator <= 0
        or isinstance(timestamp_origin_numerator, bool)
        or not isinstance(timestamp_origin_numerator, int)
        or isinstance(timestamp_origin_denominator, bool)
        or not isinstance(timestamp_origin_denominator, int)
        or timestamp_origin_denominator <= 0
    ):
        raise VideoSearchError("Video index metadata contains invalid frame timing.")
    exact_timestamp = Fraction(
        pts * time_base_numerator,
        time_base_denominator,
    ) - Fraction(timestamp_origin_numerator, timestamp_origin_denominator)
    if not math.isclose(
        float(timestamp), float(exact_timestamp), rel_tol=1e-12, abs_tol=1e-9
    ):
        raise VideoSearchError(
            "Video frame timestamp does not match its PTS and time base."
        )
    return _FrameRecord(
        video_path,
        float(timestamp),
        pts,
        time_base_numerator,
        time_base_denominator,
        timestamp_origin_numerator,
        timestamp_origin_denominator,
    )


def _validate_video_metadata(
    metadata: dict[str, Any],
    *,
    expected_root: Path | None,
    expected_precision: str | None,
    expected_interval: float | None,
) -> tuple[
    Path,
    bool,
    float,
    tuple[_VideoRecord, ...],
    tuple[_FrameRecord, ...],
    int,
    EmbeddingSpace,
]:
    if metadata.get("kind") != _VIDEO_INDEX_KIND:
        raise VideoIndexCompatibilityError(
            "The archive is not an OpenVLCLIP video index."
        )
    version = metadata.get("format_version")
    if type(version) is not int or version not in {
        _LEGACY_VIDEO_INDEX_FORMAT_VERSION,
        _VIDEO_INDEX_FORMAT_VERSION,
    }:
        raise VideoIndexCompatibilityError(
            f"Unsupported video index format version: {version!r}."
        )
    if (
        version == _VIDEO_INDEX_FORMAT_VERSION
        and not isinstance(metadata.get("embedding_strategy"), str)
    ):
        raise VideoIndexCompatibilityError(
            "Video index embedding_strategy is missing or unsupported."
        )
    try:
        embedding_dimension = validate_positive_embedding_dimension(
            metadata.get("embedding_dimension"),
            name="embedding_dimension",
        )
    except (TypeError, ValueError) as error:
        raise VideoIndexCompatibilityError(
            "Video index contains an unsupported embedding dimension."
        ) from error
    if (
        version == _LEGACY_VIDEO_INDEX_FORMAT_VERSION
        and embedding_dimension != DEFAULT_EMBEDDING_DIMENSION
    ):
        raise VideoIndexCompatibilityError(
            "Legacy video indexes must use the native 768-dimensional embeddings."
        )

    precision = metadata.get("model_precision")
    if not isinstance(precision, str) or precision not in {"fp32", "bf16"}:
        raise VideoSearchError(
            "Video index metadata contains an invalid model precision."
        )
    if expected_precision is not None and precision != expected_precision:
        raise VideoIndexCompatibilityError(
            "Video index model precision does not match the loaded model."
        )

    embedding_space = _embedding_space_from_video_metadata(
        metadata,
        embedding_dimension=embedding_dimension,
        precision=precision,
    )
    if (
        embedding_space.embedding_dimension != embedding_dimension
        or embedding_space.model_precision != precision
    ):
        raise VideoIndexCompatibilityError(
            "Video index embedding-space metadata conflicts with its legacy fields."
        )

    interval = metadata.get("interval_seconds")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or float(interval) <= 0
    ):
        raise VideoSearchError(
            "Video index metadata contains an invalid sampling interval."
        )
    interval_seconds = float(interval)
    if (
        embedding_space.schema_version == 2
        and embedding_space.processing_parameters
        != _video_processing_parameters(interval_seconds)
    ):
        raise VideoIndexCompatibilityError(
            "Video index sampling parameters do not match its embedding space."
        )
    if expected_interval is not None and interval_seconds != expected_interval:
        raise VideoIndexCompatibilityError(
            "Video index sampling interval differs from the requested interval; "
            "use a new index directory or remove the old video index to rebuild."
        )

    root_value = metadata.get("video_root")
    if not isinstance(root_value, str) or not root_value:
        raise VideoSearchError("Video index metadata contains an invalid video root.")
    video_root = Path(root_value)
    if expected_root is not None and os.path.normcase(
        str(video_root)
    ) != os.path.normcase(str(expected_root)):
        raise VideoIndexCompatibilityError(
            "Video index was created for a different video directory."
        )

    recursive = metadata.get("recursive")
    if not isinstance(recursive, bool):
        raise VideoSearchError(
            "Video index metadata contains an invalid recursion setting."
        )
    videos_value = metadata.get("videos")
    frames_value = metadata.get("frames")
    if not isinstance(videos_value, list) or not isinstance(frames_value, list):
        raise VideoSearchError(
            "Video index metadata contains invalid video or frame lists."
        )
    videos = tuple(_parse_video_record(record) for record in videos_value)
    video_keys = [_path_key(video.relative_path) for video in videos]
    if len(set(video_keys)) != len(video_keys):
        raise VideoSearchError("Video index metadata contains duplicate video paths.")
    if videos != tuple(
        sorted(
            videos,
            key=lambda video: (_path_key(video.relative_path), video.relative_path),
        )
    ):
        raise VideoSearchError("Video index records are not in canonical path order.")
    video_paths = {video.relative_path for video in videos}
    frames = tuple(_parse_frame_record(record, video_paths) for record in frames_value)
    frame_keys = [
        (
            frame.video_path,
            frame.pts,
            frame.time_base_numerator,
            frame.time_base_denominator,
        )
        for frame in frames
    ]
    if len(set(frame_keys)) != len(frame_keys):
        raise VideoSearchError(
            "Video index metadata contains duplicate frame timestamps."
        )
    if frames != tuple(sorted(frames, key=_frame_sort_key)):
        raise VideoSearchError(
            "Video frame records are not in canonical path/time order."
        )
    origins_by_video: dict[str, tuple[int, int]] = {}
    for frame in frames:
        origin = (
            frame.timestamp_origin_numerator,
            frame.timestamp_origin_denominator,
        )
        previous_origin = origins_by_video.setdefault(frame.video_path, origin)
        if previous_origin != origin:
            raise VideoSearchError(
                "Video frames contain inconsistent playback time origins."
            )
    frame_video_paths = {frame.video_path for frame in frames}
    if any(video.relative_path not in frame_video_paths for video in videos):
        raise VideoSearchError("Video index contains a video without sampled frames.")

    video_count = metadata.get("video_count")
    frame_count = metadata.get("frame_count")
    if (
        isinstance(video_count, bool)
        or not isinstance(video_count, int)
        or video_count != len(videos)
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count != len(frames)
    ):
        raise VideoSearchError("Video index counts do not match its metadata.")
    checksum = metadata.get("embedding_sha256")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise VideoSearchError(
            "Video index metadata contains an invalid embedding checksum."
        )
    return (
        video_root,
        recursive,
        interval_seconds,
        videos,
        frames,
        embedding_dimension,
        embedding_space,
    )


def _load_video_index(
    index_dir: str | os.PathLike[str],
    *,
    expected_root: Path | None = None,
    expected_precision: str | None = None,
    expected_interval: float | None = None,
    expected_embedding_dimension: int | None = None,
    expected_embedding_space: EmbeddingSpace | None = None,
) -> _VideoIndex:
    index_path = Path(index_dir).expanduser() / _VIDEO_INDEX_FILENAME
    try:
        with _open_numpy_archive(
            index_path,
            expected_arrays=frozenset({"embeddings"}),
        ) as (metadata, archive):
            (
                video_root,
                recursive,
                interval,
                videos,
                frames,
                embedding_dimension,
                embedding_space,
            ) = _validate_video_metadata(
                metadata,
                expected_root=expected_root,
                expected_precision=expected_precision,
                expected_interval=expected_interval,
            )
            embeddings = archive["embeddings"]
    except IndexArchiveError as error:
        if not index_path.exists():
            raise VideoSearchError(f"Video index not found: {index_path}") from error
        raise VideoSearchError(
            f"Could not read video index {index_path}: {error}"
        ) from error
    if expected_embedding_dimension is not None:
        expected_embedding_dimension = validate_positive_embedding_dimension(
            expected_embedding_dimension,
            name="embedding_dim",
        )
        if embedding_dimension != expected_embedding_dimension:
            raise VideoIndexCompatibilityError(
                "Video index embedding dimension does not match the requested "
                f"dimension {expected_embedding_dimension}; convert it or use a new "
                "index directory."
            )
    if expected_embedding_space is not None and not embedding_spaces_compatible(
        embedding_space,
        expected_embedding_space,
        legacy_input_type="sampled-frame",
        legacy_processing_parameters=_video_processing_parameters(interval),
    ):
        raise VideoIndexCompatibilityError(
            "Video index embedding space does not match the requested model, "
            "revision, preprocessing, prompt, sampling, or precision."
        )
    expected_shape = (len(frames), embedding_dimension)
    if embeddings.dtype != np.float32 or embeddings.shape != expected_shape:
        raise VideoSearchError(
            "Video index embeddings have an invalid dtype or shape: "
            f"expected float32{expected_shape}, received {embeddings.dtype}{embeddings.shape}."
        )
    if not np.isfinite(embeddings).all():
        raise VideoSearchError("Video index contains non-finite embeddings.")
    if _array_sha256(embeddings) != metadata["embedding_sha256"]:
        raise VideoSearchError(
            "Video index embedding checksum does not match its metadata."
        )
    return _VideoIndex(
        video_root,
        recursive,
        interval,
        videos,
        frames,
        embeddings,
        embedding_space.model_precision,
        embedding_space,
    )


def _file_signature(path: Path) -> tuple[int, int]:
    try:
        file_stat = path.stat()
    except OSError as error:
        raise VideoSearchError(
            f"Could not inspect video file {path}: {error}"
        ) from error
    return file_stat.st_size, file_stat.st_mtime_ns


def _embedding_vector(embedding: torch.Tensor, video_path: Path) -> np.ndarray:
    vector = embedding.detach().to(device="cpu", dtype=torch.float32).numpy()
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
        raise VideoSearchError(
            f"Image encoder returned an invalid frame embedding for {video_path}."
        )
    return vector / norm


def _embed_video(
    model: _ImageSearchModel,
    preprocess: _ImagePreprocess,
    video_path: Path,
    relative_path: str,
    interval: Fraction,
    batch_size: int,
    embedding_dim: int,
) -> Iterator[tuple[_FrameRecord, np.ndarray]]:
    frame_tensors = _iter_frame_tensors(
        video_path,
        relative_path,
        interval,
        preprocess,
    )
    embeddings = _iter_preprocessed_embeddings(
        model,
        frame_tensors,
        batch_size=batch_size,
        item_kind="video frames",
        embedding_dim=embedding_dim,
    )
    for frame_record, embedding in embeddings:
        yield frame_record, _embedding_vector(embedding, video_path)


def _build_or_update_video_index(
    model: _ImageSearchModel,
    preprocess: _ImagePreprocess,
    videos_dir: str | os.PathLike[str],
    index_dir: str | os.PathLike[str],
    *,
    interval: float = 1.0,
    batch_size: int = 8,
    recursive: bool = False,
    embedding_dim: int | None = None,
    query_instruction: str | None = None,
) -> VideoIndexUpdate:
    """Create or incrementally update a versioned index of sampled video frames."""
    interval_seconds, interval_fraction = _validate_interval(interval)
    _validate_batch_size(batch_size)
    embedding_dim = validate_model_embedding_dimension(
        model,
        embedding_dim,
        name="embedding_dim",
    )
    root, videos = _scan_video_files(videos_dir, recursive=recursive)
    output_dir = Path(index_dir).expanduser()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Video index output path is not a directory: {output_dir}")
    index_path = output_dir / _VIDEO_INDEX_FILENAME
    _, precision = _model_settings(model)
    embedding_space = _video_embedding_space(
        model,
        embedding_dimension=embedding_dim,
        precision=precision,
        interval_seconds=interval_seconds,
        query_instruction=query_instruction,
    )

    previous: _VideoIndex | None = None
    if index_path.exists():
        if not index_path.is_file():
            raise VideoSearchError(f"Video index path is not a file: {index_path}")
        previous = _load_video_index(
            output_dir,
            expected_root=root,
            expected_precision=precision,
            expected_interval=interval_seconds,
            expected_embedding_dimension=embedding_dim,
            expected_embedding_space=embedding_space,
        )
    elif not videos:
        raise ValueError(f"No supported video files were found in {root}.")

    previous_by_path = (
        {video.relative_path: video for video in previous.videos}
        if previous is not None
        else {}
    )
    previous_frames: dict[str, list[tuple[int, _FrameRecord]]] = {}
    if previous is not None:
        for row, frame in enumerate(previous.frames):
            previous_frames.setdefault(frame.video_path, []).append((row, frame))
    current_paths = {_path_key(video.relative_path) for video in videos}
    removed = (
        sum(
            _path_key(video.relative_path) not in current_paths
            for video in previous.videos
        )
        if previous is not None
        else 0
    )

    frame_records: list[_FrameRecord] = []
    embedding_rows: list[np.ndarray] = []
    added = 0
    updated = 0
    unchanged = 0
    for video in videos:
        previous_video = previous_by_path.get(video.relative_path)
        if (
            previous is not None
            and previous_video is not None
            and (previous_video.size_bytes, previous_video.mtime_ns)
            == (video.size_bytes, video.mtime_ns)
        ):
            unchanged += 1
            for row, frame in previous_frames.get(video.relative_path, []):
                frame_records.append(frame)
                embedding_rows.append(previous.embeddings[row])
            continue

        if previous_video is None:
            added += 1
        else:
            updated += 1
        video_path = root / PurePosixPath(video.relative_path)
        new_frame_count = 0
        for frame_record, embedding in _embed_video(
            model,
            preprocess,
            video_path,
            video.relative_path,
            interval_fraction,
            batch_size,
            embedding_dim,
        ):
            frame_records.append(frame_record)
            embedding_rows.append(embedding)
            new_frame_count += 1
        if new_frame_count == 0:
            raise VideoSearchError(f"No frames were embedded from {video_path}.")

    for video in videos:
        video_path = root / PurePosixPath(video.relative_path)
        if _file_signature(video_path) != (video.size_bytes, video.mtime_ns):
            raise VideoSearchError(
                f"Video changed during indexing: {video_path}. Run the index command again."
            )

    if (
        previous is not None
        and previous.recursive == recursive
        and added == 0
        and updated == 0
        and removed == 0
    ):
        return VideoIndexUpdate(
            index_path, added, updated, removed, unchanged, len(frame_records)
        )

    embeddings = (
        np.stack(embedding_rows).astype(np.float32, copy=False)
        if embedding_rows
        else np.empty((0, embedding_dim), dtype=np.float32)
    )
    index = _VideoIndex(
        root,
        recursive,
        interval_seconds,
        videos,
        tuple(frame_records),
        embeddings,
        precision,
        embedding_space,
    )
    saved_path = _save_video_index(output_dir, index, precision)
    return VideoIndexUpdate(
        saved_path, added, updated, removed, unchanged, len(frame_records)
    )


def _search_video_index(
    model: _ImageSearchModel,
    index_dir: str | os.PathLike[str],
    query: str,
    *,
    top_k: int = 5,
    query_instruction: str | None = None,
) -> list[VideoSearchResult]:
    """Search sampled frames using normalized cosine similarity."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be a positive integer.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    if not query.strip():
        raise ValueError("query must contain non-whitespace text.")

    device, precision = _model_settings(model)
    index = _load_video_index(index_dir, expected_precision=precision)
    if not index.frames:
        raise ValueError("The video index is empty; run the index command first.")
    expected_embedding_space = _video_embedding_space(
        model,
        embedding_dimension=index.embeddings.shape[1],
        precision=precision,
        interval_seconds=index.interval_seconds,
        query_instruction=query_instruction,
    )
    if not embedding_spaces_compatible(
        index.embedding_space,
        expected_embedding_space,
        legacy_input_type="sampled-frame",
        legacy_processing_parameters=_video_processing_parameters(
            index.interval_seconds
        ),
    ):
        raise VideoIndexCompatibilityError(
            "Video index embedding space does not match the search model, revision, "
            "preprocessing, prompt, instruction policy, sampling, or precision."
        )

    query_features = _encode_search_query(
        model,
        query,
        device=device,
        embedding_dimension=index.embeddings.shape[1],
        embedding_space=index.embedding_space,
        query_instruction=query_instruction,
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
        raise VideoSearchError("Text encoder returned an invalid query embedding.")
    query_vector = query_vector / query_norm

    limit = min(top_k, len(index.frames))
    best: list[tuple[float, int]] = []

    def sort_key(item: tuple[float, int]) -> tuple[float, str, float, int]:
        frame = index.frames[item[1]]
        return (
            -item[0],
            frame.video_path.casefold(),
            frame.timestamp_seconds,
            frame.pts,
        )

    for start in range(0, len(index.frames), 4096):
        stop = min(start + 4096, len(index.frames))
        scores = index.embeddings[start:stop] @ query_vector
        if not np.isfinite(scores).all():
            raise VideoSearchError("Video index produced non-finite similarity scores.")
        best = select_top_k(
            scores, limit=limit, start_row=start, sort_key=sort_key, current=best
        )

    return [
        VideoSearchResult(
            rank=rank,
            video_path=index.video_root / PurePosixPath(index.frames[row].video_path),
            timestamp_seconds=index.frames[row].timestamp_seconds,
            score=score,
        )
        for rank, (score, row) in enumerate(best, start=1)
    ]


def build_or_update_video_index(
    model: torch.nn.Module,
    preprocess: _ImagePreprocess,
    videos_dir: str | os.PathLike[str],
    index_dir: str | os.PathLike[str],
    *,
    interval_seconds: float = 1.0,
    batch_size: int = 8,
    recursive: bool = False,
    embedding_dim: int | None = None,
    query_instruction: str | None = None,
) -> VideoIndexUpdate:
    """Build or incrementally update an index of sampled video frames.

    Requires the optional PyAV dependency. Timestamps in the index are relative
    to the container playback start, or the video stream start when unavailable.
    If ``embedding_dim`` is omitted, the model backend's native dimension is used.
    ``query_instruction`` fixes the text-query policy for instruction-aware models.
    """
    return _build_or_update_video_index(
        model,
        preprocess,
        videos_dir,
        index_dir,
        interval=interval_seconds,
        batch_size=batch_size,
        recursive=recursive,
        embedding_dim=embedding_dim,
        query_instruction=query_instruction,
    )


def convert_video_index(
    index_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    embedding_dim: int,
) -> Path:
    """Convert a video index to a supported dimension without model inference."""
    embedding_dim = validate_positive_embedding_dimension(
        embedding_dim,
        name="embedding_dim",
    )
    index = _load_video_index(index_dir)
    try:
        backend = get_default_registry().resolve_backend_for_embedding_space(
            backend_id=index.embedding_space.backend_id,
            model_id=index.embedding_space.model_id,
            revision=index.embedding_space.model_revision,
        )
    except ValueError as error:
        raise VideoIndexCompatibilityError(
            "Video index conversion requires a registered backend matching its "
            "embedding-space identity."
        ) from error
    if (
        index.embedding_space.embedding_strategy
        != backend.embedding_space_metadata.embedding_strategy
    ):
        raise VideoIndexCompatibilityError(
            "Video index conversion is not supported for this embedding strategy."
        )
    source_dimension = index.embeddings.shape[1]
    try:
        validate_model_embedding_dimension(
            backend,
            source_dimension,
            name="source_embedding_dimension",
        )
    except (TypeError, ValueError) as error:
        raise VideoIndexCompatibilityError(
            "Video index source dimension is not supported by its registered backend."
        ) from error
    embedding_dim = validate_model_embedding_dimension(
        backend,
        embedding_dim,
        name="embedding_dim",
    )
    source_dir = Path(index_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise ValueError(f"Video index conversion output already exists: {destination}")
    if destination.resolve() == source_dir:
        raise ValueError("Video index conversion output must differ from its source.")

    if embedding_dim > source_dimension:
        raise ValueError(
            f"Cannot expand a video index from {source_dimension} to "
            f"{embedding_dim} dimensions."
        )
    truncated = np.ascontiguousarray(
        index.embeddings[:, :embedding_dim], dtype=np.float32
    )
    if len(index.frames):
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise VideoSearchError(
                "Video index contains an embedding with zero or non-finite norm "
                "after truncation."
            )
        truncated = np.ascontiguousarray(truncated / norms, dtype=np.float32)
    converted = _VideoIndex(
        index.video_root,
        index.recursive,
        index.interval_seconds,
        index.videos,
        index.frames,
        truncated,
        index.precision,
        replace(index.embedding_space, embedding_dimension=embedding_dim),
    )
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(
            f"Video index conversion output already exists: {destination}"
        ) from error
    return _save_video_index(destination, converted, index.precision)


def search_video(
    model: torch.nn.Module,
    index_dir: str | os.PathLike[str],
    query: str,
    *,
    top_k: int = 5,
    query_instruction: str | None = None,
) -> list[VideoSearchResult]:
    """Search a video-frame index with a Japanese or English text query.

    Results are ordered by descending cosine similarity and include playback-
    relative timestamps in seconds. Scores are not probabilities.
    """
    return _search_video_index(
        model,
        index_dir,
        query,
        top_k=top_k,
        query_instruction=query_instruction,
    )


def _native_video_processing_parameters(
    model: object,
    fps: float,
    max_frames: int,
) -> str:
    processor_metadata = getattr(model, "native_video_processing_metadata", None)
    if not isinstance(processor_metadata, Mapping):
        raise VideoSearchError(
            "Native video indexing requires explicit processor version and "
            "total-pixel metadata from the backend."
        )
    processor_version = processor_metadata.get("processor_version")
    total_pixels = processor_metadata.get("total_pixels")
    if not isinstance(processor_version, str) or not processor_version:
        raise VideoSearchError(
            "Native video processor metadata must include a non-empty "
            "processor_version."
        )
    if (
        isinstance(total_pixels, bool)
        or not isinstance(total_pixels, (int, float))
    ):
        raise VideoSearchError(
            "Native video processor metadata must include a finite positive "
            "total_pixels value."
        )
    try:
        total_pixels_value = float(total_pixels)
    except OverflowError as error:
        raise VideoSearchError(
            "Native video processor metadata must include a finite positive "
            "total_pixels value."
        ) from error
    if not math.isfinite(total_pixels_value) or total_pixels_value <= 0:
        raise VideoSearchError(
            "Native video processor metadata must include a finite positive "
            "total_pixels value."
        )
    parameters: dict[str, object] = {
        "fps": fps,
        "max_frames": max_frames,
        "processor_version": processor_version,
        "total_pixels": total_pixels,
    }
    return json.dumps(
        parameters,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_native_video_processing_parameters(
    processing_parameters: str,
    *,
    fps: float,
    max_frames: int,
) -> None:
    try:
        parameters = json.loads(processing_parameters)
    except json.JSONDecodeError as error:
        raise VideoSearchError(
            "Native-video processing parameters are invalid JSON."
        ) from error
    if not isinstance(parameters, dict):
        raise VideoSearchError(
            "Native-video settings conflict with its embedding-space metadata."
        )
    parameter_fps = parameters.get("fps")
    parameter_max_frames = parameters.get("max_frames")
    if (
        isinstance(parameter_fps, bool)
        or not isinstance(parameter_fps, (int, float))
        or not math.isfinite(parameter_fps)
        or parameter_fps != fps
        or type(parameter_max_frames) is not int
        or parameter_max_frames != max_frames
    ):
        raise VideoSearchError(
            "Native-video settings conflict with its embedding-space metadata."
        )
    processor_version = parameters.get("processor_version")
    total_pixels = parameters.get("total_pixels")
    if not isinstance(processor_version, str) or not processor_version:
        raise VideoSearchError(
            "Native-video processing metadata has no valid processor version."
        )
    if (
        isinstance(total_pixels, bool)
        or not isinstance(total_pixels, (int, float))
    ):
        raise VideoSearchError(
            "Native-video processing metadata has an invalid total pixel budget."
        )
    try:
        total_pixels_value = float(total_pixels)
    except OverflowError as error:
        raise VideoSearchError(
            "Native-video processing metadata has an invalid total pixel budget."
        ) from error
    if not math.isfinite(total_pixels_value) or total_pixels_value <= 0:
        raise VideoSearchError(
            "Native-video processing metadata has an invalid total pixel budget."
        )


def _validate_native_video_settings(fps: object, max_frames: object) -> tuple[float, int]:
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise TypeError("fps must be a finite positive number.")
    fps_value = float(fps)
    if not math.isfinite(fps_value) or fps_value <= 0:
        raise ValueError("fps must be a finite positive number.")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int):
        raise TypeError("max_frames must be a positive integer.")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero.")
    return fps_value, max_frames


def _validate_native_video_model_frame_limit(model: object, max_frames: int) -> None:
    model_limit = getattr(
        model,
        "native_video_max_frames",
        _NO_NATIVE_VIDEO_MAX_FRAMES,
    )
    if model_limit is _NO_NATIVE_VIDEO_MAX_FRAMES:
        return
    if isinstance(model_limit, bool) or not isinstance(model_limit, int):
        raise TypeError("model.native_video_max_frames must be a positive integer.")
    if model_limit <= 0:
        raise ValueError("model.native_video_max_frames must be greater than zero.")
    if max_frames > model_limit:
        raise ValueError(
            "max_frames must not exceed the model's native video frame limit "
            f"({model_limit})."
        )


def _require_native_video_model(model: object) -> Callable[..., torch.Tensor]:
    capabilities = getattr(model, "capabilities", None)
    if (
        not isinstance(capabilities, Mapping)
        or capabilities.get("video_embedding") is not True
    ):
        raise NotImplementedError(
            "Native video indexing requires a backend with video_embedding capability."
        )
    encoder = getattr(model, "encode_video", None)
    if not callable(encoder):
        raise NotImplementedError(
            "Native video indexing requires the model's encode_video() API."
        )
    if not callable(getattr(model, "validate_embedding_dimension", None)):
        raise VideoSearchError(
            "Native video indexing requires a backend-native embedding dimension."
        )
    return encoder


def _native_video_embedding_space(
    model: _ImageSearchModel,
    *,
    embedding_dimension: int,
    precision: str,
    fps: float,
    max_frames: int,
    query_instruction: str | None,
) -> EmbeddingSpace:
    try:
        return _model_embedding_space(
            model,
            embedding_dimension=embedding_dimension,
            precision=precision,
            input_type="native-video",
            query_instruction=query_instruction,
            processing_parameters=_native_video_processing_parameters(
                model,
                fps,
                max_frames,
            ),
        )
    except ImageIndexCompatibilityError as error:
        raise VideoIndexCompatibilityError(str(error)) from error


def _metadata_for_native_video_index(
    index: _NativeVideoIndex,
) -> dict[str, object]:
    return {
        "kind": _NATIVE_VIDEO_INDEX_KIND,
        "format_version": _NATIVE_VIDEO_INDEX_FORMAT_VERSION,
        **index.embedding_space.to_metadata(),
        "video_root": str(index.video_root),
        "recursive": index.recursive,
        "fps": index.fps,
        "max_frames": index.max_frames,
        "video_count": len(index.videos),
        "videos": [_video_record_metadata(video) for video in index.videos],
        "embedding_sha256": _array_sha256(index.embeddings),
    }


def _save_native_video_index(
    index_dir: Path,
    index: _NativeVideoIndex,
) -> Path:
    if index.embedding_space.schema_version != 2 or (
        index.embedding_space.input_type != "native-video"
    ):
        raise VideoSearchError("Refusing to save an untyped native-video index.")
    if (
        index.embeddings.ndim != 2
        or index.embeddings.shape
        != (len(index.videos), index.embedding_space.embedding_dimension)
    ):
        raise VideoSearchError(
            "Refusing to save a native-video index with inconsistent dimensions."
        )
    if index.embeddings.dtype != np.float32 or not np.isfinite(index.embeddings).all():
        raise VideoSearchError("Refusing to save invalid native-video embeddings.")
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        return _save_numpy_archive_atomic(
            index_dir / _NATIVE_VIDEO_INDEX_FILENAME,
            _metadata_for_native_video_index(index),
            {"embeddings": index.embeddings},
        )
    except IndexArchiveError as error:
        raise VideoSearchError(str(error)) from error


def _load_native_video_index(
    index_dir: str | os.PathLike[str],
    *,
    expected_root: Path | None = None,
    expected_precision: str | None = None,
    expected_fps: float | None = None,
    expected_max_frames: int | None = None,
    expected_embedding_dimension: int | None = None,
    expected_embedding_space: EmbeddingSpace | None = None,
) -> _NativeVideoIndex:
    index_path = Path(index_dir).expanduser() / _NATIVE_VIDEO_INDEX_FILENAME
    try:
        with _open_numpy_archive(
            index_path,
            expected_arrays=frozenset({"embeddings"}),
        ) as (metadata, archive):
            if (
                metadata.get("kind") != _NATIVE_VIDEO_INDEX_KIND
                or type(metadata.get("format_version")) is not int
                or metadata.get("format_version") != _NATIVE_VIDEO_INDEX_FORMAT_VERSION
            ):
                raise VideoIndexCompatibilityError(
                    "The archive is not a supported native-video index."
                )
            try:
                embedding_space = EmbeddingSpace.from_metadata(metadata)
            except (TypeError, ValueError) as error:
                raise VideoSearchError(
                    f"Native-video embedding-space metadata is invalid: {error}"
                ) from error
            if (
                embedding_space.schema_version != 2
                or embedding_space.input_type != "native-video"
            ):
                raise VideoIndexCompatibilityError(
                    "Native-video index has an incompatible embedding-space schema."
                )
            try:
                dimension = validate_positive_embedding_dimension(
                    metadata.get("embedding_dimension"),
                    name="embedding_dimension",
                )
            except (TypeError, ValueError) as error:
                raise VideoSearchError(
                    "Native-video index contains an invalid embedding dimension."
                ) from error
            if dimension != embedding_space.embedding_dimension:
                raise VideoSearchError(
                    "Native-video embedding dimension conflicts with its space metadata."
                )
            if (
                expected_embedding_dimension is not None
                and dimension != expected_embedding_dimension
            ):
                raise VideoIndexCompatibilityError(
                    "Native-video index embedding dimension does not match the "
                    "requested dimension."
                )
            precision = metadata.get("model_precision")
            if not isinstance(precision, str) or precision not in {"fp32", "bf16"}:
                raise VideoSearchError(
                    "Native-video index contains an invalid model precision."
                )
            if expected_precision is not None and precision != expected_precision:
                raise VideoIndexCompatibilityError(
                    "Native-video index precision does not match the loaded model."
                )
            fps_value = metadata.get("fps")
            max_frames_value = metadata.get("max_frames")
            try:
                fps, max_frames = _validate_native_video_settings(
                    fps_value,
                    max_frames_value,
                )
            except (TypeError, ValueError) as error:
                raise VideoSearchError(
                    "Native-video index contains invalid sampling settings."
                ) from error
            _validate_native_video_processing_parameters(
                embedding_space.processing_parameters or "",
                fps=fps,
                max_frames=max_frames,
            )
            if expected_fps is not None and fps != expected_fps:
                raise VideoIndexCompatibilityError(
                    "Native-video index fps differs from the requested fps."
                )
            if expected_max_frames is not None and max_frames != expected_max_frames:
                raise VideoIndexCompatibilityError(
                    "Native-video index max_frames differs from the requested value."
                )
            recursive = metadata.get("recursive")
            if not isinstance(recursive, bool):
                raise VideoSearchError(
                    "Native-video index contains an invalid recursion setting."
                )
            root_value = metadata.get("video_root")
            if not isinstance(root_value, str) or not root_value:
                raise VideoSearchError(
                    "Native-video index contains an invalid video root."
                )
            video_root = Path(root_value)
            if expected_root is not None and os.path.normcase(
                str(video_root)
            ) != os.path.normcase(str(expected_root)):
                raise VideoIndexCompatibilityError(
                    "Native-video index was created for a different video directory."
                )
            video_values = metadata.get("videos")
            if not isinstance(video_values, list):
                raise VideoSearchError(
                    "Native-video index contains invalid video records."
                )
            videos = tuple(_parse_video_record(value) for value in video_values)
            video_keys = [_path_key(video.relative_path) for video in videos]
            if len(set(video_keys)) != len(video_keys):
                raise VideoSearchError(
                    "Native-video index contains duplicate video paths."
                )
            if videos != tuple(
                sorted(
                    videos,
                    key=lambda video: (
                        _path_key(video.relative_path),
                        video.relative_path,
                    ),
                )
            ):
                raise VideoSearchError(
                    "Native-video index records are not in canonical path order."
                )
            video_count = metadata.get("video_count")
            if (
                isinstance(video_count, bool)
                or not isinstance(video_count, int)
                or video_count != len(videos)
            ):
                raise VideoSearchError(
                    "Native-video index count does not match its metadata."
                )
            if (
                expected_embedding_space is not None
                and embedding_space != expected_embedding_space
            ):
                raise VideoIndexCompatibilityError(
                    "Native-video embedding space does not match the requested model, "
                    "revision, processor, query policy, or video settings."
                )
            checksum = metadata.get("embedding_sha256")
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
            ):
                raise VideoSearchError(
                    "Native-video index contains an invalid embedding checksum."
                )
            embeddings = archive["embeddings"]
    except IndexArchiveError as error:
        if not index_path.exists():
            raise VideoSearchError(
                f"Native-video index not found: {index_path}"
            ) from error
        raise VideoSearchError(
            f"Could not read native-video index {index_path}: {error}"
        ) from error

    if embeddings.dtype != np.float32 or embeddings.shape != (len(videos), dimension):
        raise VideoSearchError(
            "Native-video embeddings have an invalid dtype or shape: "
            f"expected float32{(len(videos), dimension)}, "
            f"received {embeddings.dtype}{embeddings.shape}."
        )
    if not np.isfinite(embeddings).all():
        raise VideoSearchError("Native-video index contains non-finite embeddings.")
    if _array_sha256(embeddings) != checksum:
        raise VideoSearchError(
            "Native-video embedding checksum does not match its metadata."
        )
    return _NativeVideoIndex(
        video_root,
        recursive,
        fps,
        max_frames,
        videos,
        embeddings,
        precision,
        embedding_space,
    )


def _native_video_vector(
    encoder: Callable[..., torch.Tensor],
    video_path: Path,
    *,
    embedding_dimension: int,
    fps: float,
    max_frames: int,
) -> np.ndarray:
    with torch.inference_mode():
        features = encoder(
            video_path,
            normalize=True,
            truncate_dim=embedding_dimension,
            fps=fps,
            max_frames=max_frames,
        )
    if not isinstance(features, torch.Tensor) or tuple(features.shape) != (
        1,
        embedding_dimension,
    ):
        actual_shape = (
            tuple(features.shape)
            if isinstance(features, torch.Tensor)
            else type(features).__name__
        )
        raise VideoSearchError(
            "Video encoder returned an invalid embedding: "
            f"expected {(1, embedding_dimension)}, received {actual_shape}."
        )
    vector = features.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
        raise VideoSearchError(
            f"Video encoder returned an invalid embedding for {video_path}."
        )
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def build_or_update_native_video_index(
    model: torch.nn.Module,
    videos_dir: str | os.PathLike[str],
    index_dir: str | os.PathLike[str],
    *,
    fps: float = _DEFAULT_NATIVE_VIDEO_FPS,
    max_frames: int = _DEFAULT_NATIVE_VIDEO_MAX_FRAMES,
    recursive: bool = False,
    embedding_dim: int | None = None,
    verify_contents: bool = False,
    query_instruction: str | None = None,
) -> NativeVideoIndexUpdate:
    """Build or incrementally update a model-native video-embedding index.

    Set ``verify_contents=True`` to compare SHA-256 hashes and re-embed videos
    whose bytes changed; this reads every video during each index update.
    """
    encoder = _require_native_video_model(model)
    fps, max_frames = _validate_native_video_settings(fps, max_frames)
    _validate_native_video_model_frame_limit(model, max_frames)
    if not isinstance(verify_contents, bool):
        raise TypeError("verify_contents must be a bool.")
    if not callable(getattr(model, "validate_embedding_dimension", None)):
        raise VideoSearchError(
            "Native video indexing requires a backend-native embedding dimension."
        )
    embedding_dim = validate_model_embedding_dimension(
        model,
        embedding_dim,
        name="embedding_dim",
    )
    root, videos = _scan_video_files(
        videos_dir,
        recursive=recursive,
        verify_contents=verify_contents,
    )
    output_dir = Path(index_dir).expanduser()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Native-video index output path is not a directory: {output_dir}")
    index_path = output_dir / _NATIVE_VIDEO_INDEX_FILENAME
    _, precision = _model_settings(model)
    embedding_space = _native_video_embedding_space(
        model,
        embedding_dimension=embedding_dim,
        precision=precision,
        fps=fps,
        max_frames=max_frames,
        query_instruction=query_instruction,
    )

    previous: _NativeVideoIndex | None = None
    if index_path.exists():
        previous = _load_native_video_index(
            output_dir,
            expected_root=root,
            expected_precision=precision,
            expected_fps=fps,
            expected_max_frames=max_frames,
            expected_embedding_dimension=embedding_dim,
            expected_embedding_space=embedding_space,
        )
    elif not videos:
        raise ValueError(f"No supported video files were found in {root}.")

    previous_rows = (
        {
            _path_key(video.relative_path): row
            for row, video in enumerate(previous.videos)
        }
        if previous is not None
        else {}
    )
    current_keys = {_path_key(video.relative_path) for video in videos}
    removed = (
        sum(
            _path_key(video.relative_path) not in current_keys
            for video in previous.videos
        )
        if previous is not None
        else 0
    )
    embeddings_by_key: dict[str, np.ndarray] = {}
    videos_for_index = list(videos)
    added = 0
    updated = 0
    unchanged = 0
    for row, video in enumerate(videos):
        key = _path_key(video.relative_path)
        previous_row = previous_rows.get(key)
        if previous is not None and previous_row is not None:
            previous_video = previous.videos[previous_row]
            if verify_contents:
                unchanged_content = (
                    previous_video.content_sha256 is not None
                    and previous_video.content_sha256 == video.content_sha256
                )
            else:
                unchanged_content = (
                    previous_video.size_bytes,
                    previous_video.mtime_ns,
                ) == (video.size_bytes, video.mtime_ns)
            if unchanged_content:
                embeddings_by_key[key] = previous.embeddings[previous_row]
                unchanged += 1
                if not verify_contents:
                    videos_for_index[row] = replace(
                        video,
                        content_sha256=previous_video.content_sha256,
                    )
                continue
            updated += 1
        else:
            added += 1
        video_path = root / PurePosixPath(video.relative_path)
        embeddings_by_key[key] = _native_video_vector(
            encoder,
            video_path,
            embedding_dimension=embedding_dim,
            fps=fps,
            max_frames=max_frames,
        )

    videos = tuple(videos_for_index)
    for video in videos:
        video_path = root / PurePosixPath(video.relative_path)
        if _file_signature(video_path) != (video.size_bytes, video.mtime_ns):
            raise VideoSearchError(
                f"Video changed during indexing: {video_path}. Run the index command again."
            )
        if verify_contents and _video_content_sha256(video_path) != video.content_sha256:
            raise VideoSearchError(
                f"Video changed during indexing: {video_path}. Run the index command again."
            )

    if previous is not None and previous.recursive == recursive and (
        added == 0 and updated == 0 and removed == 0 and videos == previous.videos
    ):
        return NativeVideoIndexUpdate(index_path, added, updated, removed, unchanged)

    matrix = (
        np.stack(
            [embeddings_by_key[_path_key(video.relative_path)] for video in videos]
        ).astype(np.float32, copy=False)
        if videos
        else np.empty((0, embedding_dim), dtype=np.float32)
    )
    saved_path = _save_native_video_index(
        output_dir,
        _NativeVideoIndex(
            root,
            recursive,
            fps,
            max_frames,
            videos,
            matrix,
            precision,
            embedding_space,
        ),
    )
    return NativeVideoIndexUpdate(saved_path, added, updated, removed, unchanged)


def search_native_video(
    model: torch.nn.Module,
    index_dir: str | os.PathLike[str],
    query: str,
    *,
    top_k: int = 5,
    query_instruction: str | None = None,
) -> list[NativeVideoSearchResult]:
    """Search native video embeddings with text; scores are cosine similarities."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be a positive integer.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    if not query.strip():
        raise ValueError("query must contain non-whitespace text.")
    _require_native_video_model(model)
    device, precision = _model_settings(model)
    index = _load_native_video_index(index_dir, expected_precision=precision)
    _validate_native_video_model_frame_limit(model, index.max_frames)
    if not index.videos:
        raise ValueError("The native-video index is empty; run the index command first.")
    expected_embedding_space = _native_video_embedding_space(
        model,
        embedding_dimension=index.embeddings.shape[1],
        precision=precision,
        fps=index.fps,
        max_frames=index.max_frames,
        query_instruction=query_instruction,
    )
    if index.embedding_space != expected_embedding_space:
        raise VideoIndexCompatibilityError(
            "Native-video embedding space does not match the search model, revision, "
            "processor, query policy, or video settings."
        )
    query_features = _encode_search_query(
        model,
        query,
        device=device,
        embedding_dimension=index.embeddings.shape[1],
        embedding_space=index.embedding_space,
        query_instruction=query_instruction,
    )
    query_vector = (
        query_features.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
    )
    query_norm = float(np.linalg.norm(query_vector))
    if not np.isfinite(query_vector).all() or not np.isfinite(query_norm) or (
        query_norm <= 0
    ):
        raise VideoSearchError("Text encoder returned an invalid query embedding.")
    query_vector = query_vector / query_norm
    scores = index.embeddings @ query_vector
    if not np.isfinite(scores).all():
        raise VideoSearchError("Native-video index produced non-finite similarity scores.")
    best = select_top_k(
        scores,
        limit=min(top_k, len(index.videos)),
        start_row=0,
        sort_key=lambda item: (
            -item[0],
            _path_key(index.videos[item[1]].relative_path),
            index.videos[item[1]].relative_path,
        ),
    )
    return [
        NativeVideoSearchResult(
            rank=rank,
            video_path=index.video_root
            / PurePosixPath(index.videos[row].relative_path),
            score=score,
        )
        for rank, (score, row) in enumerate(best, start=1)
    ]
