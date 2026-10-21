"""Internal video-frame indexing and text search helpers."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from openclip_gemma._dimensions import (
    DEFAULT_EMBEDDING_DIMENSION,
    EMBEDDING_STRATEGY,
    validate_positive_embedding_dimension,
)
from openclip_gemma.backends.base import validate_model_embedding_dimension
from openclip_gemma.batching import _iter_preprocessed_embeddings
from openclip_gemma.constants import (
    BACKEND_ID,
    EMBEDDING_NORMALIZATION,
    IMAGE_PREPROCESSING_ID,
    MODEL_ID,
    MODEL_REVISION,
)
from openclip_gemma.embedding_space import (
    EmbeddingSpace,
    has_embedding_space_metadata,
)
from openclip_gemma.image_search import (
    _PREPROCESSING_ID,
    ImageIndexCompatibilityError,
    _get_search_tokenizer,
    _ImageSearchModel,
    _model_embedding_space,
    _model_settings,
    _path_key,
    _validate_batch_size,
)
from openclip_gemma.index_storage import (
    IndexArchiveError,
    _array_sha256,
    _open_numpy_archive,
    _save_numpy_archive_atomic,
)
from openclip_gemma.registry import get_default_registry
from openclip_gemma.tokenizer import OFFICIAL_PROMPTS, get_tokenizer

_VIDEO_INDEX_FILENAME = "video-index.npz"
_VIDEO_INDEX_KIND = "openclip-gemma.video-index"
_VIDEO_INDEX_FORMAT_VERSION = 3
_LEGACY_VIDEO_INDEX_FORMAT_VERSION = 2
_SAMPLING_METHOD = "first-presentation-frame-at-or-after-playback-origin-target-v2"
_SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"})

type _EmbeddingMatrix = NDArray[np.float32]
type _ImagePreprocess = Callable[[Image.Image], torch.Tensor]

__all__ = [
    "VideoIndexCompatibilityError",
    "VideoIndexUpdate",
    "VideoSearchError",
    "VideoSearchResult",
    "build_or_update_video_index",
    "convert_video_index",
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


def _load_av() -> Any:
    try:
        import av
    except ImportError as error:
        raise VideoSearchError(
            "Video indexing requires the optional PyAV dependency. "
            "Install it with `pip install openclip-gemma[video]`."
        ) from error
    return av


def _validate_interval(interval: float) -> tuple[float, Fraction]:
    if isinstance(interval, bool) or not isinstance(interval, (int, float)):
        raise TypeError("interval must be a finite positive number of seconds.")
    interval_seconds = float(interval)
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval must be a finite positive number of seconds.")
    return interval_seconds, Fraction(str(interval_seconds))


def _scan_video_files(
    videos_dir: str | os.PathLike[str],
    *,
    recursive: bool,
) -> tuple[Path, tuple[_VideoRecord, ...]]:
    if not isinstance(recursive, bool):
        raise TypeError("recursive must be a bool.")
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
                _VideoRecord(relative_path, file_stat.st_size, file_stat.st_mtime_ns),
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
                while next_target <= timestamp:
                    next_target += interval
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
) -> Iterator[tuple[_FrameRecord, torch.Tensor]]:
    for sampled_frame in _iter_sampled_frames(video_path, interval):
        try:
            tensor = preprocess(sampled_frame.image)
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
        yield frame_record, tensor


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
) -> EmbeddingSpace:
    try:
        return _model_embedding_space(
            model,
            embedding_dimension=embedding_dimension,
            precision=precision,
            sampling_method=_SAMPLING_METHOD,
        )
    except ImageIndexCompatibilityError as error:
        raise VideoIndexCompatibilityError(str(error)) from error


def _embedding_space_from_video_metadata(
    metadata: dict[str, Any],
    *,
    embedding_dimension: int,
    precision: str,
) -> EmbeddingSpace:
    if has_embedding_space_metadata(metadata):
        try:
            embedding_space = EmbeddingSpace.from_metadata(metadata)
        except (TypeError, ValueError) as error:
            raise VideoIndexCompatibilityError(
                f"Video index embedding-space metadata is invalid: {error}"
            ) from error
        if embedding_space.embedding_strategy != EMBEDDING_STRATEGY:
            raise VideoIndexCompatibilityError(
                "Video index embedding_strategy is missing or unsupported."
            )
        if embedding_space.sampling_method != _SAMPLING_METHOD:
            raise VideoIndexCompatibilityError(
                "Video index sampling_method does not match the current configuration."
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
        "videos": [
            {
                "path": video.relative_path,
                "size_bytes": video.size_bytes,
                "mtime_ns": video.mtime_ns,
            }
            for video in index.videos
        ],
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
    if PurePosixPath(relative_path).suffix.lower() not in _SUPPORTED_VIDEO_SUFFIXES:
        raise VideoSearchError(
            "Video index metadata contains an unsupported video extension."
        )
    return _VideoRecord(relative_path, size_bytes, mtime_ns)


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
            "The archive is not an openclip-gemma video index."
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
        and metadata.get("embedding_strategy") != EMBEDDING_STRATEGY
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
    if (
        expected_embedding_space is not None
        and embedding_space != expected_embedding_space
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
    embedding_dim: int = DEFAULT_EMBEDDING_DIMENSION,
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
    )
    if index.embedding_space != expected_embedding_space:
        raise VideoIndexCompatibilityError(
            "Video index embedding space does not match the search model, revision, "
            "preprocessing, prompt, sampling, or precision."
        )

    prompt_name = index.embedding_space.query_prompt_name
    if prompt_name is None:
        raise VideoIndexCompatibilityError(
            "Video index does not define a compatible text query prompt."
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
        raise VideoSearchError(
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
        candidates = [(float(score), row + start) for row, score in enumerate(scores)]
        best = sorted((*best, *candidates), key=sort_key)[:limit]

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
    embedding_dim: int = DEFAULT_EMBEDDING_DIMENSION,
) -> VideoIndexUpdate:
    """Build or incrementally update an index of sampled video frames.

    Requires the optional PyAV dependency. Timestamps in the index are relative
    to the container playback start, or the video stream start when unavailable.
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
) -> list[VideoSearchResult]:
    """Search a video-frame index with a Japanese or English text query.

    Results are ordered by descending cosine similarity and include playback-
    relative timestamps in seconds. Scores are not probabilities.
    """
    return _search_video_index(model, index_dir, query, top_k=top_k)
