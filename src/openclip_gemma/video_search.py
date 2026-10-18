"""Internal video-frame indexing and text search helpers."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from openclip_gemma.batching import _iter_preprocessed_embeddings
from openclip_gemma.constants import MODEL_ID, MODEL_REVISION
from openclip_gemma.image_search import (
    _PREPROCESSING_ID,
    _ImageSearchModel,
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
from openclip_gemma.tokenizer import get_tokenizer

_VIDEO_INDEX_FILENAME = "video-index.npz"
_VIDEO_INDEX_KIND = "openclip-gemma.video-index"
_VIDEO_INDEX_FORMAT_VERSION = 1
_EMBEDDING_DIMENSION = 768
_SAMPLING_METHOD = "first-presentation-frame-at-or-after-fixed-target-v1"
_SUPPORTED_VIDEO_SUFFIXES = frozenset(
    {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"}
)

type _EmbeddingMatrix = NDArray[np.float32]
type _ImagePreprocess = Callable[[Image.Image], torch.Tensor]


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
    """A matching video frame with its path, PTS-derived time, and cosine score."""

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
            if path.is_symlink() or path.suffix.lower() not in _SUPPORTED_VIDEO_SUFFIXES:
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
        raise VideoSearchError(f"Could not scan video directory {root}: {error}") from error

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
        next_target: Fraction | None = None
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
                timestamp = Fraction(frame.pts) * time_base
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise VideoSearchError(
                        f"Video presentation timestamps are not monotonic in {video_path} "
                        f"at frame {frame_index}."
                    )
                previous_timestamp = timestamp

                if next_target is None:
                    next_target = timestamp
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
                    ),
                    image,
                )
                sampled_count += 1
                while next_target <= timestamp:
                    next_target += interval
        except VideoSearchError:
            raise
        except (OSError, ValueError, av.error.FFmpegError) as error:
            raise VideoSearchError(f"Could not decode video {video_path}: {error}") from error

        if decoded_count == 0 or sampled_count == 0:
            raise VideoSearchError(f"No video frames could be sampled from {video_path}.")
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
        )
        yield frame_record, tensor


def _frame_sort_key(record: _FrameRecord) -> tuple[str, float, str, int]:
    return (
        _path_key(record.video_path),
        record.timestamp_seconds,
        record.video_path,
        record.pts,
    )


def _metadata_for_index(index: _VideoIndex, precision: str) -> dict[str, object]:
    return {
        "kind": _VIDEO_INDEX_KIND,
        "format_version": _VIDEO_INDEX_FORMAT_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_precision": precision,
        "embedding_dimension": _EMBEDDING_DIMENSION,
        "preprocessing_id": _PREPROCESSING_ID,
        "storage_dtype": "float32",
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
    if index.embeddings.shape != (len(index.frames), _EMBEDDING_DIMENSION):
        raise VideoSearchError("Refusing to save a video index with inconsistent dimensions.")
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
        raise VideoSearchError("Video index metadata contains invalid video file information.")
    if PurePosixPath(relative_path).suffix.lower() not in _SUPPORTED_VIDEO_SUFFIXES:
        raise VideoSearchError("Video index metadata contains an unsupported video extension.")
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
        raise VideoSearchError("Video index metadata contains a non-relative video path.")
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
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (float, int))
        or not math.isfinite(float(timestamp))
        or isinstance(pts, bool)
        or not isinstance(pts, int)
        or isinstance(time_base_numerator, bool)
        or not isinstance(time_base_numerator, int)
        or time_base_numerator <= 0
        or isinstance(time_base_denominator, bool)
        or not isinstance(time_base_denominator, int)
        or time_base_denominator <= 0
    ):
        raise VideoSearchError("Video index metadata contains invalid frame timing.")
    exact_timestamp = Fraction(pts * time_base_numerator, time_base_denominator)
    if not math.isclose(float(timestamp), float(exact_timestamp), rel_tol=1e-12, abs_tol=1e-9):
        raise VideoSearchError("Video frame timestamp does not match its PTS and time base.")
    return _FrameRecord(
        video_path,
        float(timestamp),
        pts,
        time_base_numerator,
        time_base_denominator,
    )


def _validate_video_metadata(
    metadata: dict[str, Any],
    *,
    expected_root: Path | None,
    expected_precision: str | None,
    expected_interval: float | None,
) -> tuple[Path, bool, float, tuple[_VideoRecord, ...], tuple[_FrameRecord, ...]]:
    if metadata.get("kind") != _VIDEO_INDEX_KIND:
        raise VideoIndexCompatibilityError("The archive is not an openclip-gemma video index.")
    version = metadata.get("format_version")
    if type(version) is not int or version != _VIDEO_INDEX_FORMAT_VERSION:
        raise VideoIndexCompatibilityError(
            f"Unsupported video index format version: {version!r}."
        )
    for name, expected in (
        ("model_id", MODEL_ID),
        ("model_revision", MODEL_REVISION),
        ("embedding_dimension", _EMBEDDING_DIMENSION),
        ("preprocessing_id", _PREPROCESSING_ID),
        ("storage_dtype", "float32"),
        ("sampling_method", _SAMPLING_METHOD),
    ):
        if metadata.get(name) != expected:
            raise VideoIndexCompatibilityError(
                f"Video index {name} does not match the current configuration."
            )

    precision = metadata.get("model_precision")
    if not isinstance(precision, str) or precision not in {"fp32", "bf16"}:
        raise VideoSearchError("Video index metadata contains an invalid model precision.")
    if expected_precision is not None and precision != expected_precision:
        raise VideoIndexCompatibilityError(
            "Video index model precision does not match the loaded model."
        )

    interval = metadata.get("interval_seconds")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or float(interval) <= 0
    ):
        raise VideoSearchError("Video index metadata contains an invalid sampling interval.")
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
    if expected_root is not None and os.path.normcase(str(video_root)) != os.path.normcase(
        str(expected_root)
    ):
        raise VideoIndexCompatibilityError(
            "Video index was created for a different video directory."
        )

    recursive = metadata.get("recursive")
    if not isinstance(recursive, bool):
        raise VideoSearchError("Video index metadata contains an invalid recursion setting.")
    videos_value = metadata.get("videos")
    frames_value = metadata.get("frames")
    if not isinstance(videos_value, list) or not isinstance(frames_value, list):
        raise VideoSearchError("Video index metadata contains invalid video or frame lists.")
    videos = tuple(_parse_video_record(record) for record in videos_value)
    video_keys = [_path_key(video.relative_path) for video in videos]
    if len(set(video_keys)) != len(video_keys):
        raise VideoSearchError("Video index metadata contains duplicate video paths.")
    if videos != tuple(
        sorted(videos, key=lambda video: (_path_key(video.relative_path), video.relative_path))
    ):
        raise VideoSearchError("Video index records are not in canonical path order.")
    video_paths = {video.relative_path for video in videos}
    frames = tuple(_parse_frame_record(record, video_paths) for record in frames_value)
    frame_keys = [
        (frame.video_path, frame.pts, frame.time_base_numerator, frame.time_base_denominator)
        for frame in frames
    ]
    if len(set(frame_keys)) != len(frame_keys):
        raise VideoSearchError("Video index metadata contains duplicate frame timestamps.")
    if frames != tuple(sorted(frames, key=_frame_sort_key)):
        raise VideoSearchError("Video frame records are not in canonical path/time order.")
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
        raise VideoSearchError("Video index metadata contains an invalid embedding checksum.")
    return video_root, recursive, interval_seconds, videos, frames


def _load_video_index(
    index_dir: str | os.PathLike[str],
    *,
    expected_root: Path | None = None,
    expected_precision: str | None = None,
    expected_interval: float | None = None,
) -> _VideoIndex:
    index_path = Path(index_dir).expanduser() / _VIDEO_INDEX_FILENAME
    try:
        with _open_numpy_archive(
            index_path,
            expected_arrays=frozenset({"embeddings"}),
        ) as (metadata, archive):
            video_root, recursive, interval, videos, frames = _validate_video_metadata(
                metadata,
                expected_root=expected_root,
                expected_precision=expected_precision,
                expected_interval=expected_interval,
            )
            embeddings = archive["embeddings"]
    except IndexArchiveError as error:
        if not index_path.exists():
            raise VideoSearchError(f"Video index not found: {index_path}") from error
        raise VideoSearchError(f"Could not read video index {index_path}: {error}") from error
    expected_shape = (len(frames), _EMBEDDING_DIMENSION)
    if embeddings.dtype != np.float32 or embeddings.shape != expected_shape:
        raise VideoSearchError(
            "Video index embeddings have an invalid dtype or shape: "
            f"expected float32{expected_shape}, received {embeddings.dtype}{embeddings.shape}."
        )
    if not np.isfinite(embeddings).all():
        raise VideoSearchError("Video index contains non-finite embeddings.")
    if _array_sha256(embeddings) != metadata["embedding_sha256"]:
        raise VideoSearchError("Video index embedding checksum does not match its metadata.")
    return _VideoIndex(video_root, recursive, interval, videos, frames, embeddings)


def _file_signature(path: Path) -> tuple[int, int]:
    try:
        file_stat = path.stat()
    except OSError as error:
        raise VideoSearchError(f"Could not inspect video file {path}: {error}") from error
    return file_stat.st_size, file_stat.st_mtime_ns


def _embedding_vector(embedding: torch.Tensor, video_path: Path) -> np.ndarray:
    vector = embedding.detach().to(device="cpu", dtype=torch.float32).numpy()
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
        raise VideoSearchError(f"Image encoder returned an invalid frame embedding for {video_path}.")
    return vector / norm


def _embed_video(
    model: _ImageSearchModel,
    preprocess: _ImagePreprocess,
    video_path: Path,
    relative_path: str,
    interval: Fraction,
    batch_size: int,
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
) -> VideoIndexUpdate:
    """Create or incrementally update a versioned index of sampled video frames."""
    interval_seconds, interval_fraction = _validate_interval(interval)
    _validate_batch_size(batch_size)
    root, videos = _scan_video_files(videos_dir, recursive=recursive)
    output_dir = Path(index_dir).expanduser()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Video index output path is not a directory: {output_dir}")
    index_path = output_dir / _VIDEO_INDEX_FILENAME
    _, precision = _model_settings(model)

    previous: _VideoIndex | None = None
    if index_path.exists():
        if not index_path.is_file():
            raise VideoSearchError(f"Video index path is not a file: {index_path}")
        previous = _load_video_index(
            output_dir,
            expected_root=root,
            expected_precision=precision,
            expected_interval=interval_seconds,
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
        sum(_path_key(video.relative_path) not in current_paths for video in previous.videos)
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
        return VideoIndexUpdate(index_path, added, updated, removed, unchanged, len(frame_records))

    embeddings = (
        np.stack(embedding_rows).astype(np.float32, copy=False)
        if embedding_rows
        else np.empty((0, _EMBEDDING_DIMENSION), dtype=np.float32)
    )
    index = _VideoIndex(
        root,
        recursive,
        interval_seconds,
        videos,
        tuple(frame_records),
        embeddings,
    )
    saved_path = _save_video_index(output_dir, index, precision)
    return VideoIndexUpdate(saved_path, added, updated, removed, unchanged, len(frame_records))


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

    tokenizer = get_tokenizer(prompt_name="SearchQuery")
    tokens = tokenizer(query).to(device)
    with torch.inference_mode():
        query_features = model.encode_text(tokens, normalize=True)
    expected_shape = (1, _EMBEDDING_DIMENSION)
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
    query_vector = query_features.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
    query_norm = float(np.linalg.norm(query_vector))
    if not np.isfinite(query_vector).all() or not np.isfinite(query_norm) or query_norm <= 0:
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
