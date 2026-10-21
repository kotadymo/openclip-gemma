from __future__ import annotations

import builtins
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from openclip_gemma import video_search
from openclip_gemma.backends.embeddinggemma2 import EmbeddingGemma2Backend
from openclip_gemma.embedding_space import EmbeddingSpaceMetadata
from openclip_gemma.index_storage import _save_numpy_archive_atomic
from openclip_gemma.video_search import (
    VideoIndexCompatibilityError,
    VideoSearchError,
    _build_or_update_video_index,
    _iter_sampled_frames,
    _load_video_index,
    _validate_interval,
    build_or_update_video_index,
    convert_video_index,
    search_video,
)


class FakeFFmpegError(Exception):
    pass


class FakeFrame:
    def __init__(
        self,
        pts: int | None,
        color: tuple[int, int, int],
        *,
        time_base: Fraction | None = Fraction(1, 30),
        size: tuple[int, int] = (8, 8),
    ) -> None:
        self.pts = pts
        self.time_base = time_base
        self.image = Image.new("RGB", size, color)
        self.converted = 0

    def to_image(self) -> Image.Image:
        self.converted += 1
        return self.image.copy()


class FakeStream:
    time_base = Fraction(1, 30)

    def __init__(self, start_time: int | None = None) -> None:
        self.start_time = start_time


class FakeContainer:
    def __init__(
        self,
        frames: list[FakeFrame],
        *,
        decode_error: bool = False,
        no_video_stream: bool = False,
        start_time: int | None = None,
        stream_start_time: int | None = None,
    ) -> None:
        self.frames = frames
        self.decode_error = decode_error
        self.start_time = start_time
        self.streams = SimpleNamespace(
            video=[] if no_video_stream else [FakeStream(stream_start_time)]
        )
        self.closed = False

    def decode(self, *, video: int):
        assert video == 0
        yield from self.frames
        if self.decode_error:
            raise FakeFFmpegError("invalid packet")

    def close(self) -> None:
        self.closed = True


class FakeAV:
    error = SimpleNamespace(FFmpegError=FakeFFmpegError)
    time_base = 1_000_000

    def __init__(self, videos: dict[Path, FakeContainer]) -> None:
        self.videos = videos
        self.opened: list[Path] = []

    def open(self, path: str, *, mode: str) -> FakeContainer:
        assert mode == "r"
        video_path = Path(path)
        self.opened.append(video_path)
        if video_path not in self.videos:
            raise FakeFFmpegError("unknown or corrupt video")
        return self.videos[video_path]


class FakeVideoModel(nn.Module):
    def __init__(
        self,
        *,
        embedding_space_metadata: EmbeddingSpaceMetadata | None = None,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.batch_sizes: list[int] = []
        self.inference_modes: list[bool] = []
        self.received_tokens: list[torch.Tensor] = []
        self.embedding_space_metadata = (
            embedding_space_metadata or EmbeddingGemma2Backend.embedding_space_metadata
        )
        self._dimension_backend = EmbeddingGemma2Backend()

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int:
        return self._dimension_backend.validate_embedding_dimension(
            value,
            name=name,
        )

    def encode_image(
        self,
        images: torch.Tensor,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        self.batch_sizes.append(images.shape[0])
        self.inference_modes.append(torch.is_inference_mode_enabled())
        features = images.mean(dim=(2, 3))
        features = torch.nn.functional.normalize(features, dim=-1)
        features = torch.nn.functional.pad(features, (0, 765))
        if truncate_dim is not None:
            features = torch.nn.functional.normalize(
                features[:, :truncate_dim],
                dim=-1,
            )
        return features

    def encode_text(
        self,
        tokens: torch.Tensor,
        *,
        normalize: bool = False,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        self.received_tokens.append(tokens.detach().cpu())
        embedding_dim = 768 if truncate_dim is None else truncate_dim
        features = torch.zeros(
            (tokens.shape[0], embedding_dim),
            device=tokens.device,
            dtype=self.anchor.dtype,
        )
        features[:, 0] = 1
        return (
            torch.nn.functional.normalize(features, dim=-1) if normalize else features
        )


class FakeTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str) -> torch.Tensor:
        self.texts.append(text)
        return torch.tensor([[1]], dtype=torch.long)


def _preprocess(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1) / 255


def _frames() -> list[FakeFrame]:
    return [
        FakeFrame(0, (255, 0, 0)),
        FakeFrame(15, (0, 255, 0)),
        FakeFrame(31, (0, 0, 255)),
        FakeFrame(61, (255, 0, 0)),
    ]


def _write_video(path: Path, data: bytes = b"fake video") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _rewrite_video_index_metadata(
    index_path: Path,
    updates: dict[str, object],
    *,
    remove: tuple[str, ...] = (),
) -> None:
    with np.load(index_path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        embeddings = archive["embeddings"].copy()
    metadata.update(updates)
    for name in remove:
        metadata.pop(name, None)
    _save_numpy_archive_atomic(
        index_path,
        metadata,
        {"embeddings": embeddings},
    )


def _rewrite_video_index_version(index_path: Path, version: int) -> None:
    _rewrite_video_index_metadata(
        index_path,
        {"format_version": version},
        remove=(
            (
                "embedding_strategy",
                "backend_id",
                "normalization",
                "index_prompt",
                "query_prompt_name",
                "query_prompt",
                "embedding_space_id",
            )
            if version == 2
            else ()
        ),
    )


def test_sampling_uses_actual_vfr_pts_and_preserves_frame_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_video(tmp_path / "vfr.mp4")
    frames = [
        FakeFrame(0, (255, 0, 0), time_base=Fraction(1, 30)),
        FakeFrame(15, (0, 255, 0), time_base=Fraction(1, 30)),
        FakeFrame(31, (0, 0, 255), time_base=Fraction(1, 30)),
        FakeFrame(2033, (255, 0, 0), time_base=Fraction(1, 1000)),
    ]
    container = FakeContainer(frames)
    monkeypatch.setattr(video_search, "_load_av", lambda: FakeAV({path: container}))

    sampled = list(_iter_sampled_frames(path, Fraction(1, 1)))

    assert [frame.record.pts for frame in sampled] == [0, 31, 2033]
    assert [frame.record.timestamp_seconds for frame in sampled] == pytest.approx(
        [0.0, 31 / 30, 2.033]
    )
    assert [frame.record.time_base_denominator for frame in sampled] == [30, 30, 1000]
    assert [frame.image.mode for frame in sampled] == ["RGB"] * 3
    assert container.closed
    for frame in sampled:
        frame.image.close()


def test_sampling_timestamps_are_relative_to_container_playback_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_video(tmp_path / "nonzero-start.mp4")
    frames = [
        FakeFrame(300, (255, 0, 0)),
        FakeFrame(315, (0, 255, 0)),
        FakeFrame(330, (0, 0, 255)),
        FakeFrame(345, (255, 255, 0)),
    ]
    container = FakeContainer(
        frames,
        start_time=10_000_000,
        stream_start_time=210,
    )
    monkeypatch.setattr(video_search, "_load_av", lambda: FakeAV({path: container}))

    sampled = list(_iter_sampled_frames(path, Fraction(1, 2)))

    assert [frame.record.timestamp_seconds for frame in sampled] == pytest.approx(
        [0.0, 0.5, 1.0, 1.5]
    )
    assert {frame.record.timestamp_origin_numerator for frame in sampled} == {10}
    assert {frame.record.timestamp_origin_denominator for frame in sampled} == {1}
    assert container.closed
    for frame in sampled:
        frame.image.close()


def test_sampling_uses_stream_start_when_container_start_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_video(tmp_path / "stream-start.mkv")
    frames = [
        FakeFrame(100, (255, 0, 0)),
        FakeFrame(130, (0, 255, 0)),
        FakeFrame(145, (0, 0, 255)),
    ]
    container = FakeContainer(frames, stream_start_time=100)
    monkeypatch.setattr(video_search, "_load_av", lambda: FakeAV({path: container}))

    sampled = list(_iter_sampled_frames(path, Fraction(1, 2)))

    assert [frame.record.timestamp_seconds for frame in sampled] == pytest.approx(
        [0.0, 1.0, 1.5]
    )
    assert {frame.record.timestamp_origin_numerator for frame in sampled} == {10}
    assert {frame.record.timestamp_origin_denominator for frame in sampled} == {3}
    assert container.closed
    for frame in sampled:
        frame.image.close()


@pytest.mark.parametrize(
    ("interval", "exception"),
    [
        (0, ValueError),
        (-1.0, ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
    ],
)
def test_rejects_invalid_sampling_interval(
    interval: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception, match="interval"):
        _validate_interval(interval)  # type: ignore[arg-type]


def test_missing_pts_or_video_stream_fails_and_closes_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_video(tmp_path / "missing-pts.mp4")
    no_pts = FakeContainer([FakeFrame(None, (255, 0, 0))])
    monkeypatch.setattr(video_search, "_load_av", lambda: FakeAV({path: no_pts}))

    with pytest.raises(VideoSearchError, match="no presentation timestamp"):
        list(_iter_sampled_frames(path, Fraction(1, 1)))
    assert no_pts.closed

    no_stream = FakeContainer([], no_video_stream=True)
    monkeypatch.setattr(video_search, "_load_av", lambda: FakeAV({path: no_stream}))
    with pytest.raises(VideoSearchError, match="no video stream"):
        list(_iter_sampled_frames(path, Fraction(1, 1)))
    assert no_stream.closed


def test_empty_or_corrupt_video_fails_without_silent_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_video(tmp_path / "empty.mp4")
    empty = FakeContainer([])
    monkeypatch.setattr(video_search, "_load_av", lambda: FakeAV({path: empty}))

    with pytest.raises(VideoSearchError, match="No video frames"):
        list(_iter_sampled_frames(path, Fraction(1, 1)))
    assert empty.closed

    broken = FakeContainer(_frames()[:1], decode_error=True)
    monkeypatch.setattr(video_search, "_load_av", lambda: FakeAV({path: broken}))
    with pytest.raises(VideoSearchError, match="Could not decode"):
        list(_iter_sampled_frames(path, Fraction(1, 1)))
    assert broken.closed


def test_missing_pyav_reports_optional_dependency_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_av(name: str, *args: object, **kwargs: object):
        if name == "av":
            raise ImportError("simulated missing PyAV")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_av)

    with pytest.raises(VideoSearchError, match=r"pip install openclip-gemma\[video\]"):
        video_search._load_av()


def test_video_index_round_trip_search_and_incremental_file_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos_dir = tmp_path / "videos"
    first_path = _write_video(videos_dir / "workshop.mp4", b"version-one")
    second_path = _write_video(videos_dir / "drill.mp4")
    index_dir = tmp_path / "video-index"
    decoder = FakeAV(
        {
            first_path: FakeContainer(_frames()),
            second_path: FakeContainer([FakeFrame(0, (255, 0, 0))]),
        }
    )
    monkeypatch.setattr(video_search, "_load_av", lambda: decoder)
    model = FakeVideoModel()

    first = build_or_update_video_index(
        model,
        _preprocess,
        videos_dir,
        index_dir,
        interval_seconds=1.0,
        batch_size=2,
    )

    assert (first.added_videos, first.updated_videos, first.removed_videos) == (2, 0, 0)
    assert first.frame_count == 4
    assert model.batch_sizes
    assert all(model.inference_modes)
    loaded = _load_video_index(
        index_dir, expected_root=videos_dir.resolve(), expected_interval=1.0
    )
    assert loaded.embeddings.shape == (4, 768)
    assert [frame.pts for frame in loaded.frames] == [0, 0, 31, 61]
    assert [frame.video_path for frame in loaded.frames] == [
        "drill.mp4",
        "workshop.mp4",
        "workshop.mp4",
        "workshop.mp4",
    ]
    np.testing.assert_allclose(
        np.linalg.norm(loaded.embeddings, axis=1), 1.0, atol=1e-6
    )

    opened_before_reuse = len(decoder.opened)
    unchanged = build_or_update_video_index(
        model,
        _preprocess,
        videos_dir,
        index_dir,
        interval_seconds=1.0,
        batch_size=2,
    )
    assert unchanged.unchanged_videos == 2
    assert len(decoder.opened) == opened_before_reuse

    first_path.write_bytes(b"version-two-with-extra-content")
    second_path.unlink()
    third_path = _write_video(videos_dir / "new" / "new.mp4")
    decoder.videos[first_path] = FakeContainer(
        [FakeFrame(0, (0, 0, 255)), FakeFrame(30, (255, 0, 0))]
    )
    decoder.videos[third_path] = FakeContainer([FakeFrame(0, (0, 255, 0))])
    update = build_or_update_video_index(
        model,
        _preprocess,
        videos_dir,
        index_dir,
        interval_seconds=1.0,
        batch_size=2,
        recursive=True,
    )
    assert (
        update.added_videos,
        update.updated_videos,
        update.removed_videos,
        update.unchanged_videos,
    ) == (1, 1, 1, 0)
    assert update.frame_count == 3

    tokenizer = FakeTokenizer()
    prompt_names: list[str | None] = []

    def fake_tokenizer(*, prompt_name: str | None = None) -> FakeTokenizer:
        prompt_names.append(prompt_name)
        return tokenizer

    monkeypatch.setattr(video_search, "get_tokenizer", fake_tokenizer)
    results = search_video(model, index_dir, "赤いドリル", top_k=2)
    assert prompt_names == ["SearchQuery"]
    assert tokenizer.texts == ["赤いドリル"]
    assert [result.rank for result in results] == [1, 2]
    assert all(result.video_path.is_absolute() for result in results)
    assert all(result.timestamp_seconds >= 0 for result in results)
    assert all(-1.0 <= result.score <= 1.0 for result in results)


@pytest.mark.parametrize("embedding_dim", [128, 256, 512, 768])
def test_video_index_builds_and_searches_at_requested_dimension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    embedding_dim: int,
) -> None:
    videos_dir = tmp_path / "videos"
    video_path = _write_video(videos_dir / "one.mp4")
    index_dir = tmp_path / "index"
    monkeypatch.setattr(
        video_search,
        "_load_av",
        lambda: FakeAV({video_path: FakeContainer([FakeFrame(0, (255, 0, 0))])}),
    )
    model = FakeVideoModel()

    update = build_or_update_video_index(
        model,
        _preprocess,
        videos_dir,
        index_dir,
        embedding_dim=embedding_dim,
    )
    loaded = _load_video_index(index_dir)
    monkeypatch.setattr(
        video_search,
        "get_tokenizer",
        lambda *, prompt_name: FakeTokenizer(),
    )
    results = search_video(model, index_dir, "red")

    assert update.frame_count == 1
    assert loaded.embeddings.shape == (1, embedding_dim)
    assert results[0].video_path == video_path
    assert results[0].score == pytest.approx(1.0)
    with np.load(index_dir / "video-index.npz", allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
    assert metadata["format_version"] == 3
    assert metadata["embedding_dimension"] == embedding_dim
    assert metadata["embedding_strategy"] == "leading-dimensions-l2-v1"


def test_video_search_rejects_model_with_same_dimension_but_different_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos_dir = tmp_path / "videos"
    video_path = _write_video(videos_dir / "one.mp4")
    index_dir = tmp_path / "index"
    monkeypatch.setattr(
        video_search,
        "_load_av",
        lambda: FakeAV({video_path: FakeContainer([FakeFrame(0, (255, 0, 0))])}),
    )
    build_or_update_video_index(
        FakeVideoModel(),
        _preprocess,
        videos_dir,
        index_dir,
    )
    metadata = replace(
        EmbeddingGemma2Backend.embedding_space_metadata,
        model_id="different/model",
    )
    monkeypatch.setattr(
        video_search,
        "get_tokenizer",
        lambda **kwargs: pytest.fail("incompatible model must fail before tokenizing"),
    )

    with pytest.raises(VideoIndexCompatibilityError, match="embedding space"):
        search_video(
            FakeVideoModel(embedding_space_metadata=metadata), index_dir, "red"
        )


@pytest.mark.parametrize("strategy", [None, "unsupported-strategy"])
def test_current_video_index_requires_supported_embedding_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strategy: str | None,
) -> None:
    videos_dir = tmp_path / "videos"
    video_path = _write_video(videos_dir / "one.mp4")
    index_dir = tmp_path / "index"
    monkeypatch.setattr(
        video_search,
        "_load_av",
        lambda: FakeAV({video_path: FakeContainer([FakeFrame(0, (255, 0, 0))])}),
    )
    update = build_or_update_video_index(
        FakeVideoModel(),
        _preprocess,
        videos_dir,
        index_dir,
    )
    if strategy is None:
        _rewrite_video_index_metadata(
            update.index_path,
            {},
            remove=("embedding_strategy",),
        )
    else:
        _rewrite_video_index_metadata(
            update.index_path,
            {"embedding_strategy": strategy},
        )

    with pytest.raises(
        VideoIndexCompatibilityError,
        match="embedding_strategy",
    ):
        _load_video_index(index_dir)


def test_legacy_video_index_remains_readable_and_can_be_converted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos_dir = tmp_path / "videos"
    video_path = _write_video(videos_dir / "one.mp4")
    index_dir = tmp_path / "legacy-index"
    monkeypatch.setattr(
        video_search,
        "_load_av",
        lambda: FakeAV({video_path: FakeContainer([FakeFrame(0, (255, 0, 0))])}),
    )
    model = FakeVideoModel()
    update = build_or_update_video_index(model, _preprocess, videos_dir, index_dir)
    _rewrite_video_index_version(update.index_path, 2)
    inference_count = sum(model.batch_sizes)

    assert _load_video_index(index_dir).embeddings.shape == (1, 768)
    converted_path = convert_video_index(
        index_dir,
        tmp_path / "converted-index",
        embedding_dim=128,
    )
    converted = _load_video_index(converted_path.parent)

    assert converted_path.exists()
    assert converted.embeddings.shape == (1, 128)
    with np.load(converted_path, allow_pickle=False) as archive:
        converted_metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
    assert converted_metadata["embedding_strategy"] == "leading-dimensions-l2-v1"
    np.testing.assert_allclose(np.linalg.norm(converted.embeddings, axis=1), 1.0)
    assert sum(model.batch_sizes) == inference_count
    assert _load_video_index(index_dir).embeddings.shape == (1, 768)


def test_legacy_video_index_requires_native_dimension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos_dir = tmp_path / "videos"
    video_path = _write_video(videos_dir / "one.mp4")
    index_dir = tmp_path / "index"
    monkeypatch.setattr(
        video_search,
        "_load_av",
        lambda: FakeAV({video_path: FakeContainer([FakeFrame(0, (255, 0, 0))])}),
    )
    update = build_or_update_video_index(
        FakeVideoModel(),
        _preprocess,
        videos_dir,
        index_dir,
        embedding_dim=128,
    )
    _rewrite_video_index_version(update.index_path, 2)

    with pytest.raises(VideoIndexCompatibilityError, match="Legacy video indexes"):
        _load_video_index(index_dir)


def test_video_update_rejects_mixed_embedding_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos_dir = tmp_path / "videos"
    video_path = _write_video(videos_dir / "one.mp4")
    index_dir = tmp_path / "index"
    monkeypatch.setattr(
        video_search,
        "_load_av",
        lambda: FakeAV({video_path: FakeContainer([FakeFrame(0, (255, 0, 0))])}),
    )
    model = FakeVideoModel()
    build_or_update_video_index(
        model,
        _preprocess,
        videos_dir,
        index_dir,
        embedding_dim=256,
    )

    with pytest.raises(
        VideoIndexCompatibilityError, match="does not match the requested"
    ):
        build_or_update_video_index(
            model,
            _preprocess,
            videos_dir,
            index_dir,
            embedding_dim=128,
        )


def test_video_index_conversion_rejects_expansion_and_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos_dir = tmp_path / "videos"
    video_path = _write_video(videos_dir / "one.mp4")
    index_dir = tmp_path / "index"
    monkeypatch.setattr(
        video_search,
        "_load_av",
        lambda: FakeAV({video_path: FakeContainer([FakeFrame(0, (255, 0, 0))])}),
    )
    build_or_update_video_index(
        FakeVideoModel(),
        _preprocess,
        videos_dir,
        index_dir,
        embedding_dim=128,
    )

    with pytest.raises(ValueError, match="Cannot expand"):
        convert_video_index(index_dir, tmp_path / "larger", embedding_dim=256)
    existing_output = tmp_path / "existing"
    existing_output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        convert_video_index(index_dir, existing_output, embedding_dim=128)


def test_video_index_conversion_rejects_output_created_after_precheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos_dir = tmp_path / "videos"
    video_path = _write_video(videos_dir / "one.mp4")
    index_dir = tmp_path / "index"
    monkeypatch.setattr(
        video_search,
        "_load_av",
        lambda: FakeAV({video_path: FakeContainer([FakeFrame(0, (255, 0, 0))])}),
    )
    build_or_update_video_index(
        FakeVideoModel(),
        _preprocess,
        videos_dir,
        index_dir,
        embedding_dim=128,
    )
    destination = tmp_path / "raced-output"
    original_mkdir = Path.mkdir
    competing_output_created = False

    def create_competing_output(
        path: Path,
        *,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        nonlocal competing_output_created
        if path == destination and not competing_output_created:
            competing_output_created = True
            original_mkdir(path, parents=parents, exist_ok=True)
        original_mkdir(path, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", create_competing_output)
    with pytest.raises(ValueError, match="already exists"):
        convert_video_index(index_dir, destination, embedding_dim=128)

    assert competing_output_created
    assert not (destination / "video-index.npz").exists()


def test_changed_video_failure_preserves_existing_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos_dir = tmp_path / "videos"
    path = _write_video(videos_dir / "valid.mp4")
    index_dir = tmp_path / "index"
    monkeypatch.setattr(
        video_search,
        "_load_av",
        lambda: FakeAV({path: FakeContainer([FakeFrame(0, (255, 0, 0))])}),
    )
    model = FakeVideoModel()
    _build_or_update_video_index(model, _preprocess, videos_dir, index_dir)
    index_path = index_dir / "video-index.npz"
    original = index_path.read_bytes()
    path.write_bytes(b"modified")
    monkeypatch.setattr(video_search, "_load_av", lambda: FakeAV({}))

    with pytest.raises(VideoSearchError, match="Could not open video"):
        _build_or_update_video_index(model, _preprocess, videos_dir, index_dir)

    assert index_path.read_bytes() == original


def test_interval_and_index_kind_mismatches_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_dir = tmp_path / "index"
    _save_numpy_archive_atomic(
        index_dir / "video-index.npz",
        {"kind": "openclip-gemma.image-index"},
        {"embeddings": np.empty((0, 768), dtype=np.float32)},
    )
    with pytest.raises(
        VideoIndexCompatibilityError, match="not an openclip-gemma video index"
    ):
        _load_video_index(index_dir)

    videos_dir = tmp_path / "videos"
    video_path = _write_video(videos_dir / "one.mp4")
    decoder = FakeAV({video_path: FakeContainer([FakeFrame(0, (255, 0, 0))])})
    monkeypatch.setattr(video_search, "_load_av", lambda: decoder)
    model = FakeVideoModel()
    # Replace the incompatible archive with a valid index before testing interval compatibility.
    (index_dir / "video-index.npz").unlink()
    video_search_result = _build_or_update_video_index(
        model,
        _preprocess,
        videos_dir,
        index_dir,
    )
    assert video_search_result.frame_count == 1
    with pytest.raises(VideoIndexCompatibilityError, match="sampling interval differs"):
        _build_or_update_video_index(
            model,
            _preprocess,
            videos_dir,
            index_dir,
            interval=2.0,
        )


def test_old_absolute_timestamp_index_version_requires_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos_dir = tmp_path / "videos"
    video_path = _write_video(videos_dir / "one.mp4")
    index_dir = tmp_path / "index"
    monkeypatch.setattr(
        video_search,
        "_load_av",
        lambda: FakeAV({video_path: FakeContainer([FakeFrame(0, (255, 0, 0))])}),
    )
    index = build_or_update_video_index(
        FakeVideoModel(), _preprocess, videos_dir, index_dir
    )

    with np.load(index.index_path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        embeddings = archive["embeddings"].copy()
    metadata["format_version"] = 1
    _save_numpy_archive_atomic(
        index.index_path,
        metadata,
        {"embeddings": embeddings},
    )

    with pytest.raises(VideoIndexCompatibilityError, match="format version: 1"):
        _load_video_index(index_dir)
