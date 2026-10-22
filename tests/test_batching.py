from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from openvlclip.backends.base import ProcessedImageBatch
from openvlclip.batching import _iter_preprocessed_embeddings, iter_image_embeddings


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.batch_sizes: list[int] = []
        self.inference_mode_states: list[bool] = []

    def encode_image(
        self,
        images: torch.Tensor,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        self.batch_sizes.append(images.shape[0])
        self.inference_mode_states.append(torch.is_inference_mode_enabled())
        embedding_dim = 768 if truncate_dim is None else truncate_dim
        embeddings = torch.zeros(
            (images.shape[0], embedding_dim),
            dtype=torch.float32,
            device=images.device,
        )
        embeddings[:, 0] = images[:, 0, 0, 0]
        embeddings[:, 1] = 1
        return embeddings


class FakeDinoEncoder(FakeEncoder):
    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int:
        if value is None:
            return 2048
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")
        if value != 2048:
            raise ValueError(
                f"{name} must be 2048; dino.txt does not support MRL or truncation."
            )
        return value

    def encode_image(
        self,
        images: torch.Tensor,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        self.batch_sizes.append(images.shape[0])
        self.inference_mode_states.append(torch.is_inference_mode_enabled())
        embedding_dim = 2048 if truncate_dim is None else truncate_dim
        return torch.zeros(
            (images.shape[0], embedding_dim),
            dtype=torch.float32,
            device=images.device,
        )


class FakeMappingEncoder(FakeEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.received_batches: list[dict[str, torch.Tensor]] = []

    def encode_image(
        self,
        images: ProcessedImageBatch,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        assert isinstance(images, Mapping)
        batch = dict(images)
        self.received_batches.append(batch)
        self.batch_sizes.append(batch["pixel_values"].shape[0])
        self.inference_mode_states.append(torch.is_inference_mode_enabled())
        embedding_dim = 768 if truncate_dim is None else truncate_dim
        features = torch.zeros(
            (batch["pixel_values"].shape[0], embedding_dim),
            dtype=torch.float32,
            device=batch["pixel_values"].device,
        )
        features[:, 0] = batch["pixel_values"][:, 0, 0]
        features[:, 1] = 1
        return features


class OutOfMemoryEncoder(FakeEncoder):
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")


def _preprocess(image: Image.Image) -> torch.Tensor:
    pixel = image.getpixel((0, 0))
    first_channel = pixel[0] if isinstance(pixel, tuple) else pixel
    red = float(first_channel) / 255
    return torch.full(
        (3, image.height, image.width),
        red,
        dtype=torch.float32,
    )


def _mapping_preprocess(image: Image.Image) -> dict[str, torch.Tensor]:
    pixel = image.getpixel((0, 0))
    first_channel = pixel[0] if isinstance(pixel, tuple) else pixel
    red = float(first_channel) / 255
    mask = torch.zeros((256,), dtype=torch.long)
    mask[0] = 1
    return {
        "pixel_values": torch.full((256, 768), red, dtype=torch.float32),
        "pixel_attention_mask": mask,
        "spatial_shapes": torch.tensor((1, 1), dtype=torch.long),
    }


def _write_image(
    path: Path,
    *,
    red: int,
    size: tuple[int, int] = (8, 8),
) -> Path:
    Image.new("RGB", size, (red, 0, 0)).save(path)
    return path


def test_streams_bounded_batches_and_restores_order_for_different_shapes(
    tmp_path: Path,
) -> None:
    paths = [
        _write_image(tmp_path / "first.png", red=51),
        _write_image(tmp_path / "second.png", red=102, size=(12, 8)),
        _write_image(tmp_path / "third.png", red=153),
    ]
    model = FakeEncoder()

    results = list(
        iter_image_embeddings(
            model,
            _preprocess,
            iter(paths),
            batch_size=3,
        )
    )

    assert [path for path, _ in results] == paths
    assert [feature.shape for _, feature in results] == [(768,)] * 3
    assert all(feature.device.type == "cpu" for _, feature in results)
    torch.testing.assert_close(
        torch.stack([feature[0] for _, feature in results]),
        torch.tensor([51, 102, 153], dtype=torch.float32) / 255,
    )
    assert model.batch_sizes == [2, 1]
    assert model.inference_mode_states == [True, True]


def test_defaults_to_dinotxt_native_embedding_dimension(
    tmp_path: Path,
) -> None:
    path = _write_image(tmp_path / "image.png", red=100)

    results = list(iter_image_embeddings(FakeDinoEncoder(), _preprocess, [path]))

    assert results[0][1].shape == (2048,)


def test_file_batching_preserves_processor_mapping_fields_and_order(
    tmp_path: Path,
) -> None:
    paths = [
        _write_image(tmp_path / "first.png", red=51),
        _write_image(tmp_path / "second.png", red=102),
        _write_image(tmp_path / "third.png", red=153),
    ]
    model = FakeMappingEncoder()

    results = list(
        iter_image_embeddings(
            model,
            _mapping_preprocess,
            paths,
            batch_size=2,
        )
    )

    assert [path for path, _ in results] == paths
    assert [feature.shape for _, feature in results] == [(768,)] * 3
    torch.testing.assert_close(
        torch.stack([feature[0] for _, feature in results]),
        torch.tensor([51, 102, 153], dtype=torch.float32) / 255,
    )
    assert model.batch_sizes == [2, 1]
    assert all(
        tuple(batch["pixel_values"].shape) == (size, 256, 768)
        for batch, size in zip(model.received_batches, (2, 1), strict=True)
    )
    assert all(
        tuple(batch["pixel_attention_mask"].shape) == (size, 256)
        for batch, size in zip(model.received_batches, (2, 1), strict=True)
    )
    assert all(
        tuple(batch["spatial_shapes"].shape) == (size, 2)
        for batch, size in zip(model.received_batches, (2, 1), strict=True)
    )


def test_preprocessed_batch_defaults_to_dinotxt_native_dimension() -> None:
    items = [("frame-1", torch.zeros((3, 8, 8), dtype=torch.float32))]

    results = list(_iter_preprocessed_embeddings(FakeDinoEncoder(), items))

    assert results[0][0] == "frame-1"
    assert results[0][1].shape == (2048,)


def test_explicit_dinotxt_native_embedding_dimension_is_accepted(
    tmp_path: Path,
) -> None:
    path = _write_image(tmp_path / "image.png", red=100)

    results = list(
        iter_image_embeddings(
            FakeDinoEncoder(),
            _preprocess,
            [path],
            embedding_dim=2048,
        )
    )

    assert results[0][1].shape == (2048,)


def test_preprocessed_generic_encoder_keeps_768_dimension_fallback() -> None:
    items = [("frame-1", torch.zeros((3, 8, 8), dtype=torch.float32))]

    results = list(_iter_preprocessed_embeddings(FakeEncoder(), items))

    assert results[0][1].shape == (768,)


@pytest.mark.parametrize("embedding_dim", [768, 1024])
def test_rejects_non_native_dinotxt_embedding_dimensions(
    embedding_dim: int,
) -> None:
    with pytest.raises(ValueError, match="does not support MRL"):
        iter_image_embeddings(
            FakeDinoEncoder(),
            _preprocess,
            [],
            embedding_dim=embedding_dim,
        )


def test_batch_size_bounds_each_encoded_batch(tmp_path: Path) -> None:
    paths = [
        _write_image(tmp_path / f"{index}.png", red=index * 30) for index in range(1, 6)
    ]
    model = FakeEncoder()

    results = list(
        iter_image_embeddings(
            model,
            _preprocess,
            paths,
            batch_size=2,
        )
    )

    assert len(results) == len(paths)
    assert model.batch_sizes == [2, 2, 1]


@pytest.mark.parametrize("embedding_dim", [128, 256, 384, 512, 768])
def test_streams_requested_embedding_dimension(
    tmp_path: Path,
    embedding_dim: int,
) -> None:
    path = _write_image(tmp_path / "image.png", red=100)

    results = list(
        iter_image_embeddings(
            FakeEncoder(),
            _preprocess,
            [path],
            embedding_dim=embedding_dim,
        )
    )

    assert results[0][1].shape == (embedding_dim,)


@pytest.mark.parametrize(
    ("embedding_dim", "exception"),
    [(0, ValueError), (-1, ValueError), (True, TypeError)],
)
def test_rejects_non_positive_or_non_integer_embedding_dimensions(
    embedding_dim: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception, match="embedding_dim"):
        iter_image_embeddings(
            FakeEncoder(),
            _preprocess,
            [],
            embedding_dim=embedding_dim,  # type: ignore[arg-type]
        )


def test_generic_encoder_accepts_a_model_specific_dimension(
    tmp_path: Path,
) -> None:
    path = _write_image(tmp_path / "image.png", red=100)

    results = list(
        iter_image_embeddings(
            FakeEncoder(),
            _preprocess,
            [path],
            embedding_dim=384,
        )
    )

    assert results[0][1].shape == (384,)


@pytest.mark.parametrize("suffix", [".jpg", ".png"])
def test_reads_common_pillow_image_formats(tmp_path: Path, suffix: str) -> None:
    path = _write_image(tmp_path / f"image{suffix}", red=100)

    results = list(iter_image_embeddings(FakeEncoder(), _preprocess, [path]))

    assert len(results) == 1
    assert results[0][0] == path
    assert results[0][1].shape == (768,)


def test_empty_input_returns_no_results_without_calling_model() -> None:
    model = FakeEncoder()

    assert list(iter_image_embeddings(model, _preprocess, [], batch_size=4)) == []
    assert model.batch_sizes == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_rejects_non_positive_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        iter_image_embeddings(FakeEncoder(), _preprocess, [], batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [True, 1.5, "2"])
def test_rejects_non_integer_batch_size(batch_size: object) -> None:
    with pytest.raises(TypeError, match="positive integer"):
        iter_image_embeddings(
            FakeEncoder(),
            _preprocess,
            [],
            batch_size=batch_size,  # type: ignore[arg-type]
        )


def test_rejects_single_path_instead_of_path_iterable(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="iterable of paths"):
        iter_image_embeddings(
            FakeEncoder(),
            _preprocess,
            tmp_path / "image.png",  # type: ignore[arg-type]
        )


def test_corrupt_image_fails_with_input_index_and_path(tmp_path: Path) -> None:
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"not an image")

    with pytest.raises(
        ValueError,
        match=rf"input index 0.*{re.escape(str(corrupt_path))}",
    ):
        list(
            iter_image_embeddings(
                FakeEncoder(),
                _preprocess,
                [corrupt_path],
            )
        )


def test_missing_image_fails_with_input_index_and_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.png"

    with pytest.raises(
        ValueError,
        match=rf"input index 0.*{re.escape(str(missing_path))}",
    ):
        list(
            iter_image_embeddings(
                FakeEncoder(),
                _preprocess,
                [missing_path],
            )
        )


def test_cuda_out_of_memory_error_suggests_lower_batch_size(tmp_path: Path) -> None:
    path = _write_image(tmp_path / "image.png", red=100)

    with pytest.raises(RuntimeError, match="CUDA ran out of memory.*reduce batch_size"):
        list(
            iter_image_embeddings(
                OutOfMemoryEncoder(),
                _preprocess,
                [path],
                batch_size=1,
            )
        )
