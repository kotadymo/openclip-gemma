from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openvlclip.model import EmbeddingGemmaCLIP


class FakeProcessor:
    image_token = "<image>"

    def __init__(self) -> None:
        self.image_processor = SimpleNamespace(
            patch_size=4,
            max_soft_tokens=280,
            pooling_kernel_size=3,
        )
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        images: list[list[torch.Tensor]],
        text: list[str],
        padding: bool,
        do_resize: bool,
        do_rescale: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        self.calls.append(
            {
                "images": images,
                "text": text,
                "padding": padding,
                "do_resize": do_resize,
                "do_rescale": do_rescale,
                "return_tensors": return_tensors,
            }
        )
        batch_size = len(images)
        pixel_values = torch.zeros((batch_size, 3, 768), dtype=torch.float32)
        for index, image_group in enumerate(images):
            pixel_values[index, 0, 0] = image_group[0].mean()
            pixel_values[index, 1, 1] = 1
            pixel_values[index, 2, 2] = 100
        return {
            "input_ids": torch.ones((batch_size, 3), dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 0]] * batch_size,
                dtype=torch.long,
            ),
            "pixel_values": pixel_values,
            "image_position_ids": torch.zeros(
                (batch_size, 3, 2),
                dtype=torch.long,
            ),
        }


class FakeBackend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor,
    ) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=pixel_values * self.scale)


@pytest.fixture
def model_and_processor() -> tuple[EmbeddingGemmaCLIP, FakeProcessor]:
    processor = FakeProcessor()
    model = EmbeddingGemmaCLIP(FakeBackend(), processor).eval()
    return model, processor


def make_batch(values: list[float]) -> torch.Tensor:
    return torch.stack(
        [torch.full((3, 8, 8), value, dtype=torch.float32) for value in values]
    )


def test_encode_image_single_image_returns_native_normalized_embedding(
    model_and_processor: tuple[EmbeddingGemmaCLIP, FakeProcessor],
) -> None:
    model, processor = model_and_processor
    image = make_batch([0.25])

    result = model.encode_image(image)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 768)
    assert result.device == image.device
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    assert not result.requires_grad
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(1))
    expected = torch.zeros((1, 768))
    expected[0, :2] = torch.tensor([0.25, 1.0])
    torch.testing.assert_close(result, torch.nn.functional.normalize(expected, dim=-1))

    call = processor.calls[0]
    assert call["text"] == ["<image>"]
    assert call["padding"] is True
    assert call["do_resize"] is False
    assert call["do_rescale"] is False
    assert call["return_tensors"] == "pt"


def test_encode_image_preserves_batch_order_and_normalize_argument(
    model_and_processor: tuple[EmbeddingGemmaCLIP, FakeProcessor],
) -> None:
    model, _ = model_and_processor
    batch = make_batch([0.2, 0.8])

    batched = model.encode_image(batch)
    expected_order = torch.cat(
        [model.encode_image(batch[index : index + 1]) for index in range(2)]
    )
    explicitly_normalized = model.encode_image(batch, normalize=True)

    assert batched.shape == (2, 768)
    torch.testing.assert_close(batched, expected_order)
    torch.testing.assert_close(explicitly_normalized, batched)
    torch.testing.assert_close(
        explicitly_normalized.norm(dim=-1),
        torch.ones(2),
    )


@pytest.mark.parametrize("truncate_dim", [128, 256, 512, 768])
def test_encode_image_truncates_and_renormalizes(
    model_and_processor: tuple[EmbeddingGemmaCLIP, FakeProcessor],
    truncate_dim: int,
) -> None:
    model, _ = model_and_processor
    image = make_batch([0.5])
    native = model.encode_image(image)

    result = model.encode_image(image, truncate_dim=truncate_dim)

    assert result.shape == (1, truncate_dim)
    torch.testing.assert_close(
        result,
        torch.nn.functional.normalize(
            native[:, :truncate_dim],
            p=2,
            dim=-1,
        ),
    )
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(1))
    if truncate_dim == 768:
        assert torch.equal(result, native)


@pytest.mark.parametrize(
    ("truncate_dim", "exception", "message"),
    [
        (127, ValueError, "one of"),
        (1024, ValueError, "one of"),
        (True, TypeError, "integer"),
        (128.0, TypeError, "integer"),
        ("128", TypeError, "integer"),
    ],
)
def test_encode_image_rejects_invalid_truncate_dimensions(
    model_and_processor: tuple[EmbeddingGemmaCLIP, FakeProcessor],
    truncate_dim: object,
    exception: type[Exception],
    message: str,
) -> None:
    model, _ = model_and_processor

    with pytest.raises(exception, match=message):
        model.encode_image(
            make_batch([0.5]),
            truncate_dim=truncate_dim,  # type: ignore[arg-type]
        )


def test_encode_image_rejects_non_tensor_input(
    model_and_processor: tuple[EmbeddingGemmaCLIP, FakeProcessor],
) -> None:
    model, _ = model_and_processor

    with pytest.raises(TypeError, match="torch.Tensor"):
        model.encode_image([[0.0]])  # type: ignore[arg-type]


def test_encode_image_rejects_processor_mapping_for_embeddinggemma(
    model_and_processor: tuple[EmbeddingGemmaCLIP, FakeProcessor],
) -> None:
    model, _ = model_and_processor

    with pytest.raises(TypeError, match="processor-aware image mappings"):
        model.encode_image({"pixel_values": torch.zeros((1, 3, 224, 224))})


@pytest.mark.parametrize(
    ("image", "exception", "message"),
    [
        (torch.zeros((3, 8, 8)), ValueError, r"\[B,3,H,W\]"),
        (torch.zeros((1, 3, 8, 8, 1)), ValueError, r"\[B,3,H,W\]"),
        (torch.zeros((1, 1, 8, 8)), ValueError, "3 channels"),
        (torch.zeros((0, 3, 8, 8)), ValueError, "at least one"),
        (torch.zeros((1, 3, 0, 8)), ValueError, "height and width"),
        (torch.zeros((1, 3, 7, 8)), ValueError, "multiples of patch_size"),
    ],
)
def test_encode_image_rejects_invalid_shapes(
    model_and_processor: tuple[EmbeddingGemmaCLIP, FakeProcessor],
    image: torch.Tensor,
    exception: type[Exception],
    message: str,
) -> None:
    model, _ = model_and_processor

    with pytest.raises(exception, match=message):
        model.encode_image(image)


@pytest.mark.parametrize(
    "dtype",
    [torch.uint8, torch.float16, torch.bfloat16, torch.float64],
)
def test_encode_image_rejects_unsupported_dtypes(
    model_and_processor: tuple[EmbeddingGemmaCLIP, FakeProcessor],
    dtype: torch.dtype,
) -> None:
    model, _ = model_and_processor

    with pytest.raises(TypeError, match="torch.float32"):
        model.encode_image(torch.zeros((1, 3, 8, 8), dtype=dtype))


@pytest.mark.parametrize(
    ("value", "message"),
    [(float("nan"), "NaN or infinite"), (float("inf"), "NaN or infinite")],
)
def test_encode_image_rejects_non_finite_values(
    model_and_processor: tuple[EmbeddingGemmaCLIP, FakeProcessor],
    value: float,
    message: str,
) -> None:
    model, _ = model_and_processor
    image = make_batch([0.5])
    image[0, 0, 0, 0] = value

    with pytest.raises(ValueError, match=message):
        model.encode_image(image)


@pytest.mark.parametrize(
    ("value", "message"),
    [(-0.01, r"\[0,1\]"), (1.01, r"\[0,1\]")],
)
def test_encode_image_rejects_values_outside_preprocess_range(
    model_and_processor: tuple[EmbeddingGemmaCLIP, FakeProcessor],
    value: float,
    message: str,
) -> None:
    model, _ = model_and_processor
    image = make_batch([0.5])
    image[0, 0, 0, 0] = value

    with pytest.raises(ValueError, match=message):
        model.encode_image(image)


def test_encode_image_rejects_device_mismatch(
    model_and_processor: tuple[EmbeddingGemmaCLIP, FakeProcessor],
) -> None:
    model, _ = model_and_processor
    image = torch.empty((1, 3, 8, 8), dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match="model backend is on"):
        model.encode_image(image)


def test_encode_image_rejects_non_bool_normalize(
    model_and_processor: tuple[EmbeddingGemmaCLIP, FakeProcessor],
) -> None:
    model, _ = model_and_processor

    with pytest.raises(TypeError, match="normalize must be a bool"):
        model.encode_image(make_batch([0.5]), normalize=1)  # type: ignore[arg-type]
