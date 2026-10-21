from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from openclip_gemma.backends.embeddinggemma2 import _truncate_and_normalize
from openclip_gemma.model import EmbeddingGemmaCLIP


def _model(
    *,
    output_dict: bool = False,
    logit_scale_mode: str = "cosine",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EmbeddingGemmaCLIP:
    backend = nn.Linear(2, 2, device=device, dtype=dtype)
    return EmbeddingGemmaCLIP(
        backend,
        processor=object(),
        output_dict=output_dict,
        logit_scale_mode=logit_scale_mode,
    )


def _install_fake_encoders(
    model: EmbeddingGemmaCLIP,
    image_features: torch.Tensor | None,
    text_features: torch.Tensor | None,
) -> tuple[Mock, Mock]:
    image_encoder = Mock(return_value=image_features)
    text_encoder = Mock(return_value=text_features)
    model.encode_image = image_encoder  # type: ignore[method-assign]
    model.encode_text = text_encoder  # type: ignore[method-assign]
    return image_encoder, text_encoder


def test_forward_tuple_has_features_and_float32_scalar_scale() -> None:
    model = _model()
    image_features = F.normalize(torch.randn(2, 768), dim=-1)
    text_features = F.normalize(torch.randn(3, 768), dim=-1)
    image_encoder, text_encoder = _install_fake_encoders(
        model,
        image_features,
        text_features,
    )

    output = model(torch.empty((2, 3, 4, 4)), torch.empty((3, 5), dtype=torch.long))

    assert isinstance(output, tuple)
    assert len(output) == 3
    torch.testing.assert_close(output[0], image_features)
    torch.testing.assert_close(output[1], text_features)
    scale = output[2]
    assert scale.shape == ()
    assert scale.dtype == torch.float32
    assert scale.device.type == "cpu"
    assert scale.item() == 1.0
    assert "logit_scale" not in dict(model.named_parameters())
    image_encoder.assert_called_once()
    text_encoder.assert_called_once()
    assert image_encoder.call_args.kwargs == {
        "normalize": True,
        "truncate_dim": None,
    }
    assert text_encoder.call_args.kwargs == {
        "normalize": True,
        "truncate_dim": None,
    }


@pytest.mark.parametrize("image_only", [True, False])
def test_forward_supports_single_modality(
    image_only: bool,
) -> None:
    model = _model(output_dict=True)
    image_features = torch.ones((2, 768)) if image_only else None
    text_features = torch.ones((3, 768)) if not image_only else None
    _install_fake_encoders(model, image_features, text_features)

    output = model(
        torch.empty((2, 3, 4, 4)) if image_only else None,
        torch.empty((3, 5), dtype=torch.long) if not image_only else None,
    )

    assert isinstance(output, dict)
    assert output["image_features"] is image_features
    assert output["text_features"] is text_features
    scale = output["logit_scale"]
    assert isinstance(scale, torch.Tensor)
    assert scale.shape == ()
    assert scale.dtype == torch.float32
    assert scale.item() == 1.0


def test_forward_accepts_different_image_and_text_batch_sizes() -> None:
    model = _model()
    image_features = torch.ones((2, 768))
    text_features = torch.ones((5, 768))
    _install_fake_encoders(model, image_features, text_features)

    output = model(torch.empty((2, 3, 4, 4)), torch.empty((5, 5), dtype=torch.long))

    assert output[0].shape == (2, 768)
    assert output[1].shape == (5, 768)


@pytest.mark.parametrize("truncate_dim", [128, 256, 512, 768])
def test_forward_uses_one_embedding_dimension_for_both_modalities(
    truncate_dim: int,
) -> None:
    model = _model()
    image_features = torch.ones((2, truncate_dim))
    text_features = torch.ones((3, truncate_dim))
    image_encoder, text_encoder = _install_fake_encoders(
        model,
        image_features,
        text_features,
    )

    output = model(
        torch.empty((2, 3, 4, 4)),
        torch.empty((3, 5), dtype=torch.long),
        truncate_dim=truncate_dim,
    )

    assert output[0].shape == (2, truncate_dim)
    assert output[1].shape == (3, truncate_dim)
    image_encoder.assert_called_once()
    text_encoder.assert_called_once()
    assert image_encoder.call_args.kwargs == {
        "normalize": True,
        "truncate_dim": truncate_dim,
    }
    assert text_encoder.call_args.kwargs == {
        "normalize": True,
        "truncate_dim": truncate_dim,
    }


@pytest.mark.parametrize(
    ("truncate_dim", "exception"),
    [(True, TypeError), (0, ValueError), (129, ValueError)],
)
def test_forward_rejects_invalid_embedding_dimensions_before_encoding(
    truncate_dim: object,
    exception: type[Exception],
) -> None:
    model = _model()
    image_encoder, text_encoder = _install_fake_encoders(model, None, None)

    with pytest.raises(exception, match="truncate_dim"):
        model(torch.empty((1, 3, 4, 4)), truncate_dim=truncate_dim)  # type: ignore[arg-type]

    image_encoder.assert_not_called()
    text_encoder.assert_not_called()


def test_forward_rejects_both_inputs_missing_before_strict_mode() -> None:
    model = _model(logit_scale_mode="strict")
    image_encoder, text_encoder = _install_fake_encoders(model, None, None)

    with pytest.raises(ValueError, match="At least one"):
        model()

    image_encoder.assert_not_called()
    text_encoder.assert_not_called()


def test_strict_mode_rejects_forward_and_get_logits_without_encoding() -> None:
    model = _model(logit_scale_mode="strict")
    image_encoder, text_encoder = _install_fake_encoders(model, None, None)
    image = torch.empty((1, 3, 4, 4))
    text = torch.empty((1, 5), dtype=torch.long)

    with pytest.raises(NotImplementedError, match="logit_scale_mode='cosine'"):
        model(image, text)
    with pytest.raises(NotImplementedError, match="logit_scale_mode='cosine'"):
        model.get_logits(image, text)

    image_encoder.assert_not_called()
    text_encoder.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"output_dict": 1}, TypeError, "output_dict must be a bool"),
        ({"logit_scale_mode": 1}, TypeError, "logit_scale_mode must be a string"),
        ({"logit_scale_mode": "scaled"}, ValueError, "logit_scale_mode"),
    ],
)
def test_model_rejects_invalid_compatibility_options(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        _model(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", torch.float32),
        ("cpu", torch.bfloat16),
        pytest.param(
            "cuda",
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable for the device contract test.",
            ),
        ),
        pytest.param(
            "cuda",
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
                reason="CUDA BF16 is unavailable for the device contract test.",
            ),
        ),
    ],
)
def test_forward_scale_uses_model_device_and_fp32_for_supported_precisions(
    device: str,
    dtype: torch.dtype,
) -> None:
    model = _model(device=device, dtype=dtype)
    image_features = torch.ones((1, 768), device=device, dtype=dtype)
    _install_fake_encoders(model, image_features, None)

    output = model(torch.empty((1, 3, 4, 4), device=device))

    scale = output[2]
    assert scale.device.type == device
    assert scale.dtype == torch.float32
    assert scale.item() == 1.0


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", torch.float32),
        ("cpu", torch.bfloat16),
        pytest.param(
            "cuda",
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable for the logits contract test.",
            ),
        ),
        pytest.param(
            "cuda",
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
                reason="CUDA BF16 is unavailable for the logits contract test.",
            ),
        ),
    ],
)
def test_get_logits_returns_cosine_matrix_and_its_transpose(
    device: str,
    dtype: torch.dtype,
) -> None:
    model = _model(device=device, dtype=dtype, output_dict=True)
    image_features = F.normalize(
        torch.randn((2, 768), dtype=dtype, device=device),
        dim=-1,
    )
    text_features = F.normalize(
        torch.randn((3, 768), dtype=dtype, device=device),
        dim=-1,
    )
    image_encoder, text_encoder = _install_fake_encoders(
        model,
        image_features,
        text_features,
    )

    image_logits, text_logits = model.get_logits(
        torch.empty((2, 3, 4, 4), device=device),
        torch.empty((3, 5), dtype=torch.long, device=device),
    )

    if device == "cpu" and dtype == torch.bfloat16:
        expected = (image_features.float() @ text_features.float().T).to(dtype=dtype)
    else:
        expected = image_features @ text_features.T
    assert image_logits.shape == (2, 3)
    assert text_logits.shape == (3, 2)
    torch.testing.assert_close(image_logits, expected)
    torch.testing.assert_close(text_logits, expected.T)
    image_encoder.assert_called_once()
    text_encoder.assert_called_once()


@pytest.mark.parametrize("truncate_dim", [128, 256, 512, 768])
def test_get_logits_uses_truncated_feature_dimension(truncate_dim: int) -> None:
    model = _model()
    image_features = F.normalize(torch.randn((2, truncate_dim)), dim=-1)
    text_features = F.normalize(torch.randn((3, truncate_dim)), dim=-1)
    image_encoder, text_encoder = _install_fake_encoders(
        model,
        image_features,
        text_features,
    )

    image_logits, text_logits = model.get_logits(
        torch.empty((2, 3, 4, 4)),
        torch.empty((3, 5), dtype=torch.long),
        truncate_dim=truncate_dim,
    )

    expected = image_features @ text_features.T
    torch.testing.assert_close(image_logits, expected)
    torch.testing.assert_close(text_logits, expected.T)
    image_encoder.assert_called_once()
    text_encoder.assert_called_once()
    assert image_encoder.call_args.kwargs == {
        "normalize": True,
        "truncate_dim": truncate_dim,
    }
    assert text_encoder.call_args.kwargs == {
        "normalize": True,
        "truncate_dim": truncate_dim,
    }


def test_truncation_rejects_an_embedding_with_zero_prefix_norm() -> None:
    embeddings = torch.zeros((1, 768))
    embeddings[0, 128] = 1.0

    with pytest.raises(RuntimeError, match="after truncation to 128"):
        _truncate_and_normalize(
            embeddings,
            truncate_dim=128,
            modality="text",
        )


@pytest.mark.parametrize(
    ("image_features", "text_features", "exception", "message"),
    [
        (
            torch.ones((1, 768)),
            torch.ones((1, 767)),
            RuntimeError,
            "768 dimensions",
        ),
        (
            torch.ones((1, 768)),
            torch.full((1, 768), float("inf")),
            ValueError,
            "finite",
        ),
        (
            torch.ones((1, 768)),
            torch.ones((1, 768), dtype=torch.bfloat16),
            TypeError,
            "same dtype",
        ),
        (
            torch.ones((1, 768)),
            torch.empty((1, 768), device="meta"),
            ValueError,
            "same device",
        ),
    ],
)
def test_get_logits_rejects_invalid_encoded_feature_contracts(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    exception: type[Exception],
    message: str,
) -> None:
    model = _model()
    _install_fake_encoders(model, image_features, text_features)

    with pytest.raises(exception, match=message):
        model.get_logits(
            torch.empty((1, 3, 4, 4)),
            torch.empty((1, 5), dtype=torch.long),
        )
