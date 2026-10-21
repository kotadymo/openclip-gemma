"""Real-model numerical comparisons for selective tower loading."""

from __future__ import annotations

import os

import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from openvlclip import create_model, create_model_and_transforms, get_tokenizer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]


@pytest.mark.parametrize(
    ("device_name", "precision", "dtype", "tolerance"),
    [
        ("cpu", "fp32", torch.float32, 1e-4),
        ("cpu", "bf16", torch.bfloat16, 2e-3),
        pytest.param(
            "cuda",
            "fp32",
            torch.float32,
            1e-4,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable.",
            ),
        ),
        pytest.param(
            "cuda",
            "bf16",
            torch.bfloat16,
            2e-3,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
                reason="CUDA BF16 is unavailable.",
            ),
        ),
    ],
)
def test_selective_loading_preserves_retained_embeddings(
    device_name: str,
    precision: str,
    dtype: torch.dtype,
    tolerance: float,
) -> None:
    reference, _, preprocess = create_model_and_transforms(
        precision=precision,
        device=device_name,
    )
    text_only = create_model(
        precision=precision,
        device=device_name,
        modalities=("text",),
    )
    image_text = create_model(
        precision=precision,
        device=device_name,
        modalities=("text", "image"),
    )

    assert reference.backend.vision_tower is not None
    assert reference.backend.audio_tower is not None
    assert reference.backend.language_model is not None
    assert text_only.backend.vision_tower is None
    assert text_only.backend.audio_tower is None
    assert text_only.backend.language_model is not None
    assert image_text.backend.vision_tower is not None
    assert image_text.backend.audio_tower is None
    assert image_text.backend.language_model is not None

    tokens = get_tokenizer(prompt_name="SearchQuery")(
        "A red circle on a blue background."
    ).to(device_name)
    image = preprocess(Image.new("RGB", (256, 192), "red")).unsqueeze(0).to(device_name)

    with torch.inference_mode():
        reference_text = reference.encode_text(tokens)
        text_only_features = text_only.encode_text(tokens)
        image_text_features = image_text.encode_text(tokens)
        reference_image = reference.encode_image(image)
        image_text_image = image_text.encode_image(image)

    for features in (
        reference_text,
        text_only_features,
        image_text_features,
        reference_image,
        image_text_image,
    ):
        assert features.dtype == dtype
        assert features.device.type == device_name
        assert torch.isfinite(features).all()
    torch.testing.assert_close(
        text_only_features,
        reference_text,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        image_text_features,
        reference_text,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        image_text_image,
        reference_image,
        atol=tolerance,
        rtol=tolerance,
    )

    for embedding_dim in (512, 256, 128):
        with torch.inference_mode():
            reference_text_mrl = reference.encode_text(
                tokens,
                truncate_dim=embedding_dim,
            )
            text_only_mrl = text_only.encode_text(
                tokens,
                truncate_dim=embedding_dim,
            )
            image_text_mrl = image_text.encode_text(
                tokens,
                truncate_dim=embedding_dim,
            )
            reference_image_mrl = reference.encode_image(
                image,
                truncate_dim=embedding_dim,
            )
            image_text_image_mrl = image_text.encode_image(
                image,
                truncate_dim=embedding_dim,
            )

        expected_text_mrl = F.normalize(
            reference_text[:, :embedding_dim],
            p=2,
            dim=-1,
        )
        expected_image_mrl = F.normalize(
            reference_image[:, :embedding_dim],
            p=2,
            dim=-1,
        )
        text_outputs = (reference_text_mrl, text_only_mrl, image_text_mrl)
        image_outputs = (reference_image_mrl, image_text_image_mrl)
        for features in (*text_outputs, *image_outputs):
            assert features.shape == (1, embedding_dim)
            assert features.dtype == dtype
            assert features.device.type == device_name
            assert torch.isfinite(features).all()
            torch.testing.assert_close(
                features.float().norm(dim=-1),
                torch.ones(1, device=features.device),
                atol=tolerance,
                rtol=tolerance,
            )
        for features in text_outputs:
            torch.testing.assert_close(
                features,
                expected_text_mrl,
                atol=tolerance,
                rtol=tolerance,
            )
        for features in image_outputs:
            torch.testing.assert_close(
                features,
                expected_image_mrl,
                atol=tolerance,
                rtol=tolerance,
            )

    with pytest.raises(RuntimeError, match="without the image modality"):
        text_only.encode_image(image)
