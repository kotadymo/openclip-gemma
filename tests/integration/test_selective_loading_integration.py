"""Real-model numerical comparisons for selective tower loading."""

from __future__ import annotations

import os

import pytest
import torch
from PIL import Image

from openclip_gemma import create_model, create_model_and_transforms, get_tokenizer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]


@pytest.mark.parametrize(
    ("precision", "dtype", "tolerance"),
    [
        ("fp32", torch.float32, 1e-4),
        ("bf16", torch.bfloat16, 2e-3),
    ],
)
def test_selective_loading_preserves_retained_embeddings(
    precision: str,
    dtype: torch.dtype,
    tolerance: float,
) -> None:
    reference, _, preprocess = create_model_and_transforms(
        precision=precision,
        device="cpu",
    )
    text_only = create_model(
        precision=precision,
        device="cpu",
        modalities=("text",),
    )
    image_text = create_model(
        precision=precision,
        device="cpu",
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
    )
    image = preprocess(Image.new("RGB", (256, 192), "red")).unsqueeze(0)

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
        assert features.device.type == "cpu"
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

    with pytest.raises(RuntimeError, match="without the image modality"):
        text_only.encode_image(image)
