"""Integration checks against the pinned EmbeddingGemma 2 checkpoint."""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
from PIL import Image, ImageDraw

from openvlclip import EmbeddingGemmaCLIP, create_model_and_transforms
from openvlclip.factory import MODEL_ID, MODEL_REVISION

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]


def _m01_reference_preprocess(
    image: Image.Image,
    processor: Any,
) -> torch.Tensor:
    image_processor = processor.image_processor
    converted = image_processor.process_image(
        image,
        do_convert_rgb=image_processor.do_convert_rgb,
    )
    resized = image_processor.aspect_ratio_preserving_resize(
        image=converted,
        patch_size=image_processor.patch_size,
        max_patches=(
            image_processor.max_soft_tokens * image_processor.pooling_kernel_size**2
        ),
        pooling_kernel_size=image_processor.pooling_kernel_size,
        resample=image_processor.resample,
    )
    return image_processor.rescale_and_normalize(
        resized,
        do_rescale=image_processor.do_rescale,
        rescale_factor=image_processor.rescale_factor,
        do_normalize=image_processor.do_normalize,
        image_mean=image_processor.image_mean,
        image_std=image_processor.image_std,
    )


def _sample_image() -> Image.Image:
    image = Image.new("RGB", (256, 192), (30, 80, 180))
    ImageDraw.Draw(image).ellipse((72, 40, 184, 152), fill=(220, 35, 45))
    return image


def test_real_model_wrapper_and_m01_preprocess() -> None:
    model, preprocess_train, preprocess_val = create_model_and_transforms(device="cpu")

    assert isinstance(model, EmbeddingGemmaCLIP)
    assert preprocess_train is preprocess_val
    assert isinstance(model.backend, torch.nn.Module)
    assert model.backend in tuple(model.children())
    assert model.processor is not None
    assert not any(child is model.processor for child in model.children())
    assert MODEL_ID == "google/embeddinggemma-2"
    assert MODEL_REVISION == "914f7f89142e33e77833254d9c9b90c3cef7303b"

    assert model.eval() is model
    assert not model.training
    assert not model.backend.training
    assert model.to("cpu") is model
    assert next(model.backend.parameters()).device.type == "cpu"

    image = _sample_image()
    actual = preprocess_val(image)
    reference = _m01_reference_preprocess(image, model.processor)
    assert torch.equal(actual, reference)

    pil_features = model.processor(images=image, return_tensors="pt")
    tensor_features = model.processor(
        images=[actual],
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    )
    assert set(pil_features) == set(tensor_features)
    for name in pil_features:
        if isinstance(pil_features[name], torch.Tensor):
            assert torch.equal(pil_features[name], tensor_features[name]), name
