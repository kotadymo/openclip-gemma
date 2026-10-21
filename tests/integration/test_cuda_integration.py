"""CUDA inference checks and basic device metrics."""

from __future__ import annotations

import json
import os
import time

import pytest
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from openvlclip import create_model, create_model_and_transforms, get_tokenizer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable."),
]

TEXTS = ["A red circle on a blue background.", "青い背景にある赤い円です。"]


def _sample_image() -> Image.Image:
    image = Image.new("RGB", (256, 192), (30, 80, 180))
    ImageDraw.Draw(image).ellipse((72, 40, 184, 152), fill=(220, 35, 45))
    return image


def _comparison(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference_fp32 = reference.detach().float()
    actual_fp32 = actual.detach().float()
    return {
        "max_abs_error": (reference_fp32 - actual_fp32).abs().max().item(),
        "cosine_similarity": F.cosine_similarity(
            reference_fp32,
            actual_fp32,
            dim=-1,
        )
        .mean()
        .item(),
    }


@pytest.mark.parametrize(
    ("precision", "max_abs_error", "min_cosine_similarity"),
    [
        ("fp32", 1e-3, 0.9999),
        ("bf16", 5e-2, 0.995),
    ],
)
def test_cuda_image_and_japanese_text_inference(
    precision: str,
    max_abs_error: float,
    min_cosine_similarity: float,
) -> None:
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("The current CUDA device does not support BF16.")

    cpu_model, _, preprocess = create_model_and_transforms(
        precision=precision,
        device="cpu",
    )
    load_started = time.perf_counter()
    cuda_model = create_model(device="cuda", precision=precision)
    model_load_seconds = time.perf_counter() - load_started

    image = preprocess(_sample_image()).unsqueeze(0)
    tokens = get_tokenizer(prompt_name="SearchQuery")(TEXTS)
    cpu_model.eval()
    cuda_model.eval()

    with torch.inference_mode():
        cpu_image_features = cpu_model.encode_image(image)
        cpu_text_features = cpu_model.encode_text(tokens)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        cuda_image_features = cuda_model.encode_image(image.to("cuda"))
        cuda_text_features = cuda_model.encode_text(tokens.to("cuda"))
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started

    for output, expected_shape in (
        (cuda_image_features, (1, 768)),
        (cuda_text_features, (len(TEXTS), 768)),
    ):
        assert output.shape == expected_shape
        assert output.device.type == "cuda"
        assert torch.isfinite(output).all()

    comparisons = {
        "image": _comparison(cpu_image_features, cuda_image_features.cpu()),
        "text": _comparison(cpu_text_features, cuda_text_features.cpu()),
    }
    for result in comparisons.values():
        assert result["max_abs_error"] <= max_abs_error
        assert result["cosine_similarity"] >= min_cosine_similarity

    print(
        json.dumps(
            {
                "precision": precision,
                "cuda_device": torch.cuda.get_device_name(),
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                "comparisons": comparisons,
            },
            sort_keys=True,
        )
    )
