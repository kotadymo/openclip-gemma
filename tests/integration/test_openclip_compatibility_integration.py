"""Run representative OpenCLIP-style inference calls against the pinned model."""

from __future__ import annotations

import json
import os

import pytest
import torch
from PIL import Image, ImageDraw

from openvlclip import (
    create_model_from_pretrained,
    get_tokenizer,
    list_models,
    list_pretrained,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]

DEVICE_PRECISIONS = [
    pytest.param("cpu", "fp32", torch.float32, id="cpu-fp32"),
    pytest.param("cpu", "bf16", torch.bfloat16, id="cpu-bf16"),
    pytest.param(
        "cuda",
        "fp32",
        torch.float32,
        id="cuda-fp32",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="CUDA is unavailable.",
        ),
    ),
    pytest.param(
        "cuda",
        "bf16",
        torch.bfloat16,
        id="cuda-bf16",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
            reason="CUDA BF16 is unavailable.",
        ),
    ),
]


def _sample_images() -> list[Image.Image]:
    images = []
    for background, foreground in (("navy", "red"), ("white", "green")):
        image = Image.new("RGB", (256, 256), background)
        ImageDraw.Draw(image).ellipse((64, 64, 192, 192), fill=foreground)
        images.append(image)
    return images


@pytest.mark.parametrize(("device", "precision", "dtype"), DEVICE_PRECISIONS)
def test_openclip_compatibility_scenarios(
    device: str,
    precision: str,
    dtype: torch.dtype,
) -> None:
    assert set(list_models()) == {
        "embeddinggemma-2",
        "dinov2-dinotxt",
        "siglip2-base",
        "siglip2-naflex-base",
        "qwen3-vl-embedding-2b",
        "qwen3-vl-embedding-8b",
    }
    assert set(list_pretrained()) == {
        ("embeddinggemma-2", "official"),
        ("dinov2-dinotxt", "official"),
        ("siglip2-base", "official"),
        ("siglip2-naflex-base", "official"),
        ("qwen3-vl-embedding-2b", "official"),
        ("qwen3-vl-embedding-8b", "official"),
    }
    assert set(list_pretrained(as_str=True)) == {
        "embeddinggemma-2:official",
        "dinov2-dinotxt:official",
        "siglip2-base:official",
        "siglip2-naflex-base:official",
        "qwen3-vl-embedding-2b:official",
        "qwen3-vl-embedding-8b:official",
    }

    model, preprocess = create_model_from_pretrained(
        "embeddinggemma-2",
        pretrained="official",
        precision=precision,
        device=device,
        output_dict=True,
        logit_scale_mode="cosine",
    )
    model.eval()
    image_batch = torch.stack([preprocess(image) for image in _sample_images()]).to(
        device
    )
    search_tokens = get_tokenizer(
        "embeddinggemma-2",
        prompt_name="SearchQuery",
    )(["a red circle", "a green circle", "a blue mountain"]).to(device)
    classification_tokens = get_tokenizer(
        "embeddinggemma-2",
        prompt_name="Classification",
    )(["a red object", "a green object"]).to(device)

    with torch.inference_mode():
        image_features = model.encode_image(image_batch, normalize=True)
        text_features = model.encode_text(search_tokens, normalize=True)
        classification_features = model.encode_text(
            classification_tokens,
            normalize=True,
        )
        similarities = image_features @ text_features.T
        classification_scores = image_features @ classification_features.T

        dict_output = model(image_batch, search_tokens)
        assert isinstance(dict_output, dict)
        assert set(dict_output) == {
            "image_features",
            "text_features",
            "logit_scale",
        }
        torch.testing.assert_close(dict_output["image_features"], image_features)
        torch.testing.assert_close(dict_output["text_features"], text_features)
        scale = dict_output["logit_scale"]
        assert isinstance(scale, torch.Tensor)
        assert scale.shape == ()
        assert scale.dtype == torch.float32
        assert scale.device.type == device
        assert scale.item() == 1.0

        model.output_dict = False
        tuple_output = model(image_batch, search_tokens)
        assert isinstance(tuple_output, tuple)
        assert len(tuple_output) == 3
        torch.testing.assert_close(tuple_output[0], image_features)
        torch.testing.assert_close(tuple_output[1], text_features)

        image_only = model(image=image_batch[:1])
        text_only = model(text=search_tokens[:1])
        assert image_only[0].shape == (1, 768)
        assert image_only[1] is None
        assert text_only[0] is None
        assert text_only[1].shape == (1, 768)

        image_logits, text_logits = model.get_logits(image_batch, search_tokens)
        assert image_logits.shape == (2, 3)
        assert text_logits.shape == (3, 2)
        torch.testing.assert_close(image_logits, similarities)
        torch.testing.assert_close(text_logits, similarities.T)
        assert classification_scores.argmax(dim=-1).shape == (2,)

    expected_dtype = dtype
    assert image_features.shape == (2, 768)
    assert text_features.shape == (3, 768)
    assert image_features.dtype == expected_dtype
    assert text_features.dtype == expected_dtype
    assert image_features.device.type == device
    assert text_features.device.type == device
    assert torch.isfinite(image_features).all()
    assert torch.isfinite(text_features).all()

    print(
        json.dumps(
            {
                "device": device,
                "precision": precision,
                "dtype": str(dtype),
                "image_shape": list(image_features.shape),
                "text_shape": list(text_features.shape),
                "similarity_shape": list(similarities.shape),
                "top_labels": classification_scores.argmax(dim=-1).cpu().tolist(),
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated() if device == "cuda" else None
                ),
            },
            sort_keys=True,
        )
    )
