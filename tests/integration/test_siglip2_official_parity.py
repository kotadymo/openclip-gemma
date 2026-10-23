"""Opt-in parity checks against the pinned official SigLIP 2 checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor

from openvlclip import create_model_and_transforms, get_tokenizer
from openvlclip.image_search import build_or_update_image_index, search_images

_CHECKPOINTS = (
    (
        "siglip2-base",
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "fixres",
        768,
    ),
    (
        "siglip2-naflex-base",
        "google/siglip2-base-patch16-naflex",
        "b53b807d3a2d5e2b3911292f2d69e5341cdc064c",
        "naflex",
        768,
    ),
    ("siglip2-base-384", "google/siglip2-base-patch16-384",
     "f775b65a79762255128c981547af89addcfe0f88", "fixres", 768),
    ("siglip2-large-256", "google/siglip2-large-patch16-256",
     "787800c8990e6f058423089178e718139608408c", "fixres", 1024),
    ("siglip2-so400m-384", "google/siglip2-so400m-patch14-384",
     "e8e487298228002f3d8a82e0cd5c8ea9c567f57f", "fixres", 1152),
)
_DEVICES_AND_PRECISIONS = (
    ("cpu", "fp32"),
    ("cuda", "fp32"),
    ("cuda", "bf16"),
)
_TEXTS = ("a red object", "a blue object")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]


def _sample_images() -> list[Image.Image]:
    return [
        Image.new("RGB", (320, 192), (210, 30, 40)),
        Image.new("RGB", (192, 320), (20, 50, 210)),
    ]


def _move_image_inputs(
    inputs: dict[str, torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(
            device=device,
            dtype=dtype if name == "pixel_values" else value.dtype,
        )
        for name, value in inputs.items()
    }


@pytest.mark.parametrize(
    ("model_name", "model_id", "revision", "variant", "dimension"),
    _CHECKPOINTS,
)
@pytest.mark.parametrize(("device", "precision"), _DEVICES_AND_PRECISIONS)
def test_siglip2_wrapper_matches_pinned_official_model(
    tmp_path: Path,
    model_name: str,
    model_id: str,
    revision: str,
    variant: str,
    dimension: int,
    device: str,
    precision: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    if device == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable.")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32

    model, _, preprocess = create_model_and_transforms(
        model_name,
        pretrained="official",
        precision=precision,
        device=device,
    )
    official_model = (
        AutoModel.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            use_safetensors=True,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    official_processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
    )
    images = _sample_images()
    processed_images = [preprocess(image) for image in images]
    reference_image_batch = official_processor(
        images=images,
        return_tensors="pt",
    )
    reference_image_inputs = _move_image_inputs(
        {
            key: value
            for key, value in reference_image_batch.items()
            if isinstance(value, torch.Tensor)
        },
        device=device,
        dtype=dtype,
    )

    if variant == "fixres":
        actual_pixels = torch.stack(processed_images)
        torch.testing.assert_close(
            actual_pixels,
            reference_image_batch["pixel_values"],
            rtol=0,
            atol=0,
        )
    else:
        assert all(isinstance(image, dict) for image in processed_images)
        for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes"):
            actual_values = torch.stack([image[key] for image in processed_images])
            torch.testing.assert_close(
                actual_values,
                reference_image_batch[key],
                rtol=0,
                atol=0,
            )

    reference_text_batch = official_processor(
        text=list(_TEXTS),
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    pad_token_id = official_processor.tokenizer.pad_token_id
    assert isinstance(pad_token_id, int)
    reference_attention_mask = reference_text_batch["input_ids"].ne(pad_token_id)
    token_ids = get_tokenizer(model_name)(list(_TEXTS))
    torch.testing.assert_close(
        token_ids,
        reference_text_batch["input_ids"],
        rtol=0,
        atol=0,
    )

    with torch.inference_mode():
        actual_images = model.encode_image(processed_images)
        actual_text = model.encode_text(token_ids)
        actual_images_normalized = model.encode_image(
            processed_images,
            normalize=True,
        )
        actual_text_normalized = model.encode_text(token_ids, normalize=True)

        reference_images = official_model.get_image_features(
            **reference_image_inputs
        ).pooler_output
        reference_text = official_model.get_text_features(
            input_ids=reference_text_batch["input_ids"].to(device),
            attention_mask=reference_attention_mask.to(device),
        ).pooler_output
        reference_images_normalized = F.normalize(reference_images, p=2, dim=-1)
        reference_text_normalized = F.normalize(reference_text, p=2, dim=-1)

        combined_inputs = official_processor(
            images=images,
            text=list(_TEXTS),
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        combined_inputs = {
            key: (
                value.to(
                    device=device,
                    dtype=dtype if key == "pixel_values" else value.dtype,
                )
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in combined_inputs.items()
        }
        combined_inputs["attention_mask"] = combined_inputs["input_ids"].ne(
            pad_token_id
        )
        reference_output = official_model(**combined_inputs)
        actual_logits = model.get_logits(processed_images, token_ids)

    tolerance = 2e-3 if precision == "bf16" else 1e-5
    for actual, reference in (
        (actual_images, reference_images),
        (actual_text, reference_text),
        (actual_images_normalized, reference_images_normalized),
        (actual_text_normalized, reference_text_normalized),
        (actual_logits[0], reference_output.logits_per_image),
        (actual_logits[1], reference_output.logits_per_text),
    ):
        torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
        assert torch.isfinite(actual).all()
    assert actual_images.shape == (len(images), dimension)
    assert actual_text.shape == (len(_TEXTS), dimension)
    assert actual_images_normalized.shape == (len(images), dimension)
    assert actual_text_normalized.shape == (len(_TEXTS), dimension)

    if device == "cpu" and precision == "fp32":
        image_dir = tmp_path / model_name / "images"
        image_dir.mkdir(parents=True)
        for index, image in enumerate(images):
            image.save(image_dir / f"{index}.png")
        update = build_or_update_image_index(
            model,
            preprocess,
            image_dir,
            tmp_path / model_name / "index",
            batch_size=2,
        )
        search_results = search_images(
            model,
            tmp_path / model_name / "index",
            "a red object",
            top_k=2,
        )
        assert update.added == len(images)
        assert len(search_results) == len(images)

    errors = {
        "image_max_abs_error": float((actual_images - reference_images).abs().max()),
        "text_max_abs_error": float((actual_text - reference_text).abs().max()),
        "logit_max_abs_error": float(
            (actual_logits[0] - reference_output.logits_per_image).abs().max()
        ),
    }
    print(
        json.dumps(
            {
                "model": model_name,
                "device": device,
                "precision": precision,
                "variant": variant,
                "max_abs_errors": errors,
            },
            sort_keys=True,
        )
    )
