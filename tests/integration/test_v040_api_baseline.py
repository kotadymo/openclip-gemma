"""Compare current inference results with the v0.4.0 real-model baseline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from openvlclip import create_model_and_transforms, get_tokenizer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
BASELINE_MANIFEST = FIXTURE_DIR / "v0.4.0_api_baseline.json"
BASELINE_EMBEDDINGS = FIXTURE_DIR / "v0.4.0_api_baseline.npz"


def _create_image(image_spec: dict[str, Any]) -> Image.Image:
    image = Image.new(
        "RGB",
        tuple(image_spec["size"]),
        tuple(image_spec["background"]),
    )
    ImageDraw.Draw(image).ellipse(
        tuple(image_spec["ellipse"]),
        fill=tuple(image_spec["fill"]),
    )
    return image


def _assert_matches_baseline(
    baseline: torch.Tensor,
    actual: torch.Tensor,
    tolerance: dict[str, float],
) -> dict[str, float]:
    assert actual.shape == baseline.shape
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    assert torch.isfinite(actual).all()

    baseline_fp32 = baseline.float()
    actual_fp32 = actual.float()
    max_abs_error = (baseline_fp32 - actual_fp32).abs().max().item()
    min_cosine_similarity = (
        F.cosine_similarity(
            baseline_fp32,
            actual_fp32,
            dim=-1,
        )
        .min()
        .item()
    )
    assert max_abs_error <= tolerance["max_abs_error"]
    assert min_cosine_similarity >= tolerance["min_cosine_similarity"]
    return {
        "max_abs_error": max_abs_error,
        "min_cosine_similarity": min_cosine_similarity,
    }


def test_v040_cpu_fp32_embeddings_match_saved_baseline() -> None:
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["release"] == "openclip-gemma==0.4.0"
    assert manifest["embedding_dimensions"] == [768, 512, 256, 128]
    assert (
        hashlib.sha256(BASELINE_EMBEDDINGS.read_bytes()).hexdigest()
        == manifest["fixture_sha256"]
    )

    with np.load(BASELINE_EMBEDDINGS, allow_pickle=False) as archive:
        image_baseline = torch.from_numpy(
            np.array(archive[manifest["npz_keys"]["image"]], copy=True)
        )
        text_baseline = torch.from_numpy(
            np.array(archive[manifest["npz_keys"]["text"]], copy=True)
        )

    model, _, preprocess = create_model_and_transforms(
        model_name=manifest["model_name"],
        pretrained=manifest["pretrained"],
        precision=manifest["precision"],
        device=manifest["device"],
    )
    image = preprocess(_create_image(manifest["image"])).unsqueeze(0)
    texts = manifest["text"]["items"]
    tokens = get_tokenizer(
        model_name=manifest["model_name"],
        prompt_name=manifest["text"]["prompt_name"],
    )(texts)

    with torch.inference_mode():
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokens)

    tolerance = manifest["tolerance"]
    comparisons = {
        "image_768": _assert_matches_baseline(
            image_baseline,
            image_features,
            tolerance,
        ),
        "text_768": _assert_matches_baseline(
            text_baseline,
            text_features,
            tolerance,
        ),
    }

    for embedding_dim in manifest["embedding_dimensions"]:
        expected_image = (
            image_baseline
            if embedding_dim == 768
            else F.normalize(image_baseline[:, :embedding_dim], dim=-1)
        )
        expected_text = (
            text_baseline
            if embedding_dim == 768
            else F.normalize(text_baseline[:, :embedding_dim], dim=-1)
        )
        with torch.inference_mode():
            actual_image = model.encode_image(image, truncate_dim=embedding_dim)
            actual_text = model.encode_text(tokens, truncate_dim=embedding_dim)

        assert actual_image.shape == (1, embedding_dim)
        assert actual_text.shape == (len(texts), embedding_dim)
        comparisons[f"image_{embedding_dim}"] = _assert_matches_baseline(
            expected_image,
            actual_image,
            tolerance,
        )
        comparisons[f"text_{embedding_dim}"] = _assert_matches_baseline(
            expected_text,
            actual_text,
            tolerance,
        )
        torch.testing.assert_close(
            actual_image.norm(dim=-1),
            torch.ones(1),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            actual_text.norm(dim=-1),
            torch.ones(len(texts)),
            atol=1e-5,
            rtol=1e-5,
        )

    print(
        json.dumps(
            {
                "baseline": manifest["release"],
                "model_revision": manifest["model_revision"],
                "comparisons": comparisons,
            },
            sort_keys=True,
        )
    )
