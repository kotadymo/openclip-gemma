"""Numerical image-inference comparisons against Sentence Transformers."""

from __future__ import annotations

import json
import os
from typing import TypedDict

import pytest
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from openvlclip import create_model_and_transforms
from openvlclip.factory import MODEL_ID, MODEL_REVISION

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]

MAX_ABS_ERROR = 1e-4
MIN_COSINE_SIMILARITY = 0.9999


class Comparison(TypedDict):
    max_abs_error: float
    cosine_similarity: list[float]


def _sample_images() -> dict[str, Image.Image]:
    rgb = Image.new("RGB", (256, 192), (30, 80, 180))
    ImageDraw.Draw(rgb).ellipse((72, 40, 184, 152), fill=(220, 35, 45))
    return {
        "rgb": rgb,
        "grayscale": ImageOps.grayscale(rgb),
        "wide": rgb.resize((384, 128), Image.Resampling.BICUBIC),
        "tall": rgb.resize((128, 384), Image.Resampling.BICUBIC),
    }


def _comparison(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> Comparison:
    reference_fp32 = reference.detach().float()
    actual_fp32 = actual.detach().float()
    cosine = F.cosine_similarity(reference_fp32, actual_fp32, dim=-1)
    return {
        "max_abs_error": (reference_fp32 - actual_fp32).abs().max().item(),
        "cosine_similarity": cosine.cpu().tolist(),
    }


@pytest.mark.parametrize(
    ("precision", "dtype"),
    [("fp32", torch.float32), ("bf16", torch.bfloat16)],
)
def test_image_embeddings_match_sentence_transformers(
    precision: str,
    dtype: torch.dtype,
) -> None:
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    model, _, preprocess = create_model_and_transforms(
        precision=precision,
        device="cpu",
    )
    reference_model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
    )
    reference_model.to(device="cpu", dtype=dtype)
    reference_model.eval()

    images = _sample_images()
    comparisons: dict[str, dict[str, object]] = {}
    with torch.inference_mode():
        for name, pil_image in images.items():
            image_tensor = preprocess(pil_image).unsqueeze(0)
            actual = model.encode_image(image_tensor)
            reference = reference_model.encode(
                [{"image": pil_image}],
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            assert actual.shape == (1, 768)
            assert actual.dtype == dtype
            assert actual.device.type == "cpu"
            assert torch.isfinite(actual).all()
            comparisons[name] = _comparison(reference, actual)

            for truncate_dim in (512, 256, 128):
                truncated = model.encode_image(
                    image_tensor,
                    truncate_dim=truncate_dim,
                )
                expected = F.normalize(
                    reference[:, :truncate_dim],
                    p=2,
                    dim=-1,
                )
                assert truncated.shape == (1, truncate_dim)
                assert torch.isfinite(truncated).all()
                torch.testing.assert_close(
                    truncated,
                    expected,
                    atol=2e-3 if dtype == torch.bfloat16 else MAX_ABS_ERROR,
                    rtol=2e-3 if dtype == torch.bfloat16 else MAX_ABS_ERROR,
                )

            if dtype == torch.float32:
                assert comparisons[name]["max_abs_error"] <= MAX_ABS_ERROR
                assert all(
                    value >= MIN_COSINE_SIMILARITY
                    for value in comparisons[name]["cosine_similarity"]
                )

        same_shape_names = ["rgb", "grayscale"]
        image_batch = torch.stack(
            [preprocess(images[name]) for name in same_shape_names]
        )
        actual_batch = model.encode_image(image_batch)
        reference_batch = reference_model.encode(
            [{"image": images[name]} for name in same_shape_names],
            batch_size=len(same_shape_names),
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        comparisons["same_shape_batch"] = _comparison(
            reference_batch,
            actual_batch,
        )
        assert actual_batch.shape == (2, 768)
        assert torch.isfinite(actual_batch).all()
        if dtype == torch.float32:
            assert comparisons["same_shape_batch"]["max_abs_error"] <= MAX_ABS_ERROR
            assert all(
                value >= MIN_COSINE_SIMILARITY
                for value in comparisons["same_shape_batch"]["cosine_similarity"]
            )

        normalized = model.encode_image(image_batch, normalize=True)
        normalized_reference = reference_model.encode(
            [{"image": images[name]} for name in same_shape_names],
            batch_size=len(same_shape_names),
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        comparisons["normalize_true"] = _comparison(
            normalized_reference,
            normalized,
        )
        assert torch.allclose(
            normalized.norm(dim=-1).float(),
            torch.ones(2),
            atol=2e-3 if dtype == torch.bfloat16 else 1e-6,
        )

    print(
        json.dumps(
            {"precision": precision, "comparisons": comparisons},
            sort_keys=True,
        )
    )
    if dtype == torch.bfloat16:
        assert all(
            torch.isfinite(torch.tensor(values["cosine_similarity"])).all()
            for values in comparisons.values()
        )
