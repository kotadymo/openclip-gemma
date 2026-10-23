"""Opt-in parity checks against the pinned timm OpenCLIP PE-Core snapshots."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from openvlclip import create_model_and_transforms, get_tokenizer
from openvlclip.backends.pe_core import PE_CORE_S_MODEL_NAME, PE_CORE_T_MODEL_NAME
from openvlclip.image_search import (
    ImageIndexCompatibilityError,
    build_or_update_image_index,
    search_images,
)
from openvlclip.registry import get_default_registry

_CHECKPOINTS = (
    (
        PE_CORE_T_MODEL_NAME,
        "timm/PE-Core-T-16-384",
        "7fe539ed578ac49a1c2b4f946e4b0747704c825a",
        PE_CORE_S_MODEL_NAME,
    ),
    (
        PE_CORE_S_MODEL_NAME,
        "timm/PE-Core-S-16-384",
        "3249a38eb1c432ec19231c6fe1774acb6a4e4efe",
        PE_CORE_T_MODEL_NAME,
    ),
)
_TEXTS = (
    "a red object on a blue background",
    "a small dog sitting by the window",
)
_TOLERANCE = 1e-5

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]


class _DifferentSpaceModel:
    def __init__(self, model: object, metadata: object) -> None:
        self._model = model
        self.embedding_space_metadata = metadata

    def __getattr__(self, name: str) -> object:
        return getattr(self._model, name)


def _sample_images() -> list[Image.Image]:
    return [
        Image.new("RGB", (321, 193), (210, 30, 40)),
        Image.new("RGB", (191, 319), (20, 50, 210)),
    ]


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max().item())


def _max_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        ((actual - expected).abs() / expected.abs().clamp_min(1e-8)).max().item()
    )


@pytest.mark.parametrize(
    ("model_name", "model_id", "revision", "other_model_name"),
    _CHECKPOINTS,
)
def test_pe_core_wrapper_matches_independent_openclip_reference(
    tmp_path: Path,
    model_name: str,
    model_id: str,
    revision: str,
    other_model_name: str,
) -> None:
    import open_clip

    from openvlclip.backends.pe_core import _snapshot

    local = _snapshot(model_id, revision, None)
    reference_model, _, reference_preprocess = open_clip.create_model_and_transforms(
        f"local-dir:{local}",
        device="cpu",
        precision="fp32",
    )
    reference_model = reference_model.eval()
    reference_tokenizer = open_clip.get_tokenizer(f"local-dir:{local}")

    model, _, preprocess = create_model_and_transforms(
        model_name,
        pretrained="official",
        precision="fp32",
        device="cpu",
    )
    images = _sample_images()
    reference_pixels = torch.stack([reference_preprocess(image) for image in images])
    actual_pixels = torch.stack([preprocess(image) for image in images])
    torch.testing.assert_close(actual_pixels, reference_pixels, rtol=0, atol=0)
    assert actual_pixels.shape == (2, 3, 384, 384)

    reference_tokens = reference_tokenizer(list(_TEXTS))
    actual_tokens = get_tokenizer(model_name)(list(_TEXTS))
    torch.testing.assert_close(actual_tokens, reference_tokens, rtol=0, atol=0)
    assert actual_tokens.shape == (2, 32)

    with torch.inference_mode():
        reference_images = reference_model.encode_image(
            reference_pixels, normalize=False
        )
        reference_text = reference_model.encode_text(reference_tokens, normalize=False)
        actual_images = model.encode_image(actual_pixels, normalize=False)
        actual_text = model.encode_text(actual_tokens, normalize=False)
        reference_images_normalized = F.normalize(reference_images, p=2, dim=-1)
        reference_text_normalized = F.normalize(reference_text, p=2, dim=-1)
        actual_images_normalized = model.encode_image(actual_pixels, normalize=True)
        actual_text_normalized = model.encode_text(actual_tokens, normalize=True)
        reference_scale = reference_model.logit_scale.exp()
        reference_logits = (
            reference_images_normalized @ reference_text_normalized.T
        ) * reference_scale
        actual_logits, actual_logits_transposed = model.get_logits(
            actual_pixels, actual_tokens
        )

    for actual, expected in (
        (actual_images, reference_images),
        (actual_text, reference_text),
        (actual_images_normalized, reference_images_normalized),
        (actual_text_normalized, reference_text_normalized),
        (actual_logits, reference_logits),
        (actual_logits_transposed, reference_logits.T),
        (model.backend.logit_scale, reference_model.logit_scale),
    ):
        torch.testing.assert_close(actual, expected, rtol=_TOLERANCE, atol=_TOLERANCE)
        assert torch.isfinite(actual).all()
    assert actual_images.shape == (2, 512)
    assert actual_text.shape == (2, 512)

    images_dir = tmp_path / model_name / "images"
    images_dir.mkdir(parents=True)
    for index, image in enumerate(images):
        image.save(images_dir / f"{index}.png")
    index_dir = tmp_path / model_name / "index"
    update = build_or_update_image_index(
        model, preprocess, images_dir, index_dir, batch_size=2
    )
    results = search_images(model, index_dir, _TEXTS[0], top_k=2)
    assert update.added == len(images)
    assert len(results) == len(images)

    other_descriptor = get_default_registry().resolve(other_model_name)
    different_space_model = _DifferentSpaceModel(
        model, other_descriptor.embedding_space_metadata
    )
    with pytest.raises(ImageIndexCompatibilityError, match="embedding space"):
        search_images(different_space_model, index_dir, _TEXTS[0], top_k=1)

    error_metrics = {
        "image_max_abs": _max_abs_error(actual_images, reference_images),
        "image_max_relative": _max_relative_error(actual_images, reference_images),
        "normalized_image_max_abs": _max_abs_error(
            actual_images_normalized, reference_images_normalized
        ),
        "normalized_text_max_abs": _max_abs_error(
            actual_text_normalized, reference_text_normalized
        ),
        "text_max_abs": _max_abs_error(actual_text, reference_text),
        "text_max_relative": _max_relative_error(actual_text, reference_text),
        "logits_max_abs": _max_abs_error(actual_logits, reference_logits),
        "logits_max_relative": _max_relative_error(actual_logits, reference_logits),
    }
    print(
        json.dumps(
            {
                "model": model_name,
                "model_id": model_id,
                "revision": revision,
                "device": "cpu",
                "dtype": "float32",
                "tolerance": _TOLERANCE,
                "open_clip_torch": importlib.metadata.version("open-clip-torch"),
                "torch": torch.__version__,
                "max_errors": error_metrics,
            },
            sort_keys=True,
        )
    )
