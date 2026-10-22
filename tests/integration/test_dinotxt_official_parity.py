from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from openvlclip import create_model_and_transforms, get_tokenizer
from openvlclip.backends.dinov2_dinotxt import (
    DINO_MODEL_NAME,
    DINO_SOURCE_REVISION,
)
from openvlclip.image_search import build_or_update_image_index, search_images

pytestmark = pytest.mark.integration


def test_dinotxt_wrapper_matches_pinned_official_model(tmp_path: Path) -> None:
    model, _, preprocess = create_model_and_transforms(
        DINO_MODEL_NAME,
        pretrained="official",
        precision="fp32",
        device="cpu",
    )
    tokenizer = get_tokenizer(DINO_MODEL_NAME)
    bpe_path = Path(torch.hub.get_dir()) / "checkpoints" / "bpe_simple_vocab_16e6.txt.gz"
    checkout = (
        Path(torch.hub.get_dir())
        / f"facebookresearch_dinov2_{DINO_SOURCE_REVISION}"
    )
    assert checkout.is_dir()
    sys.path.insert(0, str(checkout))
    try:
        from dinov2.data.transforms import make_classification_eval_transform
        from dinov2.hub.text.tokenizer import Tokenizer as OfficialTokenizer

        official_preprocess = make_classification_eval_transform()
        official_tokenizer = OfficialTokenizer(vocab_path=str(bpe_path))
        images = [
            Image.new("RGB", (320, 256), color=(20, 90, 180)),
            Image.new("RGB", (256, 320), color=(180, 80, 20)),
        ]
        expected_pixels = torch.stack([official_preprocess(image) for image in images])
        actual_pixels = torch.stack([preprocess(image) for image in images])
        texts = ["a red car", "a blue car", ""]
        expected_tokens = official_tokenizer.tokenize(texts, context_length=77)
        actual_tokens = tokenizer(texts)
    finally:
        sys.path.remove(str(checkout))

    assert torch.equal(actual_pixels, expected_pixels)
    assert torch.equal(actual_tokens, expected_tokens)
    assert actual_tokens.shape == (3, 77)
    assert actual_tokens[0, :5].tolist() == [49406, 320, 736, 1615, 49407]
    assert actual_tokens[1, :5].tolist() == [49406, 320, 1746, 1615, 49407]

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    comparisons: dict[str, dict[str, float | str]] = {}
    for device in devices:
        dtypes = [torch.float32, torch.bfloat16]
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            dtypes.remove(torch.bfloat16)
        for dtype in dtypes:
            model.to(device=device, dtype=dtype)
            image_batch = actual_pixels.to(device=device)
            token_batch = actual_tokens.to(device=device)
            model_images = image_batch.to(dtype=dtype)
            with torch.inference_mode():
                official_image = model.backend.encode_image(
                    model_images,
                    normalize=False,
                )
                official_text = model.backend.encode_text(
                    token_batch,
                    normalize=False,
                )
                actual_image = model.encode_image(image_batch, normalize=False)
                actual_text = model.encode_text(token_batch, normalize=False)
                official_image_normalized = model.backend.encode_image(
                    model_images,
                    normalize=True,
                )
                official_text_normalized = model.backend.encode_text(
                    token_batch,
                    normalize=True,
                )
                actual_image_normalized = model.encode_image(
                    image_batch,
                    normalize=True,
                )
                actual_text_normalized = model.encode_text(
                    token_batch,
                    normalize=True,
                )

            torch.testing.assert_close(actual_image, official_image, rtol=0, atol=0)
            torch.testing.assert_close(actual_text, official_text, rtol=0, atol=0)
            torch.testing.assert_close(
                actual_image_normalized,
                official_image_normalized,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual_text_normalized,
                official_text_normalized,
                rtol=0,
                atol=0,
            )
            with torch.inference_mode():
                official_forward = model.backend(model_images, token_batch)
                actual_forward = model(image_batch, token_batch)
                official_logits = model.backend.get_logits(
                    model_images,
                    token_batch,
                )
                actual_logits = model.get_logits(image_batch, token_batch)
            for actual, official in zip(
                actual_forward,
                official_forward,
                strict=True,
            ):
                torch.testing.assert_close(actual, official, rtol=0, atol=0)
            for actual, official in zip(
                actual_logits,
                official_logits,
                strict=True,
            ):
                torch.testing.assert_close(actual, official, rtol=0, atol=0)

            model.logit_scale_mode = "cosine"
            with torch.inference_mode():
                cosine_forward = model(image_batch, token_batch)
                cosine_logits = model.get_logits(image_batch, token_batch)
            if dtype == torch.bfloat16 and device == "cpu":
                expected_cosine_logits = (
                    actual_image_normalized.float()
                    @ actual_text_normalized.float().T
                ).to(dtype=dtype)
            else:
                expected_cosine_logits = (
                    actual_image_normalized @ actual_text_normalized.T
                )
            torch.testing.assert_close(
                cosine_logits[0],
                expected_cosine_logits,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                cosine_logits[1],
                expected_cosine_logits.T,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                cosine_forward[0],
                actual_image_normalized,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                cosine_forward[1],
                actual_text_normalized,
                rtol=0,
                atol=0,
            )
            assert cosine_forward[2].item() == 1.0
            model.logit_scale_mode = "strict"

            for features in (
                actual_image,
                actual_text,
                actual_image_normalized,
                actual_text_normalized,
            ):
                assert features.shape[-1] == 2048
                assert features.dtype == dtype
                assert features.device.type == device
                assert torch.isfinite(features).all()
            comparisons[f"{device}-{str(dtype).removeprefix('torch.')}"] = {
                "image_max_abs_error": float(
                    (actual_image - official_image).abs().max().item()
                ),
                "text_max_abs_error": float(
                    (actual_text - official_text).abs().max().item()
                ),
                "forward_image_max_abs_error": float(
                    (actual_forward[0] - official_forward[0]).abs().max().item()
                ),
                "forward_text_max_abs_error": float(
                    (actual_forward[1] - official_forward[1]).abs().max().item()
                ),
                "logit_scale_max_abs_error": float(
                    (actual_forward[2] - official_forward[2]).abs().max().item()
                ),
                "strict_logits_max_abs_error": float(
                    (actual_logits[0] - official_logits[0]).abs().max().item()
                ),
                "cosine_logits_max_abs_error": float(
                    (cosine_logits[0] - expected_cosine_logits).abs().max().item()
                ),
            }

    image_root = tmp_path / "search-images"
    image_root.mkdir()
    image_path = image_root / "sample.png"
    images[0].save(image_path)
    index_dir = tmp_path / "search-index"
    build_or_update_image_index(model, preprocess, image_root, index_dir)
    query = "a blue object"
    search_results = search_images(model, index_dir, query)
    tokens = tokenizer(query).to(device="cpu")
    pixels = preprocess(images[0]).unsqueeze(0)
    with torch.inference_mode():
        image_features = model.encode_image(
            pixels,
            truncate_dim=2048,
        ).float()
        text_features = model.encode_text(
            tokens,
            normalize=True,
            truncate_dim=2048,
        ).float()
    expected_search_score = (
        torch.nn.functional.normalize(image_features, dim=-1)
        @ torch.nn.functional.normalize(text_features, dim=-1).T
    ).item()
    assert search_results[0].path == image_path
    assert search_results[0].score == pytest.approx(expected_search_score, abs=1e-6)
    print(json.dumps(comparisons, sort_keys=True))
