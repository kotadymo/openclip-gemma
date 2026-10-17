"""End-to-end cosine-scoring validation against Sentence Transformers."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from openclip_gemma import create_model_and_transforms, get_tokenizer
from openclip_gemma.factory import MODEL_ID, MODEL_REVISION

_EXAMPLE = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "examples" / "zero_shot_classification.py")
)
relative_softmax_scores = _EXAMPLE["relative_softmax_scores"]
zero_shot_scores = _EXAMPLE["zero_shot_scores"]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]

MAX_ABS_ERROR = 1e-4
MIN_COSINE_SIMILARITY = 0.9999
CANDIDATES = [
    "a red circle on a blue background",
    "a green triangle on a yellow background",
    "青い背景にある赤い円",
    "黄色い背景にある緑の三角形",
]


def _images() -> list[Image.Image]:
    first = Image.new("RGB", (256, 192), (30, 80, 180))
    ImageDraw.Draw(first).ellipse((72, 40, 184, 152), fill=(220, 35, 45))
    second = Image.new("RGB", (256, 192), (230, 200, 30))
    ImageDraw.Draw(second).polygon(
        [(128, 32), (220, 160), (36, 160)],
        fill=(25, 170, 60),
    )
    return [first, second]


def _track_loader(
    monkeypatch: pytest.MonkeyPatch,
    loader_class: type,
    name: str,
    counts: dict[str, int],
) -> None:
    original = loader_class.from_pretrained

    def tracked(cls: type, *args: object, **kwargs: object) -> object:
        counts[name] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(loader_class, "from_pretrained", classmethod(tracked))


def _assert_embedding_match(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    check_fp32_threshold: bool,
) -> dict[str, object]:
    assert reference.shape == actual.shape
    assert torch.isfinite(reference).all()
    assert torch.isfinite(actual).all()
    reference_fp32 = reference.float()
    actual_fp32 = actual.float()
    max_abs_error = (reference_fp32 - actual_fp32).abs().max().item()
    cosine = F.cosine_similarity(reference_fp32, actual_fp32, dim=-1)
    comparison: dict[str, object] = {
        "max_abs_error": max_abs_error,
        "cosine_similarity": cosine.cpu().tolist(),
    }
    if check_fp32_threshold:
        assert max_abs_error <= MAX_ABS_ERROR
        assert (cosine >= MIN_COSINE_SIMILARITY).all()
    return comparison


@pytest.mark.parametrize(
    ("precision", "dtype"),
    [("fp32", torch.float32), ("bf16", torch.bfloat16)],
)
def test_zero_shot_scores_match_reference_and_do_not_reload(
    precision: str,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer
    from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

    reference_model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
    )
    reference_model.to(device="cpu", dtype=dtype)
    reference_model.eval()

    loader_counts = {"model": 0, "processor": 0, "config": 0, "tokenizer": 0}
    _track_loader(monkeypatch, AutoModel, "model", loader_counts)
    _track_loader(monkeypatch, AutoProcessor, "processor", loader_counts)
    _track_loader(monkeypatch, AutoConfig, "config", loader_counts)
    _track_loader(monkeypatch, AutoTokenizer, "tokenizer", loader_counts)

    model, _, preprocess = create_model_and_transforms(
        precision=precision,
        device="cpu",
    )
    tokenizer = get_tokenizer(prompt_name="Classification")
    assert tokenizer.context_length == 8192
    assert loader_counts["model"] == 1
    assert loader_counts["processor"] == 1
    loads_after_initialization = loader_counts.copy()

    images = _images()
    with torch.inference_mode():
        image_features = model.encode_image(
            torch.stack([preprocess(image) for image in images]),
            normalize=True,
        )
        text_features = model.encode_text(
            tokenizer(CANDIDATES),
            normalize=True,
        )
        reference_images = reference_model.encode(
            [{"image": image} for image in images],
            batch_size=2,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        reference_texts = reference_model.encode(
            CANDIDATES,
            prompt_name="Classification",
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    assert image_features.shape == (2, 768)
    assert text_features.shape == (4, 768)
    assert image_features.dtype == dtype
    assert text_features.dtype == dtype
    assert image_features.device.type == "cpu"
    assert text_features.device.type == "cpu"
    image_comparison = _assert_embedding_match(
        reference_images,
        image_features,
        check_fp32_threshold=dtype == torch.float32,
    )
    text_comparison = _assert_embedding_match(
        reference_texts,
        text_features,
        check_fp32_threshold=dtype == torch.float32,
    )

    scores = zero_shot_scores(model, preprocess, tokenizer, images, CANDIDATES)
    one_image_scores = zero_shot_scores(
        model,
        preprocess,
        tokenizer,
        images[:1],
        CANDIDATES,
    )
    reversed_image_scores = zero_shot_scores(
        model,
        preprocess,
        tokenizer,
        images[::-1],
        CANDIDATES,
    )
    reversed_candidate_scores = zero_shot_scores(
        model,
        preprocess,
        tokenizer,
        images,
        CANDIDATES[::-1],
    )
    expected_scores = reference_images @ reference_texts.T

    assert scores.shape == (2, 4)
    assert one_image_scores.shape == (1, 4)
    torch.testing.assert_close(scores, image_features @ text_features.T)
    torch.testing.assert_close(scores[:1], one_image_scores)
    torch.testing.assert_close(reversed_image_scores, scores.flip(0))
    torch.testing.assert_close(reversed_candidate_scores, scores.flip(1))
    assert torch.isfinite(scores).all()
    score_comparison = _assert_embedding_match(
        expected_scores,
        scores,
        check_fp32_threshold=dtype == torch.float32,
    )
    if dtype == torch.float32:
        assert score_comparison["max_abs_error"] <= MAX_ABS_ERROR
    relative_scores = relative_softmax_scores(scores)
    torch.testing.assert_close(
        relative_scores.sum(dim=-1),
        torch.ones(2, dtype=relative_scores.dtype),
    )

    assert loader_counts == loads_after_initialization
    print(
        json.dumps(
            {
                "precision": precision,
                "image_embeddings": image_comparison,
                "text_embeddings": text_comparison,
                "cosine_score_matrix": score_comparison,
                "score_shape": list(scores.shape),
                "model_reload_count": loader_counts["model"],
                "processor_reload_count": loader_counts["processor"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable.")
def test_zero_shot_scores_match_reference_on_cuda() -> None:
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    reference_model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cuda",
    )
    model, _, preprocess = create_model_and_transforms(device="cuda")
    tokenizer = get_tokenizer(prompt_name="Classification")
    images = _images()
    scores = zero_shot_scores(model, preprocess, tokenizer, images, CANDIDATES)
    reference_images = reference_model.encode(
        [{"image": image} for image in images],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    reference_texts = reference_model.encode(
        CANDIDATES,
        prompt_name="Classification",
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    expected_scores = reference_images @ reference_texts.T

    assert scores.device.type == "cuda"
    assert scores.shape == (2, 4)
    assert torch.isfinite(scores).all()
    torch.testing.assert_close(scores, expected_scores, atol=MAX_ABS_ERROR, rtol=0)
