"""Text embedding comparisons against the pinned Sentence Transformers model."""

from __future__ import annotations

import json
import os
from typing import TypedDict

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from openclip_gemma import create_model, get_tokenizer
from openclip_gemma.factory import MODEL_ID, MODEL_REVISION

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]

MAX_ABS_ERROR = 1e-4
MIN_COSINE_SIMILARITY = 0.9999
TEXTS = [
    "A small red circle on a blue background.",
    "猫は青い背景にある赤い円を見ています。",
    "",
]


class Comparison(TypedDict):
    max_abs_error: float
    cosine_similarity: list[float]


def _comparison(reference: torch.Tensor, actual: torch.Tensor) -> Comparison:
    reference_fp32 = reference.detach().float()
    actual_fp32 = actual.detach().float()
    return {
        "max_abs_error": (reference_fp32 - actual_fp32).abs().max().item(),
        "cosine_similarity": F.cosine_similarity(
            reference_fp32,
            actual_fp32,
            dim=-1,
        )
        .cpu()
        .tolist(),
    }


@pytest.mark.parametrize(
    ("precision", "dtype"),
    [("fp32", torch.float32), ("bf16", torch.bfloat16)],
)
def test_text_embeddings_match_sentence_transformers(
    precision: str,
    dtype: torch.dtype,
) -> None:
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    model = create_model(precision=precision, device="cpu")
    reference_model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
    )
    reference_model.to(device="cpu", dtype=dtype)
    reference_model.eval()
    official_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    max_context_length = config.text_config.max_position_embeddings

    comparisons: dict[str, dict[str, object]] = {}
    for prompt_name in (None, "SearchQuery", "Classification"):
        tokenizer = get_tokenizer(prompt_name=prompt_name)
        assert tokenizer.max_context_length == 8192
        assert tokenizer.model_max_position_embeddings == max_context_length
        assert tokenizer.prompts == reference_model.prompts
        tokens = tokenizer(TEXTS)
        prefix = (
            ""
            if prompt_name is None
            else reference_model.prompts[prompt_name]
        )
        official_features = official_tokenizer(
            [f"{prefix}{text}" for text in TEXTS],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=tokenizer.context_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        torch.testing.assert_close(tokens, official_features["input_ids"])

        with torch.inference_mode():
            actual = model.encode_text(tokens)
            prompt_args = (
                {"prompt": ""}
                if prompt_name is None
                else {"prompt_name": prompt_name}
            )
            reference = reference_model.encode(
                TEXTS,
                **prompt_args,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            normalized = model.encode_text(tokens, normalize=True)
            normalized_reference = reference_model.encode(
                TEXTS,
                **prompt_args,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        key = "none" if prompt_name is None else prompt_name
        comparisons[key] = _comparison(reference, actual)
        comparisons[f"{key}_normalize_true"] = _comparison(
            normalized_reference,
            normalized,
        )
        for output in (actual, normalized):
            assert output.shape == (len(TEXTS), 768)
            assert output.dtype == dtype
            assert output.device.type == "cpu"
            assert torch.isfinite(output).all()

        if dtype == torch.float32:
            for result in (comparisons[key], comparisons[f"{key}_normalize_true"]):
                assert result["max_abs_error"] <= MAX_ABS_ERROR
                assert all(
                    value >= MIN_COSINE_SIMILARITY
                    for value in result["cosine_similarity"]
                )

    tokenizer = get_tokenizer(context_length=16)
    long_tokens = tokenizer("word " * 40)
    assert long_tokens.shape == (1, 16)
    assert long_tokens[0, 0].item() == official_tokenizer.bos_token_id
    assert long_tokens[0, -1].item() == official_tokenizer.eos_token_id
    assert model.encode_text(long_tokens).shape == (1, 768)

    special_tokens = get_tokenizer()("<eos>")
    with torch.inference_mode():
        special_actual = model.encode_text(special_tokens)
        special_reference = reference_model.encode(
            ["<eos>"],
            prompt="",
            convert_to_tensor=True,
            show_progress_bar=False,
        )
    comparisons["special_eos_token"] = _comparison(
        special_reference,
        special_actual,
    )
    if dtype == torch.float32:
        assert comparisons["special_eos_token"]["max_abs_error"] <= MAX_ABS_ERROR
        assert all(
            value >= MIN_COSINE_SIMILARITY
            for value in comparisons["special_eos_token"]["cosine_similarity"]
        )

    with pytest.raises(ValueError, match="PAD token as content"):
        get_tokenizer()("<pad>")

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable.")
def test_text_embeddings_match_sentence_transformers_on_cuda() -> None:
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    model = create_model(device="cuda")
    reference_model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cuda",
    )
    tokenizer = get_tokenizer(prompt_name="SearchQuery")
    tokens = tokenizer(TEXTS).to("cuda")

    with torch.inference_mode():
        actual = model.encode_text(tokens)
        reference = reference_model.encode(
            TEXTS,
            prompt_name="SearchQuery",
            convert_to_tensor=True,
            show_progress_bar=False,
        )

    result = _comparison(reference, actual)
    assert actual.device.type == "cuda"
    assert actual.shape == (len(TEXTS), 768)
    assert torch.isfinite(actual).all()
    assert result["max_abs_error"] <= MAX_ABS_ERROR
    assert all(
        value >= MIN_COSINE_SIMILARITY
        for value in result["cosine_similarity"]
    )
