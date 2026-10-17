from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import openclip_gemma.tokenizer as tokenizer_module
from openclip_gemma.factory import MODEL_ID, MODEL_REVISION
from openclip_gemma.tokenizer import OFFICIAL_PROMPTS, get_tokenizer


class FakeHFAutoTokenizer:
    pad_token_id = 0
    bos_token_id = 2
    eos_token_id = 1
    padding_side = "right"
    truncation_side = "right"
    vocab_size = 128
    model_max_length = 10**30

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append((texts, kwargs))
        sequences = []
        for text in texts:
            if "<pad>" in text:
                content = [self.pad_token_id]
            elif text == "<eos>":
                content = [self.eos_token_id]
            else:
                content = [
                    3 + (sum(map(ord, piece)) % 100)
                    for piece in text.split()
                ]
            sequence = [self.bos_token_id, *content, self.eos_token_id]
            if kwargs["truncation"] and len(sequence) > kwargs["max_length"]:
                sequence = sequence[: kwargs["max_length"]]
                sequence[-1] = self.eos_token_id
            sequences.append(sequence)

        padded_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), padded_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.tensor(sequence)
            attention_mask[row, : len(sequence)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture
def fake_hf_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeHFAutoTokenizer, list[dict[str, Any]]]:
    backend = FakeHFAutoTokenizer()
    calls: list[dict[str, Any]] = []

    def load_config(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append({"loader": "config", "args": args, **kwargs})
        return SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=32)
        )

    def load_tokenizer(*args: Any, **kwargs: Any) -> FakeHFAutoTokenizer:
        calls.append({"loader": "tokenizer", "args": args, **kwargs})
        return backend

    monkeypatch.setattr(
        tokenizer_module,
        "AutoConfig",
        SimpleNamespace(from_pretrained=load_config),
    )
    monkeypatch.setattr(
        tokenizer_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=load_tokenizer),
    )
    return backend, calls


def test_get_tokenizer_uses_pinned_revision_and_dynamic_padding(
    fake_hf_loaders: tuple[FakeHFAutoTokenizer, list[dict[str, Any]]],
) -> None:
    backend, calls = fake_hf_loaders

    tokenizer = get_tokenizer("EmbeddingGemma_2", cache_dir="cache")
    tokens = tokenizer(["a cat", "a small blue dog"])

    assert tokens.dtype == torch.long
    assert tokens.shape[0] == 2
    assert tokens.shape[1] == 6
    assert tokens[0, -1].item() == backend.pad_token_id
    assert tokenizer.max_context_length == 32
    assert tokenizer.context_length == 32
    assert [call["loader"] for call in calls] == ["config", "tokenizer"]
    for call in calls:
        assert call["args"] == (MODEL_ID,)
        assert call["revision"] == MODEL_REVISION
        assert call["cache_dir"] == "cache"
    assert backend.calls[0][1] == {
        "add_special_tokens": True,
        "padding": True,
        "truncation": True,
        "max_length": 32,
        "return_attention_mask": True,
        "return_tensors": "pt",
    }


def test_tokenizer_accepts_single_and_empty_strings(
    fake_hf_loaders: tuple[FakeHFAutoTokenizer, list[dict[str, Any]]],
) -> None:
    backend, _ = fake_hf_loaders
    tokenizer = get_tokenizer()

    single = tokenizer("a cat")
    empty = tokenizer("")

    assert single.shape[0] == 1
    assert empty.tolist() == [[backend.bos_token_id, backend.eos_token_id]]


def test_tokenizer_preserves_user_supplied_special_tokens(
    fake_hf_loaders: tuple[FakeHFAutoTokenizer, list[dict[str, Any]]],
) -> None:
    backend, _ = fake_hf_loaders

    tokens = get_tokenizer()("<eos>")

    assert tokens.tolist() == [
        [backend.bos_token_id, backend.eos_token_id, backend.eos_token_id]
    ]


def test_tokenizer_applies_official_prompt_once(
    fake_hf_loaders: tuple[FakeHFAutoTokenizer, list[dict[str, Any]]],
) -> None:
    backend, _ = fake_hf_loaders
    tokenizer = get_tokenizer(prompt_name="SearchQuery")

    tokenizer(["a cat"])

    assert tokenizer.prompts == OFFICIAL_PROMPTS
    assert backend.calls[0][0] == [
        f"{OFFICIAL_PROMPTS['SearchQuery']}a cat"
    ]
    with pytest.raises(ValueError, match="already starts with an official task prompt"):
        tokenizer(f"{OFFICIAL_PROMPTS['SearchQuery']}a cat")


def test_tokenizer_rejects_pad_token_as_text_content(
    fake_hf_loaders: tuple[FakeHFAutoTokenizer, list[dict[str, Any]]],
) -> None:
    tokenizer = get_tokenizer()

    with pytest.raises(ValueError, match="PAD token as content"):
        tokenizer("<pad>")


def test_tokenizer_truncates_and_preserves_eos(
    fake_hf_loaders: tuple[FakeHFAutoTokenizer, list[dict[str, Any]]],
) -> None:
    backend, _ = fake_hf_loaders
    tokenizer = get_tokenizer(context_length=5)

    tokens = tokenizer("one two three four five six")

    assert tokens.shape == (1, 5)
    assert tokens[0, 0].item() == backend.bos_token_id
    assert tokens[0, -1].item() == backend.eos_token_id


@pytest.mark.parametrize(
    ("model_name", "prompt_name", "context_length", "exception", "message"),
    [
        ("unknown", None, None, ValueError, "Unsupported model_name"),
        ("embeddinggemma-2", "Unknown", None, ValueError, "Unsupported prompt_name"),
        ("embeddinggemma-2", None, 1, ValueError, "at least 2"),
        ("embeddinggemma-2", None, 33, ValueError, "exceeds the model positional limit"),
        ("embeddinggemma-2", None, True, ValueError, "integer of at least 2"),
    ],
)
def test_get_tokenizer_rejects_unsupported_options(
    fake_hf_loaders: tuple[FakeHFAutoTokenizer, list[dict[str, Any]]],
    model_name: str,
    prompt_name: str | None,
    context_length: int | None,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        get_tokenizer(
            model_name,
            prompt_name=prompt_name,
            context_length=context_length,
        )


@pytest.mark.parametrize(
    ("texts", "exception", "message"),
    [
        ([], ValueError, "at least one"),
        ([1], TypeError, "texts\\[0\\]"),
        (None, TypeError, "string or a sequence"),
    ],
)
def test_tokenizer_rejects_invalid_input(
    fake_hf_loaders: tuple[FakeHFAutoTokenizer, list[dict[str, Any]]],
    texts: Any,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        get_tokenizer()(texts)
