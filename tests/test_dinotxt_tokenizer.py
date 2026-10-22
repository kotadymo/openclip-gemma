from __future__ import annotations

import gzip
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest
import torch

from openvlclip.backends.dinotxt_tokenizer import DinoTxtTokenizer, SimpleTokenizer


class FakeBpeTokenizer:
    encoder: ClassVar[dict[str, int]] = {
        "<|startoftext|>": 49406,
        "<|endoftext|>": 49407,
    }

    def encode(self, text: str) -> list[int]:
        if text == "long":
            return [101, 102, 103, 104, 105]
        return [10] if text else []


def test_dinotxt_tokenizer_matches_fixed_width_sot_eot_padding_and_truncation() -> (
    None
):
    tokenizer = DinoTxtTokenizer(FakeBpeTokenizer(), context_length=5)  # type: ignore[arg-type]

    tokens = tokenizer(["word", "", "long"])

    assert tokens.dtype == torch.long
    assert tokens.tolist() == [
        [49406, 10, 49407, 0, 0],
        [49406, 49407, 0, 0, 0],
        [49406, 101, 102, 103, 49407],
    ]
    assert tokenizer.context_length == 5
    assert tokenizer.prompt_name is None


@pytest.mark.parametrize(
    ("texts", "error", "message"),
    [
        ([], ValueError, "at least one"),
        ([1], TypeError, r"texts\[0\]"),
        (None, TypeError, "string or a sequence"),
    ],
)
def test_dinotxt_tokenizer_rejects_invalid_inputs(
    texts: str | Sequence[str] | None,
    error: type[Exception],
    message: str,
) -> None:
    tokenizer = DinoTxtTokenizer(FakeBpeTokenizer(), context_length=77)  # type: ignore[arg-type]

    with pytest.raises(error, match=message):
        tokenizer(texts)  # type: ignore[arg-type]


def test_dinotxt_tokenizer_rejects_unexpected_special_token_ids() -> None:
    class InvalidBpeTokenizer:
        encoder: ClassVar[dict[str, int]] = {
            "<|startoftext|>": 1,
            "<|endoftext|>": 2,
        }

    with pytest.raises(RuntimeError, match="unexpected special-token IDs"):
        DinoTxtTokenizer(InvalidBpeTokenizer(), context_length=77)  # type: ignore[arg-type]


def test_simple_tokenizer_builds_finite_byte_vocabulary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bpe_path = tmp_path / "bpe_simple_vocab.txt.gz"
    with gzip.open(bpe_path, "wt", encoding="utf-8") as vocabulary_file:
        vocabulary_file.write("version")

    ftfy = ModuleType("ftfy")
    ftfy.__dict__["fix_text"] = lambda text: text
    regex = ModuleType("regex")
    regex.__dict__["compile"] = lambda pattern, flags: pattern
    regex.__dict__["findall"] = lambda pattern, text: []
    regex.__dict__["IGNORECASE"] = 2
    monkeypatch.setitem(sys.modules, "ftfy", ftfy)
    monkeypatch.setitem(sys.modules, "regex", regex)

    tokenizer = SimpleTokenizer(bpe_path)

    assert len(tokenizer.encoder) == 514
