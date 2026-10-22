"""Pinned CLIP BPE tokenizer used by Meta's dino.txt model."""

from __future__ import annotations

import gzip
import html
import importlib
import itertools
import re
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = byte_values[:]
    next_codepoint = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            unicode_values.append(256 + next_codepoint)
            next_codepoint += 1
    return dict(zip(byte_values, map(chr, unicode_values), strict=True))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(itertools.pairwise(word))


class SimpleTokenizer:
    """CLIP byte-level BPE tokenizer with its vocabulary supplied explicitly."""

    def __init__(self, bpe_path: str | Path) -> None:
        try:
            ftfy = importlib.import_module("ftfy")
            regex = importlib.import_module("regex")
        except ModuleNotFoundError as error:
            raise ImportError(
                'DINOv2 + dino.txt tokenization requires the "dinotxt" extra; '
                'install it with `pip install "openvlclip[dinotxt]"`.'
            ) from error

        fix_text = getattr(ftfy, "fix_text", None)
        regex_namespace = vars(regex)
        compile_pattern = regex_namespace.get("compile")
        findall = regex_namespace.get("findall")
        ignore_case = regex_namespace.get("IGNORECASE")
        if (
            not callable(fix_text)
            or not callable(compile_pattern)
            or not callable(findall)
            or ignore_case is None
        ):
            raise RuntimeError(
                "The installed dino.txt tokenizer dependencies are invalid."
            )
        self._fix_text: Callable[[str], str] = fix_text
        self._regex_findall: Callable[[Any, str], list[str]] = findall
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        with gzip.open(bpe_path, "rt", encoding="utf-8") as vocabulary_file:
            raw_merges = vocabulary_file.read().split("\n")
        merges = [
            tuple(merge.split())
            for merge in raw_merges[1 : 49152 - 256 - 2 + 1]
        ]
        vocabulary = list(self.byte_encoder.values())
        vocabulary.extend(value + "</w>" for value in tuple(vocabulary))
        vocabulary.extend("".join(merge) for merge in merges)
        vocabulary.extend(("<|startoftext|>", "<|endoftext|>"))
        self.encoder = {token: index for index, token in enumerate(vocabulary)}
        self.decoder = {index: token for token, index in self.encoder.items()}
        self.bpe_ranks = {merge: index for index, merge in enumerate(merges)}
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pattern = compile_pattern(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            ignore_case,
        )

    def _clean_text(self, text: str) -> str:
        text = self._fix_text(text)
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, float("inf")),
            )
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_index])
                index = next_index
                if (
                    word[index] == first
                    and index < len(word) - 1
                    and word[index + 1] == second
                ):
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        for token in self._regex_findall(self.pattern, self._clean_text(text)):
            encoded = "".join(self.byte_encoder[value] for value in token.encode("utf-8"))
            tokens.extend(
                self.encoder[part] for part in self._bpe(encoded).split(" ")
            )
        return tokens


class DinoTxtTokenizer:
    """Return fixed-width token IDs following the official dino.txt tokenizer."""

    def __init__(self, tokenizer: SimpleTokenizer, *, context_length: int) -> None:
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prompt_name = None
        self.start_token_id = tokenizer.encoder["<|startoftext|>"]
        self.end_token_id = tokenizer.encoder["<|endoftext|>"]
        if self.start_token_id != 49406 or self.end_token_id != 49407:
            raise RuntimeError(
                "The verified dino.txt BPE vocabulary has unexpected special-token IDs."
            )

    def __call__(self, texts: str | Sequence[str]) -> torch.Tensor:
        if isinstance(texts, str):
            batch = [texts]
        elif isinstance(texts, Sequence):
            batch = list(texts)
        else:
            raise TypeError("texts must be a string or a sequence of strings.")
        if not batch:
            raise ValueError("texts must contain at least one string.")
        for index, text in enumerate(batch):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] must be a string.")

        output = torch.zeros((len(batch), self.context_length), dtype=torch.long)
        for row, text in enumerate(batch):
            token_ids = [
                self.start_token_id,
                *self.tokenizer.encode(text),
                self.end_token_id,
            ]
            if len(token_ids) > self.context_length:
                token_ids = token_ids[: self.context_length]
                token_ids[-1] = self.end_token_id
            output[row, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        return output
