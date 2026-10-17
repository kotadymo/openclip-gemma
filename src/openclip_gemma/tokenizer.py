"""Official EmbeddingGemma 2 tokenization for OpenCLIP-style callers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from openclip_gemma.constants import (
    MODEL_CARD_MAX_CONTEXT_LENGTH,
    MODEL_ID,
    MODEL_NAME,
    MODEL_REVISION,
)

OFFICIAL_PROMPTS = {
    "BitextMining": "task: search result | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Document": "title: none | text: ",
    "FactChecking": "task: fact checking | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "QuestionAnswering": "task: question answering | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "Retrieval-query": "task: search result | query: ",
    "STS": "task: sentence similarity | query: ",
    "SearchQuery": "task: search result | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
    "Summarization": "task: sentence similarity | query: ",
    "document": "title: none | text: ",
    "query": "task: search result | query: ",
}


class EmbeddingGemmaTokenizer:
    """Return dynamically padded token IDs using the pinned official tokenizer."""

    def __init__(
        self,
        backend: Any,
        *,
        max_context_length: int,
        model_max_position_embeddings: int,
        context_length: int,
        prompt_name: str | None,
    ) -> None:
        self.backend = backend
        self.max_context_length = max_context_length
        self.model_max_position_embeddings = model_max_position_embeddings
        self.context_length = context_length
        self.prompt_name = prompt_name
        self.prompt = OFFICIAL_PROMPTS.get(prompt_name) if prompt_name else None
        self.prompts = dict(OFFICIAL_PROMPTS)

        if backend.padding_side != "right" or backend.truncation_side != "right":
            raise RuntimeError(
                "The pinned EmbeddingGemma tokenizer must use right padding and "
                "right truncation."
            )
        for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
            token_id = getattr(backend, name, None)
            if not isinstance(token_id, int):
                raise TypeError(
                    f"The pinned EmbeddingGemma tokenizer has no valid {name}."
                )

    def __call__(self, texts: str | Sequence[str]) -> torch.Tensor:
        """Tokenize one string or a non-empty sequence into a LongTensor batch."""
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

        tokenization_inputs: list[str] = []
        for index, text in enumerate(batch):
            if self.prompt is not None and any(
                text.startswith(prefix) for prefix in self.prompts.values()
            ):
                raise ValueError(
                    f"texts[{index}] already starts with an official task prompt; "
                    "omit it when using prompt_name."
                )
            tokenization_inputs.append(
                f"{self.prompt}{text}" if self.prompt is not None else text
            )

        features = self.backend(
            tokenization_inputs,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.context_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = features.get("input_ids")
        attention_mask = features.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(
            attention_mask, torch.Tensor
        ):
            raise TypeError(
                "The official tokenizer must return input_ids and attention_mask "
                "as PyTorch tensors."
            )
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise RuntimeError(
                "The official tokenizer must return input_ids as LongTensor[B,L]."
            )
        if tuple(input_ids.shape) != tuple(attention_mask.shape):
            raise RuntimeError(
                "The official tokenizer returned mismatched input_ids and "
                "attention_mask shapes."
            )
        if input_ids.shape[0] != len(batch):
            raise RuntimeError("The official tokenizer changed the input batch size.")
        if input_ids.shape[1] > self.context_length:
            raise RuntimeError(
                "The official tokenizer returned a sequence longer than context_length."
            )

        pad_token_id = self.backend.pad_token_id
        official_mask = attention_mask.to(dtype=torch.bool)
        recovered_mask = input_ids.ne(pad_token_id)
        if (input_ids.eq(pad_token_id) & official_mask).any().item():
            raise ValueError(
                "Input text contains the PAD token as content. Its attention mask "
                "cannot be represented by the OpenCLIP-compatible token-ID tensor."
            )
        if not torch.equal(official_mask, recovered_mask):
            raise RuntimeError(
                "The official attention mask cannot be reconstructed from the "
                "token IDs and PAD token."
            )

        lengths = official_mask.sum(dim=1)
        if (input_ids[:, 0] != self.backend.bos_token_id).any().item():
            raise RuntimeError("The official tokenizer omitted the BOS token.")
        final_tokens = input_ids.gather(
            1,
            (lengths - 1).unsqueeze(1),
        ).squeeze(1)
        if (final_tokens != self.backend.eos_token_id).any().item():
            raise RuntimeError(
                "The official tokenizer omitted the final EOS token after truncation."
            )
        return input_ids


def get_tokenizer(
    model_name: str = MODEL_NAME,
    *,
    prompt_name: str | None = None,
    context_length: int | None = None,
    cache_dir: str | None = None,
) -> EmbeddingGemmaTokenizer:
    """Load the tokenizer and optionally apply an official task prompt.

    ``context_length`` bounds truncation; returned batches are padded to the
    longest sequence in that batch rather than to this full length.
    """
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string.")
    normalized_model_name = model_name.strip().lower().replace("_", "-")
    if normalized_model_name != MODEL_NAME:
        raise ValueError(
            f"Unsupported model_name {model_name!r}; only {MODEL_NAME!r} is supported."
        )
    if prompt_name is not None and prompt_name not in OFFICIAL_PROMPTS:
        available = ", ".join(sorted(OFFICIAL_PROMPTS))
        raise ValueError(
            f"Unsupported prompt_name {prompt_name!r}; choose one of: {available}."
        )
    if context_length is not None and (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length < 2
    ):
        raise ValueError("context_length must be an integer of at least 2.")
    if cache_dir is not None and not isinstance(cache_dir, str):
        raise TypeError("cache_dir must be a string or None.")

    try:
        config = AutoConfig.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to load the configuration for {MODEL_ID!r} "
            f"at revision {MODEL_REVISION}."
        ) from error

    text_config = getattr(config, "text_config", None)
    max_context_length = getattr(text_config, "max_position_embeddings", None)
    if (
        isinstance(max_context_length, bool)
        or not isinstance(max_context_length, int)
        or max_context_length < 2
    ):
        raise RuntimeError(
            "The model configuration has no valid text max_position_embeddings."
        )
    supported_context_length = min(
        max_context_length,
        MODEL_CARD_MAX_CONTEXT_LENGTH,
    )
    resolved_context_length = (
        supported_context_length if context_length is None else context_length
    )
    if resolved_context_length > supported_context_length:
        raise ValueError(
            f"context_length={resolved_context_length} exceeds the documented "
            f"supported context limit of {supported_context_length}."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to load the tokenizer for {MODEL_ID!r} "
            f"at revision {MODEL_REVISION}."
        ) from error

    return EmbeddingGemmaTokenizer(
        tokenizer,
        max_context_length=supported_context_length,
        model_max_position_embeddings=max_context_length,
        context_length=resolved_context_length,
        prompt_name=prompt_name,
    )
