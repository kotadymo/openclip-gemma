"""Minimal protocol shared by registered model backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeVar

import torch

from openvlclip._dimensions import validate_positive_embedding_dimension
from openvlclip.embedding_space import EmbeddingSpaceMetadata

ModelT = TypeVar("ModelT", bound=torch.nn.Module)
TokenizerT = TypeVar("TokenizerT")
PreprocessT = TypeVar("PreprocessT")


class Backend(Protocol[ModelT, TokenizerT, PreprocessT]):
    """Model-specific loading and inference operations used by the shared Factory."""

    backend_id: str
    capabilities: Mapping[str, bool]
    embedding_space_metadata: EmbeddingSpaceMetadata

    def capabilities_for(self, modalities: tuple[str, ...]) -> Mapping[str, bool]: ...

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int: ...

    def load_model(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: str | None,
        device: torch.device,
        dtype: torch.dtype,
        output_dict: bool,
        logit_scale_mode: str,
        modalities: tuple[str, ...] | None,
    ) -> ModelT: ...

    def create_tokenizer(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: str | None,
        prompt_name: str | None,
        context_length: int | None,
    ) -> TokenizerT: ...

    def create_preprocess(self, model: ModelT) -> PreprocessT: ...

    def encode_image(
        self,
        model: ModelT,
        image: torch.Tensor,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor: ...

    def encode_text(
        self,
        model: ModelT,
        text: torch.Tensor,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor: ...


def validate_model_embedding_dimension(
    model: object,
    value: object,
    *,
    name: str,
) -> int:
    """Delegate dimension policy to a model backend when one is available."""
    validator = getattr(model, "validate_embedding_dimension", None)
    if callable(validator):
        dimension = validator(value, name=name)
    else:
        dimension = validate_positive_embedding_dimension(value, name=name)

    return validate_positive_embedding_dimension(dimension, name=name)
