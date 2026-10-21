"""Minimal protocol shared by registered model backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import torch
from PIL import Image

from openclip_gemma.embedding_space import EmbeddingSpaceMetadata

if TYPE_CHECKING:
    from openclip_gemma.model import EmbeddingGemmaCLIP
    from openclip_gemma.tokenizer import EmbeddingGemmaTokenizer

type ImagePreprocess = Callable[[Image.Image], torch.Tensor]
type BackendTokenizer = Callable[[str | Sequence[str]], torch.Tensor]


class Backend(Protocol):
    """Model-specific loading and inference operations used by the shared Factory."""

    backend_id: str
    capabilities: Mapping[str, bool]
    embedding_space_metadata: EmbeddingSpaceMetadata

    def capabilities_for(self, modalities: tuple[str, ...]) -> Mapping[str, bool]: ...

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
    ) -> EmbeddingGemmaCLIP: ...

    def create_tokenizer(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: str | None,
        prompt_name: str | None,
        context_length: int | None,
    ) -> EmbeddingGemmaTokenizer: ...

    def create_preprocess(self, model: EmbeddingGemmaCLIP) -> ImagePreprocess: ...

    def encode_image(
        self,
        model: EmbeddingGemmaCLIP,
        image: torch.Tensor,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor: ...

    def encode_text(
        self,
        model: EmbeddingGemmaCLIP,
        text: torch.Tensor,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor: ...
