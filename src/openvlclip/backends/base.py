"""Minimal protocol shared by registered model backends."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from os import PathLike
from typing import Any, Protocol, Self, TypeVar

import torch
from PIL import Image

from openvlclip._dimensions import (
    DEFAULT_EMBEDDING_DIMENSION,
    validate_positive_embedding_dimension,
)
from openvlclip.embedding_space import EmbeddingSpaceMetadata

ModelT = TypeVar("ModelT", bound=torch.nn.Module)
TokenizerT = TypeVar("TokenizerT")
PreprocessT = TypeVar("PreprocessT")

type ProcessedImage = torch.Tensor | Mapping[str, object]
type ProcessedImageBatch = ProcessedImage | Sequence[ProcessedImage]
type MultimodalInput = Mapping[str, object] | Sequence[Mapping[str, object]]
type VideoInput = str | PathLike[str] | Sequence[Image.Image | str | PathLike[str]]


class InferenceModelAPI(Protocol):
    """Model-independent operations returned by the public Factory."""

    @property
    def capabilities(self) -> Mapping[str, bool]: ...

    @property
    def embedding_space_metadata(self) -> EmbeddingSpaceMetadata: ...

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int: ...

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]: ...

    def encode_image(
        self,
        image: ProcessedImageBatch,
        normalize: bool = False,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor: ...

    def encode_text(
        self,
        text: torch.Tensor,
        normalize: bool = False,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor: ...

    def encode_texts(
        self,
        texts: str | Sequence[str],
        *,
        instruction: str | None = None,
        normalize: bool = False,
        truncate_dim: int | None = None,
    ) -> torch.Tensor: ...

    def encode_multimodal(
        self,
        items: MultimodalInput,
        *,
        normalize: bool = False,
        truncate_dim: int | None = None,
    ) -> torch.Tensor: ...

    def encode_video(
        self,
        video: VideoInput,
        *,
        normalize: bool = False,
        truncate_dim: int | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> torch.Tensor: ...

    def encode_patches(
        self,
        image: ProcessedImageBatch,
        normalize: bool = False,
    ) -> torch.Tensor: ...

    def similarity_map(
        self,
        image: ProcessedImageBatch,
        text: str | Sequence[str],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor: ...

    def forward(
        self,
        image: ProcessedImageBatch | None = None,
        text: torch.Tensor | None = None,
        *,
        truncate_dim: int | None = None,
    ) -> (
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]
        | dict[str, object]
    ): ...

    def get_logits(
        self,
        image: ProcessedImageBatch,
        text: torch.Tensor,
        *,
        truncate_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def eval(self) -> Self: ...

    def to(self, *args: Any, **kwargs: Any) -> Self: ...


class TokenizerAPI(Protocol):
    """Callable tokenizer contract shared by registered model backends."""

    def __call__(self, texts: str | Sequence[str]) -> torch.Tensor: ...


class ImagePreprocessAPI(Protocol):
    """Callable PIL-image preprocessing contract shared by model backends."""

    def __call__(self, image: Image.Image) -> ProcessedImage: ...


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
        image: ProcessedImageBatch,
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


def require_tensor_image(
    image: ProcessedImageBatch,
    *,
    backend_name: str,
) -> torch.Tensor:
    """Reject processor-aware inputs in backends that only consume image tensors."""
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            f"image must be a torch.Tensor for the {backend_name} backend; "
            "processor-aware image mappings are not supported."
        )
    return image


def validate_model_embedding_dimension(
    model: object,
    value: object | None,
    *,
    name: str,
) -> int:
    """Delegate dimension policy to a model backend when one is available."""
    validator = getattr(model, "validate_embedding_dimension", None)
    if callable(validator):
        dimension = validator(value, name=name)
    else:
        dimension = validate_positive_embedding_dimension(
            DEFAULT_EMBEDDING_DIMENSION if value is None else value,
            name=name,
        )

    return validate_positive_embedding_dimension(dimension, name=name)
