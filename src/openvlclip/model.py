"""PyTorch model wrapper for the supported EmbeddingGemma 2 backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NoReturn

import torch
from torch import nn

from openvlclip.embedding_space import EmbeddingSpaceMetadata

if TYPE_CHECKING:
    from openvlclip.backends.base import (
        Backend,
        MultimodalInput,
        ProcessedImageBatch,
        TokenizerAPI,
        VideoInput,
    )


class EmbeddingGemmaCLIP(nn.Module):
    """Wrap the Transformers backend with an OpenCLIP-inspired inference API."""

    def __init__(
        self,
        backend: nn.Module,
        processor: Any,
        *,
        output_dict: bool = False,
        logit_scale_mode: str = "strict",
        modalities: tuple[str, ...] = ("text", "image"),
        inference_backend: Backend[Any, Any, Any] | None = None,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(backend, nn.Module):
            raise TypeError("backend must be a torch.nn.Module.")
        if not isinstance(output_dict, bool):
            raise TypeError("output_dict must be a bool.")
        if not isinstance(logit_scale_mode, str):
            raise TypeError("logit_scale_mode must be a string.")
        if logit_scale_mode not in {"strict", "cosine"}:
            raise ValueError("logit_scale_mode must be 'strict' or 'cosine'.")
        if cache_dir is not None and not isinstance(cache_dir, str):
            raise TypeError("cache_dir must be a string or None.")
        if not isinstance(modalities, tuple):
            raise TypeError(
                "modalities must be a tuple containing 'text' and optionally 'image'."
            )
        if any(not isinstance(modality, str) for modality in modalities):
            raise TypeError("modalities must contain only strings.")
        if (
            "text" not in modalities
            or len(set(modalities)) != len(modalities)
            or any(modality not in {"text", "image"} for modality in modalities)
        ):
            raise ValueError("modalities must include 'text' and may include 'image'.")
        if inference_backend is None:
            from openvlclip.backends.embeddinggemma2 import EmbeddingGemma2Backend

            inference_backend = EmbeddingGemma2Backend()
        if not callable(
            getattr(inference_backend, "encode_image", None)
        ) or not callable(getattr(inference_backend, "encode_text", None)):
            raise TypeError("inference_backend must implement image and text encoding.")
        if not callable(
            getattr(inference_backend, "validate_embedding_dimension", None)
        ):
            raise TypeError(
                "inference_backend must implement validate_embedding_dimension()."
            )
        self.backend = backend
        object.__setattr__(self, "_processor", processor)
        object.__setattr__(self, "_inference_backend", inference_backend)
        self.output_dict = output_dict
        self.logit_scale_mode = logit_scale_mode
        self.cache_dir = cache_dir
        self.modalities = tuple(
            modality for modality in ("text", "image") if modality in modalities
        )

    @property
    def processor(self) -> Any:
        """Return the processor without registering it as a PyTorch submodule."""
        return self._processor

    @property
    def capabilities(self) -> Mapping[str, bool]:
        """Return read-only capabilities for the loaded model modalities."""
        return self._inference_backend.capabilities_for(self.modalities)

    @property
    def embedding_space_metadata(self) -> EmbeddingSpaceMetadata:
        """Return backend-owned identity settings used by persisted indexes."""
        return self._inference_backend.embedding_space_metadata

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int:
        """Validate a dimension using this model's backend policy."""
        return self._inference_backend.validate_embedding_dimension(
            value,
            name=name,
        )

    def _create_tokenizer(
        self,
        *,
        prompt_name: str | None,
    ) -> TokenizerAPI:
        """Create a query tokenizer through this model's selected backend."""
        metadata = self.embedding_space_metadata
        return self._inference_backend.create_tokenizer(
            model_id=metadata.model_id,
            revision=metadata.model_revision,
            cache_dir=self.cache_dir,
            prompt_name=prompt_name,
            context_length=None,
        )

    def to(self, *args: Any, **kwargs: Any) -> EmbeddingGemmaCLIP:
        """Move the model while preventing unsupported FP16 conversions."""
        requested_dtype = kwargs.get("dtype")
        for argument in args:
            if isinstance(argument, torch.dtype):
                requested_dtype = argument
            elif isinstance(argument, torch.Tensor):
                requested_dtype = argument.dtype
        if requested_dtype == torch.float16:
            raise ValueError("FP16 is not supported; use FP32 or BF16.")
        return super().to(*args, **kwargs)

    def half(self) -> EmbeddingGemmaCLIP:
        """Reject the PyTorch convenience conversion to unsupported FP16."""
        raise ValueError("FP16 is not supported; use FP32 or BF16.")

    def forward(
        self,
        image: ProcessedImageBatch | None = None,
        text: torch.Tensor | None = None,
        *,
        truncate_dim: int | None = None,
    ) -> (
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]
        | dict[str, object]
    ):
        """Return normalized features and an explicit, untrained cosine scale."""
        if image is None and text is None:
            raise ValueError("At least one of image or text must be provided.")
        if self.logit_scale_mode != "cosine":
            raise NotImplementedError(
                "forward() requires logit_scale_mode='cosine'; EmbeddingGemma 2 "
                "has no trained OpenCLIP logit_scale."
            )

        expected_dimension = self.validate_embedding_dimension(
            truncate_dim,
            name="truncate_dim",
        )
        image_features = (
            self.encode_image(
                image,
                normalize=True,
                truncate_dim=truncate_dim,
            )
            if image is not None
            else None
        )
        text_features = (
            self.encode_text(
                text,
                normalize=True,
                truncate_dim=truncate_dim,
            )
            if text is not None
            else None
        )
        for features in (image_features, text_features):
            if features is not None and (
                features.ndim != 2 or features.shape[1] != expected_dimension
            ):
                raise RuntimeError(
                    "Encoded image and text features must have "
                    f"{expected_dimension} dimensions."
                )
        if image_features is not None:
            feature_device = image_features.device
        elif text_features is not None:
            feature_device = text_features.device
        else:
            raise RuntimeError("At least one encoded feature tensor is required.")
        logit_scale = torch.ones((), dtype=torch.float32, device=feature_device)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": logit_scale,
            }
        return image_features, text_features, logit_scale

    def get_logits(
        self,
        image: ProcessedImageBatch,
        text: torch.Tensor,
        *,
        truncate_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image/text cosine similarity matrices in both orientations."""
        if self.logit_scale_mode != "cosine":
            raise NotImplementedError(
                "get_logits() requires logit_scale_mode='cosine'; EmbeddingGemma 2 "
                "has no trained OpenCLIP logit_scale."
            )

        expected_dimension = self.validate_embedding_dimension(
            truncate_dim,
            name="truncate_dim",
        )
        image_features = self.encode_image(
            image,
            normalize=True,
            truncate_dim=truncate_dim,
        )
        text_features = self.encode_text(
            text,
            normalize=True,
            truncate_dim=truncate_dim,
        )
        if image_features.ndim != 2 or text_features.ndim != 2:
            raise RuntimeError(
                "Encoded image and text features must be rank-2 tensors."
            )
        if (
            image_features.shape[1] != expected_dimension
            or text_features.shape[1] != expected_dimension
        ):
            raise RuntimeError(
                "Encoded image and text features must have "
                f"{expected_dimension} dimensions."
            )
        if image_features.device != text_features.device:
            raise ValueError(
                "Encoded image and text features must be on the same device."
            )
        if image_features.dtype != text_features.dtype:
            raise TypeError("Encoded image and text features must have the same dtype.")
        if (
            not torch.isfinite(image_features).all().item()
            or not torch.isfinite(text_features).all().item()
        ):
            raise ValueError("Encoded image and text features must be finite.")

        if (
            image_features.device.type == "cpu"
            and image_features.dtype == torch.bfloat16
        ):
            # CPU BF16 matmul is not supported by every processor feature set.
            image_logits = image_features.float() @ text_features.float().T
            image_logits = image_logits.to(dtype=image_features.dtype)
        else:
            image_logits = image_features @ text_features.T
        if not torch.isfinite(image_logits).all().item():
            raise RuntimeError("Cosine similarity produced non-finite values.")
        return image_logits, image_logits.T

    def encode_image(
        self,
        image: ProcessedImageBatch,
        normalize: bool = False,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        """Delegate image validation, inference, pooling, and MRL to the backend."""
        if "image" not in self.modalities:
            raise RuntimeError(
                "encode_image() is unavailable because the model was loaded "
                "without the image modality."
            )
        return self._inference_backend.encode_image(
            self,
            image,
            normalize=normalize,
            truncate_dim=truncate_dim,
        )

    def encode_text(
        self,
        text: torch.Tensor,
        normalize: bool = False,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        """Delegate token validation, inference, pooling, and MRL to the backend."""
        return self._inference_backend.encode_text(
            self,
            text,
            normalize=normalize,
            truncate_dim=truncate_dim,
        )

    def encode_texts(
        self,
        texts: str | Sequence[str],
        *,
        instruction: str | None = None,
        normalize: bool = False,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        """Tokenize plain text and delegate embedding inference to the backend."""
        if instruction is not None:
            raise NotImplementedError(
                "EmbeddingGemma 2 uses registered task prompts. Create tokens with "
                "get_tokenizer(prompt_name=...) and call encode_text()."
            )
        if "text" not in self.modalities:
            raise RuntimeError(
                "encode_texts() is unavailable because the model was loaded "
                "without the text modality."
            )
        token_ids = self._create_tokenizer(prompt_name=None)(texts)
        return self.encode_text(
            token_ids,
            normalize=normalize,
            truncate_dim=truncate_dim,
        )

    def encode_multimodal(
        self,
        items: MultimodalInput,
        *,
        normalize: bool = False,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        """Raise because EmbeddingGemma 2 does not jointly embed modalities."""
        del items, normalize, truncate_dim
        raise NotImplementedError(
            "encode_multimodal() is not supported by the EmbeddingGemma 2 backend."
        )

    def encode_video(
        self,
        video: VideoInput,
        *,
        normalize: bool = False,
        truncate_dim: int | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> torch.Tensor:
        """Raise because EmbeddingGemma 2 does not expose video embeddings."""
        del video, normalize, truncate_dim, fps, max_frames
        raise NotImplementedError(
            "encode_video() is not supported by the EmbeddingGemma 2 backend."
        )

    def encode_patches(
        self,
        image: ProcessedImageBatch,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Raise because EmbeddingGemma 2 does not expose aligned patch features."""
        del image, normalize
        raise NotImplementedError(
            "encode_patches() is not supported by the EmbeddingGemma 2 backend."
        )

    def similarity_map(
        self,
        image: ProcessedImageBatch,
        text: str | Sequence[str],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Raise because EmbeddingGemma 2 has no text-aligned patch features."""
        del image, text, output_size
        raise NotImplementedError(
            "similarity_map() is not supported by the EmbeddingGemma 2 backend."
        )

    def encode(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Sentence Transformers encode compatibility is not in the current scope."""
        raise NotImplementedError("encode is not part of the OpenCLIP-compatible API.")
