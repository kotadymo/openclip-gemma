"""EmbeddingGemma 2 loader, tokenizer, preprocessing, and inference backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

from openvlclip._dimensions import (
    DEFAULT_EMBEDDING_DIMENSION,
    EMBEDDING_STRATEGY,
    validate_positive_embedding_dimension,
)
from openvlclip.backends.base import ProcessedImageBatch, require_tensor_image
from openvlclip.capabilities import capabilities_for_modalities, freeze_capabilities
from openvlclip.constants import (
    BACKEND_ID,
    EMBEDDING_NORMALIZATION,
    IMAGE_PREPROCESSING_ID,
    MODEL_CARD_MAX_CONTEXT_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
)
from openvlclip.embedding_space import EmbeddingSpaceMetadata
from openvlclip.model import EmbeddingGemmaCLIP
from openvlclip.tokenizer import (
    OFFICIAL_PROMPTS,
    EmbeddingGemmaTokenizer,
)
from openvlclip.transforms import EmbeddingGemmaPreprocess

SUPPORTED_EMBEDDING_DIMENSIONS = (128, 256, 512, 768)
_NATIVE_EMBEDDING_DIMENSION = DEFAULT_EMBEDDING_DIMENSION


def _truncate_and_normalize(
    embeddings: torch.Tensor,
    *,
    truncate_dim: int,
    modality: str,
) -> torch.Tensor:
    if truncate_dim == _NATIVE_EMBEDDING_DIMENSION:
        return embeddings

    truncated = embeddings[..., :truncate_dim]
    norms = torch.linalg.vector_norm(truncated.float(), dim=-1)
    if not torch.isfinite(norms).all().item() or (norms <= 0).any().item():
        raise RuntimeError(
            f"The backend produced an invalid {modality} embedding after truncation "
            f"to {truncate_dim} dimensions."
        )
    return F.normalize(truncated, p=2, dim=-1)


def _pool_and_normalize(
    outputs: Any,
    attention_mask: torch.Tensor,
    *,
    batch_size: int,
    normalize: bool,
    truncate_dim: int,
    modality: str,
) -> torch.Tensor:
    hidden = outputs.last_hidden_state
    if hidden.ndim != 3 or hidden.shape[0] != batch_size:
        raise RuntimeError(
            f"The backend returned an invalid {modality} last_hidden_state shape: "
            f"{tuple(hidden.shape)}."
        )
    if hidden.shape[-1] != _NATIVE_EMBEDDING_DIMENSION:
        raise RuntimeError(
            "The backend returned an unexpected embedding dimension "
            f"{hidden.shape[-1]}; expected {_NATIVE_EMBEDDING_DIMENSION}."
        )
    if tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
        raise RuntimeError(
            "attention_mask shape does not match last_hidden_state: "
            f"{tuple(attention_mask.shape)} != {tuple(hidden.shape[:2])}."
        )

    mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    counts = weights.sum(dim=1)
    if (counts == 0).any().item():
        raise RuntimeError("The processor returned an empty attention mask.")
    pooled = (hidden * weights).sum(dim=1) / counts

    embeddings = F.normalize(pooled, p=2, dim=-1)
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    embeddings = _truncate_and_normalize(
        embeddings,
        truncate_dim=truncate_dim,
        modality=modality,
    )
    if not torch.isfinite(embeddings).all().item():
        raise RuntimeError(f"The backend produced non-finite {modality} embeddings.")
    return embeddings


class EmbeddingGemma2Backend:
    """Implement the pinned Transformers path without changing inference math."""

    backend_id = BACKEND_ID
    capabilities: Mapping[str, bool] = freeze_capabilities(
        {
            "image_embedding": True,
            "text_embedding": True,
            "image_text_similarity": True,
            "patch_embedding": False,
            "text_similarity_map": False,
            "mrl": True,
            "selective_loading": True,
            "video_embedding": False,
            "multimodal_embedding": False,
            "instruction_aware": True,
            "variable_resolution": True,
            "sigmoid_logits": False,
        }
    )
    embedding_space_metadata = EmbeddingSpaceMetadata(
        backend_id=BACKEND_ID,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        embedding_strategy=EMBEDDING_STRATEGY,
        normalization=EMBEDDING_NORMALIZATION,
        preprocessing_id=IMAGE_PREPROCESSING_ID,
        index_prompt=None,
        query_prompt_name="SearchQuery",
        query_prompt=OFFICIAL_PROMPTS["SearchQuery"],
    )

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int:
        """Validate dimensions supported by EmbeddingGemma 2's MRL head."""
        if value is None:
            return _NATIVE_EMBEDDING_DIMENSION
        dimension = validate_positive_embedding_dimension(value, name=name)
        if dimension not in SUPPORTED_EMBEDDING_DIMENSIONS:
            supported = ", ".join(str(item) for item in SUPPORTED_EMBEDDING_DIMENSIONS)
            raise ValueError(
                f"{name} must be one of {{{supported}}}; received {dimension}."
            )
        return dimension

    def capabilities_for(self, modalities: tuple[str, ...]) -> Mapping[str, bool]:
        """Expose only operations supported by the loaded modalities."""
        return capabilities_for_modalities(self.capabilities, modalities)

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
    ) -> EmbeddingGemmaCLIP:
        """Load the pinned model and processor using the existing selective path."""
        try:
            model_kwargs: dict[str, Any] = {
                "revision": revision,
                "cache_dir": cache_dir,
                "dtype": dtype,
            }
            if modalities is not None:
                config = AutoConfig.from_pretrained(
                    model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                )
                if "image" not in modalities:
                    config.vision_config = None
                config.audio_config = None
                model_kwargs["config"] = config
            backend = AutoModel.from_pretrained(model_id, **model_kwargs)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load {model_id!r} at revision {revision}."
            ) from error

        if not isinstance(backend, torch.nn.Module):
            raise TypeError(
                "The Transformers model loader did not return an nn.Module."
            )

        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load the processor for {model_id!r} at revision {revision}."
            ) from error

        if (modalities is None or "image" in modalities) and getattr(
            processor, "image_processor", None
        ) is None:
            raise RuntimeError("The loaded processor has no image_processor.")

        backend = backend.to(device=device, dtype=dtype).eval()
        return EmbeddingGemmaCLIP(
            backend=backend,
            processor=processor,
            output_dict=output_dict,
            logit_scale_mode=logit_scale_mode,
            modalities=modalities or ("text", "image"),
            inference_backend=self,
        ).eval()

    def create_tokenizer(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: str | None,
        prompt_name: str | None,
        context_length: int | None,
    ) -> EmbeddingGemmaTokenizer:
        """Load the official tokenizer and validate its documented input limit."""
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
                model_id,
                revision=revision,
                cache_dir=cache_dir,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load the configuration for {model_id!r} "
                f"at revision {revision}."
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
                model_id,
                revision=revision,
                cache_dir=cache_dir,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load the tokenizer for {model_id!r} at revision {revision}."
            ) from error

        return EmbeddingGemmaTokenizer(
            tokenizer,
            max_context_length=supported_context_length,
            model_max_position_embeddings=max_context_length,
            context_length=resolved_context_length,
            prompt_name=prompt_name,
        )

    def create_preprocess(self, model: EmbeddingGemmaCLIP) -> EmbeddingGemmaPreprocess:
        """Create the official deterministic image transform for a loaded model."""
        image_processor = getattr(model.processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("The loaded processor has no image_processor.")
        return EmbeddingGemmaPreprocess(image_processor)

    def encode_image(
        self,
        model: EmbeddingGemmaCLIP,
        image: ProcessedImageBatch,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor:
        """Encode preprocessed images through the official model and pooling path."""
        image = require_tensor_image(image, backend_name="EmbeddingGemma 2")
        if image.ndim != 4:
            raise ValueError(
                f"image must have shape [B,3,H,W]; received {tuple(image.shape)}."
            )
        if image.shape[0] == 0:
            raise ValueError("image batch must contain at least one image.")
        if image.shape[1] != 3:
            raise ValueError(f"image must have 3 channels; received {image.shape[1]}.")
        if image.shape[2] <= 0 or image.shape[3] <= 0:
            raise ValueError("image height and width must be positive.")
        if image.dtype != torch.float32:
            raise TypeError(
                f"image must use torch.float32 from preprocess; received {image.dtype}."
            )
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool.")
        resolved_truncate_dim = self.validate_embedding_dimension(
            truncate_dim,
            name="truncate_dim",
        )

        try:
            model_device = next(model.backend.parameters()).device
        except StopIteration as error:
            raise RuntimeError(
                "The backend has no parameters to determine its device."
            ) from error
        if image.device != model_device:
            raise ValueError(
                f"image is on {image.device}, but the model backend is on {model_device}."
            )
        if not torch.isfinite(image).all().item():
            raise ValueError("image contains NaN or infinite values.")
        if image.min().item() < 0 or image.max().item() > 1:
            raise ValueError("preprocessed image values must be in the [0,1] range.")

        image_processor = model.processor.image_processor
        patch_size = image_processor.patch_size
        if image.shape[2] % patch_size or image.shape[3] % patch_size:
            raise ValueError(
                "preprocessed image height and width must be multiples of "
                f"patch_size={patch_size}; received {image.shape[2:]}."
            )
        patch_count = (image.shape[2] // patch_size) * (image.shape[3] // patch_size)
        max_patches = (
            image_processor.max_soft_tokens * image_processor.pooling_kernel_size**2
        )
        if patch_count > max_patches:
            raise ValueError(
                f"preprocessed image contains {patch_count} patches, exceeding "
                f"the processor limit of {max_patches}."
            )

        image_groups = [[sample] for sample in image.unbind(dim=0)]
        processor_features = model.processor(
            images=image_groups,
            text=[model.processor.image_token] * image.shape[0],
            padding=True,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        required_features = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_position_ids",
        }
        missing_features = required_features.difference(processor_features)
        if missing_features:
            raise RuntimeError(
                "The official processor omitted required model inputs: "
                f"{sorted(missing_features)}."
            )
        model_inputs = {
            name: value.to(model_device) if isinstance(value, torch.Tensor) else value
            for name, value in processor_features.items()
        }

        with torch.inference_mode():
            outputs = model.backend(**model_inputs)
            return _pool_and_normalize(
                outputs,
                model_inputs["attention_mask"],
                batch_size=image.shape[0],
                normalize=normalize,
                truncate_dim=resolved_truncate_dim,
                modality="image",
            )

    def encode_text(
        self,
        model: EmbeddingGemmaCLIP,
        text: torch.Tensor,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor:
        """Encode official token IDs with the pinned pooling and normalization."""
        if not isinstance(text, torch.Tensor):
            raise TypeError("text must be a torch.Tensor.")
        if text.ndim != 2:
            raise ValueError(
                f"text must have shape [B,L]; received {tuple(text.shape)}."
            )
        if text.shape[0] == 0:
            raise ValueError("text batch must contain at least one sequence.")
        if text.shape[1] == 0:
            raise ValueError("text sequence length must be positive.")
        if text.dtype != torch.long:
            raise TypeError(
                f"text must use torch.int64 token IDs; received {text.dtype}."
            )
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool.")
        resolved_truncate_dim = self.validate_embedding_dimension(
            truncate_dim,
            name="truncate_dim",
        )

        try:
            model_device = next(model.backend.parameters()).device
        except StopIteration as error:
            raise RuntimeError(
                "The backend has no parameters to determine its device."
            ) from error
        if text.device != model_device:
            raise ValueError(
                f"text is on {text.device}, but the model backend is on {model_device}."
            )

        tokenizer = getattr(model.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("The processor has no tokenizer.")
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        vocab_size = getattr(tokenizer, "vocab_size", None)
        text_config = getattr(
            getattr(model.backend, "config", None), "text_config", None
        )
        max_context_length = getattr(text_config, "max_position_embeddings", None)
        if not all(
            isinstance(token_id, int)
            for token_id in (pad_token_id, bos_token_id, eos_token_id, vocab_size)
        ):
            raise RuntimeError("The processor tokenizer has incomplete token metadata.")
        if (
            isinstance(max_context_length, bool)
            or not isinstance(max_context_length, int)
            or max_context_length < 2
        ):
            raise RuntimeError(
                "The backend configuration has no valid text positional limit."
            )
        supported_context_length = min(
            max_context_length,
            MODEL_CARD_MAX_CONTEXT_LENGTH,
        )
        if text.shape[1] > supported_context_length:
            raise ValueError(
                f"text sequence length {text.shape[1]} exceeds the documented "
                f"supported context limit of {supported_context_length}."
            )
        if (text < 0).any().item() or (text >= vocab_size).any().item():
            raise ValueError(f"text token IDs must be in the range [0, {vocab_size}).")

        attention_mask = text.ne(pad_token_id)
        padding_seen = text.eq(pad_token_id).cumsum(dim=1).gt(0)
        if (attention_mask & padding_seen).any().item():
            raise ValueError(
                "PAD token IDs must occur only as a right-padding suffix; PAD cannot "
                "be represented as text content in the token-ID-only API."
            )
        if (text[:, 0] != bos_token_id).any().item():
            raise ValueError(
                "Each text sequence must begin with the official BOS token."
            )
        lengths = attention_mask.sum(dim=1)
        if (lengths < 2).any().item():
            raise ValueError("Each text sequence must contain BOS and EOS tokens.")
        final_tokens = text.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        if (final_tokens != eos_token_id).any().item():
            raise ValueError(
                "Each text sequence must end with the official EOS token before "
                "right-padding."
            )

        with torch.inference_mode():
            outputs = model.backend(
                input_ids=text,
                attention_mask=attention_mask.to(dtype=torch.long),
            )
            return _pool_and_normalize(
                outputs,
                attention_mask,
                batch_size=text.shape[0],
                normalize=normalize,
                truncate_dim=resolved_truncate_dim,
                modality="text",
            )
