"""Pinned Transformers backends for SigLIP 2 FixRes and NaFlex."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from openvlclip._dimensions import validate_positive_embedding_dimension
from openvlclip.backends.base import (
    MultimodalInput,
    ProcessedImage,
    ProcessedImageBatch,
    VideoInput,
    stack_processed_images,
)
from openvlclip.capabilities import capabilities_for_modalities, freeze_capabilities
from openvlclip.embedding_space import EmbeddingSpaceMetadata
from openvlclip.model import EmbeddingGemmaCLIP

SIGLIP2_BACKEND_ID = "google-siglip2"

SIGLIP2_FIXRES_MODEL_NAME = "siglip2-base"
SIGLIP2_FIXRES_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP2_FIXRES_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"

SIGLIP2_NAFLEX_MODEL_NAME = "siglip2-naflex-base"
SIGLIP2_NAFLEX_MODEL_ID = "google/siglip2-base-patch16-naflex"
SIGLIP2_NAFLEX_REVISION = "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"

# Additional v0.8.0 fixed-resolution checkpoints (official Hugging Face revisions).
SIGLIP2_BASE_384_MODEL_NAME = "siglip2-base-384"
SIGLIP2_BASE_384_MODEL_ID = "google/siglip2-base-patch16-384"
SIGLIP2_BASE_384_REVISION = "f775b65a79762255128c981547af89addcfe0f88"

SIGLIP2_LARGE_256_MODEL_NAME = "siglip2-large-256"
SIGLIP2_LARGE_256_MODEL_ID = "google/siglip2-large-patch16-256"
SIGLIP2_LARGE_256_REVISION = "787800c8990e6f058423089178e718139608408c"

SIGLIP2_SO400M_384_MODEL_NAME = "siglip2-so400m-384"
SIGLIP2_SO400M_384_MODEL_ID = "google/siglip2-so400m-patch14-384"
SIGLIP2_SO400M_384_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"

_SIGLIP2_NATIVE_DIMENSION = 768
_MAX_CONTEXT_LENGTH = 64
_FIXRES_IMAGE_SHAPE = (3, 224, 224)
_NAFLEX_INPUT_KEYS = frozenset(
    {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
)
type _SiglipVariant = Literal["fixres", "naflex"]


class Siglip2Tokenizer:
    """Return fixed-length token IDs with an attention mask recoverable from PAD."""

    def __init__(self, tokenizer: Any, *, context_length: int) -> None:
        self.backend = tokenizer
        self.context_length = context_length
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
            raise TypeError("The pinned SigLIP 2 tokenizer has no valid pad token.")
        self.pad_token_id = pad_token_id

    def __call__(self, texts: str | Sequence[str]) -> torch.Tensor:
        """Tokenize text using the fixed 64-token SigLIP 2 processor contract."""
        if isinstance(texts, str):
            batch = [texts]
        elif isinstance(texts, Sequence) and not isinstance(texts, (bytes, bytearray)):
            batch = list(texts)
        else:
            raise TypeError("texts must be a string or a sequence of strings.")
        if not batch:
            raise ValueError("texts must contain at least one string.")
        for index, text in enumerate(batch):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] must be a string.")

        features = self.backend(
            batch,
            padding="max_length",
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
                "The pinned SigLIP 2 tokenizer must return input_ids and "
                "attention_mask as PyTorch tensors."
            )
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise RuntimeError(
                "The SigLIP 2 tokenizer must return input_ids as LongTensor[B,L]."
            )
        if tuple(input_ids.shape) != tuple(attention_mask.shape):
            raise RuntimeError(
                "The SigLIP 2 tokenizer returned mismatched input_ids and "
                "attention_mask shapes."
            )
        if (
            input_ids.shape[0] != len(batch)
            or input_ids.shape[1] != self.context_length
        ):
            raise RuntimeError(
                "The SigLIP 2 tokenizer changed the requested batch or context length."
            )
        if not torch.equal(
            attention_mask.to(dtype=torch.bool),
            input_ids.ne(self.pad_token_id),
        ):
            raise ValueError(
                "The SigLIP 2 attention mask cannot be reconstructed from token IDs "
                "and the PAD token."
            )
        return input_ids


class Siglip2ImagePreprocess:
    """Apply the pinned official FixRes or NaFlex image processor."""

    def __init__(
        self, processor: Any, *, variant: _SiglipVariant, image_size: int = 224
    ) -> None:
        self.processor = processor
        self.variant = variant
        self.image_shape = (3, image_size, image_size)

    def __call__(self, image: Image.Image) -> ProcessedImage:
        if not isinstance(image, Image.Image):
            raise TypeError("preprocess expects a PIL.Image.Image.")
        features = self.processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )
        if not isinstance(features, Mapping):
            raise TypeError("The official SigLIP 2 processor must return a mapping.")

        if self.variant == "fixres":
            if set(features) != {"pixel_values"}:
                raise RuntimeError(
                    "The pinned FixRes processor must return only pixel_values."
                )
            pixel_values = features.get("pixel_values")
            if (
                not isinstance(pixel_values, torch.Tensor)
                or tuple(pixel_values.shape) != (1, *self.image_shape)
                or not pixel_values.is_floating_point()
                or not torch.isfinite(pixel_values).all().item()
            ):
                raise RuntimeError(
                    "The pinned FixRes processor must return finite "
                    f"FloatTensor[1,{self.image_shape[0]},"
                    f"{self.image_shape[1]},{self.image_shape[2]}]."
                )
            return pixel_values[0]

        if set(features) != _NAFLEX_INPUT_KEYS:
            raise RuntimeError(
                "The pinned NaFlex processor must return pixel_values, "
                "pixel_attention_mask, and spatial_shapes."
            )
        image_inputs: dict[str, torch.Tensor] = {}
        for name in ("pixel_values", "pixel_attention_mask", "spatial_shapes"):
            value = features.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim < 2
                or value.shape[0] != 1
            ):
                raise RuntimeError(
                    f"The pinned NaFlex processor returned invalid {name}."
                )
            image_inputs[name] = value[0]
        pixel_values = image_inputs["pixel_values"]
        if (
            pixel_values.ndim != 2
            or pixel_values.shape[-1] != _SIGLIP2_NATIVE_DIMENSION
            or not pixel_values.is_floating_point()
            or not torch.isfinite(pixel_values).all().item()
        ):
            raise RuntimeError(
                "The pinned NaFlex processor must return finite "
                "FloatTensor[N,768] patch values."
            )
        return image_inputs


class Siglip2CLIP(EmbeddingGemmaCLIP):
    """OpenCLIP-style API wrapper with SigLIP 2-specific score semantics."""

    def encode_texts(
        self,
        texts: str | Sequence[str],
        *,
        instruction: str | None = None,
        normalize: bool = False,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        if instruction is not None:
            raise NotImplementedError(
                "SigLIP 2 is not instruction-aware; omit instruction."
            )
        return super().encode_texts(
            texts,
            normalize=normalize,
            truncate_dim=truncate_dim,
        )

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
        if self.logit_scale_mode != "cosine":
            raise NotImplementedError(
                "SigLIP 2 forward() supports only logit_scale_mode='cosine'; "
                "use get_logits() for the official learned scale and bias."
            )
        return super().forward(image, text, truncate_dim=truncate_dim)

    def encode_multimodal(
        self,
        items: MultimodalInput,
        *,
        normalize: bool = False,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "encode_multimodal() is not supported by the SigLIP 2 backend."
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
        raise NotImplementedError(
            "encode_video() is not supported by the SigLIP 2 backend."
        )

    def encode_patches(
        self,
        image: ProcessedImageBatch,
        normalize: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "encode_patches() is not supported by the SigLIP 2 backend."
        )

    def similarity_map(
        self,
        image: ProcessedImageBatch,
        text: str | Sequence[str],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "similarity_map() is not supported by the SigLIP 2 backend."
        )


class Siglip2Backend:
    """Load and infer with one of the pinned official SigLIP 2 checkpoints."""

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        revision: str,
        variant: _SiglipVariant,
        native_dimension: int = _SIGLIP2_NATIVE_DIMENSION,
        image_size: int = 224,
    ) -> None:
        if variant not in {"fixres", "naflex"}:
            raise ValueError("variant must be 'fixres' or 'naflex'.")
        if type(native_dimension) is not int or native_dimension <= 0:
            raise ValueError("native_dimension must be a positive integer.")
        if type(image_size) is not int or image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if variant == "naflex" and (native_dimension != 768 or image_size != 224):
            raise ValueError("NaFlex Base requires its existing 768D/224 configuration.")
        self.backend_id = SIGLIP2_BACKEND_ID
        self.model_name = model_name
        self.model_id = model_id
        self.revision = revision
        self.variant = variant
        self.native_dimension = native_dimension
        self.image_size = image_size
        self.fixres_image_shape = (3, image_size, image_size)
        self.capabilities: Mapping[str, bool] = freeze_capabilities(
            {
                "image_embedding": True,
                "text_embedding": True,
                "image_text_similarity": True,
                "patch_embedding": False,
                "text_similarity_map": False,
                "mrl": False,
                "selective_loading": False,
                "video_embedding": False,
                "multimodal_embedding": False,
                "instruction_aware": False,
                "variable_resolution": variant == "naflex",
                "sigmoid_logits": True,
            }
        )
        preprocessing_id = (
            f"siglip2-fixres-rgb{image_size}-bilinear-rescale1-255-normalize-0.5-v1"
            if variant == "fixres"
            else "siglip2-naflex-rgb-patch16-max256-aspect-mask-spatial-v1"
        )
        self.embedding_space_metadata = EmbeddingSpaceMetadata(
            backend_id=self.backend_id,
            model_id=model_id,
            model_revision=revision,
            embedding_strategy="siglip2-global-pooler-output-v1",
            normalization="l2-after-pooler-v1",
            preprocessing_id=preprocessing_id,
            index_prompt=None,
            query_prompt_name=None,
            query_prompt=None,
        )

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int:
        """Reject truncation because SigLIP 2 checkpoints do not use MRL."""
        if value is None:
            return self.native_dimension
        dimension = validate_positive_embedding_dimension(value, name=name)
        if dimension != self.native_dimension:
            raise ValueError(
                f"{name} must be {self.native_dimension}; SigLIP 2 does not "
                f"support MRL or truncation. Received {dimension}."
            )
        return dimension

    def capabilities_for(self, modalities: tuple[str, ...]) -> Mapping[str, bool]:
        """Expose only operations available for the loaded modalities."""
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
    ) -> Siglip2CLIP:
        if model_id != self.model_id or revision != self.revision:
            raise ValueError(
                "Only the registered SigLIP 2 checkpoint and pinned revision "
                "are supported."
            )
        if cache_dir is not None and not isinstance(cache_dir, str):
            raise TypeError("cache_dir must be a string or None.")
        if dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("SigLIP 2 supports only FP32 and BF16 precision.")
        if modalities not in (None, ("text", "image")):
            raise NotImplementedError(
                "SigLIP 2 selective modality loading is not supported; load both "
                "text and image modalities."
            )
        if not isinstance(output_dict, bool):
            raise TypeError("output_dict must be a bool.")
        if logit_scale_mode not in {"strict", "cosine"}:
            raise ValueError("logit_scale_mode must be 'strict' or 'cosine'.")

        try:
            backend = AutoModel.from_pretrained(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
                dtype=dtype,
                use_safetensors=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load {model_id!r} at pinned revision {revision}."
            ) from error
        if not isinstance(backend, nn.Module):
            raise TypeError("The Transformers loader did not return an nn.Module.")
        expected_model_type = "siglip" if self.variant == "fixres" else "siglip2"
        if getattr(getattr(backend, "config", None), "model_type", None) != (
            expected_model_type
        ):
            raise RuntimeError(
                f"The pinned {self.variant} checkpoint must use Transformers "
                f"model_type={expected_model_type!r}."
            )
        if not all(
            callable(getattr(backend, name, None))
            for name in ("get_image_features", "get_text_features")
        ):
            raise TypeError(
                "The loaded SigLIP 2 model must expose image and text feature APIs."
            )
        for name in ("logit_scale", "logit_bias"):
            parameter = getattr(backend, name, None)
            if not isinstance(parameter, torch.Tensor) or parameter.numel() != 1:
                raise RuntimeError(
                    f"The pinned SigLIP 2 model has no scalar learned {name}."
                )

        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load the processor for {model_id!r} at pinned revision "
                f"{revision}."
            ) from error
        if (
            getattr(processor, "image_processor", None) is None
            or getattr(processor, "tokenizer", None) is None
        ):
            raise RuntimeError(
                "The pinned SigLIP 2 processor must provide both image and text "
                "processing."
            )

        backend = backend.to(device=device, dtype=dtype).eval()
        return Siglip2CLIP(
            backend=backend,
            processor=processor,
            output_dict=output_dict,
            logit_scale_mode=logit_scale_mode,
            modalities=("text", "image"),
            inference_backend=self,
            cache_dir=cache_dir,
        ).eval()

    def create_tokenizer(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: str | None,
        prompt_name: str | None,
        context_length: int | None,
    ) -> Siglip2Tokenizer:
        """Load the pinned tokenizer with the official fixed-length text contract."""
        if model_id != self.model_id or revision != self.revision:
            raise ValueError("Only the registered SigLIP 2 tokenizer is supported.")
        if prompt_name is not None:
            raise ValueError("SigLIP 2 has no task prompts; prompt_name must be None.")
        if cache_dir is not None and not isinstance(cache_dir, str):
            raise TypeError("cache_dir must be a string or None.")
        resolved_context_length = (
            _MAX_CONTEXT_LENGTH if context_length is None else context_length
        )
        if (
            isinstance(resolved_context_length, bool)
            or not isinstance(resolved_context_length, int)
            or not 2 <= resolved_context_length <= _MAX_CONTEXT_LENGTH
        ):
            raise ValueError(
                f"context_length must be an integer between 2 and "
                f"{_MAX_CONTEXT_LENGTH}."
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load the tokenizer for {model_id!r} at pinned revision "
                f"{revision}."
            ) from error
        return Siglip2Tokenizer(tokenizer, context_length=resolved_context_length)

    def create_preprocess(self, model: Siglip2CLIP) -> Siglip2ImagePreprocess:
        if not isinstance(model, Siglip2CLIP):
            raise TypeError("The SigLIP 2 backend requires a Siglip2CLIP model.")
        return Siglip2ImagePreprocess(
            model.processor, variant=self.variant, image_size=self.image_size
        )

    def encode_image(
        self,
        model: Siglip2CLIP,
        image: ProcessedImageBatch,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor:
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool.")
        self.validate_embedding_dimension(truncate_dim, name="truncate_dim")
        inputs = self._prepare_image_inputs(model, image)
        batch_size = inputs["pixel_values"].shape[0]
        with torch.inference_mode():
            outputs = model.backend.get_image_features(**inputs)
        features = self._pooler_output(outputs, "image")
        return self._validate_features(
            features,
            batch_size=batch_size,
            modality="image",
            normalize=normalize,
        )

    def encode_text(
        self,
        model: Siglip2CLIP,
        text: torch.Tensor,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor:
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool.")
        self.validate_embedding_dimension(truncate_dim, name="truncate_dim")
        if not isinstance(text, torch.Tensor):
            raise TypeError("text must be a LongTensor[B,L] of token IDs.")
        if text.dtype != torch.long or text.ndim != 2 or text.shape[0] == 0:
            raise ValueError("text must be a non-empty LongTensor[B,L].")
        if text.shape[1] == 0 or text.shape[1] > _MAX_CONTEXT_LENGTH:
            raise ValueError(
                f"text sequence length must be between 1 and {_MAX_CONTEXT_LENGTH}."
            )
        tokenizer = getattr(model.processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
            raise TypeError("The loaded SigLIP 2 tokenizer has no valid pad token.")
        device, _ = self._model_device_dtype(model)
        input_ids = text.to(device=device)
        attention_mask = input_ids.ne(pad_token_id)
        if (attention_mask.sum(dim=1) == 0).any().item():
            raise ValueError("SigLIP 2 text input cannot contain an empty sequence.")
        with torch.inference_mode():
            outputs = model.backend.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        features = self._pooler_output(outputs, "text")
        return self._validate_features(
            features,
            batch_size=input_ids.shape[0],
            modality="text",
            normalize=normalize,
        )

    def get_logits(
        self,
        model: Siglip2CLIP,
        image: ProcessedImageBatch,
        text: torch.Tensor,
        *,
        truncate_dim: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return official SigLIP 2 logits using the learned scale and bias."""
        image_features = self.encode_image(
            model,
            image,
            normalize=True,
            truncate_dim=truncate_dim,
        )
        text_features = self.encode_text(
            model,
            text,
            normalize=True,
            truncate_dim=truncate_dim,
        )
        if image_features.device != text_features.device:
            raise ValueError("Image and text features must be on the same device.")
        if image_features.dtype != text_features.dtype:
            raise TypeError("Image and text features must have the same dtype.")
        if image_features.shape[1] != text_features.shape[1]:
            raise ValueError("Image and text feature dimensions must match.")

        logit_scale = model.backend.logit_scale.to(device=text_features.device)
        logit_bias = model.backend.logit_bias.to(device=text_features.device)
        logits_per_text = text_features @ image_features.T
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
        if not torch.isfinite(logits_per_text).all().item():
            raise RuntimeError("SigLIP 2 produced non-finite logits.")
        return logits_per_text.T, logits_per_text

    def _prepare_image_inputs(
        self,
        model: Siglip2CLIP,
        image: ProcessedImageBatch,
    ) -> dict[str, torch.Tensor]:
        if isinstance(image, Sequence) and not isinstance(
            image,
            (str, bytes, bytearray, Mapping),
        ):
            image = stack_processed_images(image, backend_name=self.model_name)

        device, dtype = self._model_device_dtype(model)
        if self.variant == "fixres":
            if isinstance(image, torch.Tensor):
                inputs: Mapping[str, object] = {"pixel_values": image}
            elif isinstance(image, Mapping):
                inputs = image
            else:
                raise TypeError(
                    "SigLIP 2 FixRes images must be a preprocessed Tensor or "
                    "pixel_values mapping."
                )
            if set(inputs) != {"pixel_values"}:
                raise ValueError("FixRes input must contain only pixel_values.")
            pixel_values = inputs.get("pixel_values")
            if not isinstance(pixel_values, torch.Tensor):
                raise TypeError("FixRes pixel_values must be a Tensor.")
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            if (
                pixel_values.ndim != 4
                or tuple(pixel_values.shape[1:]) != self.fixres_image_shape
                or pixel_values.shape[0] == 0
            ):
                raise ValueError(
                    "FixRes pixel_values must have shape "
                    f"[B,3,{self.image_size},{self.image_size}] or "
                    f"[3,{self.image_size},{self.image_size}]."
                )
            if not pixel_values.is_floating_point():
                raise TypeError("FixRes pixel_values must use a floating dtype.")
            if not torch.isfinite(pixel_values).all().item():
                raise ValueError("FixRes pixel_values must be finite.")
            return {"pixel_values": pixel_values.to(device=device, dtype=dtype)}

        if isinstance(image, torch.Tensor):
            raise TypeError(
                "SigLIP 2 NaFlex requires processor-aware image input; pass the "
                "mapping returned by the NaFlex preprocess function."
            )
        if not isinstance(image, Mapping):
            raise TypeError("SigLIP 2 NaFlex images must be a processor-aware mapping.")
        if set(image) != _NAFLEX_INPUT_KEYS:
            raise ValueError(
                "NaFlex input must contain pixel_values, pixel_attention_mask, "
                "and spatial_shapes."
            )
        pixel_values = image.get("pixel_values")
        pixel_attention_mask = image.get("pixel_attention_mask")
        spatial_shapes = image.get("spatial_shapes")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (pixel_values, pixel_attention_mask, spatial_shapes)
        ):
            raise TypeError("All NaFlex input values must be Tensors.")
        if (
            not isinstance(pixel_values, torch.Tensor)
            or not isinstance(pixel_attention_mask, torch.Tensor)
            or not isinstance(spatial_shapes, torch.Tensor)
        ):
            raise TypeError("All NaFlex input values must be Tensors.")

        unbatched = pixel_values.ndim == 2
        if unbatched:
            if pixel_attention_mask.ndim != 1 or spatial_shapes.ndim != 1:
                raise ValueError(
                    "Unbatched NaFlex input requires a 1D mask and spatial_shapes."
                )
            pixel_values = pixel_values.unsqueeze(0)
            pixel_attention_mask = pixel_attention_mask.unsqueeze(0)
            spatial_shapes = spatial_shapes.unsqueeze(0)
        if (
            pixel_values.ndim != 3
            or pixel_attention_mask.ndim != 2
            or spatial_shapes.ndim != 2
            or pixel_values.shape[0] == 0
            or pixel_values.shape[-1] != _SIGLIP2_NATIVE_DIMENSION
            or pixel_attention_mask.shape != pixel_values.shape[:2]
            or spatial_shapes.shape != (pixel_values.shape[0], 2)
        ):
            raise ValueError(
                "NaFlex input shapes must be pixel_values[B,N,768], "
                "pixel_attention_mask[B,N], and spatial_shapes[B,2]."
            )
        if pixel_values.shape[1] == 0 or not pixel_values.is_floating_point():
            raise TypeError("NaFlex pixel_values must be non-empty floating Tensors.")
        image_processor = getattr(model.processor, "image_processor", None)
        max_num_patches = getattr(image_processor, "max_num_patches", None)
        if (
            isinstance(max_num_patches, bool)
            or not isinstance(max_num_patches, int)
            or max_num_patches <= 0
        ):
            raise RuntimeError(
                "The pinned NaFlex image processor has no valid max_num_patches."
            )
        if pixel_values.shape[1] > max_num_patches:
            raise ValueError(
                "NaFlex pixel_values exceed the processor's "
                f"max_num_patches={max_num_patches}."
            )
        if not torch.isfinite(pixel_values).all().item():
            raise ValueError("NaFlex pixel_values must be finite.")
        if pixel_attention_mask.dtype not in {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("NaFlex pixel_attention_mask must contain integer values.")
        if spatial_shapes.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("NaFlex spatial_shapes must contain integer values.")
        if not ((pixel_attention_mask == 0) | (pixel_attention_mask == 1)).all().item():
            raise ValueError("NaFlex pixel_attention_mask values must be 0 or 1.")
        if (spatial_shapes <= 0).any().item():
            raise ValueError("NaFlex spatial_shapes values must be positive.")
        valid_patches = spatial_shapes.to(dtype=torch.long).prod(dim=1)
        mask_counts = pixel_attention_mask.to(dtype=torch.bool).sum(dim=1)
        if (valid_patches > pixel_values.shape[1]).any().item() or not torch.equal(
            valid_patches, mask_counts
        ):
            raise ValueError(
                "NaFlex spatial_shapes and pixel_attention_mask do not describe "
                "the same valid patches."
            )
        return {
            "pixel_values": pixel_values.to(device=device, dtype=dtype),
            "pixel_attention_mask": pixel_attention_mask.to(device=device),
            "spatial_shapes": spatial_shapes.to(device=device),
        }

    @staticmethod
    def _model_device_dtype(model: Siglip2CLIP) -> tuple[torch.device, torch.dtype]:
        try:
            parameter = next(model.backend.parameters())
        except StopIteration as error:
            raise RuntimeError("The SigLIP 2 model has no parameters.") from error
        return parameter.device, parameter.dtype

    @staticmethod
    def _pooler_output(outputs: Any, modality: str) -> torch.Tensor:
        features = getattr(outputs, "pooler_output", None)
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"The official SigLIP 2 {modality} feature API returned no "
                "pooler_output Tensor."
            )
        return features

    def _validate_features(
        self,
        features: torch.Tensor,
        *,
        batch_size: int,
        modality: str,
        normalize: bool,
    ) -> torch.Tensor:
        expected_shape = (batch_size, self.native_dimension)
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"The official SigLIP 2 {modality} feature API returned shape "
                f"{tuple(features.shape)}; expected {expected_shape}."
            )
        if not torch.isfinite(features).all().item():
            raise RuntimeError(
                f"The official SigLIP 2 {modality} feature API returned "
                "non-finite values."
            )
        if normalize:
            norms = torch.linalg.vector_norm(features.float(), dim=-1)
            if not torch.isfinite(norms).all().item() or (norms <= 0).any().item():
                raise RuntimeError(
                    f"The official SigLIP 2 {modality} features cannot be normalized."
                )
            features = F.normalize(features, p=2, dim=-1)
        return features
