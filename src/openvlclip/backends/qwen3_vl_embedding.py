"""Official processor-aware inference for the pinned Qwen3-VL-Embedding models."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from os import PathLike, fspath
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoProcessor

from openvlclip._dimensions import validate_positive_embedding_dimension
from openvlclip.backends.base import (
    MultimodalInput,
    ProcessedImageBatch,
    VideoInput,
)
from openvlclip.capabilities import capabilities_for_modalities, freeze_capabilities
from openvlclip.embedding_space import EmbeddingSpaceMetadata

QWEN3VL_BACKEND_ID = "qwen3-vl-embedding"

QWEN3VL_2B_MODEL_NAME = "qwen3-vl-embedding-2b"
QWEN3VL_2B_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
QWEN3VL_2B_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
QWEN3VL_2B_DIMENSION = 2048


_DEFAULT_INSTRUCTION = "Represent the user's input."
_MAX_CONTEXT_LENGTH = 8192
_MIN_IMAGE_PIXELS = 4096
_MAX_IMAGE_PIXELS = 1_843_200
_IMAGE_PATCH_SIZE = 16
_DEFAULT_FPS = 1.0
_DEFAULT_MAX_FRAMES = 64
_MAX_TOTAL_VIDEO_PIXELS = 7_864_320
_MIN_MRL_DIMENSION = 64
_MULTIMODAL_KEYS = frozenset(
    {"text", "image", "video", "instruction", "fps", "max_frames"}
)


class Qwen3VLImagePreprocess:
    """Prepare RGB images while leaving model-specific resizing to the processor."""

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("preprocess expects a PIL.Image.Image.")
        return image.convert("RGB")


class Qwen3VLCLIP(nn.Module):
    """Expose the Qwen embedding backend through the common inference API."""

    def __init__(
        self,
        backend: nn.Module,
        processor: Any,
        inference_backend: Qwen3VLBackend,
        *,
        output_dict: bool,
        logit_scale_mode: str,
        cache_dir: str | None,
    ) -> None:
        super().__init__()
        if not isinstance(backend, nn.Module):
            raise TypeError("backend must be a torch.nn.Module.")
        if not isinstance(output_dict, bool):
            raise TypeError("output_dict must be a bool.")
        if logit_scale_mode not in {"strict", "cosine"}:
            raise ValueError("logit_scale_mode must be 'strict' or 'cosine'.")
        if cache_dir is not None and not isinstance(cache_dir, str):
            raise TypeError("cache_dir must be a string or None.")
        self.backend = backend
        object.__setattr__(self, "_processor", processor)
        object.__setattr__(self, "_inference_backend", inference_backend)
        self.output_dict = output_dict
        self.logit_scale_mode = logit_scale_mode
        self.cache_dir = cache_dir
        self.modalities = ("text", "image", "video")

    @property
    def processor(self) -> Any:
        """Return the processor without registering it as a PyTorch submodule."""
        return self._processor

    @property
    def native_video_max_frames(self) -> int:
        """Return the maximum number of frames supported by this checkpoint."""
        return _DEFAULT_MAX_FRAMES

    @property
    def capabilities(self) -> Mapping[str, bool]:
        """Return the full set of supported Qwen embedding operations."""
        return self._inference_backend.capabilities_for(self.modalities)

    @property
    def embedding_space_metadata(self) -> EmbeddingSpaceMetadata:
        """Return the pinned checkpoint's embedding-space identity."""
        return self._inference_backend.embedding_space_metadata

    @property
    def native_video_processing_metadata(self) -> Mapping[str, object]:
        """Return the exact processor version and file-video pixel budget."""
        try:
            from qwen_vl_utils import vision_process

            reader_backend = vision_process.get_video_reader_backend()
            reader_package = {
                "torchcodec": "torchcodec",
                "decord": "decord",
                "torchvision": "torchvision",
            }.get(reader_backend)
            if reader_package is None:
                raise RuntimeError(
                    f"Unsupported Qwen video reader backend: {reader_backend}."
                )
            processor_version = (
                f"transformers=={version('transformers')};"
                f"qwen-vl-utils=={version('qwen-vl-utils')};"
                f"video-reader={reader_backend}=={version(reader_package)}"
            )
        except PackageNotFoundError as error:
            raise RuntimeError(
                "Native video indexing requires the optional 'qwen3vl' extra "
                "and its selected video reader; "
                'install it with `pip install "openvlclip[qwen3vl]"`.'
            ) from error
        except ImportError as error:
            raise RuntimeError(
                "Native video indexing requires the optional 'qwen3vl' extra; "
                'install it with `pip install "openvlclip[qwen3vl]"`.'
            ) from error
        return {
            "processor_version": processor_version,
            "total_pixels": _default_file_video_total_pixels(),
        }

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int:
        """Validate the Qwen MRL dimension range for this checkpoint."""
        return self._inference_backend.validate_embedding_dimension(value, name=name)

    def to(self, *args: Any, **kwargs: Any) -> Qwen3VLCLIP:
        """Move the model while rejecting unsupported FP16 conversion."""
        requested_dtype = kwargs.get("dtype")
        for argument in args:
            if isinstance(argument, torch.dtype):
                requested_dtype = argument
            elif isinstance(argument, torch.Tensor):
                requested_dtype = argument.dtype
        if requested_dtype == torch.float16:
            raise ValueError("FP16 is not supported; use FP32 or BF16.")
        return super().to(*args, **kwargs)

    def half(self) -> Qwen3VLCLIP:
        """Reject the PyTorch convenience conversion to unsupported FP16."""
        raise ValueError("FP16 is not supported; use FP32 or BF16.")

    def encode_image(
        self,
        image: ProcessedImageBatch,
        normalize: bool = False,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
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
        return self._inference_backend.encode_texts(
            self,
            texts,
            instruction=instruction,
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
        return self._inference_backend.encode_multimodal(
            self,
            items,
            normalize=normalize,
            truncate_dim=truncate_dim,
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
        return self._inference_backend.encode_video(
            self,
            video,
            normalize=normalize,
            truncate_dim=truncate_dim,
            fps=fps,
            max_frames=max_frames,
        )

    def encode_patches(
        self,
        image: ProcessedImageBatch,
        normalize: bool = False,
    ) -> torch.Tensor:
        del image, normalize
        raise NotImplementedError(
            "encode_patches() is not supported by the Qwen3-VL-Embedding backend."
        )

    def similarity_map(
        self,
        image: ProcessedImageBatch,
        text: str | Sequence[str],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        del image, text, output_size
        raise NotImplementedError(
            "similarity_map() is not supported by the Qwen3-VL-Embedding backend."
        )

    def _normalize_text_features(
        self,
        text: torch.Tensor,
        *,
        dimension: int,
    ) -> torch.Tensor:
        if not isinstance(text, torch.Tensor):
            raise TypeError(
                "text must be a Tensor of previously encoded text features."
            )
        if not text.is_floating_point():
            raise NotImplementedError(
                "Qwen token IDs do not preserve the official processor attention mask "
                "or chat template; call encode_texts() and pass its features instead."
            )
        if text.ndim != 2 or text.shape[0] == 0:
            raise ValueError("text features must be a non-empty Tensor[B,D].")
        native_dimension = self._inference_backend.native_dimension
        if text.shape[1] < dimension or text.shape[1] > native_dimension:
            raise ValueError(
                "text feature width must be at least the selected truncate_dim and "
                f"no greater than the native dimension {native_dimension}."
            )
        if not torch.isfinite(text).all().item():
            raise ValueError("text features must be finite.")
        return F.normalize(text[:, :dimension], p=2, dim=-1)

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
        """Return cosine-ready image features and optional precomputed text features."""
        if image is None and text is None:
            raise ValueError("At least one of image or text must be provided.")
        if self.logit_scale_mode != "cosine":
            raise NotImplementedError(
                "forward() requires logit_scale_mode='cosine'; Qwen3-VL-Embedding "
                "has no trained OpenCLIP logit_scale."
            )

        dimension = self.validate_embedding_dimension(
            truncate_dim,
            name="truncate_dim",
        )
        image_features = (
            self.encode_image(image, normalize=True, truncate_dim=truncate_dim)
            if image is not None
            else None
        )
        text_features = (
            self._normalize_text_features(text, dimension=dimension)
            if text is not None
            else None
        )
        _validate_feature_pair(image_features, text_features, dimension=dimension)
        feature_device = (
            image_features.device
            if image_features is not None
            else text_features.device
        )
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
        """Compute cosine logits from images and encode_texts() feature tensors."""
        if self.logit_scale_mode != "cosine":
            raise NotImplementedError(
                "get_logits() requires logit_scale_mode='cosine'; Qwen3-VL-Embedding "
                "has no trained OpenCLIP logit_scale."
            )
        dimension = self.validate_embedding_dimension(
            truncate_dim,
            name="truncate_dim",
        )
        image_features = self.encode_image(
            image,
            normalize=True,
            truncate_dim=truncate_dim,
        )
        text_features = self._normalize_text_features(text, dimension=dimension)
        _validate_feature_pair(image_features, text_features, dimension=dimension)
        if image_features.device != text_features.device:
            raise ValueError(
                "Encoded image and text features must be on the same device."
            )
        if image_features.dtype != text_features.dtype:
            raise TypeError("Encoded image and text features must have the same dtype.")

        if (
            image_features.device.type == "cpu"
            and image_features.dtype == torch.bfloat16
        ):
            image_logits = image_features.float() @ text_features.float().T
            image_logits = image_logits.to(dtype=image_features.dtype)
        else:
            image_logits = image_features @ text_features.T
        if not torch.isfinite(image_logits).all().item():
            raise RuntimeError("Cosine similarity produced non-finite values.")
        return image_logits, image_logits.T


class Qwen3VLBackend:
    """Load a fixed Qwen3-VL-Embedding checkpoint without remote model code."""

    backend_id = QWEN3VL_BACKEND_ID

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        revision: str,
        native_dimension: int,
    ) -> None:
        self.model_name = model_name
        self.model_id = model_id
        self.revision = revision
        self.native_dimension = native_dimension
        self.capabilities: Mapping[str, bool] = freeze_capabilities(
            {
                "image_embedding": True,
                "text_embedding": True,
                "image_text_similarity": True,
                "patch_embedding": False,
                "text_similarity_map": False,
                "mrl": True,
                "selective_loading": False,
                "video_embedding": True,
                "multimodal_embedding": True,
                "instruction_aware": True,
                "variable_resolution": True,
                "sigmoid_logits": False,
            }
        )
        self.embedding_space_metadata = EmbeddingSpaceMetadata(
            backend_id=self.backend_id,
            model_id=model_id,
            model_revision=revision,
            embedding_strategy="qwen3-vl-chat-template-last-valid-token-pooling-v1",
            normalization="none-or-truncate-then-l2-v1",
            preprocessing_id=(
                "qwen3-vl-processor-len8192-pixel4096-1843200-fps1-frames64-v1"
            ),
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
        """Accept the model-card MRL range and reject non-native truncation."""
        if value is None:
            return self.native_dimension
        dimension = validate_positive_embedding_dimension(value, name=name)
        if not _MIN_MRL_DIMENSION <= dimension <= self.native_dimension:
            raise ValueError(
                f"{name} must be between {_MIN_MRL_DIMENSION} and "
                f"{self.native_dimension} for {self.model_name}; received {dimension}."
            )
        return dimension

    def capabilities_for(self, modalities: tuple[str, ...]) -> Mapping[str, bool]:
        """Return capabilities filtered by the loaded modalities."""
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
    ) -> Qwen3VLCLIP:
        if model_id != self.model_id or revision != self.revision:
            raise ValueError(
                "Only the registered Qwen3-VL-Embedding checkpoint and pinned "
                "revision are supported."
            )
        if cache_dir is not None and not isinstance(cache_dir, str):
            raise TypeError("cache_dir must be a string or None.")
        if dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("Qwen3-VL-Embedding supports only FP32 and BF16.")
        if modalities not in (None, ("text", "image")):
            raise NotImplementedError(
                "Qwen3-VL-Embedding does not support selective modality loading."
            )
        if not isinstance(output_dict, bool):
            raise TypeError("output_dict must be a bool.")
        if logit_scale_mode not in {"strict", "cosine"}:
            raise ValueError("logit_scale_mode must be 'strict' or 'cosine'.")

        try:
            loaded = AutoModel.from_pretrained(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=dtype,
                output_loading_info=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load {model_id!r} at pinned revision {revision} "
                "without remote code."
            ) from error
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise RuntimeError("Transformers did not return model loading details.")
        model, loading_info = loaded
        if not isinstance(model, nn.Module) or not isinstance(loading_info, Mapping):
            raise TypeError("Transformers returned an invalid Qwen model load result.")
        _validate_loading_info(loading_info, model_id=model_id)

        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", None)
        observed_dimension = getattr(text_config, "hidden_size", None)
        if (
            getattr(config, "model_type", None) != "qwen3_vl"
            or observed_dimension != self.native_dimension
        ):
            raise RuntimeError(
                f"The pinned Qwen checkpoint must expose a qwen3_vl text hidden "
                f"size of {self.native_dimension}; received {observed_dimension!r}."
            )

        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
                trust_remote_code=False,
                padding_side="right",
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load the processor for {model_id!r} at pinned "
                f"revision {revision}."
            ) from error
        tokenizer = getattr(processor, "tokenizer", None)
        if (
            not callable(getattr(processor, "apply_chat_template", None))
            or tokenizer is None
            or getattr(tokenizer, "padding_side", None) != "right"
        ):
            raise RuntimeError(
                "The pinned Qwen processor must expose apply_chat_template() and "
                "a right-padding tokenizer."
            )

        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            required_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
            )
            if required_bytes > free_bytes:
                raise RuntimeError(
                    f"{self.model_name} parameters require at least "
                    f"{required_bytes / 2**30:.2f} GiB, but {device} has only "
                    f"{free_bytes / 2**30:.2f} GiB free. Use a larger CUDA device."
                )
        try:
            model = model.to(device=device, dtype=dtype).eval()
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                f"Could not move {self.model_name} to {device}; free memory is "
                "insufficient for the requested precision."
            ) from error
        return Qwen3VLCLIP(
            model,
            processor,
            self,
            output_dict=output_dict,
            logit_scale_mode=logit_scale_mode,
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
    ) -> NoReturn:
        del cache_dir, prompt_name, context_length
        if model_id != self.model_id or revision != self.revision:
            raise ValueError("Only the registered Qwen tokenizer is supported.")
        raise NotImplementedError(
            "Qwen token IDs do not preserve the official processor attention mask "
            "or chat template; use model.encode_texts() instead of get_tokenizer()."
        )

    def create_preprocess(self, model: Qwen3VLCLIP) -> Qwen3VLImagePreprocess:
        if not isinstance(model, Qwen3VLCLIP):
            raise TypeError("The Qwen3-VL backend requires a Qwen3VLCLIP model.")
        return Qwen3VLImagePreprocess()

    def encode_image(
        self,
        model: Qwen3VLCLIP,
        image: ProcessedImageBatch,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor:
        if isinstance(image, Mapping):
            images: list[object] = [image]
        elif isinstance(image, Image.Image):
            images = [image]
        elif isinstance(image, Sequence) and not isinstance(
            image, (str, bytes, bytearray)
        ):
            images = list(image)
        else:
            raise TypeError(
                "Qwen encode_image() expects a PIL image or a non-empty sequence "
                "of PIL images returned by create_model_and_transforms()."
            )
        if not images:
            raise ValueError("Qwen encode_image() cannot encode an empty batch.")

        items: list[dict[str, object]] = []
        for image_item in images:
            if isinstance(image_item, Mapping):
                if set(image_item) != {"image"}:
                    raise TypeError(
                        "Qwen processor mappings must contain only an 'image' entry."
                    )
                image_item = image_item["image"]
            if not isinstance(image_item, Image.Image):
                raise TypeError(
                    "Qwen encode_image() requires PIL images; use the returned "
                    "preprocess function rather than normalized image tensors."
                )
            items.append({"image": image_item})
        return self.encode_multimodal(
            model,
            items,
            normalize=normalize,
            truncate_dim=truncate_dim,
        )

    def encode_text(
        self,
        model: Qwen3VLCLIP,
        text: torch.Tensor,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor:
        del model, text, normalize, truncate_dim
        raise NotImplementedError(
            "Qwen token IDs do not preserve the official processor attention mask "
            "or chat template; use model.encode_texts() instead."
        )

    def encode_texts(
        self,
        model: Qwen3VLCLIP,
        texts: str | Sequence[str],
        *,
        instruction: str | None,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor:
        text_batch = _normalize_texts(texts)
        if not text_batch:
            raise ValueError("texts must contain at least one string.")
        items = [{"text": text, "instruction": instruction} for text in text_batch]
        return self.encode_multimodal(
            model,
            items,
            normalize=normalize,
            truncate_dim=truncate_dim,
        )

    def encode_multimodal(
        self,
        model: Qwen3VLCLIP,
        items: MultimodalInput,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor:
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool.")
        dimension = self.validate_embedding_dimension(
            truncate_dim,
            name="truncate_dim",
        )
        if isinstance(items, Mapping):
            input_items = [items]
        elif isinstance(items, Sequence) and not isinstance(
            items, (str, bytes, bytearray)
        ):
            input_items = list(items)
        else:
            raise TypeError(
                "items must be a mapping or a sequence of multimodal mappings."
            )
        if not input_items:
            raise ValueError("encode_multimodal() cannot encode an empty batch.")

        conversations: list[list[dict[str, object]]] = []
        for index, item in enumerate(input_items):
            if not isinstance(item, Mapping):
                raise TypeError(f"Multimodal item at index {index} must be a mapping.")
            conversations.append(_format_conversation(item, index=index))

        processor_inputs = _prepare_processor_inputs(model.processor, conversations)
        model_device = _model_device(model.backend)
        model_inputs = {
            key: value.to(model_device) if isinstance(value, torch.Tensor) else value
            for key, value in processor_inputs.items()
        }
        attention_mask = model_inputs.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError(
                "The Qwen processor did not return an attention_mask Tensor."
            )
        with torch.inference_mode():
            outputs = model.backend(**model_inputs, use_cache=False)
        hidden_state = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden_state, torch.Tensor) or hidden_state.ndim != 3:
            raise RuntimeError(
                "The Qwen backbone did not return last_hidden_state[B,L,D]."
            )
        if hidden_state.shape[0] != len(conversations):
            raise RuntimeError("The Qwen backbone changed the input batch size.")
        if hidden_state.shape[2] != self.native_dimension:
            raise RuntimeError(
                f"The Qwen backbone returned dimension {hidden_state.shape[2]}; "
                f"expected {self.native_dimension}."
            )

        pooled = _pool_last_valid_token(hidden_state, attention_mask)
        features = pooled[:, :dimension]
        if normalize:
            features = F.normalize(features, p=2, dim=-1)
        if features.shape != (len(conversations), dimension):
            raise RuntimeError("Qwen returned an invalid embedding shape.")
        if not torch.isfinite(features).all().item():
            raise RuntimeError("Qwen returned non-finite embeddings.")
        return features

    def encode_video(
        self,
        model: Qwen3VLCLIP,
        video: VideoInput,
        *,
        normalize: bool,
        truncate_dim: int | None,
        fps: float | None,
        max_frames: int | None,
    ) -> torch.Tensor:
        features = self.encode_multimodal(
            model,
            {"video": video, "fps": fps, "max_frames": max_frames},
            normalize=normalize,
            truncate_dim=truncate_dim,
        )
        if features.shape[0] != 1:
            raise RuntimeError("encode_video() must return exactly one embedding.")
        return features


def _validate_loading_info(
    loading_info: Mapping[str, object],
    *,
    model_id: str,
) -> None:
    issue_names = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    issues: dict[str, list[str]] = {}
    for name in issue_names:
        values = loading_info.get(name, ())
        if not isinstance(values, Sequence) and not isinstance(values, set):
            raise TypeError(f"Transformers returned invalid {name} details.")
        if values:
            issues[name] = sorted(str(value) for value in values)
    if issues:
        details = "; ".join(
            f"{name}={values[:8]}" for name, values in sorted(issues.items())
        )
        raise RuntimeError(
            f"Checkpoint {model_id!r} did not load exactly into Qwen3VLModel: "
            f"{details}."
        )


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise RuntimeError("The Qwen backbone has no parameters.") from error


def _normalize_texts(texts: str | Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    if isinstance(texts, (bytes, bytearray)) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a string or a sequence of strings.")
    values = list(texts)
    if any(not isinstance(text, str) for text in values):
        raise TypeError("texts must contain only strings.")
    return values


def _normalize_instruction(instruction: object, *, index: int) -> str:
    if instruction is None:
        return _DEFAULT_INSTRUCTION
    if not isinstance(instruction, str):
        raise TypeError(f"Instruction at input index {index} must be a string or None.")
    instruction = instruction.strip()
    if not instruction:
        return _DEFAULT_INSTRUCTION
    if not unicodedata.category(instruction[-1]).startswith("P"):
        instruction = f"{instruction}."
    return instruction


def _normalize_image_reference(value: object, *, index: int) -> Image.Image | str:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, PathLike)):
        return _local_file_uri(value, name=f"image at input index {index}")
    raise TypeError(
        f"Image at input index {index} must be a PIL image or local file path."
    )


def _normalize_images(value: object, *, index: int) -> list[Image.Image | str]:
    if value is None:
        return []
    if isinstance(value, (Image.Image, str, PathLike)):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
    else:
        raise TypeError(
            f"Images at input index {index} must be a PIL image, local path, "
            "or a sequence of them."
        )
    return [_normalize_image_reference(image, index=index) for image in values]


def _normalize_video(
    value: object,
    *,
    index: int,
    fps: object,
    max_frames: object,
) -> dict[str, object]:
    resolved_max_frames = _validate_max_frames(max_frames)
    if isinstance(value, (str, PathLike)):
        return {
            "type": "video",
            "video": str(_local_file_path(value, name=f"video at input index {index}")),
            "fps": _validate_fps(fps),
            "max_frames": resolved_max_frames,
            "total_pixels": _default_file_video_total_pixels(),
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if fps is not None:
            raise ValueError(
                "fps applies only to video file paths; frame sequences have no "
                "source timestamp metadata."
            )
        frames = list(value)
        if not frames:
            raise ValueError(f"Video frame sequence at input index {index} is empty.")
        if len(frames) > resolved_max_frames:
            indices = np.linspace(
                0,
                len(frames) - 1,
                resolved_max_frames,
                dtype=int,
            )
            frames = [frames[int(frame_index)] for frame_index in indices]
        return {
            "type": "video",
            "video": [
                _normalize_image_reference(frame, index=index) for frame in frames
            ],
            "total_pixels": _MAX_TOTAL_VIDEO_PIXELS,
        }
    raise TypeError(
        f"Video at input index {index} must be a local file path or a "
        "sequence of PIL frames/local frame paths."
    )


def _default_file_video_total_pixels() -> float:
    try:
        from qwen_vl_utils import vision_process
    except ImportError as error:
        raise RuntimeError(
            "Qwen video inference requires the optional 'qwen3vl' extra; "
            'install it with `pip install "openvlclip[qwen3vl]"`.'
        ) from error
    try:
        model_sequence_length = vision_process.MODEL_SEQ_LEN
        spatial_merge_size = vision_process.SPATIAL_MERGE_SIZE
    except AttributeError as error:
        raise RuntimeError(
            "The installed qwen-vl-utils package does not expose its video "
            "pixel-budget settings."
        ) from error
    if (
        type(model_sequence_length) is not int
        or model_sequence_length <= 0
        or type(spatial_merge_size) is not int
        or spatial_merge_size <= 0
    ):
        raise RuntimeError(
            "The installed qwen-vl-utils package exposes invalid video "
            "pixel-budget settings."
        )
    image_factor = _IMAGE_PATCH_SIZE * spatial_merge_size
    total_pixels = float(model_sequence_length * image_factor**2 * 0.9)
    if not isfinite(total_pixels) or total_pixels <= 0:
        raise RuntimeError(
            "The installed qwen-vl-utils package exposes an invalid video "
            "pixel budget."
        )
    return total_pixels


def _validate_fps(value: object) -> float:
    if value is None:
        return _DEFAULT_FPS
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("fps must be a positive finite number or None.")
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError("fps must be a positive finite number.")
    return resolved


def _validate_max_frames(value: object) -> int:
    if value is None:
        return _DEFAULT_MAX_FRAMES
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_frames must be a positive integer or None.")
    if not 1 <= value <= _DEFAULT_MAX_FRAMES:
        raise ValueError(f"max_frames must be between 1 and {_DEFAULT_MAX_FRAMES}.")
    return value


def _local_file_uri(value: str | PathLike[str], *, name: str) -> str:
    return _local_file_path(value, name=name).as_uri()


def _local_file_path(value: str | PathLike[str], *, name: str) -> Path:
    path_value = fspath(value)
    if not isinstance(path_value, str):
        raise TypeError(f"{name} path must be a string path.")
    if "://" in path_value:
        raise ValueError(f"Remote URLs and URI paths are not supported for {name}.")
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"The {name} file does not exist: {path}.")
    return path.resolve()


def _format_conversation(
    item: Mapping[str, object],
    *,
    index: int,
) -> list[dict[str, object]]:
    unknown_keys = set(item).difference(_MULTIMODAL_KEYS)
    if unknown_keys:
        raise TypeError(
            f"Unsupported multimodal key(s) at input index {index}: "
            f"{', '.join(sorted(str(key) for key in unknown_keys))}."
        )
    texts = _normalize_texts(item.get("text")) if item.get("text") is not None else []
    instruction = _normalize_instruction(item.get("instruction"), index=index)
    content: list[dict[str, object]] = []

    video = item.get("video")
    if video is not None:
        content.append(
            _normalize_video(
                video,
                index=index,
                fps=item.get("fps"),
                max_frames=item.get("max_frames"),
            )
        )
    elif item.get("fps") is not None or item.get("max_frames") is not None:
        raise ValueError(
            f"fps and max_frames at input index {index} require a video input."
        )

    for image in _normalize_images(item.get("image"), index=index):
        content.append(
            {
                "type": "image",
                "image": image,
                "min_pixels": _MIN_IMAGE_PIXELS,
                "max_pixels": _MAX_IMAGE_PIXELS,
            }
        )
    content.extend({"type": "text", "text": text} for text in texts)
    if not content:
        raise ValueError(
            f"Multimodal item at input index {index} must contain text, image, "
            "or video input."
        )

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {"role": "user", "content": content},
    ]


def _process_vision_info(
    conversations: list[list[dict[str, object]]],
) -> tuple[object, object, object]:
    try:
        from qwen_vl_utils.vision_process import process_vision_info
    except ImportError as error:
        raise RuntimeError(
            "Qwen image/video inference requires the optional 'qwen3vl' extra; "
            'install it with `pip install "openvlclip[qwen3vl]"`.'
        ) from error
    return process_vision_info(
        conversations,
        image_patch_size=_IMAGE_PATCH_SIZE,
        return_video_metadata=True,
        return_video_kwargs=True,
    )


def _prepare_processor_inputs(
    processor: Any,
    conversations: list[list[dict[str, object]]],
) -> Mapping[str, object]:
    text = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=False,
    )
    has_vision = any(
        element.get("type") in {"image", "video"}
        for conversation in conversations
        for message in conversation
        for element in message.get("content", [])
        if isinstance(element, Mapping)
    )
    if has_vision:
        images, video_inputs, video_kwargs = _process_vision_info(conversations)
    else:
        images, video_inputs, video_kwargs = None, None, {}

    if video_inputs is None:
        videos = None
        video_metadata = None
    else:
        if not isinstance(video_inputs, Sequence):
            raise TypeError("Qwen video preprocessing must return a sequence.")
        videos = []
        metadata_values = []
        for index, video_input in enumerate(video_inputs):
            if not isinstance(video_input, Sequence) or len(video_input) != 2:
                raise TypeError(
                    f"Qwen video preprocessing result at index {index} must "
                    "contain frames and metadata."
                )
            videos.append(video_input[0])
            metadata_values.append(video_input[1])
        video_metadata = metadata_values

    if not isinstance(video_kwargs, Mapping):
        raise TypeError("Qwen video preprocessing must return keyword arguments.")
    processor_inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        truncation=True,
        max_length=_MAX_CONTEXT_LENGTH,
        padding=True,
        do_resize=False,
        return_tensors="pt",
        **dict(video_kwargs),
    )
    if not isinstance(processor_inputs, Mapping):
        raise TypeError("The Qwen processor must return a mapping of model inputs.")
    return processor_inputs


def _pool_last_valid_token(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(
        hidden_state.shape[:2]
    ):
        raise RuntimeError(
            "The Qwen attention mask must match the batch and token dimensions."
        )
    if not torch.logical_or(attention_mask == 0, attention_mask == 1).all().item():
        raise ValueError("The Qwen attention mask must contain only zeros and ones.")
    valid_mask = attention_mask.to(dtype=torch.bool)
    if not valid_mask.any(dim=1).all().item():
        raise ValueError("Every Qwen input must contain at least one attended token.")
    last_positions = (
        valid_mask.shape[1]
        - 1
        - valid_mask.flip(dims=[1]).to(dtype=torch.long).argmax(dim=1)
    )
    rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    pooled = hidden_state[rows, last_positions]
    if not torch.isfinite(pooled).all().item():
        raise RuntimeError("Qwen returned non-finite pooled features.")
    return pooled


def _validate_feature_pair(
    image_features: torch.Tensor | None,
    text_features: torch.Tensor | None,
    *,
    dimension: int,
) -> None:
    for features in (image_features, text_features):
        if features is not None and (
            features.ndim != 2 or features.shape[1] != dimension
        ):
            raise RuntimeError(f"Encoded features must have {dimension} dimensions.")
    if (
        image_features is not None
        and text_features is not None
        and image_features.dtype != text_features.dtype
    ):
        raise TypeError("Encoded image and text features must have the same dtype.")
    if (
        image_features is not None
        and text_features is not None
        and image_features.device != text_features.device
    ):
        raise ValueError("Encoded image and text features must be on the same device.")
