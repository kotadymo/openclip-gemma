"""Model factories for the supported EmbeddingGemma 2 checkpoint."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from openclip_gemma.constants import MODEL_ID, MODEL_NAME, MODEL_REVISION
from openclip_gemma.model import EmbeddingGemmaCLIP
from openclip_gemma.transforms import EmbeddingGemmaPreprocess

PRETRAINED_OFFICIAL = "official"


def _validate_modalities(
    modalities: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if modalities is None:
        return None
    if isinstance(modalities, (str, bytes)) or not isinstance(modalities, Sequence):
        raise TypeError(
            "modalities must be a sequence of 'text' and optionally 'image'."
        )
    if any(not isinstance(modality, str) for modality in modalities):
        raise TypeError("modalities must contain only strings.")
    if (
        "text" not in modalities
        or len(set(modalities)) != len(modalities)
        or any(modality not in {"text", "image"} for modality in modalities)
    ):
        raise ValueError("modalities must include 'text' and may include 'image'.")
    return tuple(modality for modality in ("text", "image") if modality in modalities)


def _validate_factory_options(
    model_name: str,
    pretrained: str | None,
    precision: str,
    device: str | torch.device,
    cache_dir: str | None,
    output_dict: bool,
    logit_scale_mode: str,
    modalities: Sequence[str] | None,
    kwargs: dict[str, Any],
) -> tuple[torch.device, torch.dtype, tuple[str, ...] | None]:
    if not isinstance(output_dict, bool):
        raise TypeError("output_dict must be a bool.")
    if not isinstance(logit_scale_mode, str):
        raise TypeError("logit_scale_mode must be a string.")
    if logit_scale_mode not in {"strict", "cosine"}:
        raise ValueError("logit_scale_mode must be 'strict' or 'cosine'.")
    validated_modalities = _validate_modalities(modalities)
    if cache_dir is not None and not isinstance(cache_dir, str):
        raise TypeError("cache_dir must be a string or None.")
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported keyword argument(s): {names}.")

    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string.")
    normalized_model_name = model_name.strip().lower().replace("_", "-")
    if normalized_model_name != MODEL_NAME:
        raise ValueError(
            f"Unsupported model_name {model_name!r}; only {MODEL_NAME!r} is supported."
        )

    if pretrained is None:
        raise ValueError(
            "pretrained=None is not supported; specify 'google/embeddinggemma-2'."
        )
    if not isinstance(pretrained, str):
        raise TypeError("pretrained must be a string.")
    if pretrained not in {PRETRAINED_OFFICIAL, MODEL_ID}:
        raise ValueError(
            f"Unsupported pretrained checkpoint {pretrained!r}; "
            f"use {PRETRAINED_OFFICIAL!r} or {MODEL_ID!r}."
        )

    if not isinstance(precision, str):
        raise TypeError("precision must be a string ('fp32' or 'bf16').")
    normalized_precision = precision.strip().lower()
    precision_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if normalized_precision in {"fp16", "float16"}:
        raise ValueError("FP16 is not supported; use 'fp32' or 'bf16'.")
    try:
        dtype = precision_map[normalized_precision]
    except KeyError as error:
        raise ValueError(
            f"Unsupported precision {precision!r}; use 'fp32' or 'bf16'."
        ) from error

    if isinstance(device, torch.device):
        resolved_device = device
    elif isinstance(device, str):
        try:
            resolved_device = torch.device(device)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"Invalid device {device!r}.") from error
    else:
        raise TypeError("device must be a string or torch.device.")

    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device type {resolved_device.type!r}; use CPU or CUDA."
        )
    if resolved_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {resolved_device} was requested but CUDA is unavailable."
            )
        device_count = torch.cuda.device_count()
        if resolved_device.index is not None and not (
            0 <= resolved_device.index < device_count
        ):
            raise RuntimeError(
                f"CUDA device index {resolved_device.index} is unavailable; "
                f"found {device_count} CUDA device(s)."
            )

    return resolved_device, dtype, validated_modalities


def _load_model(
    *,
    cache_dir: str | None,
    device: torch.device,
    dtype: torch.dtype,
    output_dict: bool,
    logit_scale_mode: str,
    modalities: tuple[str, ...] | None,
) -> EmbeddingGemmaCLIP:
    try:
        model_kwargs: dict[str, Any] = {
            "revision": MODEL_REVISION,
            "cache_dir": cache_dir,
            "dtype": dtype,
        }
        if modalities is not None:
            config = AutoConfig.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=cache_dir,
            )
            if "image" not in modalities:
                config.vision_config = None
            config.audio_config = None
            model_kwargs["config"] = config
        backend = AutoModel.from_pretrained(
            MODEL_ID,
            **model_kwargs,
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to load {MODEL_ID!r} at revision {MODEL_REVISION}."
        ) from error

    if not isinstance(backend, torch.nn.Module):
        raise TypeError("The Transformers model loader did not return an nn.Module.")

    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to load the processor for {MODEL_ID!r} "
            f"at revision {MODEL_REVISION}."
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
    ).eval()


def create_model(
    model_name: str = MODEL_NAME,
    pretrained: str | None = MODEL_ID,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    cache_dir: str | None = None,
    *,
    output_dict: bool = False,
    logit_scale_mode: str = "strict",
    modalities: Sequence[str] | None = None,
    **kwargs: Any,
) -> EmbeddingGemmaCLIP:
    """Load the pinned EmbeddingGemma 2 checkpoint as an OpenCLIP-style module."""
    resolved_device, dtype, validated_modalities = _validate_factory_options(
        model_name,
        pretrained,
        precision,
        device,
        cache_dir,
        output_dict,
        logit_scale_mode,
        modalities,
        kwargs,
    )
    return _load_model(
        cache_dir=cache_dir,
        device=resolved_device,
        dtype=dtype,
        output_dict=output_dict,
        logit_scale_mode=logit_scale_mode,
        modalities=validated_modalities,
    )


def create_model_and_transforms(
    model_name: str = MODEL_NAME,
    pretrained: str | None = MODEL_ID,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    cache_dir: str | None = None,
    *,
    output_dict: bool = False,
    logit_scale_mode: str = "strict",
    modalities: Sequence[str] | None = None,
    **kwargs: Any,
) -> tuple[EmbeddingGemmaCLIP, EmbeddingGemmaPreprocess, EmbeddingGemmaPreprocess]:
    """Load the model and return deterministic train and validation transforms."""
    validated_modalities = _validate_modalities(modalities)
    if validated_modalities is not None and "image" not in validated_modalities:
        raise ValueError("create_model_and_transforms() requires the image modality.")
    model = create_model(
        model_name,
        pretrained,
        precision,
        device,
        cache_dir,
        output_dict=output_dict,
        logit_scale_mode=logit_scale_mode,
        modalities=validated_modalities,
        **kwargs,
    )
    preprocess = EmbeddingGemmaPreprocess(model.processor.image_processor)
    return model, preprocess, preprocess


def create_model_from_pretrained(
    model_name: str = MODEL_NAME,
    pretrained: str | None = PRETRAINED_OFFICIAL,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    *,
    return_transform: bool = True,
    cache_dir: str | None = None,
    output_dict: bool = False,
    logit_scale_mode: str = "strict",
    modalities: Sequence[str] | None = None,
    **kwargs: Any,
) -> tuple[EmbeddingGemmaCLIP, EmbeddingGemmaPreprocess] | EmbeddingGemmaCLIP:
    """Load the supported checkpoint and optionally return its image transform."""
    if not isinstance(return_transform, bool):
        raise TypeError("return_transform must be a bool.")
    validated_modalities = _validate_modalities(modalities)
    if (
        return_transform
        and validated_modalities is not None
        and "image" not in validated_modalities
    ):
        raise ValueError("return_transform=True requires the image modality.")
    model = create_model(
        model_name,
        pretrained,
        precision,
        device,
        cache_dir,
        output_dict=output_dict,
        logit_scale_mode=logit_scale_mode,
        modalities=validated_modalities,
        **kwargs,
    )
    if not return_transform:
        return model

    preprocess = EmbeddingGemmaPreprocess(model.processor.image_processor)
    return model, preprocess


def list_models() -> list[str]:
    """Return the supported model names without loading model resources."""
    return [MODEL_NAME]


def list_pretrained(as_str: bool = False) -> list[tuple[str, str]] | list[str]:
    """Return supported model/pretrained-tag pairs without loading model resources."""
    if not isinstance(as_str, bool):
        raise TypeError("as_str must be a bool.")
    entries = [(MODEL_NAME, PRETRAINED_OFFICIAL)]
    if as_str:
        return [f"{model_name}:{tag}" for model_name, tag in entries]
    return entries
