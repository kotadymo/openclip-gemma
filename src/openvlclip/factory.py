"""Model factories for the supported EmbeddingGemma 2 checkpoint."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from openvlclip.constants import MODEL_ID, MODEL_NAME, MODEL_REVISION
from openvlclip.model import EmbeddingGemmaCLIP
from openvlclip.registry import ModelDescriptor, get_default_registry
from openvlclip.transforms import EmbeddingGemmaPreprocess

__all__ = [
    "MODEL_ID",
    "MODEL_NAME",
    "MODEL_REVISION",
    "PRETRAINED_OFFICIAL",
    "EmbeddingGemmaCLIP",
    "EmbeddingGemmaPreprocess",
    "create_model",
    "create_model_and_transforms",
    "create_model_from_pretrained",
    "list_models",
    "list_pretrained",
]

PRETRAINED_OFFICIAL = "official"


@dataclass(frozen=True)
class _FactoryOptions:
    descriptor: ModelDescriptor
    checkpoint_id: str
    device: torch.device
    dtype: torch.dtype
    modalities: tuple[str, ...] | None


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
) -> _FactoryOptions:
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

    descriptor = get_default_registry().resolve(model_name)

    if pretrained is None:
        raise ValueError(
            "pretrained=None is not supported; specify 'google/embeddinggemma-2'."
        )
    if not isinstance(pretrained, str):
        raise TypeError("pretrained must be a string.")
    checkpoint_id = descriptor.checkpoint_for(pretrained)

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

    return _FactoryOptions(
        descriptor=descriptor,
        checkpoint_id=checkpoint_id,
        device=resolved_device,
        dtype=dtype,
        modalities=validated_modalities,
    )


def _load_model(
    options: _FactoryOptions,
    *,
    cache_dir: str | None,
    output_dict: bool,
    logit_scale_mode: str,
) -> EmbeddingGemmaCLIP:
    """Delegate model loading to the Registry-selected backend."""
    return options.descriptor.backend.load_model(
        model_id=options.checkpoint_id,
        revision=options.descriptor.revision,
        cache_dir=cache_dir,
        device=options.device,
        dtype=options.dtype,
        output_dict=output_dict,
        logit_scale_mode=logit_scale_mode,
        modalities=options.modalities,
    )


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
    options = _validate_factory_options(
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
        options,
        cache_dir=cache_dir,
        output_dict=output_dict,
        logit_scale_mode=logit_scale_mode,
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
    descriptor = get_default_registry().resolve(model_name)
    preprocess = descriptor.backend.create_preprocess(model)
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

    descriptor = get_default_registry().resolve(model_name)
    preprocess = descriptor.backend.create_preprocess(model)
    return model, preprocess


def list_models() -> list[str]:
    """Return the supported model names without loading model resources."""
    return get_default_registry().list_models()


def list_pretrained(as_str: bool = False) -> list[tuple[str, str]] | list[str]:
    """Return supported model/pretrained-tag pairs without loading model resources."""
    if not isinstance(as_str, bool):
        raise TypeError("as_str must be a bool.")
    entries = get_default_registry().list_pretrained()
    if as_str:
        return [f"{model_name}:{tag}" for model_name, tag in entries]
    return entries
