"""Model factories for the supported EmbeddingGemma 2 checkpoint."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, AutoProcessor

from openclip_gemma.model import EmbeddingGemmaCLIP
from openclip_gemma.transforms import EmbeddingGemmaPreprocess

MODEL_NAME = "embeddinggemma-2"
MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def _validate_factory_options(
    model_name: str,
    pretrained: str | None,
    precision: str,
    device: str | torch.device,
    cache_dir: str | None,
    kwargs: dict[str, Any],
) -> tuple[torch.device, torch.dtype]:
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
    if pretrained != MODEL_ID:
        raise ValueError(
            f"Unsupported pretrained checkpoint {pretrained!r}; "
            f"only {MODEL_ID!r} is supported."
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

    if cache_dir is not None and not isinstance(cache_dir, str):
        raise TypeError("cache_dir must be a string or None.")
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported keyword argument(s): {names}.")

    return resolved_device, dtype


def _load_model(
    *,
    cache_dir: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> EmbeddingGemmaCLIP:
    try:
        backend = AutoModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            dtype=dtype,
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

    if getattr(processor, "image_processor", None) is None:
        raise RuntimeError("The loaded processor has no image_processor.")

    backend = backend.to(device=device, dtype=dtype).eval()
    return EmbeddingGemmaCLIP(backend=backend, processor=processor).eval()


def create_model(
    model_name: str = MODEL_NAME,
    pretrained: str | None = MODEL_ID,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    cache_dir: str | None = None,
    **kwargs: Any,
) -> torch.nn.Module:
    """Load the pinned EmbeddingGemma 2 checkpoint as an OpenCLIP-style module."""
    resolved_device, dtype = _validate_factory_options(
        model_name,
        pretrained,
        precision,
        device,
        cache_dir,
        kwargs,
    )
    return _load_model(cache_dir=cache_dir, device=resolved_device, dtype=dtype)


def create_model_and_transforms(
    model_name: str = MODEL_NAME,
    pretrained: str | None = MODEL_ID,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    cache_dir: str | None = None,
    **kwargs: Any,
) -> tuple[torch.nn.Module, EmbeddingGemmaPreprocess, EmbeddingGemmaPreprocess]:
    """Load the model and return deterministic train and validation transforms."""
    model = create_model(
        model_name,
        pretrained,
        precision,
        device,
        cache_dir,
        **kwargs,
    )
    preprocess = EmbeddingGemmaPreprocess(model.processor.image_processor)
    return model, preprocess, preprocess
