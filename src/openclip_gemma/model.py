"""PyTorch model wrapper for the supported EmbeddingGemma 2 backend."""

from __future__ import annotations

from typing import Any, NoReturn

import torch
from torch import nn
from torch.nn import functional as F

from openclip_gemma.constants import MODEL_CARD_MAX_CONTEXT_LENGTH


def _pool_and_normalize(
    outputs: Any,
    attention_mask: torch.Tensor,
    *,
    batch_size: int,
    normalize: bool,
    modality: str,
) -> torch.Tensor:
    hidden = outputs.last_hidden_state
    if hidden.ndim != 3 or hidden.shape[0] != batch_size:
        raise RuntimeError(
            f"The backend returned an invalid {modality} last_hidden_state shape: "
            f"{tuple(hidden.shape)}."
        )
    if hidden.shape[-1] != 768:
        raise RuntimeError(
            "The backend returned an unexpected embedding dimension "
            f"{hidden.shape[-1]}; expected 768."
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

    # EmbeddingGemma's Sentence Transformers pipeline includes Normalize.
    embeddings = F.normalize(pooled, p=2, dim=-1)
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    if not torch.isfinite(embeddings).all().item():
        raise RuntimeError(f"The backend produced non-finite {modality} embeddings.")
    return embeddings


class EmbeddingGemmaCLIP(nn.Module):
    """Wrap the Transformers backend with the planned OpenCLIP-style API."""

    def __init__(self, backend: nn.Module, processor: Any) -> None:
        super().__init__()
        if not isinstance(backend, nn.Module):
            raise TypeError("backend must be a torch.nn.Module.")
        self.backend = backend
        object.__setattr__(self, "_processor", processor)

    @property
    def processor(self) -> Any:
        """Return the processor without registering it as a PyTorch submodule."""
        return self._processor

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

    def encode_image(
        self,
        image: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Encode preprocessed OpenCLIP-style image batches."""
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor.")
        if image.ndim != 4:
            raise ValueError(
                f"image must have shape [B,3,H,W]; received {tuple(image.shape)}."
            )
        if image.shape[0] == 0:
            raise ValueError("image batch must contain at least one image.")
        if image.shape[1] != 3:
            raise ValueError(
                f"image must have 3 channels; received {image.shape[1]}."
            )
        if image.shape[2] <= 0 or image.shape[3] <= 0:
            raise ValueError("image height and width must be positive.")
        if image.dtype != torch.float32:
            raise TypeError(
                f"image must use torch.float32 from preprocess; received {image.dtype}."
            )
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool.")

        try:
            model_device = next(self.backend.parameters()).device
        except StopIteration as error:
            raise RuntimeError("The backend has no parameters to determine its device.") from error
        if image.device != model_device:
            raise ValueError(
                f"image is on {image.device}, but the model backend is on {model_device}."
            )
        if not torch.isfinite(image).all().item():
            raise ValueError("image contains NaN or infinite values.")
        if image.min().item() < 0 or image.max().item() > 1:
            raise ValueError("preprocessed image values must be in the [0,1] range.")

        image_processor = self.processor.image_processor
        patch_size = image_processor.patch_size
        if image.shape[2] % patch_size or image.shape[3] % patch_size:
            raise ValueError(
                "preprocessed image height and width must be multiples of "
                f"patch_size={patch_size}; received {image.shape[2:]}."
            )
        patch_count = (image.shape[2] // patch_size) * (
            image.shape[3] // patch_size
        )
        max_patches = (
            image_processor.max_soft_tokens * image_processor.pooling_kernel_size**2
        )
        if patch_count > max_patches:
            raise ValueError(
                f"preprocessed image contains {patch_count} patches, exceeding "
                f"the processor limit of {max_patches}."
            )

        image_groups = [[sample] for sample in image.unbind(dim=0)]
        processor_features = self.processor(
            images=image_groups,
            text=[self.processor.image_token] * image.shape[0],
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
            outputs = self.backend(**model_inputs)
            return _pool_and_normalize(
                outputs,
                model_inputs["attention_mask"],
                batch_size=image.shape[0],
                normalize=normalize,
                modality="image",
            )

    def encode_text(
        self,
        text: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Encode official EmbeddingGemma token IDs with reconstructed padding mask."""
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

        try:
            model_device = next(self.backend.parameters()).device
        except StopIteration as error:
            raise RuntimeError("The backend has no parameters to determine its device.") from error
        if text.device != model_device:
            raise ValueError(
                f"text is on {text.device}, but the model backend is on {model_device}."
            )

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("The processor has no tokenizer.")
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        vocab_size = getattr(tokenizer, "vocab_size", None)
        text_config = getattr(getattr(self.backend, "config", None), "text_config", None)
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
            raise ValueError(
                f"text token IDs must be in the range [0, {vocab_size})."
            )

        attention_mask = text.ne(pad_token_id)
        padding_seen = text.eq(pad_token_id).cumsum(dim=1).gt(0)
        if (attention_mask & padding_seen).any().item():
            raise ValueError(
                "PAD token IDs must occur only as a right-padding suffix; PAD cannot "
                "be represented as text content in the token-ID-only API."
            )
        if (text[:, 0] != bos_token_id).any().item():
            raise ValueError("Each text sequence must begin with the official BOS token.")
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
            outputs = self.backend(
                input_ids=text,
                attention_mask=attention_mask.to(dtype=torch.long),
            )
            return _pool_and_normalize(
                outputs,
                attention_mask,
                batch_size=text.shape[0],
                normalize=normalize,
                modality="text",
            )

    def encode(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Sentence Transformers encode compatibility is not in the current scope."""
        raise NotImplementedError("encode is not part of the OpenCLIP-compatible API.")
