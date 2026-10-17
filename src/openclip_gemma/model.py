"""PyTorch model wrapper for the supported EmbeddingGemma 2 backend."""

from __future__ import annotations

from typing import Any, NoReturn

import torch
from torch import nn
from torch.nn import functional as F


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
            hidden = outputs.last_hidden_state
            attention_mask = model_inputs["attention_mask"]
            if hidden.ndim != 3 or hidden.shape[0] != image.shape[0]:
                raise RuntimeError(
                    "The backend returned an invalid last_hidden_state shape: "
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
                raise RuntimeError("The backend produced non-finite image embeddings.")
            return embeddings

    def encode_text(
        self,
        text: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Encode token tensors; inference is implemented in a later milestone."""
        raise NotImplementedError("encode_text is not implemented in M1.1.")

    def encode(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Sentence Transformers encode compatibility is not in the current scope."""
        raise NotImplementedError("encode is not part of the OpenCLIP-compatible API.")
