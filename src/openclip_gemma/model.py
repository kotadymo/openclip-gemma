"""PyTorch model wrapper for the supported EmbeddingGemma 2 backend."""

from __future__ import annotations

from typing import Any, NoReturn

import torch
from torch import nn


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
        """Encode image tensors; inference is implemented in a later milestone."""
        raise NotImplementedError("encode_image is not implemented in M1.1.")

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
