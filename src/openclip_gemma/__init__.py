"""OpenCLIP-compatible factory API for EmbeddingGemma 2."""

from openclip_gemma.factory import (
    create_model,
    create_model_and_transforms,
    create_model_from_pretrained,
    list_models,
    list_pretrained,
)
from openclip_gemma.model import EmbeddingGemmaCLIP
from openclip_gemma.tokenizer import get_tokenizer

__all__ = [
    "EmbeddingGemmaCLIP",
    "create_model",
    "create_model_and_transforms",
    "create_model_from_pretrained",
    "get_tokenizer",
    "list_models",
    "list_pretrained",
]
