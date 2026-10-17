"""OpenCLIP-compatible factory API for EmbeddingGemma 2."""

from openclip_gemma.factory import create_model, create_model_and_transforms
from openclip_gemma.model import EmbeddingGemmaCLIP
from openclip_gemma.tokenizer import get_tokenizer

__all__ = [
    "EmbeddingGemmaCLIP",
    "create_model",
    "create_model_and_transforms",
    "get_tokenizer",
]
