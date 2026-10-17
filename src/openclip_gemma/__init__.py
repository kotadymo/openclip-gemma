"""OpenCLIP-compatible factory API for EmbeddingGemma 2."""

from openclip_gemma.factory import create_model, create_model_and_transforms
from openclip_gemma.model import EmbeddingGemmaCLIP

__all__ = [
    "EmbeddingGemmaCLIP",
    "create_model",
    "create_model_and_transforms",
]


def main() -> None:
    print("Hello from openclip-gemma!")
