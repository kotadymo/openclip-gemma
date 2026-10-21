"""Shared positive-dimension validation and compatibility defaults."""

from __future__ import annotations

DEFAULT_EMBEDDING_DIMENSION = 768
EMBEDDING_STRATEGY = "leading-dimensions-l2-v1"


def validate_positive_embedding_dimension(value: object, *, name: str) -> int:
    """Validate a model-independent positive embedding dimension."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value
