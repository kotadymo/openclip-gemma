"""Shared supported embedding dimensions and validation."""

from __future__ import annotations

SUPPORTED_EMBEDDING_DIMENSIONS = (128, 256, 512, 768)
DEFAULT_EMBEDDING_DIMENSION = 768
EMBEDDING_STRATEGY = "leading-dimensions-l2-v1"


def validate_embedding_dimension(value: object, *, name: str) -> int:
    """Return a supported embedding dimension or raise an explicit input error."""
    dimensions = ", ".join(
        str(dimension) for dimension in SUPPORTED_EMBEDDING_DIMENSIONS
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer in {{{dimensions}}}.")
    if value not in SUPPORTED_EMBEDDING_DIMENSIONS:
        raise ValueError(f"{name} must be one of {{{dimensions}}}; received {value}.")
    return value


def resolve_embedding_dimension(value: object | None, *, name: str) -> int:
    """Resolve an optional dimension to the native 768-dimensional output."""
    if value is None:
        return DEFAULT_EMBEDDING_DIMENSION
    return validate_embedding_dimension(value, name=name)
