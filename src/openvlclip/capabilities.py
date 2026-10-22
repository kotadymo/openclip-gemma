"""Read-only capability mappings for registered inference models."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

CAPABILITY_KEYS = (
    "image_embedding",
    "text_embedding",
    "image_text_similarity",
    "patch_embedding",
    "text_similarity_map",
    "mrl",
    "selective_loading",
)


def freeze_capabilities(values: Mapping[str, bool]) -> Mapping[str, bool]:
    """Validate and return an immutable mapping with the supported capability keys."""
    if not isinstance(values, Mapping):
        raise TypeError("capabilities must be a mapping.")
    missing = set(CAPABILITY_KEYS).difference(values)
    unexpected = set(values).difference(CAPABILITY_KEYS)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected keys: {', '.join(sorted(unexpected))}")
        raise ValueError(f"Invalid capabilities mapping ({'; '.join(details)}).")
    if any(type(values[key]) is not bool for key in CAPABILITY_KEYS):
        raise TypeError("Capability values must be bools.")
    return MappingProxyType({key: values[key] for key in CAPABILITY_KEYS})


def capabilities_for_modalities(
    capabilities: Mapping[str, bool],
    modalities: tuple[str, ...],
) -> Mapping[str, bool]:
    """Return immutable capabilities constrained by loaded model modalities."""
    resolved = dict(capabilities)
    has_image = "image" in modalities
    has_text = "text" in modalities
    if not has_image:
        resolved["image_embedding"] = False
        resolved["patch_embedding"] = False
    if not has_text:
        resolved["text_embedding"] = False
    resolved["image_text_similarity"] = (
        resolved["image_text_similarity"] and has_image and has_text
    )
    resolved["text_similarity_map"] = (
        resolved["text_similarity_map"] and has_image and has_text
    )
    return freeze_capabilities(resolved)
