"""Read-only model registry used by public Factory functions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from openvlclip.backends.base import Backend
from openvlclip.capabilities import freeze_capabilities
from openvlclip.constants import MODEL_ID, MODEL_NAME, MODEL_REVISION
from openvlclip.embedding_space import EmbeddingSpaceMetadata


def normalize_model_name(model_name: str) -> str:
    """Normalize a model name using the v0.4.0 Factory convention."""
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string.")
    return model_name.strip().lower().replace("_", "-")


@dataclass(frozen=True)
class ModelDescriptor:
    """Immutable metadata that binds a public model name to one backend."""

    model_name: str
    aliases: tuple[str, ...]
    backend_id: str
    model_id: str
    revision: str
    pretrained_tags: Mapping[str, str]
    pretrained_aliases: Mapping[str, str]
    capabilities: Mapping[str, bool]
    embedding_space_metadata: EmbeddingSpaceMetadata
    backend: Backend[Any, Any, Any]

    def __post_init__(self) -> None:
        canonical_name = normalize_model_name(self.model_name)
        if not canonical_name or canonical_name != self.model_name:
            raise ValueError("model_name must be a normalized, non-empty name.")
        if not isinstance(self.aliases, tuple) or any(
            not isinstance(alias, str) or not alias.strip() for alias in self.aliases
        ):
            raise TypeError("aliases must be a tuple of non-empty strings.")
        if not all(
            isinstance(value, str) and value
            for value in (self.backend_id, self.model_id, self.revision)
        ):
            raise ValueError("backend_id, model_id, and revision must be non-empty.")
        if getattr(self.backend, "backend_id", None) != self.backend_id:
            raise ValueError("The registered backend_id does not match the backend.")
        for method_name in (
            "capabilities_for",
            "load_model",
            "create_tokenizer",
            "create_preprocess",
            "validate_embedding_dimension",
            "encode_image",
            "encode_text",
        ):
            if not callable(getattr(self.backend, method_name, None)):
                raise TypeError(f"The registered backend is missing {method_name}().")
        if not isinstance(
            self.embedding_space_metadata,
            EmbeddingSpaceMetadata,
        ):
            raise TypeError(
                "embedding_space_metadata must be an EmbeddingSpaceMetadata."
            )
        if (
            self.embedding_space_metadata.backend_id != self.backend_id
            or self.embedding_space_metadata.model_id != self.model_id
            or self.embedding_space_metadata.model_revision != self.revision
        ):
            raise ValueError(
                "Model descriptor and embedding-space metadata must identify "
                "the same backend and checkpoint."
            )

        tags = dict(self.pretrained_tags)
        if not tags or any(
            not isinstance(tag, str)
            or not tag
            or not isinstance(checkpoint, str)
            or not checkpoint
            for tag, checkpoint in tags.items()
        ):
            raise ValueError("pretrained_tags must map non-empty tags to checkpoints.")
        aliases = dict(self.pretrained_aliases)
        if any(
            not isinstance(alias, str)
            or not alias
            or not isinstance(tag, str)
            or tag not in tags
            for alias, tag in aliases.items()
        ):
            raise ValueError(
                "pretrained_aliases must map non-empty aliases to registered tags."
            )
        capabilities = freeze_capabilities(self.capabilities)
        if dict(capabilities) != dict(self.backend.capabilities):
            raise ValueError("Registry capabilities must match the registered backend.")
        if dict(capabilities) != dict(self.capabilities_for_full_model()):
            raise ValueError("The backend's full-model capabilities are inconsistent.")

        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "pretrained_tags", MappingProxyType(tags))
        object.__setattr__(self, "pretrained_aliases", MappingProxyType(aliases))
        object.__setattr__(self, "capabilities", capabilities)

    def capabilities_for_full_model(self) -> Mapping[str, bool]:
        """Get the backend's full-modality capabilities for consistency checks."""
        return self.backend.capabilities_for(("text", "image", "video"))

    def checkpoint_for(self, pretrained: str) -> str:
        """Resolve a public pretrained tag or an accepted checkpoint alias."""
        tag = pretrained if pretrained in self.pretrained_tags else None
        if tag is None:
            tag = self.pretrained_aliases.get(pretrained)
        if tag is None:
            supported = sorted((*self.pretrained_tags, *self.pretrained_aliases))
            raise ValueError(
                f"Unsupported pretrained checkpoint {pretrained!r}; use one of: "
                f"{', '.join(repr(value) for value in supported)}."
            )
        return self.pretrained_tags[tag]


class ModelRegistry:
    """Resolve registered model names without loading models or accessing the Hub."""

    def __init__(self, descriptors: Iterable[ModelDescriptor]) -> None:
        resolved = tuple(descriptors)
        if not resolved:
            raise ValueError("A model registry must contain at least one model.")
        names: dict[str, ModelDescriptor] = {}
        for descriptor in resolved:
            if not isinstance(descriptor, ModelDescriptor):
                raise TypeError("Registry entries must be ModelDescriptor objects.")
            for name in (descriptor.model_name, *descriptor.aliases):
                normalized = normalize_model_name(name)
                if not normalized:
                    raise ValueError("Model names and aliases cannot be empty.")
                if normalized in names:
                    raise ValueError(
                        f"Duplicate model name or alias {name!r} in the registry."
                    )
                names[normalized] = descriptor
        self._descriptors = resolved
        self._names = MappingProxyType(names)

    @property
    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        """Return the immutable registered descriptor sequence."""
        return self._descriptors

    def resolve(self, model_name: str) -> ModelDescriptor:
        """Resolve a canonical name or alias without performing model I/O."""
        normalized = normalize_model_name(model_name)
        try:
            return self._names[normalized]
        except KeyError as error:
            supported = ", ".join(self.list_models())
            raise ValueError(
                f"Unsupported model_name {model_name!r}; supported models: {supported}."
            ) from error

    def resolve_backend_for_embedding_space(
        self,
        *,
        backend_id: str,
        model_id: str,
        revision: str,
    ) -> Backend[Any, Any, Any]:
        """Find a registered backend by the stable checkpoint identity in an index."""
        for descriptor in self._descriptors:
            if (
                descriptor.backend_id == backend_id
                and descriptor.model_id == model_id
                and descriptor.revision == revision
            ):
                return descriptor.backend
        raise ValueError(
            "No registered backend matches the embedding-space identity "
            f"{backend_id!r}/{model_id!r}@{revision!r}."
        )

    def list_models(self) -> list[str]:
        """Return canonical model names in registry order."""
        return [descriptor.model_name for descriptor in self._descriptors]

    def list_pretrained(self) -> list[tuple[str, str]]:
        """Return public pretrained tags in registry order."""
        return [
            (descriptor.model_name, tag)
            for descriptor in self._descriptors
            for tag in descriptor.pretrained_tags
        ]


@lru_cache(maxsize=1)
def get_default_registry() -> ModelRegistry:
    """Return the built-in registry; imports and listing do not load model weights."""
    from openvlclip.backends.dinov2_dinotxt import (
        DINO_BACKEND_ID,
        DINO_HUB_MODEL_ID,
        DINO_MODEL_NAME,
        DINO_MODEL_REVISION,
        DinoV2DinoTxtBackend,
    )
    from openvlclip.backends.embeddinggemma2 import EmbeddingGemma2Backend
    from openvlclip.backends.qwen3_vl_embedding import (
        QWEN3VL_2B_DIMENSION,
        QWEN3VL_2B_MODEL_ID,
        QWEN3VL_2B_MODEL_NAME,
        QWEN3VL_2B_REVISION,
        Qwen3VLBackend,
    )
    from openvlclip.backends.siglip2 import (
        SIGLIP2_BASE_384_MODEL_ID,
        SIGLIP2_BASE_384_MODEL_NAME,
        SIGLIP2_BASE_384_REVISION,
        SIGLIP2_FIXRES_MODEL_ID,
        SIGLIP2_FIXRES_MODEL_NAME,
        SIGLIP2_FIXRES_REVISION,
        SIGLIP2_LARGE_256_MODEL_ID,
        SIGLIP2_LARGE_256_MODEL_NAME,
        SIGLIP2_LARGE_256_REVISION,
        SIGLIP2_NAFLEX_MODEL_ID,
        SIGLIP2_NAFLEX_MODEL_NAME,
        SIGLIP2_NAFLEX_REVISION,
        SIGLIP2_SO400M_384_MODEL_ID,
        SIGLIP2_SO400M_384_MODEL_NAME,
        SIGLIP2_SO400M_384_REVISION,
        Siglip2Backend,
    )

    gemma_backend = EmbeddingGemma2Backend()
    dinotxt_backend = DinoV2DinoTxtBackend()
    siglip2_fixres_backend = Siglip2Backend(
        model_name=SIGLIP2_FIXRES_MODEL_NAME,
        model_id=SIGLIP2_FIXRES_MODEL_ID,
        revision=SIGLIP2_FIXRES_REVISION,
        variant="fixres",
    )
    siglip2_naflex_backend = Siglip2Backend(
        model_name=SIGLIP2_NAFLEX_MODEL_NAME,
        model_id=SIGLIP2_NAFLEX_MODEL_ID,
        revision=SIGLIP2_NAFLEX_REVISION,
        variant="naflex",
    )
    siglip2_base_384_backend = Siglip2Backend(
        model_name=SIGLIP2_BASE_384_MODEL_NAME,
        model_id=SIGLIP2_BASE_384_MODEL_ID,
        revision=SIGLIP2_BASE_384_REVISION,
        variant="fixres",
        native_dimension=768,
        image_size=384,
    )
    siglip2_large_256_backend = Siglip2Backend(
        model_name=SIGLIP2_LARGE_256_MODEL_NAME,
        model_id=SIGLIP2_LARGE_256_MODEL_ID,
        revision=SIGLIP2_LARGE_256_REVISION,
        variant="fixres",
        native_dimension=1024,
        image_size=256,
    )
    siglip2_so400m_384_backend = Siglip2Backend(
        model_name=SIGLIP2_SO400M_384_MODEL_NAME,
        model_id=SIGLIP2_SO400M_384_MODEL_ID,
        revision=SIGLIP2_SO400M_384_REVISION,
        variant="fixres",
        native_dimension=1152,
        image_size=384,
    )
    qwen3vl_2b_backend = Qwen3VLBackend(
        model_name=QWEN3VL_2B_MODEL_NAME,
        model_id=QWEN3VL_2B_MODEL_ID,
        revision=QWEN3VL_2B_REVISION,
        native_dimension=QWEN3VL_2B_DIMENSION,
    )
    return ModelRegistry(
        (
            ModelDescriptor(
                model_name=MODEL_NAME,
                aliases=(),
                backend_id=gemma_backend.backend_id,
                model_id=MODEL_ID,
                revision=MODEL_REVISION,
                pretrained_tags={"official": MODEL_ID},
                pretrained_aliases={MODEL_ID: "official"},
                capabilities=gemma_backend.capabilities,
                embedding_space_metadata=gemma_backend.embedding_space_metadata,
                backend=gemma_backend,
            ),
            ModelDescriptor(
                model_name=DINO_MODEL_NAME,
                aliases=(),
                backend_id=DINO_BACKEND_ID,
                model_id=DINO_HUB_MODEL_ID,
                revision=DINO_MODEL_REVISION,
                pretrained_tags={"official": DINO_HUB_MODEL_ID},
                pretrained_aliases={},
                capabilities=dinotxt_backend.capabilities,
                embedding_space_metadata=dinotxt_backend.embedding_space_metadata,
                backend=dinotxt_backend,
            ),
            ModelDescriptor(
                model_name=SIGLIP2_FIXRES_MODEL_NAME,
                aliases=(),
                backend_id=siglip2_fixres_backend.backend_id,
                model_id=SIGLIP2_FIXRES_MODEL_ID,
                revision=SIGLIP2_FIXRES_REVISION,
                pretrained_tags={"official": SIGLIP2_FIXRES_MODEL_ID},
                pretrained_aliases={},
                capabilities=siglip2_fixres_backend.capabilities,
                embedding_space_metadata=(
                    siglip2_fixres_backend.embedding_space_metadata
                ),
                backend=siglip2_fixres_backend,
            ),
            ModelDescriptor(
                model_name=SIGLIP2_NAFLEX_MODEL_NAME,
                aliases=(),
                backend_id=siglip2_naflex_backend.backend_id,
                model_id=SIGLIP2_NAFLEX_MODEL_ID,
                revision=SIGLIP2_NAFLEX_REVISION,
                pretrained_tags={"official": SIGLIP2_NAFLEX_MODEL_ID},
                pretrained_aliases={},
                capabilities=siglip2_naflex_backend.capabilities,
                embedding_space_metadata=(
                    siglip2_naflex_backend.embedding_space_metadata
                ),
                backend=siglip2_naflex_backend,
            ),
            ModelDescriptor(
                model_name=SIGLIP2_BASE_384_MODEL_NAME,
                aliases=(),
                backend_id=siglip2_base_384_backend.backend_id,
                model_id=SIGLIP2_BASE_384_MODEL_ID,
                revision=SIGLIP2_BASE_384_REVISION,
                pretrained_tags={"official": SIGLIP2_BASE_384_MODEL_ID},
                pretrained_aliases={},
                capabilities=siglip2_base_384_backend.capabilities,
                embedding_space_metadata=siglip2_base_384_backend.embedding_space_metadata,
                backend=siglip2_base_384_backend,
            ),
            ModelDescriptor(
                model_name=SIGLIP2_LARGE_256_MODEL_NAME,
                aliases=(),
                backend_id=siglip2_large_256_backend.backend_id,
                model_id=SIGLIP2_LARGE_256_MODEL_ID,
                revision=SIGLIP2_LARGE_256_REVISION,
                pretrained_tags={"official": SIGLIP2_LARGE_256_MODEL_ID},
                pretrained_aliases={},
                capabilities=siglip2_large_256_backend.capabilities,
                embedding_space_metadata=siglip2_large_256_backend.embedding_space_metadata,
                backend=siglip2_large_256_backend,
            ),
            ModelDescriptor(
                model_name=SIGLIP2_SO400M_384_MODEL_NAME,
                aliases=(),
                backend_id=siglip2_so400m_384_backend.backend_id,
                model_id=SIGLIP2_SO400M_384_MODEL_ID,
                revision=SIGLIP2_SO400M_384_REVISION,
                pretrained_tags={"official": SIGLIP2_SO400M_384_MODEL_ID},
                pretrained_aliases={},
                capabilities=siglip2_so400m_384_backend.capabilities,
                embedding_space_metadata=siglip2_so400m_384_backend.embedding_space_metadata,
                backend=siglip2_so400m_384_backend,
            ),
            ModelDescriptor(
                model_name=QWEN3VL_2B_MODEL_NAME,
                aliases=(),
                backend_id=qwen3vl_2b_backend.backend_id,
                model_id=QWEN3VL_2B_MODEL_ID,
                revision=QWEN3VL_2B_REVISION,
                pretrained_tags={"official": QWEN3VL_2B_MODEL_ID},
                pretrained_aliases={},
                capabilities=qwen3vl_2b_backend.capabilities,
                embedding_space_metadata=qwen3vl_2b_backend.embedding_space_metadata,
                backend=qwen3vl_2b_backend,
            ),
        )
    )
