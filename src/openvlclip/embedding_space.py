"""Deterministic identity metadata for stored embedding vectors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from openvlclip._dimensions import validate_positive_embedding_dimension

_SIGNATURE_FIELDS = (
    "backend_id",
    "model_id",
    "model_revision",
    "embedding_dimension",
    "embedding_strategy",
    "normalization",
    "preprocessing_id",
    "index_prompt",
    "query_prompt_name",
    "query_prompt",
    "model_precision",
    "storage_dtype",
    "sampling_method",
)

_SPACE_METADATA_FIELDS = frozenset((*_SIGNATURE_FIELDS, "embedding_space_id"))


@dataclass(frozen=True)
class EmbeddingSpace:
    """The model, preprocessing, prompt, and numeric settings of an index."""

    backend_id: str
    model_id: str
    model_revision: str
    embedding_dimension: int
    embedding_strategy: str
    normalization: str
    preprocessing_id: str
    index_prompt: str | None
    query_prompt_name: str | None
    query_prompt: str | None
    model_precision: str
    storage_dtype: str = "float32"
    sampling_method: str | None = None

    def __post_init__(self) -> None:
        try:
            dimension = validate_positive_embedding_dimension(
                self.embedding_dimension,
                name="embedding_dimension",
            )
        except (TypeError, ValueError) as error:
            raise ValueError("embedding_dimension is invalid.") from error
        if dimension != self.embedding_dimension:
            raise ValueError("embedding_dimension must be an integer.")
        for name in (
            "backend_id",
            "model_id",
            "model_revision",
            "embedding_strategy",
            "normalization",
            "preprocessing_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if self.index_prompt is not None and not isinstance(self.index_prompt, str):
            raise TypeError("index_prompt must be a string or None.")
        if (self.query_prompt_name is None) != (self.query_prompt is None):
            raise ValueError(
                "query_prompt_name and query_prompt must both be set or both be None."
            )
        if self.query_prompt_name is not None and (
            not isinstance(self.query_prompt_name, str)
            or not self.query_prompt_name
            or not isinstance(self.query_prompt, str)
            or not self.query_prompt
        ):
            raise ValueError("Query prompt metadata must contain non-empty strings.")
        if not isinstance(self.model_precision, str) or self.model_precision not in {
            "fp32",
            "bf16",
        }:
            raise ValueError("model_precision must be 'fp32' or 'bf16'.")
        if self.storage_dtype != "float32":
            raise ValueError("storage_dtype must be 'float32'.")
        if self.sampling_method is not None and (
            not isinstance(self.sampling_method, str) or not self.sampling_method
        ):
            raise ValueError("sampling_method must be a non-empty string or None.")

    def _signature_payload(self) -> list[tuple[str, str | int | None]]:
        return [(name, getattr(self, name)) for name in _SIGNATURE_FIELDS]

    @property
    def embedding_space_id(self) -> str:
        """Return a stable SHA-256 signature over fields in a fixed order."""
        payload = json.dumps(
            self._signature_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_metadata(self) -> dict[str, object]:
        """Serialize all space fields and their deterministic signature."""
        result = dict(self._signature_payload())
        result["embedding_space_id"] = self.embedding_space_id
        return result

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> EmbeddingSpace:
        """Parse extended index metadata and verify its embedding-space signature."""
        missing = _SPACE_METADATA_FIELDS.difference(metadata)
        if missing:
            raise ValueError(
                f"Embedding-space metadata is incomplete: {', '.join(sorted(missing))}."
            )
        string_fields = (
            "backend_id",
            "model_id",
            "model_revision",
            "embedding_strategy",
            "normalization",
            "preprocessing_id",
            "model_precision",
            "storage_dtype",
            "embedding_space_id",
        )
        for name in string_fields:
            if not isinstance(metadata[name], str):
                raise TypeError(f"{name} must be a string.")
        dimension = metadata["embedding_dimension"]
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("embedding_dimension must be an integer.")
        index_prompt = metadata["index_prompt"]
        if index_prompt is not None and not isinstance(index_prompt, str):
            raise TypeError("index_prompt must be a string or None.")
        query_prompt_name = metadata["query_prompt_name"]
        if query_prompt_name is not None and not isinstance(
            query_prompt_name,
            str,
        ):
            raise TypeError("query_prompt_name must be a string or None.")
        query_prompt = metadata["query_prompt"]
        if query_prompt is not None and not isinstance(query_prompt, str):
            raise TypeError("query_prompt must be a string or None.")
        sampling_method = metadata["sampling_method"]
        if sampling_method is not None and not isinstance(sampling_method, str):
            raise TypeError("sampling_method must be a string or None.")
        space = cls(
            backend_id=metadata["backend_id"],
            model_id=metadata["model_id"],
            model_revision=metadata["model_revision"],
            embedding_dimension=dimension,
            embedding_strategy=metadata["embedding_strategy"],
            normalization=metadata["normalization"],
            preprocessing_id=metadata["preprocessing_id"],
            index_prompt=index_prompt,
            query_prompt_name=query_prompt_name,
            query_prompt=query_prompt,
            model_precision=metadata["model_precision"],
            storage_dtype=metadata["storage_dtype"],
            sampling_method=sampling_method,
        )
        signature = metadata["embedding_space_id"]
        if not isinstance(signature, str) or signature != space.embedding_space_id:
            raise ValueError("embedding_space_id does not match its metadata.")
        return space


@dataclass(frozen=True)
class EmbeddingSpaceMetadata:
    """Backend-owned settings shared by indexes at different dimensions."""

    backend_id: str
    model_id: str
    model_revision: str
    embedding_strategy: str
    normalization: str
    preprocessing_id: str
    index_prompt: str | None
    query_prompt_name: str | None
    query_prompt: str | None

    def for_index(
        self,
        *,
        embedding_dimension: int,
        model_precision: str,
        sampling_method: str | None = None,
        storage_dtype: str = "float32",
    ) -> EmbeddingSpace:
        """Create the complete identity used by one index."""
        return EmbeddingSpace(
            backend_id=self.backend_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            embedding_dimension=embedding_dimension,
            embedding_strategy=self.embedding_strategy,
            normalization=self.normalization,
            preprocessing_id=self.preprocessing_id,
            index_prompt=self.index_prompt,
            query_prompt_name=self.query_prompt_name,
            query_prompt=self.query_prompt,
            model_precision=model_precision,
            storage_dtype=storage_dtype,
            sampling_method=sampling_method,
        )


def has_embedding_space_metadata(metadata: Mapping[str, object]) -> bool:
    """Return whether any extension identity field is present in an archive."""
    extension_fields = _SPACE_METADATA_FIELDS.difference(
        {
            "model_id",
            "model_revision",
            "embedding_dimension",
            "embedding_strategy",
            "preprocessing_id",
            "model_precision",
            "storage_dtype",
            "sampling_method",
        }
    )
    return bool(extension_fields.intersection(metadata))
