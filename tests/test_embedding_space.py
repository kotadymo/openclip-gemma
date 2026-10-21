from __future__ import annotations

from dataclasses import replace

import pytest

from openclip_gemma.embedding_space import EmbeddingSpace, EmbeddingSpaceMetadata


def _space(embedding_dimension: int = 384) -> EmbeddingSpace:
    metadata = EmbeddingSpaceMetadata(
        backend_id="test-backend",
        model_id="test/model",
        model_revision="test-revision",
        embedding_strategy="native-v1",
        normalization="l2",
        preprocessing_id="test-preprocess-v1",
        index_prompt=None,
        query_prompt_name=None,
        query_prompt=None,
    )
    return metadata.for_index(
        embedding_dimension=embedding_dimension,
        model_precision="fp32",
    )


def test_embedding_space_accepts_and_round_trips_non_gemma_dimension() -> None:
    space = _space()

    assert space.embedding_dimension == 384
    assert EmbeddingSpace.from_metadata(space.to_metadata()) == space


def test_embedding_space_signature_tracks_dimension_and_checkpoint_identity() -> None:
    space = _space()

    assert (
        replace(space, embedding_dimension=512).embedding_space_id
        != space.embedding_space_id
    )
    assert replace(space, model_id="other/model").embedding_space_id != (
        space.embedding_space_id
    )
    assert replace(space, model_revision="other-revision").embedding_space_id != (
        space.embedding_space_id
    )


@pytest.mark.parametrize("dimension", [0, -1, True, 1.5])
def test_embedding_space_rejects_non_positive_or_non_integer_dimensions(
    dimension: object,
) -> None:
    with pytest.raises(ValueError, match="embedding_dimension"):
        _space(dimension)  # type: ignore[arg-type]
