from __future__ import annotations

from dataclasses import replace

import pytest

from openvlclip.embedding_space import (
    EmbeddingSpace,
    EmbeddingSpaceMetadata,
    embedding_spaces_compatible,
)


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


def test_legacy_embedding_space_signature_remains_unchanged() -> None:
    assert (
        _space().embedding_space_id
        == "2c525ea9f54a1ba8fbc5973e06abe0bdc06eb10d221ce80a0e8ca10ade0ff7ee"
    )


def test_versioned_embedding_space_round_trips_and_signs_policy_fields() -> None:
    space = _space()
    versioned = replace(
        space,
        schema_version=2,
        input_type="native-video",
        index_policy='{"instruction":null}',
        query_policy='{"instruction":"Describe the clip."}',
        processing_parameters='{"fps":1.0,"max_frames":64}',
    )

    assert EmbeddingSpace.from_metadata(versioned.to_metadata()) == versioned
    assert versioned.embedding_space_id != space.embedding_space_id
    assert (
        replace(versioned, query_policy='{"instruction":null}').embedding_space_id
        != versioned.embedding_space_id
    )
    assert (
        replace(versioned, processing_parameters='{"fps":2.0,"max_frames":64}')
        .embedding_space_id
        != versioned.embedding_space_id
    )


@pytest.mark.parametrize("input_type", ["image", "text", "mixed", "sampled-frame", "native-video"])
def test_versioned_embedding_space_accepts_supported_input_types(
    input_type: str,
) -> None:
    versioned = replace(
        _space(),
        schema_version=2,
        input_type=input_type,
        processing_parameters="{}",
    )

    assert EmbeddingSpace.from_metadata(versioned.to_metadata()) == versioned


def test_legacy_space_matches_only_compatible_default_v2_identity() -> None:
    legacy = _space()
    expected = replace(
        legacy,
        schema_version=2,
        input_type="image",
        processing_parameters="{}",
    )

    assert embedding_spaces_compatible(
        legacy,
        expected,
        legacy_input_type="image",
    )
    assert not embedding_spaces_compatible(
        legacy,
        replace(expected, input_type="sampled-frame"),
        legacy_input_type="image",
    )
    assert not embedding_spaces_compatible(
        legacy,
        replace(expected, query_policy="explicit"),
        legacy_input_type="image",
    )


def test_legacy_prompted_space_matches_its_default_instruction_policy() -> None:
    metadata = EmbeddingSpaceMetadata(
        backend_id="prompted-backend",
        model_id="prompted/model",
        model_revision="prompted-revision",
        embedding_strategy="native-v1",
        normalization="l2",
        preprocessing_id="prompted-preprocess-v1",
        index_prompt=None,
        query_prompt_name="SearchQuery",
        query_prompt="Represent this sentence for searching relevant passages: ",
    )
    legacy = metadata.for_index(
        embedding_dimension=768,
        model_precision="fp32",
    )
    expected = replace(
        legacy,
        schema_version=2,
        input_type="image",
        index_policy='{"input_type":"image","instruction":"backend-default"}',
        query_policy='{"instruction":null}',
        processing_parameters="{}",
    )

    assert embedding_spaces_compatible(
        legacy,
        expected,
        legacy_input_type="image",
    )
    assert not embedding_spaces_compatible(
        legacy,
        replace(expected, query_policy='{"instruction":"custom"}'),
        legacy_input_type="image",
    )


@pytest.mark.parametrize("dimension", [0, -1, True, 1.5])
def test_embedding_space_rejects_non_positive_or_non_integer_dimensions(
    dimension: object,
) -> None:
    with pytest.raises(ValueError, match="embedding_dimension"):
        _space(dimension)  # type: ignore[arg-type]
