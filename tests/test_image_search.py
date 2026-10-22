from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from openvlclip import image_search
from openvlclip.backends.dinov2_dinotxt import DinoV2DinoTxtBackend
from openvlclip.backends.embeddinggemma2 import EmbeddingGemma2Backend
from openvlclip.embedding_space import EmbeddingSpaceMetadata
from openvlclip.image_search import (
    ImageIndexCompatibilityError,
    ImageIndexError,
    _load_index,
    _search_index,
    build_or_update_image_index,
    convert_image_index,
    search_images,
)
from openvlclip.index_storage import _save_numpy_archive_atomic


class FakeImageSearchModel(nn.Module):
    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        *,
        embedding_space_metadata: EmbeddingSpaceMetadata | None = None,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=dtype))
        self.encoded_image_count = 0
        self.received_tokens: list[torch.Tensor] = []
        self.embedding_space_metadata = (
            embedding_space_metadata or EmbeddingGemma2Backend.embedding_space_metadata
        )
        self._dimension_backend = EmbeddingGemma2Backend()

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int:
        return self._dimension_backend.validate_embedding_dimension(
            value,
            name=name,
        )

    def encode_image(
        self,
        image: torch.Tensor,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        self.encoded_image_count += image.shape[0]
        features = image.mean(dim=(2, 3))
        features = torch.nn.functional.normalize(features, dim=-1)
        features = torch.nn.functional.pad(features, (0, 765))
        if truncate_dim is not None:
            features = torch.nn.functional.normalize(
                features[:, :truncate_dim],
                dim=-1,
            )
        return features

    def encode_text(
        self,
        text: torch.Tensor,
        *,
        normalize: bool = False,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        self.received_tokens.append(text.detach().cpu())
        embedding_dim = 768 if truncate_dim is None else truncate_dim
        features = torch.zeros(
            (text.shape[0], embedding_dim),
            device=text.device,
            dtype=self.anchor.dtype,
        )
        features[:, 0] = 1
        return (
            torch.nn.functional.normalize(features, dim=-1) if normalize else features
        )


class FakeDinoImageSearchModel(nn.Module):
    def __init__(
        self,
        *,
        embedding_space_metadata: EmbeddingSpaceMetadata | None = None,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self._dimension_backend = DinoV2DinoTxtBackend()
        self.embedding_space_metadata = (
            embedding_space_metadata
            or self._dimension_backend.embedding_space_metadata
        )
        self.received_prompt_names: list[str | None] = []
        self.tokenizer = FakeTokenizer()

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int:
        return self._dimension_backend.validate_embedding_dimension(
            value,
            name=name,
        )

    def encode_image(
        self,
        image: torch.Tensor,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        dimension = self.validate_embedding_dimension(
            truncate_dim,
            name="truncate_dim",
        )
        features = torch.zeros(
            (image.shape[0], dimension),
            device=image.device,
            dtype=self.anchor.dtype,
        )
        features[:, :3] = image.mean(dim=(2, 3))
        return features

    def encode_text(
        self,
        text: torch.Tensor,
        *,
        normalize: bool = False,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        dimension = self.validate_embedding_dimension(
            truncate_dim,
            name="truncate_dim",
        )
        features = torch.zeros(
            (text.shape[0], dimension),
            device=text.device,
            dtype=self.anchor.dtype,
        )
        features[:, 0] = 1
        return (
            torch.nn.functional.normalize(features, dim=-1) if normalize else features
        )

    def _create_tokenizer(self, *, prompt_name: str | None) -> FakeTokenizer:
        self.received_prompt_names.append(prompt_name)
        return self.tokenizer


class FakeTokenizer:
    def __init__(self) -> None:
        self.received_texts: list[str] = []

    def __call__(self, text: str) -> torch.Tensor:
        self.received_texts.append(text)
        return torch.tensor([[1]], dtype=torch.long)


def _preprocess(image: Image.Image) -> torch.Tensor:
    rgb = image.convert("RGB")
    return (
        torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy()).permute(2, 0, 1)
        / 255
    )


def _write_image(
    path: Path,
    color: tuple[int, int, int],
    *,
    size: tuple[int, int] = (8, 8),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _build(
    model: FakeImageSearchModel,
    images_dir: Path,
    index_dir: Path,
    *,
    batch_size: int = 4,
    recursive: bool = False,
    embedding_dim: int = 768,
):
    return build_or_update_image_index(
        model,
        _preprocess,
        images_dir,
        index_dir,
        batch_size=batch_size,
        recursive=recursive,
        embedding_dim=embedding_dim,
    )


def _rewrite_index_metadata(
    index_path: Path,
    updates: dict[str, object],
    *,
    remove: tuple[str, ...] = (),
) -> None:
    with np.load(index_path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        embeddings = archive["embeddings"].copy()
    metadata.update(updates)
    for name in remove:
        metadata.pop(name, None)
    _save_numpy_archive_atomic(
        index_path,
        metadata,
        {"embeddings": embeddings},
    )


def _rewrite_index_version(index_path: Path, version: int) -> None:
    _rewrite_index_metadata(
        index_path,
        {"format_version": version},
        remove=(
            (
                "embedding_strategy",
                "backend_id",
                "normalization",
                "index_prompt",
                "query_prompt_name",
                "query_prompt",
                "sampling_method",
                "embedding_space_id",
            )
            if version == 1
            else ()
        ),
    )


def test_index_build_round_trip_and_incremental_changes(tmp_path: Path) -> None:
    images_dir = tmp_path / "photos"
    index_dir = tmp_path / "index"
    _write_image(images_dir / "a.png", (255, 0, 0))
    _write_image(images_dir / "b.jpg", (0, 255, 0))
    _write_image(images_dir / "nested" / "c.webp", (0, 0, 255))
    _write_image(images_dir / "ignored.gif", (255, 255, 0))
    model = FakeImageSearchModel()

    first = _build(model, images_dir, index_dir, recursive=True)

    assert (first.added, first.updated, first.removed, first.unchanged) == (3, 0, 0, 0)
    assert model.encoded_image_count == 3
    original_archive = first.index_path.read_bytes()
    loaded = _load_index(
        index_dir, expected_root=images_dir.resolve(), expected_precision="fp32"
    )
    assert [record.relative_path for record in loaded.records] == [
        "a.png",
        "b.jpg",
        "nested/c.webp",
    ]
    assert loaded.embeddings.shape == (3, 768)
    assert loaded.embeddings.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(loaded.embeddings, axis=1), 1.0, atol=1e-6
    )

    unchanged = _build(model, images_dir, index_dir, recursive=True)
    assert (
        unchanged.added,
        unchanged.updated,
        unchanged.removed,
        unchanged.unchanged,
    ) == (
        0,
        0,
        0,
        3,
    )
    assert model.encoded_image_count == 3
    assert first.index_path.read_bytes() == original_archive

    _write_image(images_dir / "a.png", (0, 0, 255), size=(16, 16))
    (images_dir / "b.jpg").unlink()
    _write_image(images_dir / "nested" / "d.png", (255, 0, 0))
    updated = _build(model, images_dir, index_dir, recursive=True)

    assert (updated.added, updated.updated, updated.removed, updated.unchanged) == (
        1,
        1,
        1,
        1,
    )
    assert model.encoded_image_count == 5
    reloaded = _load_index(
        index_dir, expected_root=images_dir.resolve(), expected_precision="fp32"
    )
    assert [record.relative_path for record in reloaded.records] == [
        "a.png",
        "nested/c.webp",
        "nested/d.png",
    ]


@pytest.mark.parametrize("embedding_dim", [128, 256, 512, 768])
def test_index_builds_and_searches_at_requested_dimension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    embedding_dim: int,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "red.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    model = FakeImageSearchModel()

    _build(model, images_dir, index_dir, embedding_dim=embedding_dim)
    loaded = _load_index(index_dir)
    monkeypatch.setattr(
        image_search,
        "get_tokenizer",
        lambda *, prompt_name: FakeTokenizer(),
    )
    results = search_images(model, index_dir, "red")

    assert loaded.embeddings.shape == (1, embedding_dim)
    assert results[0].path.name == "red.png"
    assert results[0].score == pytest.approx(1.0)
    with np.load(index_dir / "index.npz", allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
    assert metadata["format_version"] == 2
    assert metadata["embedding_dimension"] == embedding_dim
    assert metadata["embedding_strategy"] == "leading-dimensions-l2-v1"


def test_dino_index_uses_native_dimension_and_unprompted_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "red.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    model = FakeDinoImageSearchModel()

    update = build_or_update_image_index(
        model,
        _preprocess,
        images_dir,
        index_dir,
    )
    loaded = _load_index(index_dir)
    monkeypatch.setattr(
        image_search,
        "get_tokenizer",
        lambda **_: pytest.fail("dino.txt search must use its backend tokenizer."),
    )
    results = search_images(model, index_dir, "red object")

    assert update.added == 1
    assert loaded.embeddings.shape == (1, 2048)
    assert loaded.embedding_space.embedding_strategy == (
        "dinotxt-global-cls-plus-mean-patches-v1"
    )
    assert model.received_prompt_names == [None]
    assert model.tokenizer.received_texts == ["red object"]
    assert results[0].score == pytest.approx(1.0)


def test_dino_image_index_rejects_mrl_dimensions(tmp_path: Path) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "red.png", (255, 0, 0))

    with pytest.raises(ValueError, match="does not support MRL"):
        build_or_update_image_index(
            FakeDinoImageSearchModel(),
            _preprocess,
            images_dir,
            tmp_path / "index",
            embedding_dim=768,
        )


def test_search_rejects_another_registered_embedding_space(tmp_path: Path) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "red.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    build_or_update_image_index(
        FakeDinoImageSearchModel(),
        _preprocess,
        images_dir,
        index_dir,
    )

    with pytest.raises(ImageIndexCompatibilityError, match="embedding-space"):
        search_images(FakeImageSearchModel(), index_dir, "red object")


@pytest.mark.parametrize(
    "identity_change",
    [
        {"model_id": "different/model"},
        {"model_revision": "unregistered-revision"},
    ],
)
def test_search_rejects_different_dinotxt_checkpoint_identity(
    tmp_path: Path,
    identity_change: dict[str, str],
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "red.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    build_or_update_image_index(
        FakeDinoImageSearchModel(),
        _preprocess,
        images_dir,
        index_dir,
    )
    changed_metadata = replace(
        DinoV2DinoTxtBackend.embedding_space_metadata,
        **identity_change,
    )
    model = FakeDinoImageSearchModel(embedding_space_metadata=changed_metadata)

    with pytest.raises(ImageIndexCompatibilityError, match="embedding space"):
        search_images(model, index_dir, "red object")


@pytest.mark.parametrize("strategy", [None, "unsupported-strategy"])
def test_current_image_index_requires_supported_embedding_strategy(
    tmp_path: Path,
    strategy: str | None,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "red.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    _build(FakeImageSearchModel(), images_dir, index_dir)
    if strategy is None:
        _rewrite_index_metadata(
            index_dir / "index.npz",
            {},
            remove=("embedding_strategy",),
        )
    else:
        _rewrite_index_metadata(
            index_dir / "index.npz",
            {"embedding_strategy": strategy},
        )

    with pytest.raises(ImageIndexCompatibilityError, match="embedding_strategy"):
        _load_index(index_dir)


def test_legacy_768_index_remains_readable_and_can_be_converted(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "red.png", (255, 0, 0))
    source_dir = tmp_path / "legacy-index"
    model = FakeImageSearchModel()
    _build(model, images_dir, source_dir)
    _rewrite_index_version(source_dir / "index.npz", 1)
    encoded_count = model.encoded_image_count

    assert _load_index(source_dir).embeddings.shape == (1, 768)
    converted_path = convert_image_index(
        source_dir,
        tmp_path / "converted-index",
        embedding_dim=128,
    )
    converted = _load_index(converted_path.parent)

    assert converted_path.exists()
    assert converted.embeddings.shape == (1, 128)
    with np.load(converted_path, allow_pickle=False) as archive:
        converted_metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
    assert converted_metadata["embedding_strategy"] == "leading-dimensions-l2-v1"
    np.testing.assert_allclose(np.linalg.norm(converted.embeddings, axis=1), 1.0)
    assert model.encoded_image_count == encoded_count
    assert _load_index(source_dir).embeddings.shape == (1, 768)


def test_legacy_image_index_requires_native_dimension(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "red.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    _build(
        FakeImageSearchModel(),
        images_dir,
        index_dir,
        embedding_dim=128,
    )
    _rewrite_index_version(index_dir / "index.npz", 1)

    with pytest.raises(ImageIndexCompatibilityError, match="Legacy image indexes"):
        _load_index(index_dir)


def test_image_update_rejects_mixed_embedding_dimensions(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "red.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    model = FakeImageSearchModel()
    _build(model, images_dir, index_dir, embedding_dim=256)

    with pytest.raises(
        ImageIndexCompatibilityError, match="does not match the requested"
    ):
        _build(model, images_dir, index_dir, embedding_dim=128)


def test_image_index_conversion_rejects_expansion_and_existing_output(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "red.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    _build(FakeImageSearchModel(), images_dir, index_dir, embedding_dim=128)

    with pytest.raises(ValueError, match="Cannot expand"):
        convert_image_index(index_dir, tmp_path / "larger", embedding_dim=256)
    existing_output = tmp_path / "existing"
    existing_output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        convert_image_index(index_dir, existing_output, embedding_dim=128)


def test_image_index_conversion_rejects_output_created_after_precheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "red.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    _build(FakeImageSearchModel(), images_dir, index_dir, embedding_dim=128)
    destination = tmp_path / "raced-output"
    original_mkdir = Path.mkdir
    competing_output_created = False

    def create_competing_output(
        path: Path,
        *,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        nonlocal competing_output_created
        if path == destination and not competing_output_created:
            competing_output_created = True
            original_mkdir(path, parents=parents, exist_ok=True)
        original_mkdir(path, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", create_competing_output)
    with pytest.raises(ValueError, match="already exists"):
        convert_image_index(index_dir, destination, embedding_dim=128)

    assert competing_output_created
    assert not (destination / "index.npz").exists()


def test_non_recursive_index_excludes_nested_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "top.png", (255, 0, 0))
    _write_image(images_dir / "nested" / "inside.png", (0, 255, 0))

    update = _build(FakeImageSearchModel(), images_dir, tmp_path / "index")

    assert update.added == 1
    assert [
        record.relative_path for record in _load_index(tmp_path / "index").records
    ] == ["top.png"]


def test_empty_directory_is_rejected_for_new_index(tmp_path: Path) -> None:
    images_dir = tmp_path / "empty"
    images_dir.mkdir()

    with pytest.raises(ValueError, match="No supported images"):
        _build(FakeImageSearchModel(), images_dir, tmp_path / "index")


def test_deleting_every_image_writes_an_empty_index(tmp_path: Path) -> None:
    images_dir = tmp_path / "photos"
    image_path = _write_image(images_dir / "only.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    model = FakeImageSearchModel()
    _build(model, images_dir, index_dir)
    image_path.unlink()

    update = _build(model, images_dir, index_dir)

    assert update.removed == 1
    assert update.index_path.exists()
    assert _load_index(index_dir).embeddings.shape == (0, 768)
    with pytest.raises(ValueError, match="index is empty"):
        _search_index(model, index_dir, "赤い画像")


def test_corrupt_image_does_not_replace_existing_index(tmp_path: Path) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "valid.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    _build(FakeImageSearchModel(), images_dir, index_dir)
    index_path = index_dir / "index.npz"
    original_archive = index_path.read_bytes()
    (images_dir / "broken.png").write_bytes(b"not an image")

    with pytest.raises(ValueError, match="input index"):
        _build(FakeImageSearchModel(), images_dir, index_dir)

    assert index_path.read_bytes() == original_archive
    assert len(_load_index(index_dir).records) == 1


def test_search_uses_search_query_prompt_and_stable_top_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "red-b.png", (255, 0, 0))
    _write_image(images_dir / "red-a.png", (255, 0, 0))
    _write_image(images_dir / "green.png", (0, 255, 0))
    index_dir = tmp_path / "index"
    model = FakeImageSearchModel()
    _build(model, images_dir, index_dir)
    tokenizer = FakeTokenizer()
    prompts: list[str | None] = []

    def get_fake_tokenizer(*, prompt_name: str | None = None) -> FakeTokenizer:
        prompts.append(prompt_name)
        return tokenizer

    monkeypatch.setattr(image_search, "get_tokenizer", get_fake_tokenizer)
    results = search_images(model, index_dir, "赤い画像", top_k=2)

    assert prompts == ["SearchQuery"]
    assert tokenizer.received_texts == ["赤い画像"]
    assert [result.path.name for result in results] == ["red-a.png", "red-b.png"]
    assert [result.score for result in results] == pytest.approx([1.0, 1.0])
    assert len(model.received_tokens) == 1


def test_search_returns_all_results_when_top_k_exceeds_index_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "one.png", (255, 0, 0))
    _write_image(images_dir / "two.png", (0, 255, 0))
    index_dir = tmp_path / "index"
    model = FakeImageSearchModel()
    _build(model, images_dir, index_dir)
    monkeypatch.setattr(
        image_search,
        "get_tokenizer",
        lambda *, prompt_name: FakeTokenizer(),
    )

    assert len(_search_index(model, index_dir, "image", top_k=10)) == 2


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_non_positive_top_k(top_k: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        _search_index(
            FakeImageSearchModel(), tmp_path / "missing", "query", top_k=top_k
        )


def test_search_rejects_empty_query(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        _search_index(FakeImageSearchModel(), tmp_path / "missing", "   ")


def test_update_rejects_incompatible_model_revision(tmp_path: Path) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "one.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    _build(FakeImageSearchModel(), images_dir, index_dir)
    metadata = replace(
        EmbeddingGemma2Backend.embedding_space_metadata,
        model_revision="different-revision",
    )

    with pytest.raises(ImageIndexCompatibilityError, match="embedding space"):
        _build(
            FakeImageSearchModel(embedding_space_metadata=metadata),
            images_dir,
            index_dir,
        )


def test_update_rejects_incompatible_preprocessing(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "one.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    _build(FakeImageSearchModel(), images_dir, index_dir)
    metadata = replace(
        EmbeddingGemma2Backend.embedding_space_metadata,
        preprocessing_id="different-preprocess",
    )

    with pytest.raises(ImageIndexCompatibilityError, match="embedding space"):
        _build(
            FakeImageSearchModel(embedding_space_metadata=metadata),
            images_dir,
            index_dir,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_id", "different-backend"),
        ("model_id", "different/model"),
        ("normalization", "different-normalization"),
        ("query_prompt", "different query prefix"),
    ],
)
def test_update_rejects_equal_dimension_from_incompatible_embedding_space(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "one.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    _build(FakeImageSearchModel(), images_dir, index_dir)
    metadata = replace(
        EmbeddingGemma2Backend.embedding_space_metadata,
        **{field: value},
    )

    with pytest.raises(ImageIndexCompatibilityError, match="embedding space"):
        _build(
            FakeImageSearchModel(embedding_space_metadata=metadata),
            images_dir,
            index_dir,
        )


def test_loader_rejects_different_model_precision(tmp_path: Path) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "one.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    _build(FakeImageSearchModel(), images_dir, index_dir)

    with pytest.raises(ImageIndexCompatibilityError, match="model precision"):
        _load_index(index_dir, expected_precision="bf16")


def test_loader_rejects_corrupt_and_pickle_archives(tmp_path: Path) -> None:
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "index.npz").write_bytes(b"not an npz archive")
    with pytest.raises(ImageIndexError, match="Could not read image index"):
        _load_index(corrupt_dir)

    pickle_dir = tmp_path / "pickle"
    pickle_dir.mkdir()
    np.savez(
        pickle_dir / "index.npz",
        metadata=np.array([{"unsafe": True}], dtype=object),
        embeddings=np.zeros((0, 768), dtype=np.float32),
    )
    with pytest.raises(ImageIndexError):
        _load_index(pickle_dir)
