from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from openclip_gemma import image_search
from openclip_gemma.image_search import (
    ImageIndexCompatibilityError,
    ImageIndexError,
    _build_or_update_image_index,
    _load_index,
    _search_index,
)


class FakeImageSearchModel(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=dtype))
        self.encoded_image_count = 0
        self.received_tokens: list[torch.Tensor] = []

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        self.encoded_image_count += images.shape[0]
        features = images.mean(dim=(2, 3))
        features = torch.nn.functional.normalize(features, dim=-1)
        return torch.nn.functional.pad(features, (0, 765))

    def encode_text(
        self,
        tokens: torch.Tensor,
        *,
        normalize: bool = False,
    ) -> torch.Tensor:
        self.received_tokens.append(tokens.detach().cpu())
        features = torch.zeros(
            (tokens.shape[0], 768),
            device=tokens.device,
            dtype=self.anchor.dtype,
        )
        features[:, 0] = 1
        return torch.nn.functional.normalize(features, dim=-1) if normalize else features


class FakeTokenizer:
    def __init__(self) -> None:
        self.received_texts: list[str] = []

    def __call__(self, text: str) -> torch.Tensor:
        self.received_texts.append(text)
        return torch.tensor([[1]], dtype=torch.long)


def _preprocess(image: Image.Image) -> torch.Tensor:
    rgb = image.convert("RGB")
    return torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy()).permute(2, 0, 1) / 255


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
):
    return _build_or_update_image_index(
        model,
        _preprocess,
        images_dir,
        index_dir,
        batch_size=batch_size,
        recursive=recursive,
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
    loaded = _load_index(index_dir, expected_root=images_dir.resolve(), expected_precision="fp32")
    assert [record.relative_path for record in loaded.records] == [
        "a.png",
        "b.jpg",
        "nested/c.webp",
    ]
    assert loaded.embeddings.shape == (3, 768)
    assert loaded.embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(loaded.embeddings, axis=1), 1.0, atol=1e-6)

    unchanged = _build(model, images_dir, index_dir, recursive=True)
    assert (unchanged.added, unchanged.updated, unchanged.removed, unchanged.unchanged) == (
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
    reloaded = _load_index(index_dir, expected_root=images_dir.resolve(), expected_precision="fp32")
    assert [record.relative_path for record in reloaded.records] == [
        "a.png",
        "nested/c.webp",
        "nested/d.png",
    ]


def test_non_recursive_index_excludes_nested_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "top.png", (255, 0, 0))
    _write_image(images_dir / "nested" / "inside.png", (0, 255, 0))

    update = _build(FakeImageSearchModel(), images_dir, tmp_path / "index")

    assert update.added == 1
    assert [record.relative_path for record in _load_index(tmp_path / "index").records] == [
        "top.png"
    ]


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
    results = _search_index(model, index_dir, "赤い画像", top_k=2)

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
        _search_index(FakeImageSearchModel(), tmp_path / "missing", "query", top_k=top_k)


def test_search_rejects_empty_query(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        _search_index(FakeImageSearchModel(), tmp_path / "missing", "   ")


def test_loader_rejects_incompatible_model_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "one.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    _build(FakeImageSearchModel(), images_dir, index_dir)
    monkeypatch.setattr(image_search, "MODEL_REVISION", "different-revision")

    with pytest.raises(ImageIndexCompatibilityError, match="model_revision"):
        _load_index(index_dir)


def test_loader_rejects_incompatible_preprocessing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images_dir = tmp_path / "photos"
    _write_image(images_dir / "one.png", (255, 0, 0))
    index_dir = tmp_path / "index"
    _build(FakeImageSearchModel(), images_dir, index_dir)
    monkeypatch.setattr(image_search, "_PREPROCESSING_ID", "different-preprocess")

    with pytest.raises(ImageIndexCompatibilityError, match="preprocessing_id"):
        _load_index(index_dir)


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
