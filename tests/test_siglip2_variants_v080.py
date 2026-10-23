"""Lightweight v0.8.0 regression checks for additional pinned SigLIP 2 checkpoints."""

from __future__ import annotations

from math import log
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from torch import nn

from openvlclip import create_model, create_model_and_transforms, list_models
from openvlclip.backends import siglip2
from openvlclip.image_search import (
    ImageIndexCompatibilityError,
    build_or_update_image_index,
    search_images,
)
from openvlclip.registry import get_default_registry

_CASES = (
    ("siglip2-base-384", "google/siglip2-base-patch16-384",
     "f775b65a79762255128c981547af89addcfe0f88", 768, 384),
    ("siglip2-large-256", "google/siglip2-large-patch16-256",
     "787800c8990e6f058423089178e718139608408c", 1024, 256),
    ("siglip2-so400m-384", "google/siglip2-so400m-patch14-384",
     "e8e487298228002f3d8a82e0cd5c8ea9c567f57f", 1152, 384),
)


class _Model(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.logit_scale = nn.Parameter(torch.tensor(log(2.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-0.25))
        self.dimension = dimension
        self.config = SimpleNamespace(model_type="siglip")
        self.last_pixels = None

    def get_image_features(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.last_pixels = pixel_values
        return SimpleNamespace(pooler_output=torch.ones((pixel_values.shape[0], self.dimension)))

    def get_text_features(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        assert attention_mask.shape == input_ids.shape
        return SimpleNamespace(pooler_output=torch.ones((input_ids.shape[0], self.dimension)))


class _Processor:
    def __init__(self, size: int) -> None:
        self.size = size
        self.image_processor = SimpleNamespace()
        self.tokenizer = SimpleNamespace(pad_token_id=0)

    def __call__(self, *, images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert images.mode == "RGB"
        return {"pixel_values": torch.ones((1, 3, self.size, self.size))}


@pytest.mark.parametrize(("name", "checkpoint", "revision", "dimension", "size"), _CASES)
def test_new_fixres_models_keep_official_identity_and_native_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    name: str, checkpoint: str, revision: str, dimension: int, size: int
) -> None:
    descriptor = get_default_registry().resolve(name)
    assert descriptor.model_id == checkpoint
    assert descriptor.revision == revision
    assert descriptor.backend.native_dimension == dimension
    assert descriptor.backend.image_size == size
    assert descriptor.capabilities["variable_resolution"] is False
    assert descriptor.capabilities["mrl"] is False
    assert name in list_models()

    fake = _Model(dimension)
    model_loader = Mock(return_value=fake)
    proc_loader = Mock(return_value=_Processor(size))
    monkeypatch.setattr(siglip2.AutoModel, "from_pretrained", model_loader)
    monkeypatch.setattr(siglip2.AutoProcessor, "from_pretrained", proc_loader)

    model, preprocess_train, preprocess_val = create_model_and_transforms(name)
    assert preprocess_train is preprocess_val
    pixels = preprocess_val(Image.new("RGB", (75, 45), "red"))
    assert tuple(pixels.shape) == (3, size, size)
    assert model.validate_embedding_dimension(None, name="embedding_dim") == dimension

    features = model.encode_image(pixels, normalize=True)
    text_features = model.encode_text(torch.tensor([[2, 3, 0]]), normalize=True)
    assert features.shape == (1, dimension)
    assert text_features.shape == (1, dimension)
    logits_image, logits_text = model.get_logits(pixels, torch.tensor([[2, 3, 0]]))
    assert logits_image.shape == logits_text.shape == (1, 1)
    torch.testing.assert_close(logits_image, torch.tensor([[1.75]]), atol=1e-5, rtol=1e-5)
    assert fake.last_pixels.shape == (1, 3, size, size)

    model_loader.assert_called_once_with(
        checkpoint, revision=revision, cache_dir=None,
        dtype=torch.float32, use_safetensors=True
    )
    proc_loader.assert_called_once_with(checkpoint, revision=revision, cache_dir=None)
    with pytest.raises(ValueError, match="does not support MRL or truncation"):
        model.validate_embedding_dimension(128, name="embedding_dim")


def test_preprocessing_and_embedding_spaces_distinguish_the_checkpoints() -> None:
    registry = get_default_registry()
    old_base = registry.resolve("siglip2-base").embedding_space_metadata
    other = [
        registry.resolve(name).embedding_space_metadata
        for name, *_ in _CASES
    ]
    assert old_base.preprocessing_id != other[0].preprocessing_id
    assert all(old_base.model_id != item.model_id for item in other)
    assert len({item.model_id for item in other}) == len(other)
    assert all(item.backend_id == siglip2.SIGLIP2_BACKEND_ID for item in other)


def test_old_base_and_naflex_dimensions_still_unchanged() -> None:
    registry = get_default_registry()
    assert registry.resolve("siglip2-base").backend.validate_embedding_dimension(
        None, name="embedding_dim"
    ) == 768
    assert registry.resolve("siglip2-naflex-base").backend.validate_embedding_dimension(
        None, name="embedding_dim"
    ) == 768


def test_new_backend_rejects_incorrect_projection_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Model(768)
    monkeypatch.setattr(siglip2.AutoModel, "from_pretrained", Mock(return_value=fake))
    monkeypatch.setattr(
        siglip2.AutoProcessor, "from_pretrained", Mock(return_value=_Processor(256))
    )
    model = create_model("siglip2-large-256")
    with pytest.raises(RuntimeError, match="expected"):
        model.encode_image(torch.ones((3, 256, 256)))


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        assert kwargs["max_length"] == 64
        token_ids = torch.zeros((len(texts), 64), dtype=torch.long)
        token_ids[:, :2] = torch.tensor([2, 3])
        return {"input_ids": token_ids, "attention_mask": token_ids.ne(0).long()}


@pytest.mark.parametrize(("name", "checkpoint", "revision", "dimension", "size"), _CASES)
def test_new_model_index_build_search_and_model_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str, checkpoint: str, revision: str, dimension: int, size: int
) -> None:
    del checkpoint, revision
    model_loader = Mock(return_value=_Model(dimension))
    processor_loader = Mock(return_value=_Processor(size))
    monkeypatch.setattr(siglip2.AutoModel, "from_pretrained", model_loader)
    monkeypatch.setattr(siglip2.AutoProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(
        siglip2.AutoTokenizer, "from_pretrained", Mock(return_value=_Tokenizer())
    )

    model, _, preprocess = create_model_and_transforms(name)
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (42, 68), "red").save(images / "red.png")
    index_dir = tmp_path / "index"
    result = build_or_update_image_index(model, preprocess, images, index_dir)
    assert result.added == 1
    hits = search_images(model, index_dir, "red object", top_k=1)
    assert [hit.path.name for hit in hits] == ["red.png"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)

    model_loader.return_value = _Model(768)
    processor_loader.return_value = _Processor(224)
    old_model = create_model("siglip2-base")
    with pytest.raises(ImageIndexCompatibilityError, match="embedding space"):
        search_images(old_model, index_dir, "red object")
