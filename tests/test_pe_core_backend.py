"""Lightweight PE-Core T/S integration with OpenVLCLIP Factory (no checkpoint downloads)."""

from __future__ import annotations

import json
import sys
from math import log
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from torch import nn

from openvlclip import create_model_and_transforms, get_tokenizer, list_models
from openvlclip.backends import pe_core
from openvlclip.registry import get_default_registry

_CASES = (
    ("pe-core-t16-384", "timm/PE-Core-T-16-384", "7fe539e"),
    ("pe-core-s16-384", "timm/PE-Core-S-16-384", "3249a38"),
)


class FakePE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.logit_scale = nn.Parameter(torch.tensor(log(2.0)))

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = torch.ones((image.shape[0], 512), device=image.device)
        return torch.nn.functional.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = torch.ones((text.shape[0], 512), device=text.device)
        return torch.nn.functional.normalize(features, dim=-1) if normalize else features


def _mock_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    (tmp_path / "open_clip_config.json").write_text(
        json.dumps({"model_cfg": {"embed_dim": 512, "vision_cfg": {"image_size": 384},
                                  "text_cfg": {"context_length": 32}}}),
        encoding="utf-8",
    )
    (tmp_path / "open_clip_model.safetensors").write_bytes(b"fake")
    monkeypatch.setattr(pe_core, "_snapshot", Mock(return_value=tmp_path))

    def preprocess(image: Image.Image) -> torch.Tensor:
        assert image.mode == "RGB"
        return torch.ones((3, 384, 384))

    model_loader = Mock(return_value=(FakePE(), None, preprocess))
    tokenizer = Mock(return_value=torch.ones((1, 32), dtype=torch.long))
    get_tokenizer_mock = Mock(return_value=tokenizer)
    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        SimpleNamespace(
            create_model_and_transforms=model_loader, get_tokenizer=get_tokenizer_mock
        ),
    )
    return model_loader, get_tokenizer_mock


@pytest.mark.parametrize(("name", "checkpoint", "revision"), _CASES)
def test_model_factory_embeddings_logits_and_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    name: str, checkpoint: str, revision: str,
) -> None:
    model_loader, tokenizer_loader = _mock_loading(tmp_path, monkeypatch)
    descriptor = get_default_registry().resolve(name)
    assert descriptor.model_id == checkpoint
    assert descriptor.revision == revision
    assert descriptor.capabilities["video_embedding"] is False
    assert descriptor.capabilities["image_text_similarity"] is True
    assert name in list_models()

    model, prep_train, prep_val = create_model_and_transforms(name)
    assert prep_train is prep_val
    pixels = prep_val(Image.new("RGBA", (64, 96), "red"))
    assert pixels.shape == (3, 384, 384)
    image_embedding = model.encode_image(pixels, normalize=True)
    tokens = torch.ones((1, 32), dtype=torch.long)
    text_embedding = model.encode_text(tokens, normalize=True)
    assert image_embedding.shape == (1, 512)
    assert text_embedding.shape == (1, 512)
    img_logits, txt_logits = model.get_logits(pixels, tokens)
    torch.testing.assert_close(img_logits, torch.full((1, 1), 2.0))
    torch.testing.assert_close(txt_logits, img_logits)
    features = model(pixels, tokens)
    assert len(features) == 3
    torch.testing.assert_close(features[2], torch.tensor(2.0))
    assert model.validate_embedding_dimension(None, name="embedding_dim") == 512
    with pytest.raises(ValueError, match="512"):
        model.validate_embedding_dimension(256, name="embedding_dim")

    model_loader.assert_called_once_with(
        f"local-dir:{tmp_path}", device="cpu", precision="fp32"
    )
    tokenize = get_tokenizer(name)
    assert tokenize("a cat").shape == (1, 32)
    tokenizer_loader.assert_called_once_with(f"local-dir:{tmp_path}")


def test_pe_core_model_spaces_are_distinct_and_list_is_side_effect_free() -> None:
    t = get_default_registry().resolve("pe-core-t16-384")
    s = get_default_registry().resolve("pe-core-s16-384")
    assert t.embedding_space_metadata != s.embedding_space_metadata
    assert t.embedding_space_metadata.model_revision != s.embedding_space_metadata.model_revision


def test_pe_core_refuses_unsupported_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_loading(tmp_path, monkeypatch)
    model, _, _ = create_model_and_transforms("pe-core-t16-384")
    with pytest.raises(NotImplementedError):
        model.encode_video("video.mp4")
    with pytest.raises(NotImplementedError):
        model.encode_patches(torch.ones((3, 384, 384)))
    with pytest.raises(ValueError):
        get_tokenizer("pe-core-t16-384", prompt_name="query")
