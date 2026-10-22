from __future__ import annotations

import hashlib
import io
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from torch import nn

from openvlclip import (
    create_model,
    create_model_and_transforms,
    create_model_from_pretrained,
    get_tokenizer,
    list_models,
    list_pretrained,
)
from openvlclip.backends import dinov2_dinotxt as backend_module
from openvlclip.backends.dinov2_dinotxt import (
    DINO_HUB_MODEL_ID,
    DINO_MODEL_NAME,
    DINO_MODEL_REVISION,
    DINO_SOURCE_REVISION,
    DinoTxtCLIP,
    DinoTxtPreprocess,
    DinoV2DinoTxtBackend,
    _Artifact,
)


class FakeDinoTxt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        del normalize
        return self.anchor.expand(image.shape[0], 2048)

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        del normalize
        return self.anchor.expand(text.shape[0], 2048)


def _mock_hub_loader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cache_dir: Path | None = None,
) -> tuple[FakeDinoTxt, list[tuple[Path, _Artifact]], Mock]:
    fake_model = FakeDinoTxt()
    verified: list[tuple[Path, _Artifact]] = []
    hub_loader = Mock(return_value=fake_model)

    def verify_artifact(hub_dir: Path, artifact: _Artifact) -> Path:
        verified.append((hub_dir, artifact))
        return hub_dir / "checkpoints" / artifact.filename

    monkeypatch.setattr(backend_module, "_ensure_verified_artifact", verify_artifact)
    monkeypatch.setattr(torch.hub, "load", hub_loader)
    return fake_model, verified, hub_loader


def test_registry_and_listing_are_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub_loader = Mock(side_effect=AssertionError("the Hub must not load"))
    monkeypatch.setattr(torch.hub, "load", hub_loader)

    assert list_models() == ["embeddinggemma-2", DINO_MODEL_NAME]
    assert list_pretrained() == [
        ("embeddinggemma-2", "official"),
        (DINO_MODEL_NAME, "official"),
    ]
    assert list_pretrained(as_str=True) == [
        "embeddinggemma-2:official",
        f"{DINO_MODEL_NAME}:official",
    ]
    hub_loader.assert_not_called()


def test_factory_loads_only_the_pinned_model_and_verified_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_model, verified, hub_loader = _mock_hub_loader(monkeypatch)
    original_hub_dir = torch.hub.get_dir()
    cache_dir = tmp_path / "dino-cache"

    model = create_model(
        DINO_MODEL_NAME,
        device="cpu",
        precision="fp32",
        cache_dir=str(cache_dir),
    )

    assert isinstance(model, DinoTxtCLIP)
    assert model.backend is fake_model
    assert model.embedding_space_metadata.model_revision == DINO_MODEL_REVISION
    assert model.capabilities["image_embedding"] is True
    assert model.capabilities["mrl"] is False
    assert model.capabilities["selective_loading"] is False
    assert torch.hub.get_dir() == original_hub_dir
    assert {path for path, _ in verified} == {cache_dir.resolve()}
    assert [artifact.filename for _, artifact in verified] == [
        "dinov2_vitl14_reg4_pretrain.pth",
        "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
        "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth",
    ]
    hub_loader.assert_called_once_with(
        f"facebookresearch/dinov2:{DINO_SOURCE_REVISION}",
        "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
        source="github",
        trust_repo=True,
    )


def test_all_factory_variants_return_the_dinotxt_preprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_model, _, _ = _mock_hub_loader(monkeypatch)
    model, train_preprocess, validation_preprocess = create_model_and_transforms(
        DINO_MODEL_NAME,
        pretrained="official",
        cache_dir=str(tmp_path / "factory-cache"),
    )
    assert isinstance(model, DinoTxtCLIP)
    assert model.backend is fake_model
    assert isinstance(train_preprocess, DinoTxtPreprocess)
    assert train_preprocess is validation_preprocess

    fake_model, _, _ = _mock_hub_loader(monkeypatch)
    model_and_preprocess = create_model_from_pretrained(
        DINO_MODEL_NAME,
        pretrained="official",
        cache_dir=str(tmp_path / "pretrained-cache"),
    )
    assert isinstance(model_and_preprocess, tuple)
    model, preprocess = model_and_preprocess
    assert isinstance(model, DinoTxtCLIP)
    assert model.backend is fake_model
    assert isinstance(preprocess, DinoTxtPreprocess)

    fake_model, _, _ = _mock_hub_loader(monkeypatch)
    model_only = create_model_from_pretrained(
        DINO_MODEL_NAME,
        pretrained="official",
        cache_dir=str(tmp_path / "model-only-cache"),
        return_transform=False,
    )
    assert isinstance(model_only, DinoTxtCLIP)
    assert model_only.backend is fake_model


def test_dinotxt_rejects_unsupported_checkpoints_prompts_and_selective_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_loader = Mock(side_effect=AssertionError("artifact access is unsupported"))
    monkeypatch.setattr(backend_module, "_ensure_verified_artifact", artifact_loader)
    hub_loader = Mock(side_effect=AssertionError("the Hub must not load"))
    monkeypatch.setattr(torch.hub, "load", hub_loader)

    with pytest.raises(ValueError, match="Unsupported pretrained checkpoint"):
        create_model(DINO_MODEL_NAME, pretrained="another/model")
    with pytest.raises(ValueError, match="pretrained=None"):
        create_model(DINO_MODEL_NAME, pretrained=None)
    with pytest.raises(NotImplementedError, match="selective loading"):
        create_model(DINO_MODEL_NAME, modalities=("text",))
    with pytest.raises(ValueError, match="dino.txt has no task prompts"):
        get_tokenizer(DINO_MODEL_NAME, prompt_name="SearchQuery")
    with pytest.raises(ValueError, match="between 2 and 77"):
        get_tokenizer(DINO_MODEL_NAME, context_length=78)

    artifact_loader.assert_not_called()
    hub_loader.assert_not_called()


@pytest.mark.parametrize("dimension", [128, 256, 512, 768, 1024])
def test_dinotxt_only_accepts_native_embedding_dimension(dimension: int) -> None:
    backend = DinoV2DinoTxtBackend()

    assert backend.validate_embedding_dimension(None, name="truncate_dim") == 2048
    assert backend.validate_embedding_dimension(2048, name="truncate_dim") == 2048
    with pytest.raises(ValueError, match="does not support MRL or truncation"):
        backend.validate_embedding_dimension(dimension, name="truncate_dim")


def test_dinotxt_encoders_preserve_official_raw_and_normalized_features() -> None:
    backend = DinoV2DinoTxtBackend()
    model = DinoTxtCLIP(
        FakeDinoTxt(),
        backend,
        output_dict=False,
        logit_scale_mode="strict",
    )
    image = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    tokens = torch.tensor(
        [[49406, 320, 49407, 0], [49406, 49407, 0, 0]],
        dtype=torch.long,
    )

    raw_image = model.encode_image(image)
    raw_text = model.encode_text(tokens)
    normalized_image = model.encode_image(image, normalize=True)
    normalized_text = model.encode_text(tokens, normalize=True)

    assert raw_image.shape == raw_text.shape == (2, 2048)
    assert torch.equal(raw_image, torch.ones_like(raw_image))
    assert torch.equal(raw_text, torch.ones_like(raw_text))
    torch.testing.assert_close(
        torch.linalg.vector_norm(normalized_image, dim=-1),
        torch.ones(2),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(normalized_text, dim=-1),
        torch.ones(2),
    )
    with pytest.raises(ValueError, match="does not support MRL or truncation"):
        model.encode_image(image, truncate_dim=768)


def test_dinotxt_encoders_reject_invalid_image_and_token_inputs() -> None:
    model = DinoTxtCLIP(
        FakeDinoTxt(),
        DinoV2DinoTxtBackend(),
        output_dict=False,
        logit_scale_mode="strict",
    )

    with pytest.raises(TypeError, match="torch.float32"):
        model.encode_image(torch.zeros((1, 3, 224, 224), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="multiples of patch_size"):
        model.encode_image(torch.zeros((1, 3, 223, 224), dtype=torch.float32))
    with pytest.raises(ValueError, match="start with SOT"):
        model.encode_text(torch.tensor([[49407, 0]], dtype=torch.long))
    with pytest.raises(ValueError, match="contain EOT"):
        model.encode_text(torch.tensor([[49406, 123]], dtype=torch.long))
    with pytest.raises(ValueError, match="after the final EOT"):
        model.encode_text(torch.tensor([[49406, 49407, 123]], dtype=torch.long))


def test_dinotxt_preprocess_returns_rgb_imagenet_tensor() -> None:
    preprocess = DinoTxtPreprocess()
    image = Image.new("L", (320, 256), color=127)

    pixels = preprocess(image)

    assert pixels.shape == (3, 224, 224)
    assert pixels.dtype == torch.float32
    assert torch.isfinite(pixels).all()


def test_verified_artifact_download_is_atomic_and_cache_is_checked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"official artifact bytes"
    artifact = _Artifact(
        filename="small-test.bin",
        url="https://example.invalid/small-test.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    response = io.BytesIO(payload)
    monkeypatch.setattr(
        backend_module.urllib.request,
        "urlopen",
        lambda url, timeout: response,
    )

    downloaded = backend_module._ensure_verified_artifact(tmp_path, artifact)
    assert downloaded.read_bytes() == payload
    assert backend_module._ensure_verified_artifact(tmp_path, artifact) == downloaded

    downloaded.write_bytes(b"corrupt cache")
    with pytest.raises(RuntimeError, match="Cached dino.txt artifact"):
        backend_module._ensure_verified_artifact(tmp_path, artifact)


def test_wrong_embedding_revision_is_not_registered() -> None:
    registry = backend_module.DinoV2DinoTxtBackend().embedding_space_metadata

    assert registry.model_id == DINO_HUB_MODEL_ID
    assert registry.model_revision == DINO_MODEL_REVISION
