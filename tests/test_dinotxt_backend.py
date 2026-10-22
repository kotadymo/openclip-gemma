from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

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
        self.logit_scale = nn.Parameter(torch.tensor([math.log(10.0)]))

    def encode_image(
        self, image: torch.Tensor, normalize: bool = False
    ) -> torch.Tensor:
        features = self.anchor.expand(image.shape[0], 2048)
        return (
            torch.nn.functional.normalize(features, dim=-1) if normalize else features
        )

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self.anchor.expand(text.shape[0], 2048)
        return (
            torch.nn.functional.normalize(features, dim=-1) if normalize else features
        )

    def get_visual_class_and_patch_tokens(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = image.shape
        grid_height = height // backend_module._PATCH_SIZE
        grid_width = width // backend_module._PATCH_SIZE
        patch_count = grid_height * grid_width
        patch_offsets = torch.arange(
            patch_count,
            device=image.device,
            dtype=torch.float32,
        ).view(1, patch_count, 1)
        channel_offsets = torch.arange(
            backend_module._PATCH_EMBEDDING_DIMENSION,
            device=image.device,
            dtype=torch.float32,
        ).view(1, 1, backend_module._PATCH_EMBEDDING_DIMENSION)
        patches = (
            0.25
            + patch_offsets / max(patch_count, 1)
            + channel_offsets / backend_module._PATCH_EMBEDDING_DIMENSION
        ).expand(batch_size, -1, -1)
        class_features = self.anchor.to(
            device=image.device,
            dtype=image.dtype,
        ).expand(batch_size, backend_module._PATCH_EMBEDDING_DIMENSION)
        return class_features, patches.to(dtype=image.dtype)

    def get_logits(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_features = self.encode_text(text, normalize=True)
        image_features = self.encode_image(image, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        return image_logits, image_logits.T

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_features = self.encode_text(text, normalize=True)
        image_features = self.encode_image(image, normalize=True)
        return image_features, text_features, self.logit_scale.exp()


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

    assert list_models() == [
        "embeddinggemma-2",
        DINO_MODEL_NAME,
        "siglip2-base",
        "siglip2-naflex-base",
        "qwen3-vl-embedding-2b",
        "qwen3-vl-embedding-8b",
    ]
    assert list_pretrained() == [
        ("embeddinggemma-2", "official"),
        (DINO_MODEL_NAME, "official"),
        ("siglip2-base", "official"),
        ("siglip2-naflex-base", "official"),
        ("qwen3-vl-embedding-2b", "official"),
        ("qwen3-vl-embedding-8b", "official"),
    ]
    assert list_pretrained(as_str=True) == [
        "embeddinggemma-2:official",
        f"{DINO_MODEL_NAME}:official",
        "siglip2-base:official",
        "siglip2-naflex-base:official",
        "qwen3-vl-embedding-2b:official",
        "qwen3-vl-embedding-8b:official",
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
    assert model.capabilities["patch_embedding"] is True
    assert model.capabilities["text_similarity_map"] is True
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


def test_dinotxt_encode_texts_uses_its_tokenizer_and_rejects_instructions() -> None:
    model = DinoTxtCLIP(
        FakeDinoTxt(),
        DinoV2DinoTxtBackend(),
        output_dict=False,
        logit_scale_mode="strict",
    )
    token_ids = torch.tensor([[49406, 320, 49407]], dtype=torch.long)
    tokenizer = Mock(return_value=token_ids)
    create_tokenizer = Mock(return_value=tokenizer)
    features = torch.ones((1, 2048))
    encoder = Mock(return_value=features)
    model._create_tokenizer = create_tokenizer  # type: ignore[method-assign]
    model.encode_text = encoder  # type: ignore[method-assign]

    result = model.encode_texts("a photo")

    assert result is features
    create_tokenizer.assert_called_once_with(prompt_name=None)
    tokenizer.assert_called_once_with("a photo")
    encoder.assert_called_once_with(
        token_ids,
        normalize=False,
        truncate_dim=None,
    )
    with pytest.raises(NotImplementedError, match="does not support task instructions"):
        model.encode_texts("a photo", instruction="describe this image")
    with pytest.raises(NotImplementedError, match="encode_multimodal"):
        model.encode_multimodal({"text": "a photo"})
    with pytest.raises(NotImplementedError, match="encode_video"):
        model.encode_video("video.mp4")


def test_dinotxt_rejects_processor_mapping_image_inputs() -> None:
    model = DinoTxtCLIP(
        FakeDinoTxt(),
        DinoV2DinoTxtBackend(),
        output_dict=False,
        logit_scale_mode="strict",
    )

    with pytest.raises(TypeError, match="processor-aware image mappings"):
        model.encode_image({"pixel_values": torch.zeros((1, 3, 224, 224))})


def test_dinotxt_forward_and_logits_distinguish_official_scale_from_cosine() -> None:
    backend = DinoV2DinoTxtBackend()
    official_model = FakeDinoTxt()
    model = DinoTxtCLIP(
        official_model,
        backend,
        output_dict=False,
        logit_scale_mode="strict",
    )
    image = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    tokens = torch.tensor(
        [
            [49406, 320, 49407, 0],
            [49406, 49407, 0, 0],
            [49406, 334, 49407, 0],
        ],
        dtype=torch.long,
    )

    expected_forward = official_model(image, tokens)
    actual_forward = model(image, tokens)
    expected_logits = official_model.get_logits(image, tokens)
    actual_logits = model.get_logits(image, tokens)

    for actual, expected in zip(actual_forward, expected_forward, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(actual_logits, expected_logits, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_logits[0], actual_logits[1].T, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_logits[0],
        actual_forward[2] * actual_forward[0] @ actual_forward[1].T,
        rtol=0,
        atol=0,
    )
    assert actual_forward[2].item() == pytest.approx(10.0)

    model.logit_scale_mode = "cosine"
    cosine_forward = model(image, tokens)
    cosine_logits = model.get_logits(image, tokens)
    expected_cosine_logits = (
        official_model.encode_image(image, normalize=True)
        @ official_model.encode_text(tokens, normalize=True).T
    )
    torch.testing.assert_close(
        cosine_logits[0],
        expected_cosine_logits,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(cosine_logits[1], cosine_logits[0].T, rtol=0, atol=0)
    assert cosine_forward[2].item() == 1.0

    single_modality = model(image=image)
    assert single_modality[0] is not None
    assert single_modality[1] is None
    assert single_modality[2].item() == 1.0

    model.output_dict = True
    dictionary_output = model(image, tokens)
    assert isinstance(dictionary_output, dict)
    assert dictionary_output["image_features"] is not None
    assert dictionary_output["text_features"] is not None
    assert dictionary_output["logit_scale"].item() == 1.0


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


def test_dinotxt_encode_patches_uses_head_tokens_and_handles_non_square_inputs() -> (
    None
):
    official_model = FakeDinoTxt()
    model = DinoTxtCLIP(
        official_model,
        DinoV2DinoTxtBackend(),
        output_dict=False,
        logit_scale_mode="strict",
    )
    image = torch.rand((2, 3, 224, 196), dtype=torch.float32).transpose(2, 3)
    assert not image.is_contiguous()

    _, official_patches = official_model.get_visual_class_and_patch_tokens(image)
    patches = model.encode_patches(image)
    normalized = model.encode_patches(image, normalize=True)

    assert patches.shape == (2, 14 * 16, 1024)
    assert patches.dtype == image.dtype
    assert patches.device == image.device
    torch.testing.assert_close(patches, official_patches, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(normalized.float(), dim=-1),
        torch.ones((2, 14 * 16)),
        rtol=1e-6,
        atol=1e-6,
    )
    with pytest.raises(TypeError, match="normalize must be a bool"):
        model.encode_patches(image, normalize=1)  # type: ignore[arg-type]


def _notebook_similarity_map(
    model: FakeDinoTxt,
    image: torch.Tensor,
    tokens: torch.Tensor,
    output_size: tuple[int, int] | None,
) -> torch.Tensor:
    _, patches = model.get_visual_class_and_patch_tokens(image)
    grid_height = image.shape[2] // backend_module._PATCH_SIZE
    grid_width = image.shape[3] // backend_module._PATCH_SIZE
    patch_map = (
        patches.movedim(2, 1)
        .unflatten(
            2,
            (grid_height, grid_width),
        )
        .float()
    )
    if output_size is not None:
        patch_map = F.interpolate(
            patch_map,
            size=output_size,
            mode="bicubic",
            align_corners=False,
        )
    patch_map = F.normalize(patch_map, p=2, dim=1)
    text_features = model.encode_text(tokens, normalize=False)[:, 1024:].float()
    text_features = F.normalize(text_features, p=2, dim=1)
    return torch.einsum("bdhw,kd->bkhw", patch_map, text_features)


def test_dinotxt_similarity_map_matches_notebook_and_supports_text_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    official_model = FakeDinoTxt()
    inference_backend = DinoV2DinoTxtBackend()
    model = DinoTxtCLIP(
        official_model,
        inference_backend,
        output_dict=False,
        logit_scale_mode="strict",
    )
    image = torch.rand((1, 3, 224, 196), dtype=torch.float32).transpose(2, 3)
    tokens = torch.tensor(
        [
            [49406, 320, 49407, 0],
            [49406, 334, 49407, 0],
        ],
        dtype=torch.long,
    )

    def tokenize(texts: list[str]) -> torch.Tensor:
        return tokens[: len(texts)]

    tokenizer = Mock(side_effect=tokenize)
    create_tokenizer = Mock(return_value=tokenizer)
    monkeypatch.setattr(inference_backend, "create_tokenizer", create_tokenizer)

    actual_multi = model.similarity_map(
        image,
        ("a cat", "a dog"),
        output_size=(28, 32),
    )
    expected_multi = _notebook_similarity_map(
        official_model,
        image,
        tokens,
        (28, 32),
    )
    actual_single = model.similarity_map(image, "a cat")
    expected_single = _notebook_similarity_map(
        official_model,
        image,
        tokens[:1],
        None,
    )[:, 0]

    assert actual_multi.shape == (1, 2, 28, 32)
    assert actual_single.shape == (1, 14, 16)
    assert actual_multi.dtype == actual_single.dtype == torch.float32
    assert actual_multi.device == actual_single.device == image.device
    torch.testing.assert_close(actual_multi, expected_multi, rtol=0, atol=0)
    torch.testing.assert_close(actual_single, expected_single, rtol=0, atol=0)
    create_tokenizer.assert_any_call(
        model_id=DINO_HUB_MODEL_ID,
        revision=DINO_MODEL_REVISION,
        cache_dir=None,
        prompt_name=None,
        context_length=None,
    )
    assert tokenizer.call_args_list[0].args == (["a cat", "a dog"],)
    assert tokenizer.call_args_list[1].args == (["a cat"],)


def test_dinotxt_similarity_map_rejects_empty_text_invalid_size_and_large_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DinoTxtCLIP(
        FakeDinoTxt(),
        DinoV2DinoTxtBackend(),
        output_dict=False,
        logit_scale_mode="strict",
    )
    image = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    tokenizer = Mock(return_value=torch.tensor([[49406, 49407]], dtype=torch.long))
    create_tokenizer = Mock(return_value=tokenizer)
    monkeypatch.setattr(model._inference_backend, "create_tokenizer", create_tokenizer)
    invalid_text: Any = ["cat", 1]
    invalid_size: Any = [10, 10]

    with pytest.raises(ValueError, match="must not be empty"):
        model.similarity_map(image, "")
    with pytest.raises(ValueError, match="at least one"):
        model.similarity_map(image, [])
    with pytest.raises(TypeError, match="must be a string"):
        model.similarity_map(image, invalid_text)
    with pytest.raises(TypeError, match="height, width"):
        model.similarity_map(image, "cat", output_size=invalid_size)
    with pytest.raises(ValueError, match="must be positive"):
        model.similarity_map(image, "cat", output_size=(0, 10))
    with pytest.raises(ValueError, match="memory limit"):
        model.similarity_map(image, ["cat", "dog"], output_size=(10_000, 10_000))
    create_tokenizer.assert_not_called()


def test_dinotxt_patch_normalization_rejects_zero_norms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DinoTxtCLIP(
        FakeDinoTxt(),
        DinoV2DinoTxtBackend(),
        output_dict=False,
        logit_scale_mode="strict",
    )
    image = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    monkeypatch.setattr(
        model.backend,
        "get_visual_class_and_patch_tokens",
        lambda pixels: (
            torch.ones((pixels.shape[0], 1024), dtype=pixels.dtype),
            torch.zeros((pixels.shape[0], 256, 1024), dtype=pixels.dtype),
        ),
    )

    with pytest.raises(ValueError, match="zero or invalid L2 norm"):
        model.encode_patches(image, normalize=True)


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
