from __future__ import annotations

from math import log
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from torch import nn

from openvlclip import create_model, create_model_and_transforms, get_tokenizer
from openvlclip.backends import siglip2
from openvlclip.image_search import build_or_update_image_index, search_images
from openvlclip.registry import get_default_registry

_MODEL_CASES = [
    (
        "siglip2-base",
        "fixres",
        "siglip",
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    ),
    (
        "siglip2-naflex-base",
        "naflex",
        "siglip2",
        "google/siglip2-base-patch16-naflex",
        "b53b807d3a2d5e2b3911292f2d69e5341cdc064c",
    ),
]


class FakeTokenizer:
    pad_token_id = 0
    model_max_length = 64
    padding_side = "right"

    def __init__(self) -> None:
        self.options: dict[str, object] | None = None

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        self.options = kwargs
        max_length = kwargs["max_length"]
        assert isinstance(max_length, int)
        input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, text in enumerate(texts):
            values = [2, len(text) + 3, 4]
            input_ids[row, : len(values)] = torch.tensor(values)
            attention_mask[row, : len(values)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeSiglipProcessor:
    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.image_processor = SimpleNamespace(patch_size=16, max_num_patches=256)
        self.tokenizer = FakeTokenizer()
        self.received_images: list[Image.Image] = []

    def __call__(
        self,
        *,
        images: Image.Image,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        self.received_images.append(images)
        if self.variant == "fixres":
            return {"pixel_values": torch.ones((1, 3, 224, 224))}
        masks = torch.zeros((1, 256), dtype=torch.long)
        masks[:, :4] = 1
        return {
            "pixel_values": torch.full((1, 256, 768), 0.25),
            "pixel_attention_mask": masks,
            "spatial_shapes": torch.tensor([[2, 2]], dtype=torch.long),
        }


class FakeSiglipModel(nn.Module):
    def __init__(self, model_type: str) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.logit_scale = nn.Parameter(torch.tensor(log(2.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-0.25))
        self.config = SimpleNamespace(model_type=model_type)
        self.last_image_inputs: dict[str, torch.Tensor] | None = None
        self.last_text_inputs: dict[str, torch.Tensor] | None = None

    def get_image_features(
        self,
        *,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        self.last_image_inputs = {
            "pixel_values": pixel_values,
            **(
                {"pixel_attention_mask": pixel_attention_mask}
                if pixel_attention_mask is not None
                else {}
            ),
            **(
                {"spatial_shapes": spatial_shapes} if spatial_shapes is not None else {}
            ),
        }
        if pixel_values.ndim == 4:
            signal = pixel_values.float().mean(dim=(1, 2, 3))
        else:
            assert pixel_attention_mask is not None
            mask = pixel_attention_mask.to(dtype=torch.float32)
            signal = (pixel_values.float() * mask.unsqueeze(-1)).sum(dim=(1, 2))
            signal = signal / (mask.sum(dim=1) * pixel_values.shape[-1])
        basis = torch.linspace(0.5, 1.5, 768, device=pixel_values.device)
        return SimpleNamespace(pooler_output=(signal[:, None] + 1) * basis)

    def get_text_features(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        self.last_text_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        signal = (input_ids * attention_mask).sum(dim=1).float()
        basis = torch.linspace(0.25, 1.25, 768, device=input_ids.device)
        return SimpleNamespace(pooler_output=(signal[:, None] + 1) * basis)


def _load_fake_model(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    *,
    logit_scale_mode: str = "strict",
) -> tuple[
    nn.Module,
    FakeSiglipModel,
    FakeSiglipProcessor,
    Mock,
    Mock,
]:
    descriptor = get_default_registry().resolve(model_name)
    backend = descriptor.backend
    hf_model = FakeSiglipModel("siglip" if backend.variant == "fixres" else "siglip2")
    processor = FakeSiglipProcessor(backend.variant)
    model_loader = Mock(return_value=hf_model)
    processor_loader = Mock(return_value=processor)
    monkeypatch.setattr(siglip2.AutoModel, "from_pretrained", model_loader)
    monkeypatch.setattr(siglip2.AutoProcessor, "from_pretrained", processor_loader)
    model = create_model(
        model_name,
        cache_dir="model-cache",
        logit_scale_mode=logit_scale_mode,
    )
    return model, hf_model, processor, model_loader, processor_loader


@pytest.mark.parametrize(
    ("model_name", "variant", "model_type", "model_id", "revision"),
    _MODEL_CASES,
)
def test_factory_loads_pinned_checkpoint_and_preprocesses_official_inputs(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    variant: str,
    model_type: str,
    model_id: str,
    revision: str,
) -> None:
    descriptor = get_default_registry().resolve(model_name)
    hf_model = FakeSiglipModel(model_type)
    processor = FakeSiglipProcessor(variant)
    model_loader = Mock(return_value=hf_model)
    processor_loader = Mock(return_value=processor)
    monkeypatch.setattr(siglip2.AutoModel, "from_pretrained", model_loader)
    monkeypatch.setattr(siglip2.AutoProcessor, "from_pretrained", processor_loader)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        cache_dir="model-cache",
    )

    assert preprocess_train is preprocess_val
    assert model.backend is hf_model
    assert model.processor is processor
    assert not model.training
    assert not hf_model.training
    model_loader.assert_called_once_with(
        model_id,
        revision=revision,
        cache_dir="model-cache",
        dtype=torch.float32,
        use_safetensors=True,
    )
    processor_loader.assert_called_once_with(
        model_id,
        revision=revision,
        cache_dir="model-cache",
    )
    assert descriptor.revision == revision

    image = Image.new("RGB", (96, 64), (20, 80, 160))
    processed = preprocess_val(image)
    assert processor.received_images[-1].mode == "RGB"
    if variant == "fixres":
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (3, 224, 224)
    else:
        assert isinstance(processed, dict)
        assert set(processed) == {
            "pixel_values",
            "pixel_attention_mask",
            "spatial_shapes",
        }
        assert processed["pixel_values"].shape == (256, 768)
        assert processed["pixel_attention_mask"].shape == (256,)
        assert processed["spatial_shapes"].tolist() == [2, 2]


def test_siglip2_registry_exposes_model_specific_capabilities() -> None:
    registry = get_default_registry()
    fixres = registry.resolve("siglip2-base")
    naflex = registry.resolve("siglip2-naflex-base")

    assert fixres.backend_id == naflex.backend_id == siglip2.SIGLIP2_BACKEND_ID
    assert fixres.embedding_space_metadata.model_id != (
        naflex.embedding_space_metadata.model_id
    )
    assert (
        fixres.embedding_space_metadata.preprocessing_id
        != naflex.embedding_space_metadata.preprocessing_id
    )
    for descriptor in (fixres, naflex):
        assert descriptor.capabilities_for_full_model()["image_embedding"] is True
        assert descriptor.capabilities_for_full_model()["text_embedding"] is True
        assert descriptor.capabilities_for_full_model()["sigmoid_logits"] is True
        assert descriptor.capabilities_for_full_model()["mrl"] is False
        assert descriptor.capabilities_for_full_model()["selective_loading"] is False
        assert descriptor.capabilities_for_full_model()["patch_embedding"] is False
    assert fixres.capabilities_for_full_model()["variable_resolution"] is False
    assert naflex.capabilities_for_full_model()["variable_resolution"] is True


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [(case[0], case[1]) for case in _MODEL_CASES],
)
def test_feature_encoding_and_strict_sigmoid_logits(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    variant: str,
) -> None:
    model, hf_model, _, _, _ = _load_fake_model(monkeypatch, model_name)
    image = Image.new("RGB", (96, 64), (20, 80, 160))
    preprocess = model._inference_backend.create_preprocess(model)
    processed = preprocess(image)
    token_ids = torch.tensor([[2, 3, 4, 0], [2, 5, 4, 0]], dtype=torch.long)

    image_features = model.encode_image(processed)
    text_features = model.encode_text(token_ids)
    normalized_image = model.encode_image(processed, normalize=True)
    normalized_text = model.encode_text(token_ids, normalize=True)

    assert image_features.shape == (1, 768)
    assert text_features.shape == (2, 768)
    assert normalized_image.dtype == torch.float32
    torch.testing.assert_close(
        torch.linalg.vector_norm(normalized_image, dim=-1),
        torch.ones(1),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(normalized_text, dim=-1),
        torch.ones(2),
    )
    assert hf_model.last_text_inputs is not None
    torch.testing.assert_close(
        hf_model.last_text_inputs["attention_mask"],
        token_ids.ne(0),
    )
    if variant == "naflex":
        assert hf_model.last_image_inputs is not None
        assert hf_model.last_image_inputs["pixel_attention_mask"].shape == (1, 256)
        assert hf_model.last_image_inputs["spatial_shapes"].tolist() == [[2, 2]]

    logits_per_image, logits_per_text = model.get_logits(processed, token_ids)
    expected_per_text = (
        normalized_text @ normalized_image.T
    ) * hf_model.logit_scale.exp() + hf_model.logit_bias
    torch.testing.assert_close(logits_per_image, expected_per_text.T)
    torch.testing.assert_close(logits_per_text, expected_per_text)
    with pytest.raises(NotImplementedError, match="use get_logits"):
        model(processed, token_ids)


def test_cosine_mode_remains_distinct_from_learned_siglip_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, _, _, _ = _load_fake_model(
        monkeypatch,
        "siglip2-base",
        logit_scale_mode="cosine",
    )
    image = torch.ones((1, 3, 224, 224))
    tokens = torch.tensor([[2, 3, 0]], dtype=torch.long)

    logits_per_image, logits_per_text = model.get_logits(image, tokens)
    image_features = model.encode_image(image, normalize=True)
    text_features = model.encode_text(tokens, normalize=True)
    expected = image_features @ text_features.T

    torch.testing.assert_close(logits_per_image, expected)
    torch.testing.assert_close(logits_per_text, expected.T)
    assert not torch.allclose(
        logits_per_image,
        expected * model.backend.logit_scale.exp() + model.backend.logit_bias,
    )


def test_siglip2_tokenizer_matches_fixed_length_processor_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = get_default_registry().resolve("siglip2-base")
    fake_tokenizer = FakeTokenizer()
    tokenizer_loader = Mock(return_value=fake_tokenizer)
    monkeypatch.setattr(siglip2.AutoTokenizer, "from_pretrained", tokenizer_loader)

    tokenizer = get_tokenizer("siglip2-base", cache_dir="token-cache")
    tokens = tokenizer(["a cat", "a dog"])

    assert tokens.shape == (2, 64)
    assert tokens.dtype == torch.long
    assert fake_tokenizer.options == {
        "padding": "max_length",
        "truncation": True,
        "max_length": 64,
        "return_attention_mask": True,
        "return_tensors": "pt",
    }
    tokenizer_loader.assert_called_once_with(
        descriptor.model_id,
        revision=descriptor.revision,
        cache_dir="token-cache",
    )

    with pytest.raises(ValueError, match="no task prompts"):
        get_tokenizer("siglip2-base", prompt_name="SearchQuery")
    with pytest.raises(ValueError, match="context_length"):
        get_tokenizer("siglip2-base", context_length=65)


@pytest.mark.parametrize("dimension", [128, 256, 512, 1024])
def test_siglip2_rejects_mrl_truncation(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
) -> None:
    model, *_ = _load_fake_model(monkeypatch, "siglip2-base")

    with pytest.raises(ValueError, match="does not support MRL or truncation"):
        model.validate_embedding_dimension(dimension, name="truncate_dim")


def test_siglip2_rejects_naflex_tensors_and_invalid_processor_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, _, _, _ = _load_fake_model(monkeypatch, "siglip2-naflex-base")

    with pytest.raises(TypeError, match="processor-aware"):
        model.encode_image(torch.zeros((1, 3, 224, 224)))

    invalid = {
        "pixel_values": torch.zeros((256, 768)),
        "pixel_attention_mask": torch.zeros((256,), dtype=torch.long),
        "spatial_shapes": torch.tensor([2, 2]),
    }
    with pytest.raises(ValueError, match="do not describe the same valid patches"):
        model.encode_image(invalid)

    missing = {"pixel_values": torch.zeros((256, 768))}
    with pytest.raises(ValueError, match="must contain pixel_values"):
        model.encode_image(missing)


def test_siglip2_instruction_input_is_explicitly_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, *_ = _load_fake_model(monkeypatch, "siglip2-base")

    with pytest.raises(NotImplementedError, match="not instruction-aware"):
        model.encode_texts("a cat", instruction="retrieve")


def test_naflex_image_search_batches_structured_processor_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, _, _, _ = _load_fake_model(monkeypatch, "siglip2-naflex-base")
    fake_tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        siglip2.AutoTokenizer,
        "from_pretrained",
        Mock(return_value=fake_tokenizer),
    )
    preprocess = model._inference_backend.create_preprocess(model)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (96, 64), (210, 30, 40)).save(images_dir / "red.png")
    Image.new("RGB", (64, 96), (20, 50, 210)).save(images_dir / "blue.png")
    index_dir = tmp_path / "index"

    update = build_or_update_image_index(
        model,
        preprocess,
        images_dir,
        index_dir,
        batch_size=2,
    )
    results = search_images(model, index_dir, "a red object", top_k=2)

    assert update.added == 2
    assert len(results) == 2
    assert [result.path.name for result in results] == ["blue.png", "red.png"]
    assert all(-1.0 <= result.score <= 1.0 for result in results)
