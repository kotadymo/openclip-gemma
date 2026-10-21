from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import openclip_gemma.factory as factory_module
import openclip_gemma.tokenizer as tokenizer_module
from openclip_gemma.backends.embeddinggemma2 import EmbeddingGemma2Backend
from openclip_gemma.capabilities import CAPABILITY_KEYS
from openclip_gemma.constants import MODEL_ID, MODEL_NAME, MODEL_REVISION
from openclip_gemma.factory import create_model_and_transforms
from openclip_gemma.model import EmbeddingGemmaCLIP
from openclip_gemma.registry import (
    ModelDescriptor,
    ModelRegistry,
    get_default_registry,
)
from openclip_gemma.tokenizer import get_tokenizer


class FakeBackend(EmbeddingGemma2Backend):
    backend_id = "test-backend"
    embedding_space_metadata = replace(
        EmbeddingGemma2Backend.embedding_space_metadata,
        backend_id="test-backend",
        model_id="test/model",
        model_revision="test-revision",
    )

    def __init__(self) -> None:
        self.load_options: dict[str, object] | None = None
        self.preprocessed_model: EmbeddingGemmaCLIP | None = None

    def load_model(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: str | None,
        device: torch.device,
        dtype: torch.dtype,
        output_dict: bool,
        logit_scale_mode: str,
        modalities: tuple[str, ...] | None,
    ) -> EmbeddingGemmaCLIP:
        self.load_options = {
            "model_id": model_id,
            "revision": revision,
            "cache_dir": cache_dir,
            "device": device,
            "dtype": dtype,
            "output_dict": output_dict,
            "logit_scale_mode": logit_scale_mode,
            "modalities": modalities,
        }
        raw_model = nn.Linear(2, 2, device=device, dtype=dtype)
        processor = SimpleNamespace(image_processor=object())
        return EmbeddingGemmaCLIP(
            raw_model,
            processor,
            output_dict=output_dict,
            logit_scale_mode=logit_scale_mode,
            modalities=modalities or ("text", "image"),
            inference_backend=self,
        ).eval()

    def create_preprocess(self, model: EmbeddingGemmaCLIP):
        self.preprocessed_model = model
        return super().create_preprocess(model)


def _fake_registry(backend: FakeBackend) -> ModelRegistry:
    descriptor = ModelDescriptor(
        model_name="fake-model",
        aliases=("fake-alias",),
        backend_id=backend.backend_id,
        model_id=backend.embedding_space_metadata.model_id,
        revision=backend.embedding_space_metadata.model_revision,
        pretrained_tags={"unit-test": "test/checkpoint"},
        pretrained_aliases={"test/checkpoint": "unit-test"},
        capabilities=backend.capabilities,
        embedding_space_metadata=backend.embedding_space_metadata,
        backend=backend,
    )
    return ModelRegistry((descriptor,))


def test_default_registry_lists_only_pinned_model_without_loading_weights() -> None:
    registry = get_default_registry()

    assert registry.list_models() == [MODEL_NAME]
    assert registry.list_pretrained() == [(MODEL_NAME, "official")]
    assert registry.resolve(" EMBEDDINGGEMMA_2 ") is registry.descriptors[0]
    assert registry.descriptors[0].model_id == MODEL_ID
    assert registry.descriptors[0].revision == MODEL_REVISION


def test_registry_rejects_duplicate_names_and_invalid_descriptors() -> None:
    descriptor = get_default_registry().descriptors[0]
    duplicate = replace(
        descriptor,
        model_name="other-model",
        aliases=("embeddinggemma_2",),
    )

    with pytest.raises(ValueError, match="Duplicate model name or alias"):
        ModelRegistry((descriptor, duplicate))
    with pytest.raises(ValueError, match="backend_id"):
        replace(descriptor, backend_id="unregistered-backend")
    with pytest.raises(ValueError, match="pretrained_aliases"):
        replace(descriptor, pretrained_aliases={MODEL_ID: "unknown-tag"})


def test_registry_fake_backend_receives_factory_tokenizer_and_transform_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    registry = _fake_registry(backend)
    monkeypatch.setattr(factory_module, "get_default_registry", lambda: registry)
    monkeypatch.setattr(tokenizer_module, "get_default_registry", lambda: registry)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        "fake-alias",
        pretrained="test/checkpoint",
        precision="bf16",
        output_dict=True,
        logit_scale_mode="cosine",
    )

    assert isinstance(model, EmbeddingGemmaCLIP)
    assert backend.load_options == {
        "model_id": "test/checkpoint",
        "revision": "test-revision",
        "cache_dir": None,
        "device": torch.device("cpu"),
        "dtype": torch.bfloat16,
        "output_dict": True,
        "logit_scale_mode": "cosine",
        "modalities": None,
    }
    assert backend.preprocessed_model is model
    assert preprocess_train is preprocess_val
    assert model.embedding_space_metadata is backend.embedding_space_metadata

    tokenizer_result = object()
    tokenizer_loader = Mock(return_value=tokenizer_result)
    backend.create_tokenizer = tokenizer_loader  # type: ignore[method-assign]

    result = get_tokenizer(
        "fake-model",
        prompt_name="SearchQuery",
        context_length=128,
        cache_dir="token-cache",
    )

    assert result is tokenizer_result
    tokenizer_loader.assert_called_once_with(
        model_id="test/model",
        revision="test-revision",
        cache_dir="token-cache",
        prompt_name="SearchQuery",
        context_length=128,
    )


def test_model_capabilities_are_immutable_and_respect_selective_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    registry = _fake_registry(backend)
    monkeypatch.setattr(factory_module, "get_default_registry", lambda: registry)

    full_model = factory_module.create_model("fake-model", pretrained="unit-test")
    text_model = factory_module.create_model(
        "fake-model",
        pretrained="unit-test",
        modalities=("text",),
    )

    assert tuple(full_model.capabilities) == CAPABILITY_KEYS
    assert full_model.capabilities["image_embedding"] is True
    assert full_model.capabilities["image_text_similarity"] is True
    assert text_model.capabilities["image_embedding"] is False
    assert text_model.capabilities["image_text_similarity"] is False
    assert text_model.capabilities["text_embedding"] is True
    with pytest.raises(TypeError):
        text_model.capabilities["image_embedding"] = True  # type: ignore[index]
