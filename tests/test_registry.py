from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from torch import nn

import openvlclip.factory as factory_module
import openvlclip.tokenizer as tokenizer_module
from openvlclip.backends.dinov2_dinotxt import (
    DINO_BACKEND_ID,
    DINO_HUB_MODEL_ID,
    DINO_MODEL_NAME,
    DINO_MODEL_REVISION,
)
from openvlclip.backends.qwen3_vl_embedding import (
    QWEN3VL_2B_DIMENSION,
    QWEN3VL_2B_MODEL_ID,
    QWEN3VL_2B_MODEL_NAME,
    QWEN3VL_2B_REVISION,
    Qwen3VLBackend,
)
from openvlclip.capabilities import (
    CAPABILITY_KEYS,
    capabilities_for_modalities,
    freeze_capabilities,
)
from openvlclip.constants import MODEL_ID, MODEL_NAME, MODEL_REVISION
from openvlclip.embedding_space import EmbeddingSpaceMetadata
from openvlclip.factory import create_model_and_transforms
from openvlclip.model import EmbeddingGemmaCLIP
from openvlclip.registry import (
    ModelDescriptor,
    ModelRegistry,
    get_default_registry,
)
from openvlclip.tokenizer import get_tokenizer


class FakeTokenizer:
    pass


class FakePreprocess:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return torch.zeros((3, image.height, image.width))


class FakeModel(nn.Module):
    def __init__(
        self,
        backend: FakeBackend,
        modalities: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.backend = backend
        self.modalities = modalities
        self.embedding_space_metadata = backend.embedding_space_metadata
        self.capabilities = backend.capabilities_for(modalities)

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int:
        return self.backend.validate_embedding_dimension(value, name=name)

    def encode_image(
        self,
        image: torch.Tensor,
        normalize: bool = False,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        return self.backend.encode_image(
            self,
            image,
            normalize=normalize,
            truncate_dim=truncate_dim,
        )

    def encode_text(
        self,
        text: torch.Tensor,
        normalize: bool = False,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        return self.backend.encode_text(
            self,
            text,
            normalize=normalize,
            truncate_dim=truncate_dim,
        )


class FakeBackend:
    backend_id = "test-backend"
    capabilities = freeze_capabilities(
        {
            "image_embedding": True,
            "text_embedding": True,
            "image_text_similarity": True,
            "patch_embedding": False,
            "text_similarity_map": False,
            "mrl": False,
            "selective_loading": True,
            "video_embedding": False,
            "multimodal_embedding": False,
            "instruction_aware": False,
            "variable_resolution": False,
            "sigmoid_logits": False,
        }
    )
    embedding_space_metadata = EmbeddingSpaceMetadata(
        backend_id=backend_id,
        model_id="test/model",
        model_revision="test-revision",
        embedding_strategy="test-native-v1",
        normalization="l2",
        preprocessing_id="test-preprocess-v1",
        index_prompt=None,
        query_prompt_name=None,
        query_prompt=None,
    )

    def __init__(self) -> None:
        self.load_options: dict[str, object] | None = None
        self.preprocessed_model: FakeModel | None = None
        self.tokenizer_options: dict[str, object] | None = None
        self.loaded_model: FakeModel | None = None
        self.preprocess = FakePreprocess()

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
    ) -> FakeModel:
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
        model = FakeModel(self, modalities or ("text", "image"))
        self.loaded_model = model.to(device=device, dtype=dtype).eval()
        return self.loaded_model

    def create_tokenizer(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: str | None,
        prompt_name: str | None,
        context_length: int | None,
    ) -> FakeTokenizer:
        self.tokenizer_options = {
            "model_id": model_id,
            "revision": revision,
            "cache_dir": cache_dir,
            "prompt_name": prompt_name,
            "context_length": context_length,
        }
        return FakeTokenizer()

    def create_preprocess(self, model: FakeModel) -> FakePreprocess:
        self.preprocessed_model = model
        return self.preprocess

    def capabilities_for(self, modalities: tuple[str, ...]) -> Mapping[str, bool]:
        return capabilities_for_modalities(self.capabilities, modalities)

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int:
        if value is None:
            return 384
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be the integer 384.")
        if value != 384:
            raise ValueError(f"{name} must be 384; received {value}.")
        return value

    def encode_image(
        self,
        _model: FakeModel,
        image: torch.Tensor,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor:
        del _model
        del normalize
        dimension = self.validate_embedding_dimension(
            truncate_dim,
            name="truncate_dim",
        )
        return image.new_zeros((image.shape[0], dimension))

    def encode_text(
        self,
        _model: FakeModel,
        text: torch.Tensor,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor:
        del _model
        del normalize
        dimension = self.validate_embedding_dimension(
            truncate_dim,
            name="truncate_dim",
        )
        return torch.zeros(
            (text.shape[0], dimension),
            device=text.device,
            dtype=torch.float32,
        )


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


def test_default_registry_lists_models_without_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_loader = Mock(side_effect=AssertionError("model weights must not load"))
    monkeypatch.setattr(torch.hub, "load", model_loader)
    registry = get_default_registry()

    assert registry.list_models() == [
        MODEL_NAME,
        DINO_MODEL_NAME,
        "siglip2-base",
        "siglip2-naflex-base",
        "siglip2-base-384",
        "siglip2-large-256",
        "siglip2-so400m-384",
        QWEN3VL_2B_MODEL_NAME,
    ]
    assert registry.list_pretrained() == [
        (MODEL_NAME, "official"),
        (DINO_MODEL_NAME, "official"),
        ("siglip2-base", "official"),
        ("siglip2-naflex-base", "official"),
        ("siglip2-base-384", "official"),
        ("siglip2-large-256", "official"),
        ("siglip2-so400m-384", "official"),
        (QWEN3VL_2B_MODEL_NAME, "official"),
    ]
    assert registry.resolve(" EMBEDDINGGEMMA_2 ") is registry.descriptors[0]
    assert registry.descriptors[0].model_id == MODEL_ID
    assert registry.descriptors[0].revision == MODEL_REVISION
    dinotxt = registry.resolve(DINO_MODEL_NAME)
    assert dinotxt.backend_id == DINO_BACKEND_ID
    assert dinotxt.model_id == DINO_HUB_MODEL_ID
    assert dinotxt.revision == DINO_MODEL_REVISION
    qwen_2b = registry.resolve(QWEN3VL_2B_MODEL_NAME)
    assert qwen_2b.model_id == QWEN3VL_2B_MODEL_ID
    assert qwen_2b.revision == QWEN3VL_2B_REVISION
    with pytest.raises(ValueError, match="Unsupported model_name"):
        registry.resolve("qwen3-vl-embedding-8b")
    model_loader.assert_not_called()


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


def test_registry_resolves_backend_only_for_exact_checkpoint_identity() -> None:
    backend = FakeBackend()
    registry = _fake_registry(backend)
    metadata = backend.embedding_space_metadata

    assert (
        registry.resolve_backend_for_embedding_space(
            backend_id=metadata.backend_id,
            model_id=metadata.model_id,
            revision=metadata.model_revision,
        )
        is backend
    )
    for backend_id, model_id, revision in (
        ("other-backend", metadata.model_id, metadata.model_revision),
        (metadata.backend_id, "other/model", metadata.model_revision),
        (metadata.backend_id, metadata.model_id, "other-revision"),
    ):
        with pytest.raises(ValueError, match="No registered backend"):
            registry.resolve_backend_for_embedding_space(
                backend_id=backend_id,
                model_id=model_id,
                revision=revision,
            )


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
    assert isinstance(preprocess_train, FakePreprocess)
    assert isinstance(model, FakeModel)
    assert not isinstance(model, EmbeddingGemmaCLIP)
    assert model.embedding_space_metadata is backend.embedding_space_metadata
    assert model.validate_embedding_dimension(None, name="embedding_dim") == 384

    tokenizer_result = get_tokenizer(
        "fake-model",
        prompt_name="SearchQuery",
        context_length=128,
        cache_dir="token-cache",
    )

    assert isinstance(tokenizer_result, FakeTokenizer)
    assert backend.tokenizer_options == {
        "model_id": "test/model",
        "revision": "test-revision",
        "cache_dir": "token-cache",
        "prompt_name": "SearchQuery",
        "context_length": 128,
    }


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

    assert isinstance(full_model, FakeModel)
    with pytest.raises(ValueError, match="must be 384"):
        full_model.validate_embedding_dimension(768, name="embedding_dim")


def test_dense_capabilities_require_the_relevant_modalities() -> None:
    capabilities = freeze_capabilities(
        {
            "image_embedding": True,
            "text_embedding": True,
            "image_text_similarity": True,
            "patch_embedding": True,
            "text_similarity_map": True,
            "mrl": False,
            "selective_loading": True,
            "video_embedding": True,
            "multimodal_embedding": True,
            "instruction_aware": True,
            "variable_resolution": True,
            "sigmoid_logits": True,
        }
    )

    text_only = capabilities_for_modalities(capabilities, ("text",))
    image_only = capabilities_for_modalities(capabilities, ("image",))
    full = capabilities_for_modalities(
        capabilities,
        ("text", "image", "video"),
    )

    assert text_only["patch_embedding"] is False
    assert text_only["text_similarity_map"] is False
    assert text_only["variable_resolution"] is False
    assert text_only["instruction_aware"] is True
    assert text_only["video_embedding"] is False
    assert text_only["multimodal_embedding"] is False
    assert text_only["sigmoid_logits"] is False
    assert image_only["patch_embedding"] is True
    assert image_only["text_similarity_map"] is False
    assert image_only["variable_resolution"] is True
    assert image_only["instruction_aware"] is False
    assert image_only["video_embedding"] is False
    assert image_only["multimodal_embedding"] is False
    assert image_only["sigmoid_logits"] is False
    assert full["video_embedding"] is True
    assert full["multimodal_embedding"] is True
    assert full["instruction_aware"] is True
    assert full["variable_resolution"] is True
    assert full["sigmoid_logits"] is True


def test_registered_backends_declare_the_extended_capability_contract() -> None:
    registry = get_default_registry()
    gemma = registry.resolve(MODEL_NAME)
    dinotxt = registry.resolve(DINO_MODEL_NAME)
    qwen_2b = registry.resolve(QWEN3VL_2B_MODEL_NAME)

    assert tuple(gemma.backend.capabilities) == CAPABILITY_KEYS
    assert gemma.capabilities_for_full_model()["instruction_aware"] is True
    assert gemma.capabilities_for_full_model()["variable_resolution"] is True
    assert gemma.capabilities_for_full_model()["video_embedding"] is False
    assert gemma.capabilities_for_full_model()["multimodal_embedding"] is False
    assert gemma.capabilities_for_full_model()["sigmoid_logits"] is False

    assert tuple(dinotxt.backend.capabilities) == CAPABILITY_KEYS
    assert dinotxt.capabilities_for_full_model()["instruction_aware"] is False
    assert dinotxt.capabilities_for_full_model()["variable_resolution"] is False
    assert dinotxt.capabilities_for_full_model()["video_embedding"] is False
    assert dinotxt.capabilities_for_full_model()["multimodal_embedding"] is False
    assert dinotxt.capabilities_for_full_model()["sigmoid_logits"] is False

    for descriptor, dimension in (
        (qwen_2b, QWEN3VL_2B_DIMENSION),
    ):
        assert isinstance(descriptor.backend, Qwen3VLBackend)
        assert descriptor.backend.native_dimension == dimension
        capabilities = descriptor.capabilities_for_full_model()
        assert capabilities["image_embedding"] is True
        assert capabilities["text_embedding"] is True
        assert capabilities["image_text_similarity"] is True
        assert capabilities["mrl"] is True
        assert capabilities["selective_loading"] is False
        assert capabilities["video_embedding"] is True
        assert capabilities["multimodal_embedding"] is True
        assert capabilities["instruction_aware"] is True
        assert capabilities["patch_embedding"] is False
        assert capabilities["text_similarity_map"] is False
