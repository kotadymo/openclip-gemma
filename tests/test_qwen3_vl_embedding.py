from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

import openvlclip.backends.qwen3_vl_embedding as qwen_module
from openvlclip import create_model_and_transforms, get_tokenizer
from openvlclip.backends.qwen3_vl_embedding import (
    QWEN3VL_2B_DIMENSION,
    QWEN3VL_2B_MODEL_ID,
    QWEN3VL_2B_MODEL_NAME,
    QWEN3VL_2B_REVISION,
    Qwen3VLBackend,
    Qwen3VLCLIP,
    Qwen3VLImagePreprocess,
    _format_conversation,
)
from openvlclip.batching import iter_image_embeddings


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(padding_side="right")
        self.conversations: list[list[dict[str, object]]] = []
        self.template_options: dict[str, object] = {}
        self.processor_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        conversations: list[list[dict[str, object]]],
        **kwargs: object,
    ) -> list[str]:
        self.conversations = conversations
        self.template_options = kwargs
        prompts: list[str] = []
        for conversation in conversations:
            parts = []
            for message in conversation:
                content = message.get("content", [])
                if not isinstance(content, Sequence):
                    continue
                for element in content:
                    if not isinstance(element, Mapping):
                        continue
                    if element.get("type") == "text":
                        parts.append(str(element.get("text", "")))
                    else:
                        parts.append(f"<{element.get('type')}>")
            prompts.append(" ".join(parts))
        return prompts

    def __call__(
        self,
        *,
        text: str | Sequence[str],
        images: object,
        videos: object,
        video_metadata: object,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        texts = [text] if isinstance(text, str) else list(text)
        self.processor_calls.append(
            {
                "text": texts,
                "images": images,
                "videos": videos,
                "video_metadata": video_metadata,
                **kwargs,
            }
        )
        lengths = [max(2, min(len(value) // 24, 8)) for value in texts]
        width = max(lengths)
        input_ids = torch.zeros((len(texts), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeQwenBackbone(nn.Module):
    def __init__(self, native_dimension: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            model_type="qwen3_vl",
            text_config=SimpleNamespace(hidden_size=native_dimension),
        )
        self.last_attention_mask: torch.Tensor | None = None
        self.last_hidden_state: torch.Tensor | None = None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        self.last_attention_mask = attention_mask
        batch, sequence = input_ids.shape
        positions = torch.arange(
            sequence,
            device=self.anchor.device,
            dtype=self.anchor.dtype,
        ).view(1, sequence, 1)
        dimensions = torch.arange(
            self.config.text_config.hidden_size,
            device=self.anchor.device,
            dtype=self.anchor.dtype,
        ).view(1, 1, -1)
        rows = torch.arange(
            batch,
            device=self.anchor.device,
            dtype=self.anchor.dtype,
        ).view(batch, 1, 1)
        hidden_state = self.anchor + rows + positions * 0.1 + dimensions * 0.001
        self.last_hidden_state = hidden_state
        return SimpleNamespace(last_hidden_state=hidden_state)


def _make_backend(
    *,
    model_name: str = QWEN3VL_2B_MODEL_NAME,
    model_id: str = QWEN3VL_2B_MODEL_ID,
    revision: str = QWEN3VL_2B_REVISION,
    native_dimension: int = QWEN3VL_2B_DIMENSION,
) -> Qwen3VLBackend:
    return Qwen3VLBackend(
        model_name=model_name,
        model_id=model_id,
        revision=revision,
        native_dimension=native_dimension,
    )


def _make_model(
    *,
    logit_scale_mode: str = "cosine",
    native_dimension: int = QWEN3VL_2B_DIMENSION,
    processor: FakeProcessor | None = None,
) -> Qwen3VLCLIP:
    inference_backend = _make_backend(native_dimension=native_dimension)
    return Qwen3VLCLIP(
        FakeQwenBackbone(native_dimension),
        processor or FakeProcessor(),
        inference_backend,
        output_dict=False,
        logit_scale_mode=logit_scale_mode,
        cache_dir=None,
    ).eval()


def _install_fake_vision_utils(
    monkeypatch: pytest.MonkeyPatch,
) -> Mock:
    def process_vision_info(
        conversations: list[list[dict[str, object]]],
        **kwargs: object,
    ) -> tuple[
        list[object] | None, list[tuple[object, object]] | None, dict[str, object]
    ]:
        del kwargs
        images: list[object] = []
        videos: list[tuple[object, object]] = []
        for conversation in conversations:
            for message in conversation:
                content = message.get("content", [])
                if not isinstance(content, Sequence):
                    continue
                for element in content:
                    if not isinstance(element, Mapping):
                        continue
                    if element.get("type") == "image":
                        images.append(element["image"])
                    elif element.get("type") == "video":
                        videos.append(
                            (
                                element["video"],
                                {"fps": element.get("fps", 1.0)},
                            )
                        )
        return images or None, videos or None, {}

    vision_module = ModuleType("qwen_vl_utils.vision_process")
    vision_module.MODEL_SEQ_LEN = 128_000
    vision_module.SPATIAL_MERGE_SIZE = 2
    vision_module.get_video_reader_backend = Mock(return_value="torchcodec")
    vision_module.process_vision_info = Mock(side_effect=process_vision_info)
    package_module = ModuleType("qwen_vl_utils")
    package_module.__path__ = []
    package_module.vision_process = vision_module
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", package_module)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", vision_module)
    monkeypatch.setattr(
        qwen_module,
        "version",
        lambda package: {
            "transformers": "5.20.1",
            "qwen-vl-utils": "0.0.14",
            "torchcodec": "0.17.0",
        }[package],
    )
    return vision_module.process_vision_info


def _expected_pooled_features(model: Qwen3VLCLIP) -> torch.Tensor:
    backbone = model.backend
    assert isinstance(backbone, FakeQwenBackbone)
    assert backbone.last_hidden_state is not None
    assert backbone.last_attention_mask is not None
    last_positions = backbone.last_attention_mask.sum(dim=1) - 1
    rows = torch.arange(backbone.last_hidden_state.shape[0])
    return backbone.last_hidden_state[rows, last_positions]


def test_qwen_registry_and_capabilities_are_pinned_and_side_effect_free() -> None:
    from openvlclip import list_models, list_pretrained
    from openvlclip.registry import get_default_registry

    assert list_models() == [
        "embeddinggemma-2",
        "dinov2-dinotxt",
        "siglip2-base",
        "siglip2-naflex-base",
        QWEN3VL_2B_MODEL_NAME,
    ]
    assert list_pretrained()[-1:] == [
        (QWEN3VL_2B_MODEL_NAME, "official"),
    ]
    registry = get_default_registry()
    with pytest.raises(ValueError, match="Unknown model"):
        registry.resolve("qwen3-vl-embedding-8b")
    for name, model_id, revision, dimension in (
        (
            QWEN3VL_2B_MODEL_NAME,
            QWEN3VL_2B_MODEL_ID,
            QWEN3VL_2B_REVISION,
            QWEN3VL_2B_DIMENSION,
        ),
    ):
        descriptor = registry.resolve(name)
        assert descriptor.model_id == model_id
        assert descriptor.revision == revision
        assert descriptor.capabilities_for_full_model()["video_embedding"] is True
        assert descriptor.capabilities_for_full_model()["multimodal_embedding"] is True
        assert descriptor.backend.native_dimension == dimension


@pytest.mark.parametrize(
    ("native_dimension", "name"),
    [
        (QWEN3VL_2B_DIMENSION, QWEN3VL_2B_MODEL_NAME),
    ],
)
def test_qwen_mrl_accepts_any_dimension_in_model_card_range(
    native_dimension: int,
    name: str,
) -> None:
    model_id = QWEN3VL_2B_MODEL_ID
    revision = QWEN3VL_2B_REVISION
    backend = _make_backend(
        model_name=name,
        model_id=model_id,
        revision=revision,
        native_dimension=native_dimension,
    )

    assert backend.validate_embedding_dimension(None, name="truncate_dim") == (
        native_dimension
    )
    for dimension in (64, 65, 257, native_dimension):
        assert (
            backend.validate_embedding_dimension(
                dimension,
                name="truncate_dim",
            )
            == dimension
        )
    for invalid in (True, 63, native_dimension + 1, "128"):
        with pytest.raises((TypeError, ValueError)):
            backend.validate_embedding_dimension(invalid, name="truncate_dim")


def test_factory_loads_fixed_qwen_checkpoint_without_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backbone = FakeQwenBackbone(QWEN3VL_2B_DIMENSION)
    fake_processor = FakeProcessor()
    model_loader = Mock(
        return_value=(
            fake_backbone,
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            },
        )
    )
    processor_loader = Mock(return_value=fake_processor)
    monkeypatch.setattr(qwen_module.AutoModel, "from_pretrained", model_loader)
    monkeypatch.setattr(qwen_module.AutoProcessor, "from_pretrained", processor_loader)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        QWEN3VL_2B_MODEL_NAME,
        pretrained="official",
        device="cpu",
        logit_scale_mode="cosine",
    )

    assert isinstance(model, Qwen3VLCLIP)
    assert model.training is False
    model_loader.assert_called_once_with(
        QWEN3VL_2B_MODEL_ID,
        revision=QWEN3VL_2B_REVISION,
        cache_dir=None,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        output_loading_info=True,
    )
    processor_loader.assert_called_once_with(
        QWEN3VL_2B_MODEL_ID,
        revision=QWEN3VL_2B_REVISION,
        cache_dir=None,
        trust_remote_code=False,
        padding_side="right",
    )
    image = Image.new("L", (8, 6), 1)
    assert preprocess_train is preprocess_val
    processed = preprocess_train(image)
    assert processed.mode == "RGB"
    with pytest.raises(NotImplementedError, match="encode_texts"):
        get_tokenizer(QWEN3VL_2B_MODEL_NAME)


def test_text_encoding_uses_last_valid_token_and_normalizes_after_truncation() -> None:
    model = _make_model()
    texts = ("cat", "a much longer description for the second input")

    raw = model.encode_texts(texts)
    expected_raw = _expected_pooled_features(model)

    assert raw.shape == (2, QWEN3VL_2B_DIMENSION)
    torch.testing.assert_close(raw, expected_raw)
    assert model.processor.template_options == {
        "add_generation_prompt": True,
        "tokenize": False,
    }

    truncated = model.encode_texts(
        texts,
        normalize=True,
        truncate_dim=64,
    )
    expected = F.normalize(_expected_pooled_features(model)[:, :64], p=2, dim=-1)
    assert truncated.shape == (2, 64)
    torch.testing.assert_close(truncated, expected)
    torch.testing.assert_close(
        torch.linalg.vector_norm(truncated, dim=-1),
        torch.ones(2),
    )


def test_token_ids_and_embeddinggemma_specific_operations_are_not_emulated() -> None:
    model = _make_model()
    with pytest.raises(NotImplementedError, match="encode_texts"):
        model.encode_text(torch.ones((1, 4), dtype=torch.long))
    with pytest.raises(NotImplementedError, match="logit_scale_mode='cosine'"):
        _make_model(logit_scale_mode="strict").forward(image=Image.new("RGB", (8, 8)))
    with pytest.raises(NotImplementedError, match="logit_scale_mode='cosine'"):
        _make_model(logit_scale_mode="strict").get_logits(
            Image.new("RGB", (8, 8)),
            torch.ones((1, QWEN3VL_2B_DIMENSION)),
        )
    with pytest.raises(NotImplementedError, match="not supported"):
        model.encode_patches(Image.new("RGB", (8, 8)))
    with pytest.raises(NotImplementedError, match="not supported"):
        model.similarity_map(Image.new("RGB", (8, 8)), "a cat")


def test_multimodal_and_native_video_encoding_preserve_input_batch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_vision_info = _install_fake_vision_utils(monkeypatch)
    model = _make_model()
    image = Image.new("RGB", (16, 12), (200, 30, 40))
    frames = [
        Image.new("RGB", (16, 12), (10, 20, 30)),
        Image.new("RGB", (16, 12), (40, 50, 60)),
    ]
    items = [
        {"text": "red object", "image": image},
        {"video": frames},
        {"text": "caption", "video": frames},
    ]

    embeddings = model.encode_multimodal(items)

    assert embeddings.shape == (3, QWEN3VL_2B_DIMENSION)
    assert [
        [
            element["type"]
            for element in message["content"]
            if isinstance(element, Mapping)
        ]
        for conversation in model.processor.conversations
        for message in conversation
        if message["role"] == "user"
    ] == [["image", "text"], ["video"], ["video", "text"]]
    assert process_vision_info.call_count == 1
    assert process_vision_info.call_args.kwargs == {
        "image_patch_size": 16,
        "return_video_metadata": True,
        "return_video_kwargs": True,
    }
    last_call = model.processor.processor_calls[-1]
    assert last_call["videos"] == [frames, frames]
    assert len(last_call["video_metadata"]) == 2

    video_embedding = model.encode_video(frames)
    multimodal_video = model.encode_multimodal({"video": frames})
    assert video_embedding.shape == (1, QWEN3VL_2B_DIMENSION)
    torch.testing.assert_close(video_embedding, multimodal_video)


def test_file_video_records_and_uses_the_processor_pixel_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vision_utils(monkeypatch)
    model = _make_model()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    features = model.encode_video(video_path, fps=2.0, max_frames=12)

    assert features.shape == (1, QWEN3VL_2B_DIMENSION)
    assert model.native_video_processing_metadata == {
        "processor_version": (
            "transformers==5.20.1;qwen-vl-utils==0.0.14;"
            "video-reader=torchcodec==0.17.0"
        ),
        "total_pixels": 117_964_800.0,
    }
    video_item = next(
        element
        for element in model.processor.conversations[0][1]["content"]
        if isinstance(element, Mapping) and element.get("type") == "video"
    )
    assert video_item["fps"] == 2.0
    assert video_item["max_frames"] == 12
    assert video_item["total_pixels"] == 117_964_800.0
    assert video_item["video"] == str(video_path.resolve())


def test_qwen_batching_passes_pil_images_to_processor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vision_utils(monkeypatch)
    model = _make_model()
    preprocess = Qwen3VLImagePreprocess()
    paths = []
    for name, color in (("red.png", (255, 0, 0)), ("blue.png", (0, 0, 255))):
        path = tmp_path / name
        Image.new("RGB", (12, 8), color).save(path)
        paths.append(path)

    embeddings = list(iter_image_embeddings(model, preprocess, paths, batch_size=2))

    assert [path for path, _ in embeddings] == paths
    assert all(vector.shape == (QWEN3VL_2B_DIMENSION,) for _, vector in embeddings)
    assert len(model.processor.processor_calls) == 1
    images = model.processor.processor_calls[0]["images"]
    assert isinstance(images, list)
    assert len(images) == 2
    assert all(isinstance(image, Image.Image) for image in images)


def test_qwen_image_and_text_routes_share_multimodal_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vision_utils(monkeypatch)
    model = _make_model()
    image = Image.new("RGB", (12, 8), (20, 40, 60))

    torch.testing.assert_close(
        model.encode_image(image),
        model.encode_multimodal({"image": image}),
    )
    torch.testing.assert_close(
        model.encode_texts("a blue object"),
        model.encode_multimodal({"text": "a blue object"}),
    )


def test_qwen_cosine_logits_are_distinct_from_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vision_utils(monkeypatch)
    model = _make_model(logit_scale_mode="cosine")
    image = Image.new("RGB", (12, 8), (20, 40, 60))
    text_features = model.encode_texts(("a blue object", "a red object"))

    logits_per_image, logits_per_text = model.get_logits(image, text_features)
    image_features = model.encode_image(image, normalize=True)
    normalized_text = F.normalize(text_features, p=2, dim=-1)
    expected = image_features @ normalized_text.T

    torch.testing.assert_close(logits_per_image, expected)
    torch.testing.assert_close(logits_per_text, expected.T)
    image_features, text_features, logit_scale = model(
        image=image,
        text=text_features,
    )
    torch.testing.assert_close(
        image_features, model.encode_image(image, normalize=True)
    )
    assert text_features is not None
    assert logit_scale.item() == 1.0


def test_qwen_rejects_empty_inputs_remote_urls_and_missing_vision_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="at least one string"):
        model.encode_texts([])
    with pytest.raises(ValueError, match="cannot encode an empty batch"):
        model.encode_multimodal([])
    with pytest.raises(ValueError, match="must contain text, image, or video"):
        model.encode_multimodal({})
    with pytest.raises(ValueError, match="Remote URLs"):
        _format_conversation({"image": "https://example.com/image.png"}, index=0)

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", None)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", None)
    with pytest.raises(RuntimeError, match="openvlclip\\[qwen3vl\\]"):
        model.encode_image(Image.new("RGB", (8, 8)))
