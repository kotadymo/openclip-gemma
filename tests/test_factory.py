from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from openvlclip import (
    EmbeddingGemmaCLIP,
    create_model,
    create_model_and_transforms,
    create_model_from_pretrained,
    list_models,
    list_pretrained,
)
from openvlclip.constants import MODEL_ID, MODEL_REVISION
from openvlclip.transforms import EmbeddingGemmaPreprocess


class FakeImageProcessor:
    do_convert_rgb = True
    patch_size = 16
    pooling_kernel_size = 3
    max_soft_tokens = 280
    resample = 3
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = False
    image_mean: ClassVar[list[float]] = [0.5, 0.5, 0.5]
    image_std: ClassVar[list[float]] = [0.5, 0.5, 0.5]

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def process_image(
        self, image: Image.Image, *, do_convert_rgb: bool
    ) -> torch.Tensor:
        self.calls.append(("convert", do_convert_rgb))
        return torch.zeros((3, image.height, image.width), dtype=torch.uint8)

    def aspect_ratio_preserving_resize(
        self,
        *,
        image: torch.Tensor,
        patch_size: int,
        max_patches: int,
        pooling_kernel_size: int,
        resample: int,
    ) -> torch.Tensor:
        self.calls.append(
            ("resize", (patch_size, max_patches, pooling_kernel_size, resample))
        )
        return image

    def rescale_and_normalize(
        self,
        image: torch.Tensor,
        *,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: list[float],
        image_std: list[float],
    ) -> torch.Tensor:
        self.calls.append(
            (
                "rescale",
                (do_rescale, rescale_factor, do_normalize, image_mean, image_std),
            )
        )
        return image.float() / 255


@pytest.fixture
def loaded_model(monkeypatch: pytest.MonkeyPatch) -> torch.nn.Module:
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained",
        Mock(return_value=model),
    )
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoProcessor.from_pretrained",
        Mock(return_value=SimpleNamespace(image_processor=FakeImageProcessor())),
    )
    return model


def test_create_model_returns_eval_module_and_loads_pinned_fp32(
    loaded_model: torch.nn.Module,
) -> None:
    result = create_model()

    assert isinstance(result, EmbeddingGemmaCLIP)
    assert result.backend is loaded_model
    assert not result.training
    assert not result.backend.training
    assert next(result.backend.parameters()).dtype == torch.float32


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("fp32", torch.float32),
        ("float32", torch.float32),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("BF16", torch.bfloat16),
    ],
)
def test_create_model_applies_precision(
    loaded_model: torch.nn.Module,
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
    expected_dtype: torch.dtype,
) -> None:
    loader = Mock(return_value=loaded_model)
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained", loader
    )

    result = create_model(precision=precision)

    assert next(result.backend.parameters()).dtype == expected_dtype
    assert loader.call_args.kwargs["dtype"] == expected_dtype


def test_create_model_applies_cpu_device(
    loaded_model: torch.nn.Module,
) -> None:
    result = create_model(device=torch.device("cpu"))

    assert result.backend is loaded_model
    assert next(result.backend.parameters()).device.type == "cpu"


def test_create_model_can_load_text_only_without_vision_or_audio(
    loaded_model: torch.nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(vision_config=object(), audio_config=object())
    config_loader = Mock(return_value=config)
    model_loader = Mock(return_value=loaded_model)
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoConfig.from_pretrained",
        config_loader,
    )
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained",
        model_loader,
    )
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoProcessor.from_pretrained",
        Mock(return_value=SimpleNamespace()),
    )

    model = create_model(modalities=("text",))

    assert model.modalities == ("text",)
    assert config.vision_config is None
    assert config.audio_config is None
    config_loader.assert_called_once_with(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=None,
    )
    assert model_loader.call_args.kwargs["config"] is config
    with pytest.raises(RuntimeError, match="without the image modality"):
        model.encode_image(torch.zeros((1, 3, 8, 8)))


def test_create_model_can_keep_vision_but_omit_audio(
    loaded_model: torch.nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vision_config = object()
    config = SimpleNamespace(vision_config=vision_config, audio_config=object())
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoConfig.from_pretrained",
        Mock(return_value=config),
    )
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoProcessor.from_pretrained",
        Mock(return_value=SimpleNamespace(image_processor=FakeImageProcessor())),
    )

    model = create_model(modalities=["image", "text"])

    assert model.modalities == ("text", "image")
    assert config.vision_config is vision_config
    assert config.audio_config is None


@pytest.mark.parametrize(
    ("modalities", "exception"),
    [
        ("text", TypeError),
        (("image",), ValueError),
        (("text", "audio"), ValueError),
        (("text", "text"), ValueError),
    ],
)
def test_create_model_rejects_unsupported_modalities(
    modalities: object,
    exception: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_loader = Mock()
    model_loader = Mock()
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoConfig.from_pretrained",
        config_loader,
    )
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained",
        model_loader,
    )

    with pytest.raises(exception, match="modalities"):
        create_model(modalities=modalities)  # type: ignore[arg-type]

    config_loader.assert_not_called()
    model_loader.assert_not_called()


def test_transform_factories_reject_text_only_modalities_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_loader = Mock()
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained",
        model_loader,
    )

    with pytest.raises(ValueError, match="requires the image modality"):
        create_model_and_transforms(modalities=("text",))
    with pytest.raises(ValueError, match="return_transform=True requires"):
        create_model_from_pretrained(modalities=("text",))

    model_loader.assert_not_called()


def test_create_model_and_transforms_returns_three_elements(
    loaded_model: torch.nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = SimpleNamespace(image_processor=FakeImageProcessor())
    processor_loader = Mock(return_value=processor)
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoProcessor.from_pretrained",
        processor_loader,
    )

    result = create_model_and_transforms()

    assert isinstance(result, tuple)
    assert len(result) == 3
    model, preprocess_train, preprocess_val = result
    assert isinstance(model, EmbeddingGemmaCLIP)
    assert model.backend is loaded_model
    assert model.processor is processor
    assert callable(preprocess_train)
    assert preprocess_train is preprocess_val
    processor_loader.assert_called_once_with(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=None,
    )


def test_preprocess_uses_official_order_and_returns_float32_chw() -> None:
    image_processor = FakeImageProcessor()
    preprocess = EmbeddingGemmaPreprocess(image_processor)

    result = preprocess(Image.new("RGB", (12, 8)))

    assert result.shape == (3, 8, 12)
    assert result.dtype == torch.float32
    assert [name for name, _ in image_processor.calls] == [
        "convert",
        "resize",
        "rescale",
    ]
    assert image_processor.calls[1][1] == (16, 2520, 3, 3)


def test_preprocess_rejects_non_pil_input() -> None:
    preprocess = EmbeddingGemmaPreprocess(FakeImageProcessor())

    with pytest.raises(TypeError, match="PIL.Image.Image"):
        preprocess(torch.zeros((3, 8, 12)))  # type: ignore[arg-type]


def test_preprocess_rejects_non_rgb_processor_output() -> None:
    image_processor = FakeImageProcessor()
    image_processor.process_image = Mock(
        return_value=torch.zeros((1, 8, 12), dtype=torch.uint8)
    )

    with pytest.raises(ValueError, match="3-channel CHW"):
        EmbeddingGemmaPreprocess(image_processor)(Image.new("L", (12, 8)))


def test_rejects_unknown_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = Mock()
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained", loader
    )

    with pytest.raises(ValueError, match="Unsupported model_name"):
        create_model("unknown-model")

    loader.assert_not_called()


def test_create_model_normalizes_model_name(
    loaded_model: torch.nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = Mock(return_value=loaded_model)
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained", loader
    )

    result = create_model(" EMBEDDINGGEMMA_2 ")

    assert result.backend is loaded_model
    loader.assert_called_once()


def test_rejects_pretrained_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = Mock()
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained", loader
    )

    with pytest.raises(ValueError, match="pretrained=None"):
        create_model(pretrained=None)

    loader.assert_not_called()


@pytest.mark.parametrize("precision", ["fp16", "float16"])
def test_rejects_fp16(precision: str) -> None:
    with pytest.raises(ValueError, match="FP16 is not supported"):
        create_model(precision=precision)


def test_rejects_unsupported_factory_arguments() -> None:
    with pytest.raises(TypeError, match="Unsupported keyword.*unknown_option"):
        create_model(unknown_option=True)


def test_rejects_non_official_pretrained_checkpoint() -> None:
    with pytest.raises(ValueError, match="Unsupported pretrained checkpoint"):
        create_model(pretrained="some/other-model")


def test_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = Mock()
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained", loader
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        create_model(device="cuda")

    loader.assert_not_called()


def test_model_load_failure_is_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = OSError("checkpoint unavailable")
    loader = Mock(side_effect=cause)
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained", loader
    )

    with pytest.raises(RuntimeError, match="Failed to load") as error:
        create_model()

    assert error.value.__cause__ is cause


def test_processor_is_not_registered_and_module_operations_reach_backend() -> None:
    backend = torch.nn.Linear(2, 2)
    processor = torch.nn.Linear(2, 2)
    model = EmbeddingGemmaCLIP(backend=backend, processor=processor)

    assert model.processor is processor
    assert list(model.children()) == [backend]
    assert all("processor" not in name for name, _ in model.named_parameters())

    assert model.eval() is model
    assert not backend.training
    assert processor.training

    assert model.to(device="cpu", dtype=torch.float64) is model
    assert next(backend.parameters()).device.type == "cpu"
    assert next(backend.parameters()).dtype == torch.float64
    assert next(processor.parameters()).dtype == torch.float32
    assert model.cpu() is model


@pytest.mark.parametrize(
    "conversion",
    [
        lambda model: model.half(),
        lambda model: model.to(dtype=torch.float16),
        lambda model: model.to("cpu", torch.float16),
        lambda model: model.to(torch.empty((), dtype=torch.float16)),
    ],
)
def test_model_rejects_fp16_module_conversions(conversion: object) -> None:
    model = EmbeddingGemmaCLIP(
        backend=torch.nn.Linear(2, 2),
        processor=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="FP16 is not supported"):
        conversion(model)  # type: ignore[operator]


def test_encode_text_requires_processor_tokenizer_and_generic_encode_is_unimplemented() -> (
    None
):
    model = EmbeddingGemmaCLIP(
        backend=torch.nn.Linear(2, 2),
        processor=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="processor has no tokenizer"):
        model.encode_text(torch.zeros((1, 4), dtype=torch.long))
    with pytest.raises(NotImplementedError, match="encode"):
        model.encode("sample")


def test_factory_defaults_preserve_v02_and_use_strict_scale_mode(
    loaded_model: torch.nn.Module,
) -> None:
    model = create_model()

    assert model.backend is loaded_model
    assert model.output_dict is False
    assert model.logit_scale_mode == "strict"


def test_list_apis_do_not_load_model_or_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_loader = Mock(side_effect=AssertionError("model weights must not load"))
    processor_loader = Mock(side_effect=AssertionError("processor must not load"))
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained",
        model_loader,
    )
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoProcessor.from_pretrained",
        processor_loader,
    )

    assert list_models() == [
        "embeddinggemma-2",
        "dinov2-dinotxt",
        "siglip2-base",
        "siglip2-naflex-base",
        "qwen3-vl-embedding-2b",
    ]
    assert list_pretrained() == [
        ("embeddinggemma-2", "official"),
        ("dinov2-dinotxt", "official"),
        ("siglip2-base", "official"),
        ("siglip2-naflex-base", "official"),
        ("qwen3-vl-embedding-2b", "official"),
    ]
    assert list_pretrained(as_str=True) == [
        "embeddinggemma-2:official",
        "dinov2-dinotxt:official",
        "siglip2-base:official",
        "siglip2-naflex-base:official",
        "qwen3-vl-embedding-2b:official",
    ]
    model_loader.assert_not_called()
    processor_loader.assert_not_called()


def test_list_pretrained_requires_bool_as_str() -> None:
    with pytest.raises(TypeError, match="as_str must be a bool"):
        list_pretrained(as_str=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("pretrained", ["official", MODEL_ID])
def test_create_model_accepts_official_alias_and_existing_model_id(
    loaded_model: torch.nn.Module,
    monkeypatch: pytest.MonkeyPatch,
    pretrained: str,
) -> None:
    loader = Mock(return_value=loaded_model)
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained", loader
    )

    model = create_model(
        pretrained=pretrained, output_dict=True, logit_scale_mode="cosine"
    )

    assert model.output_dict is True
    assert model.logit_scale_mode == "cosine"
    assert loader.call_args.args == (MODEL_ID,)
    assert loader.call_args.kwargs["revision"] == MODEL_REVISION


def test_create_model_from_pretrained_returns_model_and_one_transform(
    loaded_model: torch.nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_loader = Mock(return_value=loaded_model)
    processor = SimpleNamespace(image_processor=FakeImageProcessor())
    processor_loader = Mock(return_value=processor)
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained",
        model_loader,
    )
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoProcessor.from_pretrained",
        processor_loader,
    )

    result = create_model_from_pretrained(pretrained="official")

    assert isinstance(result, tuple)
    model, preprocess = result
    assert isinstance(model, EmbeddingGemmaCLIP)
    assert model.backend is loaded_model
    assert isinstance(preprocess, EmbeddingGemmaPreprocess)
    model_loader.assert_called_once()
    processor_loader.assert_called_once()


def test_create_model_from_pretrained_can_return_model_only(
    loaded_model: torch.nn.Module,
) -> None:
    model = create_model_from_pretrained(return_transform=False)

    assert isinstance(model, EmbeddingGemmaCLIP)
    assert model.backend is loaded_model


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"return_transform": 1}, TypeError, "return_transform must be a bool"),
        ({"output_dict": 1}, TypeError, "output_dict must be a bool"),
        ({"logit_scale_mode": "unknown"}, ValueError, "logit_scale_mode"),
        ({"weights_only": False}, TypeError, "Unsupported keyword"),
    ],
)
def test_factory_rejects_unsupported_or_invalid_options_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    model_loader = Mock()
    monkeypatch.setattr(
        "openvlclip.backends.embeddinggemma2.AutoModel.from_pretrained",
        model_loader,
    )

    with pytest.raises(exception, match=message):
        create_model_from_pretrained(**kwargs)  # type: ignore[arg-type]

    model_loader.assert_not_called()
