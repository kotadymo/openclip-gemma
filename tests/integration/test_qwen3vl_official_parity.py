"""Opt-in parity tests against the pinned Qwen3-VL-Embedding checkpoints."""

from __future__ import annotations

import gc
import os
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from openvlclip import create_model_and_transforms
from openvlclip.backends.qwen3_vl_embedding import (
    QWEN3VL_2B_DIMENSION,
    QWEN3VL_2B_MODEL_ID,
    QWEN3VL_2B_MODEL_NAME,
    QWEN3VL_2B_REVISION,
    Qwen3VLCLIP,
)

_QWEN_SOURCE_REVISION = "393e2978d27852b0d0230d6994f37f9c15bed73c"
_DEVICE_PRECISIONS = (
    pytest.param("cpu", "fp32", id="cpu-fp32"),
    pytest.param("cpu", "bf16", id="cpu-bf16"),
    pytest.param("cuda", "fp32", id="cuda-fp32"),
    pytest.param("cuda", "bf16", id="cuda-bf16"),
)
_DEFAULT_INSTRUCTION = "Represent the user's input."


class _ReferenceProcessor(Protocol):
    def apply_chat_template(
        self,
        conversations: object,
        **kwargs: object,
    ) -> object: ...

    def __call__(self, **kwargs: object) -> object: ...


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]


def _sample_inputs() -> list[dict[str, object]]:
    images = [
        Image.new("RGB", (256, 192), (210, 30, 40)),
        Image.new("RGB", (192, 256), (20, 50, 210)),
    ]
    frames = [
        Image.new("RGB", (128, 96), (30, 90, 150)),
        Image.new("RGB", (128, 96), (60, 120, 180)),
    ]
    return [
        {"text": "a red object"},
        {"image": images[0]},
        {"text": "a blue object", "image": images[1]},
        {"video": frames},
        {
            "text": "a person walking",
            "video": frames,
            "instruction": "Represent the user's video",
        },
    ]


def _reference_conversations(
    items: Sequence[Mapping[str, object]],
) -> list[list[dict[str, object]]]:
    conversations: list[list[dict[str, object]]] = []
    for item in items:
        instruction_value = item.get("instruction")
        if instruction_value is None:
            instruction = _DEFAULT_INSTRUCTION
        elif isinstance(instruction_value, str):
            instruction = instruction_value.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith(
                "P"
            ):
                instruction = f"{instruction}."
            instruction = instruction or _DEFAULT_INSTRUCTION
        else:
            raise TypeError("Reference instructions must be strings or None.")
        content: list[dict[str, object]] = []
        video = item.get("video")
        video_kwargs: dict[str, object] = {"total_pixels": 7_864_320}
        if video is not None:
            if isinstance(video, str):
                video_content: object = video
                video_kwargs = {
                    "fps": item.get("fps") or 1.0,
                    "max_frames": item.get("max_frames") or 64,
                }
            elif isinstance(video, Sequence):
                frames = list(video)
                max_frames_value = item.get("max_frames")
                if max_frames_value is not None and (
                    isinstance(max_frames_value, bool)
                    or not isinstance(max_frames_value, int)
                ):
                    raise TypeError("Reference max_frames must be an integer.")
                max_frames = max_frames_value or 64
                if len(frames) > max_frames:
                    frame_indices = np.linspace(
                        0,
                        len(frames) - 1,
                        max_frames,
                        dtype=int,
                    )
                    frames = [frames[int(index)] for index in frame_indices]
                video_content = frames
            else:
                raise TypeError("Unsupported reference video input.")
            content.append(
                {
                    "type": "video",
                    "video": video_content,
                    **video_kwargs,
                }
            )

        image = item.get("image")
        if image is not None:
            images = image if isinstance(image, list) else [image]
            for image_item in images:
                if not isinstance(image_item, Image.Image):
                    raise TypeError("Reference images must be PIL images.")
                content.append(
                    {
                        "type": "image",
                        "image": image_item,
                        "min_pixels": 4096,
                        "max_pixels": 1_843_200,
                    }
                )
        text_value = item.get("text")
        if text_value is not None:
            texts = text_value if isinstance(text_value, list) else [text_value]
            for text_item in texts:
                if not isinstance(text_item, str):
                    raise TypeError("Reference texts must be strings.")
                content.append({"type": "text", "text": text_item})
        if not content:
            raise ValueError("Reference inputs must contain text or media.")

        conversations.append(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": instruction}],
                },
                {"role": "user", "content": content},
            ]
        )
    return conversations


def _official_reference(
    model: torch.nn.Module,
    processor: _ReferenceProcessor,
    items: Sequence[Mapping[str, object]],
    *,
    normalize: bool = False,
    truncate_dim: int | None = None,
) -> torch.Tensor:
    from qwen_vl_utils.vision_process import process_vision_info

    conversations = _reference_conversations(items)
    rendered_text = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=False,
    )
    images, video_inputs, video_kwargs = process_vision_info(
        conversations,
        image_patch_size=16,
        return_video_metadata=True,
        return_video_kwargs=True,
    )
    if video_inputs is not None:
        videos, metadata = zip(*video_inputs, strict=True)
        video_values = list(videos)
        video_metadata = list(metadata)
    else:
        video_values = None
        video_metadata = None
    processor_inputs = processor(
        text=rendered_text,
        images=images,
        videos=video_values,
        video_metadata=video_metadata,
        truncation=True,
        max_length=8192,
        padding=True,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )
    if not isinstance(processor_inputs, Mapping):
        raise TypeError("The official Qwen processor must return a mapping.")
    device = next(model.parameters()).device
    model_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in processor_inputs.items()
    }
    attention_mask = model_inputs.get("attention_mask")
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("The official Qwen processor must return an attention mask.")
    with torch.inference_mode():
        outputs = model(**model_inputs, use_cache=False)
    hidden_state = outputs.last_hidden_state
    last_positions = (
        attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
    )
    rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    features = hidden_state[rows, last_positions]
    if truncate_dim is not None:
        features = features[:, :truncate_dim]
    if normalize:
        features = F.normalize(features, p=2, dim=-1)
    return features


def _load_official_reference_model(
    model_id: str,
    revision: str,
    *,
    dtype: torch.dtype,
    device: str,
) -> torch.nn.Module:
    loaded = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=dtype,
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise TypeError("Transformers did not return reference loading details.")
    model, loading_info = loaded
    if not isinstance(model, torch.nn.Module) or not isinstance(loading_info, Mapping):
        raise TypeError("Transformers returned an invalid reference model.")
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = loading_info.get(key, ())
        if values:
            raise AssertionError(f"Official reference model has {key}: {values!r}")
    return model.to(device=device, dtype=dtype).eval()


@pytest.mark.parametrize(
    ("model_name", "model_id", "revision", "native_dimension"),
    [
        (
            QWEN3VL_2B_MODEL_NAME,
            QWEN3VL_2B_MODEL_ID,
            QWEN3VL_2B_REVISION,
            QWEN3VL_2B_DIMENSION,
        ),
    ],
)
@pytest.mark.parametrize(("device", "precision"), _DEVICE_PRECISIONS)
def test_qwen_wrapper_matches_pinned_official_processor_and_model(
    model_name: str,
    model_id: str,
    revision: str,
    native_dimension: int,
    device: str,
    precision: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    if device == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable.")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    model, _, _ = create_model_and_transforms(
        model_name,
        pretrained="official",
        precision=precision,
        device=device,
    )
    if not isinstance(model, Qwen3VLCLIP):
        raise TypeError("The Qwen registry must return a Qwen3VLCLIP model.")
    items = _sample_inputs()
    actual = model.encode_multimodal(items, normalize=False).cpu()
    dimensions = (64, 257, 1024, native_dimension)
    actual_mrl = {
        dimension: model.encode_texts(
            "a short MRL validation input",
            normalize=True,
            truncate_dim=dimension,
        ).cpu()
        for dimension in dimensions
    }
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    reference_model = _load_official_reference_model(
        model_id,
        revision,
        dtype=dtype,
        device=device,
    )
    reference_processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        padding_side="right",
    )
    expected = _official_reference(
        reference_model,
        reference_processor,
        items,
    ).cpu()
    assert actual.shape == (len(items), native_dimension)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    max_abs_error = (actual.float() - expected.float()).abs().max().item()
    print(
        f"{_QWEN_SOURCE_REVISION} {model_name} {device}/{precision} "
        f"global max_abs_error={max_abs_error:.8g}"
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    for dimension in dimensions:
        expected_mrl = _official_reference(
            reference_model,
            reference_processor,
            [{"text": "a short MRL validation input"}],
            normalize=True,
            truncate_dim=dimension,
        ).cpu()
        observed_mrl = actual_mrl[dimension]
        assert observed_mrl.shape == (1, dimension)
        assert torch.isfinite(observed_mrl).all()
        mrl_max_abs_error = (
            (observed_mrl.float() - expected_mrl.float()).abs().max().item()
        )
        print(
            f"{_QWEN_SOURCE_REVISION} {model_name} {device}/{precision} "
            f"MRL-{dimension} max_abs_error={mrl_max_abs_error:.8g}"
        )
        torch.testing.assert_close(
            observed_mrl,
            expected_mrl,
            rtol=1e-5,
            atol=1e-6,
        )
