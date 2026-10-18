"""Run the v0.3.0 OpenCLIP-style APIs with a synthetic or supplied image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

import openclip_gemma as open_clip
from openclip_gemma.constants import MODEL_NAME

DEFAULT_TEXTS = [
    "a red circle on a blue background",
    "青い背景にある赤い円",
]


def _synthetic_image() -> Image.Image:
    image = Image.new("RGB", (256, 192), (30, 80, 180))
    ImageDraw.Draw(image).ellipse((72, 40, 184, 152), fill=(220, 35, 45))
    return image


def _load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def _max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left.float() - right.float()).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional input image; a deterministic synthetic image is used by default.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face cache directory.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but CUDA is unavailable.")
    if (
        args.device == "cuda"
        and args.precision == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        parser.error("CUDA BF16 was requested, but the current GPU does not support it.")

    models = open_clip.list_models()
    pretrained = open_clip.list_pretrained()
    pretrained_strings = open_clip.list_pretrained(as_str=True)
    if MODEL_NAME not in models or (MODEL_NAME, "official") not in pretrained:
        raise RuntimeError("The expected EmbeddingGemma 2 factory entry is missing.")

    cache_dir = str(args.cache_dir) if args.cache_dir is not None else None
    model, preprocess = open_clip.create_model_from_pretrained(
        MODEL_NAME,
        pretrained="official",
        precision=args.precision,
        device=args.device,
        cache_dir=cache_dir,
        output_dict=True,
        logit_scale_mode="cosine",
    )
    model.eval()

    image = _load_image(args.image) if args.image is not None else _synthetic_image()
    image_tensor = preprocess(image).unsqueeze(0).to(args.device)
    tokenizer = open_clip.get_tokenizer(
        MODEL_NAME,
        prompt_name="SearchQuery",
        cache_dir=cache_dir,
    )
    text_tokens = tokenizer(DEFAULT_TEXTS).to(args.device)

    if args.device == "cuda":
        torch.cuda.synchronize()
        allocated_before_inference = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        allocated_before_inference = None

    with torch.inference_mode():
        dict_output = model(image_tensor, text_tokens)
        if not isinstance(dict_output, dict):
            raise TypeError("forward() did not return a dictionary.")

        model.output_dict = False
        tuple_output = model(image_tensor, text_tokens)
        if not isinstance(tuple_output, tuple) or len(tuple_output) != 3:
            raise RuntimeError("forward() did not return the expected three-item tuple.")
        model.output_dict = True

        image_logits, text_logits = model.get_logits(image_tensor, text_tokens)

    image_features = dict_output.get("image_features")
    text_features = dict_output.get("text_features")
    logit_scale = dict_output.get("logit_scale")
    if not isinstance(image_features, torch.Tensor):
        raise TypeError("Dictionary output has no image_features tensor.")
    if not isinstance(text_features, torch.Tensor):
        raise TypeError("Dictionary output has no text_features tensor.")
    if not isinstance(logit_scale, torch.Tensor):
        raise TypeError("Dictionary output has no logit_scale tensor.")
    tuple_image_features, tuple_text_features, tuple_scale = tuple_output
    if not all(
        isinstance(value, torch.Tensor)
        for value in (tuple_image_features, tuple_text_features, tuple_scale)
    ):
        raise TypeError("Tuple output contains an unexpected None value.")
    if image_logits.shape != (image_tensor.shape[0], text_tokens.shape[0]):
        raise RuntimeError(f"Unexpected image logits shape: {tuple(image_logits.shape)}.")
    if text_logits.shape != (text_tokens.shape[0], image_tensor.shape[0]):
        raise RuntimeError(f"Unexpected text logits shape: {tuple(text_logits.shape)}.")

    if args.device == "cpu" and image_features.dtype == torch.bfloat16:
        direct_logits = image_features.float() @ text_features.float().T
        direct_logits = direct_logits.to(dtype=image_features.dtype)
    else:
        direct_logits = image_features @ text_features.T
    logits_max_abs_error = _max_abs_difference(image_logits, direct_logits)

    cuda_memory: dict[str, int] | None = None
    if args.device == "cuda":
        torch.cuda.synchronize()
        cuda_memory = {
            "allocated_before_inference_bytes": allocated_before_inference,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        }

    print(
        json.dumps(
            {
                "models": models,
                "pretrained": pretrained,
                "pretrained_as_strings": pretrained_strings,
                "image_source": str(args.image) if args.image is not None else "synthetic",
                "device": args.device,
                "precision": args.precision,
                "image_features_shape": list(image_features.shape),
                "text_features_shape": list(text_features.shape),
                "image_features_dtype": str(image_features.dtype),
                "text_features_dtype": str(text_features.dtype),
                "image_features_device": str(image_features.device),
                "text_features_device": str(text_features.device),
                "logit_scale": logit_scale.item(),
                "tuple_dict_image_max_abs_difference": _max_abs_difference(
                    tuple_image_features,
                    image_features,
                ),
                "tuple_dict_text_max_abs_difference": _max_abs_difference(
                    tuple_text_features,
                    text_features,
                ),
                "tuple_dict_scale_max_abs_difference": _max_abs_difference(
                    tuple_scale,
                    logit_scale,
                ),
                "get_logits_vs_cosine_max_abs_difference": logits_max_abs_error,
                "image_logits_shape": list(image_logits.shape),
                "text_logits_shape": list(text_logits.shape),
                "similarities_are_probabilities": False,
                "cuda_memory": cuda_memory,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
