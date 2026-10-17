"""Compare official Sentence Transformers and Transformers inference paths."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
TEXTS = [
    "A small red circle on a blue background.",
    "The image contains a simple geometric shape.",
]
MASK_TEXTS = [
    "cat",
    "A much longer sentence used to ensure that right padding is observable in this batch.",
]
PROMPT_NAMES = ("SearchQuery", "Classification")
MAX_ABS_ERROR_LIMIT = 1e-4
COSINE_SIMILARITY_LIMIT = 0.9999


def create_sample_image() -> Image.Image:
    image = Image.new("RGB", (256, 192), (30, 80, 180))
    draw = ImageDraw.Draw(image)
    draw.ellipse((72, 40, 184, 152), fill=(220, 35, 45))
    return image


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    return device


def move_features(features: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for name, value in features.items():
        moved[name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def tensor_details(
    value: torch.Tensor,
    *,
    include_row_norms: bool = True,
) -> dict[str, Any]:
    value_fp32 = value.detach().float()
    details: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": bool(torch.isfinite(value_fp32).all().item()),
    }
    if include_row_norms:
        norms = value_fp32.norm(dim=-1) if value.ndim > 1 else value_fp32.norm()
        details["row_l2_norms"] = norms.cpu().tolist()
    return details


def compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    check_fp32_tolerance: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reference": tensor_details(reference),
        "candidate": tensor_details(candidate),
        "shape_matches": list(reference.shape) == list(candidate.shape),
        "max_abs_error": None,
        "cosine_similarity_per_row": None,
        "within_provisional_tolerance": None,
    }
    if not result["shape_matches"]:
        return result

    reference_fp32 = reference.detach().float()
    candidate_fp32 = candidate.detach().float()
    max_abs_error = (reference_fp32 - candidate_fp32).abs().max().item()
    cosine = F.cosine_similarity(reference_fp32, candidate_fp32, dim=-1).clamp(-1, 1)
    result["max_abs_error"] = max_abs_error
    result["cosine_similarity_per_row"] = cosine.cpu().tolist()
    if check_fp32_tolerance:
        result["within_provisional_tolerance"] = (
            max_abs_error <= MAX_ABS_ERROR_LIMIT
            and bool((cosine >= COSINE_SIMILARITY_LIMIT).all().item())
        )
    return result


def pool_and_normalize(
    outputs: Any,
    model_inputs: Mapping[str, Any],
) -> tuple[torch.Tensor, str]:
    hidden = outputs.last_hidden_state
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        mask_source = "all sequence positions; processor returned no attention_mask"
    else:
        if tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
            raise ValueError(
                "attention_mask shape does not match last_hidden_state: "
                f"{tuple(attention_mask.shape)} != {tuple(hidden.shape[:2])}"
            )
        mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
        mask_source = "processor attention_mask"

    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    return F.normalize(pooled, p=2, dim=-1), mask_source


def processor_tensor_details(features: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: tensor_details(value, include_row_norms=False)
        for name, value in features.items()
        if isinstance(value, torch.Tensor)
    }


def run_validation(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    sentence_model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device=str(device),
    )
    sentence_model.to(device=device, dtype=dtype)
    sentence_model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    transformer_model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
    )
    transformer_model.to(device=device)
    transformer_model.eval()

    configured_prompts = dict(sentence_model.prompts)
    missing_prompts = [name for name in PROMPT_NAMES if name not in configured_prompts]
    if missing_prompts:
        raise ValueError(
            f"Required official prompts are missing: {missing_prompts}; "
            f"available prompts: {sorted(configured_prompts)}"
        )

    result: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": str(device),
        "dtype": str(dtype),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sentence_transformers_version": __import__(
            "sentence_transformers"
        ).__version__,
        "processor_class": type(processor).__name__,
        "model_class": type(transformer_model).__name__,
        "model_prompts": configured_prompts,
        "image_inputs": {},
        "text_comparisons": {},
        "tokenizer_mask_recovery": {},
        "fp32_tolerance_applies": dtype == torch.float32,
    }

    image = create_sample_image()
    with torch.inference_mode():
        sentence_image = sentence_model.encode(
            [{"image": image}],
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        image_features = processor(images=image, return_tensors="pt")
        image_tensor = torch.frombuffer(
            bytearray(image.tobytes()),
            dtype=torch.uint8,
        ).reshape(image.height, image.width, 3).permute(2, 0, 1)
        tensor_image_features = processor(
            images=[image_tensor.float()],
            return_tensors="pt",
        )
        if set(image_features) != set(tensor_image_features):
            raise ValueError(
                "Processor returned different fields for PIL and Tensor images: "
                f"{sorted(image_features)} != {sorted(tensor_image_features)}"
            )
        tensor_processor_comparison = {
            name: {
                "shape_matches": tuple(image_features[name].shape)
                == tuple(tensor_image_features[name].shape),
                "exact_match": torch.equal(
                    image_features[name],
                    tensor_image_features[name],
                ),
                "max_abs_error": (
                    (image_features[name].float() - tensor_image_features[name].float())
                    .abs()
                    .max()
                    .item()
                    if image_features[name].is_floating_point()
                    else None
                ),
            }
            for name in image_features
        }
        moved_image_features = move_features(image_features, device)
        image_outputs = transformer_model(**moved_image_features)
        low_level_image, image_mask_source = pool_and_normalize(
            image_outputs, moved_image_features
        )
        moved_tensor_image_features = move_features(tensor_image_features, device)
        tensor_image_outputs = transformer_model(**moved_tensor_image_features)
        low_level_tensor_image, tensor_image_mask_source = pool_and_normalize(
            tensor_image_outputs, moved_tensor_image_features
        )

        result["image_inputs"] = {
            "processor_tensors": processor_tensor_details(image_features),
            "model_input_keys": sorted(image_features),
            "raw_image_tensor": {
                "shape": list(image_tensor.shape),
                "dtype": str(image_tensor.float().dtype),
                "value_range": [
                    float(image_tensor.min().item()),
                    float(image_tensor.max().item()),
                ],
            },
            "pil_vs_raw_tensor_processor_comparison": tensor_processor_comparison,
            "pooling_mask_source": image_mask_source,
            "raw_tensor_pooling_mask_source": tensor_image_mask_source,
            "sentence_transformers_embedding": tensor_details(sentence_image),
            "auto_model_embedding": tensor_details(low_level_image),
            "comparison": compare_tensors(
                sentence_image,
                low_level_image,
                check_fp32_tolerance=dtype == torch.float32,
            ),
            "raw_tensor_auto_model_embedding": tensor_details(low_level_tensor_image),
            "raw_tensor_embedding_comparison": compare_tensors(
                sentence_image,
                low_level_tensor_image,
                check_fp32_tolerance=dtype == torch.float32,
            ),
        }

        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("The official tokenizer has no pad_token_id.")
        mask_checks: dict[str, Any] = {}
        for prompt_name in (None, *PROMPT_NAMES):
            prefix = "" if prompt_name is None else configured_prompts[prompt_name]
            mask_features = processor(
                text=[prefix + text for text in MASK_TEXTS],
                padding=True,
                return_tensors="pt",
            )
            input_ids = mask_features["input_ids"]
            attention_mask = mask_features["attention_mask"]
            recovered_mask = (input_ids != pad_token_id).to(
                dtype=attention_mask.dtype
            )
            result_key = "none" if prompt_name is None else prompt_name
            mask_checks[result_key] = {
                "input_ids_shape": list(input_ids.shape),
                "attention_mask_shape": list(attention_mask.shape),
                "input_ids": input_ids.cpu().tolist(),
                "attention_mask": attention_mask.cpu().tolist(),
                "mask_recovered_from_ids_and_pad_id": recovered_mask.cpu().tolist(),
                "exact_match": bool(torch.equal(recovered_mask, attention_mask)),
                "padded_positions": int((attention_mask == 0).sum().item()),
                "valid_pad_id_positions": int(
                    ((input_ids == pad_token_id) & (attention_mask != 0))
                    .sum()
                    .item()
                ),
            }
        result["tokenizer_mask_recovery"] = {
            "pad_token_id": pad_token_id,
            "padding_side": processor.tokenizer.padding_side,
            "by_prompt": mask_checks,
            "all_exact_matches": all(
                check["exact_match"] for check in mask_checks.values()
            ),
        }

        text_embeddings_by_prompt: dict[str, torch.Tensor] = {}
        for prompt_name in (None, *PROMPT_NAMES):
            if prompt_name is None:
                sentence_text = sentence_model.encode(
                    TEXTS,
                    prompt="",
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                prefix = ""
                result_key = "none"
            else:
                sentence_text = sentence_model.encode(
                    TEXTS,
                    prompt_name=prompt_name,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                prefix = configured_prompts[prompt_name]
                result_key = prompt_name

            prompted_texts = [prefix + text for text in TEXTS]
            text_features = processor(
                text=prompted_texts,
                padding=True,
                return_tensors="pt",
            )
            moved_text_features = move_features(text_features, device)
            text_outputs = transformer_model(**moved_text_features)
            low_level_text, text_mask_source = pool_and_normalize(
                text_outputs, moved_text_features
            )
            result["text_comparisons"][result_key] = {
                "prompt": prefix,
                "pooling_mask_source": text_mask_source,
                "sentence_transformers_embedding": tensor_details(sentence_text),
                "auto_model_embedding": tensor_details(low_level_text),
                "comparison": compare_tensors(
                    sentence_text,
                    low_level_text,
                    check_fp32_tolerance=dtype == torch.float32,
                ),
            }
            text_embeddings_by_prompt[result_key] = sentence_text

        result["prompt_embedding_cosines"] = {}
        prompt_keys = list(text_embeddings_by_prompt)
        for first_index, first_prompt in enumerate(prompt_keys):
            for second_prompt in prompt_keys[first_index + 1 :]:
                similarity = F.cosine_similarity(
                    text_embeddings_by_prompt[first_prompt].float(),
                    text_embeddings_by_prompt[second_prompt].float(),
                    dim=-1,
                ).clamp(-1, 1)
                result["prompt_embedding_cosines"][
                    f"{first_prompt} vs {second_prompt}"
                ] = similarity.cpu().tolist()
        result["image_text_cosine_scores"] = {
            prompt: torch.matmul(
                embedding.float(),
                sentence_image.float().transpose(0, 1),
            )
            .squeeze(-1)
            .cpu()
            .tolist()
            for prompt, embedding in text_embeddings_by_prompt.items()
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Defaults to CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Model dtype. FP32 is the baseline; BF16 is a separate experiment.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON report path. No model weights or outputs are saved by default.",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    report = run_validation(device, dtype)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as report_file:
            report_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
