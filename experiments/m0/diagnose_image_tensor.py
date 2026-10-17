"""Diagnose exact image-tensor preprocessing for EmbeddingGemma 2."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Mapping
from typing import Any

import torch
import transformers
from PIL import Image, ImageDraw, ImageOps
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor

from validate import (
    MODEL_ID,
    MODEL_REVISION,
    compare_tensors,
    move_features,
    pool_and_normalize,
)

MAX_ABS_ERROR_LIMIT = 1e-4
COSINE_SIMILARITY_LIMIT = 0.9999


def create_images() -> dict[str, Image.Image]:
    rgb = Image.new("RGB", (256, 192), (30, 80, 180))
    ImageDraw.Draw(rgb).ellipse((72, 40, 184, 152), fill=(220, 35, 45))
    return {
        "m0_rgb": rgb,
        "grayscale": ImageOps.grayscale(rgb),
        "wide_rgb": rgb.resize((384, 128), Image.Resampling.BICUBIC),
        "tall_rgb": rgb.resize((128, 384), Image.Resampling.BICUBIC),
    }


def compare_feature_mappings(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for name in sorted(set(reference) | set(candidate)):
        if name not in reference or name not in candidate:
            comparisons[name] = {"field_present_in_both": False}
            continue
        first = reference[name]
        second = candidate[name]
        if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
            shape_matches = tuple(first.shape) == tuple(second.shape)
            comparisons[name] = {
                "field_present_in_both": True,
                "shape_matches": shape_matches,
                "reference_shape": list(first.shape),
                "candidate_shape": list(second.shape),
                "exact_match": torch.equal(first, second) if shape_matches else False,
                "max_abs_error": (
                    (first.float() - second.float()).abs().max().item()
                    if shape_matches and first.is_floating_point()
                    else 0.0 if shape_matches else None
                ),
            }
        else:
            comparisons[name] = {
                "field_present_in_both": True,
                "equal": first == second,
                "reference": first,
                "candidate": second,
            }
    return comparisons


def processor_config(processor: Any) -> dict[str, Any]:
    image_processor = processor.image_processor
    return {
        "do_convert_rgb": image_processor.do_convert_rgb,
        "do_resize": image_processor.do_resize,
        "resample": str(image_processor.resample),
        "do_rescale": image_processor.do_rescale,
        "rescale_factor": image_processor.rescale_factor,
        "do_normalize": image_processor.do_normalize,
        "image_mean": image_processor.image_mean,
        "image_std": image_processor.image_std,
        "patch_size": image_processor.patch_size,
        "pooling_kernel_size": image_processor.pooling_kernel_size,
        "max_soft_tokens": image_processor.max_soft_tokens,
        "max_patches": (
            image_processor.max_soft_tokens * image_processor.pooling_kernel_size**2
        ),
        "boi_token_id": processor.tokenizer.convert_tokens_to_ids(
            processor.boi_token
        ),
        "image_token_id": processor.tokenizer.image_token_id,
        "eoi_token_id": processor.tokenizer.convert_tokens_to_ids(
            processor.eoi_token
        ),
    }


def official_tensor_preprocess(
    image: Image.Image,
    processor: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Use official conversion, resize, and rescale methods and return CHW float pixels."""
    image_processor = processor.image_processor
    config = processor_config(processor)
    prepared = image_processor.process_image(
        image,
        do_convert_rgb=config["do_convert_rgb"],
    )
    converted_shape = list(prepared.shape)
    converted_dtype = str(prepared.dtype)
    converted_range = [
        int(prepared.min().item()),
        int(prepared.max().item()),
    ]
    resized = image_processor.aspect_ratio_preserving_resize(
        image=prepared,
        patch_size=config["patch_size"],
        max_patches=config["max_patches"],
        pooling_kernel_size=config["pooling_kernel_size"],
        resample=image_processor.resample,
    )
    resized_shape = list(resized.shape)
    resized_dtype = str(resized.dtype)
    resized_range = [
        float(resized.min().item()),
        float(resized.max().item()),
    ]
    scaled = image_processor.rescale_and_normalize(
        resized,
        do_rescale=config["do_rescale"],
        rescale_factor=config["rescale_factor"],
        do_normalize=config["do_normalize"],
        image_mean=config["image_mean"],
        image_std=config["image_std"],
    )
    details = {
        "input_mode": image.mode,
        "input_size_wh": list(image.size),
        "converted_chw_shape": converted_shape,
        "converted_dtype": converted_dtype,
        "converted_range": converted_range,
        "resized_chw_shape": resized_shape,
        "resized_dtype": resized_dtype,
        "resized_range": resized_range,
        "output_chw_shape": list(scaled.shape),
        "output_dtype": str(scaled.dtype),
        "output_range": [
            float(scaled.min().item()),
            float(scaled.max().item()),
        ],
        "pixel_rescale_applied": config["do_rescale"],
        "pixel_rescale_factor": config["rescale_factor"],
    }
    return scaled, details


def inference_from_features(
    model: torch.nn.Module,
    features: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    moved = move_features(features, device)
    outputs = model(**moved)
    embedding, _ = pool_and_normalize(outputs, moved)
    return embedding


def mapping_bytes(features: Mapping[str, Any]) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in features.values()
        if isinstance(value, torch.Tensor)
    )


def median_ms(operation: Any, repeats: int = 5) -> float:
    operation()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def tensor_image_variants(image: Image.Image) -> dict[str, torch.Tensor]:
    chw_u8 = torch.frombuffer(
        bytearray(image.tobytes()),
        dtype=torch.uint8,
    ).reshape(image.height, image.width, len(image.getbands())).permute(2, 0, 1)
    return {
        "uint8_chw_0_255": chw_u8,
        "float_chw_0_255": chw_u8.float(),
        "float_chw_0_1": chw_u8.float() / 255,
        "float_hwc_0_255": chw_u8.permute(1, 2, 0).float(),
    }


def tolerance_result(comparison: Mapping[str, Any]) -> dict[str, Any]:
    max_abs_error = comparison["max_abs_error"]
    cosines = comparison["cosine_similarity_per_row"]
    return {
        "max_abs_error_limit": MAX_ABS_ERROR_LIMIT,
        "cosine_similarity_limit": COSINE_SIMILARITY_LIMIT,
        "max_abs_error_pass": max_abs_error is not None
        and max_abs_error <= MAX_ABS_ERROR_LIMIT,
        "cosine_similarity_pass": cosines is not None
        and all(value >= COSINE_SIMILARITY_LIMIT for value in cosines),
        "overall_pass": comparison.get("shape_matches", False)
        and max_abs_error is not None
        and max_abs_error <= MAX_ABS_ERROR_LIMIT
        and cosines is not None
        and all(value >= COSINE_SIMILARITY_LIMIT for value in cosines),
    }


def run_diagnostics() -> dict[str, Any]:
    device = torch.device("cpu")
    dtype = torch.float32
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
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
    ).to(device).eval()

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": str(device),
        "dtype": str(dtype),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "processor_class": type(processor).__name__,
        "processor_config": processor_config(processor),
        "thresholds": {
            "max_abs_error": MAX_ABS_ERROR_LIMIT,
            "cosine_similarity": COSINE_SIMILARITY_LIMIT,
        },
        "images": {},
    }

    prepared_tensors: dict[str, torch.Tensor] = {}
    pil_images = create_images()
    for image_name, pil_image in create_images().items():
        with torch.inference_mode():
            reference_features = processor(images=pil_image, return_tensors="pt")
            high_level = sentence_model.encode(
                [{"image": pil_image}],
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            reference_low_level = inference_from_features(
                model,
                reference_features,
                device,
            )

            prepared_tensor, preprocessing_details = official_tensor_preprocess(
                pil_image,
                processor,
            )
            prepared_tensors[image_name] = prepared_tensor
            exact_rebuilt_features = processor(
                images=[prepared_tensor],
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            )
            exact_tensor_embedding = inference_from_features(
                model,
                exact_rebuilt_features,
                device,
            )
            exact_mapping_comparison = compare_feature_mappings(
                reference_features,
                exact_rebuilt_features,
            )
            exact_embedding_comparison = compare_tensors(
                high_level,
                exact_tensor_embedding,
                check_fp32_tolerance=True,
            )

            tensor_routes: dict[str, Any] = {}
            for route_name, tensor in tensor_image_variants(pil_image).items():
                try:
                    candidate_features = processor(
                        images=[tensor],
                        return_tensors="pt",
                    )
                    tensor_embedding = inference_from_features(
                        model,
                        candidate_features,
                        device,
                    )
                    feature_comparison = compare_feature_mappings(
                        reference_features,
                        candidate_features,
                    )
                    embedding_comparison = compare_tensors(
                        high_level,
                        tensor_embedding,
                        check_fp32_tolerance=True,
                    )
                    tensor_routes[route_name] = {
                        "input_shape": list(tensor.shape),
                        "input_dtype": str(tensor.dtype),
                        "input_range": [
                            float(tensor.min().item()),
                            float(tensor.max().item()),
                        ],
                        "feature_comparison": feature_comparison,
                        "embedding_comparison": embedding_comparison,
                        "thresholds": tolerance_result(embedding_comparison),
                    }
                except (TypeError, ValueError, RuntimeError) as error:
                    tensor_routes[route_name] = {
                        "input_shape": list(tensor.shape),
                        "input_dtype": str(tensor.dtype),
                        "input_range": [
                            float(tensor.min().item()),
                            float(tensor.max().item()),
                        ],
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }

            double_processed_features = processor(
                images=[prepared_tensor],
                return_tensors="pt",
            )
            double_processed_comparison = compare_feature_mappings(
                reference_features,
                double_processed_features,
            )
            double_processed_embedding = inference_from_features(
                model,
                double_processed_features,
                device,
            )
            double_processed_embedding_comparison = compare_tensors(
                high_level,
                double_processed_embedding,
                check_fp32_tolerance=True,
            )

            # Rebuild model inputs using only the returned Tensor and its spatial dimensions.
            # The official processor still creates the multimodal token sequence and patch metadata.
            tensor_only_mapping = processor(
                images=[prepared_tensor],
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            )
            reconstructed_tensor_embedding = inference_from_features(
                model,
                tensor_only_mapping,
                device,
            )

        report["images"][image_name] = {
            "input": {
                "mode": pil_image.mode,
                "size_wh": list(pil_image.size),
            },
            "official_preprocess_stages": preprocessing_details,
            "official_pil_feature_shapes": {
                name: list(value.shape)
                for name, value in reference_features.items()
                if isinstance(value, torch.Tensor)
            },
            "patch_layout": {
                "image_token_count": int(
                    reference_features["input_ids"]
                    .eq(processor.tokenizer.image_token_id)
                    .sum()
                    .item()
                ),
                "position_id_shape": list(
                    reference_features["image_position_ids"].shape
                ),
                "valid_patch_positions": int(
                    (reference_features["image_position_ids"][0, :, 0] >= 0)
                    .sum()
                    .item()
                ),
                "padded_patch_positions": int(
                    (reference_features["image_position_ids"][0, :, 0] < 0)
                    .sum()
                    .item()
                ),
                "first_position_ids": (
                    reference_features["image_position_ids"][0, :5]
                    .tolist()
                ),
                "last_valid_position_ids": reference_features[
                    "image_position_ids"
                ][
                    0,
                    max(
                        0,
                        int(
                            (
                                reference_features["image_position_ids"][
                                    0, :, 0
                                ]
                                >= 0
                            )
                            .sum()
                            .item()
                        )
                        - 5,
                    ) : int(
                        (
                            reference_features["image_position_ids"][0, :, 0] >= 0
                        )
                        .sum()
                        .item()
                    ),
                ].tolist(),
            },
            "tensor_contract_storage": {
                "preprocess_tensor_bytes": (
                    prepared_tensor.numel() * prepared_tensor.element_size()
                ),
                "processor_mapping_bytes": mapping_bytes(reference_features),
                "both_live_bytes_lower_bound": (
                    prepared_tensor.numel() * prepared_tensor.element_size()
                    + mapping_bytes(reference_features)
                ),
            },
            "resize_reproduction": {
                "feature_mapping_differences": exact_mapping_comparison,
                "max_abs_error_pixel_values": exact_mapping_comparison[
                    "pixel_values"
                ]["max_abs_error"],
                "max_abs_error_position_ids": exact_mapping_comparison[
                    "image_position_ids"
                ]["max_abs_error"],
            },
            "embedding_comparisons": {
                "high_level_vs_official_pil_low_level": compare_tensors(
                    high_level,
                    reference_low_level,
                    check_fp32_tolerance=True,
                ),
                "high_level_vs_preprocess_tensor_mapping": exact_embedding_comparison,
                "preprocess_tensor_mapping_thresholds": tolerance_result(
                    exact_embedding_comparison
                ),
                "preprocess_tensor_reconstructed_mapping": compare_tensors(
                    high_level,
                    reconstructed_tensor_embedding,
                    check_fp32_tolerance=True,
                ),
                "double_preprocess_default_processor": {
                    "feature_mapping_differences": double_processed_comparison,
                    "embedding": double_processed_embedding_comparison,
                    "thresholds": tolerance_result(
                        double_processed_embedding_comparison
                    ),
                },
            },
            "raw_tensor_input_routes": tensor_routes,
        }

    processed_tensors = list(prepared_tensors.values())
    try:
        torch.stack(processed_tensors)
        report["variable_aspect_ratio_batch"] = {
            "stacked": True,
            "shapes": [list(tensor.shape) for tensor in processed_tensors],
        }
    except RuntimeError as error:
        report["variable_aspect_ratio_batch"] = {
            "stacked": False,
            "shapes": [list(tensor.shape) for tensor in processed_tensors],
            "error_type": type(error).__name__,
            "error": str(error),
        }

    image_names = list(pil_images)
    pil_image_batch = [pil_images[name] for name in image_names]
    tensor_image_batch = [prepared_tensors[name] for name in image_names]
    image_placeholders = [processor.image_token] * len(image_names)
    with torch.inference_mode():
        reference_batch_features = processor(
            images=[[image] for image in pil_image_batch],
            text=image_placeholders,
            padding=True,
            return_tensors="pt",
        )
        tensor_batch_features = processor(
            images=[[image] for image in tensor_image_batch],
            text=image_placeholders,
            padding=True,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        high_level_batch = sentence_model.encode(
            [{"image": image} for image in pil_image_batch],
            batch_size=len(pil_image_batch),
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        tensor_batch_embedding = inference_from_features(
            model,
            tensor_batch_features,
            device,
        )
    batch_mapping_comparison = compare_feature_mappings(
        reference_batch_features,
        tensor_batch_features,
    )
    batch_embedding_comparison = compare_tensors(
        high_level_batch,
        tensor_batch_embedding,
        check_fp32_tolerance=True,
    )
    report["variable_aspect_ratio_tensor_list_batch"] = {
        "image_names": image_names,
        "reference_pil_mapping_shapes": {
            name: list(value.shape)
            for name, value in reference_batch_features.items()
            if isinstance(value, torch.Tensor)
        },
        "tensor_mapping_shapes": {
            name: list(value.shape)
            for name, value in tensor_batch_features.items()
            if isinstance(value, torch.Tensor)
        },
        "feature_comparison": batch_mapping_comparison,
        "high_level_vs_tensor_list_embedding": batch_embedding_comparison,
        "thresholds": tolerance_result(batch_embedding_comparison),
        "note": (
            "Variable-sized CHW tensors are passed as a list; this is not the "
            "OpenCLIP torch.Tensor[B,C,H,W] batch contract."
        ),
    }

    fixed_shape_names = ["m0_rgb", "grayscale"]
    fixed_shape_tensor_batch = torch.stack(
        [prepared_tensors[name] for name in fixed_shape_names]
    )
    fixed_shape_pil_images = [pil_images[name] for name in fixed_shape_names]
    fixed_shape_placeholders = [processor.image_token] * len(fixed_shape_names)
    with torch.inference_mode():
        fixed_shape_reference = processor(
            images=[[image] for image in fixed_shape_pil_images],
            text=fixed_shape_placeholders,
            padding=True,
            return_tensors="pt",
        )
        fixed_shape_tensor_features = processor(
            images=[[image] for image in fixed_shape_tensor_batch.unbind(0)],
            text=fixed_shape_placeholders,
            padding=True,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        fixed_shape_high_level = sentence_model.encode(
            [{"image": image} for image in fixed_shape_pil_images],
            batch_size=len(fixed_shape_pil_images),
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        fixed_shape_tensor_embedding = inference_from_features(
            model,
            fixed_shape_tensor_features,
            device,
        )
    fixed_shape_batch_comparison = compare_tensors(
        fixed_shape_high_level,
        fixed_shape_tensor_embedding,
        check_fp32_tolerance=True,
    )
    report["fixed_shape_tensor_batch"] = {
        "image_names": fixed_shape_names,
        "tensor_shape": list(fixed_shape_tensor_batch.shape),
        "feature_comparison": compare_feature_mappings(
            fixed_shape_reference,
            fixed_shape_tensor_features,
        ),
        "high_level_vs_tensor_batch_embedding": fixed_shape_batch_comparison,
        "thresholds": tolerance_result(fixed_shape_batch_comparison),
    }

    baseline_image = create_images()["m0_rgb"]
    baseline_tensor, _ = official_tensor_preprocess(baseline_image, processor)
    report["cpu_preprocessing_timing_median_ms"] = {
        "official_pil_processor_complete_mapping": median_ms(
            lambda: processor(images=baseline_image, return_tensors="pt")
        ),
        "tensor_preprocess_resize_and_rescale": median_ms(
            lambda: official_tensor_preprocess(baseline_image, processor)
        ),
        "tensor_to_feature_mapping_patchify_and_tokenize": median_ms(
            lambda: processor(
                images=[baseline_tensor],
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            )
        ),
        "tensor_preprocess_plus_mapping_pipeline": median_ms(
            lambda: processor(
                images=[
                    official_tensor_preprocess(baseline_image, processor)[0]
                ],
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            )
        ),
        "measurement_repeats": 5,
        "note": "CPU wall time only; excludes model forward and is not a benchmark.",
    }
    return report


def summarize(report: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for image_name, image_result in report["images"].items():
        summary[image_name] = {
            "mode": image_result["input"]["mode"],
            "size_wh": image_result["input"]["size_wh"],
            "official_preprocess_tensor": image_result[
                "official_preprocess_stages"
            ]["output_chw_shape"],
            "preprocessed_tensor_error": image_result[
                "embedding_comparisons"
            ]["high_level_vs_preprocess_tensor_mapping"]["max_abs_error"],
            "preprocessed_tensor_cosine": image_result[
                "embedding_comparisons"
            ]["high_level_vs_preprocess_tensor_mapping"][
                "cosine_similarity_per_row"
            ],
            "preprocessed_tensor_pass": image_result[
                "embedding_comparisons"
            ]["preprocess_tensor_mapping_thresholds"]["overall_pass"],
            "raw_tensor_routes": {
                name: (
                    route["embedding_comparison"]["max_abs_error"],
                    route["embedding_comparison"]["cosine_similarity_per_row"],
                )
                if "embedding_comparison" in route
                else route["error_type"]
                for name, route in image_result["raw_tensor_input_routes"].items()
            },
            "valid_and_padded_patches": (
                image_result["patch_layout"]["valid_patch_positions"],
                image_result["patch_layout"]["padded_patch_positions"],
            ),
            "all_processor_fields_exact": all(
                field.get("exact_match", field.get("equal", True))
                for field in image_result["resize_reproduction"][
                    "feature_mapping_differences"
                ].values()
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()
    started = time.perf_counter()
    report = run_diagnostics()
    report["elapsed_seconds"] = time.perf_counter() - started
    report["summary"] = summarize(report)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
