"""Measure MRL embedding latency, storage size, and optional paired recall."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import warnings
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from platform import processor, python_version, release, system
from typing import Any

import torch
from PIL import Image

from openclip_gemma import create_model_and_transforms, get_tokenizer
from openclip_gemma._dimensions import SUPPORTED_EMBEDDING_DIMENSIONS
from openclip_gemma.constants import MODEL_ID, MODEL_REVISION

_SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer"
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _image_paths(
    images_dir: Path,
    *,
    recursive: bool,
) -> list[Path]:
    try:
        root = images_dir.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Image directory does not exist: {images_dir}") from error
    if not root.is_dir():
        raise ValueError(f"Image path is not a directory: {root}")
    candidates = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(
        (
            path
            for path in candidates
            if not path.is_symlink()
            and path.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES
            and path.is_file()
        ),
        key=lambda path: (
            os.path.normcase(path.relative_to(root).as_posix().replace("/", os.sep)),
            path.relative_to(root).as_posix(),
        ),
    )
    if not paths:
        raise ValueError(f"No supported images were found in {root}.")
    return paths


def _load_images(
    paths: Sequence[Path],
    preprocess: Callable[[Image.Image], torch.Tensor],
) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for path in paths:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    image.load()
                    tensor = preprocess(image)
        except (
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as error:
            raise ValueError(f"Could not preprocess image {path}: {error}") from error
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise TypeError(
                f"Preprocessing image {path} did not return a [C,H,W] Tensor."
            )
        tensors.append(tensor)
    return tensors


def _image_batches(
    tensors: Sequence[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> list[tuple[tuple[int, ...], torch.Tensor]]:
    grouped: dict[tuple[int, ...], list[tuple[int, torch.Tensor]]] = {}
    for position, tensor in enumerate(tensors):
        grouped.setdefault(tuple(tensor.shape), []).append((position, tensor))

    batches: list[tuple[tuple[int, ...], torch.Tensor]] = []
    for group in grouped.values():
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            positions = tuple(position for position, _ in chunk)
            batch = torch.stack([tensor for _, tensor in chunk]).to(device)
            batches.append((positions, batch))
    return batches


def _encode_images(
    model: Any,
    batches: Sequence[tuple[tuple[int, ...], torch.Tensor]],
    *,
    count: int,
    embedding_dim: int,
) -> torch.Tensor:
    features: list[torch.Tensor | None] = [None] * count
    for positions, batch in batches:
        encoded = model.encode_image(batch, truncate_dim=embedding_dim)
        if tuple(encoded.shape) != (len(positions), embedding_dim):
            raise RuntimeError(
                "Image encoder returned an unexpected MRL shape: "
                f"expected {(len(positions), embedding_dim)}, received "
                f"{tuple(encoded.shape)}."
            )
        for position, embedding in zip(positions, encoded.unbind(0), strict=True):
            features[position] = embedding
    if any(feature is None for feature in features):
        raise RuntimeError("Image encoder did not produce every benchmark embedding.")
    return torch.stack([feature for feature in features if feature is not None])


def _encode_texts(
    model: Any,
    tokens: torch.Tensor,
    *,
    batch_size: int,
    embedding_dim: int,
) -> torch.Tensor:
    batches = [
        model.encode_text(
            tokens[start : start + batch_size],
            truncate_dim=embedding_dim,
        )
        for start in range(0, tokens.shape[0], batch_size)
    ]
    if not batches:
        raise RuntimeError("Text tokenizer returned an empty benchmark batch.")
    return torch.cat(batches)


def _measure(
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        operation()
        synchronize()

    samples: list[float] = []
    result: torch.Tensor | None = None
    for _ in range(iterations):
        synchronize()
        started = time.perf_counter()
        result = operation()
        synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    if result is None:
        raise RuntimeError("Benchmark did not produce a measured result.")
    return (
        {
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        },
        result,
    )


def _recall_at_k(
    scores: torch.Tensor,
    *,
    direction: str,
    k: int,
) -> float:
    if direction == "image_to_text":
        ranked = scores
    else:
        ranked = scores.T
    count = ranked.shape[0]
    top_indices = ranked.topk(min(k, count), dim=1).indices
    expected = torch.arange(count, device=ranked.device).unsqueeze(1)
    return float((top_indices == expected).any(dim=1).float().mean().item())


def _paired_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> dict[str, float]:
    if image_features.shape[0] != text_features.shape[0]:
        raise ValueError("Paired image and caption batches must have the same length.")
    scores = image_features.float().cpu() @ text_features.float().cpu().T
    if not torch.isfinite(scores).all().item():
        raise RuntimeError("Paired retrieval produced non-finite similarity scores.")
    return {
        "image_to_text_recall_at_1": _recall_at_k(
            scores,
            direction="image_to_text",
            k=1,
        ),
        "image_to_text_recall_at_5": _recall_at_k(
            scores,
            direction="image_to_text",
            k=5,
        ),
        "text_to_image_recall_at_1": _recall_at_k(
            scores,
            direction="text_to_image",
            k=1,
        ),
        "text_to_image_recall_at_5": _recall_at_k(
            scores,
            direction="text_to_image",
            k=5,
        ),
    }


def run_benchmark(
    *,
    images_dir: Path,
    captions_file: Path | None,
    limit: int,
    batch_size: int,
    warmup: int,
    iterations: int,
    device_name: str,
    precision: str,
    recursive: bool,
) -> dict[str, object]:
    all_paths = _image_paths(images_dir, recursive=recursive)
    paths = all_paths[:limit]
    paired = captions_file is not None
    if captions_file is None:
        texts = ["A benchmark query about the supplied images."]
    else:
        try:
            texts = captions_file.expanduser().read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError(
                f"Could not read captions file {captions_file}: {error}"
            ) from error
        texts = [text.strip() for text in texts]
        if len(texts) != len(all_paths):
            raise ValueError(
                "The captions file must contain exactly one line per supported image "
                f"({len(all_paths)} lines expected, {len(texts)} received)."
            )
        texts = texts[: len(paths)]
        if any(not text for text in texts):
            raise ValueError("The captions file must not contain empty lines.")

    model, _, preprocess = create_model_and_transforms(
        precision=precision,
        device=device_name,
    )
    device = next(model.parameters()).device
    image_tensors = _load_images(paths, preprocess)
    batches = _image_batches(
        image_tensors,
        batch_size=batch_size,
        device=device,
    )
    tokenizer = get_tokenizer(prompt_name="SearchQuery")
    tokens = tokenizer(texts).to(device)

    dimensions: dict[str, object] = {}
    with torch.inference_mode():
        for embedding_dim in SUPPORTED_EMBEDDING_DIMENSIONS:
            image_timing, image_features = _measure(
                lambda embedding_dim=embedding_dim: _encode_images(
                    model,
                    batches,
                    count=len(paths),
                    embedding_dim=embedding_dim,
                ),
                warmup=warmup,
                iterations=iterations,
                device=device,
            )
            text_timing, text_features = _measure(
                lambda embedding_dim=embedding_dim: _encode_texts(
                    model,
                    tokens,
                    batch_size=batch_size,
                    embedding_dim=embedding_dim,
                ),
                warmup=warmup,
                iterations=iterations,
                device=device,
            )
            image_norms = image_features.float().norm(dim=-1)
            text_norms = text_features.float().norm(dim=-1)
            tolerance = 1e-2 if precision == "bf16" else 1e-4
            if not torch.allclose(
                image_norms,
                torch.ones_like(image_norms),
                atol=tolerance,
                rtol=tolerance,
            ) or not torch.allclose(
                text_norms,
                torch.ones_like(text_norms),
                atol=tolerance,
                rtol=tolerance,
            ):
                raise RuntimeError(
                    f"MRL outputs at {embedding_dim} dimensions are not normalized."
                )
            image_result: dict[str, object] = {
                "latency": image_timing,
                "median_ms_per_image": image_timing["median_ms"] / len(paths),
            }
            text_result: dict[str, object] = {
                "latency": text_timing,
                "median_ms_per_text": text_timing["median_ms"] / len(texts),
            }
            dimensions[str(embedding_dim)] = {
                "image": image_result,
                "text": text_result,
                "float32_storage_bytes_per_embedding": embedding_dim * 4,
                "storage_reduction_percent_vs_768": (100 * (1 - embedding_dim / 768)),
                "paired_retrieval": (
                    _paired_retrieval_metrics(image_features, text_features)
                    if paired
                    else None
                ),
            }

    return {
        "schema_version": 1,
        "measured_at": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": str(device),
        "precision": precision,
        "environment": {
            "python": python_version(),
            "os": f"{system()} {release()}",
            "cpu": processor(),
            "torch": torch.__version__,
        },
        "inputs": {
            "image_count": len(paths),
            "caption_file": str(captions_file) if captions_file is not None else None,
            "paired_caption_metrics": paired,
            "image_batch_size": batch_size,
            "preprocessing_included_in_latency": False,
            "tokenization_included_in_latency": False,
        },
        "warmup_runs_per_operation": warmup,
        "measurement_runs_per_operation": iterations,
        "dimensions": dimensions,
        "notes": [
            "Retrieval recall is measured only when one line-aligned caption is supplied per image.",
            "No paired retrieval quality is claimed when captions are not provided.",
            "Latency includes model encoding and output concatenation, but excludes image batch preparation, preprocessing, and tokenization.",
            "Storage estimates use float32, matching the local image and video index format.",
            "The backbone still computes the native 768-dimensional representation; lower MRL dimensions reduce output and index storage, not transformer FLOPs.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--captions-file", type=Path, default=None)
    parser.add_argument("--limit", type=_positive_int, default=100)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--warmup", type=_nonnegative_int, default=1)
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = json.dumps(
        run_benchmark(
            images_dir=args.images,
            captions_file=args.captions_file,
            limit=args.limit,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            device_name=args.device,
            precision=args.precision,
            recursive=args.recursive,
        ),
        ensure_ascii=False,
        indent=2,
    )
    if args.output is None:
        print(report)
    else:
        args.output.expanduser().write_text(report + "\n", encoding="utf-8")
        print(f"MRL benchmark report written to {args.output}")


if __name__ == "__main__":
    main()
