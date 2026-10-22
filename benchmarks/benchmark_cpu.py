"""Measure CPU model load, embedding latency, and peak process RSS."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from platform import processor, python_version, release, system
from tempfile import TemporaryDirectory
from typing import Any, Self

import numpy as np
import torch
from PIL import Image, ImageDraw

from openvlclip import create_model_and_transforms, get_tokenizer
from openvlclip.constants import MODEL_REVISION
from openvlclip.image_search import build_or_update_image_index, search_images

DINO_MODEL_NAME = "dinov2-dinotxt"


class PeakRssSampler:
    def __init__(self, process: Any, interval_seconds: float = 0.02) -> None:
        self._process = process
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = 0

    def _sample(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)

    def __enter__(self) -> Self:
        self.peak_bytes = self._process.memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _measure(
    operation: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _demo_image() -> Image.Image:
    image = Image.new("RGB", (256, 192), (30, 80, 180))
    ImageDraw.Draw(image).ellipse((72, 40, 184, 152), fill=(220, 35, 45))
    return image


def run_benchmark(
    *,
    model_name: str,
    warmup: int,
    iterations: int,
    threads: int | None,
) -> dict[str, object]:
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError(
            "Install the optional benchmark dependency with "
            "`uv sync --extra benchmark`."
        ) from error

    if threads is not None:
        torch.set_num_threads(threads)
    if model_name not in {"embeddinggemma-2", DINO_MODEL_NAME}:
        raise ValueError(
            "model_name must be 'embeddinggemma-2' or "
            f"{DINO_MODEL_NAME!r}."
        )

    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    with PeakRssSampler(process) as sampler:
        load_started = time.perf_counter()
        model, _, preprocess = create_model_and_transforms(
            model_name,
            precision="fp32",
            device="cpu",
        )
        model_load_ms = (time.perf_counter() - load_started) * 1000
        model_loaded_rss = process.memory_info().rss

        tokenizer_started = time.perf_counter()
        tokenizer = (
            get_tokenizer(model_name, context_length=128)
            if model_name == "embeddinggemma-2"
            else get_tokenizer(model_name)
        )
        tokenizer_load_ms = (time.perf_counter() - tokenizer_started) * 1000

        image = preprocess(_demo_image())
        image_one = image.unsqueeze(0)
        image_batch = image.repeat(2, 1, 1, 1)
        texts = [
            "a red circle on a blue background",
            "a green triangle on a yellow background",
            "青い背景にある赤い円",
            "黄色い背景にある緑の三角形",
        ]
        text_batch = tokenizer(texts)

        timings = {
            "image_one": _measure(
                lambda: model.encode_image(image_one),
                warmup=warmup,
                iterations=iterations,
            ),
            "image_batch_2": _measure(
                lambda: model.encode_image(image_batch),
                warmup=warmup,
                iterations=iterations,
            ),
            "text_one": _measure(
                lambda: model.encode_text(text_batch[:1]),
                warmup=warmup,
                iterations=iterations,
            ),
            "text_batch_4": _measure(
                lambda: model.encode_text(text_batch),
                warmup=warmup,
                iterations=iterations,
            ),
        }
        metadata = model.embedding_space_metadata
        dense_features: dict[str, object] | str
        if model.capabilities["patch_embedding"]:
            timings["patches"] = _measure(
                lambda: model.encode_patches(image_one),
                warmup=warmup,
                iterations=iterations,
            )
            timings["similarity_map"] = _measure(
                lambda: model.similarity_map(image_one, texts[0]),
                warmup=warmup,
                iterations=iterations,
            )
            patch_shape = tuple(model.encode_patches(image_one).shape)
            similarity_map_shape = tuple(
                model.similarity_map(image_one, texts[0]).shape
            )
            dense_features = {
                "patch_dimension": patch_shape[-1],
                "patch_shape": list(patch_shape),
                "similarity_map_shape": list(similarity_map_shape),
                "scores_are_probabilities": False,
            }
        else:
            dense_features = "not supported by this model"

        with TemporaryDirectory(prefix="openvlclip-v060-benchmark-") as root:
            images_dir = Path(root) / "images"
            images_dir.mkdir()
            _demo_image().save(images_dir / "sample.png")
            index_dir = Path(root) / "index"
            index_started = time.perf_counter()
            index_update = build_or_update_image_index(
                model,
                preprocess,
                images_dir,
                index_dir,
                batch_size=1,
            )
            index_build_ms = (time.perf_counter() - index_started) * 1000
            search_timing = _measure(
                lambda: search_images(model, index_dir, texts[0], top_k=1),
                warmup=warmup,
                iterations=iterations,
            )
            with np.load(index_update.index_path, allow_pickle=False) as archive:
                vector_bytes = int(archive["embeddings"].nbytes)
            index_storage = {
                "image_count": 1,
                "serialized_index_bytes": index_update.index_path.stat().st_size,
                "vector_bytes": vector_bytes,
                "build_latency_ms": index_build_ms,
                "query_search_latency_ms": search_timing,
            }

        if model_name == "embeddinggemma-2" and metadata.model_revision != MODEL_REVISION:
            raise RuntimeError("The registered EmbeddingGemma revision is unexpected.")
        if model_name == DINO_MODEL_NAME:
            from openvlclip.backends.dinov2_dinotxt import DINO_MODEL_REVISION

            if metadata.model_revision != DINO_MODEL_REVISION:
                raise RuntimeError("The registered dino.txt revision is unexpected.")
    return {
        "measured_at": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "model_id": metadata.model_id,
        "model_revision": metadata.model_revision,
        "embedding_dimension": model.validate_embedding_dimension(
            None,
            name="embedding_dim",
        ),
        "embedding_strategy": metadata.embedding_strategy,
        "preprocessing_id": metadata.preprocessing_id,
        "query_prompt_name": metadata.query_prompt_name,
        "device": "cpu",
        "dtype": "float32",
        "environment": {
            "python": python_version(),
            "os": f"{system()} {release()}",
            "cpu": processor(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "inputs": {
            "image_batch_shape": list(image_batch.shape),
            "text_batch_shape": list(text_batch.shape),
            "text_count": len(texts),
            "synthetic_smoke_input": True,
            "preprocessing_included_in_inference_latency": False,
        },
        "warmup_runs_per_operation": warmup,
        "measurement_runs_per_operation": iterations,
        "load": {
            "model_and_processor_ms": model_load_ms,
            "tokenizer_and_config_ms": tokenizer_load_ms,
        },
        "latency": timings,
        "index": index_storage,
        "dense_features": dense_features,
        "quality_metrics": {
            "cosine_distribution_and_rank": "not evaluated",
            "zero_shot_top_1_top_5": "not evaluated",
            "retrieval_recall_at_k": "not evaluated",
            "reason": (
                "The synthetic smoke input is not labeled evaluation data; "
                "no licensed benchmark dataset was supplied."
            ),
        },
        "gpu_memory": "not measured by this CPU-only benchmark.",
        "memory": {
            "baseline_rss_bytes": baseline_rss,
            "after_model_and_processor_load_rss_bytes": model_loaded_rss,
            "peak_rss_bytes": sampler.peak_bytes,
            "peak_delta_from_baseline_bytes": max(
                0,
                sampler.peak_bytes - baseline_rss,
            ),
            "sampling_interval_ms": 20,
        },
        "notes": [
            "Weights are loaded from the local Hugging Face cache or downloaded if absent.",
            "Preprocessing and tokenization time are excluded from encoder latency.",
            "Peak RSS is sampled and can miss short-lived allocation spikes.",
            "The generated image is for execution-path measurement only, not accuracy, retrieval quality, or model superiority.",
            "Index size describes one synthetic image and is not representative of a real collection.",
            "CUDA memory and dataset-based cosine, zero-shot, and Recall@K metrics are not evaluated.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("embeddinggemma-2", DINO_MODEL_NAME),
        default="embeddinggemma-2",
    )
    parser.add_argument("--warmup", type=_positive_int, default=2)
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--threads", type=_positive_int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = json.dumps(
        run_benchmark(
            model_name=args.model,
            warmup=args.warmup,
            iterations=args.iterations,
            threads=args.threads,
        ),
        ensure_ascii=False,
        indent=2,
    )
    if args.output is None:
        print(report)
    else:
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"Benchmark report written to {args.output}")


if __name__ == "__main__":
    main()
