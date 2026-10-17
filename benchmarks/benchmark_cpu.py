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
from typing import Any, Self

import torch
from PIL import Image, ImageDraw

from openclip_gemma import create_model_and_transforms, get_tokenizer
from openclip_gemma.constants import MODEL_ID, MODEL_REVISION


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
    operation: Callable[[], torch.Tensor],
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


def run_benchmark(*, warmup: int, iterations: int, threads: int | None) -> dict[str, object]:
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError(
            "Install the optional benchmark dependency with "
            "`uv sync --extra benchmark`."
        ) from error

    if threads is not None:
        torch.set_num_threads(threads)

    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    with PeakRssSampler(process) as sampler:
        load_started = time.perf_counter()
        model, _, preprocess = create_model_and_transforms(
            "embeddinggemma-2",
            precision="fp32",
            device="cpu",
        )
        model_load_ms = (time.perf_counter() - load_started) * 1000
        model_loaded_rss = process.memory_info().rss

        tokenizer_started = time.perf_counter()
        tokenizer = get_tokenizer(
            "embeddinggemma-2",
            context_length=128,
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
    return {
        "measured_at": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
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
            "preprocessing_included_in_inference_latency": False,
        },
        "warmup_runs_per_operation": warmup,
        "measurement_runs_per_operation": iterations,
        "load": {
            "model_and_processor_ms": model_load_ms,
            "tokenizer_and_config_ms": tokenizer_load_ms,
        },
        "latency": timings,
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
            "This benchmark does not measure the 8192-token context limit.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=_positive_int, default=2)
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--threads", type=_positive_int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = json.dumps(
        run_benchmark(
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
