"""Measure MRL embedding latency, index storage, and query retrieval behavior."""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import os
import statistics
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from platform import processor, python_version, release, system
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from PIL import Image

from openclip_gemma import create_model_and_transforms
from openclip_gemma._dimensions import SUPPORTED_EMBEDDING_DIMENSIONS
from openclip_gemma.constants import MODEL_ID, MODEL_REVISION
from openclip_gemma.image_search import (
    build_or_update_image_index,
    search_images,
)

_SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_RANKING_K = 10
_RECALL_KS = (1, 5, 10)
_MISSING = object()


@dataclass(frozen=True)
class _Query:
    text: str
    relevant_images: frozenset[str] | None

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
) -> list[tuple[tuple[int, ...], torch.Tensor]]:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero.")
    grouped: dict[tuple[int, ...], list[tuple[int, torch.Tensor]]] = {}
    for position, tensor in enumerate(tensors):
        grouped.setdefault(tuple(tensor.shape), []).append((position, tensor))

    batches: list[tuple[tuple[int, ...], torch.Tensor]] = []
    for group in grouped.values():
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            positions = tuple(position for position, _ in chunk)
            batch = torch.stack(
                [tensor.detach().to(device="cpu") for _, tensor in chunk]
            )
            batches.append((positions, batch))
    return batches


def _encode_images(
    model: Any,
    batches: Sequence[tuple[tuple[int, ...], torch.Tensor]],
    *,
    count: int,
    embedding_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    features: list[torch.Tensor | None] = [None] * count
    encoder_time_ms = 0.0
    for positions, batch in batches:
        device_batch = batch.to(device)
        _synchronize(device)
        started = time.perf_counter()
        encoded = model.encode_image(device_batch, truncate_dim=embedding_dim)
        _synchronize(device)
        encoder_time_ms += (time.perf_counter() - started) * 1000
        if not isinstance(encoded, torch.Tensor):
            raise TypeError("Image encoder did not return a Tensor.")
        if tuple(encoded.shape) != (len(positions), embedding_dim):
            raise RuntimeError(
                "Image encoder returned an unexpected MRL shape: "
                f"expected {(len(positions), embedding_dim)}, received "
                f"{tuple(encoded.shape)}."
            )
        for position, embedding in zip(positions, encoded.unbind(0), strict=True):
            features[position] = embedding.detach().to(device="cpu")
        del device_batch, encoded
    if any(feature is None for feature in features):
        raise RuntimeError("Image encoder did not produce every benchmark embedding.")
    return (
        torch.stack([feature for feature in features if feature is not None]),
        encoder_time_ms,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latency_summary(samples: Sequence[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("At least one timing sample is required.")
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _measure[T](
    operation: Callable[[], T],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[dict[str, float], T]:
    for _ in range(warmup):
        operation()
        _synchronize(device)

    samples: list[float] = []
    result: T | None = None
    for _ in range(iterations):
        _synchronize(device)
        started = time.perf_counter()
        result = operation()
        _synchronize(device)
        samples.append((time.perf_counter() - started) * 1000)
    if result is None:
        raise RuntimeError("Benchmark did not produce a measured result.")
    return _latency_summary(samples), result


def _measure_image_encoding(
    model: Any,
    batches: Sequence[tuple[tuple[int, ...], torch.Tensor]],
    *,
    count: int,
    embedding_dim: int,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], torch.Tensor]:
    for _ in range(warmup):
        _encode_images(
            model,
            batches,
            count=count,
            embedding_dim=embedding_dim,
            device=device,
        )

    samples: list[float] = []
    result: torch.Tensor | None = None
    for _ in range(iterations):
        result, encoder_time_ms = _encode_images(
            model,
            batches,
            count=count,
            embedding_dim=embedding_dim,
            device=device,
        )
        samples.append(encoder_time_ms)
    if result is None:
        raise RuntimeError("Image benchmark did not produce embeddings.")
    return _latency_summary(samples), result


def _load_query_manifest(
    manifest_path: Path | None,
    *,
    image_root: Path,
    image_paths: Sequence[Path],
) -> tuple[_Query, ...]:
    if manifest_path is None:
        return ()
    try:
        manifest_text = manifest_path.expanduser().read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(
            f"Could not read query manifest {manifest_path}: {error}"
        ) from error
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Query manifest {manifest_path} is not valid JSON: {error}"
        ) from error
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("The query manifest must be a non-empty JSON array.")

    available_images = {
        path.relative_to(image_root).as_posix() for path in image_paths
    }
    queries: list[_Query] = []
    for position, item in enumerate(manifest):
        if not isinstance(item, dict):
            raise TypeError(
                f"Query manifest entry {position} must be a JSON object."
            )
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Query manifest entry {position} must contain non-empty query text."
            )

        labels_value = item.get("relevant_images", _MISSING)
        if labels_value is _MISSING:
            relevant_images = None
        else:
            if not isinstance(labels_value, list) or not labels_value:
                raise ValueError(
                    f"Query manifest entry {position} relevant_images must be a "
                    "non-empty array when provided."
                )
            labels: set[str] = set()
            for label in labels_value:
                if not isinstance(label, str) or not label or "\\" in label:
                    raise ValueError(
                        f"Query manifest entry {position} contains an invalid "
                        "relevant_images path."
                    )
                relative_path = PurePosixPath(label)
                if (
                    not relative_path.parts
                    or relative_path.is_absolute()
                    or ".." in relative_path.parts
                    or "\x00" in label
                    or relative_path.as_posix() != label
                    or (
                        len(relative_path.parts[0]) == 2
                        and relative_path.parts[0][1] == ":"
                    )
                ):
                    raise ValueError(
                        f"Query manifest entry {position} contains a non-relative "
                        f"image path: {label!r}."
                    )
                if label not in available_images:
                    raise ValueError(
                        f"Query manifest entry {position} references an image that "
                        f"is not in the indexed collection: {label!r}."
                    )
                if label in labels:
                    raise ValueError(
                        f"Query manifest entry {position} contains duplicate "
                        f"relevant_images path {label!r}."
                    )
                labels.add(label)
            relevant_images = frozenset(labels)
        queries.append(_Query(query.strip(), relevant_images))
    return tuple(queries)


def _ranking_agreement(
    reference: Sequence[str],
    candidate: Sequence[str],
    *,
    k: int,
) -> dict[str, float]:
    reference_top = tuple(reference[:k])
    candidate_top = tuple(candidate[:k])
    if not reference_top:
        raise ValueError("Cannot compare empty image rankings.")
    positionwise = sum(
        expected == actual
        for expected, actual in zip(reference_top, candidate_top, strict=True)
    ) / len(reference_top)
    overlap = len(set(reference_top) & set(candidate_top)) / len(reference_top)
    return {
        "positionwise_at_k": positionwise,
        "top_k_overlap": overlap,
    }


def _recall_at_k(
    ranked_images: Sequence[str],
    relevant_images: frozenset[str],
    *,
    k: int,
) -> float:
    if not relevant_images:
        raise ValueError("Recall requires at least one labeled relevant image.")
    return len(set(ranked_images[:k]) & relevant_images) / len(relevant_images)


def _mean_or_unmeasured(values: Sequence[float]) -> float | str:
    return statistics.mean(values) if values else "unmeasured"


def _process_rss_bytes() -> int:
    try:
        psutil = importlib.import_module("psutil")
    except ModuleNotFoundError as error:
        if error.name != "psutil":
            raise
        raise RuntimeError(
            "Process RSS measurement requires the benchmark extra; install "
            "openclip-gemma[benchmark]."
        ) from error
    return int(psutil.Process().memory_info().rss)


def _reset_cuda_peak(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    _synchronize(device)
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    return baseline


def _cuda_allocator_peak(
    device: torch.device,
    baseline: int | None,
) -> dict[str, int] | str:
    if baseline is None:
        return "unmeasured"
    peak = int(torch.cuda.max_memory_allocated(device))
    return {
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "peak_increment_bytes": max(0, peak - baseline),
    }


def _query_metrics(
    measurements: Sequence[tuple[_Query, dict[str, float], tuple[str, ...]]],
    baseline: Sequence[tuple[_Query, dict[str, float], tuple[str, ...]]],
) -> dict[str, Any]:
    if not measurements:
        return {
            "query_count": 0,
            "latency_ms": "unmeasured",
            "ranking_agreement_vs_768": "unmeasured",
            "recall_at_k": "unmeasured",
            "per_query": "unmeasured",
        }
    if len(measurements) != len(baseline):
        raise RuntimeError("Dimension rankings do not cover the same query set.")

    per_query: list[dict[str, object]] = []
    latency_samples: list[float] = []
    positionwise_agreements: list[float] = []
    top_k_overlaps: list[float] = []
    recalls: dict[int, list[float]] = {k: [] for k in _RECALL_KS}
    for (query, timing, ranked), (baseline_query, _, baseline_ranked) in zip(
        measurements,
        baseline,
        strict=True,
    ):
        if query.text != baseline_query.text:
            raise RuntimeError("Dimension rankings are not aligned by query.")
        agreement = _ranking_agreement(
            baseline_ranked,
            ranked,
            k=_RANKING_K,
        )
        positionwise_agreements.append(agreement["positionwise_at_k"])
        top_k_overlaps.append(agreement["top_k_overlap"])
        latency_samples.append(timing["median_ms"])

        if query.relevant_images is None:
            query_recall: dict[str, float] | str = "unmeasured"
        else:
            query_recall = {}
            for k in _RECALL_KS:
                recall = _recall_at_k(ranked, query.relevant_images, k=k)
                recalls[k].append(recall)
                query_recall[str(k)] = recall
        per_query.append(
            {
                "query": query.text,
                "search_latency_ms": timing,
                "top_results": list(ranked[:_RANKING_K]),
                "ranking_agreement_at_k_vs_768": agreement,
                "recall_at_k": query_recall,
            }
        )

    return {
        "query_count": len(measurements),
        "latency_ms": _latency_summary(latency_samples),
        "ranking_agreement_vs_768": {
            "k": _RANKING_K,
            "positionwise_at_k": statistics.mean(positionwise_agreements),
            "top_k_overlap": statistics.mean(top_k_overlaps),
        },
        "recall_at_k": {
            str(k): _mean_or_unmeasured(values) for k, values in recalls.items()
        },
        "per_query": per_query,
    }


def run_benchmark(
    *,
    images_dir: Path,
    query_manifest: Path | None,
    limit: int,
    batch_size: int,
    warmup: int,
    iterations: int,
    device_name: str,
    precision: str,
    recursive: bool,
) -> dict[str, Any]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")

    all_paths = _image_paths(images_dir, recursive=recursive)
    image_root = images_dir.expanduser().resolve(strict=True)
    sample_paths = all_paths[:limit]
    queries = _load_query_manifest(
        query_manifest,
        image_root=image_root,
        image_paths=all_paths,
    )

    rss_before_model_load = _process_rss_bytes()
    model, _, preprocess = create_model_and_transforms(
        precision=precision,
        device=device_name,
    )
    device = next(model.parameters()).device
    rss_after_model_load = _process_rss_bytes()
    image_tensors = _load_images(sample_paths, preprocess)
    batches = _image_batches(
        image_tensors,
        batch_size=batch_size,
    )

    dimensions: dict[str, Any] = {}
    query_measurements: dict[
        int,
        list[tuple[_Query, dict[str, float], tuple[str, ...]]],
    ] = {}
    with torch.inference_mode():
        for embedding_dim in SUPPORTED_EMBEDDING_DIMENSIONS:
            rss_before_dimension = _process_rss_bytes()
            cuda_baseline = _reset_cuda_peak(device)
            image_timing, image_features = _measure_image_encoding(
                model,
                batches,
                count=len(sample_paths),
                embedding_dim=embedding_dim,
                device=device,
                warmup=warmup,
                iterations=iterations,
            )
            image_norms = image_features.float().norm(dim=-1)
            tolerance = 1e-2 if precision == "bf16" else 1e-4
            if not torch.allclose(
                image_norms,
                torch.ones_like(image_norms),
                atol=tolerance,
                rtol=tolerance,
            ):
                raise RuntimeError(
                    f"MRL image outputs at {embedding_dim} dimensions are not normalized."
                )
            rss_after_image_embedding = _process_rss_bytes()

            with TemporaryDirectory(prefix="openclip-gemma-mrl-") as temporary_root:
                index_dir = Path(temporary_root) / str(embedding_dim)
                _synchronize(device)
                index_started = time.perf_counter()
                index_update = build_or_update_image_index(
                    model,
                    preprocess,
                    image_root,
                    index_dir,
                    batch_size=batch_size,
                    recursive=recursive,
                    embedding_dim=embedding_dim,
                )
                _synchronize(device)
                index_build_ms = (time.perf_counter() - index_started) * 1000
                serialized_index_bytes = index_update.index_path.stat().st_size
                with np.load(index_update.index_path, allow_pickle=False) as archive:
                    vector_bytes = int(archive["embeddings"].nbytes)
                rss_after_index_build = _process_rss_bytes()

                current_query_measurements: list[
                    tuple[_Query, dict[str, float], tuple[str, ...]]
                ] = []
                for query in queries:
                    search_timing, results = _measure(
                        lambda query=query, index_dir=index_dir: search_images(
                            model,
                            index_dir,
                            query.text,
                            top_k=_RANKING_K,
                        ),
                        warmup=warmup,
                        iterations=iterations,
                        device=device,
                    )
                    ranked_paths = tuple(
                        result.path.relative_to(image_root).as_posix()
                        for result in results
                    )
                    current_query_measurements.append(
                        (query, search_timing, ranked_paths)
                    )
                query_measurements[embedding_dim] = current_query_measurements
                rss_after_query_search = _process_rss_bytes()
                cuda_peak = _cuda_allocator_peak(device, cuda_baseline)

            dimensions[str(embedding_dim)] = {
                "image_embedding": {
                    "latency_ms": image_timing,
                    "median_ms_per_image": (
                        image_timing["median_ms"] / len(sample_paths)
                    ),
                    "sample_count": len(sample_paths),
                },
                "index": {
                    "build_latency_ms": index_build_ms,
                    "image_count": len(all_paths),
                    "serialized_index_bytes": serialized_index_bytes,
                    "vector_bytes": vector_bytes,
                    "storage_reduction_percent_vs_768": "unmeasured",
                },
                "process_rss_bytes": {
                    "before_dimension": rss_before_dimension,
                    "after_image_embedding": rss_after_image_embedding,
                    "after_index_build": rss_after_index_build,
                    "after_query_search": rss_after_query_search,
                },
                "cuda_allocator": cuda_peak,
            }

    baseline_measurements = query_measurements[768]
    for embedding_dim in SUPPORTED_EMBEDDING_DIMENSIONS:
        dimension = dimensions[str(embedding_dim)]
        dimension["query_search"] = _query_metrics(
            query_measurements[embedding_dim],
            baseline_measurements,
        )
        if embedding_dim == 768:
            dimension["index"]["storage_reduction_percent_vs_768"] = 0.0
        else:
            native_vector_bytes = dimensions["768"]["index"]["vector_bytes"]
            vector_bytes = dimension["index"]["vector_bytes"]
            dimension["index"]["storage_reduction_percent_vs_768"] = (
                100 * (1 - vector_bytes / native_vector_bytes)
            )

    return {
        "schema_version": 2,
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
            "cuda_runtime": torch.version.cuda or "unavailable",
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "process_rss_before_model_load_bytes": rss_before_model_load,
            "process_rss_after_model_load_bytes": rss_after_model_load,
        },
        "inputs": {
            "image_root": str(image_root),
            "image_limit": limit,
            "recursive": recursive,
            "image_count": len(all_paths),
            "image_embedding_sample_count": len(sample_paths),
            "query_manifest": (
                str(query_manifest.expanduser())
                if query_manifest is not None
                else "unmeasured"
            ),
            "query_count": len(queries),
            "labeled_query_count": sum(
                query.relevant_images is not None for query in queries
            ),
            "image_batch_size": batch_size,
            "image_preprocessing_included_in_embedding_latency": False,
            "host_to_device_transfer_in_embedding_latency": False,
            "query_search_includes_tokenization_and_index_loading": True,
        },
        "ranking_top_k": _RANKING_K,
        "recall_k_values": list(_RECALL_KS),
        "warmup_runs_per_operation": warmup,
        "measurement_runs_per_operation": iterations,
        "dimensions": dimensions,
        "notes": [
            "Image embedding latency sums encoder-call time only; preprocessing, host-to-device transfer, and CPU result assembly are excluded.",
            "The image timing limit selects a deterministic sample; index size and query rankings use the complete indexed collection.",
            "Query search latency measures the public search API, including tokenization, query encoding, index loading, and ranking.",
            "Serialized index size and vector bytes are read from the built index archive, not estimated.",
            "Process RSS values are point-in-time measurements after the named phase; CUDA allocator peaks are measured only on CUDA.",
            "Ranking agreement reports positionwise top-10 agreement and top-10 set overlap against the 768-dimensional index.",
            "Recall@K is computed only for queries with relevant_images labels and is macro-averaged over labeled queries.",
            "Unmeasured metrics are reported as 'unmeasured'; unlabeled or synthetic queries are not evidence of general retrieval quality.",
            "The backbone still computes the native 768-dimensional representation; lower MRL dimensions reduce output and index storage, not transformer FLOPs.",
        ],
    }


def _csv_rows(report: dict[str, Any]) -> list[dict[str, object]]:
    dimensions = report.get("dimensions")
    if not isinstance(dimensions, dict):
        raise TypeError("Benchmark report does not contain dimension results.")
    inputs = report["inputs"]
    environment = report["environment"]
    run_metadata: dict[str, object] = {
        "schema_version": report["schema_version"],
        "measured_at": report["measured_at"],
        "model_id": report["model_id"],
        "model_revision": report["model_revision"],
        "device": report["device"],
        "device_name": environment["device_name"],
        "precision": report["precision"],
        "image_root": inputs["image_root"],
        "image_limit": inputs["image_limit"],
        "recursive": inputs["recursive"],
        "query_manifest": inputs["query_manifest"],
        "image_batch_size": inputs["image_batch_size"],
        "warmup_runs_per_operation": report["warmup_runs_per_operation"],
        "measurement_runs_per_operation": report["measurement_runs_per_operation"],
        "ranking_top_k": report["ranking_top_k"],
        "recall_k_values": ",".join(str(value) for value in report["recall_k_values"]),
        "labeled_query_count": inputs["labeled_query_count"],
        "image_preprocessing_included_in_embedding_latency": inputs[
            "image_preprocessing_included_in_embedding_latency"
        ],
        "host_to_device_transfer_in_embedding_latency": inputs[
            "host_to_device_transfer_in_embedding_latency"
        ],
        "query_search_includes_tokenization_and_index_loading": inputs[
            "query_search_includes_tokenization_and_index_loading"
        ],
        "python": environment["python"],
        "os": environment["os"],
        "cpu": environment["cpu"],
        "torch": environment["torch"],
        "cuda_runtime": environment["cuda_runtime"],
        "process_rss_before_model_load_bytes": environment[
            "process_rss_before_model_load_bytes"
        ],
        "process_rss_after_model_load_bytes": environment[
            "process_rss_after_model_load_bytes"
        ],
    }
    rows: list[dict[str, object]] = []
    for embedding_dim, metrics in dimensions.items():
        image = metrics["image_embedding"]
        index = metrics["index"]
        search = metrics["query_search"]
        rss = metrics["process_rss_bytes"]
        cuda = metrics["cuda_allocator"]
        query_latency = search["latency_ms"]
        agreement = search["ranking_agreement_vs_768"]
        recall = search["recall_at_k"]
        rows.append(
            {
                **run_metadata,
                "dimension": embedding_dim,
                "image_count": inputs["image_count"],
                "image_embedding_sample_count": image["sample_count"],
                "image_embedding_median_ms": image["latency_ms"]["median_ms"],
                "image_embedding_median_ms_per_image": image[
                    "median_ms_per_image"
                ],
                "query_count": search["query_count"],
                "query_search_median_ms": (
                    query_latency["median_ms"]
                    if isinstance(query_latency, dict)
                    else query_latency
                ),
                "index_build_latency_ms": index["build_latency_ms"],
                "serialized_index_bytes": index["serialized_index_bytes"],
                "vector_bytes": index["vector_bytes"],
                "storage_reduction_percent_vs_768": index[
                    "storage_reduction_percent_vs_768"
                ],
                "process_rss_after_query_search_bytes": rss["after_query_search"],
                "cuda_peak_allocated_bytes": (
                    cuda["peak_allocated_bytes"]
                    if isinstance(cuda, dict)
                    else cuda
                ),
                "ranking_positionwise_at_k_vs_768": (
                    agreement["positionwise_at_k"]
                    if isinstance(agreement, dict)
                    else agreement
                ),
                "ranking_top_k_overlap_vs_768": (
                    agreement["top_k_overlap"]
                    if isinstance(agreement, dict)
                    else agreement
                ),
                "recall_at_1": (
                    recall.get("1", "unmeasured")
                    if isinstance(recall, dict)
                    else recall
                ),
                "recall_at_5": (
                    recall.get("5", "unmeasured")
                    if isinstance(recall, dict)
                    else recall
                ),
                "recall_at_10": (
                    recall.get("10", "unmeasured")
                    if isinstance(recall, dict)
                    else recall
                ),
            }
        )
    return rows


def _serialize_report(report: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output_format == "csv":
        rows = _csv_rows(report)
        if not rows:
            raise ValueError("Benchmark report does not contain dimension results.")
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()
    raise ValueError("output_format must be 'json' or 'csv'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, default=None)
    parser.add_argument("--limit", type=_positive_int, default=100)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--warmup", type=_nonnegative_int, default=1)
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    args = parser.parse_args()

    report = _serialize_report(
        run_benchmark(
            images_dir=args.images,
            query_manifest=args.query_manifest,
            limit=args.limit,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            device_name=args.device,
            precision=args.precision,
            recursive=args.recursive,
        ),
        args.format,
    )
    if args.output is None:
        print(report, end="")
    else:
        args.output.expanduser().write_text(report, encoding="utf-8")
        print(f"MRL benchmark report written to {args.output}")


if __name__ == "__main__":
    main()
