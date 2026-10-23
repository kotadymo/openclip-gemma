"""Compare exact-search top-k selection with the pre-optimization full sort."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from openvlclip._ranking import select_top_k

_BLOCK_SIZE = 4096
_Candidate = tuple[float, int]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _sort_key(paths: Sequence[str]) -> Callable[[_Candidate], tuple[float, str, str]]:
    return lambda item: (-item[0], paths[item[1]].casefold(), paths[item[1]])


def _full_sort(
    scores: NDArray[np.float32], limit: int, sort_key: Callable
) -> list[_Candidate]:
    return sorted(
        ((float(score), row) for row, score in enumerate(scores)),
        key=sort_key,
    )[:limit]


def _partition_sort(
    scores: NDArray[np.float32],
    limit: int,
    sort_key: Callable,
) -> list[_Candidate]:
    return select_top_k(scores, limit=limit, start_row=0, sort_key=sort_key)


def _chunked_full_sort(
    embeddings: NDArray[np.float32],
    query: NDArray[np.float32],
    limit: int,
    sort_key: Callable,
) -> list[_Candidate]:
    best: list[_Candidate] = []
    for start in range(0, len(embeddings), _BLOCK_SIZE):
        stop = min(start + _BLOCK_SIZE, len(embeddings))
        scores = embeddings[start:stop] @ query
        candidates = [(float(score), row + start) for row, score in enumerate(scores)]
        best = sorted((*best, *candidates), key=sort_key)[:limit]
    return best


def _chunked_partition_sort(
    embeddings: NDArray[np.float32],
    query: NDArray[np.float32],
    limit: int,
    sort_key: Callable,
) -> list[_Candidate]:
    best: list[_Candidate] = []
    for start in range(0, len(embeddings), _BLOCK_SIZE):
        stop = min(start + _BLOCK_SIZE, len(embeddings))
        scores = embeddings[start:stop] @ query
        best = select_top_k(
            scores,
            limit=limit,
            start_row=start,
            sort_key=sort_key,
            current=best,
        )
    return best


def _single_full_sort(
    embeddings: NDArray[np.float32],
    query: NDArray[np.float32],
    limit: int,
    sort_key: Callable,
) -> list[_Candidate]:
    return _full_sort(embeddings @ query, limit, sort_key)


def _single_partition_sort(
    embeddings: NDArray[np.float32],
    query: NDArray[np.float32],
    limit: int,
    sort_key: Callable,
) -> list[_Candidate]:
    return _partition_sort(embeddings @ query, limit, sort_key)


def _median_ms(
    operation: Callable[[], object], *, warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def _workload(
    size: int, dimension: int, tie_mode: str, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[str]]:
    if tie_mode == "high":
        embeddings = np.zeros((size, dimension), dtype=np.float32)
        embeddings[:, 0] = 1
        query = np.zeros(dimension, dtype=np.float32)
        query[0] = 1
    else:
        rng = np.random.default_rng(seed)
        embeddings = rng.standard_normal((size, dimension), dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        query = rng.standard_normal(dimension, dtype=np.float32)
        query /= np.linalg.norm(query)
    paths = [f"images/{size - row:08d}.png" for row in range(size)]
    return embeddings, query, paths


def benchmark(
    *,
    sizes: Sequence[int],
    dimensions: Sequence[int],
    top_ks: Sequence[int],
    tie_modes: Sequence[str],
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    results = []
    seed = 7300
    for tie_mode in tie_modes:
        for size in sizes:
            for dimension in dimensions:
                embeddings, query, paths = _workload(size, dimension, tie_mode, seed)
                seed += 1
                sort_key = _sort_key(paths)
                for requested_k in top_ks:
                    limit = min(requested_k, size)
                    scores = embeddings @ query
                    expected_top_k = _full_sort(scores, limit, sort_key)
                    actual_top_k = _partition_sort(scores, limit, sort_key)
                    if actual_top_k != expected_top_k:
                        raise AssertionError(
                            f"top-k mismatch: n={size}, d={dimension}, "
                            f"k={limit}, ties={tie_mode}"
                        )

                    expected_chunked = _chunked_full_sort(
                        embeddings, query, limit, sort_key
                    )
                    actual_chunked = _chunked_partition_sort(
                        embeddings, query, limit, sort_key
                    )
                    if actual_chunked != expected_chunked:
                        raise AssertionError(
                            f"chunked-query mismatch: n={size}, d={dimension}, "
                            f"k={limit}, ties={tie_mode}"
                        )

                    expected_single = _single_full_sort(
                        embeddings, query, limit, sort_key
                    )
                    actual_single = _single_partition_sort(
                        embeddings, query, limit, sort_key
                    )
                    if actual_single != expected_single:
                        raise AssertionError(
                            f"native-query mismatch: n={size}, d={dimension}, "
                            f"k={limit}, ties={tie_mode}"
                        )

                    old_top_k_ms = _median_ms(
                        partial(_full_sort, scores, limit, sort_key),
                        warmup=warmup,
                        iterations=iterations,
                    )
                    new_top_k_ms = _median_ms(
                        partial(_partition_sort, scores, limit, sort_key),
                        warmup=warmup,
                        iterations=iterations,
                    )
                    old_chunked_ms = _median_ms(
                        partial(_chunked_full_sort, embeddings, query, limit, sort_key),
                        warmup=warmup,
                        iterations=iterations,
                    )
                    new_chunked_ms = _median_ms(
                        partial(
                            _chunked_partition_sort,
                            embeddings,
                            query,
                            limit,
                            sort_key,
                        ),
                        warmup=warmup,
                        iterations=iterations,
                    )
                    old_single_ms = _median_ms(
                        partial(_single_full_sort, embeddings, query, limit, sort_key),
                        warmup=warmup,
                        iterations=iterations,
                    )
                    new_single_ms = _median_ms(
                        partial(
                            _single_partition_sort,
                            embeddings,
                            query,
                            limit,
                            sort_key,
                        ),
                        warmup=warmup,
                        iterations=iterations,
                    )
                    results.append(
                        {
                            "n": size,
                            "dimension": dimension,
                            "requested_k": requested_k,
                            "effective_k": limit,
                            "tie_mode": tie_mode,
                            "top_k_old_median_ms": old_top_k_ms,
                            "top_k_new_median_ms": new_top_k_ms,
                            "chunked_query_old_median_ms": old_chunked_ms,
                            "chunked_query_new_median_ms": new_chunked_ms,
                            "single_query_old_median_ms": old_single_ms,
                            "single_query_new_median_ms": new_single_ms,
                            "top_k_speedup": old_top_k_ms / new_top_k_ms,
                            "chunked_query_speedup": old_chunked_ms / new_chunked_ms,
                            "single_query_speedup": old_single_ms / new_single_ms,
                            "exact_order_and_scores_match": True,
                        }
                    )
                del embeddings, query, paths, scores
                gc.collect()
    return {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "warmup_runs": warmup,
            "timed_runs": iterations,
            "score_block_size": _BLOCK_SIZE,
            "peak_memory_measured": False,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", type=_positive_int, default=[1000, 10000, 100000]
    )
    parser.add_argument(
        "--dimensions", nargs="+", type=_positive_int, default=[512, 768, 1152]
    )
    parser.add_argument("--top-ks", nargs="+", type=_positive_int, default=[5, 10, 100])
    parser.add_argument(
        "--ties", nargs="+", choices=["low", "high"], default=["low", "high"]
    )
    parser.add_argument("--warmup", type=_positive_int, default=2)
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(
        sizes=args.sizes,
        dimensions=args.dimensions,
        top_ks=args.top_ks,
        tie_modes=args.ties,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
