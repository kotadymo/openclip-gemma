from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_NAMESPACE = runpy.run_path(
    str(_REPOSITORY_ROOT / "benchmarks" / "benchmark_mrl.py")
)
_paired_retrieval_metrics = _BENCHMARK_NAMESPACE["_paired_retrieval_metrics"]


def test_paired_retrieval_metrics_report_perfect_aligned_recall() -> None:
    features = torch.eye(3)

    metrics = _paired_retrieval_metrics(features, features)  # type: ignore[operator]

    assert metrics == {
        "image_to_text_recall_at_1": 1.0,
        "image_to_text_recall_at_5": 1.0,
        "text_to_image_recall_at_1": 1.0,
        "text_to_image_recall_at_5": 1.0,
    }


def test_paired_retrieval_metrics_reject_unaligned_counts() -> None:
    with pytest.raises(ValueError, match="same length"):
        _paired_retrieval_metrics(  # type: ignore[operator]
            torch.eye(2),
            torch.eye(1),
        )
