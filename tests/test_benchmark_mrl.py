from __future__ import annotations

import csv
import io
import json
import runpy
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from openclip_gemma import image_search

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_NAMESPACE: dict[str, Any] = runpy.run_path(
    str(_REPOSITORY_ROOT / "benchmarks" / "benchmark_mrl.py")
)
_image_batches = _BENCHMARK_NAMESPACE["_image_batches"]
_encode_images = _BENCHMARK_NAMESPACE["_encode_images"]
_load_query_manifest = _BENCHMARK_NAMESPACE["_load_query_manifest"]
_query_metrics = _BENCHMARK_NAMESPACE["_query_metrics"]
_ranking_agreement = _BENCHMARK_NAMESPACE["_ranking_agreement"]
_recall_at_k = _BENCHMARK_NAMESPACE["_recall_at_k"]
_serialize_report = _BENCHMARK_NAMESPACE["_serialize_report"]
run_benchmark = _BENCHMARK_NAMESPACE["run_benchmark"]


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.image_batch_sizes: list[int] = []

    def encode_image(
        self,
        images: torch.Tensor,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        self.image_batch_sizes.append(images.shape[0])
        embedding_dim = 768 if truncate_dim is None else truncate_dim
        features = torch.zeros(
            (images.shape[0], embedding_dim),
            dtype=images.dtype,
            device=images.device,
        )
        channels = min(images.shape[1], embedding_dim)
        features[:, :channels] = images.mean(dim=(2, 3))[:, :channels]
        return F.normalize(features, p=2, dim=-1)

    def encode_text(
        self,
        tokens: torch.Tensor,
        *,
        normalize: bool = False,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        embedding_dim = 768 if truncate_dim is None else truncate_dim
        features = torch.zeros(
            (tokens.shape[0], embedding_dim),
            dtype=self.anchor.dtype,
            device=tokens.device,
        )
        token_indices = (tokens[:, 0] - 1).clamp(min=0, max=2)
        features.scatter_(1, token_indices.unsqueeze(1), 1)
        return F.normalize(features, p=2, dim=-1) if normalize else features


class FakeTokenizer:
    _TOKENS: ClassVar[dict[str, int]] = {"red": 1, "green": 2, "blue": 3}

    def __call__(self, text: str) -> torch.Tensor:
        return torch.tensor([[self._TOKENS[text]]], dtype=torch.long)


def _preprocess(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1) / 255


def _write_image(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return path


def test_image_batches_remain_on_cpu_and_encode_in_input_order() -> None:
    tensors = [
        torch.full((1, 2, 2), value, dtype=torch.float32)
        for value in (1.0, 2.0, 3.0)
    ]
    batches = _image_batches(tensors, batch_size=2)  # type: ignore[operator]

    assert all(batch.device.type == "cpu" for _, batch in batches)
    model = FakeModel()
    features, _ = _encode_images(
        model,
        batches,
        count=3,
        embedding_dim=128,
        device=torch.device("cpu"),
    )

    assert model.image_batch_sizes == [2, 1]
    assert features.shape == (3, 128)
    torch.testing.assert_close(features[:, 0], torch.ones(3))


def test_query_manifest_supports_optional_labels_and_rejects_unknown_images(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "photos"
    red_path = _write_image(image_root / "red.png", (255, 0, 0))
    _write_image(image_root / "nested" / "green.png", (0, 255, 0))
    manifest_path = tmp_path / "queries.json"
    manifest_path.write_text(
        json.dumps(
            [
                {"query": " red ", "relevant_images": ["red.png"]},
                {"query": "green"},
            ]
        ),
        encoding="utf-8",
    )

    queries = _load_query_manifest(  # type: ignore[operator]
        manifest_path,
        image_root=image_root,
        image_paths=[red_path, image_root / "nested" / "green.png"],
    )

    assert queries[0].text == "red"
    assert queries[0].relevant_images == frozenset({"red.png"})
    assert queries[1].relevant_images is None

    manifest_path.write_text(
        json.dumps([{"query": "red", "relevant_images": ["missing.png"]}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not in the indexed collection"):
        _load_query_manifest(  # type: ignore[operator]
            manifest_path,
            image_root=image_root,
            image_paths=[red_path],
        )


def test_ranking_agreement_and_recall_use_retrieved_image_paths() -> None:
    agreement = _ranking_agreement(  # type: ignore[operator]
        ("a.png", "b.png", "c.png"),
        ("b.png", "a.png", "c.png"),
        k=2,
    )
    recall = _recall_at_k(  # type: ignore[operator]
        ("a.png", "b.png", "c.png"),
        frozenset({"b.png", "c.png"}),
        k=2,
    )

    assert agreement == {"positionwise_at_k": 0.0, "top_k_overlap": 1.0}
    assert recall == pytest.approx(0.5)
    assert _query_metrics((), ())["recall_at_k"] == "unmeasured"  # type: ignore[operator]


def test_run_benchmark_measures_real_index_and_labeled_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_root = tmp_path / "photos"
    _write_image(image_root / "red.png", (255, 0, 0))
    _write_image(image_root / "green.png", (0, 255, 0))
    _write_image(image_root / "blue.png", (0, 0, 255))
    manifest_path = tmp_path / "queries.json"
    manifest_path.write_text(
        json.dumps(
            [
                {"query": "red", "relevant_images": ["red.png"]},
                {"query": "green"},
            ]
        ),
        encoding="utf-8",
    )
    model = FakeModel()

    def create_fake_model_and_transforms(
        *,
        precision: str,
        device: str,
    ) -> tuple[FakeModel, None, object]:
        assert precision == "fp32"
        assert device == "cpu"
        return model, None, _preprocess

    def create_fake_tokenizer(*, prompt_name: str) -> FakeTokenizer:
        assert prompt_name == "SearchQuery"
        return FakeTokenizer()

    benchmark_globals = run_benchmark.__globals__
    monkeypatch.setitem(
        benchmark_globals,
        "create_model_and_transforms",
        create_fake_model_and_transforms,
    )
    rss_call_count = 0

    def record_rss() -> int:
        nonlocal rss_call_count
        rss_call_count += 1
        return 12344 + rss_call_count

    original_load_images = benchmark_globals["_load_images"]

    def load_images_after_model_rss(
        paths: list[Path],
        preprocess: Any,
    ) -> list[torch.Tensor]:
        assert rss_call_count == 2
        return original_load_images(paths, preprocess)

    monkeypatch.setitem(
        benchmark_globals,
        "_process_rss_bytes",
        record_rss,
    )
    monkeypatch.setitem(
        benchmark_globals,
        "_load_images",
        load_images_after_model_rss,
    )
    monkeypatch.setattr(image_search, "get_tokenizer", create_fake_tokenizer)

    report = run_benchmark(  # type: ignore[operator]
        images_dir=image_root,
        query_manifest=manifest_path,
        limit=1,
        batch_size=2,
        warmup=0,
        iterations=1,
        device_name="cpu",
        precision="fp32",
        recursive=False,
    )

    assert report["schema_version"] == 2
    assert report["environment"]["process_rss_before_model_load_bytes"] == 12345
    assert report["environment"]["process_rss_after_model_load_bytes"] == 12346
    assert report["inputs"]["image_count"] == 3
    assert report["inputs"]["image_embedding_sample_count"] == 1
    assert report["inputs"]["labeled_query_count"] == 1
    for dimension_index, dimension in enumerate((128, 256, 512, 768)):
        metrics = report["dimensions"][str(dimension)]
        assert metrics["index"]["image_count"] == 3
        assert metrics["index"]["vector_bytes"] == 3 * dimension * 4
        assert metrics["index"]["serialized_index_bytes"] > metrics["index"][
            "vector_bytes"
        ]
        assert (
            metrics["process_rss_bytes"]["after_query_search"]
            == 12350 + dimension_index * 4
        )
        assert metrics["cuda_allocator"] == "unmeasured"
        assert metrics["query_search"]["recall_at_k"]["1"] == 1.0
    assert report["dimensions"]["128"]["query_search"][
        "ranking_agreement_vs_768"
    ]["top_k_overlap"] == 1.0

    csv_report = _serialize_report(report, "csv")  # type: ignore[operator]
    assert "serialized_index_bytes" in csv_report
    assert csv_report.count("\n") == 5
    csv_rows = list(csv.DictReader(io.StringIO(csv_report)))
    assert len(csv_rows) == 4
    metadata_columns = {
        "model_revision",
        "device",
        "precision",
        "image_root",
        "image_limit",
        "recursive",
        "query_manifest",
        "image_batch_size",
        "warmup_runs_per_operation",
        "measurement_runs_per_operation",
        "python",
        "os",
        "cpu",
        "torch",
        "cuda_runtime",
        "device_name",
    }
    assert metadata_columns <= set(csv_rows[0])
    for row in csv_rows:
        assert row["model_revision"] == report["model_revision"]
        assert row["device"] == report["device"]
        assert row["precision"] == report["precision"]
        assert row["image_root"] == str(image_root.resolve())
        assert row["image_limit"] == "1"
        assert row["recursive"] == "False"
        assert row["query_manifest"] == str(manifest_path)
        assert row["image_batch_size"] == "2"
        assert row["warmup_runs_per_operation"] == "0"
        assert row["measurement_runs_per_operation"] == "1"
        assert row["python"] == report["environment"]["python"]
        assert row["torch"] == report["environment"]["torch"]
        assert row["process_rss_after_model_load_bytes"] == "12346"
    json_report = _serialize_report(report, "json")  # type: ignore[operator]
    assert json.loads(json_report)["dimensions"]["768"]["index"]["vector_bytes"] == (
        3 * 768 * 4
    )
