"""Opt-in real-model integration tests for sampled-frame video search."""

from __future__ import annotations

import gc
import json
import os
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from openclip_gemma import create_model_and_transforms
from openclip_gemma.video_search import (
    _load_video_index,
    build_or_update_video_index,
    search_video,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]


def _write_synthetic_video(path: Path) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = 256
        stream.height = 192
        stream.pix_fmt = "yuv420p"
        for frame_index, color in enumerate(("red", "green", "blue")):
            image = Image.new("RGB", (256, 192), "white")
            try:
                ImageDraw.Draw(image).rectangle((48, 32, 208, 160), fill=color)
                frame = av.VideoFrame.from_image(image)
                frame.pts = frame_index
                frame.time_base = Fraction(1, 2)
                for packet in stream.encode(frame):
                    container.mux(packet)
            finally:
                image.close()
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _run_video_search(video_path: Path, index_path: Path, device: str):
    model, _, preprocess = create_model_and_transforms(
        precision="fp32",
        device=device,
    )
    model.eval()
    update = build_or_update_video_index(
        model,
        preprocess,
        video_path.parent,
        index_path,
        interval_seconds=0.5,
        batch_size=2,
    )
    index = _load_video_index(index_path)
    results = search_video(model, index_path, "赤い四角", top_k=3)
    return update, index, results


def test_cpu_video_frame_index_and_japanese_search(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "videos" / "sample.mp4"
    _write_synthetic_video(video_path)

    update, index, results = _run_video_search(
        video_path,
        tmp_path / "cpu-index",
        "cpu",
    )

    assert update.added_videos == 1
    assert update.frame_count == 3
    assert [frame.timestamp_seconds for frame in index.frames] == pytest.approx(
        [0.0, 0.5, 1.0]
    )
    assert len(results) == 3
    assert all(result.video_path == video_path for result in results)
    assert all(-1.0 <= result.score <= 1.0 for result in results)
    assert np.isfinite(index.embeddings).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable.")
def test_cuda_video_search_matches_cpu_fp32(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "videos" / "sample.mp4"
    _write_synthetic_video(video_path)

    cpu_update, cpu_index, _ = _run_video_search(
        video_path,
        tmp_path / "cpu-index",
        "cpu",
    )
    del cpu_update
    gc.collect()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    cuda_update, cuda_index, results = _run_video_search(
        video_path,
        tmp_path / "cuda-index",
        "cuda",
    )
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - load_started

    assert cuda_update.frame_count == cpu_index.embeddings.shape[0] == 3
    assert cuda_index.frames == cpu_index.frames
    assert len(results) == 3

    cpu_embeddings = torch.from_numpy(cpu_index.embeddings)
    cuda_embeddings = torch.from_numpy(cuda_index.embeddings)
    max_abs_error = (cpu_embeddings - cuda_embeddings).abs().max().item()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        cpu_embeddings,
        cuda_embeddings,
        dim=-1,
    ).mean().item()
    assert max_abs_error <= 1e-3
    assert cosine_similarity >= 0.9999

    print(
        json.dumps(
            {
                "cuda_device": torch.cuda.get_device_name(),
                "frames": cuda_update.frame_count,
                "max_abs_error_vs_cpu": max_abs_error,
                "cosine_similarity_vs_cpu": cosine_similarity,
                "model_load_and_index_seconds": elapsed_seconds,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            },
            sort_keys=True,
        )
    )
