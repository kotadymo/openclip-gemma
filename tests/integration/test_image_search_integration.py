"""Opt-in real-model integration tests for text-to-image search."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from openclip_gemma import create_model_and_transforms
from openclip_gemma.image_search import (
    build_or_update_image_index,
    search_images,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]


def _write_sample(path: Path, color: str, shape: str) -> Path:
    image = Image.new("RGB", (256, 192), "white")
    draw = ImageDraw.Draw(image)
    if shape == "circle":
        draw.ellipse((56, 24, 200, 168), fill=color)
    else:
        draw.rectangle((56, 24, 200, 168), fill=color)
    image.save(path)
    return path


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_japanese_text_search_runs_with_real_model(
    tmp_path: Path,
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    paths = [
        _write_sample(images_dir / "red-circle.png", "red", "circle"),
        _write_sample(images_dir / "blue-square.png", "blue", "square"),
    ]
    index_dir = tmp_path / "index"
    model, _, preprocess = create_model_and_transforms(
        precision="fp32",
        device=device,
    )

    update = build_or_update_image_index(
        model,
        preprocess,
        images_dir,
        index_dir,
        batch_size=2,
    )
    results = search_images(
        model,
        index_dir,
        "赤い円",
        top_k=2,
    )

    assert update.added == 2
    assert len(results) == 2
    assert all(result.path in paths for result in results)
    assert all(-1.0 <= result.score <= 1.0 for result in results)
