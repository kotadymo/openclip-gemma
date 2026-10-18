"""Real-model CUDA test for streaming image-file batches."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from openclip_gemma import create_model_and_transforms
from openclip_gemma.batching import _iter_image_embeddings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENCLIP_GEMMA_RUN_INTEGRATION") != "1",
        reason="Set OPENCLIP_GEMMA_RUN_INTEGRATION=1 to load model weights.",
    ),
]


def _write_sample(path: Path, size: tuple[int, int], color: str) -> Path:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((size[0] // 4, size[1] // 4, size[0] * 3 // 4, size[1] * 3 // 4), fill=color)
    image.save(path)
    return path


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_image_file_batches_match_individual_encoding(
    tmp_path: Path,
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")

    model, _, preprocess = create_model_and_transforms(
        precision="fp32",
        device=device,
    )
    paths = [
        _write_sample(tmp_path / "first.png", (256, 192), "red"),
        _write_sample(tmp_path / "second.png", (384, 128), "green"),
        _write_sample(tmp_path / "third.png", (256, 192), "blue"),
    ]

    actual = list(
        _iter_image_embeddings(
            model,
            preprocess,
            paths,
            batch_size=3,
        )
    )

    assert [path for path, _ in actual] == paths
    assert all(embedding.shape == (768,) for _, embedding in actual)
    assert all(embedding.device.type == "cpu" for _, embedding in actual)
    assert all(torch.isfinite(embedding).all() for _, embedding in actual)

    with torch.inference_mode():
        for path, actual_embedding in actual:
            with Image.open(path) as image:
                image_tensor = preprocess(image).unsqueeze(0).to(device)
            expected = model.encode_image(image_tensor).squeeze(0).cpu()
            torch.testing.assert_close(
                actual_embedding,
                expected,
                atol=1e-5,
                rtol=1e-5,
            )
