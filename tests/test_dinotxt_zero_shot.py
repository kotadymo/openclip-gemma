from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = runpy.run_path(
    str(_REPOSITORY_ROOT / "examples" / "dinotxt_zero_shot_classification.py")
)
zero_shot_scores = _EXAMPLE["zero_shot_scores"]


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.received_images: torch.Tensor | None = None
        self.received_tokens: torch.Tensor | None = None

    def get_logits(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.received_images = image.detach().cpu()
        self.received_tokens = text.detach().cpu()
        image_logits = torch.arange(
            image.shape[0] * text.shape[0],
            dtype=torch.float32,
        ).reshape(image.shape[0], text.shape[0])
        return image_logits, image_logits.T


def _preprocess(image: Image.Image) -> torch.Tensor:
    return torch.full(
        (3, 4, 4),
        float(image.getpixel((0, 0))[0]),
        dtype=torch.float32,
    )


def _tokenize(texts: str | list[str]) -> torch.Tensor:
    values = [texts] if isinstance(texts, str) else texts
    return torch.arange(len(values), dtype=torch.long).unsqueeze(1)


def test_zero_shot_scores_preserve_image_and_candidate_order() -> None:
    model = FakeModel()
    images = [
        Image.new("RGB", (8, 8), color=(10, 20, 30)),
        Image.new("RGB", (8, 8), color=(40, 50, 60)),
    ]
    candidates = ["a red object", "a blue object", "a green object"]

    scores = zero_shot_scores(
        model,
        _preprocess,
        _tokenize,
        images,
        candidates,
        device=torch.device("cpu"),
    )

    assert scores.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert model.received_images is not None
    assert model.received_images[:, 0, 0, 0].tolist() == [10.0, 40.0]
    assert model.received_tokens is not None
    assert model.received_tokens[:, 0].tolist() == [0, 1, 2]
    assert not model.training


@pytest.mark.parametrize(
    ("images", "candidates", "message"),
    [
        ([], ["candidate"], "At least one image"),
        ([Image.new("RGB", (8, 8))], [], "At least one candidate"),
        ([Image.new("RGB", (8, 8))], ["  "], "non-empty text"),
    ],
)
def test_zero_shot_scores_reject_empty_inputs(
    images: list[Image.Image],
    candidates: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        zero_shot_scores(
            FakeModel(),
            _preprocess,
            _tokenize,
            images,
            candidates,
            device=torch.device("cpu"),
        )
