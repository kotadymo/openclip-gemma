"""Compare image and text candidates with the official dino.txt score."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from openvlclip import (
    create_model_and_transforms,
    get_tokenizer,
)
from openvlclip.backends.base import ImagePreprocessAPI, InferenceModelAPI, TokenizerAPI
from openvlclip.backends.dinov2_dinotxt import DINO_MODEL_NAME


def zero_shot_scores(
    model: InferenceModelAPI,
    preprocess: ImagePreprocessAPI,
    tokenizer: TokenizerAPI,
    images: Sequence[Image.Image],
    candidates: Sequence[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return an image-by-candidate score matrix using the selected score mode."""
    if not images:
        raise ValueError("At least one image is required.")
    if not candidates:
        raise ValueError("At least one candidate text is required.")
    for index, image in enumerate(images):
        if not isinstance(image, Image.Image):
            raise TypeError(f"images[{index}] must be a PIL.Image.Image.")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError(f"candidates[{index}] must be non-empty text.")

    image_tensors = [preprocess(image.convert("RGB")) for image in images]
    first_shape = tuple(image_tensors[0].shape)
    if any(tuple(image.shape) != first_shape for image in image_tensors[1:]):
        raise ValueError(
            "All images must produce the same preprocessed shape for dense batching."
        )
    if any(
        image.dtype != torch.float32 or not torch.isfinite(image).all().item()
        for image in image_tensors
    ):
        raise ValueError("Preprocessed images must contain finite float32 values.")

    image_batch = torch.stack(image_tensors).to(device)
    text_batch = tokenizer(candidates).to(device)
    model.eval()
    with torch.inference_mode():
        image_scores, _ = model.get_logits(image_batch, text_batch)
    expected_shape = (len(images), len(candidates))
    if tuple(image_scores.shape) != expected_shape:
        raise RuntimeError(
            f"Model returned scores with shape {tuple(image_scores.shape)}; "
            f"expected {expected_shape}."
        )
    if not torch.isfinite(image_scores).all().item():
        raise RuntimeError("Model returned non-finite zero-shot scores.")
    return image_scores


def _demo_images() -> list[Image.Image]:
    circle = Image.new("RGB", (256, 192), (30, 80, 180))
    ImageDraw.Draw(circle).ellipse((72, 40, 184, 152), fill=(220, 35, 45))

    triangle = Image.new("RGB", (256, 192), (230, 200, 30))
    ImageDraw.Draw(triangle).polygon(
        [(128, 32), (220, 160), (36, 160)],
        fill=(25, 170, 60),
    )
    return [circle, triangle]


def _load_images(paths: Sequence[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    image_source = parser.add_mutually_exclusive_group(required=True)
    image_source.add_argument(
        "--image",
        type=Path,
        nargs="+",
        help="One or more image files.",
    )
    image_source.add_argument(
        "--demo",
        action="store_true",
        help="Use two generated same-size images.",
    )
    parser.add_argument(
        "--candidate",
        nargs="+",
        default=[
            "a red circle on a blue background",
            "a green triangle on a yellow background",
        ],
        help="Candidate texts compared against every image.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
    )
    parser.add_argument(
        "--score-mode",
        choices=("strict", "cosine"),
        default="strict",
        help="Use official scaled logits or unscaled cosine similarity.",
    )
    args = parser.parse_args(argv)

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    images = _demo_images() if args.demo else _load_images(args.image)
    image_names = (
        [f"demo image {index + 1}" for index in range(len(images))]
        if args.demo
        else [str(path) for path in args.image]
    )
    model, _, preprocess = create_model_and_transforms(
        DINO_MODEL_NAME,
        precision=args.precision,
        device=device,
        logit_scale_mode=args.score_mode,
    )
    tokenizer = get_tokenizer(DINO_MODEL_NAME, prompt_name=None)
    scores = zero_shot_scores(
        model,
        preprocess,
        tokenizer,
        images,
        args.candidate,
        device=device,
    )

    score_description = (
        "Official dino.txt logits (scaled; not probabilities)"
        if args.score_mode == "strict"
        else "Cosine similarity scores (unscaled; not probabilities)"
    )
    print(f"{score_description}:")
    for image_name, row in zip(image_names, scores, strict=True):
        print(f"\n{image_name}")
        for candidate, score in zip(args.candidate, row, strict=True):
            print(f"  {candidate}: {score.item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
