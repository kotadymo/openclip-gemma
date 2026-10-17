"""Compute unscaled cosine similarities for EmbeddingGemma 2 candidates."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from openclip_gemma import (
    EmbeddingGemmaCLIP,
    create_model_and_transforms,
    get_tokenizer,
)
from openclip_gemma.constants import MODEL_NAME
from openclip_gemma.tokenizer import OFFICIAL_PROMPTS, EmbeddingGemmaTokenizer

DEFAULT_CANDIDATES = [
    "a red circle on a blue background",
    "a green triangle on a yellow background",
    "青い背景にある赤い円",
]


def zero_shot_scores(
    model: EmbeddingGemmaCLIP,
    preprocess: Callable[[Image.Image], torch.Tensor],
    tokenizer: EmbeddingGemmaTokenizer,
    images: Sequence[Image.Image],
    candidates: Sequence[str],
) -> torch.Tensor:
    """Return an image-by-candidate matrix of cosine similarity scores."""
    if not images:
        raise ValueError("At least one image is required.")
    if not candidates:
        raise ValueError("At least one candidate text is required.")
    for index, image in enumerate(images):
        if not isinstance(image, Image.Image):
            raise TypeError(f"images[{index}] must be a PIL.Image.Image.")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, str):
            raise TypeError(f"candidates[{index}] must be a string.")

    image_tensors = [preprocess(image.convert("RGB")) for image in images]
    first_shape = tuple(image_tensors[0].shape)
    if any(tuple(image.shape) != first_shape for image in image_tensors[1:]):
        raise ValueError(
            "All images must produce the same preprocessed shape for dense batching; "
            "no additional resizing or padding is applied."
        )

    try:
        device = next(model.backend.parameters()).device
    except StopIteration as error:
        raise RuntimeError("The model backend has no parameters to determine its device.") from error

    image_batch = torch.stack(image_tensors).to(device)
    text_batch = tokenizer(candidates).to(device)
    model.eval()
    with torch.inference_mode():
        image_features = model.encode_image(image_batch, normalize=True)
        text_features = model.encode_text(text_batch, normalize=True)
        return image_features @ text_features.T


def relative_softmax_scores(cosine_scores: torch.Tensor) -> torch.Tensor:
    """Normalize candidate scores for display; this is not a probability estimate."""
    if not isinstance(cosine_scores, torch.Tensor):
        raise TypeError("cosine_scores must be a torch.Tensor.")
    if cosine_scores.ndim != 2 or cosine_scores.shape[1] == 0:
        raise ValueError("cosine_scores must have shape [images, non-empty candidates].")
    if not torch.isfinite(cosine_scores).all().item():
        raise ValueError("cosine_scores must contain only finite values.")
    return cosine_scores.softmax(dim=-1)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    image_source = parser.add_mutually_exclusive_group(required=True)
    image_source.add_argument(
        "--image",
        type=Path,
        nargs="+",
        help="One or more image files. All must preprocess to the same H/W.",
    )
    image_source.add_argument(
        "--demo",
        action="store_true",
        help="Use two generated same-size images for a self-contained example.",
    )
    parser.add_argument(
        "--candidate",
        nargs="+",
        default=DEFAULT_CANDIDATES,
        help="Candidate texts in any supported language.",
    )
    parser.add_argument(
        "--prompt-name",
        choices=sorted(OFFICIAL_PROMPTS),
        default="Classification",
        help="Official EmbeddingGemma task prompt; defaults to Classification.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Truncation limit from 2 to 8192; padding remains batch-dynamic.",
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
        "--show-relative-softmax",
        action="store_true",
        help="Also print relative softmax scores, not calibrated probabilities.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    images = _demo_images() if args.demo else _load_images(args.image)
    image_names = (
        [f"demo image {index + 1}" for index in range(len(images))]
        if args.demo
        else [str(path) for path in args.image]
    )

    model, _, preprocess = create_model_and_transforms(
        MODEL_NAME,
        precision=args.precision,
        device=device,
    )
    tokenizer = get_tokenizer(
        prompt_name=args.prompt_name,
        context_length=args.context_length,
    )
    scores = zero_shot_scores(
        model,
        preprocess,
        tokenizer,
        images,
        args.candidate,
    )

    print("Cosine similarity scores (unscaled; not probabilities):")
    for image_name, row in zip(image_names, scores, strict=True):
        print(f"\n{image_name}")
        for candidate, score in zip(args.candidate, row, strict=True):
            print(f"  {candidate}: {score.item():.6f}")

    if args.show_relative_softmax:
        relative_scores = relative_softmax_scores(scores)
        print(
            "\nRelative softmax over these cosine scores "
            "(not calibrated classification probabilities):"
        )
        for image_name, row in zip(image_names, relative_scores, strict=True):
            print(f"\n{image_name}")
            for candidate, score in zip(args.candidate, row, strict=True):
                print(f"  {candidate}: {score.item():.6f}")


if __name__ == "__main__":
    main()
