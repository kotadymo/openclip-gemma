from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from openvlclip import create_model_and_transforms
from openvlclip.backends.dinov2_dinotxt import DINO_MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute dino.txt text-to-patch similarity scores."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("text", nargs="+")
    parser.add_argument(
        "--output-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Resize the score map after bicubic patch-feature interpolation.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = create_model_and_transforms(
        DINO_MODEL_NAME,
        pretrained="official",
        device=device,
    )
    image = Image.open(args.image).convert("RGB")
    pixels = preprocess(image).unsqueeze(0).to(device=device)
    output_size = tuple(args.output_size) if args.output_size is not None else None
    score_maps = model.similarity_map(
        pixels,
        args.text if len(args.text) > 1 else args.text[0],
        output_size=output_size,
    )
    print(
        f"Similarity scores: shape={tuple(score_maps.shape)}, "
        f"min={score_maps.min().item():.4f}, max={score_maps.max().item():.4f}"
    )


if __name__ == "__main__":
    main()
