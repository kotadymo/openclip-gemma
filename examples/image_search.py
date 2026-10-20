"""Build an image index and search it with Japanese or English text."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from openclip_gemma import create_model, create_model_and_transforms
from openclip_gemma.image_search import (
    ImageIndexError,
    build_or_update_image_index,
    search_images,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser(
        "index", help="Create or incrementally update an index."
    )
    index_parser.add_argument(
        "--images", type=Path, required=True, help="Image directory."
    )
    index_parser.add_argument(
        "--output", type=Path, required=True, help="Index directory."
    )
    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum number of images encoded together (default: 8).",
    )
    index_parser.add_argument(
        "--embedding-dim",
        type=int,
        choices=(128, 256, 512, 768),
        default=768,
        help="Embedding dimension to store (default: 768).",
    )
    index_parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device (default: cpu).",
    )
    index_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include images in subdirectories.",
    )

    search_parser = commands.add_parser("search", help="Search a saved image index.")
    search_parser.add_argument(
        "--index", type=Path, required=True, help="Index directory."
    )
    search_parser.add_argument(
        "--query", required=True, help="Japanese or English search text."
    )
    search_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results."
    )
    search_parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device (default: cpu).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            model, _, preprocess = create_model_and_transforms(device=args.device)
            update = build_or_update_image_index(
                model,
                preprocess,
                args.images,
                args.output,
                batch_size=args.batch_size,
                recursive=args.recursive,
                embedding_dim=args.embedding_dim,
            )
            print(
                "Image index updated: "
                f"added={update.added}, updated={update.updated}, "
                f"removed={update.removed}, unchanged={update.unchanged}"
            )
            print(f"Index: {update.index_path}")
            return 0

        model = create_model(device=args.device)
        results = search_images(
            model,
            args.index,
            args.query,
            top_k=args.top_k,
        )
    except (ImageIndexError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print("Cosine similarity scores (not probabilities):")
    for result in results:
        print(f"{result.score:.6f}\t{result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
