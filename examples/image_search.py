"""Build an image index and search it with Japanese or English text."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from openvlclip import create_model, create_model_and_transforms, list_models
from openvlclip.constants import MODEL_NAME
from openvlclip.image_search import (
    ImageIndexError,
    build_or_update_image_index,
    search_images,
)
from openvlclip.registry import get_default_registry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser(
        "index", help="Create or incrementally update an index."
    )
    index_parser.add_argument(
        "--model",
        choices=list_models(),
        default=MODEL_NAME,
        help=f"Registered model to index (default: {MODEL_NAME}).",
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
        default=None,
        help="Embedding dimension to store (default: the model's native dimension).",
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
    index_parser.add_argument(
        "--verify-contents",
        action="store_true",
        help="Hash file contents to detect changes with unchanged size and mtime.",
    )
    index_parser.add_argument(
        "--query-instruction",
        help="Fix a text-query instruction for instruction-aware models.",
    )

    search_parser = commands.add_parser("search", help="Search a saved image index.")
    search_parser.add_argument(
        "--model",
        choices=list_models(),
        default=MODEL_NAME,
        help=f"Registered model used to search the index (default: {MODEL_NAME}).",
    )
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
    search_parser.add_argument(
        "--query-instruction",
        help="Text-query instruction; match the instruction stored in the index.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            backend = get_default_registry().resolve(args.model).backend
            embedding_dim = backend.validate_embedding_dimension(
                args.embedding_dim,
                name="embedding_dim",
            )
            model, _, preprocess = create_model_and_transforms(
                args.model,
                device=args.device,
            )
            update = build_or_update_image_index(
                model,
                preprocess,
                args.images,
                args.output,
                batch_size=args.batch_size,
                recursive=args.recursive,
                embedding_dim=embedding_dim,
                verify_contents=args.verify_contents,
                query_instruction=args.query_instruction,
            )
            print(
                "Image index updated: "
                f"added={update.added}, updated={update.updated}, "
                f"removed={update.removed}, unchanged={update.unchanged}"
            )
            print(f"Index: {update.index_path}")
            return 0

        model = create_model(args.model, device=args.device)
        results = search_images(
            model,
            args.index,
            args.query,
            top_k=args.top_k,
            query_instruction=args.query_instruction,
        )
    except (ImageIndexError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print("Cosine similarity scores (not probabilities):")
    for result in results:
        print(f"{result.score:.6f}\t{result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
