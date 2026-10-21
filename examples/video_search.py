"""Build a sampled-frame video index and search it with text."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from openvlclip import create_model, create_model_and_transforms
from openvlclip.constants import MODEL_NAME
from openvlclip.registry import get_default_registry
from openvlclip.video_search import (
    build_or_update_video_index,
    search_video,
)


def _parse_embedding_dim(value: str) -> int:
    try:
        dimension = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("embedding_dim must be an integer.") from error
    return (
        get_default_registry()
        .resolve(MODEL_NAME)
        .backend.validate_embedding_dimension(
            dimension,
            name="embedding_dim",
        )
    )


def _format_timestamp(timestamp_seconds: float) -> str:
    sign = "-" if timestamp_seconds < 0 else ""
    milliseconds = round(abs(timestamp_seconds) * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser(
        "index", help="Create or update a video-frame index."
    )
    index_parser.add_argument(
        "--videos", type=Path, required=True, help="Video directory."
    )
    index_parser.add_argument(
        "--output", type=Path, required=True, help="Index directory."
    )
    index_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Target sampling interval in seconds (default: 1.0).",
    )
    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum number of frames encoded together (default: 8).",
    )
    index_parser.add_argument(
        "--embedding-dim",
        type=_parse_embedding_dim,
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
        help="Include videos in subdirectories.",
    )

    search_parser = commands.add_parser(
        "search", help="Search an existing video-frame index."
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            model, _, preprocess = create_model_and_transforms(device=args.device)
            update = build_or_update_video_index(
                model,
                preprocess,
                args.videos,
                args.output,
                interval_seconds=args.interval,
                batch_size=args.batch_size,
                recursive=args.recursive,
                embedding_dim=args.embedding_dim,
            )
            print(
                "Video index updated: "
                f"added={update.added_videos}, updated={update.updated_videos}, "
                f"removed={update.removed_videos}, unchanged={update.unchanged_videos}, "
                f"frames={update.frame_count}"
            )
            print(f"Index: {update.index_path}")
            return 0

        model = create_model(device=args.device)
        results = search_video(
            model,
            args.index,
            args.query,
            top_k=args.top_k,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print("Cosine similarity scores (not probabilities):")
    for result in results:
        print(
            f"{result.rank}. {result.video_path}  "
            f"{_format_timestamp(result.timestamp_seconds)}  "
            f"score={result.score:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
