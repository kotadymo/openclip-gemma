"""Build a sampled-frame video index and search it with text."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from openvlclip import create_model, create_model_and_transforms, list_models
from openvlclip.constants import MODEL_NAME
from openvlclip.registry import get_default_registry
from openvlclip.video_search import (
    build_or_update_native_video_index,
    build_or_update_video_index,
    search_native_video,
    search_video,
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
        "--model",
        choices=list_models(),
        default=MODEL_NAME,
        help=f"Registered model to index (default: {MODEL_NAME}).",
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
        help="Include videos in subdirectories.",
    )
    index_parser.add_argument(
        "--query-instruction",
        help="Fix a text-query instruction for instruction-aware models.",
    )

    search_parser = commands.add_parser(
        "search", help="Search an existing video-frame index."
    )
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

    native_index_parser = commands.add_parser(
        "native-index",
        help="Create or update a model-native whole-video embedding index.",
    )
    native_index_parser.add_argument(
        "--model",
        choices=list_models(),
        required=True,
        help="Registered model with video_embedding capability.",
    )
    native_index_parser.add_argument(
        "--videos", type=Path, required=True, help="Video directory."
    )
    native_index_parser.add_argument(
        "--output", type=Path, required=True, help="Index directory."
    )
    native_index_parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Video sampling rate in frames per second (default: 1.0).",
    )
    native_index_parser.add_argument(
        "--max-frames",
        type=int,
        default=64,
        help="Maximum frames per video (default: 64).",
    )
    native_index_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dimension to store (default: the model's native dimension).",
    )
    native_index_parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device (default: cpu).",
    )
    native_index_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include videos in subdirectories.",
    )
    native_index_parser.add_argument(
        "--verify-contents",
        action="store_true",
        help="Hash videos to detect content changes despite matching size and mtime.",
    )
    native_index_parser.add_argument(
        "--query-instruction",
        help="Fix a text-query instruction for instruction-aware models.",
    )

    native_search_parser = commands.add_parser(
        "native-search",
        help="Search a model-native whole-video embedding index.",
    )
    native_search_parser.add_argument(
        "--model",
        choices=list_models(),
        required=True,
        help="Registered model with video_embedding capability.",
    )
    native_search_parser.add_argument(
        "--index", type=Path, required=True, help="Index directory."
    )
    native_search_parser.add_argument(
        "--query", required=True, help="Japanese or English search text."
    )
    native_search_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results."
    )
    native_search_parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device (default: cpu).",
    )
    native_search_parser.add_argument(
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
            update = build_or_update_video_index(
                model,
                preprocess,
                args.videos,
                args.output,
                interval_seconds=args.interval,
                batch_size=args.batch_size,
                recursive=args.recursive,
                embedding_dim=embedding_dim,
                query_instruction=args.query_instruction,
            )
            print(
                "Video index updated: "
                f"added={update.added_videos}, updated={update.updated_videos}, "
                f"removed={update.removed_videos}, unchanged={update.unchanged_videos}, "
                f"frames={update.frame_count}"
            )
            print(f"Index: {update.index_path}")
            return 0

        if args.command == "search":
            model = create_model(args.model, device=args.device)
            results = search_video(
                model,
                args.index,
                args.query,
                top_k=args.top_k,
                query_instruction=args.query_instruction,
            )
        elif args.command == "native-index":
            backend = get_default_registry().resolve(args.model).backend
            embedding_dim = backend.validate_embedding_dimension(
                args.embedding_dim,
                name="embedding_dim",
            )
            model = create_model(args.model, device=args.device)
            update = build_or_update_native_video_index(
                model,
                args.videos,
                args.output,
                fps=args.fps,
                max_frames=args.max_frames,
                recursive=args.recursive,
                embedding_dim=embedding_dim,
                verify_contents=args.verify_contents,
                query_instruction=args.query_instruction,
            )
            print(
                "Native-video index updated: "
                f"added={update.added_videos}, updated={update.updated_videos}, "
                f"removed={update.removed_videos}, "
                f"unchanged={update.unchanged_videos}"
            )
            print(f"Index: {update.index_path}")
            return 0
        else:
            model = create_model(args.model, device=args.device)
            results = search_native_video(
                model,
                args.index,
                args.query,
                top_k=args.top_k,
                query_instruction=args.query_instruction,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print("Cosine similarity scores (not probabilities):")
    for result in results:
        if hasattr(result, "timestamp_seconds"):
            print(
                f"{result.rank}. {result.video_path}  "
                f"{_format_timestamp(result.timestamp_seconds)}  "
                f"score={result.score:.4f}"
            )
        else:
            print(f"{result.rank}. {result.video_path}  score={result.score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
