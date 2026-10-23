from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_cli(script_name: str) -> dict[str, object]:
    return runpy.run_path(str(_REPOSITORY_ROOT / "examples" / f"{script_name}.py"))


@pytest.mark.parametrize(
    ("script_name", "arguments"),
    [
        (
            "image_search",
            ["index", "--images", "photos", "--output", "image-index"],
        ),
        (
            "video_search",
            ["index", "--videos", "videos", "--output", "video-index"],
        ),
    ],
)
def test_search_cli_uses_model_native_embedding_dimension_by_default(
    script_name: str,
    arguments: list[str],
) -> None:
    module = _load_cli(script_name)
    args = module["_build_parser"]().parse_args(arguments)  # type: ignore[operator]

    assert args.embedding_dim is None


@pytest.mark.parametrize(
    ("script_name", "arguments"),
    [
        (
            "image_search",
            [
                "index",
                "--images",
                "photos",
                "--output",
                "image-index",
                "--embedding-dim",
                "256",
                "--query-instruction",
                "Represent the user's image query.",
            ],
        ),
        (
            "video_search",
            [
                "index",
                "--videos",
                "videos",
                "--output",
                "video-index",
                "--embedding-dim",
                "256",
                "--query-instruction",
                "Represent the user's video query.",
            ],
        ),
    ],
)
def test_search_cli_forwards_embedding_dimension(
    script_name: str,
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli(script_name)
    model = object()
    preprocess = object()
    index_update = SimpleNamespace(
        index_path=Path("index.npz"),
        added=0,
        updated=0,
        removed=0,
        unchanged=0,
        added_videos=0,
        updated_videos=0,
        removed_videos=0,
        unchanged_videos=0,
        frame_count=0,
    )
    loader = Mock(return_value=(model, preprocess, preprocess))
    builder = Mock(return_value=index_update)
    namespace = module["main"].__globals__  # type: ignore[union-attr]
    monkeypatch.setitem(namespace, "create_model_and_transforms", loader)
    if script_name == "image_search":
        monkeypatch.setitem(namespace, "build_or_update_image_index", builder)
    else:
        monkeypatch.setitem(namespace, "build_or_update_video_index", builder)

    assert module["main"](arguments) == 0  # type: ignore[operator]

    assert builder.call_args.kwargs["embedding_dim"] == 256
    expected_instruction = (
        "Represent the user's image query."
        if script_name == "image_search"
        else "Represent the user's video query."
    )
    assert builder.call_args.kwargs["query_instruction"] == expected_instruction
    loader.assert_called_once()


def test_image_search_cli_uses_dino_native_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli("image_search")
    index_update = SimpleNamespace(
        index_path=Path("index.npz"),
        added=0,
        updated=0,
        removed=0,
        unchanged=0,
    )
    loader = Mock(return_value=(object(), object(), object()))
    builder = Mock(return_value=index_update)
    namespace = module["main"].__globals__  # type: ignore[union-attr]
    monkeypatch.setitem(namespace, "create_model_and_transforms", loader)
    monkeypatch.setitem(namespace, "build_or_update_image_index", builder)

    assert (
        module["main"](
            [
                "index",
                "--model",
                "dinov2-dinotxt",
                "--images",
                "photos",
                "--output",
                "image-index",
            ]
        )
        == 0
    )

    assert loader.call_args.args[0] == "dinov2-dinotxt"
    assert builder.call_args.kwargs["embedding_dim"] == 2048


def test_video_search_cli_uses_registered_model_native_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli("video_search")
    index_update = SimpleNamespace(
        index_path=Path("video-index.npz"),
        added_videos=0,
        updated_videos=0,
        removed_videos=0,
        unchanged_videos=0,
        frame_count=0,
    )
    loader = Mock(return_value=(object(), object(), object()))
    builder = Mock(return_value=index_update)
    namespace = module["main"].__globals__  # type: ignore[union-attr]
    monkeypatch.setitem(namespace, "create_model_and_transforms", loader)
    monkeypatch.setitem(namespace, "build_or_update_video_index", builder)

    assert (
        module["main"](
            [
                "index",
                "--model",
                "dinov2-dinotxt",
                "--videos",
                "videos",
                "--output",
                "video-index",
            ]
        )
        == 0
    )

    assert loader.call_args.args[0] == "dinov2-dinotxt"
    assert builder.call_args.kwargs["embedding_dim"] == 2048


def test_native_video_cli_forwards_model_and_video_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli("video_search")
    namespace = module["main"].__globals__  # type: ignore[union-attr]
    validator = Mock(return_value=2048)
    backend = SimpleNamespace(validate_embedding_dimension=validator)
    registry = SimpleNamespace(resolve=Mock(return_value=SimpleNamespace(backend=backend)))
    monkeypatch.setitem(namespace, "list_models", lambda: ["qwen-test"])
    monkeypatch.setitem(namespace, "get_default_registry", lambda: registry)
    model = object()
    loader = Mock(return_value=model)
    update = SimpleNamespace(
        index_path=Path("native-video-index.npz"),
        added_videos=1,
        updated_videos=0,
        removed_videos=0,
        unchanged_videos=0,
    )
    builder = Mock(return_value=update)
    monkeypatch.setitem(namespace, "create_model", loader)
    monkeypatch.setitem(namespace, "build_or_update_native_video_index", builder)

    assert (
        module["main"](
            [
                "native-index",
                "--model",
                "qwen-test",
                "--videos",
                "videos",
                "--output",
                "native-index",
                "--fps",
                "2",
                "--max-frames",
                "12",
                "--verify-contents",
                "--query-instruction",
                "Represent this user's video query.",
            ]
        )
        == 0
    )

    assert loader.call_args.kwargs["device"] == "cpu"
    assert builder.call_args.args == (model, Path("videos"), Path("native-index"))
    assert builder.call_args.kwargs == {
        "fps": 2.0,
        "max_frames": 12,
        "recursive": False,
        "embedding_dim": 2048,
        "verify_contents": True,
        "query_instruction": "Represent this user's video query.",
    }
    validator.assert_called_once_with(None, name="embedding_dim")


def test_native_video_cli_forwards_search_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli("video_search")
    namespace = module["main"].__globals__  # type: ignore[union-attr]
    monkeypatch.setitem(namespace, "list_models", lambda: ["qwen-test"])
    model = object()
    loader = Mock(return_value=model)
    result = SimpleNamespace(
        rank=1,
        video_path=Path("videos/one.mp4"),
        score=0.75,
    )
    search = Mock(return_value=[result])
    monkeypatch.setitem(namespace, "create_model", loader)
    monkeypatch.setitem(namespace, "search_native_video", search)

    assert (
        module["main"](
            [
                "native-search",
                "--model",
                "qwen-test",
                "--index",
                "native-index",
                "--query",
                "a clip",
                "--query-instruction",
                "Represent this user's video query.",
            ]
        )
        == 0
    )

    assert search.call_args.args == (model, Path("native-index"), "a clip")
    assert search.call_args.kwargs == {
        "top_k": 5,
        "query_instruction": "Represent this user's video query.",
    }


@pytest.mark.parametrize(
    ("script_name", "arguments", "result"),
    [
        (
            "image_search",
            [
                "search",
                "--index",
                "index",
                "--query",
                "a mountain",
                "--query-instruction",
                "Represent the user's image query.",
            ],
            SimpleNamespace(path=Path("photos/mountain.jpg"), score=0.9),
        ),
        (
            "video_search",
            [
                "search",
                "--index",
                "index",
                "--query",
                "a mountain",
                "--query-instruction",
                "Represent the user's video query.",
            ],
            SimpleNamespace(
                rank=1,
                video_path=Path("videos/mountain.mp4"),
                timestamp_seconds=1.5,
                score=0.9,
            ),
        ),
    ],
)
def test_image_and_frame_search_cli_forward_query_instructions(
    script_name: str,
    arguments: list[str],
    result: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli(script_name)
    namespace = module["main"].__globals__  # type: ignore[union-attr]
    model = object()
    loader = Mock(return_value=model)
    search_name = "search_images" if script_name == "image_search" else "search_video"
    search = Mock(return_value=[result])
    monkeypatch.setitem(namespace, "create_model", loader)
    monkeypatch.setitem(namespace, search_name, search)

    assert module["main"](arguments) == 0  # type: ignore[operator]

    assert search.call_args.args == (model, Path("index"), "a mountain")
    assert search.call_args.kwargs == {
        "top_k": 5,
        "query_instruction": (
            "Represent the user's image query."
            if script_name == "image_search"
            else "Represent the user's video query."
        ),
    }


def test_image_search_cli_verify_contents_is_opt_in_and_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli("image_search")
    parser = module["_build_parser"]()  # type: ignore[operator]
    default_args = parser.parse_args(
        ["index", "--images", "photos", "--output", "image-index"]
    )
    assert default_args.verify_contents is False

    arguments = [
        "index",
        "--images",
        "photos",
        "--output",
        "image-index",
        "--verify-contents",
    ]
    index_update = SimpleNamespace(
        index_path=Path("index.npz"),
        added=0,
        updated=0,
        removed=0,
        unchanged=0,
    )
    loader = Mock(return_value=(object(), object(), object()))
    builder = Mock(return_value=index_update)
    namespace = module["main"].__globals__  # type: ignore[union-attr]
    monkeypatch.setitem(namespace, "create_model_and_transforms", loader)
    monkeypatch.setitem(namespace, "build_or_update_image_index", builder)

    assert module["main"](arguments) == 0  # type: ignore[operator]

    assert builder.call_args.kwargs["verify_contents"] is True


@pytest.mark.parametrize(
    ("script_name", "arguments"),
    [
        (
            "image_search",
            [
                "index",
                "--images",
                "photos",
                "--output",
                "image-index",
                "--embedding-dim",
                "64",
            ],
        ),
        (
            "image_search",
            [
                "index",
                "--images",
                "photos",
                "--output",
                "image-index",
                "--embedding-dim",
                "384",
            ],
        ),
        (
            "video_search",
            [
                "index",
                "--videos",
                "videos",
                "--output",
                "video-index",
                "--embedding-dim",
                "64",
            ],
        ),
        (
            "video_search",
            [
                "index",
                "--videos",
                "videos",
                "--output",
                "video-index",
                "--embedding-dim",
                "384",
            ],
        ),
    ],
)
def test_search_cli_rejects_unsupported_embedding_dimension(
    script_name: str,
    arguments: list[str],
) -> None:
    module = _load_cli(script_name)
    with pytest.raises(SystemExit, match="2"):
        module["main"](arguments)  # type: ignore[operator]


@pytest.mark.parametrize("script_name", ["image_search", "video_search"])
def test_search_cli_delegates_dimension_validation_to_backend(
    script_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli(script_name)
    validator = Mock(return_value=384)
    backend = SimpleNamespace(validate_embedding_dimension=validator)
    registry = SimpleNamespace(
        resolve=Mock(return_value=SimpleNamespace(backend=backend))
    )
    namespace = module["_build_parser"].__globals__  # type: ignore[union-attr]
    monkeypatch.setitem(namespace, "get_default_registry", lambda: registry)
    arguments = [
        "index",
        "--images" if script_name == "image_search" else "--videos",
        "input",
        "--output",
        "index",
        "--embedding-dim",
        "384",
    ]
    if script_name == "image_search":
        index_update = SimpleNamespace(
            index_path=Path("index.npz"),
            added=0,
            updated=0,
            removed=0,
            unchanged=0,
        )
        loader = Mock(return_value=(object(), object(), object()))
        builder = Mock(return_value=index_update)
        module_namespace = module["main"].__globals__  # type: ignore[union-attr]
        monkeypatch.setitem(module_namespace, "create_model_and_transforms", loader)
        monkeypatch.setitem(module_namespace, "build_or_update_image_index", builder)

        assert module["main"](arguments) == 0  # type: ignore[operator]
        assert builder.call_args.kwargs["embedding_dim"] == 384
        validator.assert_called_once_with(384, name="embedding_dim")
        return

    index_update = SimpleNamespace(
        index_path=Path("index.npz"),
        added_videos=0,
        updated_videos=0,
        removed_videos=0,
        unchanged_videos=0,
        frame_count=0,
    )
    loader = Mock(return_value=(object(), object(), object()))
    builder = Mock(return_value=index_update)
    module_namespace = module["main"].__globals__  # type: ignore[union-attr]
    monkeypatch.setitem(module_namespace, "create_model_and_transforms", loader)
    monkeypatch.setitem(module_namespace, "build_or_update_video_index", builder)

    assert module["main"](arguments) == 0  # type: ignore[operator]
    assert builder.call_args.kwargs["embedding_dim"] == 384
    validator.assert_called_once_with(384, name="embedding_dim")
    loader.assert_called_once()
