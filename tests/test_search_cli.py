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

    assert args.embedding_dim == (None if script_name == "image_search" else 768)


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
    if script_name == "image_search":
        with pytest.raises(SystemExit, match="2"):
            module["main"](arguments)  # type: ignore[operator]
    else:
        with pytest.raises(SystemExit, match="2"):
            module["_build_parser"]().parse_args(arguments)  # type: ignore[operator]


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

    args = module["_build_parser"]().parse_args(arguments)
    assert args.embedding_dim == 384
    validator.assert_called_once_with(384, name="embedding_dim")
