"""Internal streaming helpers for batched image-file inference."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator
from os import PathLike
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image

type ImagePath = str | PathLike[str]
type ImageEmbedding = tuple[Path, torch.Tensor]
type _ImageItem = tuple[int, Path, torch.Tensor]

_DEFAULT_BATCH_SIZE = 8
_EMBEDDING_DIMENSION = 768


class _ImageEncoder(Protocol):
    def parameters(self) -> Iterator[torch.nn.Parameter]: ...

    def encode_image(self, image: torch.Tensor) -> torch.Tensor: ...


def _iter_image_embeddings(
    model: _ImageEncoder,
    preprocess: Callable[[Image.Image], torch.Tensor],
    image_paths: Iterable[ImagePath],
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> Iterator[ImageEmbedding]:
    """Yield path/embedding pairs in input order, keeping output tensors on CPU."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if isinstance(image_paths, (str, PathLike)):
        raise TypeError("image_paths must be an iterable of paths, not one path.")

    try:
        paths = iter(image_paths)
    except TypeError as error:
        raise TypeError("image_paths must be an iterable of paths.") from error

    return _stream_image_embeddings(model, preprocess, paths, batch_size)


def _stream_image_embeddings(
    model: _ImageEncoder,
    preprocess: Callable[[Image.Image], torch.Tensor],
    image_paths: Iterator[ImagePath],
    batch_size: int,
) -> Iterator[ImageEmbedding]:
    pending: list[_ImageItem] = []

    for input_index, image_path in enumerate(image_paths):
        try:
            path = Path(image_path)
        except TypeError as error:
            raise TypeError(
                f"Image path at input index {input_index} must be a string or Path."
            ) from error

        image_tensor = _load_preprocessed_image(path, preprocess, input_index)
        pending.append((input_index, path, image_tensor))
        if len(pending) == batch_size:
            yield from _encode_image_chunk(model, pending, batch_size)
            pending.clear()

    if pending:
        yield from _encode_image_chunk(model, pending, batch_size)


def _load_preprocessed_image(
    path: Path,
    preprocess: Callable[[Image.Image], torch.Tensor],
    input_index: int,
) -> torch.Tensor:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.load()
                image_tensor = preprocess(image)
    except (
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError(
            f"Could not read image at input index {input_index} "
            f"({path}): {error}"
        ) from error

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(
            f"Preprocessing image at input index {input_index} "
            f"({path}) did not return a torch.Tensor."
        )
    if image_tensor.ndim != 3:
        raise ValueError(
            f"Preprocessing image at input index {input_index} "
            f"({path}) must return a [C,H,W] Tensor; "
            f"received {tuple(image_tensor.shape)}."
        )
    return image_tensor


def _encode_image_chunk(
    model: _ImageEncoder,
    items: list[_ImageItem],
    batch_size: int,
) -> Iterator[ImageEmbedding]:
    try:
        model_device = next(model.parameters()).device
    except StopIteration as error:
        raise RuntimeError("The model has no parameters to determine its device.") from error

    groups: dict[tuple[int, ...], list[tuple[int, _ImageItem]]] = {}
    for position, item in enumerate(items):
        groups.setdefault(tuple(item[2].shape), []).append((position, item))

    ordered_features: list[torch.Tensor | None] = [None] * len(items)
    for group in groups.values():
        positions = [position for position, _ in group]
        input_indices = [item[0] for _, item in group]
        image_batch = torch.stack([item[2] for _, item in group])
        try:
            image_batch = image_batch.to(model_device)
            with torch.inference_mode():
                features = model.encode_image(image_batch)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                "CUDA ran out of memory while embedding images at input indices "
                f"{input_indices} (batch_size={batch_size}); reduce batch_size."
            ) from error

        expected_shape = (len(group), _EMBEDDING_DIMENSION)
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected_shape:
            actual_shape = (
                tuple(features.shape)
                if isinstance(features, torch.Tensor)
                else type(features).__name__
            )
            raise RuntimeError(
                "Image encoder returned an invalid embedding batch: "
                f"expected {expected_shape}, received {actual_shape}."
            )
        if not torch.isfinite(features).all().item():
            raise RuntimeError("Image encoder returned non-finite embeddings.")

        cpu_features = features.detach().to(device="cpu")
        for position, embedding in zip(positions, cpu_features.unbind(0), strict=True):
            ordered_features[position] = embedding

    for item, embedding in zip(items, ordered_features, strict=True):
        if embedding is None:
            raise RuntimeError("Image encoder did not produce an embedding.")
        yield item[1], embedding
