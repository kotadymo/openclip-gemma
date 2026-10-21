"""Official EmbeddingGemma 2 image preprocessing for OpenCLIP-style callers."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image


class EmbeddingGemmaPreprocess:
    """Convert a PIL image to the verified float32 CHW Tensor representation."""

    def __init__(self, image_processor: Any) -> None:
        self.image_processor = image_processor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError(
                "preprocess expects a PIL.Image.Image; Tensor inputs are not accepted."
            )

        processor = self.image_processor
        patch_size = processor.patch_size
        pooling_kernel_size = processor.pooling_kernel_size
        max_patches = processor.max_soft_tokens * pooling_kernel_size**2

        converted = processor.process_image(
            image,
            do_convert_rgb=processor.do_convert_rgb,
        )
        resized = processor.aspect_ratio_preserving_resize(
            image=converted,
            patch_size=patch_size,
            max_patches=max_patches,
            pooling_kernel_size=pooling_kernel_size,
            resample=processor.resample,
        )
        pixels = processor.rescale_and_normalize(
            resized,
            do_rescale=processor.do_rescale,
            rescale_factor=processor.rescale_factor,
            do_normalize=processor.do_normalize,
            image_mean=processor.image_mean,
            image_std=processor.image_std,
        )
        if pixels.ndim != 3 or pixels.shape[0] != 3:
            raise ValueError(
                "The official image processor must return a 3-channel CHW Tensor; "
                f"received shape {tuple(pixels.shape)}."
            )
        if pixels.dtype != torch.float32:
            raise TypeError(
                "The verified image preprocess output must be float32; "
                f"received {pixels.dtype}."
            )
        return pixels
