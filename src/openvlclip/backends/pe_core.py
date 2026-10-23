"""Optional OpenCLIP PE-Core T/S backend using pinned Hugging Face snapshots.

Uses timm's Apache-2.0 OpenCLIP-converted PE weights, not copies of Meta's
perception_models implementation. No weights are included in this package.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from openvlclip._dimensions import validate_positive_embedding_dimension
from openvlclip.backends.base import (
    MultimodalInput,
    ProcessedImageBatch,
    VideoInput,
    stack_processed_images,
)
from openvlclip.capabilities import capabilities_for_modalities, freeze_capabilities
from openvlclip.embedding_space import EmbeddingSpaceMetadata
from openvlclip.model import EmbeddingGemmaCLIP

PE_CORE_BACKEND_ID = "meta-pecore-openclip"
PE_CORE_T_MODEL_NAME = "pe-core-t16-384"
PE_CORE_T_MODEL_ID = "timm/PE-Core-T-16-384"
PE_CORE_T_REVISION = "7fe539ed578ac49a1c2b4f946e4b0747704c825a"
PE_CORE_S_MODEL_NAME = "pe-core-s16-384"
PE_CORE_S_MODEL_ID = "timm/PE-Core-S-16-384"
PE_CORE_S_REVISION = "3249a38eb1c432ec19231c6fe1774acb6a4e4efe"
PE_CORE_DIMENSION = 512
PE_CORE_IMAGE_SIZE = 384
PE_CORE_CONTEXT_LENGTH = 32
_SNAPSHOT_PATTERNS = (
    "open_clip_config.json", "open_clip_model.safetensors", "vocab.json",
    "merges.txt", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json",
)


def _snapshot(model_id: str, revision: str, cache_dir: str | None) -> Path:
    from huggingface_hub import snapshot_download

    try:
        local = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=list(_SNAPSHOT_PATTERNS),
            )
        )
    except Exception as error:
        raise RuntimeError(f"Cannot download pinned PE-Core {model_id}@{revision}.") from error
    for filename in ("open_clip_config.json", "open_clip_model.safetensors"):
        if not (local / filename).is_file():
            raise RuntimeError(f"PE-Core snapshot missing required file {filename}.")
    return local


class PECoreTokenizer:
    """Return the native 32-token PE-Core ID tensor."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.context_length = PE_CORE_CONTEXT_LENGTH

    def __call__(self, texts: str | Sequence[str]) -> torch.Tensor:
        if isinstance(texts, str):
            batch = [texts]
        elif isinstance(texts, Sequence) and not isinstance(texts, (bytes, bytearray)):
            batch = list(texts)
        else:
            raise TypeError("texts must be a string or sequence of strings.")
        if not batch or any(not isinstance(item, str) for item in batch):
            raise ValueError("texts must contain at least one string.")
        tokens = self.tokenizer(batch)
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.dtype != torch.long
            or tokens.shape != (len(batch), PE_CORE_CONTEXT_LENGTH)
        ):
            raise RuntimeError("PE-Core tokenizer must return LongTensor[B,32].")
        return tokens


class PECorePreprocess:
    """Run the local OpenCLIP reference preprocessing on PIL RGB."""

    def __init__(self, transform: Any) -> None:
        if not callable(transform):
            raise TypeError("PE-Core preprocess must be callable.")
        self.transform = transform

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("PE-Core preprocessing requires a PIL image.")
        result = self.transform(image.convert("RGB"))
        if (
            not isinstance(result, torch.Tensor)
            or result.shape != (3, 384, 384)
            or not result.is_floating_point()
            or not torch.isfinite(result).all().item()
        ):
            raise RuntimeError("PE-Core preprocess must return finite float [3,384,384].")
        return result


class PECoreCLIP(EmbeddingGemmaCLIP):
    """OpenVLCLIP wrapper retaining PE-Core's learned CLIP scale."""

    def forward(
        self,
        image: ProcessedImageBatch | None = None,
        text: torch.Tensor | None = None,
        *,
        truncate_dim: int | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor] | dict[str, object]:
        if image is None and text is None:
            raise ValueError("Provide image or text.")
        self.validate_embedding_dimension(truncate_dim, name="truncate_dim")
        a = self.encode_image(image, normalize=True) if image is not None else None
        b = self.encode_text(text, normalize=True) if text is not None else None
        reference = a if a is not None else b
        assert reference is not None
        scale = (
            self.backend.logit_scale.exp().to(device=reference.device)
            if self.logit_scale_mode == "strict"
            else torch.ones((), device=reference.device, dtype=reference.dtype)
        )
        if self.output_dict:
            return {"image_features": a, "text_features": b, "logit_scale": scale}
        return a, b, scale

    def encode_multimodal(
        self, items: MultimodalInput, *, normalize: bool = False,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PE-Core T/S has no mixed-input API.")

    def encode_video(
        self, video: VideoInput, *, normalize: bool = False,
        truncate_dim: int | None = None, fps: float | None = None,
        max_frames: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Native video encoding is not exposed for PE-Core T/S.")

    def encode_patches(
        self, image: ProcessedImageBatch, normalize: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError("PE-Core patch embeddings are not exposed.")

    def similarity_map(
        self, image: ProcessedImageBatch, text: str | Sequence[str],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PE-Core similarity maps are not exposed.")


class PECoreBackend:
    """Load and encode a pinned timm OpenCLIP PE-Core model."""

    def __init__(self, *, model_name: str, model_id: str, revision: str) -> None:
        self.backend_id = PE_CORE_BACKEND_ID
        self.model_name = model_name
        self.model_id = model_id
        self.revision = revision
        self.capabilities: Mapping[str, bool] = freeze_capabilities(
            {
                "image_embedding": True,
                "text_embedding": True,
                "image_text_similarity": True,
                "patch_embedding": False,
                "text_similarity_map": False,
                "mrl": False,
                "selective_loading": False,
                "video_embedding": False,
                "multimodal_embedding": False,
                "instruction_aware": False,
                "variable_resolution": False,
                "sigmoid_logits": False,
            }
        )
        self.embedding_space_metadata = EmbeddingSpaceMetadata(
            backend_id=self.backend_id,
            model_id=model_id,
            model_revision=revision,
            embedding_strategy="pecore-openclip-global-v1",
            normalization="l2-on-request-v1",
            preprocessing_id="pecore-rgb384-bilinear-squash-normalize-0.5-v1",
            index_prompt=None,
            query_prompt_name=None,
            query_prompt=None,
        )

    def validate_embedding_dimension(self, value: object | None, *, name: str) -> int:
        if value is None:
            return PE_CORE_DIMENSION
        dimension = validate_positive_embedding_dimension(value, name=name)
        if dimension != PE_CORE_DIMENSION:
            raise ValueError(f"{name} must be 512; PE-Core has no MRL.")
        return dimension

    def capabilities_for(self, modalities: tuple[str, ...]) -> Mapping[str, bool]:
        return capabilities_for_modalities(self.capabilities, modalities)

    def _validate_identity(self, model_id: str, revision: str) -> None:
        if model_id != self.model_id or revision != self.revision:
            raise ValueError("Only the pinned PE-Core model revision is registered.")

    def load_model(
        self, *, model_id: str, revision: str, cache_dir: str | None,
        device: torch.device, dtype: torch.dtype, output_dict: bool,
        logit_scale_mode: str, modalities: tuple[str, ...] | None,
    ) -> PECoreCLIP:
        self._validate_identity(model_id, revision)
        if modalities not in (None, ("text", "image")):
            raise NotImplementedError("PE-Core selective loading is not supported.")
        if dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("PE-Core supports FP32 or BF16.")
        try:
            import open_clip
        except ImportError as error:
            raise ImportError(
                "PE-Core needs the optional package: pip install open-clip-torch"
            ) from error
        local = _snapshot(model_id, revision, cache_dir)
        with (local / "open_clip_config.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        model_cfg = config.get("model_cfg", {})
        if (
            model_cfg.get("embed_dim") != PE_CORE_DIMENSION
            or model_cfg.get("vision_cfg", {}).get("image_size") != PE_CORE_IMAGE_SIZE
            or model_cfg.get("text_cfg", {}).get("context_length") != PE_CORE_CONTEXT_LENGTH
        ):
            raise RuntimeError("Unexpected pinned PE-Core model configuration.")
        loaded, _, preprocess = open_clip.create_model_and_transforms(
            f"local-dir:{local}", device="cpu", precision="fp32"
        )
        if not isinstance(loaded, nn.Module):
            raise TypeError("OpenCLIP must return a PyTorch module.")
        loaded = loaded.to(device=device, dtype=dtype).eval()
        if not isinstance(getattr(loaded, "logit_scale", None), torch.Tensor):
            raise TypeError("PE-Core checkpoint must expose learned logit_scale.")
        return PECoreCLIP(
            backend=loaded, processor=preprocess, output_dict=output_dict,
            logit_scale_mode=logit_scale_mode, modalities=("text", "image"),
            inference_backend=self, cache_dir=cache_dir,
        ).eval()

    def create_tokenizer(
        self, *, model_id: str, revision: str, cache_dir: str | None,
        prompt_name: str | None, context_length: int | None,
    ) -> PECoreTokenizer:
        self._validate_identity(model_id, revision)
        if prompt_name is not None:
            raise ValueError("PE-Core does not support task prompts.")
        if context_length is not None and context_length != PE_CORE_CONTEXT_LENGTH:
            raise ValueError("PE-Core context_length must be 32.")
        try:
            import open_clip
        except ImportError as error:
            raise ImportError("Install open-clip-torch to use PE-Core.") from error
        local = _snapshot(model_id, revision, cache_dir)
        return PECoreTokenizer(open_clip.get_tokenizer(f"local-dir:{local}"))

    def create_preprocess(self, model: PECoreCLIP) -> PECorePreprocess:
        if not isinstance(model, PECoreCLIP):
            raise TypeError("PE-Core model wrapper required.")
        return PECorePreprocess(model.processor)

    def encode_image(
        self, model: PECoreCLIP, image: ProcessedImageBatch, *,
        normalize: bool, truncate_dim: int | None,
    ) -> torch.Tensor:
        self.validate_embedding_dimension(truncate_dim, name="truncate_dim")
        if isinstance(image, Sequence) and not isinstance(
            image, (str, bytes, bytearray, Mapping)
        ):
            image = stack_processed_images(image, backend_name=self.model_name)
        if not isinstance(image, torch.Tensor):
            raise TypeError("PE-Core expects a preprocessed image tensor.")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 384, 384):
            raise ValueError("PE-Core image must have shape [B,3,384,384].")
        if not image.is_floating_point() or not torch.isfinite(image).all().item():
            raise ValueError("PE-Core image must be finite floating-point values.")
        param = next(model.backend.parameters())
        with torch.inference_mode():
            output = model.backend.encode_image(
                image.to(device=param.device, dtype=param.dtype), normalize=normalize
            )
        return self._validate_features(output, image.shape[0])

    def encode_text(
        self, model: PECoreCLIP, text: torch.Tensor, *,
        normalize: bool, truncate_dim: int | None,
    ) -> torch.Tensor:
        self.validate_embedding_dimension(truncate_dim, name="truncate_dim")
        if not isinstance(text, torch.Tensor) or text.dtype != torch.long or text.ndim != 2:
            raise TypeError("PE-Core text requires LongTensor[B,L].")
        if text.shape[0] == 0 or not 1 <= text.shape[1] <= PE_CORE_CONTEXT_LENGTH:
            raise ValueError("PE-Core text length must be between 1 and 32.")
        param = next(model.backend.parameters())
        with torch.inference_mode():
            output = model.backend.encode_text(
                text.to(device=param.device), normalize=normalize
            )
        return self._validate_features(output, text.shape[0])

    def get_logits(
        self, model: PECoreCLIP, image: ProcessedImageBatch, text: torch.Tensor,
        *, truncate_dim: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.encode_image(model, image, normalize=True, truncate_dim=truncate_dim)
        b = self.encode_text(model, text, normalize=True, truncate_dim=truncate_dim)
        if a.device != b.device or a.dtype != b.dtype:
            raise ValueError("Image/text embeddings must share device and dtype.")
        if a.device.type == "cpu" and a.dtype == torch.bfloat16:
            sims = (a.float() @ b.float().T).to(dtype=a.dtype)
        else:
            sims = a @ b.T
        logits = sims * model.backend.logit_scale.exp().to(device=a.device)
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("PE-Core returned non-finite logits.")
        return logits, logits.T

    @staticmethod
    def _validate_features(output: object, batch_size: int) -> torch.Tensor:
        if not isinstance(output, torch.Tensor) or output.shape != (batch_size, 512):
            raise RuntimeError("PE-Core must output [B,512] embeddings.")
        if not torch.isfinite(output).all().item():
            raise RuntimeError("PE-Core returned non-finite features.")
        return output
