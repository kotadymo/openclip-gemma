"""Pinned Meta DINOv2 + dino.txt model backend."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import urllib.error
import urllib.request
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from openvlclip._dimensions import validate_positive_embedding_dimension
from openvlclip.backends.dinotxt_tokenizer import DinoTxtTokenizer
from openvlclip.capabilities import capabilities_for_modalities, freeze_capabilities
from openvlclip.embedding_space import EmbeddingSpaceMetadata

DINO_MODEL_NAME = "dinov2-dinotxt"
DINO_HUB_MODEL_ID = "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l"
DINO_SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_BACKEND_ID = "meta-dinov2-dinotxt"

_BACKBONE_SHA256 = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
_VISION_HEAD_SHA256 = "706fba5bcb7019d80e127f0ff123872a75e0e6099527cf1911db85bd61c88ef5"
_TEXT_ENCODER_SHA256 = "13378274c79f67af8b47b11dd57827d659e7b1ee01e4dde58d2266422ea7846a"
_BPE_VOCAB_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
_DINO_MODEL_REVISION = (
    f"{DINO_SOURCE_REVISION}+backbone-{_BACKBONE_SHA256}"
    f"+vision-{_VISION_HEAD_SHA256}+text-{_TEXT_ENCODER_SHA256}"
    f"+bpe-{_BPE_VOCAB_SHA256}"
)
DINO_MODEL_REVISION = _DINO_MODEL_REVISION
_DINO_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
_HUB_REPOSITORY = f"facebookresearch/dinov2:{DINO_SOURCE_REVISION}"
_HUB_MODEL = "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l"
_IMAGE_SIZE = 224
_PATCH_SIZE = 14
_EMBEDDING_DIMENSION = 2048
_START_TOKEN_ID = 49406
_END_TOKEN_ID = 49407
_VOCAB_SIZE = 49408
_TORCH_HUB_LOCK = threading.RLock()


@dataclass(frozen=True)
class _Artifact:
    filename: str
    url: str
    sha256: str


_BACKBONE_ARTIFACT = _Artifact(
    filename="dinov2_vitl14_reg4_pretrain.pth",
    url=f"{_DINO_BASE_URL}/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    sha256=_BACKBONE_SHA256,
)
_VISION_HEAD_ARTIFACT = _Artifact(
    filename="dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
    url=(
        f"{_DINO_BASE_URL}/dinov2_vitl14/"
        "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth"
    ),
    sha256=_VISION_HEAD_SHA256,
)
_TEXT_ENCODER_ARTIFACT = _Artifact(
    filename="dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth",
    url=(
        f"{_DINO_BASE_URL}/dinov2_vitl14/"
        "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth"
    ),
    sha256=_TEXT_ENCODER_SHA256,
)
_BPE_VOCAB_ARTIFACT = _Artifact(
    filename="bpe_simple_vocab_16e6.txt.gz",
    url=f"{_DINO_BASE_URL}/thirdparty/bpe_simple_vocab_16e6.txt.gz",
    sha256=_BPE_VOCAB_SHA256,
)
_MODEL_ARTIFACTS = (
    _BACKBONE_ARTIFACT,
    _VISION_HEAD_ARTIFACT,
    _TEXT_ENCODER_ARTIFACT,
)


@contextmanager
def _hub_cache_directory(cache_dir: str | None) -> Generator[Path, None, None]:
    if cache_dir is not None and not isinstance(cache_dir, str):
        raise TypeError("cache_dir must be a string or None.")

    with _TORCH_HUB_LOCK:
        original_hub_dir = torch.hub.get_dir()
        if cache_dir is not None:
            hub_dir = Path(cache_dir).expanduser().resolve()
            hub_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(hub_dir))
        try:
            yield Path(torch.hub.get_dir())
        finally:
            if cache_dir is not None:
                torch.hub.set_dir(original_hub_dir)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_verified_artifact(hub_dir: Path, artifact: _Artifact) -> Path:
    checkpoint_dir = hub_dir / "checkpoints"
    target = checkpoint_dir / artifact.filename
    if target.exists():
        if not target.is_file():
            raise RuntimeError(f"Cached dino.txt artifact is not a file: {target}.")
        actual_sha256 = _sha256_file(target)
        if actual_sha256 != artifact.sha256:
            raise RuntimeError(
                f"Cached dino.txt artifact {target} has SHA-256 "
                f"{actual_sha256}; expected {artifact.sha256}."
            )
        return target

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact.filename}.",
        suffix=".part",
        dir=checkpoint_dir,
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        with (
            urllib.request.urlopen(artifact.url, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(8 * 1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != artifact.sha256:
            raise RuntimeError(
                f"Downloaded dino.txt artifact {artifact.filename} has SHA-256 "
                f"{actual_sha256}; expected {artifact.sha256}."
            )
        os.replace(temporary, target)
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Failed to download official dino.txt artifact {artifact.filename} "
            f"from {artifact.url}."
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return target


class DinoTxtPreprocess:
    """Apply the deterministic official DINOv2 classification evaluation transform."""

    def __init__(self) -> None:
        from torchvision.transforms import (
            CenterCrop,
            Compose,
            InterpolationMode,
            Normalize,
            Resize,
            ToTensor,
        )

        self._transform = Compose(
            (
                Resize(256, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(_IMAGE_SIZE),
                ToTensor(),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            )
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("preprocess expects a PIL.Image.Image.")
        pixels = self._transform(image.convert("RGB"))
        if (
            pixels.dtype != torch.float32
            or pixels.ndim != 3
            or pixels.shape != (3, _IMAGE_SIZE, _IMAGE_SIZE)
            or not torch.isfinite(pixels).all().item()
        ):
            raise RuntimeError(
                "The official dino.txt preprocessing must return finite "
                "float32 Tensor[3,224,224]."
            )
        return pixels


class DinoTxtCLIP(nn.Module):
    """OpenCLIP-style wrapper around the official dino.txt model."""

    def __init__(
        self,
        backend: nn.Module,
        inference_backend: DinoV2DinoTxtBackend,
        *,
        output_dict: bool,
        logit_scale_mode: str,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(backend, nn.Module):
            raise TypeError("backend must be a torch.nn.Module.")
        self.backend = backend
        object.__setattr__(self, "_inference_backend", inference_backend)
        self.output_dict = output_dict
        self.logit_scale_mode = logit_scale_mode
        self.cache_dir = cache_dir
        self.modalities = ("text", "image")

    @property
    def capabilities(self) -> Mapping[str, bool]:
        return self._inference_backend.capabilities_for(self.modalities)

    @property
    def embedding_space_metadata(self) -> EmbeddingSpaceMetadata:
        return self._inference_backend.embedding_space_metadata

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int:
        return self._inference_backend.validate_embedding_dimension(value, name=name)

    def to(self, *args: Any, **kwargs: Any) -> DinoTxtCLIP:
        requested_dtype = kwargs.get("dtype")
        for argument in args:
            if isinstance(argument, torch.dtype):
                requested_dtype = argument
            elif isinstance(argument, torch.Tensor):
                requested_dtype = argument.dtype
        if requested_dtype == torch.float16:
            raise ValueError("FP16 is not supported; use FP32 or BF16.")
        return super().to(*args, **kwargs)

    def half(self) -> DinoTxtCLIP:
        raise ValueError("FP16 is not supported; use FP32 or BF16.")

    def encode_image(
        self,
        image: torch.Tensor,
        normalize: bool = False,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        return self._inference_backend.encode_image(
            self,
            image,
            normalize=normalize,
            truncate_dim=truncate_dim,
        )

    def encode_text(
        self,
        text: torch.Tensor,
        normalize: bool = False,
        *,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        return self._inference_backend.encode_text(
            self,
            text,
            normalize=normalize,
            truncate_dim=truncate_dim,
        )

    def forward(
        self,
        image: torch.Tensor | None = None,
        text: torch.Tensor | None = None,
        *,
        truncate_dim: int | None = None,
    ) -> (
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]
        | dict[str, object]
    ):
        return self._inference_backend.forward(
            self,
            image,
            text,
            truncate_dim=truncate_dim,
        )

    def get_logits(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        *,
        truncate_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._inference_backend.get_logits(
            self,
            image,
            text,
            truncate_dim=truncate_dim,
        )

    def _create_tokenizer(self, *, prompt_name: str | None) -> DinoTxtTokenizer:
        metadata = self.embedding_space_metadata
        return self._inference_backend.create_tokenizer(
            model_id=metadata.model_id,
            revision=metadata.model_revision,
            cache_dir=self.cache_dir,
            prompt_name=prompt_name,
            context_length=None,
        )


class DinoV2DinoTxtBackend:
    """Load the fixed official model and preserve its global embedding semantics."""

    backend_id = DINO_BACKEND_ID
    capabilities: Mapping[str, bool] = freeze_capabilities(
        {
            "image_embedding": True,
            "text_embedding": True,
            "image_text_similarity": True,
            "patch_embedding": False,
            "text_similarity_map": False,
            "mrl": False,
            "selective_loading": False,
        }
    )
    embedding_space_metadata = EmbeddingSpaceMetadata(
        backend_id=DINO_BACKEND_ID,
        model_id=DINO_HUB_MODEL_ID,
        model_revision=_DINO_MODEL_REVISION,
        embedding_strategy="dinotxt-global-cls-plus-mean-patches-v1",
        normalization="none-or-l2-global-2048-v1",
        preprocessing_id="dinov2-official-bicubic256-centercrop224-imagenet-v1",
        index_prompt=None,
        query_prompt_name=None,
        query_prompt=None,
    )

    def validate_embedding_dimension(
        self,
        value: object | None,
        *,
        name: str,
    ) -> int:
        if value is None:
            return _EMBEDDING_DIMENSION
        dimension = validate_positive_embedding_dimension(value, name=name)
        if dimension != _EMBEDDING_DIMENSION:
            raise ValueError(
                f"{name} must be {_EMBEDDING_DIMENSION}; dino.txt does not support "
                f"MRL or truncation. Received {dimension}."
            )
        return dimension

    def capabilities_for(self, modalities: tuple[str, ...]) -> Mapping[str, bool]:
        return capabilities_for_modalities(self.capabilities, modalities)

    def load_model(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: str | None,
        device: torch.device,
        dtype: torch.dtype,
        output_dict: bool,
        logit_scale_mode: str,
        modalities: tuple[str, ...] | None,
    ) -> DinoTxtCLIP:
        if model_id != DINO_HUB_MODEL_ID or revision != _DINO_MODEL_REVISION:
            raise ValueError(
                "Only the registered official DINOv2 ViT-L/14 reg4 dino.txt "
                "checkpoint and pinned revision are supported."
            )
        if modalities not in (None, ("text", "image")):
            raise NotImplementedError(
                "dino.txt selective loading is not supported; load both text and "
                "image modalities."
            )
        if dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("dino.txt supports only FP32 and BF16 precision.")
        if logit_scale_mode not in {"strict", "cosine"}:
            raise ValueError("logit_scale_mode must be 'strict' or 'cosine'.")
        if not isinstance(output_dict, bool):
            raise TypeError("output_dict must be a bool.")

        with _hub_cache_directory(cache_dir) as hub_dir:
            for artifact in _MODEL_ARTIFACTS:
                _ensure_verified_artifact(hub_dir, artifact)
            try:
                backend = torch.hub.load(
                    _HUB_REPOSITORY,
                    _HUB_MODEL,
                    source="github",
                    trust_repo=True,
                )
            except (ImportError, OSError, RuntimeError) as error:
                raise RuntimeError(
                    "Failed to load the official dino.txt model from pinned "
                    f"DINOv2 source commit {DINO_SOURCE_REVISION}."
                ) from error

        if not isinstance(backend, nn.Module):
            raise TypeError("The pinned DINOv2 Hub entry did not return an nn.Module.")
        backend = backend.to(device=device, dtype=dtype).eval()
        return DinoTxtCLIP(
            backend=backend,
            inference_backend=self,
            output_dict=output_dict,
            logit_scale_mode=logit_scale_mode,
            cache_dir=cache_dir,
        ).eval()

    def create_tokenizer(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: str | None,
        prompt_name: str | None,
        context_length: int | None,
    ) -> DinoTxtTokenizer:
        if model_id != DINO_HUB_MODEL_ID or revision != _DINO_MODEL_REVISION:
            raise ValueError(
                "Only the registered official dino.txt tokenizer is supported."
            )
        if prompt_name is not None:
            raise ValueError(
                "dino.txt has no task prompts; prompt_name must be None."
            )
        if cache_dir is not None and not isinstance(cache_dir, str):
            raise TypeError("cache_dir must be a string or None.")
        resolved_context_length = 77 if context_length is None else context_length
        if (
            isinstance(resolved_context_length, bool)
            or not isinstance(resolved_context_length, int)
            or not 2 <= resolved_context_length <= 77
        ):
            raise ValueError("context_length must be an integer between 2 and 77.")

        from openvlclip.backends.dinotxt_tokenizer import (
            SimpleTokenizer,
        )

        with _hub_cache_directory(cache_dir) as hub_dir:
            bpe_path = _ensure_verified_artifact(hub_dir, _BPE_VOCAB_ARTIFACT)
        try:
            tokenizer = SimpleTokenizer(bpe_path)
        except ImportError as error:
            raise ImportError(
                'dino.txt tokenization requires the "dinotxt" extra; install it with '
                '`pip install "openvlclip[dinotxt]"`.'
            ) from error
        return DinoTxtTokenizer(
            tokenizer,
            context_length=resolved_context_length,
        )

    def create_preprocess(self, model: DinoTxtCLIP) -> DinoTxtPreprocess:
        if not isinstance(model, DinoTxtCLIP):
            raise TypeError("The dino.txt backend requires a DinoTxtCLIP model.")
        return DinoTxtPreprocess()

    def forward(
        self,
        model: DinoTxtCLIP,
        image: torch.Tensor | None,
        text: torch.Tensor | None,
        *,
        truncate_dim: int | None,
    ) -> (
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]
        | dict[str, object]
    ):
        if image is None and text is None:
            raise ValueError("At least one of image or text must be provided.")
        self.validate_embedding_dimension(truncate_dim, name="truncate_dim")

        prepared_image: torch.Tensor | None = None
        prepared_text: torch.Tensor | None = None
        model_device, model_dtype = self._model_device_dtype(model)
        if image is not None:
            prepared_image, image_device, image_dtype = self._prepare_image_input(
                model,
                image,
                truncate_dim,
            )
            model_device, model_dtype = image_device, image_dtype
        if text is not None:
            prepared_text, text_device, text_dtype = self._prepare_text_input(
                model,
                text,
                truncate_dim,
            )
            model_device, model_dtype = text_device, text_dtype

        with torch.inference_mode():
            if prepared_image is not None and prepared_text is not None:
                output = model.backend(prepared_image, prepared_text)
                if not isinstance(output, tuple) or len(output) != 3:
                    raise RuntimeError(
                        "The official dino.txt forward() must return three values."
                    )
                image_features, text_features, official_scale = output
            else:
                image_features = (
                    model.backend.encode_image(prepared_image, normalize=True)
                    if prepared_image is not None
                    else None
                )
                text_features = (
                    model.backend.encode_text(prepared_text, normalize=True)
                    if prepared_text is not None
                    else None
                )
                official_scale = model.backend.logit_scale.exp()

        if image_features is not None:
            image_features = self._validate_features(
                image_features,
                batch_size=image.shape[0] if image is not None else 0,
                device=model_device,
                dtype=model_dtype,
                normalize=False,
                modality="image",
            )
        if text_features is not None:
            text_features = self._validate_features(
                text_features,
                batch_size=text.shape[0] if text is not None else 0,
                device=model_device,
                dtype=model_dtype,
                normalize=False,
                modality="text",
            )

        if model.logit_scale_mode == "strict":
            if (
                not isinstance(official_scale, torch.Tensor)
                or official_scale.numel() != 1
                or official_scale.device != model_device
                or not torch.isfinite(official_scale).all().item()
            ):
                raise RuntimeError(
                    "The official dino.txt forward() returned an invalid logit scale."
                )
            logit_scale = official_scale
        else:
            logit_scale = torch.ones((), dtype=torch.float32, device=model_device)

        if model.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": logit_scale,
            }
        return image_features, text_features, logit_scale

    def get_logits(
        self,
        model: DinoTxtCLIP,
        image: torch.Tensor,
        text: torch.Tensor,
        *,
        truncate_dim: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepared_image, device, dtype = self._prepare_image_input(
            model,
            image,
            truncate_dim,
        )
        prepared_text, _, _ = self._prepare_text_input(
            model,
            text,
            truncate_dim,
        )
        with torch.inference_mode():
            if model.logit_scale_mode == "strict":
                logits = model.backend.get_logits(prepared_image, prepared_text)
            else:
                image_features = model.backend.encode_image(
                    prepared_image,
                    normalize=True,
                )
                text_features = model.backend.encode_text(
                    prepared_text,
                    normalize=True,
                )
                if dtype == torch.bfloat16 and device.type == "cpu":
                    image_logits = image_features.float() @ text_features.float().T
                    image_logits = image_logits.to(dtype=dtype)
                else:
                    image_logits = image_features @ text_features.T
                logits = image_logits, image_logits.T
        return self._validate_logits(
            logits,
            image_count=image.shape[0],
            text_count=text.shape[0],
            device=device,
            dtype=dtype,
        )

    def encode_image(
        self,
        model: DinoTxtCLIP,
        image: torch.Tensor,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor:
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool.")
        image_batch, model_device, model_dtype = self._prepare_image_input(
            model,
            image,
            truncate_dim,
        )
        with torch.inference_mode():
            features = model.backend.encode_image(
                image_batch,
                normalize=False,
            )
        return self._validate_features(
            features,
            batch_size=image.shape[0],
            device=model_device,
            dtype=model_dtype,
            normalize=normalize,
            modality="image",
        )

    def _prepare_image_input(
        self,
        model: DinoTxtCLIP,
        image: torch.Tensor,
        truncate_dim: int | None,
    ) -> tuple[torch.Tensor, torch.device, torch.dtype]:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor.")
        if image.ndim != 4:
            raise ValueError(
                f"image must have shape [B,3,H,W]; received {tuple(image.shape)}."
            )
        if image.shape[0] == 0:
            raise ValueError("image batch must contain at least one image.")
        if image.shape[1] != 3:
            raise ValueError(f"image must have 3 channels; received {image.shape[1]}.")
        if image.shape[2] <= 0 or image.shape[3] <= 0:
            raise ValueError("image height and width must be positive.")
        if image.shape[2] % _PATCH_SIZE or image.shape[3] % _PATCH_SIZE:
            raise ValueError(
                "preprocessed image height and width must be multiples of "
                f"patch_size={_PATCH_SIZE}; received {image.shape[2:]}."
            )
        if image.dtype != torch.float32:
            raise TypeError(
                f"image must use torch.float32 from preprocess; received {image.dtype}."
            )
        self.validate_embedding_dimension(truncate_dim, name="truncate_dim")
        model_device, model_dtype = self._model_device_dtype(model)
        if image.device != model_device:
            raise ValueError(
                f"image is on {image.device}, but the model backend is on {model_device}."
            )
        if not torch.isfinite(image).all().item():
            raise ValueError("image contains NaN or infinite values.")
        return image.to(dtype=model_dtype), model_device, model_dtype

    def encode_text(
        self,
        model: DinoTxtCLIP,
        text: torch.Tensor,
        *,
        normalize: bool,
        truncate_dim: int | None,
    ) -> torch.Tensor:
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool.")
        text_batch, model_device, model_dtype = self._prepare_text_input(
            model,
            text,
            truncate_dim,
        )
        with torch.inference_mode():
            features = model.backend.encode_text(text_batch, normalize=False)
        return self._validate_features(
            features,
            batch_size=text.shape[0],
            device=model_device,
            dtype=model_dtype,
            normalize=normalize,
            modality="text",
        )

    def _prepare_text_input(
        self,
        model: DinoTxtCLIP,
        text: torch.Tensor,
        truncate_dim: int | None,
    ) -> tuple[torch.Tensor, torch.device, torch.dtype]:
        if not isinstance(text, torch.Tensor):
            raise TypeError("text must be a torch.Tensor.")
        if text.ndim != 2:
            raise ValueError(f"text must have shape [B,L]; received {tuple(text.shape)}.")
        if text.shape[0] == 0:
            raise ValueError("text batch must contain at least one sequence.")
        if not 2 <= text.shape[1] <= 77:
            raise ValueError("text sequence length must be between 2 and 77.")
        if text.dtype != torch.long:
            raise TypeError(
                f"text must use torch.int64 token IDs; received {text.dtype}."
            )
        self.validate_embedding_dimension(truncate_dim, name="truncate_dim")
        if text.min().item() < 0 or text.max().item() >= _VOCAB_SIZE:
            raise ValueError(f"text token IDs must be in [0, {_VOCAB_SIZE - 1}].")
        if (text[:, 0] != _START_TOKEN_ID).any().item():
            raise ValueError("Each dino.txt token sequence must start with SOT ID 49406.")
        end_tokens = text.eq(_END_TOKEN_ID)
        if not end_tokens.any(dim=1).all().item():
            raise ValueError("Each dino.txt token sequence must contain EOT ID 49407.")
        positions = torch.arange(text.shape[1], device=text.device).expand_as(text)
        final_end_positions = positions.masked_fill(~end_tokens, -1).max(dim=1).values
        if (
            text.masked_fill(positions <= final_end_positions[:, None], 0) != 0
        ).any().item():
            raise ValueError("Token IDs after the final EOT must be zero padding.")

        model_device, model_dtype = self._model_device_dtype(model)
        if text.device != model_device:
            raise ValueError(
                f"text is on {text.device}, but the model backend is on {model_device}."
            )
        return text, model_device, model_dtype

    @staticmethod
    def _validate_logits(
        logits: object,
        *,
        image_count: int,
        text_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(logits, tuple) or len(logits) != 2:
            raise RuntimeError("The official dino.txt get_logits() must return two values.")
        image_logits, text_logits = logits
        expected_image_shape = (image_count, text_count)
        expected_text_shape = (text_count, image_count)
        for values, expected_shape, modality in (
            (image_logits, expected_image_shape, "image"),
            (text_logits, expected_text_shape, "text"),
        ):
            if not isinstance(values, torch.Tensor) or tuple(values.shape) != expected_shape:
                actual_shape = (
                    tuple(values.shape)
                    if isinstance(values, torch.Tensor)
                    else type(values).__name__
                )
                raise RuntimeError(
                    f"The official dino.txt returned invalid {modality} logits: "
                    f"expected {expected_shape}, received {actual_shape}."
                )
            if values.device != device or values.dtype != dtype:
                raise RuntimeError(
                    f"The official dino.txt returned {modality} logits on "
                    f"{values.device} with {values.dtype}; expected {device} with {dtype}."
                )
            if not torch.isfinite(values).all().item():
                raise RuntimeError(
                    f"The official dino.txt returned non-finite {modality} logits."
                )
        if not torch.equal(text_logits, image_logits.T):
            raise RuntimeError(
                "The official dino.txt text logits must transpose image logits."
            )
        return image_logits, text_logits

    @staticmethod
    def _model_device_dtype(model: DinoTxtCLIP) -> tuple[torch.device, torch.dtype]:
        try:
            parameter = next(model.backend.parameters())
        except StopIteration as error:
            raise RuntimeError(
                "The official dino.txt model has no parameters to determine "
                "its device and dtype."
            ) from error
        return parameter.device, parameter.dtype

    @staticmethod
    def _validate_features(
        features: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        normalize: bool,
        modality: str,
    ) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"The official model returned non-Tensor {modality} features.")
        if features.shape != (batch_size, _EMBEDDING_DIMENSION):
            raise RuntimeError(
                f"The official model returned {modality} features with shape "
                f"{tuple(features.shape)}; expected "
                f"[{batch_size},{_EMBEDDING_DIMENSION}]."
            )
        if features.device != device or features.dtype != dtype:
            raise RuntimeError(
                f"The official model returned {modality} features on "
                f"{features.device} with {features.dtype}; expected {device} with {dtype}."
            )
        if not torch.isfinite(features).all().item():
            raise RuntimeError(f"The official model returned non-finite {modality} features.")
        if normalize:
            features = F.normalize(features, p=2, dim=-1)
        return features
