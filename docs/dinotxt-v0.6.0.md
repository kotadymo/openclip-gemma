# dino.txt Integration Reference for OpenVLCLIP v0.6.0

## Scope and pinned source

This document records the Phase 0 source, artifact, and licensing investigation and the Phase 1 OpenVLCLIP backend implementation and parity checks. Phase 2 scoring/search, Phase 3 dense features, and Phase 4 release validation remain out of scope here until completed.

The official source was pinned to commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` in `facebookresearch/dinov2` (commit date: 2026-06-03). The pinned commit, rather than a moving `main` branch, is the reference for implementation and regression tests.

- [Pinned DINOv2 source tree](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)
- [Official Hub loader](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/dinotxt.py)
- [dino.txt model API](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/text/dinotxt_model.py)
- [Vision Tower and Head](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/text/vision_tower.py)
- [Text Tower](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/text/text_tower.py)
- [Text Transformer](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/text/text_transformer.py)
- [Tokenizer](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/text/tokenizer.py)
- [Official preprocessing](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/data/transforms.py)
- [Official dino.txt notebook](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/notebooks/dinotxt.ipynb)
- [DINOv2 Hub configuration](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/hubconf.py)

The registered OpenVLCLIP model name is `dinov2-dinotxt`; the exact official Hub entry is `dinov2_vitl14_reg4_dinotxt_tet1280d20h24l`.

## Official architecture and inference contract

The pinned Hub loader constructs the model as follows:

- Vision backbone: pretrained DINOv2 ViT-L/14 with four register tokens (`dinov2_vitl14_reg()`).
- Vision Tower: a learned two-block Vision Head applied to the class, register, and patch tokens. The learned Head must be used; raw DINOv2 backbone features are not aligned with the text encoder.
- Global image representation: the Head output for the class token concatenated with the mean of the Head output for patch tokens, producing 2048 dimensions (1024 + 1024).
- Patch representation: the Head output for image patches after removing the class and four register tokens, with 1024 dimensions per patch.
- Text backbone: causal Transformer with 77-token context, vocabulary size 49,408, width 1,280, 20 heads, and 24 layers.
- Text output: the Text Tower applies its learned 2048-dimensional linear projection and pools at the position of the largest token ID (the end-of-text token).
- Global text representation: 2048 dimensions; the first and second 1024-dimensional halves have distinct alignment roles. The second half is the text representation aligned to patch features.

The official tokenizer adds start-of-text ID `49406` and end-of-text ID `49407`, truncates to the requested context length, replaces the final retained token with end-of-text when truncating, and zero-pads to 77 tokens by default. An empty string is tokenized as start-of-text, end-of-text, then padding; APIs that require non-empty search/map text must reject it explicitly rather than infer different tokenizer behavior.

`make_classification_eval_transform()` uses bicubic `Resize(256)` (short side), `CenterCrop(224)`, tensor conversion, and ImageNet normalization with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. The transform does not itself convert to RGB; the notebook supplies RGB images. OpenVLCLIP must make its RGB input contract explicit and match these operations for RGB inputs.

The official `DinoTxt.encode_image()` and `encode_text()` return raw features when `normalize=False` and L2-normalize the full 2048-dimensional feature when `normalize=True`. This differs from EmbeddingGemma's established behavior and must remain backend-specific. No MRL/truncation within the 2048-dimensional dino.txt representation is supported.

### Logits and scale

`DinoTxt.get_logits()` normalizes the complete image and text vectors and computes their dot product multiplied by `exp(logit_scale)`. The Hub configuration initializes `logit_scale` to `log(1 / 0.07)`. The loader loads the Vision Head and Text Tower state dictionaries, not a full `DinoTxt` state dictionary containing `logit_scale`; therefore this value is the Hub-initialized scale, not a learned scale loaded from the published checkpoints. The verified value was approximately `14.285715`.

`DinoTxt.forward()` returns normalized global image/text features and the same scale. Cosine similarity is the scale-1 dot product of normalized features and is not interchangeable with `get_logits()`. The implementation must not substitute the notebook's evaluation calculations or a hard-coded scale for the official model API.

### Dense inference

The official notebook obtains patch features from `get_visual_class_and_patch_tokens()`, obtains the text-side patch-aligned representation from `encode_text(tokens)[:, 1024:]`, and reshapes patch tokens to a spatial grid. It converts the feature map to float32, applies bicubic `F.interpolate(..., align_corners=False)` when resizing, then L2-normalizes each spatial feature and the text suffix before computing cosine scores with an `einsum`.

The interpolation-before-normalization order is material. The notebook's square-grid example uses `H = W = int(P**0.5)` for a 224-by-224 input; a general OpenVLCLIP implementation must derive both grid dimensions from the image/patch size rather than assume a square grid. The result is a similarity score map, not a probability or a pixel-accurate segmentation mask.

## Official artifacts and hashes

The pinned Hub loader and official README identify these artifacts. Checkpoint files were present in the PyTorch Hub cache; their byte lengths matched the official HTTP `Content-Length`, and their multipart S3 ETags matched when recalculated with 8 MiB parts. The BPE vocabulary was fetched from the official URL, and its MD5 matched the single-part ETag. SHA-256 values below were computed from those bytes.

| Component | Official URL | Bytes | SHA-256 | Official ETag |
| --- | --- | ---: | --- | --- |
| DINOv2 ViT-L/14 reg4 backbone | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth` | 1,217,607,321 | `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51` | `b6cbe2bf3ce2f370d5a67bcd465144b0-146` |
| dino.txt Vision Head | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth` | 101,317,814 | `706fba5bcb7019d80e127f0ff123872a75e0e6099527cf1911db85bd61c88ef5` | `250de829d127ad9e0115173186d9719b-13` |
| dino.txt Text Encoder | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth` | 2,152,639,923 | `13378274c79f67af8b47b11dd57827d659e7b1ee01e4dde58d2266422ea7846a` | `48d005e5e9a8900e0a13aba88ab92ba6-257` |
| CLIP BPE vocabulary | `https://dl.fbaipublicfiles.com/dinov2/thirdparty/bpe_simple_vocab_16e6.txt.gz` | 1,356,917 | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` | `933b7abbbbde62c36f02f0e6ccde464f` |

The dino.txt weights are external runtime downloads and are not included in the OpenVLCLIP wheel or sdist. Phase 1 must use these fixed URLs and verify the expected SHA-256 before accepting cached or newly downloaded artifacts; a mismatching cache must fail explicitly.

## License and dependency findings

- The pinned DINOv2 repository `LICENSE` is Apache-2.0. The official `MODEL_CARD.md` also lists Apache-2.0 for the DINOv2 ViT-S/B/L/g model family.
- The official README links the dino.txt Vision Head and Text Encoder checkpoints but does not state their license or link a dino.txt-specific checkpoint license. The pinned repository-level Apache-2.0 license is evidence for the repository, but it does not conclusively establish rights for the externally hosted dino.txt checkpoints. Do not redistribute these weights; obtain explicit licensing confirmation before the v0.6.0 release. This is a release gate, not legal advice.
- The official README explicitly links a separate license for the BPE vocabulary. That license is the MIT License (OpenAI, 2021). The repository also has a third-party CLIP license file; retain applicable notices if any upstream tokenizer code is ever copied rather than loaded from the pinned source.
- The pinned repository contains additional Cell-DINO and X-Ray-DINO license files for those separate model families. Neither those models nor their weights/code are used by this integration.
- The dino.txt tokenizer requires `ftfy` and `regex`, so these are declared in the optional `[dinotxt]` extra rather than being required for a base EmbeddingGemma-only import. BPE retrieval uses the standard-library URL loader and verifies the pinned SHA-256. xFormers is optional in the tested path: the official implementation emitted fallback warnings and completed inference without xFormers.
- The pinned safe state-dict loader passes `weights_only=True` for PyTorch 2.1 and newer. The Hub source itself is executable Python; production loading must always use the pinned commit, not a mutable branch.

## Phase 0 runtime verification

The official Hub loader was run on the pinned commit in the existing Windows CUDA environment:

- Python `3.12.10`
- PyTorch `2.14.1+cu130`, torchvision `0.29.1+cu130`
- CUDA `13.0`, NVIDIA GeForce RTX 4070 Ti SUPER, BF16 supported
- Pillow `13.0.0`, `ftfy 6.3.1`, `requests 2.34.2`, `regex 2026.9.29`
- xFormers was not installed; the official PyTorch fallback path completed the checks.

The smoke input was a deterministic synthetic RGB image (256 by 320 pixels), not an evaluation dataset. The official preprocessing returned `(1, 3, 224, 224)`. The official tokenizer returned `(3, 77)` for `"a red car"`, `"a blue car"`, and `""`; the first five IDs for the first two strings were respectively `[49406, 320, 736, 1615, 49407]` and `[49406, 320, 1746, 1615, 49407]`.

| Check | Observed result |
| --- | --- |
| Raw global image features | `(1, 2048)`, finite |
| Raw global text features | `(3, 2048)`, finite |
| Vision Head class output | `(1, 1024)` |
| Vision Head patch output | `(1, 256, 1024)`, finite |
| `get_logits()` / transpose | `(1, 3)` / `(3, 1)`, finite |
| `exp(logit_scale)` | `14.285715103149414` |
| Official logits vs normalized-feature dot product times official scale | maximum absolute difference `0.0` |
| `encode_image()` vs class token concatenated with mean patch feature | maximum absolute difference `0.0` |
| `forward()` feature norms | image `0.99999994`; text `1.0`, `0.99999994`, `1.0` |

These checks establish that the pinned official model and tokenizer load and that the expected computation paths/shapes are reachable on CUDA FP32. They are not a wrapper parity test, accuracy benchmark, or evidence of CPU/BF16 support.

## Phase 1 backend verification

The new `dinov2-dinotxt` Registry entry resolves only the official Hub model and pinned revision. The shared Factory defaults `pretrained` to `"official"` for both registered models; for EmbeddingGemma 2 this still resolves to the existing `google/embeddinggemma-2` checkpoint. Explicit `pretrained=None` remains rejected. Listing models and pretrained tags does not load weights. dino.txt rejects selective loading, task prompts, unsupported checkpoint identifiers, and every truncation dimension other than its native 2048 dimensions.

The wrapper was compared with the pinned official implementation using the same model instance and identical inference calls. Its deterministic preprocessing output and tokenizer token IDs were exactly equal to the official outputs for the tested RGB images and text strings. Global image/text embeddings had 2048 dimensions. Maximum absolute error for raw and normalized image/text embeddings was `0.0` in all tested cases:

| Device / precision | Image max absolute error | Text max absolute error |
| --- | ---: | ---: |
| CPU FP32 | `0.0` | `0.0` |
| CPU BF16 | `0.0` | `0.0` |
| CUDA FP32 | `0.0` | `0.0` |
| CUDA BF16 | `0.0` | `0.0` |

This is parity of the adapter/wrapper against the same pinned official model path, not an independent model reimplementation or a dataset accuracy benchmark. The tested environment was the Windows/Python 3.12.10/CUDA 13.0 environment described above. xFormers was unavailable; the official PyTorch fallback path passed.

Phase 1 local checks: `pytest -m "not integration"` completed with 288 passed and 27 deselected; the pinned dino.txt parity integration test passed; Ruff passed; `uv lock --check` passed; and wheel/sdist builds passed. The built wheel metadata identifies `openvlclip==0.5.0`, advertises the `dinotxt` extra with its optional dependencies, and contains the CLIP license notice and both new backend modules. The sdist contains the same license notice and backend modules. Linux, Python 3.13/3.14, CUDA-less systems, and clean installation of the built wheel have not yet been checked in this phase.

## Remaining validation before release

Phase 1 verified the backend, tokenizer, preprocessing, and global embedding adapter. The following remain required in later phases and must not be reported as verified until actually run:

- Official logits/`forward()` and cosine-mode parity, zero-shot classification, text-to-image search, and embedding-index compatibility.
- Patch-feature and similarity-map parity, including interpolation/normalization order.
- Linux and Python 3.13/3.14 compatibility, including the project's Python 3.14.1 exclusion.
- Gemma actual-model regression, v0.4.0 fixture/API regressions, image/video index compatibility, search-space mismatch rejection, and ordinary CI.
- Clean installation of built distributions with `[dinotxt]` and compatibility with `[video]`.
- Dataset-based accuracy and performance benchmarks. No dataset benchmark was run in Phase 0; those metrics are `not evaluated`.
- Explicit licensing confirmation for the externally hosted dino.txt Vision Head and Text Encoder weights. Until this gate is resolved, the v0.6.0 release must not be declared ready.
