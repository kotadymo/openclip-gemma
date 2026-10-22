# OpenVLCLIP

An independent Python library that provides an OpenCLIP-inspired inference API for Google's EmbeddingGemma 2 and Meta's DINOv2 + dino.txt models. It lets Python applications use familiar image and text encoding calls while retaining each model's own tokenizer, image processing, prompts, and embedding behavior.

> [!IMPORTANT]
> **Unofficial project**
>
> `openvlclip` is an independent, unofficial project. It is not affiliated with, endorsed by, sponsored by, or officially maintained by Google, Google DeepMind, or the OpenCLIP project.
>
> This library provides an OpenCLIP-inspired interface for EmbeddingGemma 2 and DINOv2 + dino.txt. It is not a drop-in replacement for OpenCLIP and does not claim full API compatibility.

## Features

- OpenCLIP-style `create_model`, `create_model_and_transforms`, and `get_tokenizer` entry points.
- Factory discovery with `list_models()`, `list_pretrained()`, and `create_model_from_pretrained()`.
- A Model Registry, shared Backend interface, and per-model Capabilities for `embeddinggemma-2` and `dinov2-dinotxt`; this is not a complete drop-in replacement for OpenCLIP.
- EmbeddingGemma uses the official Transformers model/processor and Sentence Transformers numerical reference; dino.txt follows the pinned Meta Hub implementation and checkpoints.
- Model API includes `forward()`, `get_logits()`, and factory option `output_dict=True`.
- EmbeddingGemma `forward()` and `get_logits()` require explicit `logit_scale_mode="cosine"`; dino.txt strict mode preserves its official scaled-logit path.
- Cosine mode returns normalized features with scale `1.0`; these scores are not trained OpenCLIP logits or calibrated probabilities.
- EmbeddingGemma 2's native 768-dimensional output and MRL dimensions 512, 256, and 128; dino.txt's native 2048-dimensional global and 1024-dimensional patch features.
- Optional text-only or text-and-image model loading; the default factory load remains unchanged.
- Pinned model revisions; model weights are downloaded at runtime and are not bundled with this package.
- Bounded-batch image embedding and incremental Japanese/English text-to-image search.
- dino.txt zero-shot classification, image search, patch embeddings, and text similarity maps.
- Optional PyAV-based video-frame indexing and text search.

## Installation

Install the package into an active virtual environment. Python 3.12 and later
are supported, up to but not including Python 3.15, except Python 3.14.1
(`>=3.12,<3.15,!=3.14.1`).

```bash
pip install openvlclip==0.6.0
```

Install dino.txt's optional tokenizer dependencies with:

```bash
pip install "openvlclip[dinotxt]==0.6.0"
```

Install the optional video dependency for video-frame indexing and search:

```bash
pip install "openvlclip[video]==0.6.0"
```

OpenVLCLIP v0.6.0 was published on 2026-10-22 to
[PyPI](https://pypi.org/project/openvlclip/0.6.0/) with a public
[GitHub Release](https://github.com/kotadymo/openvlclip/releases/tag/v0.6.0).
The earlier v0.5.0 distribution remains EmbeddingGemma-only; install v0.6.0
for DINOv2 + dino.txt support. The wheel and sdist do not include model
weights.

For development from a source checkout, install the development and optional
dino.txt dependencies with:

```bash
uv sync --locked --group dev --extra dinotxt
```

The first model or tokenizer load downloads the required files unless they
are already cached. EmbeddingGemma 2 is pinned to revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`; dino.txt code and weight hashes
are recorded in the [dino.txt reference](docs/dinotxt-v0.6.0.md).

For migration from the v0.4.0 distribution and import namespace, see the
[v0.5.0 migration guide](docs/migration-v0.5.0.md). `openvlclip` is a separate
distribution; installing it does not uninstall `openclip-gemma` or provide an
`openclip_gemma` import alias.

For v0.6.0 model-specific behavior and index compatibility, see the
[v0.6.0 migration guide](docs/migration-v0.6.0.md). For benchmark method and
results, see [the v0.6.0 evaluation report](docs/benchmark-v0.6.0.md).

Batch image embedding, image search, and video-frame search are documented in
the [batch embedding guide](docs/batch-embeddings-v0.2.0.md),
[image search guide](docs/image-search-v0.2.0.md), and
[video search guide](docs/video-search-v0.2.0.md).

## Quick Start

This example creates an image in memory, embeds it and a few candidate descriptions, and computes cosine similarities. It requires no external image file.

```python
from PIL import Image, ImageDraw
import torch

from openvlclip import create_model_and_transforms, get_tokenizer

device = "cpu"
model, _, preprocess = create_model_and_transforms(
    "embeddinggemma-2",
    device=device,
)
tokenizer = get_tokenizer("embeddinggemma-2")

image = Image.new("RGB", (256, 256), "navy")
ImageDraw.Draw(image).ellipse((64, 64, 192, 192), fill="red")
image_tensor = preprocess(image).unsqueeze(0).to(device)

descriptions = [
    "a red circle on a blue background",
    "a green triangle on a white background",
]
text_tokens = tokenizer(descriptions).to(device)

with torch.inference_mode():
    image_features = model.encode_image(image_tensor, normalize=True)
    text_features = model.encode_text(text_tokens, normalize=True)
    cosine_similarities = image_features @ text_features.T

print(image_features.shape)       # torch.Size([1, 768])
print(text_features.shape)        # torch.Size([2, 768])
print(cosine_similarities)
```

The in-memory graphic only demonstrates how to call the API; it is not evidence of the model's image-recognition accuracy. For real use, pass an image relevant to your application.

`create_model_and_transforms()` returns `(model, preprocess_train, preprocess_val)`. This inference-only package uses the same official preprocessing for both transforms. `encode_image()` expects the output of the supplied preprocessing function; do not resize or normalize that tensor a second time.

The examples use CPU by default. To request CUDA, set `device="cuda"` when creating the model and move input tensors to the same device. CUDA inference was tested on the specific GPU listed below; this result does not guarantee support for every NVIDIA GPU or driver combination.

## DINOv2 + dino.txt

The `dinov2-dinotxt` backend uses the pinned DINOv2 ViT-L/14 register-token
backbone and its trained dino.txt heads. Install the `[dinotxt]` extra when
using the published package:

```bash
pip install "openvlclip[dinotxt]==0.6.0"
```

Unlike EmbeddingGemma 2, dino.txt uses no task prompt, does not support
selective loading or MRL truncation, and produces native 2048-dimensional
global image/text embeddings:

```python
from openvlclip import create_model_and_transforms, get_tokenizer

model, _, preprocess = create_model_and_transforms(
    "dinov2-dinotxt",
    device="cpu",
)
tokenizer = get_tokenizer("dinov2-dinotxt")
```

`model.encode_patches()` returns 1024-dimensional patch features;
`model.similarity_map()` compares those features with the text encoder's
patch-aligned 1024-dimensional suffix. Its outputs are cosine scores, not
probabilities or pixel-accurate segmentation masks. See the
[dino.txt reference](docs/dinotxt-v0.6.0.md) and runnable
[similarity-map example](examples/dinotxt_similarity_map.py).

The dino.txt Vision Head and Text Encoder checkpoints are fetched at runtime
from the official Meta-hosted URLs and are not included in the wheel or sdist.
The release is public; this does not establish or expand the license terms for
those external checkpoints. See [License and model terms](#license-and-model-terms)
and the [dino.txt reference](docs/dinotxt-v0.6.0.md) for the documented
sources and remaining license-scope uncertainty.

## Japanese text

EmbeddingGemma 2 supports multilingual text, including Japanese. Use the returned Gemma token IDs rather than CLIP token IDs or raw strings when calling `encode_text()`:

```python
japanese_tokens = tokenizer(["青い背景にある赤い円"]).to(device)

with torch.inference_mode():
    japanese_features = model.encode_text(japanese_tokens, normalize=True)

print(japanese_features.shape)  # torch.Size([1, 768])
```

## Image and text similarity

Both backends can return normalized image and text embeddings with `normalize=True`; compare them with a matrix product as in the Quick Start. A larger cosine similarity indicates a closer match among the supplied candidates, but is not a calibrated probability.

## Matryoshka Representation Learning (MRL)

`encode_image()` and `encode_text()` accept a keyword-only `truncate_dim`
parameter. Supported values are `768`, `512`, `256`, and `128`. The default
`None` keeps the existing 768-dimensional output exactly; lower dimensions take
the leading dimensions of the native embedding and L2-normalize the truncated
vector. Use the same dimension for image and text features before comparing
them:

MRL applies only to EmbeddingGemma 2. The dino.txt backend rejects truncation:
its global embedding space is 2048-dimensional and is not interchangeable with
EmbeddingGemma MRL vectors or 1024-dimensional patch features.

```python
with torch.inference_mode():
    image_features = model.encode_image(image_tensor, truncate_dim=256)
    text_features = model.encode_text(text_tokens, truncate_dim=256)
    cosine_similarities = image_features @ text_features.T
```

For EmbeddingGemma 2, `forward()` and `get_logits()` accept the same
`truncate_dim` when explicitly constructed with
`logit_scale_mode="cosine"`. dino.txt rejects `truncate_dim`; its official
strict-logit path and scale-1 cosine path remain distinct.

EmbeddingGemma model loading can omit unused towers with
`create_model(modalities=...)`:

```python
from openvlclip import create_model

text_model = create_model(modalities=("text",))
image_text_model = create_model(modalities=("text", "image"))
```

`modalities=None` remains the default and preserves the previous full
EmbeddingGemma checkpoint load. Text is required because the text transformer
is shared; image-only and audio inference are not supported. A text-only model
raises an explicit error from `encode_image()`. Factories that return image
transforms require the image modality. dino.txt rejects selective loading and
loads its full vision/text model.

`model.capabilities` is a read-only mapping with `image_embedding`,
`text_embedding`, `image_text_similarity`, `patch_embedding`,
`text_similarity_map`, `mrl`, and `selective_loading` flags. Values depend on
the selected backend and loaded modalities: dino.txt supports patch features
and text similarity maps, while EmbeddingGemma does not; EmbeddingGemma alone
supports MRL and selective loading.

## Zero-shot image classification

For EmbeddingGemma, use its `Classification` prompt with candidate labels,
encode images and descriptions, then rank each image's candidate scores. The
dino.txt counterpart uses its own tokenizer without Gemma prompts; see
[`examples/dinotxt_zero_shot_classification.py`](examples/dinotxt_zero_shot_classification.py).

```python
from PIL import Image, ImageDraw
import torch

from openvlclip import create_model_and_transforms, get_tokenizer

device = "cpu"
model, _, preprocess = create_model_and_transforms(
    "embeddinggemma-2",
    device=device,
)
tokenizer = get_tokenizer("embeddinggemma-2", prompt_name="Classification")

images = []
for color in ("red", "green"):
    image = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(image).rectangle((48, 48, 208, 208), fill=color)
    images.append(image)

image_batch = torch.stack([preprocess(image) for image in images]).to(device)
candidate_tokens = tokenizer(["a red square", "a green square"]).to(device)

with torch.inference_mode():
    image_features = model.encode_image(image_batch, normalize=True)
    candidate_features = model.encode_text(candidate_tokens, normalize=True)
    scores = image_features @ candidate_features.T

print(scores.argmax(dim=-1))  # Index of the highest-scoring candidate per image
```

This example illustrates scoring only; its synthetic images do not establish recognition quality. If you apply softmax to candidate scores for display, those values are relative to the candidate set and are not calibrated probabilities. A runnable example with real-image and demo options is available at [`examples/zero_shot_classification.py`](examples/zero_shot_classification.py).

## Task-specific prompts

Task prompts are part of EmbeddingGemma's text encoding behavior. The
tokenizer accepts `prompt_name`; the default is no task prompt. dino.txt uses
its own official tokenizer without EmbeddingGemma task prompts. For EmbeddingGemma:

```python
search_tokenizer = get_tokenizer("embeddinggemma-2", prompt_name="SearchQuery")
classification_tokenizer = get_tokenizer(
    "embeddinggemma-2",
    prompt_name="Classification",
)
```

Use the prompt appropriate to the role of the text (for example, a search query or a classification label). Prompt names and behavior are defined by the pinned official model/tokenizer configuration; this is not OpenCLIP's text-prompting behavior.

## Image search

Image indexing and text search are provided by the `openvlclip` package.
The CLI demo is available from the repository source (examples are not
included in the wheel). From a source checkout, install the development
environment and run:

```bash
uv sync --locked --group dev
uv run python examples/image_search.py index \
  --images ./photos --output ./image-index-256 --batch-size 8 \
  --embedding-dim 256 --device cpu --recursive
uv run python examples/image_search.py search \
  --index ./image-index-256 --query "富士山と青空" --top-k 5 --device cpu
```

Indexing again updates added, changed, and deleted files. Change `--device cpu` to `--device cuda` to request CUDA inference. Lower `--batch-size` if GPU memory is limited. EmbeddingGemma search uses its `SearchQuery` task prompt; dino.txt uses no such prompt. Both report unscaled cosine scores, not probabilities. Results with equal scores are ordered by relative image path.

To index with dino.txt from the source tree, select the registered model and
its native dimension:

```bash
uv run python examples/image_search.py index \
  --model dinov2-dinotxt --images ./photos --output ./dino-index \
  --batch-size 1 --embedding-dim 2048 --device cpu
uv run python examples/image_search.py search \
  --model dinov2-dinotxt --index ./dino-index \
  --query "a red car" --top-k 5 --device cpu
```

Image indexes record model, revision, preprocessing, prompt, normalization,
and embedding-strategy identity. A search or conversion across incompatible
embedding spaces is rejected; dino.txt, EmbeddingGemma MRL, and patch-feature
vectors cannot be mixed.

The local index is a versioned NumPy `.npz` archive containing JSON metadata and float32 embeddings; it is read with pickle loading disabled and replaced atomically after a successful update. Image and video index archives are limited to 512 MiB each; split larger collections into multiple indexes. File changes are detected using relative path, size, and modification time in nanoseconds. Touching or rewriting an unchanged image can trigger an unnecessary re-embedding; rewriting a file while preserving both size and timestamp can evade detection. Remove the index and rebuild it if a full refresh is needed. A damaged image or incompatible/corrupt index stops the operation with an error rather than silently skipping data. See the [image search design and storage notes](docs/image-search-v0.2.0.md).

The Python API is available from `openvlclip.batching` and
`openvlclip.image_search`:

```python
from openvlclip import create_model_and_transforms
from openvlclip.batching import iter_image_embeddings
from openvlclip.image_search import build_or_update_image_index, search_images

model, _, preprocess = create_model_and_transforms(device="cpu")
for path, embedding in iter_image_embeddings(model, preprocess, ["photos/a.jpg"], batch_size=8):
    print(path, embedding.shape, embedding.device)

build_or_update_image_index(
    model,
    preprocess,
    "photos",
    "image-index-256",
    batch_size=8,
    embedding_dim=256,
)
for result in search_images(model, "image-index-256", "青空の山", top_k=5):
    print(result.path, f"{result.score:.4f}")
```

The streaming iterator yields CPU tensors in input order. Image-search results
contain raw cosine similarity scores, not probabilities.
Image indexes use format v2 and store one supported embedding dimension.
Existing v1 indexes remain readable at 768 dimensions. An incremental update
cannot mix dimensions; use a separate index directory or convert an existing
768-dimensional index with `convert_image_index()`. Conversion truncates and
renormalizes stored vectors without rerunning the model.

## Video frame search (optional `video` extra)

Video indexing and search sample frames and match them to Japanese or English
text. The optional PyAV dependency is installed with
`pip install "openvlclip[video]"`. From a source checkout:

```bash
uv sync --locked --group dev --extra video
uv run python examples/video_search.py index \
  --videos ./videos --output ./video-index --interval 1.0 \
  --batch-size 8 --embedding-dim 256 --device cpu --recursive
uv run python examples/video_search.py search \
  --index ./video-index --query "人がドリルで穴を開けている" \
  --top-k 5 --device cpu
```

MP4 is the primary target; `.mp4`, `.m4v`, `.mov`, `.mkv`, `.webm`, and `.avi` paths are considered. Actual container and codec support depends on the FFmpeg build bundled with PyAV. The decoder samples the first frame whose presentation timestamp reaches each interval target and records its PTS-derived time; the requested interval is a target, not a promise that a frame exists at that exact time. Video is decoded sequentially, while only a bounded frame batch is sent through the image encoder. Reduce `--batch-size` if GPU memory is limited.

Search ranks individual sampled frames by cosine similarity. Timestamps are relative to the container playback start, or the video stream start when the container start is unavailable; if neither is present, the first decoded frame defines zero. It does not understand motion, event order, or temporal relationships, and a scene between sampled timestamps can be missed. The local `video-index.npz` archive stores frame embeddings and timing metadata separately from image indexes, uses pickle-disabled NumPy loading, and is atomically replaced. Video additions, changes, and deletions are reconciled by relative path, file size, and modification time in nanoseconds; preserving both size and timestamp can evade change detection. See the [video search design notes](docs/video-search-v0.2.0.md) for storage, sampling, validation, and decoder limitations.

The API is available in `openvlclip.video_search`:

```python
from openvlclip import create_model_and_transforms
from openvlclip.video_search import build_or_update_video_index, search_video

model, _, preprocess = create_model_and_transforms(device="cpu")
build_or_update_video_index(
    model, preprocess, "videos", "video-index", interval_seconds=1.0, batch_size=8
)
for match in search_video(model, "video-index", "人がドリルで穴を開けている", top_k=5):
    print(match.video_path, match.timestamp_seconds, match.score)
```

The documented video workflow uses the EmbeddingGemma backend; dino.txt video
support is not claimed. Video search ranks individual sampled frames. It does not understand motion,
event order, or temporal relationships, and scenes between sampled timestamps
can be missed. Search loads the full index into RAM and uses brute-force cosine
scoring, so memory and latency grow with collection size.
Video indexes use format v3 and support EmbeddingGemma's MRL dimensions. Existing
version-2 indexes remain readable at 768 dimensions; version-1 timestamp
indexes are still unsupported. Use `convert_video_index()` to create a
separate, smaller-dimensional index without model inference.

## OpenCLIP differences and compatibility

`openvlclip` offers a **partial, OpenCLIP-inspired inference API**, not an implementation of OpenCLIP. The registry currently contains `embeddinggemma-2` and `dinov2-dinotxt`, with model-specific capabilities and embedding spaces.

| Area | `openvlclip` behavior |
| --- | --- |
| Embeddings | EmbeddingGemma 2 produces 768 dimensions with MRL outputs of 512, 256, or 128. dino.txt produces 2048-dimensional global embeddings and 1024-dimensional patch features; truncation is rejected. |
| Images | Use the selected model's returned preprocessing transform. `encode_image()` accepts preprocessed float32 `[B, 3, H, W]` tensors; do not apply preprocessing twice. |
| Text | `get_tokenizer()` returns model-specific token IDs. CLIP, Gemma, and dino.txt token IDs are not interchangeable. |
| Similarity | EmbeddingGemma cosine mode returns scale-1 similarities. dino.txt strict mode preserves the official scaled logits; cosine mode is distinct. Neither is a calibrated probability. |
| Factory discovery | `list_models()`, `list_pretrained()`, and `create_model_from_pretrained()` resolve the registered backends and their checkpoints. |
| Selective loading | Selective loading is supported by EmbeddingGemma 2; dino.txt requires its complete vision/text model. |
| Capabilities | `model.capabilities` is immutable and model-specific. dino.txt exposes patch embeddings and text similarity maps; EmbeddingGemma reports these as unsupported. |
| Model methods | Both backends support global image/text encoding, `eval()`, and `to()`. `forward()`/`get_logits()` semantics and defaults are backend-specific; consult the model reference before choosing strict or cosine mode. |
| Scope | The core model API does not support training, fine-tuning, audio, or arbitrary architectures. Video-frame search is a separate v0.2.0 API, not a core OpenCLIP-compatible model feature. |

`list_pretrained()` reports the registered checkpoints for the selected
backend; `official` is a package alias, not an OpenCLIP pretrained tag.
EmbeddingGemma cosine mode returns normalized features and scale `1.0`.
dino.txt strict mode uses the Hub-initialized official logit scale; cosine
mode uses scale `1.0`. Neither reproduces a trained OpenCLIP `logit_scale`.

For scenario-by-scenario results and migration examples, see the
[OpenCLIP compatibility report](docs/openclip-compatibility-report-v0.3.0.md)
and [v0.3.0 migration guide](docs/migration-v0.3.0.md). For MRL index and factory changes, see the [v0.4.0 migration guide](docs/migration-v0.4.0.md)
and [MRL benchmark guide](docs/benchmark-mrl-v0.4.0.md).

### Migration example

The factory and encoder calls are intentionally familiar, but use the EmbeddingGemma-specific model name, preprocessing transform, and tokenizer:

```python
# EmbeddingGemma-specific model, processor, and retrieval prompt
model, _, preprocess = create_model_and_transforms(
    "embeddinggemma-2",
    pretrained="official",
    device="cpu",
    logit_scale_mode="cosine",
)
tokenizer = get_tokenizer("embeddinggemma-2", prompt_name="SearchQuery")

image_features = model.encode_image(preprocess(image).unsqueeze(0), normalize=True)
text_features = model.encode_text(tokenizer(["a description"]), normalize=True)
similarity = image_features @ text_features.T
```

This is not a drop-in replacement: an OpenCLIP tokenizer, checkpoint, or
arbitrary OpenCLIP model name cannot be substituted. The cosine mode is
explicit because its output is not numerically equivalent to trained CLIP
logits.

## Supported environments and limitations

| Environment | Status |
| --- | --- |
| Windows 11 x64, Python 3.12, 3.13, and 3.14 (except 3.14.1), CPU FP32/BF16 | Real-model image and text inference tested. Python 3.14 testing used 3.14.8. |
| Windows and Ubuntu CI, Python 3.12, 3.13, and 3.14 | Unit tests, lint, build, and wheel checks passed. Ubuntu CI does not test real-model inference. |
| Ubuntu/Linux real-model inference | Not verified. |
| CUDA real-model inference | Tested on an NVIDIA GeForce RTX 4070 Ti SUPER (16 GiB), driver 610.88, with Python 3.12.10, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, and CUDA runtime 13.0. FP32 and BF16 image and text inference passed. Other GPUs and driver/runtime combinations are not guaranteed. |
| v0.2.0rc1 TestPyPI validation (2026-10-18), CPU | Clean wheel install on Windows 11, Python 3.12.10, PyTorch 2.14.1+cpu, and PyAV 19.0.1. FP32 image/text embeddings, batch embedding, image index/search, and synthetic-MP4 indexing/search passed. |
| v0.2.0rc1 TestPyPI validation (2026-10-18), CUDA | Clean wheel install on the RTX 4070 Ti SUPER above, with PyTorch 2.14.1+cu130 and CUDA runtime 13.0. FP32 embeddings, batch embedding, image index/search, and synthetic-MP4 indexing/search passed. This is not a real-world retrieval-quality evaluation. |
| macOS | Not verified. |

### CUDA verification results

The CUDA integration tests completed with **2 passed, 0 failed, 0 skipped** on the GPU listed above. They compared CPU and CUDA results using the same precision and fixed model revision (`914f7f89142e33e77833254d9c9b90c3cef7303b`). Both modalities returned finite 768-dimensional embeddings.

| Precision | Image max absolute error vs CPU | Text max absolute error vs CPU | Image cosine similarity | Text cosine similarity |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 9.45e-8 | 6.52e-8 | 1.000000 | 1.000000 |
| BF16 | 0.001953125 | 0.001953125 | 0.999968 | 0.999975 |

| Precision | Synchronized model load | Image latency median | Text batch (2) latency median | PyTorch peak allocated |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 4.43 s | 170.1 ms | 53.1 ms | 2,992 MiB |
| BF16 | 4.16 s | 124.3 ms | 62.8 ms | 1,522 MiB |

Latency measurements used three warm-up runs followed by ten measured runs per modality, with CUDA synchronization around each timed call. Model files were already cached, so model-load times do not include downloading. Peak allocated memory is PyTorch's measurement for the model and inference workload, not total system VRAM use. These timings are measurements from one Windows development PC, not performance guarantees.

Additional input and numerical constraints:

- Image tensors must come from the model's returned preprocessing transform and have shape `[B, 3, H, W]`. Pass RGB PIL images to the transform; it performs the required conversion and processing. Do not resize, rescale, or normalize the resulting tensor again.
- Every image in one tensor batch must have the same processed dimensions so that the tensors can be stacked.
- Tokenizer output is dynamically padded for each batch, rather than fixed to OpenCLIP's usual context length. The pinned model configuration supports up to 8192 tokens; long inputs and batches can use substantial memory. Pass the tokenizer's output directly to `encode_text()`; do not substitute CLIP token IDs.
- `encode_image()` and `encode_text()` return normalized embeddings by default. `normalize=True` is available to explicitly normalize the returned features. FP16 is not a supported precision; FP32 is the reference, and BF16 has been separately tested on the stated Windows CPU environment.
- `pretrained=None` is unsupported and raises an error. The library uses the pinned EmbeddingGemma 2 revision; it does not load arbitrary OpenCLIP checkpoints.

## Development and tests

Clone the repository and install the development dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --group dev
uv run pytest
uv run ruff check .
uv build
```

The standard CI does not download model weights. Real-model integration tests require the `integration` extra and access to the pinned Hugging Face model files (or a populated local cache):

```powershell
uv sync --locked --group dev --extra integration
$env:OPENCLIP_GEMMA_RUN_INTEGRATION = "1"
uv run pytest tests/integration -v
```

For a CUDA test environment on Windows, create a separate environment so the existing environment is left untouched. The PyTorch and torchvision versions below were used for the RTX 4070 Ti SUPER verification. Install the CUDA wheels from PyTorch's official CUDA 13.0 index; check [PyTorch's installation selector](https://pytorch.org/get-started/locally/) for other supported configurations.

```powershell
uv venv --python 3.12 .venv-cuda
uv pip install --python .venv-cuda\Scripts\python.exe -e ".[test,integration]"
uv pip install --python .venv-cuda\Scripts\python.exe --reinstall --no-deps `
  --index-url https://download.pytorch.org/whl/cu130 `
  torch==2.14.1 torchvision==0.29.1

.venv-cuda\Scripts\python.exe -c "import torch; cuda=torch.cuda.is_available(); print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda, 'CUDA available:', cuda, 'BF16 supported:', torch.cuda.is_bf16_supported() if cuda else False)"
$env:OPENCLIP_GEMMA_RUN_INTEGRATION = "1"
.venv-cuda\Scripts\python.exe -m pytest tests/integration/test_cuda_integration.py -v -s
```

The test skips CUDA checks when CUDA is unavailable; verify the probe reports `torch.cuda.is_available() == True` before treating a skip as a hardware validation. This CUDA result does not establish Linux real-model inference support. Linux and macOS real-model inference remain unverified.

## License and model terms

The OpenVLCLIP library is distributed under the [Apache License 2.0](LICENSE).
It does not include model weights or OpenCLIP implementation code. OpenCLIP
is distributed under the MIT License; EmbeddingGemma 2 weights are governed
by the terms and license stated by Google for the pinned model revision on its
[Hugging Face model card](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md).

Meta's pinned DINOv2 repository [LICENSE](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE)
is Apache-2.0, and its
[model card](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/MODEL_CARD.md)
lists Apache License 2.0 for the DINOv2 ViT model family. The
[official dino.txt README](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md)
links the [Vision Head](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth)
and [Text Encoder](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth)
checkpoints. Their individual license terms are not separately identified in
the pinned repository. Do not infer that the repository/model-card license
conclusively covers these distinct dino.txt checkpoints; review their terms
before redistributing them. The v0.6.0 publication status is separate from
this unresolved license-scope question.

The official README also links the
[BPE vocabulary](https://dl.fbaipublicfiles.com/dinov2/thirdparty/bpe_simple_vocab_16e6.txt.gz)
and its separate [vocabulary license](https://dl.fbaipublicfiles.com/dinov2/thirdparty/LICENSE),
which is the MIT License (OpenAI, 2021). The library license does not replace
the separate terms for downloaded weights or third-party dependencies.

## About the name

The name OpenVLCLIP reflects the goal of providing familiar interfaces across vision and vision-language foundation models. The v0.5.0 PyPI distribution supports EmbeddingGemma 2 only; v0.6.0 is the first published release with the dino.txt backend. The project does not claim OpenCLIP affiliation or full OpenCLIP compatibility.

This is an independently developed library, not an official extension of OpenCLIP, Google, or Meta.

The names OpenCLIP and Gemma refer to their respective projects and technologies. No affiliation or endorsement is implied.
