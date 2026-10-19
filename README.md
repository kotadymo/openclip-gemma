# openclip-gemma

An independent Python library that provides an OpenCLIP-inspired inference API for Google's EmbeddingGemma 2 model. It lets Python applications use familiar image and text encoding calls while retaining EmbeddingGemma's own tokenizer, image processing, task prompts, and embedding behavior.

> [!IMPORTANT]
> **Unofficial project**
>
> `openclip-gemma` is an independent, unofficial project. It is not affiliated with, endorsed by, sponsored by, or officially maintained by Google, Google DeepMind, or the OpenCLIP project.
>
> This library provides an OpenCLIP-inspired interface for Google's EmbeddingGemma 2 model. It is not a drop-in replacement for OpenCLIP and does not claim full API compatibility.

## Features

- OpenCLIP-style `create_model`, `create_model_and_transforms`, and `get_tokenizer` entry points.
- Factory discovery with `list_models()`, `list_pretrained()`, and `create_model_from_pretrained()`.
- Only EmbeddingGemma 2 is supported; this is not a complete drop-in replacement for OpenCLIP.
- Image and text inference use the official Transformers model and processor; Sentence Transformers is the pinned numerical reference.
- Model API includes `forward()`, `get_logits()`, and factory option `output_dict=True`.
- `forward()` and `get_logits()` require explicit `logit_scale_mode="cosine"` (default: `"strict"`).
- `forward()` returns normalized features and scale `1.0`; `get_logits()` returns unscaled cosine similarities, not trained CLIP logits or probabilities.
- 768-dimensional embeddings, task-specific text prompts, and CPU or CUDA device selection.
- A pinned EmbeddingGemma 2 model revision; model weights are downloaded from Hugging Face on first use and are not bundled with this package.
- Bounded-batch image embedding and incremental Japanese/English text-to-image search.
- Optional PyAV-based video-frame indexing and text search.

## Installation

Install the package into an active virtual environment. Python 3.12 and later
are supported, up to but not including Python 3.15, except Python 3.14.1
(`>=3.12,<3.15,!=3.14.1`).

```bash
pip install openclip-gemma
```

Install the optional video dependency for video-frame indexing and search:

```bash
pip install "openclip-gemma[video]"
```

To pin the exact version for reproducible environments:

```bash
pip install openclip-gemma==0.3.0
```

The first model or tokenizer load downloads the required files from Hugging Face unless they are already cached. The package pins the model revision `914f7f89142e33e77833254d9c9b90c3cef7303b`.

Batch image embedding, image search, and video-frame search are documented in
the [batch embedding guide](docs/batch-embeddings-v0.2.0.md),
[image search guide](docs/image-search-v0.2.0.md), and
[video search guide](docs/video-search-v0.2.0.md).

## Quick Start

This example creates an image in memory, embeds it and a few candidate descriptions, and computes cosine similarities. It requires no external image file.

```python
from PIL import Image, ImageDraw
import torch

from openclip_gemma import create_model_and_transforms, get_tokenizer

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

## Japanese text

EmbeddingGemma 2 supports multilingual text, including Japanese. Use the returned Gemma token IDs rather than CLIP token IDs or raw strings when calling `encode_text()`:

```python
japanese_tokens = tokenizer(["青い背景にある赤い円"]).to(device)

with torch.inference_mode():
    japanese_features = model.encode_text(japanese_tokens, normalize=True)

print(japanese_features.shape)  # torch.Size([1, 768])
```

## Image and text similarity

The model returns normalized sentence embeddings. You can also request normalization explicitly with `normalize=True`; normalized image and text features can be compared with a matrix product, as in the Quick Start. A larger cosine similarity indicates a closer match among the supplied candidates, but is not a calibrated probability.

## Zero-shot image classification

For a set of candidate labels, encode images and candidate descriptions, then rank each image's candidate scores:

```python
from PIL import Image, ImageDraw
import torch

from openclip_gemma import create_model_and_transforms, get_tokenizer

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

Task prompts are part of EmbeddingGemma's text encoding behavior. The tokenizer accepts `prompt_name`; the default is no task prompt. For example:

```python
search_tokenizer = get_tokenizer("embeddinggemma-2", prompt_name="SearchQuery")
classification_tokenizer = get_tokenizer(
    "embeddinggemma-2",
    prompt_name="Classification",
)
```

Use the prompt appropriate to the role of the text (for example, a search query or a classification label). Prompt names and behavior are defined by the pinned official model/tokenizer configuration; this is not OpenCLIP's text-prompting behavior.

## Image search

Image indexing and text search are provided by the `openclip_gemma` package.
The CLI demo is available from the repository source (examples are not
included in the wheel). From a source checkout, install the development
environment and run:

```bash
uv sync --locked --group dev
uv run python examples/image_search.py index \
  --images ./photos --output ./image-index --batch-size 8 --device cpu --recursive
uv run python examples/image_search.py search \
  --index ./image-index --query "富士山と青空" --top-k 5 --device cpu
```

Indexing again updates added, changed, and deleted files. Change `--device cpu` to `--device cuda` to request CUDA inference. Lower `--batch-size` if GPU memory is limited. Search uses EmbeddingGemma's `SearchQuery` task prompt and reports unscaled cosine similarity scores, not probabilities. Results with equal scores are ordered by relative image path.

The local index is a versioned NumPy `.npz` archive containing JSON metadata and float32 embeddings; it is read with pickle loading disabled and replaced atomically after a successful update. Image and video index archives are limited to 512 MiB each; split larger collections into multiple indexes. File changes are detected using relative path, size, and modification time in nanoseconds. Touching or rewriting an unchanged image can trigger an unnecessary re-embedding; rewriting a file while preserving both size and timestamp can evade detection. Remove the index and rebuild it if a full refresh is needed. A damaged image or incompatible/corrupt index stops the operation with an error rather than silently skipping data. See the [image search design and storage notes](docs/image-search-v0.2.0.md).

The Python API is available from `openclip_gemma.batching` and
`openclip_gemma.image_search`:

```python
from openclip_gemma import create_model_and_transforms
from openclip_gemma.batching import iter_image_embeddings
from openclip_gemma.image_search import build_or_update_image_index, search_images

model, _, preprocess = create_model_and_transforms(device="cpu")
for path, embedding in iter_image_embeddings(model, preprocess, ["photos/a.jpg"], batch_size=8):
    print(path, embedding.shape, embedding.device)

build_or_update_image_index(model, preprocess, "photos", "image-index", batch_size=8)
for result in search_images(model, "image-index", "青空の山", top_k=5):
    print(result.path, f"{result.score:.4f}")
```

The streaming iterator yields CPU tensors in input order. Image-search results
contain raw cosine similarity scores, not probabilities.

## Video frame search (optional `video` extra)

Video indexing and search sample frames and match them to Japanese or English
text. The optional PyAV dependency is installed with
`pip install "openclip-gemma[video]"`. From a source checkout:

```bash
uv sync --locked --group dev --extra video
uv run python examples/video_search.py index \
  --videos ./videos --output ./video-index --interval 1.0 \
  --batch-size 8 --device cpu --recursive
uv run python examples/video_search.py search \
  --index ./video-index --query "人がドリルで穴を開けている" \
  --top-k 5 --device cpu
```

MP4 is the primary target; `.mp4`, `.m4v`, `.mov`, `.mkv`, `.webm`, and `.avi` paths are considered. Actual container and codec support depends on the FFmpeg build bundled with PyAV. The decoder samples the first frame whose presentation timestamp reaches each interval target and records its PTS-derived time; the requested interval is a target, not a promise that a frame exists at that exact time. Video is decoded sequentially, while only a bounded frame batch is sent through the image encoder. Reduce `--batch-size` if GPU memory is limited.

Search ranks individual sampled frames by cosine similarity. Timestamps are relative to the container playback start, or the video stream start when the container start is unavailable; if neither is present, the first decoded frame defines zero. It does not understand motion, event order, or temporal relationships, and a scene between sampled timestamps can be missed. The local `video-index.npz` archive stores frame embeddings and timing metadata separately from image indexes, uses pickle-disabled NumPy loading, and is atomically replaced. Video additions, changes, and deletions are reconciled by relative path, file size, and modification time in nanoseconds; preserving both size and timestamp can evade change detection. See the [video search design notes](docs/video-search-v0.2.0.md) for storage, sampling, validation, and decoder limitations.

The API is available in `openclip_gemma.video_search`:

```python
from openclip_gemma import create_model_and_transforms
from openclip_gemma.video_search import build_or_update_video_index, search_video

model, _, preprocess = create_model_and_transforms(device="cpu")
build_or_update_video_index(
    model, preprocess, "videos", "video-index", interval_seconds=1.0, batch_size=8
)
for match in search_video(model, "video-index", "人がドリルで穴を開けている", top_k=5):
    print(match.video_path, match.timestamp_seconds, match.score)
```

Video search ranks individual sampled frames. It does not understand motion,
event order, or temporal relationships, and scenes between sampled timestamps
can be missed. Search loads the full index into RAM and uses brute-force cosine
scoring, so memory and latency grow with collection size.

## OpenCLIP differences and compatibility

`openclip-gemma` offers a **partial, OpenCLIP-inspired inference API**, not an implementation of OpenCLIP. Only the `embeddinggemma-2` model is supported.

| Area | `openclip-gemma` behavior |
| --- | --- |
| Embeddings | Image and text outputs have 768 features and follow EmbeddingGemma 2's embedding behavior. |
| Images | Use the returned official processor transform. `encode_image()` accepts preprocessed float32 `[B, 3, H, W]` tensors; do not feed arbitrary OpenCLIP-normalized tensors or apply preprocessing twice. All images in a dense batch must have the same shape. |
| Text | `get_tokenizer()` returns Gemma token IDs with dynamically right-padded batches. CLIP token IDs and OpenCLIP's tokenizer are not interchangeable. |
| Similarity | Normalize embeddings and compute cosine similarity. The model does not provide an OpenCLIP-trained `logit_scale`. |
| Factory discovery | `list_models()`, `list_pretrained()`, and `create_model_from_pretrained()` expose only the supported EmbeddingGemma 2 checkpoint. |
| Model methods | `encode_image()`, `encode_text()`, `eval()`, and `to()` are supported. `forward()` and `get_logits()` require explicit `logit_scale_mode="cosine"`; the default strict mode rejects them. |
| Scope | The core model API does not support training, fine-tuning, audio, or arbitrary architectures. Video-frame search is a separate v0.2.0 API, not a core OpenCLIP-compatible model feature. |

`list_pretrained()` reports `("embeddinggemma-2", "official")`; `official` is
this package's alias for its pinned Google checkpoint, not an OpenCLIP
pretrained tag. `forward()` in cosine mode returns normalized features and
scalar `1.0`, not CLIP logits. `get_logits()` returns unscaled cosine
similarities. Neither API reproduces a trained OpenCLIP `logit_scale`.

For scenario-by-scenario results and migration examples, see the
[OpenCLIP compatibility report](docs/openclip-compatibility-report-v0.3.0.md)
and [migration guide](docs/migration-v0.3.0.md).

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

The `openclip-gemma` library is distributed under the [Apache License 2.0](LICENSE). It does not include EmbeddingGemma 2 model weights or OpenCLIP implementation code. OpenCLIP is distributed under the MIT License; EmbeddingGemma 2 weights are governed by the terms and license stated by Google for the pinned model revision on its [Hugging Face model card](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md).

The library license does not replace or alter the separate terms that apply to downloaded model weights or third-party dependencies. Review those terms before use.

## About the name

The name `openclip-gemma` reflects the purpose of this project: providing an OpenCLIP-inspired interface for Google's EmbeddingGemma 2 model.

This is an independently developed library, not an official extension of OpenCLIP or a Google-maintained package.

The names OpenCLIP and Gemma refer to their respective projects and technologies. No affiliation or endorsement is implied.
