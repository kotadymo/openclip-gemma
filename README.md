# openclip-gemma

An independent Python library that provides an OpenCLIP-inspired inference API for Google's EmbeddingGemma 2 model. It lets Python applications use familiar image and text encoding calls while retaining EmbeddingGemma's own tokenizer, image processing, task prompts, and embedding behavior.

> [!IMPORTANT]
> **Unofficial project**
>
> `openclip-gemma` is an independent, unofficial project. It is not affiliated with, endorsed by, sponsored by, or officially maintained by Google, Google DeepMind, or the OpenCLIP project.
>
> This library provides an OpenCLIP-inspired interface for Google's EmbeddingGemma 2 model. It is not a drop-in replacement for OpenCLIP and does not claim full API compatibility.

## Release status

Version **0.1.1** is available on [PyPI](https://pypi.org/project/openclip-gemma/0.1.1/) and as a [GitHub release](https://github.com/kotadymo/openclip-gemma/releases/tag/v0.1.1).

```bash
pip install openclip-gemma
```

For a reproducible environment, pin the version:

```bash
pip install openclip-gemma==0.1.1
```

## Features

- OpenCLIP-style `create_model`, `create_model_and_transforms`, and `get_tokenizer` entry points.
- Image and text embedding through the official Sentence Transformers implementation of EmbeddingGemma 2.
- 768-dimensional embeddings, task-specific text prompts, and CPU or CUDA device selection.
- A pinned EmbeddingGemma 2 model revision; model weights are downloaded from Hugging Face on first use and are not bundled with this package.

## Installation

Python 3.12 and later are supported, up to but not including Python 3.15, except Python 3.14.1 (`>=3.12,<3.15,!=3.14.1`). Install into your active virtual environment using the command above.

The first model or tokenizer load downloads the required files from Hugging Face unless they are already cached. The package pins the model revision `914f7f89142e33e77833254d9c9b90c3cef7303b`.

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

The examples use CPU by default. To request CUDA, set `device="cuda"` when creating the model and move input tensors to the same device. This requires a compatible CUDA-enabled PyTorch installation and GPU; see the environment table below because physical GPU inference has not yet been verified for this release.

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

## OpenCLIP differences and compatibility

`openclip-gemma` offers a **partial, OpenCLIP-inspired inference API**, not an implementation of OpenCLIP. Only the `embeddinggemma-2` model is supported.

| Area | `openclip-gemma` behavior |
| --- | --- |
| Embeddings | Image and text outputs have 768 features and follow EmbeddingGemma 2's embedding behavior. |
| Images | Use the returned official processor transform. `encode_image()` accepts preprocessed float32 `[B, 3, H, W]` tensors; do not feed arbitrary OpenCLIP-normalized tensors or apply preprocessing twice. All images in a dense batch must have the same shape. |
| Text | `get_tokenizer()` returns Gemma token IDs with dynamically right-padded batches. CLIP token IDs and OpenCLIP's tokenizer are not interchangeable. |
| Similarity | Normalize embeddings and compute cosine similarity. The model does not provide an OpenCLIP-trained `logit_scale`. |
| Model methods | `encode_image()`, `encode_text()`, `eval()`, and `to()` are supported. `forward()` and `get_logits()` are not implemented. |
| Scope | Training, fine-tuning, audio, video, and arbitrary OpenCLIP architectures are not supported. |

For migration details and the tested compatibility surface, see the [OpenCLIP compatibility report](docs/openclip-compatibility-report.md).

### Migration example

The factory and encoder calls are intentionally familiar, but use the EmbeddingGemma-specific model name, preprocessing transform, and tokenizer:

```python
# OpenCLIP-style calls
model, _, preprocess = create_model_and_transforms("embeddinggemma-2", device="cpu")
tokenizer = get_tokenizer("embeddinggemma-2")

image_features = model.encode_image(preprocess(image).unsqueeze(0), normalize=True)
text_features = model.encode_text(tokenizer(["a description"]), normalize=True)
similarity = image_features @ text_features.T
```

This is not a drop-in replacement: an OpenCLIP tokenizer, checkpoint, or arbitrary OpenCLIP model name cannot be substituted.

## Supported environments and limitations

| Environment | Status |
| --- | --- |
| Windows 11 x64, Python 3.12, 3.13, and 3.14 (except 3.14.1), CPU FP32/BF16 | Real-model image and text inference tested. Python 3.14 testing used 3.14.8. |
| Windows and Ubuntu CI, Python 3.12, 3.13, and 3.14 | Unit tests, lint, build, and wheel checks passed. Ubuntu CI does not test real-model inference. |
| Ubuntu/Linux real-model inference | Not verified. |
| CUDA hardware inference | Not verified on a GPU. The implementation accepts CUDA devices when a compatible CUDA-enabled PyTorch installation and GPU are available. |
| macOS | Not verified. |

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

The implementation has CUDA test coverage that skips when CUDA is unavailable, but CUDA inference has not yet been verified on physical GPU hardware. Use a CUDA-enabled PyTorch/torchvision build compatible with your driver and GPU, then run the integration suite on that GPU; a CPU-only PyTorch installation cannot validate CUDA behavior. Linux CI success should not be interpreted as Linux real-model inference validation.

## License and model terms

The `openclip-gemma` library is distributed under the [Apache License 2.0](LICENSE). It does not include EmbeddingGemma 2 model weights or OpenCLIP implementation code. OpenCLIP is distributed under the MIT License; EmbeddingGemma 2 weights are governed by the terms and license stated by Google for the pinned model revision on its [Hugging Face model card](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md).

The library license does not replace or alter the separate terms that apply to downloaded model weights or third-party dependencies. Review those terms before use.

## About the name

The name `openclip-gemma` reflects the purpose of this project: providing an OpenCLIP-inspired interface for Google's EmbeddingGemma 2 model.

This is an independently developed library, not an official extension of OpenCLIP or a Google-maintained package.

The names OpenCLIP and Gemma refer to their respective projects and technologies. No affiliation or endorsement is implied.
