# openclip-gemma

An independent Python inference library that exposes a small OpenCLIP-style API for Google's EmbeddingGemma 2. It lets compatible image-and-text embedding workflows use EmbeddingGemma 2 while keeping the model's official preprocessing, tokenizer, prompts, pooling, and normalization.

> [!IMPORTANT]
> **Unofficial Project**
>
> `openclip-gemma` is an independent, unofficial project.
> It is not affiliated with, endorsed by, sponsored by,
> or officially maintained by Google, Google DeepMind,
> or the OpenCLIP project.
>
> This library provides an OpenCLIP-inspired interface
> for Google's EmbeddingGemma 2 model.
> It is not a drop-in replacement for OpenCLIP and
> does not claim full API compatibility.

## Release status

Version 0.1.0 is available on PyPI.

```bash
pip install openclip-gemma==0.1.0
```

The initial release has been validated on Windows 11 x64
with Python 3.12 and CPU inference.

CUDA, Linux, and macOS have not yet been validated.

## Features

- `create_model()` and `create_model_and_transforms()` for the pinned `google/embeddinggemma-2` checkpoint.
- OpenCLIP-style `encode_image()` and `encode_text()` methods returning 768-dimensional PyTorch embeddings.
- Callable `get_tokenizer()` returning `torch.LongTensor[B, L]`.
- Official EmbeddingGemma task prompts, selected when creating the tokenizer.
- Image and text batches, CPU inference, FP32 baseline, and separately measured BF16.
- A zero-shot example that computes cosine similarities without inventing an OpenCLIP `logit_scale`.

The v0.1.0 wrapper supports text and still-image embeddings only. It does not implement training, fine-tuning, audio, video, `forward()`, or `get_logits()`.

## Compatibility and tested environment

| Item | v0.1.0 status |
|---|---|
| Python | Python 3.12 |
| OS | Windows 11 x64 tested locally with Python 3.12. Linux and macOS are unverified. |
| Device | CPU FP32 and BF16 tested. CUDA is implemented but has not been validated on CUDA hardware. |
| Precision | FP32 is the numerical baseline. BF16 was measured separately on CPU. FP16 is rejected. |
| Model | `google/embeddinggemma-2`, pinned revision `914f7f89142e33e77833254d9c9b90c3cef7303b` |
| Context | Up to 8192 tokens, following the model card. Batches use dynamic right padding; this is not OpenCLIP's fixed-width 77-token padding. |

The environment used for the CPU reference checks was Python 3.12.10, PyTorch 2.14.1, torchvision 0.29.1, Transformers 5.20.1, Pillow 13.0.0, and Sentence Transformers 6.1.0. Development validation used uv 0.13.1, pytest 9.1.1, and Ruff 0.17.1; the optional CPU benchmark used psutil 7.2.2. Sentence Transformers and psutil are optional and are not needed by the runtime API. Other dependency versions allowed by the package metadata have not all been tested.

## Installation

Install the stable v0.1.0 release from PyPI:

```bash
python -m pip install openclip-gemma==0.1.0
```

To install directly from the GitHub repository:

```bash
python -m pip install "git+https://github.com/kotadymo/openclip-gemma.git"
```

To work from a local checkout:

```bash
git clone https://github.com/kotadymo/openclip-gemma.git
cd openclip-gemma
uv sync --group dev
```

The package does not contain model weights. The first `create_model()` call fetches the pinned model checkpoint and processor from Hugging Face; `get_tokenizer()` fetches the pinned configuration and tokenizer. Files are stored in the Hugging Face cache. A network connection is needed unless the required files are already cached. A custom model cache directory can be passed as `cache_dir` to the factory and tokenizer.

For CUDA, install a PyTorch/torchvision build appropriate for the local driver using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then confirm `torch.cuda.is_available()` before selecting `device="cuda"`. CUDA was unavailable in the v0.1.0 validation environment and is not claimed as verified.

## Basic image and text embeddings

```python
import torch
from PIL import Image

import openclip_gemma

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = openclip_gemma.create_model_and_transforms(
    "embeddinggemma-2",
    device=device,
)
model.eval()

# Replace this generated image with any RGB or grayscale PIL image.
image = Image.new("RGB", (256, 192), color=(80, 120, 180))
image_tensor = preprocess(image).unsqueeze(0).to(device)

tokenizer = openclip_gemma.get_tokenizer("embeddinggemma-2")
text_tokens = tokenizer(["a blue image", "a red image"]).to(device)

with torch.inference_mode():
    image_features = model.encode_image(image_tensor, normalize=True)
    text_features = model.encode_text(text_tokens, normalize=True)

print(image_features.shape)  # torch.Size([1, 768])
print(text_features.shape)   # torch.Size([2, 768])
```

The image tensor must come from the returned EmbeddingGemma preprocessing function. It must be `[B, 3, H, W]`, `float32`, on the same device as the model, with all images in a dense batch sharing the same preprocessed height and width. Do not apply another resize or rescale. `preprocess_train` is intentionally the same deterministic transform as `preprocess_val`; it does not provide training augmentation.

`encode_text()` accepts Gemma token IDs, not strings or OpenCLIP/CLIP token IDs. The tokenizer adds the official special tokens and right padding. `normalize=False` preserves the official backend's native embedding, which is already normalized; it does not provide an OpenCLIP-style unnormalized hidden state.

## Japanese text

The official tokenizer supports multilingual text, including Japanese:

```python
japanese_tokens = tokenizer(
    ["青い背景にある赤い円", "黄色い背景にある緑の三角形"]
).to(device)

with torch.inference_mode():
    japanese_features = model.encode_text(japanese_tokens, normalize=True)

print(japanese_features.shape)  # torch.Size([2, 768])
```

## Task-specific prompts

No prompt is applied by default. Select an official task prompt when creating the tokenizer; the prefix is added exactly once before tokenization:

```python
search_tokenizer = openclip_gemma.get_tokenizer(
    "embeddinggemma-2",
    prompt_name="SearchQuery",
)
classification_tokenizer = openclip_gemma.get_tokenizer(
    "embeddinggemma-2",
    prompt_name="Classification",
)

search_tokens = search_tokenizer(["How do I reset my password?"]).to(device)
```

Do not add the same official prefix to input text yourself when using `prompt_name`. The default `None` is appropriate when no task-specific prefix is wanted. `context_length` controls truncation and is capped at 8192; it does not set a fixed padding width.

## Zero-shot image classification

For images already preprocessed to a common shape and candidate texts:

```python
candidate_texts = [
    "a red circle on a blue background",
    "a green triangle on a yellow background",
    "青い背景にある赤い円",
]
tokenizer = openclip_gemma.get_tokenizer(prompt_name="Classification")
candidate_tokens = tokenizer(candidate_texts).to(device)

with torch.inference_mode():
    image_features = model.encode_image(image_tensor, normalize=True)
    text_features = model.encode_text(candidate_tokens, normalize=True)
    cosine_scores = image_features @ text_features.T

print(cosine_scores.shape)  # [number of images, number of candidates]
```

These are unscaled cosine similarities, not calibrated classification probabilities. The optional softmax in the example script is a relative distribution over the supplied candidates only; it does not reproduce OpenCLIP's learned `logit_scale`.

Run the repository's self-contained image demo from a checkout:

```bash
uv run python examples/zero_shot_classification.py --demo --show-relative-softmax --device cpu
```

## Migrating an OpenCLIP inference call

An OpenCLIP workflow commonly creates an architecture-specific model and tokenizer. For the supported EmbeddingGemma path, the import, model name, and tokenizer should be changed, and the official task/preprocessing contract must be followed:

```python
# Before:
# import open_clip
# model, _, preprocess = open_clip.create_model_and_transforms(
#     "ViT-B-32", pretrained="..."
# )
# tokenizer = open_clip.get_tokenizer("ViT-B-32")

# After:
import openclip_gemma as open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    "embeddinggemma-2",
    device="cpu",
)
tokenizer = open_clip.get_tokenizer("embeddinggemma-2")
```

The familiar factory/encoder shape does not imply full behavioral or numerical compatibility with a CLIP model. Review the [compatibility report](docs/openclip-compatibility-report.md) before adapting an existing application.

## Compatibility limitations

- Only `embeddinggemma-2` and its pinned Google checkpoint are supported. `pretrained=None`, arbitrary architectures, and unknown factory arguments are rejected.
- `create_model_and_transforms()` returns the familiar three values, but train/validation preprocessing is deterministic and identical; it is not OpenCLIP augmentation.
- Image and text token IDs, prompts, normalization behavior, and preprocessing are EmbeddingGemma-specific. Existing CLIP tokens and CLIP-normalized image tensors are not interchangeable.
- Image batches require a common preprocessed height and width. No extra resize or padding is applied to make different sizes stackable.
- Tokenizer padding is dynamic, not fixed-width 77. The maximum supported context is 8192; practical memory usage at that limit has not been measured.
- `forward()`, `get_logits()`, training, audio, and video are unsupported.
- CPU FP32 and BF16 were tested against the official Sentence Transformers inference path for the pinned revision. CUDA validation is outstanding. BF16 was measured separately and has no general numerical acceptance guarantee.

## About the name

The name `openclip-gemma` reflects the purpose of this
project: providing an OpenCLIP-inspired interface for
Google's EmbeddingGemma 2 model.

This is an independently developed library, not an
official extension of OpenCLIP or a Google-maintained
package.

The names OpenCLIP and Gemma refer to their respective
projects and technologies. No affiliation or endorsement
is implied.

## License and model

This library is distributed under the [Apache License 2.0](LICENSE). The model card at the pinned [EmbeddingGemma 2 revision](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b) declares `apache-2.0` and links Google's [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license); those separate terms apply to model weights. No model weights are included in the source distribution or wheel.

OpenCLIP is a separate project licensed under [MIT](https://github.com/mlfoundations/open_clip/blob/main/LICENSE). This package uses OpenCLIP as an API compatibility target but does not bundle or copy its implementation code. It also uses the PyTorch, Transformers, torchvision, and Pillow APIs without bundling their code; their licenses remain separate: [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE), [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE), and [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE).
