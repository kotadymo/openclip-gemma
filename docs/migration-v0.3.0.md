# Migration Guide: OpenCLIP Inference to openclip-gemma v0.3.0

`openclip-gemma` provides an OpenCLIP-inspired inference API for the fixed
EmbeddingGemma 2 checkpoint. It does not load OpenCLIP architectures or
checkpoints, and it does not claim numerical equivalence to CLIP.

## Module and model selection

Use the `openclip_gemma` import name; the package does not shadow `open_clip`.
Only `embeddinggemma-2` and the logical `official` pretrained alias are
supported:

```python
import openclip_gemma as open_clip

print(open_clip.list_models())
print(open_clip.list_pretrained())
print(open_clip.list_pretrained(as_str=True))
```

`official` resolves to `google/embeddinggemma-2` at the revision pinned by the
package. It is not an OpenCLIP checkpoint tag. Unsupported model names,
pretrained tags, and unknown factory options raise explicit errors.

## Model loading and image encoding

`create_model_and_transforms()` preserves its three-element return shape. Its
training and validation transforms are the same deterministic official
EmbeddingGemma processor transform:

```python
import torch
from PIL import Image

from openclip_gemma import create_model_and_transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_train, preprocess_val = create_model_and_transforms(
    "embeddinggemma-2",
    pretrained="official",
    precision="fp32",
    device=device,
)
model.eval()

images = [
    Image.open("photos/a.jpg").convert("RGB"),
    Image.open("photos/b.jpg").convert("RGB"),
]
image_batch = torch.stack([preprocess_val(image) for image in images]).to(device)

with torch.inference_mode():
    image_features = model.encode_image(image_batch, normalize=True)

assert image_features.shape == (2, 768)
```

Use the returned preprocessing function exactly once. It may produce
aspect-ratio-preserving, variable-sized tensors; images combined in one dense
tensor batch must have matching shapes. The supported input is float32
`[B, 3, H, W]` in the processor's `[0, 1]` range, not OpenCLIP-normalized
pixels.

To get only the model and a single transform, use
`create_model_from_pretrained()`:

```python
model, preprocess = open_clip.create_model_from_pretrained(
    "embeddinggemma-2", pretrained="official", device=device
)
```

Use `return_transform=False` when preprocessing is managed separately:

```python
model = open_clip.create_model_from_pretrained(
    "embeddinggemma-2",
    pretrained="official",
    device=device,
    return_transform=False,
)
```

## Text encoding and task prompts

CLIP token IDs cannot be reused. The EmbeddingGemma tokenizer returns Gemma
token IDs as a dynamically right-padded `torch.long` tensor. Provide the
official prompt suited to the text's task:

```python
from openclip_gemma import get_tokenizer

search_tokenizer = get_tokenizer(
    "embeddinggemma-2",
    prompt_name="SearchQuery",
)
text_tokens = search_tokenizer(
    ["a person riding a bicycle", "a red car on a road"]
).to(device)

with torch.inference_mode():
    text_features = model.encode_text(text_tokens, normalize=True)

assert text_features.shape == (2, 768)
```

Other prompt choices include `"Classification"` for candidate labels and
`None` for no task prompt. Do not prepend an official prompt manually when
`prompt_name` is set.

## Similarity and model outputs

The model returns normalized sentence embeddings. Compare image and text
features with cosine similarity:

```python
with torch.inference_mode():
    image_features = model.encode_image(image_batch, normalize=True)
    text_features = model.encode_text(text_tokens, normalize=True)
    cosine_similarity = image_features @ text_features.T
```

The result is a similarity score, not a probability.

`forward()` and `get_logits()` default to strict mode and raise
`NotImplementedError`: EmbeddingGemma 2 does not have a trained OpenCLIP
`logit_scale`. If an application explicitly wants the cosine alternative,
select it while constructing the model:

```python
model = open_clip.create_model(
    "embeddinggemma-2",
    pretrained="official",
    device=device,
    logit_scale_mode="cosine",
    output_dict=True,
)
output = model(image_batch, text_tokens)
cosine_similarity = (
    output["image_features"] @ output["text_features"].T
)
```

The `logit_scale` key in this mode contains scalar `1.0`, not a learned CLIP
parameter; do not exponentiate it or describe the resulting values as
OpenCLIP logits. `get_logits()` returns cosine similarities in image-to-text
and text-to-image orientation. This behavior is opt-in and semantically
different from trained CLIP logits.

## Device and precision

The model and input tensors must use the same device. CPU and CUDA are
supported when the corresponding PyTorch build and device are available.
FP32 is the reference precision; BF16 is supported. FP16 is intentionally
rejected. The verified CUDA hardware and numerical results are listed in the
repository README and compatibility report; they do not imply support for
every GPU or driver combination.

## Compatibility boundary

Existing v0.2 APIs for image batching, image search, and optional video-frame
search remain separate from the OpenCLIP-inspired model API. The v0.3.0
factory and model additions do not change their index formats, tokenization,
preprocessing, or embedding behavior.

For case-by-case results and observed tensor shapes, see the
[v0.3.0 OpenCLIP compatibility report](openclip-compatibility-report-v0.3.0.md).
