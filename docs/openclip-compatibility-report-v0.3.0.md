# OpenCLIP Compatibility Report for v0.3.0

## Scope and evidence

This report compares the public inference surface at
`mlfoundations/open_clip` commit
[`7067b350050dc409bfdfdbb3f0cc6c9a1651c000`](https://github.com/mlfoundations/open_clip/tree/7067b350050dc409bfdfdbb3f0cc6c9a1651c000)
with `openclip-gemma` v0.3.0 development code. It covers ten inference
scenarios, not all OpenCLIP APIs or architectures.

The OpenCLIP reference was imported from that exact source commit with
`timm`, `ftfy`, and `wcwidth` installed in a temporary dependency directory.
No project environment or repository dependency was changed. Its CPU smoke
tests used a randomly initialized `ViT-B-32` (`load_weights=False`): factory
discovery, preprocessing, tokenization, encoding, `forward()`, and
`get_logits()` ran successfully. Those runs establish API behavior and tensor
shapes only; they are not pretrained-quality or numerical-equivalence tests.
The `create_model_from_pretrained()` wrapper was also executed with its
underlying model loader and transform builder mocked, so its return-shape
behavior was checked without loading an unrelated checkpoint.

The migrated EmbeddingGemma path was exercised by
`tests/integration/test_openclip_compatibility_integration.py` on CPU FP32,
CPU BF16, CUDA FP32, and CUDA BF16. All four cases passed on the pinned
EmbeddingGemma 2 revision. Factory model-only return behavior was separately
executed by the mocked unit test
`test_create_model_from_pretrained_can_return_model_only` in
`tests/test_factory.py`; no pretrained weights were loaded for that case.

Classification:

- **A** — works with only the import/module name and supported model identifier
  changed.
- **B** — works after an EmbeddingGemma-specific option, preprocessing,
  prompting, or output handling change.
- **C** — the original OpenCLIP numerical meaning is not supported. A
  differently named cosine-similarity alternative may exist, but it is not
  equivalent to OpenCLIP's trained logits.

The edit counts below are the number of line-level substitutions/additions in
the shown example, not a diff of a caller's entire application.

## Ten representative scenarios

### 1. Discover supported models — B, 1 edited line

OpenCLIP:

```python
import open_clip

model_names = open_clip.list_models()
```

Migration:

```python
import openclip_gemma as open_clip

model_names = open_clip.list_models()
```

Execution: both implementations returned a list without loading weights.
OpenCLIP reported 191 architectures at the pinned source commit;
`openclip-gemma` returned only `["embeddinggemma-2"]`. The call shape is
compatible, but the available model set is intentionally narrower.

### 2. Discover pretrained tags — B, 1 edited line

OpenCLIP:

```python
tags = open_clip.list_pretrained(as_str=True)
```

Migration:

```python
tags = open_clip.list_pretrained(as_str=True)
```

Execution: both calls returned strings. OpenCLIP has many architecture/tag
pairs; `openclip-gemma` returns `["embeddinggemma-2:official"]`. Here,
`"official"` is this library's alias for the pinned Google model revision, not
an OpenCLIP checkpoint tag.

### 3. Load a model and image transform — B, 1 edited line

OpenCLIP:

```python
model, preprocess = open_clip.create_model_from_pretrained(
    "hf-hub:organization/model"
)
```

Migration:

```python
model, preprocess = open_clip.create_model_from_pretrained(
    "embeddinggemma-2", pretrained="official", precision="fp32", device="cpu"
)
```

Execution: the EmbeddingGemma factory was exercised with the pinned checkpoint
in the four-device/precision integration matrix. The OpenCLIP wrapper was run
with its underlying model loader and transform builder mocked; its real
checkpoint-loading path was not executed because it would download an
unrelated checkpoint. The API shape is similar, but identifiers and
checkpoint semantics differ.

### 4. Load only the model — A, 1 edited line

OpenCLIP:

```python
model = open_clip.create_model_from_pretrained(
    "hf-hub:organization/model", return_transform=False
)
```

Migration:

```python
model = open_clip.create_model_from_pretrained(
    "embeddinggemma-2", pretrained="official", return_transform=False
)
```

Execution: both wrapper implementations were called with their underlying
model loaders mocked. Each returned one model object with
`return_transform=False`; neither pretrained checkpoint was loaded for this
case.

### 5. Preprocess and encode an image batch — B, 2 edited lines

OpenCLIP:

```python
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
images = torch.stack([preprocess(image) for image in pil_images]).to(device)
image_features = model.encode_image(images)
```

Migration:

```python
model, _, preprocess = open_clip.create_model_and_transforms(
    "embeddinggemma-2", pretrained="official", device=device
)
images = torch.stack([preprocess(image) for image in pil_images]).to(device)
image_features = model.encode_image(images, normalize=True)
```

Execution: the pinned OpenCLIP source successfully preprocessed and encoded a
two-image batch using a random CPU model. The migrated path produced
`[2, 768]` embeddings in the real-model CPU/CUDA FP32/BF16 integration cases.
The returned transform is mandatory: EmbeddingGemma uses its own
aspect-ratio-preserving processor and variable image dimensions; do not apply
CLIP normalization or resizing afterward.

### 6. Tokenize and encode text — B, 2 edited lines

OpenCLIP:

```python
tokenizer = open_clip.get_tokenizer("ViT-B-32")
text_tokens = tokenizer(["a red object", "a blue object"]).to(device)
text_features = model.encode_text(text_tokens)
```

Migration:

```python
tokenizer = open_clip.get_tokenizer("embeddinggemma-2", prompt_name="SearchQuery")
text_tokens = tokenizer(["a red object", "a blue object"]).to(device)
text_features = model.encode_text(text_tokens, normalize=True)
```

Execution: both pinned implementations tokenized and encoded two strings. The
OpenCLIP smoke test returned 512-dimensional features; the migrated real-model
test returned 768-dimensional features. Token IDs are model-specific and are
not interchangeable. Select the official task prompt matching the text's
role; `SearchQuery` is appropriate for retrieval queries.

### 7. Compute an image/text similarity matrix — B, 2 edited lines

OpenCLIP:

```python
scores = image_features @ text_features.T
```

Migration:

```python
image_features = model.encode_image(images, normalize=True)
text_features = model.encode_text(text_tokens, normalize=True)
scores = image_features @ text_features.T
```

Execution: both paths produced a finite `[2, 2]` matrix in their CPU smoke
tests. For EmbeddingGemma, the scores are cosine similarities between
normalized embeddings, not calibrated probabilities or OpenCLIP-trained
logits.

### 8. Call `model(image, text)` — C for OpenCLIP logits, 3 adapted lines

OpenCLIP:

```python
image_features, text_features, logit_scale = model(images, text_tokens)
logits_per_image = logit_scale * (image_features @ text_features.T)
logits_per_text = logits_per_image.T
```

Migration:

```python
model = open_clip.create_model(
    "embeddinggemma-2", pretrained="official", device=device,
    logit_scale_mode="cosine",
)
image_features, text_features, scale = model(images, text_tokens)
cosine_scores = image_features @ text_features.T
```

Execution: both model `forward()` implementations were run. The OpenCLIP
reference returned its normal inference output and its scale parameter,
initialized to approximately `14.2857` in this random model. EmbeddingGemma's
default `"strict"` mode raises
`NotImplementedError`; explicit `"cosine"` mode returns normalized features
and scalar `1.0`, not CLIP logits. Callers must adapt how scores are produced.

### 9. Request dictionary output — C for scale/logit semantics, 3 adapted lines

OpenCLIP:

```python
model.output_dict = True
output = model(images, text_tokens)
scores = output["logit_scale"] * (
    output["image_features"] @ output["text_features"].T
)
```

Migration:

```python
model = open_clip.create_model(
    "embeddinggemma-2", pretrained="official", device=device,
    output_dict=True, logit_scale_mode="cosine",
)
output = model(images, text_tokens)
cosine_scores = (
    output["image_features"] @ output["text_features"].T
)
```

Execution: the pinned OpenCLIP reference and migrated real-model tests both
returned a dictionary containing `image_features`, `text_features`, and
`logit_scale`. The key layout is similar, but the migrated scale is a
non-trained scalar `1.0`; exponentiating it or treating it as a learned CLIP
scale is incorrect. Use the feature keys and calculate cosine similarity
explicitly.

### 10. Call `get_logits()` — C for trained-logit semantics, 1 setup line

OpenCLIP:

```python
image_logits, text_logits = model.get_logits(images, text_tokens)
```

Migration:

```python
model = create_model(
    "embeddinggemma-2", pretrained="official", device=device,
    logit_scale_mode="cosine",
)
image_scores, text_scores = model.get_logits(images, text_tokens)
```

Execution: both functions ran. OpenCLIP applies its exponentiated
`logit_scale` parameter (and may apply bias); `openclip-gemma` returns
unscaled cosine similarities and the transposed matrix. Shapes and orientation
match, but numerical meanings do not. The default strict mode rejects this
call to avoid suggesting otherwise.

## Numerical and scope conclusions

- The OpenCLIP CPU smoke model produced 512-dimensional features and a
  `logit_scale` parameter whose initial exponentiated value was approximately
  `14.2857`; weights were random, so this was not a trained checkpoint result.
- EmbeddingGemma 2 returns 768-dimensional embeddings and has no learned CLIP
  `logit_scale`. There is no numerical-equivalence claim for OpenCLIP logits.
- The migrated EmbeddingGemma calls passed on CPU FP32/BF16 and RTX 4070 Ti
  SUPER CUDA FP32/BF16. This validates the fixed EmbeddingGemma inference
  route, not retrieval quality or equivalence to a CLIP checkpoint.
- `openclip-gemma` does not replace the `open_clip` import namespace. Use
  `import openclip_gemma as open_clip` or import its APIs explicitly.
- Training, fine-tuning, arbitrary OpenCLIP architectures, OpenCLIP checkpoint
  loading, and FP16 remain outside the supported surface.

## Revalidation of the executable example and CPU BF16 path

The standalone example
[`examples/openclip_compatibility_v030.py`](../examples/openclip_compatibility_v030.py)
was run on 2026-10-19 with its generated synthetic image and two search-query
texts. `--image` can instead be used to provide a local image. The example
exercises Factory discovery/loading, both `forward()` return formats, and
`get_logits()`.

| Device and precision | Feature dtype | Image/text feature shapes | Tuple/dict max difference | `get_logits()` vs direct cosine max difference | PyTorch peak allocated / reserved |
| --- | --- | --- | ---: | ---: | --- |
| CPU FP32 | `torch.float32` | `[1, 768]` / `[2, 768]` | 0.0 | 0.0 | Not measured |
| CPU BF16 | `torch.bfloat16` | `[1, 768]` / `[2, 768]` | 0.0 | 0.0 | Not measured |
| CUDA FP32 | `torch.float32` | `[1, 768]` / `[2, 768]` | 0.0 | 0.0 | 3,137,347,072 / 3,214,934,016 bytes |
| CUDA BF16 | `torch.bfloat16` | `[1, 768]` / `[2, 768]` | 0.0 | 0.0 | 1,594,758,656 / 1,646,264,320 bytes |

The example's zero differences validate API output consistency for these
inputs; they are **not** reference-model embedding errors. Its memory figures
are PyTorch allocator peaks after the model has loaded, for one synthetic
image, two texts, and the example's forward and logits calls. The measured
GPU was an NVIDIA GeForce RTX 4070 Ti SUPER with 17,170,956,288 bytes reported
as total device memory. Host process RSS was not measured.

The complete opt-in integration selection was rerun with
`OPENCLIP_GEMMA_RUN_INTEGRATION=1` using Python 3.12.10, PyTorch 2.14.1+cu130,
Transformers 5.20.1, Sentence Transformers 6.1.0, and PyAV 19.0.1. It completed
with **21 passed, 0 failed, 0 skipped, 164 deselected** in 863.50 seconds.
The CUDA device was the RTX 4070 Ti SUPER (CUDA runtime 13.0; BF16 supported).

The integration comparisons against the pinned Sentence Transformers
implementation measured:

| Comparison | Dtype | Device | Maximum absolute error | Notes |
| --- | --- | --- | ---: | --- |
| Image embeddings | FP32 | CPU | `3.2223761081695557e-7` | Maximum over RGB, grayscale, wide, tall, batch, and normalized cases |
| Text embeddings | FP32 | CPU | `3.982335329055786e-6` | Maximum over no prompt, `SearchQuery`, `Classification`, normalized, and EOS cases |
| Image embeddings | BF16 | CPU | `0.0` | Exact after BF16 quantization for tested fixtures |
| Text embeddings | BF16 | CPU | `0.0` | Exact after BF16 quantization for tested fixtures |
| Image embeddings | FP32 | CUDA | `2.682209014892578e-7` | Additional pinned-reference run; two synthetic images |
| Text embeddings | FP32 | CUDA | `5.345791578292847e-6` | Same run; `Classification` prompt |
| Cosine score matrix | FP32 | CUDA | `2.1457672119140625e-6` | Compared with the reference image/text matrix |

The CUDA FP32 reference run reported minimum embedding cosine similarity
`1.0` for both modalities. Separately, the CUDA integration test measured
cross-device CPU-to-CUDA maximum errors of `9.452924132347107e-8` for images
and `6.51925802230835e-8` for text in FP32, with a peak allocated value of
`3,171,772,416` bytes. In BF16 it measured `0.001953125` for both modalities,
with cosine similarities `0.9999682903289795` (image) and
`0.9999750852584839` (text), and peak allocated memory of `1,587,407,872`
bytes. These BF16 CUDA figures are CPU-versus-CUDA comparisons, not a
CUDA-BF16-versus-Sentence-Transformers comparison.

The full integration run also covered image preprocessing, batch inference,
search, video frame indexing/search, and the OpenCLIP-style API. The examples
use synthetic fixtures; they do not establish real-media retrieval quality.

See the [migration guide](migration-v0.3.0.md) for end-to-end examples and
strict/cosine mode guidance.
