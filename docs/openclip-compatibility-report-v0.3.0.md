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

See the [migration guide](migration-v0.3.0.md) for end-to-end examples and
strict/cosine mode guidance.
