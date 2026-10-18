# Batch image embedding prototype (v0.2.0 development)

## Status

Phase 1 adds an internal streaming implementation for embedding image files. It is not exported by `openclip_gemma`, is not part of the published v0.1.1 package API, and may change before a public API is selected. Image indexing/search and video-frame search are not implemented by this phase.

## Internal contract

The source-tree helper is `_iter_image_embeddings(model, preprocess, image_paths, *, batch_size=8)`. It accepts an iterable of string or path-like paths readable by Pillow and lazily yields `(Path, embedding)` pairs in input order. Each embedding is a CPU tensor with shape `[768]` and the model's output dtype. Keeping results on CPU and yielding one at a time avoids retaining all output vectors or GPU tensors for a large collection.

Example from a checkout of this development branch:

```python
from pathlib import Path

from openclip_gemma import create_model_and_transforms
from openclip_gemma.batching import _iter_image_embeddings

model, _, preprocess = create_model_and_transforms(
    "embeddinggemma-2",
    device="cuda",
)
image_paths = Path("images").glob("**/*.jpg")

for path, embedding in _iter_image_embeddings(
    model,
    preprocess,
    image_paths,
    batch_size=8,
):
    print(path, embedding.shape, embedding.device)
```

The example is development-only; `batch_size=8` is an example, not a universally safe value. Lower it if the selected device runs out of memory. The helper uses the supplied official preprocessing transform and the model's existing `encode_image()` method. It does not alter the model revision, preprocessing, normalization, or public exports.

## Batching and memory behavior

- At most `batch_size` source images are opened and preprocessed at a time; image files are closed after preprocessing.
- The official aspect-preserving transform can produce different `H,W` sizes. Within each bounded chunk, only tensors with identical shapes are stacked and encoded together. Results are then yielded in original input order. No resize, crop, or padding is introduced to force heterogeneous images into one tensor.
- CUDA receives at most one same-shape group at a time. Encoded vectors are copied to CPU before they are yielded.
- Pillow decompression-bomb warnings/errors and image read/decode failures stop iteration with an error containing the input position and path. The helper does not silently skip damaged files; results from earlier chunks may already have been yielded.
- CUDA out-of-memory errors are re-raised as actionable `RuntimeError`s that identify the affected input indices and recommend lowering `batch_size`. No automatic retry changes the caller's requested batch behavior.
- An empty path iterable yields no results. `batch_size` must be a positive integer; booleans, non-integers, zero, and negative values are rejected.

## Phase 1 validation

Unit tests cover bounded chunks, ordering across differing preprocessed shapes, JPEG/PNG inputs, empty and invalid inputs, damaged/missing files, inference mode, CPU result tensors, and CUDA out-of-memory messaging. A separate opt-in integration test compares CPU and CUDA batch results with individual image encodings using the pinned real model.

## Decisions deferred to Phase 2

- Whether the stable API should return a dense tensor, an iterator, or write batches directly to an index.
- Image index metadata, persistence format, incremental detection of added/changed/deleted files, and cache validation.
- Search CLI design and top-k score semantics.

The iterator is used for this prototype because a dense result tensor grows with the full image collection. Phase 2 can decide whether to retain this streaming shape or provide a separate convenience API without changing `encode_image()`.
