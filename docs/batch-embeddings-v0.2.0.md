# Batch image embedding (v0.2.0)

## Status

Version 0.2.0 provides a module-level streaming API for embedding image files. The API is available from `openvlclip.batching` and is not re-exported from the package root. Image indexing/search and video-frame search are documented separately.

## Internal contract

The module-level API is `openvlclip.batching.iter_image_embeddings(model, preprocess, image_paths, *, batch_size=8)`. It accepts an iterable of string or path-like paths readable by Pillow and lazily yields `(Path, embedding)` pairs in input order. Each embedding is a CPU tensor with shape `[768]` and the model's output dtype. Keeping results on CPU and yielding one at a time avoids retaining all output vectors or GPU tensors for a large collection.

Example from a v0.2.0 source checkout:

```python
from pathlib import Path

from openvlclip import create_model_and_transforms
from openvlclip.batching import iter_image_embeddings

model, _, preprocess = create_model_and_transforms(
    "embeddinggemma-2",
    device="cuda",
)
image_paths = Path("images").glob("**/*.jpg")

for path, embedding in iter_image_embeddings(
    model,
    preprocess,
    image_paths,
    batch_size=8,
):
    print(path, embedding.shape, embedding.device)
```

The example is available from a source checkout only; `batch_size=8` is an example, not a universally safe value. Lower it if the selected device runs out of memory. The helper uses the supplied official preprocessing transform and the model's existing `encode_image()` method. It does not alter the model revision, preprocessing, or normalization.

## Batching and memory behavior

- At most `batch_size` source images are opened and preprocessed at a time; image files are closed after preprocessing.
- The official aspect-preserving transform can produce different `H,W` sizes. Within each bounded chunk, only tensors with identical shapes are stacked and encoded together. Results are then yielded in original input order. No resize, crop, or padding is introduced to force heterogeneous images into one tensor.
- CUDA receives at most one same-shape group at a time. Encoded vectors are copied to CPU before they are yielded.
- Pillow decompression-bomb warnings/errors and image read/decode failures stop iteration with an error containing the input position and path. The helper does not silently skip damaged files; results from earlier chunks may already have been yielded.
- CUDA out-of-memory errors are re-raised as actionable `RuntimeError`s that identify the affected input indices and recommend lowering `batch_size`. No automatic retry changes the caller's requested batch behavior.
- An empty path iterable yields no results. `batch_size` must be a positive integer; booleans, non-integers, zero, and negative values are rejected.

## Phase 1 validation

Unit tests cover bounded chunks, ordering across differing preprocessed shapes, JPEG/PNG inputs, empty and invalid inputs, damaged/missing files, inference mode, CPU result tensors, and CUDA out-of-memory messaging. A separate opt-in integration test compares CPU and CUDA batch results with individual image encodings using the pinned real model.

## Phase 1 boundary and Phase 2 decisions

The streaming iterator is a module-level development API and is not re-exported from the package root. Phase 2 uses it to update an on-disk image index without collecting all image embeddings in GPU memory. Index metadata, persistence, update detection, search scoring, and CLI behavior are described in the [image search design notes](image-search-v0.2.0.md).

The image-search index uses JSON metadata and a NumPy embedding matrix in a pickle-disabled `.npz` archive, with atomic replacement for updates. File-change detection uses path, size, and modification time; its limitations are documented in the Phase 2 notes. These choices do not change `encode_image()`.
