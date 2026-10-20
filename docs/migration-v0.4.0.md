# Migration Guide: v0.3.0 to v0.4.0

This release adds Matryoshka Representation Learning (MRL) dimensions and
selective loading of the supported model's optional towers. Existing calls
continue to return the native 768-dimensional embeddings unless a smaller
dimension is requested.

## Encode at a smaller dimension

Use one of the supported dimensions, `768`, `512`, `256`, or `128`, for both
modalities in a similarity comparison:

```python
with torch.inference_mode():
    image_features = model.encode_image(image_batch, truncate_dim=256)
    text_features = model.encode_text(text_tokens, truncate_dim=256)
    scores = image_features @ text_features.T
```

For values below 768, the implementation takes the leading dimensions of the
native embedding and L2-normalizes the truncated vector, matching the pinned
EmbeddingGemma 2 MRL behavior. The default `truncate_dim=None` is unchanged.
`forward()` and `get_logits()` accept the keyword-only parameter when
`logit_scale_mode="cosine"` is selected; strict mode remains the default.

## Select loaded modalities

The default `modalities=None` keeps the prior full checkpoint configuration.
Opt in to a smaller model with `create_model()`:

```python
text_model = create_model(modalities=("text",))
image_text_model = create_model(modalities=("text", "image"))
```

The text transformer is required and shared. `("text",)` omits both vision
and audio configs; `("text", "image")` retains vision and omits audio.
`encode_image()` raises an explicit error for a text-only model. Factories
that return image transforms require the image modality. Audio inference and
image-only loading are not supported.

## Image and video indexes

Index builders accept `embedding_dim=768` by default, or `512`, `256`, or
`128`. The search functions read the stored dimension and encode the query at
the matching size. An incremental update cannot change a saved index's
dimension; use a separate output directory to build another size.

Image index format v2 reads legacy v1 indexes only at 768 dimensions. Video
format v3 reads legacy v2 indexes only at 768 dimensions; v1 video indexes
remain unsupported because they contain the older absolute-PTS timing
contract. The conversion APIs do not require model weights or source media:

```python
from openclip_gemma.image_search import convert_image_index
from openclip_gemma.video_search import convert_video_index

convert_image_index("image-index-768", "image-index-256", embedding_dim=256)
convert_video_index("video-index-768", "video-index-256", embedding_dim=256)
```

Each conversion requires a new, unused destination directory and cannot
increase the source dimension. It truncates and re-normalizes existing
vectors, updates metadata and checksums, and never modifies the source index.

The image and video CLI index commands accept `--embedding-dim` with the same
choices. Their default remains 768.

## Validation and limits

The MRL dimensions follow the official model's truncation-and-renormalization
contract. Reduced dimensions lower stored float32 vector size but do not
reduce the transformer's internal computation. Run the
[MRL benchmark guide](benchmark-mrl-v0.4.0.md) with an application-relevant
image set and, for retrieval metrics, a JSON query manifest with `query` text
and optional `relevant_images` paths relative to the image root. No
application-specific retrieval-quality conclusion follows from shape or
normalization tests alone.

The release preserves the existing preprocessing, tokenizer prompts,
default 768-dimensional values, strict logit-scale behavior, and safe
NumPy-index loading.
