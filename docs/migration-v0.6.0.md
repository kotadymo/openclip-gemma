# Migration Guide: OpenVLCLIP v0.5.0 to v0.6.0

The Python distribution (`openvlclip`) and import namespace (`openvlclip`) do
not change in v0.6.0. EmbeddingGemma 2 remains the default model and retains
its existing Factory, tokenizer, preprocessing, task-prompt, MRL, image-search,
video-search, and index contracts. The new `dinov2-dinotxt` backend is
opt-in.

## Install the optional dino.txt dependencies

The v0.6.0 source tree declares dino.txt-only runtime dependencies in the
`dinotxt` extra. From a source checkout:

```bash
uv sync --locked --group dev --extra dinotxt
```

The base install remains independent of `ftfy` and `regex`. Do not install
weights from this package; the model downloads the pinned external artifacts
at runtime and checks their SHA-256 hashes.

## Select the DINOv2 + dino.txt model

Factory calls that omit `model_name` continue to select EmbeddingGemma 2.
Select dino.txt explicitly:

```python
from openvlclip import create_model_and_transforms, get_tokenizer

model, _, preprocess = create_model_and_transforms(
    "dinov2-dinotxt",
    device="cpu",
)
tokenizer = get_tokenizer("dinov2-dinotxt")
```

The global image and text embeddings are 2048-dimensional. `normalize=True`
normalizes the complete vector. `truncate_dim` and MRL dimensions are not
supported by dino.txt. Its tokenizer accepts no EmbeddingGemma task prompts,
and the complete vision/text model must be loaded together.

`forward()` and `get_logits()` preserve the official dino.txt strict scoring
path, including the Hub-initialized logit scale. `logit_scale_mode="cosine"`
selects unscaled cosine scores instead. The modes must not be treated as
equivalent or as calibrated probabilities.

For dense inference, `encode_patches()` returns `[B, N, 1024]` features;
`similarity_map()` uses the text embedding's patch-aligned last 1024
dimensions. Similarity-map values are scores, not probabilities or
pixel-accurate masks. EmbeddingGemma reports these APIs as unsupported.

## Image and video index compatibility

An index records its backend/model identity, revision, embedding dimension,
strategy, normalization, preprocessing, and prompt settings. Searches,
updates, and conversions reject mismatched spaces. In particular, dino.txt
2048-dimensional global embeddings, its 1024-dimensional patch features, and
EmbeddingGemma MRL embeddings cannot be mixed.

Existing v0.4.0/v0.5.0 EmbeddingGemma image and video indexes retain their
legacy read and conversion behavior. No re-embedding is required solely due
to the v0.6.0 package update. This guide makes no claim that dino.txt is
supported by the video-search API.

## Release and licensing status

The dino.txt Vision Head and Text Encoder checkpoints are fetched from the
official DINOv2 asset URLs and are not bundled in distributions. The pinned
upstream repository does not clearly state a license for those specific
checkpoints. This remains a release blocker until explicit licensing
confirmation is recorded; the Apache-2.0 repository license alone is not
treated as conclusive permission for the separate weights.

The `0.6.0` source version is prepared independently of PyPI publication.
Consult the [v0.6.0 release checklist](release-checklist-v0.6.0.md) before
creating a release tag.
