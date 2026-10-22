# Migration Guide: OpenVLCLIP v0.5.0 to v0.6.0

The Python distribution (`openvlclip`) and import namespace (`openvlclip`) do
not change in v0.6.0. EmbeddingGemma 2 remains the default model and retains
its existing Factory, tokenizer, preprocessing, task-prompt, MRL, image-search,
video-search, and index contracts. The new `dinov2-dinotxt` backend is
opt-in.

OpenVLCLIP v0.6.0 was published on 2026-10-22 to
[PyPI](https://pypi.org/project/openvlclip/0.6.0/) with a public
[GitHub Release](https://github.com/kotadymo/openvlclip/releases/tag/v0.6.0).

## Install the released package

Install the base package:

```bash
pip install openvlclip==0.6.0
```

For DINOv2 + dino.txt, install the optional tokenizer dependencies:

```bash
pip install "openvlclip[dinotxt]==0.6.0"
```

For video-frame indexing and search, install:

```bash
pip install "openvlclip[video]==0.6.0"
```

The base install remains independent of dino.txt's `ftfy` and `regex`
dependencies. For source development, use
`uv sync --locked --group dev --extra dinotxt`.

The package does not include model weights. The model loader retrieves the
pinned external artifacts at runtime and verifies their SHA-256 hashes.

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

The DINOv2 repository's pinned [LICENSE](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE)
is Apache-2.0, and its [model card](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/MODEL_CARD.md)
lists Apache License 2.0 for the DINOv2 ViT family. The official
[dino.txt sources](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md)
link the externally hosted Vision Head and Text Encoder checkpoints, but
their individual license terms are not separately specified in the pinned
repository. Do not infer that the Apache-2.0 repository/model-card terms
conclusively cover those assets. OpenVLCLIP's wheel and sdist do not bundle
the weights; publication of the package is separate from that unresolved
license-scope question.

The official [BPE vocabulary license](https://dl.fbaipublicfiles.com/dinov2/thirdparty/LICENSE)
is MIT (OpenAI, 2021). See the
[dino.txt reference](dinotxt-v0.6.0.md) for the pinned source, artifact
hashes, test results, and license notes.

The release record is in the
[v0.6.0 release checklist](release-checklist-v0.6.0.md). The public release
must not be confused with a license grant for separate third-party assets.
