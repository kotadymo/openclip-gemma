# OpenVLCLIP v0.6.0

> Draft only. Do not publish while the dino.txt Vision Head and Text Encoder
> checkpoint licensing gate remains unresolved.

OpenVLCLIP v0.6.0 adds a DINOv2 + dino.txt backend while preserving the
existing EmbeddingGemma 2 API and embedding behavior.

## Highlights

- Add the `dinov2-dinotxt` model to the Registry and shared Factory API.
- Add native 2048-dimensional image/text embeddings, official scaled logits,
  and separate scale-1 cosine scoring.
- Add 1024-dimensional patch embeddings and text similarity maps using the
  pinned official dino.txt computation path.
- Add dino.txt image search with embedding-space checks that reject
  incompatible models, revisions, preprocessing, prompts, strategies, and
  dimensions.
- Preserve EmbeddingGemma 2 MRL (768/512/256/128), task prompts, selective
  loading, image/video search, and existing image/video index formats.
- Keep dino.txt dependencies optional; its model weights are external runtime
  downloads and are not included in the wheel or source distribution.

## Compatibility and limitations

- The distribution and import name remain `openvlclip`; see the
  [v0.6.0 migration guide](migration-v0.6.0.md).
- dino.txt does not support MRL truncation, EmbeddingGemma task prompts,
  selective loading, or video-search compatibility claims.
- Similarity-map values are cosine scores, not probabilities or pixel-accurate
  segmentation masks.
- No dataset-based accuracy or retrieval-quality improvement is claimed. See
  the [benchmark report](benchmark-v0.6.0.md).
- Licensing for the dino.txt Vision Head and Text Encoder checkpoints must be
  resolved before these notes can be published.

## Install

After publication, install the base distribution with:

```bash
pip install openvlclip
```

Install dino.txt's optional tokenizer dependencies with:

```bash
pip install "openvlclip[dinotxt]"
```

See the [README](../README.md), the
[migration guide](migration-v0.6.0.md), and the
[PyPI project page](https://pypi.org/project/openvlclip/0.6.0/).
