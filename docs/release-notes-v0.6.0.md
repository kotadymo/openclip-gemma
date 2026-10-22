# OpenVLCLIP v0.6.0

Released on 2026-10-22:
[GitHub Release](https://github.com/kotadymo/openvlclip/releases/tag/v0.6.0) ·
[PyPI](https://pypi.org/project/openvlclip/0.6.0/)

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
- Existing v0.5.0 EmbeddingGemma 2 usage and index compatibility are retained.
- dino.txt does not support MRL truncation, EmbeddingGemma task prompts,
  selective loading, or video-search compatibility claims.
- Similarity-map values are cosine scores, not probabilities or pixel-accurate
  segmentation masks.
- No dataset-based accuracy or retrieval-quality improvement is claimed. See
  the [benchmark report](benchmark-v0.6.0.md).
- The DINOv2 repository license and model card identify Apache-2.0 for the
  repository/model family. Separate license terms for the external dino.txt
  Vision Head and Text Encoder checkpoints are not clearly specified in the
  pinned repository. The release does not include those weights and does not
  resolve that license-scope question; see the [dino.txt reference](dinotxt-v0.6.0.md).

## Install

Install the base distribution:

```bash
pip install openvlclip==0.6.0
```

Install dino.txt's optional tokenizer dependencies:

```bash
pip install "openvlclip[dinotxt]==0.6.0"
```

See the [README](../README.md), the
[migration guide](migration-v0.6.0.md), and the
[PyPI project page](https://pypi.org/project/openvlclip/0.6.0/).
