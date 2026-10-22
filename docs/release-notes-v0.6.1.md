# OpenVLCLIP v0.6.1

OpenVLCLIP v0.6.1 is a backward-compatible bug-fix release for the
EmbeddingGemma 2 and DINOv2 + dino.txt APIs.

- [PyPI project page](https://pypi.org/project/openvlclip/0.6.1/)
- [GitHub Release](https://github.com/kotadymo/openvlclip/releases/tag/v0.6.1)
- [Changelog](../CHANGELOG.md)

## Fixes

- **dino.txt batch embeddings:** An omitted `embedding_dim` now resolves to
  the selected backend's native dimension. dino.txt defaults to 2048, while
  EmbeddingGemma 2 retains its 768-dimensional default and supported MRL
  dimensions of 512, 256, and 128. Unsupported dino.txt truncation remains
  rejected.
- **Image index updates:** `build_or_update_image_index()` accepts
  `verify_contents=True` to compare SHA-256 content hashes when deciding
  whether to reuse an image embedding. The default size/mtime fast path is
  unchanged. Strict verification safely re-embeds legacy entries without a
  stored hash.
- **Video sampling:** Sampling skips across large presentation-timestamp gaps
  using exact `Fraction` arithmetic and a single integer step calculation,
  avoiding excessive per-interval loops without changing the target times.

## Compatibility and limitations

- Existing image-index formats remain readable; the content hash is optional
  metadata and does not require an index format-version change.
- EmbeddingGemma 2 embeddings, prompts, MRL dimensions, selective loading,
  image/video search, and dino.txt's official 2048-dimensional global and
  1024-dimensional patch output contracts are unchanged.
- Strict image-content verification reads and hashes file contents, so it
  costs more than the default metadata-only update check.
- dino.txt still does not support MRL truncation or EmbeddingGemma task
  prompts. The individual license terms for its external Vision Head and
  Text Encoder checkpoints remain unclear; see the
  [dino.txt reference](dinotxt-v0.6.0.md).
- CUDA FP32/BF16 validation is not claimed for this patch release.

## Install

```bash
pip install --upgrade openvlclip
pip install --upgrade "openvlclip[dinotxt]"
```
