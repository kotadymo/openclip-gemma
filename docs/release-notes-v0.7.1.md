# OpenVLCLIP v0.7.1 — Release Notes

OpenVLCLIP v0.7.1 is a backward-compatible bug-fix release for the five
registered v0.7.0 checkpoints. It does not change model weights, embedding
calculations, or the supported model registry.

## Fixes

- **Native Video Index change detection:** `build_or_update_native_video_index()`
  accepts `verify_contents=True` to compare SHA-256 hashes and re-embed videos
  whose bytes changed, even when their size and modification time are unchanged.
  The default size/mtime path remains fast. The `native-index` CLI command now
  provides `--verify-contents`.
- **Legacy index handling:** Native Video Index metadata stores the hash as an
  optional field without changing format version 1. Existing v0.7.0 indexes
  without hashes remain readable; enabling strict verification safely
  re-embeds those entries once to establish hashes.
- **Qwen native-video limit:** The Qwen3-VL-Embedding-2B model exposes
  `native_video_max_frames == 64`. Larger `max_frames` values are rejected
  before video scanning or decoding, and searches reject indexes whose stored
  limit exceeds that of the loaded model.
- **Distribution verification:** Wheel and sdist checks require the SigLIP 2
  and Qwen 2B backend modules, verify the five registered model IDs in the
  README, and derive the default expected version from `pyproject.toml`.

## Compatibility and limitations

- The default Native Video Index update remains size/mtime based. Strict
  verification reads every video to compute SHA-256 and therefore adds I/O.
- Existing native-video and frame-video indexes remain readable. Frame-video
  indexing behavior is unchanged.
- No model weights are included in the wheel or sdist. The five v0.7.0
  checkpoints remain supported; Qwen3-VL-Embedding-8B is not registered.
- Qwen native-video decoding still depends on the optional `qwen3vl` extra and
  a compatible TorchCodec/FFmpeg runtime.
- CUDA was unavailable in the development environment; this patch does not
  claim new CUDA integration results. Existing model parity evidence remains
  documented in the [v0.7.0 model provenance report](model-provenance-v0.7.0.md).

## Install

```bash
pip install "openvlclip==0.7.1"
pip install "openvlclip[dinotxt]==0.7.1"
pip install "openvlclip[qwen3vl]==0.7.1"
```

See the [release checklist](release-checklist-v0.7.1.md) for publication and
verification evidence.
