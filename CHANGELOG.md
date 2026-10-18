# Changelog

## v0.3.0

### Added

- Add OpenCLIP-style model discovery with `list_models()` and
  `list_pretrained()`.
- Add `create_model_from_pretrained()` with optional transform return.
- Add `output_dict` and explicit strict/cosine modes to the model factories.
- Add opt-in `forward()` and `get_logits()` cosine-similarity APIs while
  rejecting unsupported trained-CLIP logit semantics by default.
- Add a ten-scenario OpenCLIP compatibility report and a migration guide.

### Validation

- Default pytest suite: 160 passed, 25 skipped; the skips are expected when
  model integration is not enabled or CUDA-only contract tests run on CPU.
- Full pinned-model integration suite on the CUDA development PC:
  21 passed, 0 failed, with 164 non-integration tests deselected.
- The pinned OpenCLIP source was exercised on CPU with randomly initialized
  `ViT-B-32` weights for factory discovery, preprocessing, tokenization,
  encoding, `forward()`, and `get_logits()`. This validates API behavior only,
  not pretrained OpenCLIP accuracy.
- The pinned EmbeddingGemma 2 compatibility integration test passed for CPU
  FP32/BF16 and CUDA FP32/BF16 on an NVIDIA GeForce RTX 4070 Ti SUPER.
- CUDA cross-implementation comparisons now explicitly use FP32 for both
  Sentence Transformers and openclip-gemma; earlier comparisons exposed a
  BF16-vs-FP32 test-condition mismatch, not a relaxed tolerance.

## v0.2.0

### Added

- Add bounded, ordered image-file embedding with
  `openclip_gemma.batching.iter_image_embeddings`.
- Add incremental image indexing and Japanese/English cosine-similarity search.
- Add optional PyAV-based video-frame indexing and text search with
  playback-relative timestamps.
- Validate `.npz` members and NPY headers before loading, with explicit
  per-index archive and metadata size limits.
- Test the optional video extra on Windows and Ubuntu CI.

### Validation

- The v0.2.0rc1 TestPyPI wheel was installed in clean CPU and CUDA environments.
- CPU FP32 image/text embedding, batch embedding, image index/search, and
  synthetic-MP4 frame index/search passed on Windows 11 with Python 3.12.10,
  PyTorch 2.14.1+cpu, and PyAV 19.0.1.
- CUDA FP32 operations passed on an NVIDIA GeForce RTX 4070 Ti SUPER (16 GiB)
  with driver 610.88, Python 3.12.10, PyTorch 2.14.1+cu130,
  torchvision 0.29.1+cu130, CUDA runtime 13.0, and PyAV 19.0.1.
- Synthetic media validates the execution path, not real-world retrieval
  quality. Linux real-model inference and macOS remain unverified.

## v0.2.0rc1 (TestPyPI; 2026-10-18)

- Publish the v0.2.0 release candidate to TestPyPI as a wheel and sdist.
- Validate clean installs with dependencies from PyPI and the project wheel
  from TestPyPI, without mixing package indexes during dependency resolution.
- Pass published-package CPU and CUDA smoke tests for embeddings, batch
  processing, image search, and sampled video-frame search.
- Confirm the optional PyAV dependency and playback-relative timestamps using
  a synthetic MP4.
- Keep retrieval-quality claims separate from synthetic-media execution tests.

## v0.1.1

- Expand the declared Python range and classifiers to Python 3.12-3.14,
  excluding Python 3.14.1 due to the torchvision dependency metadata.
- Validate the Windows and Ubuntu CI matrix on Python 3.12, 3.13, and 3.14.
- Update the CI actions to pinned Node.js 24-compatible revisions.
- Make CI wheel verification and clean-install paths work across Windows and Ubuntu.
- Add opt-in CUDA integration coverage for image and Japanese text embeddings,
  including CPU comparison, inference timing, and peak allocated GPU memory.
- Record the Python compatibility investigation and per-version validation.
- Publish the package to PyPI with the v0.1.1 release workflow.
