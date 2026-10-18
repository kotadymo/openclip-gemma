# Changelog

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
