# Changelog

## Unreleased

### Planned for v0.2.0

- Add the `openclip_gemma.batching.iter_image_embeddings` streaming API.
- Add incremental image indexing and cosine-similarity search APIs.
- Add optional PyAV-based video-frame indexing and search APIs with playback-relative timestamps.
- Preflight `.npz` members and NPY headers, with explicit archive and metadata size limits.
- Expand CI coverage for the optional video extra on Windows and Ubuntu.

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
