# Changelog

## v0.1.1 (Unreleased)

- Expand the declared Python range and classifiers to Python 3.12-3.14,
  excluding Python 3.14.1 due to the torchvision dependency metadata.
- Validate the Windows and Ubuntu CI matrix on Python 3.12, 3.13, and 3.14.
- Update the CI actions to pinned Node.js 24-compatible revisions.
- Make CI wheel verification and clean-install paths work across Windows and Ubuntu.
- Add opt-in CUDA integration coverage for image and Japanese text embeddings,
  including CPU comparison, inference timing, and peak allocated GPU memory.
- Record the Python compatibility investigation and per-version validation.
- Prepare a gated PyPI Trusted Publishing workflow for the v0.1.1 tag.
