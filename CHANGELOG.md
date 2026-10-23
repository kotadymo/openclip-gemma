# Changelog

## v0.7.1 — Unreleased

This backward-compatible patch release hardens native-video indexing without
changing registered models or embedding calculations.

### Fixed

- Add opt-in SHA-256 content verification to Native Video Index updates and
  `native-index --verify-contents`; strict updates re-embed content changes even
  when file size and modification time are unchanged.
- Expose and enforce the Qwen3-VL-Embedding-2B native-video frame limit of 64
  before scanning or decoding videos, and reject existing indexes above the
  loaded model's limit during search.
- Preserve compatibility with v0.7.0 Native Video Index metadata; legacy
  records without hashes remain readable and are re-embedded when strict
  verification is first enabled.
- Keep distribution verification aligned with the five registered v0.7.0
  models and derive its default expected version from the project metadata.

See [v0.7.1 release notes](docs/release-notes-v0.7.1.md).

## v0.7.0 — 2026-10-23

This minor release expands OpenVLCLIP to **five officially registered checkpoints**: EmbeddingGemma 2, DINOv2 + dino.txt, two SigLIP 2 Base variants (FixRes and NaFlex), and **Qwen3-VL-Embedding-2B**. Qwen3-VL-Embedding-8B is explicitly **deferred**; it is **not** registered or supported in v0.7.0.

### Added
- Google SigLIP 2 FixRes 224 and NaFlex Base backends, preserving official image/text processors and trained logit scale/bias.
- Qwen3-VL-Embedding-2B with instruction-aware text, image, native video, multimodal inputs, and MRL 64–2048 dimensional outputs.
- Processor-aware shared batching and search, native-video indexing, compatible embedding-space metadata, and opt-in `[qwen3vl]` dependencies.
- Reusable tag-only `.github/workflows/publish-pypi.yml` with dynamic SemVer validation and existing PyPI Trusted Publishing safeguards.

### Fixed
- dino.txt similarity-map resource budgeting and invalid special-token handling.
- `examples/image_search.py index --verify-contents` strict file-change detection.
- Existing frame-video embedding dimension inference for backends with non-768 native dimensions.

### Compatibility and validation
- Preserve EmbeddingGemma 2 MRL and the original dino.txt embedding/patch/map calculations.
- Preserve legacy image/video indexes via versioned metadata and compatibility checks.
- The pinned real-model parity results for SigLIP 2 FixRes, SigLIP 2 NaFlex, and Qwen 2B CPU/CUDA (FP32/BF16 as documented) are in [model provenance](docs/model-provenance-v0.7.0.md). Synthetic tests are **not** claims about retrieval quality.
- Qwen 8B CPU research results remain documented but do not imply v0.7.0 support.
- Release CI passed and the authenticated GitHub Actions workflow uploaded both distributions to [PyPI](https://pypi.org/project/openvlclip/0.7.0/) after Environment approval. The [GitHub Release](https://github.com/kotadymo/openvlclip/releases/tag/v0.7.0) is public; the published wheel and sdist hashes match the PyPI JSON metadata, and a clean Python 3.12 wheel installation verified the five-model registry and Qwen 8B absence. See the [release checklist](docs/release-checklist-v0.7.0.md) for details.

See [v0.7.0 release notes](docs/release-notes-v0.7.0.md).


## v0.6.1 — 2026-10-22

Released on [PyPI](https://pypi.org/project/openvlclip/0.6.1/) and as a
[GitHub Release](https://github.com/kotadymo/openvlclip/releases/tag/v0.6.1).

Install the base package with `pip install openvlclip==0.6.1`, or include the
optional dino.txt dependencies with
`pip install "openvlclip[dinotxt]==0.6.1"`.

This backward-compatible patch release changes no model weights, embedding
outputs, or EmbeddingGemma 2 MRL behavior.

### Fixed

- Resolve omitted batch embedding dimensions from the selected backend so
  dino.txt uses 2048 dimensions by default while EmbeddingGemma 2 retains its
  768-dimensional default and MRL options.
- Add opt-in SHA-256 image-content verification for incremental index updates,
  including safe re-embedding of legacy entries without stored hashes.
- Calculate skipped video sampling intervals with exact `Fraction` arithmetic
  rather than iterating once per missed interval.

See the [v0.6.1 release notes](docs/release-notes-v0.6.1.md).

## v0.6.0 — 2026-10-22

Released on [PyPI](https://pypi.org/project/openvlclip/0.6.0/) and as a
[GitHub Release](https://github.com/kotadymo/openvlclip/releases/tag/v0.6.0).

Install the base package with `pip install openvlclip==0.6.0`, or include the
optional dino.txt dependencies with
`pip install "openvlclip[dinotxt]==0.6.0"`.

### Added

- Register Meta DINOv2 ViT-L/14 with dino.txt vision and text heads as
  `dinov2-dinotxt`, alongside the existing EmbeddingGemma 2 backend.
- Add native 2048-dimensional dino.txt embeddings, official-scale logits,
  cosine scoring, image search, 1024-dimensional patch embeddings, and text
  similarity maps.
- Add backend-specific model capabilities and embedding-space metadata to keep
  dino.txt, EmbeddingGemma MRL, and legacy indexes from being mixed.
- Add optional `[dinotxt]` dependencies and the v0.6.0 migration, evaluation,
  and release-checklist documentation.

### Compatibility

- Preserve EmbeddingGemma 2 factories, prompts, preprocessing, selective
  loading, MRL dimensions, image/video search, and legacy index reads.
- dino.txt does not support MRL truncation, EmbeddingGemma task prompts, or
  selective loading; its checkpoints remain external downloads and are not
  included in distributions.
- Meta's pinned DINOv2 repository license and model card identify Apache-2.0
  for the repository and DINOv2 model family. The individual license terms for
  the external dino.txt Vision Head and Text Encoder checkpoints are not
  separately identified in the pinned repository; the public release does
  not resolve that scope question. See the [dino.txt reference](docs/dinotxt-v0.6.0.md).

### Validation

- `pytest -q`: 304 passed, 30 skipped, 3 warnings. The clean-wheel real-model
  suite had 16 passes and 11 CUDA-only skips on a CPU-only environment.
- CPU FP32/BF16 dino.txt parity against the pinned official implementation
  had 0.0 maximum absolute error for the compared outputs. The v0.4.0
  EmbeddingGemma image/text baseline across MRL 768/512/256/128 had maximum
  absolute error no greater than `1.27e-7`.
- Dataset-based accuracy, retrieval quality, and comparative GPU performance
  remain not evaluated. See [the v0.6.0 evaluation report](docs/benchmark-v0.6.0.md).

## v0.5.0

### Added

- Introduce the `openvlclip` distribution and import namespace while retaining
  the existing EmbeddingGemma 2 inference API.
- Add a model Registry, generic Backend interface, model capabilities, and
  backend-aware embedding-space compatibility checks.
- Add migration guidance for users moving from `openclip-gemma` v0.4.0.

### Compatibility

- Keep EmbeddingGemma 2 as the only supported real model; no model weights,
  prompts, preprocessing, pooling, MRL behavior, or selective-loading behavior
  are changed.
- Preserve existing image and video index formats and v0.4.0 index loading.
- Keep `openclip-gemma` as a separate distribution; installing `openvlclip`
  does not replace or uninstall it.

## v0.4.0

### Added

- Add official Matryoshka embedding dimensions of 768, 512, 256, and 128 to
  image/text encoding, cosine APIs, image/video indexing, and search.
- Add safe, no-inference image/video index conversion to lower dimensions while
  preserving legacy 768-dimensional index reads.
- Add opt-in text-only and text-plus-image factory loading by omitting
  unrequested Transformers vision/audio configs.
- Add an MRL benchmark harness for encoder/search latency, measured index and
  vector storage, memory, ranking agreement, and optional Recall@K from labeled
  JSON query manifests.

### Compatibility

- Preserve the default 768-dimensional outputs, default full model loading,
  strict `forward()`/`get_logits()` behavior, preprocessing, and tokenizer.
- Introduce image index format v2 and video index format v3. Legacy image v1
  and video v2 archives remain readable only at 768 dimensions; video v1
  remains unsupported.

### Validation

- See `docs/v0.4.0-development-progress.md` for executed test, build, model,
  and release-validation results. Shape and normalization tests do not
  establish application-specific retrieval quality.

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
- CPU BF16 `get_logits()` accumulates its matrix product in FP32 and casts the
  result back to BF16 to support processors without native BF16 matrix
  instructions; the CUDA computation path is unchanged.
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
