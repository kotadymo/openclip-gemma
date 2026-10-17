# v0.1.0 Release Checklist

This checklist is a release gate, not evidence that a release has happened. Do not upload to PyPI/TestPyPI or create a GitHub Release until the remaining gates are cleared.

## Package and documentation

- [x] Distribution name `openclip-gemma`, import name `openclip_gemma`, and version `0.1.0` are declared in `pyproject.toml`.
- [x] Python is restricted to the validated initial target, 3.12.
- [x] Windows x64 is the only declared/CI OS target; Linux and macOS are not yet claimed.
- [x] Runtime dependency lower bounds use versions exercised in the local environment; upper major bounds avoid unreviewed major upgrades.
- [x] Runtime requirements are limited to Pillow, PyTorch, Transformers, and torchvision; Sentence Transformers is an optional integration-test dependency.
- [x] The model ID and revision are fixed in the package; no model weights are packaged.
- [x] README examples use the public factory, tokenizer, and encoder APIs and explain their limits.
- [x] Compatibility claims and unsupported APIs are documented.
- [x] The source license is Apache-2.0. The EmbeddingGemma 2 model card specifies Apache-2.0 separately; its weights are not bundled.
- [x] No OpenCLIP implementation source is copied into the package. OpenCLIP is referenced as the compatible API target and is separately MIT-licensed.
- [x] The pinned EmbeddingGemma 2 model card was checked directly: its metadata states Apache-2.0 and links Google's Gemma 4 license.
- [x] Reviewed direct runtime/optional dependency license files: PyTorch and torchvision are BSD-3-Clause/BSD-style; Transformers and Sentence Transformers are Apache-2.0; Pillow is MIT-CMU; psutil is BSD-3-Clause; pytest is MIT; Ruff is MIT. These libraries are installed as separate dependencies, not copied into this source package.
- [x] PyPI availability checked on 2026-10-17: `https://pypi.org/pypi/openclip-gemma/json` returned HTTP 404. Recheck immediately before release; name availability can change.
- [ ] Confirm the copyright holder and authority to publish under Apache-2.0.
- [x] Check the exact pinned model revision's license; recheck its usage terms at release time.

License sources:

- [EmbeddingGemma 2 pinned model card](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b)
- [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE), [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE), [Sentence Transformers](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE), [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE)
- [psutil](https://github.com/giampaolo/psutil/blob/master/LICENSE), [pytest](https://github.com/pytest-dev/pytest/blob/main/LICENSE), [Ruff](https://github.com/astral-sh/ruff/blob/main/LICENSE), [OpenCLIP](https://github.com/mlfoundations/open_clip/blob/main/LICENSE)

## Automated validation

- [x] `uv sync`
- [x] `uv run pytest` — 75 passed, 9 skipped (the opt-in model tests).
- [x] `uv run ruff check .` — passed.
- [x] `uv build` created both sdist and wheel.
- [x] Optional integration dependencies install; Sentence Transformers 6.1.0 resolved.
- [x] Wheel and sdist inspections confirmed package/release files only; no tests, experiments, examples, or model weights.
- [x] Installed the wheel and its dependencies into a fresh Python 3.12 virtual environment; import and public exports passed.
- [x] Executed the README basic embedding, task-prompt/Japanese tokenizer, and zero-shot demo examples with the pinned cached model.
- [ ] GitHub Actions CI has completed successfully on the target branch.
- [x] Targeted real-model wrapper/preprocessing integration passed locally: 1 passed for the pinned revision.
- [ ] Run the complete manual reference-integration workflow, or explicitly accept the remaining CUDA limitation.

## Numerical and platform validation

- [x] CPU FP32 image/text/zero-shot comparisons passed the provisional thresholds for the pinned revision.
- [x] CPU BF16 was measured separately; results are environment-specific and do not establish a portable tolerance.
- [ ] CUDA image and text integration tests pass on a documented CUDA/PyTorch combination, or CUDA is removed from the v0.1.0 supported-device claim.
- [ ] Decide whether CPU-only validation is sufficient for the v0.1.0 P0 contract.
- [x] CPU load, inference latency, and sampled peak RSS were measured with the standalone benchmark.
- [x] Documentation does not claim practical memory support for 8192-token sequences; measure before making that claim.

### CPU benchmark observation

Recorded in [benchmark-cpu-v0.1.0.json](./benchmark-cpu-v0.1.0.json). The run used the cached pinned weights, Windows 11, Python 3.12.10, PyTorch 2.14.1+cpu, FP32, CPU, 20 Torch threads, a 256x192 generated image (preprocessed to 672x912), four short texts, two warmups, and five measurements per inference operation.

| Measurement | Observed |
|---|---:|
| Model and processor load from cache | 4.99 s |
| Tokenizer and config load | 2.76 s |
| Single image encoder median | 5.48 s |
| Two-image batch median | 10.24 s |
| Single text median | 76.5 ms |
| Four-text batch median | 116.7 ms |
| Sampled peak process RSS | 4.27 GB |
| RSS before load | 0.34 GB |

Inference latency excludes image preprocessing and tokenization. RSS is sampled every 20 ms and may miss short-lived peaks. These are observations from one Windows CPU, not performance guarantees; the relatively high memory and image latency should be considered when documenting deployment requirements.

## Release actions (only after approval)

- [ ] Recheck PyPI name availability and reserve/publish the package.
- [ ] Confirm GitHub repository visibility, tags, release notes, and CI status.
- [ ] Build the release artifacts from a clean checkout and verify their checksums and metadata.
- [ ] Publish to PyPI only after explicit release approval.
- [ ] Create a GitHub Release only after explicit release approval.
