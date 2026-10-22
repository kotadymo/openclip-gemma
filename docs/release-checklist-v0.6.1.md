# OpenVLCLIP v0.6.1 Release Checklist

This is a patch release. Keep the existing Trusted Publisher workflow filename
and `pypi` Environment; do not use TestPyPI.

## Source and validation

- [x] Fixes merged to `main` in [PR #36](https://github.com/kotadymo/openvlclip/pull/36).
- [x] CI passed on the fix merge commit `ffb1999c75c1e9829287698e93420c0728ea4255`.
- [x] Set `pyproject.toml` to `0.6.1` and synchronize `uv.lock`.
- [x] Run pytest, Ruff, lock validation, and targeted regression tests.
- [x] Build and inspect the wheel and sdist.
- [x] Install the wheel in a clean Python 3.12 environment and run applicable real-model checks.

### Pre-release verification evidence

- `uv lock --check`, `ruff check .`, and `git diff --check` passed.
- Default suite: 314 passed, 30 skipped, 3 warnings. The focused release
  workflow tests passed (2 tests).
- The built wheel and sdist passed `tools/verify_wheel.py` and
  `tools/verify_sdist.py`; their sizes were 74,582 and 75,599 bytes.
- A clean Python 3.12.10 environment installed the local wheel with
  `[dinotxt]`. `importlib.metadata.version("openvlclip")` returned `0.6.1`,
  and `openvlclip.__file__` resolved inside that environment's
  `site-packages`.
- Tests run against the installed wheel: 95 passed for batching, image/video
  search, v0.4.0 MRL baseline, and dino.txt official parity; 2 CPU
  selective-loading integration cases passed. CUDA-only cases were skipped.
- The v0.4.0 CPU FP32 image/text baseline across MRL 768/512/256/128 had
  maximum absolute errors of `1.27e-7` (image) and `1.20e-7` (text).
  dino.txt CPU FP32/BF16 parity had maximum absolute error `0.0` for the
  compared embeddings, logits, patch features, and similarity maps.
- The wheel's real dino.txt batch API returned a finite 2048-dimensional
  FP32 embedding when `embedding_dim` was omitted.
- CUDA is unavailable in this environment (`torch 2.14.1+cpu`); CUDA
  FP32/BF16 is not validated here.

## Trusted Publishing safeguards

- [x] Update `.github/workflows/publish-pypi-v0.6.0.yml` in place; retain that exact filename.
- [x] Restrict the workflow trigger and version checks to tag `v0.6.1` and version `0.6.1`.
- [x] Preserve exact tag/commit validation, `main` ancestry, and successful same-SHA `main` CI verification.
- [x] Preserve wheel/sdist verification, the dino.txt license gate, the `pypi` Environment, and Trusted Publishing.
- [x] Confirm only the publish job has `id-token: write`; do not alter the v0.6.0 tag or published files.
- [ ] Add `v0.6.1` to the existing `pypi` Environment tag allowlist without changing reviewers or other protections.
- [x] Confirm `OPENVLCLIP_DINOTXT_LICENSE_CONFIRMED=true`.

## Final release gates

- [ ] Confirm `v0.6.1` does not exist as a Git tag or GitHub Release and `openvlclip==0.6.1` is not on PyPI.
- [ ] Merge the release-preparation PR and confirm main CI passes on the release commit.
- [ ] Create and push `v0.6.1` at that exact main commit; do not overwrite any tag.
- [ ] Complete any required `pypi` Environment approval; do not bypass protections.
- [ ] Verify the PyPI wheel and sdist SHA-256 values and install the published wheel in a clean environment.
- [ ] Run published-wheel API and real-model regression checks; record CUDA as unverified if unavailable.
- [ ] Create the public `OpenVLCLIP v0.6.1` GitHub Release only after PyPI verification.
