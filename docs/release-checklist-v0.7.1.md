# OpenVLCLIP v0.7.1 Release Checklist

**Status (2026-10-23):** Release preparation is in progress. Do not create the
tag or publish until every pre-release gate below passes. Qwen3-VL-Embedding-8B
is outside this release.

## Scope

- [x] Keep the five v0.7.0 checkpoints and their embedding behavior unchanged.
- [x] Add opt-in SHA-256 content verification for Native Video Index updates.
- [x] Expose and enforce Qwen3-VL-Embedding-2B's 64-frame native-video limit.
- [x] Preserve native-video index format version 1 and support hashless v0.7.0
  records.
- [x] Update README release-state checks and distribution-content verification.

## Source and local validation

- [x] Fix PR [#51](https://github.com/kotadymo/openvlclip/pull/51) merged to
  `main` at `b63f0e3ee5321edb4b6d865daedec6319f75e84f`.
- [x] Merged-main CI passed for that exact SHA:
  [run 39953334852](https://github.com/kotadymo/openvlclip/actions/runs/39953334852).
- [x] Regression tests reproduce same-size/same-mtime content changes, legacy
  hashless index migration, frame-limit rejection before scanning, search-time
  limit enforcement, and the public Qwen limit property.
- [x] Local suite: 391 passed, 40 skipped, 3 warnings.
- [x] `ruff check .`, `uv lock --check`, and `git diff --check` passed before
  the release version bump.
- [x] A v0.7.0 wheel and sdist built from the fix branch passed their content
  verifiers, including the new backend-module and README checks.
- [x] Confirm `pyproject.toml` and `uv.lock` both identify `0.7.1`.
- [x] Release-preparation validation: 391 passed, 40 skipped, 3 warnings;
  Ruff, `uv lock --check`, and `git diff --check` passed.
- [x] Build the v0.7.1 wheel and sdist and verify archive contents. Installed
  the wheel in a clean Python 3.12 environment; import and version checks
  passed, `list_models()` returned the five supported models without Qwen 8B,
  and the installed Native Video Index API exposes `verify_contents`.
- [ ] Confirm release-preparation CI succeeds on Windows and Linux with Python
  3.12, 3.13, and 3.14.

## Numerical and runtime scope

- No model, tokenizer, preprocessing, pooling, normalization, logit, or MRL
  calculations changed in this patch.
- Existing EmbeddingGemma, dino.txt, SigLIP 2, and Qwen 2B model-parity evidence
  remains in the [v0.7.0 model provenance report](model-provenance-v0.7.0.md);
  this release does not claim new numerical parity measurements.
- CUDA is unavailable in the release-preparation environment. No new CUDA
  result is claimed.
- Strict SHA-256 verification is opt-in and reads every video on each update;
  the default fast size/mtime behavior is unchanged.
- Qwen 8B is not registered or supported.

## Trusted Publishing and release gates

- [x] Existing shared workflow `.github/workflows/publish-pypi.yml` is reused;
  no new Trusted Publisher is required.
- [x] Repository variables
  `OPENVLCLIP_DINOTXT_LICENSE_CONFIRMED` and
  `OPENVLCLIP_V070_MODELS_LICENSE_CONFIRMED` were confirmed `true`.
- [x] The existing `pypi` Environment has one required reviewer; its protections
  must remain enabled.
- [ ] Add `v0.7.1` to the existing Environment's tag allowlist without
  changing reviewers or other policies.
- [ ] Confirm the shared workflow's exact tag, commit, main ancestry, same-SHA
  CI, lock, wheel/sdist, license, and Trusted Publishing checks.
- [ ] Immediately before tagging, confirm `v0.7.1` does not already exist as a
  tag or GitHub Release and that PyPI has no `openvlclip==0.7.1`.
- [ ] Create and push an annotated `v0.7.1` tag at the exact CI-successful
  `main` commit without force or overwrite.
- [ ] If the `pypi` Environment requests approval, wait for the authorized
  reviewer; do not bypass deployment protections.
- [ ] Verify the publishing workflow succeeds and record wheel/sdist filenames
  and SHA-256 values against PyPI JSON metadata.
- [ ] Install the published wheel in a clean Python 3.12 environment and verify
  package version, imports, five-model registry, and absence of Qwen 8B.
- [ ] Publish the formal GitHub Release only after PyPI artifact verification.

## Publication record

- PyPI: <https://pypi.org/project/openvlclip/0.7.1/>
- GitHub Release: <https://github.com/kotadymo/openvlclip/releases/tag/v0.7.1>
- Main/tag commit: pending
- Main CI: pending
- PyPI publishing workflow: pending
- Wheel filename / SHA-256: pending
- Sdist filename / SHA-256: pending
- Clean-install verification: pending
