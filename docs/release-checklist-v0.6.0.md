# OpenVLCLIP v0.6.0 Release Checklist

This checklist prepares the direct `0.6.0` release path. It intentionally
does not use TestPyPI. Creating release tags, publishing to PyPI, or creating
a GitHub Release is outside the current preparation step.

## Blocking prerequisites

- [ ] Resolve and record explicit licensing/redistribution rights for the
  dino.txt Vision Head and Text Encoder checkpoints. The DINOv2 repository
  license does not conclusively establish rights for these separate assets.
- [ ] Only after that review, set the GitHub repository variable
  `OPENVLCLIP_DINOTXT_LICENSE_CONFIRMED` to the exact value `true`. The
  `publish-pypi-v0.6.0.yml` build job fails closed without it.
- [ ] Confirm the final `0.6.0` source and public docs do not contain
  unresolved license claims or unverified accuracy/performance claims.

## Source, CI, and distribution checks

- [ ] `pyproject.toml` declares `openvlclip==0.6.0`; `uv lock --check` passes.
- [ ] Ruff and the normal pytest suite pass.
- [ ] EmbeddingGemma v0.4.0 fixed-embedding, MRL, prompt, selective-loading,
  image/video-search, conversion, and legacy-index tests pass. Do not
  regenerate or modify the fixed `.npz` baseline.
- [ ] dino.txt tokenization, preprocessing, FP32/BF16 parity, official logits,
  cosine scoring, image search, patch features, and similarity-map tests pass.
- [ ] Record CUDA cases as untested when no compatible device/runtime is
  available; a skipped test is not a pass.
- [ ] Build wheel and sdist; run `tools/verify_wheel.py` and
  `tools/verify_sdist.py` with `--expected-version 0.6.0`.
- [ ] Confirm the distribution excludes model weights, includes the DINO
  backend/tokenizer, declares the `dinotxt` extra, and identifies the
  `kotadymo/openvlclip` repository.
- [ ] Install the wheel in a clean Python 3.12 environment and test both
  `[dinotxt]` and `[video]` dependency resolution without importing the
  repository checkout.
- [ ] Confirm the same main commit's GitHub Actions CI has completed
  successfully and required reviews/protection rules are satisfied.

## GitHub and PyPI publishing configuration

- [ ] Configure the PyPI Trusted Publisher for:
  - Owner: `kotadymo`
  - Repository: `openvlclip`
  - Workflow filename: `publish-pypi-v0.6.0.yml`
  - GitHub Environment: `pypi`
- [ ] Add `v0.6.0` to the `pypi` GitHub Environment tag allowlist without
  weakening any protection or approval rule.
- [ ] Confirm the `pypi` Environment uses the intended reviewers and
  deployment protection.
- [ ] Do not create a TestPyPI publisher for this release workflow.

After configuration, inspect the Environment in GitHub Settings and the
Trusted Publisher configuration in the PyPI project settings. Confirm the
repository, workflow filename, and Environment match exactly before tagging.

## Workflow guarantees

`.github/workflows/publish-pypi-v0.6.0.yml` is tag-triggered only for
`v0.6.0`. It must verify the exact tag/commit, main ancestry, a successful CI
run for that commit, the project version, and the lockfile; run unit tests,
Ruff, wheel/sdist build and content verification; and publish only with
Trusted Publishing from the `pypi` Environment. `id-token: write` belongs
only to the publish job. The existing v0.5.0 workflow must remain unchanged.

## Release sequence after every prerequisite is satisfied

1. Verify the current main SHA and its successful CI run:

   ```powershell
   git fetch origin main --tags
   git switch main
   git pull --ff-only origin main
   git rev-parse HEAD
   gh run list --repo kotadymo/openvlclip --workflow ci.yml --commit (git rev-parse HEAD) --status completed
   ```

2. Confirm `v0.6.0` does not already exist, then create and push it to that
   verified main commit. Never overwrite or recreate an existing tag:

   ```powershell
   git show-ref --verify refs/tags/v0.6.0
   git tag v0.6.0 (git rev-parse HEAD)
   git push origin v0.6.0
   ```

   A nonzero `git show-ref` result means the tag is absent; inspect the
   repository before proceeding. Do not use force-push.

3. Monitor the tag-triggered Actions run. If `pypi` approval is required,
   approve only through the normal GitHub Environment UI. Verify the PyPI
   wheel and sdist, their SHA-256 values, metadata, clean-install behavior,
   and required real-model regressions before creating a public GitHub
   Release for the same tag.

## Current preparation boundary

At this preparation stage, the dino.txt checkpoint license gate is
unresolved. Therefore v0.6.0 is **not release-ready**. Do not set the license
confirmation variable, create `v0.6.0` or RC tags, run a publish workflow,
upload to PyPI/TestPyPI, or create a GitHub Release until the blocker and all
required validations are resolved.
