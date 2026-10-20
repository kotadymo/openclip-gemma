# v0.4.0 Release Checklist

This checklist preserves the repository's review, CI, GitHub Environment, and
PyPI Trusted Publishing gates. A local build, branch, or tag alone does not
authorize a package publication.

## 1. Implementation and pull request

- [x] Complete model MRL, selective loading, index migration/conversion, CLI,
  benchmark, and documentation changes.
- [x] Address the four PR review findings: CPU-staged image batches,
  specification-compliant benchmark metrics and formats, selective-loading
  integration coverage, and embedding-strategy metadata validation.
- [x] Address five findings from the follow-up Copilot review: measure RSS
  directly after model loading, include run and environment metadata in CSV,
  align migration/specification text with current APIs, and reserve conversion
  destinations atomically.
- [x] Align the changelog's benchmark summary with the JSON query-manifest
  input and measured storage, memory, ranking, and recall outputs.
- [x] Pass the complete non-integration pytest suite, Ruff, locked dependency
  check, wheel/sdist build, artifact verification, and clean wheel import.
- [x] Validate the four follow-up review fixes with focused/full tests,
  selective-loading integration tests, benchmark smoke coverage, and rebuilt
  wheel/sdist artifacts.
- [x] Run available CPU pinned-model MRL comparisons and record exact results.
  CUDA MRL and selective-loading integration checks also passed for FP32 and
  BF16 on an NVIDIA RTX 4070 Ti SUPER using `.venv-cuda`; broader platform and
  application-data retrieval-quality checks remain separate.
- [x] Create the implementation PR from the v0.4.0 feature branch.
- [x] Pass CI for implementation commit `5880286` and documentation/status
  commits through `768365b` on Windows and Ubuntu with Python 3.12, 3.13, and
  3.14. The manual CPU integration jobs were skipped by workflow
  configuration.
- [x] Re-run Copilot Code Review on code/changelog commit `560e621`; the final
  review found no open issues, and all review threads are resolved.
- [x] Inspect GitHub branch protection and repository rulesets. Neither
  required an independent maintainer approval; PR #21 was merged only after
  CI passed and all review threads were resolved, under the user's explicit
  merge authorization.
- [x] Merge PR #21 at `ca3c3a4aeb524aae99d6737e70ba3e30c8dbcf33` and RC
  preparation PR #22 at `6e0058898f0d07bd08921b6bef75bb29c5c924de`; CI passed
  for both exact main commits.

## 2. TestPyPI release candidate

- [x] Register and verify the TestPyPI Trusted Publisher for project `openclip-gemma`,
  owner `kotadymo`, repository `openclip-gemma`, workflow
  `.github/workflows/publish-testpypi-v0.4.0rc1.yml`, and environment
  `testpypi`; Trusted Publishing succeeded in workflow run `39431435350`.
- [x] Confirm the workflow accepts only tag `v0.4.0rc1`, checks version
  `0.4.0rc1`, and requires successful CI for the exact tagged commit.
- [x] Confirm the `testpypi` Environment gate was satisfied by the successful
  publish run without weakening or bypassing its protection.
- [x] Change the package version to `0.4.0rc1` through reviewed PR #22; lock,
  test, lint, build, and artifact checks passed.
- [x] Tag only the verified main commit as `v0.4.0rc1`; the tag points to
  `6e0058898f0d07bd08921b6bef75bb29c5c924de`.
- [x] Verify the publish job completed and both the wheel and sdist are
  visible on TestPyPI at version `0.4.0rc1`.

## 3. Published candidate validation

- [x] Download the candidate artifact from TestPyPI without dependencies, then
  install dependencies separately from their intended sources in a clean
  environment. Do not mix package indexes with `--extra-index-url`.
- [x] Verify installed metadata is exactly `0.4.0rc1` and imports resolve from
  the clean environment rather than the repository.
- [x] Test default 768 and MRL 512/256/128 image and text outputs; verify
  shapes, dtypes, devices, finiteness, unit norms, and matching-dimension
  cosine calculations.
- [x] Test text-only loading, image-capable loading, index v1/v2 and video
  v2/v3 compatibility, dimension mismatch rejection, conversion, and search.
- [x] Test the `video` extra independently and retain a base-install check
  without PyAV.
- [x] Keep synthetic functional checks distinct from retrieval-quality
  measurements on labeled, representative data.

## 4. Formal PyPI release and GitHub Release

- [x] No release-blocking issue was found in the candidate. Any future
  candidate fix must use a new RC version rather than replace an uploaded
  artifact.
- [ ] Prepare the formal `0.4.0` version change through normal review and
  successful CI on the exact main commit.
- [x] The user confirmed registration of the PyPI Trusted Publisher for
  project `openclip-gemma`, owner `kotadymo`, repository `openclip-gemma`,
  workflow `.github/workflows/publish-pypi-v0.4.0.yml`, and environment
  `pypi`; the production workflow must still verify the registration.
- [x] Confirm the configured `pypi` Environment reviewer is `kotadymo`.
  Deployment approval is still required before publication; do not bypass or
  weaken protections.
- [x] Verify the production workflow accepts only `v0.4.0`, checks package
  version `0.4.0`, and requires successful CI for the exact main commit.
- [ ] Tag and push only after authorization and all release gates pass.
- [ ] Verify the PyPI publish job completed, both artifacts are available,
  clean installation reports version `0.4.0`, and the GitHub Release points
  to the verified tag and commit.

## Workflow references

- TestPyPI: `.github/workflows/publish-testpypi-v0.4.0rc1.yml`
- PyPI: `.github/workflows/publish-pypi-v0.4.0.yml`
- Existing workflow protections and Trusted Publisher registrations must be
  confirmed for the new versioned workflow names before tagging.
