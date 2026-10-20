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
- [x] Pass CI for latest PR head `89f8e2f` on Windows and Ubuntu with Python
  3.12, 3.13, and 3.14. The manual CPU integration jobs were skipped by
  workflow configuration.
- [x] Re-run Copilot Code Review on code/changelog commit `560e621`; the final
  review found no open issues, and all review threads are resolved.
- [ ] Obtain normal independent maintainer review and approval before merging
  to `main`.
- [ ] Record the merge SHA and confirm it is reachable from `main`.

## 2. TestPyPI release candidate

- [ ] Register the TestPyPI Trusted Publisher for project `openclip-gemma`,
  owner `kotadymo`, repository `openclip-gemma`, workflow
  `.github/workflows/publish-testpypi-v0.4.0rc1.yml`, and environment
  `testpypi`; registration has not been independently confirmed.
- [ ] Confirm the workflow accepts only tag `v0.4.0rc1`, checks version
  `0.4.0rc1`, and requires successful CI for the exact tagged commit.
- [ ] Confirm and satisfy the `testpypi` Environment's configured protection;
  its custom branch policy is enabled but currently has no configured
  branch/tag patterns. Do not bypass or weaken environment rules.
- [ ] Change the package version to `0.4.0rc1` through a reviewed release
  preparation change, then repeat lock, test, lint, build, and package checks.
- [ ] Tag only the verified main commit after the candidate release is
  authorized. Never overwrite an existing TestPyPI version.
- [ ] Verify the publish job completed and both wheel and sdist are visible on
  TestPyPI; a successful build job alone is not publication evidence.

## 3. Published candidate validation

- [ ] Download the candidate artifact from TestPyPI without dependencies, then
  install dependencies separately from their intended sources in a clean
  environment. Do not mix package indexes with `--extra-index-url`.
- [ ] Verify installed metadata is exactly `0.4.0rc1` and imports resolve from
  the clean environment rather than the repository.
- [ ] Test default 768 and MRL 512/256/128 image and text outputs; verify
  shapes, dtypes, devices, finiteness, unit norms, and matching-dimension
  cosine calculations.
- [ ] Test text-only loading, image-capable loading, index v1/v2 and video
  v2/v3 compatibility, dimension mismatch rejection, conversion, and search.
- [ ] Test the `video` extra independently and retain a base-install check
  without PyAV.
- [ ] Keep synthetic functional checks distinct from retrieval-quality
  measurements on labeled, representative data.

## 4. Formal PyPI release and GitHub Release

- [ ] Resolve release-candidate issues with a new candidate version rather
  than replacing an uploaded artifact.
- [ ] Prepare the formal `0.4.0` version change through normal review and
  successful CI on the exact main commit.
- [ ] Register the PyPI Trusted Publisher for project `openclip-gemma`,
  owner `kotadymo`, repository `openclip-gemma`, workflow
  `.github/workflows/publish-pypi-v0.4.0.yml`, and environment `pypi`;
  registration has not been independently confirmed.
- [ ] Confirm `pypi` Environment reviewer approval and every other configured
  protection is available. Its current required reviewer is `kotadymo`. Do not
  bypass or weaken these controls.
- [ ] Verify the production workflow accepts only `v0.4.0`, checks package
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
