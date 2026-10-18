# v0.3.0 Release Checklist

This checklist describes a future release procedure. It does not authorize a
merge, tag, TestPyPI/PyPI upload, or GitHub Release by itself.

## 1. Complete and merge the implementation

- [ ] Review and approve Draft PR #18.
- [ ] Confirm the PR head's Windows/Linux Python 3.12-3.14 CI, video-extra
  tests, ordinary tests, lint, and wheel/sdist checks are successful.
- [ ] Merge PR #18 to `main` only after explicit approval.
- [ ] Record the merge commit SHA and confirm it is the commit intended for the
  release preparation.

## 2. Prepare the v0.3.0 release candidate

- [ ] Create a release-preparation branch from the latest `main`.
- [ ] Change `project.version` in `pyproject.toml` to `0.3.0rc1`.
- [ ] Run `uv lock`, then `uv lock --check`; review that runtime dependencies
  and Python support constraints did not change unintentionally.
- [ ] Prepare the v0.3.0 changelog entry. Keep the README free of time-sensitive
  claims such as "unreleased" or "current version".
- [ ] Run the ordinary test suite, Ruff, wheel/sdist builds, and the artifact
  verification scripts. Do not weaken thresholds or add skips to make a check
  pass.
- [ ] Merge the release-candidate version change through the repository's
  normal review process. The candidate tag must point to a commit reachable
  from `main`.

## 3. Configure TestPyPI Trusted Publishing and verify main CI

- [ ] Before tagging, register a TestPyPI Trusted Publisher for project
  `openclip-gemma`, GitHub owner `kotadymo`, repository `openclip-gemma`,
  workflow `.github/workflows/publish-testpypi-v0.3.0rc1.yml`, and GitHub
  Environment `testpypi`.
- [ ] Confirm that the workflow is tag-triggered only for `v0.3.0rc1`, expects
  version `0.3.0rc1`, and grants `id-token: write` only to its publish job.
- [ ] Confirm the exact candidate commit is on `main` and the CI workflow has a
  successful run for that exact SHA. Resolve any newer commits or changed
  release files before choosing the tag target.
- [ ] After explicit release approval, create and push `v0.3.0rc1` at that
  exact main commit. Do not use a tag on an unmerged feature or release branch.
- [ ] Verify that the `publish-testpypi-v0.3.0rc1.yml` run builds and validates
  both distributions and completes its TestPyPI publication. A successful
  build job alone is not evidence that publication completed.

## 4. Install and test the published release candidate

Use separate clean CPU and CUDA environments when testing different PyTorch
builds. Do not mix TestPyPI and PyPI using `--extra-index-url`: download the
project artifact from TestPyPI without dependencies, then install its
dependencies from their intended sources.

Example Windows PowerShell steps for the project artifact:

```powershell
python -m venv .venv-openclip-gemma-rc
.\.venv-openclip-gemma-rc\Scripts\python.exe -m pip download `
  --no-deps `
  --index-url https://test.pypi.org/simple `
  --dest .\testpypi-dist `
  openclip-gemma==0.3.0rc1
```

Install dependencies separately from the normal PyPI index. For a CPU test,
for example:

```powershell
.\.venv-openclip-gemma-rc\Scripts\python.exe -m pip install `
  --index-url https://pypi.org/simple `
  "numpy>=1.26,<3" "pillow>=13,<14" `
  "torch>=2.14.1,<3" "torchvision>=0.29.1,<1" `
  "transformers>=5.20.1,<6" "av>=19.0.1,<20"
.\.venv-openclip-gemma-rc\Scripts\python.exe -m pip install `
  --no-deps .\testpypi-dist\openclip_gemma-0.3.0rc1-py3-none-any.whl
.\.venv-openclip-gemma-rc\Scripts\python.exe -m pip check
```

The `av` requirement is optional and is included above for video testing; omit
it for a base-install check. For CUDA testing, install a compatible
PyTorch/torchvision build from the official PyTorch package source in its own
clean environment, then install the remaining dependencies from PyPI. Avoid
`--extra-index-url` so the package artifact and its dependencies cannot be
selected ambiguously across indexes.

- [ ] Verify the installed version is exactly `0.3.0rc1` and the import path is
  in the clean virtual environment, not the repository or an editable install.
- [ ] Import all seven top-level exports and the batching, image-search, and
  video-search modules.
- [ ] Run CPU image and Japanese/English text inference with the pinned model.
- [ ] Exercise batch image embeddings, image-index save/load/search, video
  frame extraction, video-index save/load/search, and timestamp handling with
  deterministic fixtures.
- [ ] Repeat supported operations on CUDA FP32 and BF16 where the GPU supports
  them. Identify these as hardware tests, separate from CPU CI.
- [ ] Confirm the video extra installs PyAV and that the base installation
  remains usable without it.
- [ ] Record actual failures, skips, dtype/device, numerical comparison
  metrics, and available memory measurements. Keep synthetic functional
  checks distinct from retrieval-quality evaluation on real media.

## 5. Approve the release candidate

- [ ] Review the published wheel and sdist metadata and contents.
- [ ] Resolve release-blocking issues and decide explicitly whether the RC is
  acceptable. If code or metadata changes, repeat the applicable checks and
  publish a new RC version rather than overwriting `0.3.0rc1`.
- [ ] Obtain explicit approval before beginning the formal release.

## 6. Prepare and verify the formal v0.3.0 version

- [ ] Start from the latest `main` containing the accepted RC and any required
  fixes.
- [ ] Change `project.version` to `0.3.0`, run `uv lock`, and verify
  `uv lock --check`.
- [ ] Finalize the v0.3.0 changelog entry and review the README, API examples,
  dependency metadata, and release artifacts.
- [ ] Run the ordinary and applicable integration tests, Ruff, actionlint,
  wheel/sdist builds, and both artifact verification scripts.
- [ ] Merge the formal-version change using the usual review process. Confirm
  the exact release commit is reachable from `main` and has a successful CI
  run for that SHA.
- [ ] Register the PyPI Trusted Publisher for project `openclip-gemma`, GitHub
  owner `kotadymo`, repository `openclip-gemma`, workflow
  `.github/workflows/publish-pypi-v0.3.0.yml`, and Environment `pypi`.
- [ ] Verify the production workflow is limited to tag `v0.3.0`, expects
  version `0.3.0`, and uses the `pypi` environment.
- [ ] Only after explicit approval, create and push `v0.3.0` at the verified
  main commit.
- [ ] Confirm the production workflow completes the PyPI upload, then verify
  the PyPI project page, wheel/sdist downloads, installed package metadata,
  and GitHub Release. Report any discrepancy rather than editing the README
  to compensate for an artifact mismatch.

## Release workflow references

- TestPyPI RC: `.github/workflows/publish-testpypi-v0.3.0rc1.yml`
- Production: `.github/workflows/publish-pypi-v0.3.0.yml`
- Historical v0.2.0 workflow:
  `.github/workflows/publish-pypi-v0.2.0.yml` (reference only; do not modify)
- [PyPI Trusted Publishers: adding a publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
