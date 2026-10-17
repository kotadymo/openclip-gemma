# v0.1.1 Release Checklist

This checklist is a release gate, not evidence that a release has happened.
No v0.1.1 package, tag, or GitHub Release has been published or created.

## Current preparation status

- Release branch: `release/v0.1.1`
- Base commit: `5c861a7b0f696164c7c8b3a628540b496280ee06`
- Base CI run: [39041843535](https://github.com/kotadymo/openclip-gemma/actions/runs/39041843535);
  all six Windows/Ubuntu × Python 3.12/3.13/3.14 quality jobs passed.
- This run predates the version bump. The final release commit must have its
  own successful CI run before it can be tagged.
- CUDA hardware inference, Linux real-model inference, and macOS remain
  unverified. Ubuntu CI success covers quality/build/install checks only.

## Package, compatibility, and documentation

- [x] Distribution name is `openclip-gemma`; import name is `openclip_gemma`.
- [x] Version is set to `0.1.1` in `pyproject.toml`; `uv.lock` is synchronized.
- [x] Python range is `>=3.12,<3.15,!=3.14.1`. Python 3.14.1 remains excluded
  because torchvision 0.29.1 rejects it.
- [x] README and changelog describe v0.1.1 without suggesting it is already
  published.
- [x] The Python compatibility report distinguishes successful Linux CI from
  unverified Linux model inference.
- [x] CUDA hardware and macOS remain explicitly unverified.
- [x] Public API and the pinned model revision are unchanged.
- [x] Model weights are not packaged.
- [x] Checked `https://pypi.org/pypi/openclip-gemma/0.1.1/json` on
  2026-10-17; it returned 404. Recheck immediately before release.
- [ ] Reconfirm the pinned model terms and contributor authority for the
  project's Apache-2.0 license before release.

## Automated validation and release artifacts

- [x] Run `uv sync --locked`.
- [x] Run `uv run pytest` — 75 passed, 11 integration tests skipped.
- [x] Run `uv run ruff check .`.
- [x] Run actionlint 1.7.12 on `ci.yml` and `publish-pypi-v0.1.1.yml`.
- [x] Run `uv build` — v0.1.1 sdist and wheel built.
- [x] Inspect the wheel and sdist with `tools/verify_wheel.py` and
  `tools/verify_sdist.py`, expecting version `0.1.1`.
- [x] Verify wheel metadata declares `Requires-Python: >=3.12,!=3.14.1,<3.15`
  and that neither distribution contains model weights.
- [x] Install the wheel into a clean Python 3.12 environment; run `pip check`
  and verify public API imports.
- [ ] After merging the release commit, confirm all six CI jobs pass for the
  exact main commit to be tagged.

## PyPI Trusted Publishing setup

The production-only v0.1.1 workflow is
[`.github/workflows/publish-pypi-v0.1.1.yml`](../.github/workflows/publish-pypi-v0.1.1.yml).
It does not run on ordinary branch pushes, accepts only the `v0.1.1` tag, and
requires an explicit confirmation for manual dispatch from that tag. Before
upload, it verifies the exact tag SHA, reachability from `main`, a successful
CI run for that same SHA, the project version, and the wheel/sdist contents.
Build artifacts are inspected before transfer to the separate publishing job.
Only that job has `id-token: write`; no API token is used. The v0.1.0 workflow
[`publish-pypi.yml`](../.github/workflows/publish-pypi.yml) is unchanged.

Configure or confirm these PyPI Trusted Publisher values:

| Setting | Value |
|---|---|
| PyPI project | `openclip-gemma` |
| GitHub owner | `kotadymo` |
| Repository | `openclip-gemma` |
| Workflow filename | `publish-pypi-v0.1.1.yml` |
| GitHub environment | `pypi` |

On GitHub, ensure the `pypi` environment exists and has required reviewers.
On PyPI, add the v0.1.1 workflow as a Trusted Publisher for this project; do
not replace or edit the v0.1.0 workflow's publisher configuration.

## Production release procedure (not executed)

1. Review and merge the v0.1.1 release changes to `main`.
2. Wait for CI on the resulting main commit and confirm all six OS/Python jobs
   pass. Do not use the earlier base-commit CI run as a substitute.
3. After explicit release approval, create and push an annotated tag pointing
   to that exact main commit:

   ```powershell
   git fetch origin
   git switch main
   git pull --ff-only origin main
   git tag -a v0.1.1 -m "Release v0.1.1"
   git push origin v0.1.1
   ```

   The tag push starts the workflow. Alternatively, manually dispatch
   `Publish 0.1.1 to PyPI` from the `v0.1.1` tag and select
   `publish-to-pypi`; do not do both because that can start duplicate upload
   attempts.
4. Review the build job's tag, CI, version, and artifact-verification results.
   Approve the `pypi` environment only after confirming the workflow run and
   artifacts are expected. Verify that the publish job succeeds.

## Post-publication verification (not executed)

1. Confirm the project JSON and release page report `0.1.1`:

   ```powershell
   $metadata = Invoke-RestMethod "https://pypi.org/pypi/openclip-gemma/0.1.1/json"
   if ($metadata.info.version -ne "0.1.1") { throw "Unexpected PyPI version." }
   ```

2. In a clean Python 3.12 environment, install the public package and check
   dependency consistency and exports:

   ```powershell
   py -3.12 -m venv .venv-pypi-0.1.1
   .\.venv-pypi-0.1.1\Scripts\python.exe -m pip install openclip-gemma==0.1.1
   .\.venv-pypi-0.1.1\Scripts\python.exe -m pip check
   .\.venv-pypi-0.1.1\Scripts\python.exe -c "import openclip_gemma; print(openclip_gemma.__all__)"
   ```

3. Run the README image and text example on CPU with network access or cached
   weights. Confirm both outputs are finite, shape `[1, 768]`, and use the
   pinned model revision.
4. Compare PyPI wheel/sdist SHA-256 values with the successful workflow summary
   and inspect the published metadata and provenance.
5. Only after these checks pass, create a GitHub Release from the same
   `v0.1.1` tag and include the verified artifact hashes.

Do not claim CUDA or Linux model-inference validation from the Ubuntu CI run.
