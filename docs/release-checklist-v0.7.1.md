# OpenVLCLIP v0.7.1 Release Checklist

**Status (2026-10-23):** Published to PyPI and as a formal GitHub Release after
all release gates passed. Qwen3-VL-Embedding-8B is outside this release.

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
- [x] Release-preparation PR [#52](https://github.com/kotadymo/openvlclip/pull/52)
  merged to `main` at `e8f726dda242101a1be71587044e3bb43061d72a`.
- [x] Final main CI passed for that exact SHA:
  [run 39954908351](https://github.com/kotadymo/openvlclip/actions/runs/39954908351).
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
- [x] Release CI passed on Windows and Linux with Python 3.12, 3.13, and 3.14;
  the manual CPU model-integration job was skipped.
- [x] The release workflow built and verified the v0.7.1 wheel and sdist.
  Public PyPI files also passed the wheel and sdist content verifiers.
- [x] Installed the downloaded PyPI wheel in a clean Python 3.12.10
  environment; import, version, five-model registry, existing factory APIs,
  and Native Video Index `verify_contents` checks passed. Qwen 8B was absent.
- [x] Distribution verifiers confirmed that neither the wheel nor sdist
  bundles model weights.

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
  were preserved; `v0.7.1` was added to the tag allowlist without changing
  existing policies.
- [x] The shared workflow verified the exact tag and commit, main ancestry,
  same-SHA main CI, lock, wheel/sdist contents, license variables, and
  Trusted Publishing configuration.
- [x] Immediately before tagging, verified `v0.7.1` did not exist as a tag or
  GitHub Release and `openvlclip==0.7.1` was absent from PyPI.
- [x] Created and pushed the annotated `v0.7.1` tag at the exact CI-successful
  main commit without force or overwrite.
- [x] The authorized reviewer approved the `pypi` Environment deployment;
  deployment protections remained enabled.
- [x] The publishing workflow succeeded and its artifact hashes match both
  downloaded PyPI files and PyPI JSON metadata.
- [x] The published wheel was installed in a clean Python 3.12 environment;
  package version, imports, registry, and Qwen 8B absence were verified.
- [x] Published the formal GitHub Release after PyPI artifact verification.

## Publication record

- PyPI: <https://pypi.org/project/openvlclip/0.7.1/>
- GitHub Release: <https://github.com/kotadymo/openvlclip/releases/tag/v0.7.1>
- Main/tag commit: `e8f726dda242101a1be71587044e3bb43061d72a`
- Annotated tag object SHA: `56c8bf59c09acb793f55ca48fb4124179d89e8b1`
- Main CI: <https://github.com/kotadymo/openvlclip/actions/runs/39954908351>
- PyPI publishing workflow:
  <https://github.com/kotadymo/openvlclip/actions/runs/39955230402>
- Wheel filename / SHA-256:
  `openvlclip-0.7.1-py3-none-any.whl` /
  `0555451509d9386a3bf440debc08a38942b5110c6336d5a58d12aba2818fafc3`
- Sdist filename / SHA-256:
  `openvlclip-0.7.1.tar.gz` /
  `54b95f1cac5ed420def81b57d0f5c0c799701c1848d719297b869eb94f8c90b0`
- Clean-install verification: Python 3.12.10; PyPI wheel installed and
  imported from `site-packages`; package version `0.7.1`; five registered
  models; Qwen 8B absent; `verify_contents` and existing factory APIs present.
- Public wheel and sdist SHA-256 values match the workflow summary and PyPI
  JSON; both passed content verification and contain no model weights.
- GitHub Release published as a non-draft, non-prerelease release on
  2026-10-23.
