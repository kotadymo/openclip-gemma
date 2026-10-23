# OpenVLCLIP v0.7.0 release checklist

**Status (2026-10-23):** `v0.7.0` has been tagged and its wheel and sdist uploaded successfully to production PyPI. GitHub Release creation and independent post-publication checks remain incomplete. Do not push or rewrite the existing tag.

## Scope
- [x] `list_models()` contains exactly five checkpoints: Gemma, dino.txt, SigLIP2 FixRes, SigLIP2 NaFlex, Qwen3-VL-Embedding-2B.
- [x] Qwen3-VL-Embedding-8B is **not registered or advertised** as supported in v0.7.0.
- [x] `pyproject.toml` and `uv.lock` both contain `0.7.0`.

## Code and numerical checks
- [x] PR #44 (Qwen 2B only) merged; successful main push CI for exact release commit.
- [ ] SigLIP 2 FixRes/NaFlex official reference parity confirmed.
- [ ] Qwen 2B CPU FP32/BF16 and CUDA FP32/BF16 official parity confirmed from actual recorded tests.
- [ ] Qwen 2B image/text/video/mixed inference and native-video index smoke tested.
- [ ] EmbeddingGemma v0.4.0 MRL and dino.txt official global/patch/map numerical regressions pass.
- [ ] Legacy image/video index reads and new native-video index separation pass.
- [ ] Phase 4 dino.txt fixes / CLI regression pass.
- [ ] Windows and Linux × Python 3.12, 3.13 and 3.14 CI success.
- [ ] Ruff, `uv lock --check`, wheel/sdist build and clean install success.
- [ ] Official weight/license documentation reviewed; no weights in distributions.

## Shared publish workflow and user authorization
- [x] `.github/workflows/publish-pypi.yml` permits stable v0.7.0 tags and checks exact main SHA, main CI, licenses, SemVer and PyPI absence.
- [ ] Existing `OPENVLCLIP_DINOTXT_LICENSE_CONFIRMED=true` variable validated.
- [ ] New `OPENVLCLIP_V070_MODELS_LICENSE_CONFIRMED=true` variable set only after reviewing SigLIP/Qwen 2B license evidence.
- [ ] PyPI `openvlclip` project has registered Trusted Publisher: owner `kotadymo`, repo `openvlclip`, workflow `publish-pypi.yml`, environment `pypi`.
- [ ] GitHub `pypi` Environment permits the `v0.7.0` tag without disabling approval or protection rules.

## Publication and verification
- [x] Tag `v0.7.0` points to exact main CI-success SHA and is pushed once, without force.
- [x] New shared publish workflow completed; manual Environment approval given by authorized reviewer.
- [ ] PyPI wheel and sdist filenames/sha256 match published JSON metadata.
- [ ] Fresh v0.7.0 PyPI wheel install and smoke tests pass.
- [ ] GitHub Release v0.7.0 published after successful PyPI verification.
- [ ] Release notes and README updated after publication; old tags and workflow history untouched.

Previous Qwen 8B CUDA tests are **not** part of the v0.7.0 gate. They may be required for a future 8B release.

## Confirmed distribution evidence

- [x] [Production publish run 39941731530](https://github.com/kotadymo/openvlclip/actions/runs/39941731530) completed successfully with both publish uploads returning HTTP 200.
- [x] wheel `openvlclip-0.7.0-py3-none-any.whl` SHA-256 from publish log: `92944d02f57ef883d1a955452d1d9c2e1c446fa84f412c0e66752d4db4a38628`
- [x] sdist `openvlclip-0.7.0.tar.gz` SHA-256 from publish log: `7b6b2b8834659a3ba5b74bd2f4fcfed6e302898bd61c574356ea016edcc3528b`
- [ ] Compare both SHA-256 values against the **public PyPI JSON** metadata independently.
- [ ] Fresh environment: install the exact PyPI wheel (not a source checkout) and check `import openvlclip`, version, `list_models()` (five) and 8B absence.
- [ ] Confirm GitHub Release `v0.7.0` exists, is neither draft nor prerelease, and includes these release notes.

All checked items above correspond to repository content, completed GitHub CI or GitHub's publishing logs. Unchecked items must not be reported as successful until independently verified.
