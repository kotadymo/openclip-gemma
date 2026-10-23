# OpenVLCLIP v0.7.0 release checklist

**Status:** Release preparation only; do not push `v0.7.0` until all blocking items pass.

## Scope
- [ ] `list_models()` contains exactly five checkpoints: Gemma, dino.txt, SigLIP2 FixRes, SigLIP2 NaFlex, Qwen3-VL-Embedding-2B.
- [ ] Qwen3-VL-Embedding-8B is **not registered or advertised** as supported in v0.7.0.
- [ ] `pyproject.toml` and `uv.lock` both contain `0.7.0`.

## Code and numerical checks
- [ ] PR #44 (Qwen 2B only) merged; successful main push CI for exact release commit.
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
- [ ] `.github/workflows/publish-pypi.yml` permits stable v0.7.0 tags and checks exact main SHA, main CI, licenses, SemVer and PyPI absence.
- [ ] Existing `OPENVLCLIP_DINOTXT_LICENSE_CONFIRMED=true` variable validated.
- [ ] New `OPENVLCLIP_V070_MODELS_LICENSE_CONFIRMED=true` variable set only after reviewing SigLIP/Qwen 2B license evidence.
- [ ] PyPI `openvlclip` project has registered Trusted Publisher: owner `kotadymo`, repo `openvlclip`, workflow `publish-pypi.yml`, environment `pypi`.
- [ ] GitHub `pypi` Environment permits the `v0.7.0` tag without disabling approval or protection rules.

## Publication and verification
- [ ] Tag `v0.7.0` points to exact main CI-success SHA and is pushed once, without force.
- [ ] New shared publish workflow completed; manual Environment approval given by authorized reviewer.
- [ ] PyPI wheel and sdist filenames/sha256 match published JSON metadata.
- [ ] Fresh v0.7.0 PyPI wheel install and smoke tests pass.
- [ ] GitHub Release v0.7.0 published after successful PyPI verification.
- [ ] Release notes and README updated after publication; old tags and workflow history untouched.

Previous Qwen 8B CUDA tests are **not** part of the v0.7.0 gate. They may be required for a future 8B release.
