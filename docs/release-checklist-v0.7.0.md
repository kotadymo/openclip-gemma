# OpenVLCLIP v0.7.0 release checklist

**Status (2026-10-23):** `v0.7.0` is published on [PyPI](https://pypi.org/project/openvlclip/0.7.0/) and as a formal [GitHub Release](https://github.com/kotadymo/openvlclip/releases/tag/v0.7.0). The tag points to `81c8e1df9cb02d6b9f13c4ebda9ac605a54b62eb`. PyPI metadata, downloaded artifact hashes, and a clean Python 3.12 wheel installation were independently verified.

## Scope
- [x] `list_models()` contains exactly five checkpoints: Gemma, dino.txt, SigLIP2 FixRes, SigLIP2 NaFlex, Qwen3-VL-Embedding-2B.
- [x] Qwen3-VL-Embedding-8B is **not registered or advertised** as supported in v0.7.0.
- [x] `pyproject.toml` and `uv.lock` both contain `0.7.0`.

## Code and numerical checks
- [x] PR #44 (Qwen 2B only) merged; successful main push CI for exact release commit.
- [x] SigLIP 2 FixRes/NaFlex official reference parity confirmed in the [model provenance report](model-provenance-v0.7.0.md).
- [x] Qwen 2B CPU FP32/BF16 and CUDA FP32/BF16 official parity confirmed from recorded tests in the [model provenance report](model-provenance-v0.7.0.md).
- [x] Qwen 2B image/text/video/mixed inference and native-video index smoke tested; see the [model provenance report](model-provenance-v0.7.0.md).
- [x] EmbeddingGemma v0.4.0 MRL and dino.txt official global/patch/map numerical regression results are recorded in the [dino.txt reference](dinotxt-v0.6.0.md).
- [x] Legacy image/video index reads and new native-video index separation are covered by the passing main CI test matrix.
- [x] Phase 4 dino.txt fixes / CLI regression passed in the main CI test matrix.
- [x] Windows and Linux × Python 3.12, 3.13 and 3.14 CI succeeded for exact release main SHA `81c8e1df9cb02d6b9f13c4ebda9ac605a54b62eb` ([run 39940171668](https://github.com/kotadymo/openvlclip/actions/runs/39940171668)).
- [x] Ruff, `uv lock --check`, wheel/sdist build and clean install succeeded.
- [x] Official weight/license documentation reviewed; the published wheel and sdist contain no model-weight files.

## Shared publish workflow and user authorization
- [x] `.github/workflows/publish-pypi.yml` permits stable v0.7.0 tags and checks exact main SHA, main CI, licenses, SemVer and PyPI absence.
- [x] Existing `OPENVLCLIP_DINOTXT_LICENSE_CONFIRMED=true` variable validated.
- [x] `OPENVLCLIP_V070_MODELS_LICENSE_CONFIRMED=true` set after reviewing SigLIP/Qwen 2B license evidence.
- [x] PyPI `openvlclip` project used the Trusted Publisher with owner `kotadymo`, repo `openvlclip`, workflow `publish-pypi.yml`, environment `pypi`.
- [x] GitHub `pypi` Environment permitted the `v0.7.0` tag; the required reviewer approval was completed without changing protection rules.

## Publication and verification
- [x] Tag `v0.7.0` points to exact main CI-success SHA and is pushed once, without force.
- [x] New shared publish workflow completed; manual Environment approval given by authorized reviewer.
- [x] PyPI wheel and sdist filenames/SHA-256 match published JSON metadata.
- [x] Fresh v0.7.0 PyPI wheel install and smoke tests pass.
- [x] GitHub Release v0.7.0 published after successful PyPI verification.
- [x] Release notes and README updated after publication; old tags and workflow history untouched.

Previous Qwen 8B CUDA tests are **not** part of the v0.7.0 gate. They may be required for a future 8B release.

## Confirmed distribution evidence and final verification

- [x] [Production publish run 39941731530](https://github.com/kotadymo/openvlclip/actions/runs/39941731530) completed successfully with both publish uploads returning HTTP 200.
- [x] PyPI JSON reports version `0.7.0`, wheel `openvlclip-0.7.0-py3-none-any.whl` (103,088 bytes), and sdist `openvlclip-0.7.0.tar.gz` (105,278 bytes).
- [x] Wheel SHA-256 `92944d02f57ef883d1a955452d1d9c2e1c446fa84f412c0e66752d4db4a38628` matches the GitHub publishing log, PyPI JSON, and the independently downloaded wheel.
- [x] Sdist SHA-256 `7b6b2b8834659a3ba5b74bd2f4fcfed6e302898bd61c574356ea016edcc3528b` matches the GitHub publishing log, PyPI JSON, and the independently downloaded sdist.
- [x] A new Python 3.12.10 virtual environment installed the exact wheel downloaded from PyPI, resolving dependencies from PyPI. `import openvlclip` succeeded from that environment's `site-packages`; installed metadata reported `0.7.0`.
- [x] `list_models()` returned exactly `embeddinggemma-2`, `dinov2-dinotxt`, `siglip2-base`, `siglip2-naflex-base`, and `qwen3-vl-embedding-2b`; `qwen3-vl-embedding-8b` was absent.
- [x] Archive inspection found no model-weight files in either the public wheel or sdist.
- [x] The [GitHub Release](https://github.com/kotadymo/openvlclip/releases/tag/v0.7.0) is public, named `OpenVLCLIP v0.7.0`, and is neither a draft nor a prerelease.

All checked items above correspond to repository evidence, completed CI, recorded model-provenance tests, public PyPI metadata, downloaded distribution bytes, or the public GitHub Release. In the exact-main CI run, the optional `CPU model integration (manual)` job was skipped; model-specific parity evidence is recorded separately in the linked provenance documents. Qwen 8B CUDA remains unverified and is outside the v0.7.0 release scope.
