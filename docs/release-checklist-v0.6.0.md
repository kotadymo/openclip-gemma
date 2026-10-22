# OpenVLCLIP v0.6.0 Release Record

**Status: released on 2026-10-22.** TestPyPI was not used.

- [GitHub Release v0.6.0](https://github.com/kotadymo/openvlclip/releases/tag/v0.6.0)
- [PyPI openvlclip 0.6.0](https://pypi.org/project/openvlclip/0.6.0/)
- Release tag: `v0.6.0`
- Release source: main commit `97f670c4a64200d6286f48fb238a9e8446a6433d`

This file records the completed release and preserves post-release limitations.
It is not a request to recreate the tag, re-upload distributions, or recreate
the GitHub Release.

## Completed source and validation checks

- [x] `pyproject.toml` and `uv.lock` identify `openvlclip==0.6.0`;
  `uv lock --check` passed.
- [x] Ruff and `git diff --check` passed.
- [x] Normal `pytest -q`: 304 passed, 30 skipped, 3 warnings.
- [x] EmbeddingGemma regression tests passed for image/text embeddings, MRL
  768/512/256/128, task prompts, selective loading, image/video search, and
  index compatibility. The v0.4.0 `.npz` baseline was not modified.
- [x] The CPU FP32 comparison with the v0.4.0 baseline had maximum absolute
  errors of `1.27e-7` or less across the tested image/text MRL dimensions.
- [x] dino.txt official CPU FP32/BF16 parity passed with maximum absolute
  error `0.0` for the compared embeddings, logits, patches, and similarity
  maps.
- [x] The wheel and sdist passed content verification. A clean Python 3.12
  wheel install reported version `0.6.0`, loaded from `site-packages`, and
  passed the real-model integration suite (16 passed).
- [x] Main CI on the release source passed Windows/Linux and Python
  3.12/3.13/3.14. The manual real-model CI job was skipped.
- [x] PyPI JSON confirmed both published files and the GitHub repository URL
  in distribution metadata.
- [x] The public GitHub Release uses the existing `v0.6.0` tag.

### Published PyPI files

| File | Size | SHA-256 |
| --- | ---: | --- |
| `openvlclip-0.6.0-py3-none-any.whl` | 73,375 bytes | `a6fc193bd988df697b595ee10daaefcadabb8238d128023139bbb64bce5722aa` |
| `openvlclip-0.6.0.tar.gz` | 73,883 bytes | `9162351f8a133dd029385c658d6304f9bb1447b05ae80fa15255d0f8adc37a3d` |

The final Phase 4 environment had no CUDA runtime or GPU. Eleven CUDA-only
real-model integration cases were skipped, not passed. Earlier targeted
CUDA parity measurements are documented separately in the
[dino.txt validation reference](dinotxt-v0.6.0.md). Dataset-based accuracy,
retrieval quality, and comparative GPU performance remain `not evaluated`;
see the [benchmark report](benchmark-v0.6.0.md).

## License scope and post-release follow-up

The pinned DINOv2 repository [LICENSE](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE)
is Apache-2.0. Its [model card](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/MODEL_CARD.md)
also lists Apache License 2.0 for the DINOv2 ViT model family. The official
[dino.txt checkpoint links](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md)
point to external Vision Head and Text Encoder files whose individual license
terms are not separately specified in the pinned repository. Do not infer
that those distinct checkpoint terms are resolved by the repository license
or by publication of this package.

The BPE vocabulary's separate [official license](https://dl.fbaipublicfiles.com/dinov2/thirdparty/LICENSE)
is MIT (OpenAI, 2021). OpenVLCLIP's wheel and sdist do not include DINOv2 or
dino.txt model weights; they are retrieved externally at runtime. Clarify the
head/text checkpoint terms before separately redistributing those files.

## Release workflow record

The v0.6.0-specific [Trusted Publishing workflow](../.github/workflows/publish-pypi-v0.6.0.yml)
targets only `v0.6.0`, verifies the source/version/CI/artifacts, uses the
`pypi` Environment, and grants `id-token: write` only to the publish job.
TestPyPI was intentionally omitted. The v0.5.0 release files and historical
v0.4.0/v0.5.0 records are unchanged.
