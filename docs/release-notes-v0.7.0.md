# OpenVLCLIP v0.7.0 — Release Notes

## New in v0.7.0

The library supports **five registered checkpoints** (two existing and three new):

| Model | Native image/text embedding | Capabilities |
| --- | ---: | --- |
| `embeddinggemma-2` | 768 | MRL 768/512/256/128, prompt-aware image/text |
| `dinov2-dinotxt` | 2048 | Official logit scale, 1024D patch features, similarity maps |
| `siglip2-base` | 768 | FixRes 224, official scale/bias logits |
| `siglip2-naflex-base` | 768 | Variable aspect ratio, patch masks and spatial shapes |
| `qwen3-vl-embedding-2b` | 2048 | Image/text/video/mixed inputs, instruction-aware, MRL 64–2048 |

**Qwen3-VL-Embedding-8B is deferred to a later release** due to insufficient GPU VRAM for the required CUDA validation. v0.7.0 does not register or claim support for the 8B checkpoint. Existing 8B research results are preserved separately.

### Added
- SigLIP 2 FixRes and NaFlex inference using pinned Google checkpoints with reference-parity verification.
- Pinned Qwen3-VL-Embedding-2B inference; text, image, native video, mixed-input encoding and compatible MRL.
- Processor-aware batching, instruction-aware retrieval, isolated native-video indexing and versioned EmbeddingSpace compatibility.
- Optional `[qwen3vl]` dependencies.
- Unified `publish-pypi.yml` for future stable tags, with Trusted Publishing and safety checks.

### Corrected
- dino.txt similarity-map peak-memory limits and malformed special-token input handling.
- Opt-in SHA-256 image-content verification through `--verify-contents`.
- Native embedding dimension resolution in frame-video indexing.

### Installation

```bash
pip install "openvlclip==0.7.0"
pip install "openvlclip[dinotxt]==0.7.0"
pip install "openvlclip[qwen3vl]==0.7.0"
```

Weights are retrieved from pinned external official sources and are not distributed in the wheel or sdist.

### Compatibility
- EmbeddingGemma 2 and dino.txt API and fixed numerical baselines remain unchanged.
- Versioned image/video indexes preserve documented legacy compatibility.
- Qwen processor-aware inputs differ from CLIP token-ID-only APIs; unsupported operations raise explicit errors.
- Qwen 2B native video index and frame-level video index are not interchangeable.

### Verification and known limitations
- See [model provenance](model-provenance-v0.7.0.md) for real-model parity under the tested CPU and CUDA configurations and the distinction between official wrapper parity and actual retrieval quality.
- Windows full-shared FFmpeg/TorchCodec compatibility depends on the installed runtime.
- Real-world retrieval accuracy and comprehensive benchmarks remain not evaluated.
- Published through GitHub Actions Trusted Publishing after the required `pypi` Environment approval; see the [publishing run](https://github.com/kotadymo/openvlclip/actions/runs/39941731530).

### Published distribution SHA-256

| Distribution | SHA-256 |
| --- | --- |
| `openvlclip-0.7.0-py3-none-any.whl` | `92944d02f57ef883d1a955452d1d9c2e1c446fa84f412c0e66752d4db4a38628` |
| `openvlclip-0.7.0.tar.gz` | `7b6b2b8834659a3ba5b74bd2f4fcfed6e302898bd61c574356ea016edcc3528b` |

The wheel and sdist hashes match the successful [GitHub Actions publish log](https://github.com/kotadymo/openvlclip/actions/runs/39941731530), the public [PyPI JSON metadata](https://pypi.org/pypi/openvlclip/0.7.0/json), and independently downloaded PyPI files. A clean Python 3.12 environment installed the published wheel; `import openvlclip`, version `0.7.0`, and `list_models()` succeeded. The registry contains the five checkpoints listed above and does not contain Qwen3-VL-Embedding-8B.

The release is available on [PyPI](https://pypi.org/project/openvlclip/0.7.0/) and as the [OpenVLCLIP v0.7.0 GitHub Release](https://github.com/kotadymo/openvlclip/releases/tag/v0.7.0).
