# Model provenance for OpenVLCLIP v0.7.0

- **Status:** Phase 0 source audit; runtime parity is not complete.
- **Audit date:** 2026-10-22
- **OpenVLCLIP baseline:** `main` at `4d6aba4e3dd7c152c48422a5e7e8b2d9672d2cfa`
- **Distribution:** `openvlclip`

This document records fixed upstream revisions, provenance, and static API findings for the four proposed v0.7.0 models. It is not evidence that OpenVLCLIP supports them, and it is not release approval. No model weights were downloaded or executed during this audit. The local runtime blockers and unverified contracts are listed at the end.

## Fixed model revisions

The revision values below are immutable Hugging Face repository commits observed through the Hugging Face model API on the audit date. Future code must pass these revisions explicitly; it must not resolve `main` or `latest` at model-load time.

| OpenVLCLIP ID | Hugging Face repository | Pinned revision | HF card license | Config / Transformers class | Expected native dimension |
| --- | --- | --- | --- | --- | ---: |
| `siglip2-base` | [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224/tree/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2) | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | Apache-2.0 | `model_type=siglip`; `AutoModel` resolves to `SiglipModel` | 768 |
| `siglip2-naflex-base` | [`google/siglip2-base-patch16-naflex`](https://huggingface.co/google/siglip2-base-patch16-naflex/tree/b53b807d3a2d5e2b3911292f2d69e5341cdc064c) | `b53b807d3a2d5e2b3911292f2d69e5341cdc064c` | Apache-2.0 | `model_type=siglip2`, `architectures=Siglip2Model` | 768 |
| `qwen3-vl-embedding-2b` | [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/tree/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda) | `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` | Apache-2.0 | `model_type=qwen3_vl`, checkpoint config declares `Qwen3VLForConditionalGeneration`; the embedding head is supplied by the pinned Qwen implementation | 2048 |
| `qwen3-vl-embedding-8b` | [`Qwen/Qwen3-VL-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/tree/2c4565515e0f265c6511776e7193b22c0968ddc7) | `2c4565515e0f265c6511776e7193b22c0968ddc7` | Apache-2.0 | `model_type=qwen3_vl`, checkpoint config declares `Qwen3VLForConditionalGeneration`; the embedding head is supplied by the pinned Qwen implementation | 4096 |

SigLIP 2's dimensions are the locked Transformers 5.20.1 config defaults: text and vision hidden size 768, with projection size defaulting to the hidden size. These are config/source-derived expected shapes, not measured model outputs. Qwen dimensions and parameter dtypes come from the pinned model-card/API metadata; the embedding outputs have not yet been executed.

## Hugging Face file hashes

The small config, processor, and tokenizer files in these tables were downloaded from the exact pinned `resolve/<revision>/...` URLs and SHA-256 hashed locally. Weight-file SHA-256 values are the Hugging Face LFS metadata for those same revisions; the large weights themselves were not downloaded or independently rehashed.

### SigLIP 2 FixRes Base

HF metadata reports 375,187,970 F32 parameters and a 1,500,800,904-byte `model.safetensors`.

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `config.json` | 253 | `fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7` |
| `preprocessor_config.json` | 394 | `9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc` |
| `model.safetensors` | 1,500,800,904 | `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b` |
| `tokenizer.json` | 34,363,039 | `cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322` |
| `tokenizer.model` | 4,241,003 | `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` |
| `tokenizer_config.json` | 47,164 | `14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8` |
| `special_tokens_map.json` | 636 | `baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351` |

The pinned processor is `SiglipImageProcessor` with 224x224 resize, rescale by 1/255, RGB mean/std of 0.5, and resampling value 2 (Pillow bilinear). The tokenizer config names `GemmaTokenizer`; the model config sets vocabulary size 256,000. The Transformers 5.20.1 text config uses a 64-token position limit. Preserve the processor's text padding/truncation contract.

### SigLIP 2 NaFlex Base

HF metadata reports 375,234,050 F32 parameters and a 1,500,985,224-byte `model.safetensors`.

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `config.json` | 329 | `c0b8c2e7f0527b0bea1b1d9abe0381c0f294352df92439c54590b8420e539118` |
| `preprocessor_config.json` | 393 | `1125703e5446d5b6ff4d5893a33bac128cdd21dc12e3dad2469a648fb0ae3bf7` |
| `model.safetensors` | 1,500,985,224 | `ac5f28bbdf92c0c1696ccbd3ce716426049cd67ad8045b66d0d938b0f9c8bbec` |
| `tokenizer.json` | 34,356,304 | `58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0` |
| `tokenizer.model` | 4,241,003 | `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` |
| `tokenizer_config.json` | 40,160 | `d2343400f0f86133053325951b696df8fd0f53a007cf6a546e6c2b4361344f47` |
| `special_tokens_map.json` | 636 | `baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351` |

The config identifies `Siglip2Model`. The processor config identifies `Siglip2Processor` / `Siglip2ImageProcessorFast`, patch size 16, and `max_num_patches=256`; it has no fixed 224x224 image size. The Transformers 5.20.1 `AutoImageProcessor` compatibility path removes the legacy `Fast` suffix and resolves the exported `Siglip2ImageProcessor` (a `TorchvisionBackend` implementation). That processor emits `pixel_values`, `pixel_attention_mask`, and `spatial_shapes`; all fields must be carried through batching and search. It resizes while preserving aspect ratio and pads the patch sequence, rather than substituting a fixed crop.

### Qwen3-VL-Embedding-2B

HF metadata reports 2,127,532,032 BF16 parameters and a 4,255,140,312-byte `model.safetensors`.

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `config.json` | 1,570 | `9172f55b0b9cce70b7f67b10c58a408ccf3ec15c587e6efd4d5f41631237fded` |
| `preprocessor_config.json` | 783 | `d75c1f9f474add87cb0188d43c4fbf5c232d2af64f156fcefc5311089049100d` |
| `model.safetensors` | 4,255,140,312 | `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1` |
| `tokenizer.json` | 11,423,705 | `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a` |
| `tokenizer_config.json` | 5,404 | `355f2b4e5bad7b01f11ef6cb68ebc176f61b95c3276092ea225b1bea0e01e95c` |
| `video_preprocessor_config.json` | 817 | `59c5c9eb52182eb14c06ffb10ca9effd29adce5f238a95de23ca14a38dbd2cb1` |

The checkpoint config reports text hidden size 2048, vision hidden size 1024, `Qwen3VLProcessor`, `Qwen2Tokenizer`, and `Qwen2VLImageProcessorFast`. The tokenizer config's `model_max_length=262144` is not the official embedder's default context length.

### Qwen3-VL-Embedding-8B

HF metadata reports 8,144,793,840 BF16 parameters. The four checkpoint shards total 16,289,679,624 bytes (15.17 GiB).

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `config.json` | 1,539 | `1eaa88d8d8824a47304e864c14101fe9efe3286607a7573f27eb723637f0d73d` |
| `preprocessor_config.json` | 784 | `d6ccdc0fe0f6b49aaeb3c10d2687289a057233ed685b30155202898a0e5e470f` |
| `model-00001-of-00004.safetensors` | 4,998,056,552 | `79ef275ec5f751d5fb59357c00d473268f9fd74abf5e38aa30137d268e7733c4` |
| `model-00002-of-00004.safetensors` | 4,915,962,464 | `a4da61f512e84fc0f0b80bcb7bcc5137eb3bf25b658a7d55f84f4056078545f0` |
| `model-00003-of-00004.safetensors` | 4,915,962,496 | `7fb17cf8f06d6fe5aaacf114e85c4e6d8318799f24b517f50d1ec154a8d47007` |
| `model-00004-of-00004.safetensors` | 1,459,698,112 | `000213b6d1d03ed9023fac23716da51ec4c5be221a04526c2f732d31d8fed1f5` |
| `tokenizer.json` | 11,423,705 | `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a` |
| `tokenizer_config.json` | 5,404 | `355f2b4e5bad7b01f11ef6cb68ebc176f61b95c3276092ea225b1bea0e01e95c` |
| `video_preprocessor_config.json` | 817 | `59c5c9eb52182eb14c06ffb10ca9effd29adce5f238a95de23ca14a38dbd2cb1` |

The checkpoint config reports text hidden size 4096, vision hidden size 1152, `Qwen3VLProcessor`, and `Qwen2Tokenizer`.

## Official implementation and inference contracts

### SigLIP 2

The OpenVLCLIP lock resolves Transformers `5.20.1`. The corresponding upstream Transformers commit is [`0ea877911a11499053df677befd4b35b172ff502`](https://github.com/huggingface/transformers/tree/0ea877911a11499053df677befd4b35b172ff502). Static inspection of that revision confirms:

- The FixRes and NaFlex checkpoints resolve to different model families: `SiglipModel` (`model_type=siglip`) and `Siglip2Model` (`model_type=siglip2`).
- The public feature methods are `get_image_features` and `get_text_features`; their model outputs must be handled as structured Transformers outputs, using the pooled projected features rather than assuming the wrapper always receives a bare Tensor.
- Both model forwards L2-normalize the pooled image/text features for similarity, compute cosine dot products, then apply the learned `exp(logit_scale)` and `logit_bias`. The learned values are in the checkpoint, not in `config.json`; their actual values and numerical parity remain unmeasured.
- The official zero-shot pipeline applies sigmoid to each scaled logit. This is not a softmax probability distribution. The cosine-only mode must remain distinct from scaled logits.
- NaFlex processor output includes patch values, valid-patch mask, and spatial shapes. Dropping any of these is not a valid NaFlex path.

The source files inspected at this Transformers revision were:

| File | SHA-256 |
| --- | --- |
| `models/siglip/configuration_siglip.py` | `4bb206d27da475a051219f783c4542c209729fc5a0c9279b1439d75c555b9228` |
| `models/siglip/modeling_siglip.py` | `be6a88e579c8b4a69d705a7903174f975ff5cc278910539d0a5505116da9d710` |
| `models/siglip2/configuration_siglip2.py` | `444b07c3c1c45cd2ffa4abeadb7fe844349762e35a629aed6a823a7e471e998f` |
| `models/siglip2/modeling_siglip2.py` | `87fc104c588c0c72453a84b411d10dc8599ba9e7a80be0c57b246b2c83329ba7` |
| `models/siglip2/processing_siglip2.py` | `248107456d2a86267273b8f1a39da3788572361b827a07d73d837e68dfa0c419` |
| `models/siglip2/image_processing_siglip2.py` | `613a0e9cbe20c16269ac4da47b9b3cfd63553039891dfd3335945280c9c048ca` |
| `models/auto/image_processing_auto.py` | `730f482febb436e5508ba7c9bed2a606de4b584b3ecb9eaf4ff8287b7a29c7e7` |

The file implements legacy `*Fast` suffix handling. Runtime loading and processor output still require an integration test.

### Qwen3-VL-Embedding

The official source repository [`QwenLM/Qwen3-VL-Embedding`](https://github.com/QwenLM/Qwen3-VL-Embedding/tree/393e2978d27852b0d0230d6994f37f9c15bed73c) was pinned at commit `393e2978d27852b0d0230d6994f37f9c15bed73c` (2026-06-23). The audited source file is [`src/models/qwen3_vl_embedding.py`](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/393e2978d27852b0d0230d6994f37f9c15bed73c/src/models/qwen3_vl_embedding.py), SHA-256 `0e447b6a80e05527f3fd437ab4f1617a6e7d0df102b8164348bbce787a41b267`.

Static inspection of that source and the pinned model cards confirms:

- `Qwen3VLEmbedder` constructs a system message from the instruction, uses `Qwen3VLProcessor.apply_chat_template(..., add_generation_prompt=True)`, then passes text, image, and video inputs through the Qwen processor and `qwen_vl_utils.vision_process.process_vision_info`.
- The default instruction is `Represent the user's input.`. The official retrieval example uses `Retrieve images or text relevant to the user's query.` for the query and a different/no instruction for documents; the exact paired policy must be part of the embedding-space identity.
- The official wrapper defaults to `max_length=8192`, `fps=1`, and `max_frames=64`. The model cards list a 32K sequence length, while the tokenizer config sets `model_max_length=262144`; these values are not interchangeable.
- The wrapper's image defaults are `min_pixels=4096`, `max_pixels=1,843,200`; its video defaults include `total_pixels=7,864,320`. The pinned HF `preprocessor_config.json` has `max_pixels=1,310,720`, so the official embedder's explicit content-level override must be preserved and tested.
- `Qwen3VLForEmbedding` returns the last hidden state and attention mask. `_pooling_last` selects the last non-padding token. `process(normalize=True)` L2-normalizes after pooling.
- The upstream wrapper uses `trust_remote_code=True`; it also catches vision-processing exceptions and replaces the failed media input with `"NULL"`. OpenVLCLIP must not use unpinned remote code or silently produce a NULL embedding. A reviewed, pinned, locally controlled implementation and explicit media errors are required.
- The model cards advertise MRL ranges of 64-2048 and 64-4096. The audited `Qwen3VLEmbedder.process` source has no `truncate_dim` argument or truncation implementation. Exact slicing, accepted dimensions, and truncate-then-normalize parity therefore remain an open Phase 0 contract; they must be resolved against an independent official/reference path before implementation.

The official Qwen repository's `LICENSE` is Apache-2.0, SHA-256 `f08c07d73a60444803f8902b00f7b14c839647a48cc07fba59b3c1bda656ec3c`. At the pinned source commit, `pyproject.toml` has SHA-256 `12c3726efcb3028dd9055c68f33bcbda6d6b9d379deeb920b0467b206675ffe3`; the source README has SHA-256 `c57c3c6c15d15b16a7a281fe5454a88bfe43f57b014d7398e59d37f8db3f36a9`.

## License and dependency findings

- The four pinned Hugging Face model cards declare Apache-2.0. The SigLIP repositories expose that declaration in model-card metadata; a separate `LICENSE` file was not listed in the queried snapshot. The tokenizer files are pinned within those same repositories. No separate tokenizer license statement was found during this static audit, so the model-card declaration is the current evidence and must remain clearly attributed.
- The Qwen model cards declare Apache-2.0. The pinned Qwen source repository includes an Apache-2.0 `LICENSE`. The `qwen-vl-utils` PyPI metadata for version `0.0.14` declares Apache-2.0 and Python `>=3.8`; its required dependencies include `av`, `packaging`, `pillow`, and `requests`, while `decord` is an optional extra.
- The OpenVLCLIP wheel and sdist must not contain model weights. Downloaded weights remain in the user's Hugging Face cache.
- The current OpenVLCLIP lock resolves Torch `2.14.1`, Torchvision `0.29.1`, and Transformers `5.20.1`. The pinned Qwen source project's manifest requests Torch `2.8.*`, Torchvision `>=0.23.0`, and Transformers `>=4.57.3`, and lists additional development/data packages. Do not copy its Torch pin or install all of its development dependencies; compatibility with the existing locked versions has not yet been tested. Its manifest also configures an HTTP package mirror; that index must not be used.
- The upstream Qwen code imports `qwen-vl-utils`; OpenVLCLIP currently does not lock it. The project already has optional `video` extra with PyAV. Windows/Linux and Python 3.12-3.14 resolution and inference compatibility still require CI testing; the metadata minimum version alone does not establish that compatibility.
- The SigLIP tokenizer config names `GemmaTokenizer`; the verified model repositories include tokenizer JSON files. Whether a `sentencepiece` extra is needed for the chosen Transformers loading path must be settled with an actual clean-environment test.

No code, model weights, or license files have been copied into OpenVLCLIP as part of this audit. If future implementation vendors Qwen code, preserve its Apache license and required attribution/NOTICE information.

## Runtime validation status and blockers

No official model was loaded. No output shapes were measured, no numerical parity or error tolerance was established, and no BF16/CUDA behavior is claimed by this document.

The local environment inspected on 2026-10-22 is Windows with Python 3.12.10 and Torch `2.14.0+cpu`; `torch.cuda.is_available()` is false, and Transformers and Torchvision are not installed in that interpreter. The repository lock instead requires Torch `2.14.1`, Torchvision `0.29.1`, and Transformers `5.20.1`.

`nvidia-smi` reports a 16,376 MiB GPU with 1,641 MiB currently used and 14,735 MiB free. The Qwen 8B BF16 checkpoint shards total 15.17 GiB, exceeding currently free device memory before model activations and runtime buffers. Even an idle 16 GiB device would leave less than 1 GiB after the checkpoint weights alone. The repository has zero configured self-hosted Actions runners. This environment cannot perform the mandatory Qwen 8B GPU parity test.

**Stop gate:** do not start Phase 1 or declare any new model supported until a CUDA-enabled, higher-memory GPU environment is available and the unresolved Qwen MRL contract is independently verified. A 24 GiB-or-larger GPU is a practical environment to try, not a measured minimum; record actual peak memory before accepting it. The 8B test must use the fixed checkpoint, a recorded precision, and the pinned official reference implementation. Do not use quantization or unverified CPU offload to bypass the requirement.

Official references:

- [SigLIP 2 FixRes model card](https://huggingface.co/google/siglip2-base-patch16-224)
- [SigLIP 2 NaFlex model card](https://huggingface.co/google/siglip2-base-patch16-naflex)
- [Transformers 5.20.1 SigLIP documentation](https://huggingface.co/docs/transformers/v5.20.1/model_doc/siglip)
- [Transformers 5.20.1 SigLIP 2 documentation](https://huggingface.co/docs/transformers/v5.20.1/model_doc/siglip2)
- [Qwen3-VL-Embedding 2B model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- [Qwen3-VL-Embedding 8B model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)
- [Pinned Qwen3-VL-Embedding source](https://github.com/QwenLM/Qwen3-VL-Embedding/tree/393e2978d27852b0d0230d6994f37f9c15bed73c)
- [Qwen3-VL-Embedding official implementation file](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/393e2978d27852b0d0230d6994f37f9c15bed73c/src/models/qwen3_vl_embedding.py)
- [qwen-vl-utils on PyPI](https://pypi.org/project/qwen-vl-utils/)
