# Migration Guide: openclip-gemma v0.4.0 to OpenVLCLIP v0.5.0

OpenVLCLIP v0.5.0 changes the Python distribution and import names. It does
not add a model or change EmbeddingGemma 2 inference behavior.

| Item | v0.4.0 | v0.5.0 |
| --- | --- | --- |
| Distribution | `openclip-gemma` | `openvlclip` |
| Import | `openclip_gemma` | `openvlclip` |
| Model | `embeddinggemma-2` | `embeddinggemma-2` |

## Install

Install the new distribution in the target environment:

```bash
pip install openvlclip
```

For video indexing and search, install the optional extra:

```bash
pip install "openvlclip[video]"
```

`openvlclip` and `openclip-gemma` are separate PyPI distributions. Installing
the new package does not uninstall or replace the old one. The new package
does not provide an `openclip_gemma` compatibility import. Co-installation of
both distributions in one environment is not guaranteed; use a fresh
environment if dependency isolation is important.

## Update imports

Change package imports from the old namespace to the new one:

```python
from openclip_gemma import create_model_and_transforms, get_tokenizer

model, _, preprocess = create_model_and_transforms("embeddinggemma-2")
tokenizer = get_tokenizer("embeddinggemma-2")
```

```python
from openvlclip import create_model_and_transforms, get_tokenizer

model, _, preprocess = create_model_and_transforms("embeddinggemma-2")
tokenizer = get_tokenizer("embeddinggemma-2")
```

Submodule imports change the same way, for example:

```python
from openvlclip.batching import iter_image_embeddings
from openvlclip.image_search import build_or_update_image_index, search_images
from openvlclip.video_search import build_or_update_video_index, search_video
```

## Inference and index compatibility

The public Factory, tokenizer, preprocessing, task prompts, and model methods
retain their v0.4.0 contracts. `embeddinggemma-2` remains the only supported
real model. The default embedding remains 768 dimensions; MRL dimensions
512/256/128, selective loading, FP32, and the validated CPU/CUDA BF16 behavior
are preserved.

Existing image and video indexes created by v0.4.0 remain readable. Their
format versions and validation rules are unchanged; no index conversion or
re-embedding is required solely because of the Python package rename.

The old distribution and its published versions remain available
independently. This migration does not modify or delete `openclip-gemma`.
