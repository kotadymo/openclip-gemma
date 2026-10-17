# Python compatibility investigation for v0.1.1

## Scope and conclusion

The project runtime and pinned model were tested on Windows 11 x64 with CPython
3.12.10, 3.13.16, and 3.14.8. Python 3.14.1 is excluded from the declared
version range because the required torchvision release rejects it. For each
tested interpreter, locked dependency setup,
unit tests, Ruff, wheel/sdist builds, artifact checks, and clean wheel install
with public API imports passed. CPU FP32 and BF16 integration tests against the
pinned EmbeddingGemma 2 revision also passed.

This evidence supports Python 3.12, 3.13, and Python 3.14 except 3.14.1 for the
tested Windows CPU environment. The only 3.14 patch release tested end-to-end
is 3.14.8; this does not certify every other 3.14 patch release. It also does
not certify other operating systems or CUDA. A Windows/Ubuntu CI matrix is
configured, but has not run on GitHub from this branch. Linux CI results and
Linux real-model inference are therefore both unverified.

## Compatibility evidence

OpenCLIP is an API compatibility reference only; it is not a runtime dependency
and its Python metadata does not define this package's support policy. The
upstream PyPI classifiers reviewed for OpenCLIP extend through Python 3.12.
The [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b)
does not specify a Python version range. Runtime model checks use the pinned
revision shown in that link.

| Component | Locked version | Published Python metadata reviewed | Observed validation |
|---|---:|---|---|
| PyTorch | 2.14.1 | `Requires-Python >=3.10`; compatible wheels were available for 3.12-3.14 | Dependencies, unit tests, build/install, and model integration passed on 3.12.10, 3.13.16, and 3.14.8 |
| torchvision | 0.29.1 | `Requires-Python !=3.14.1,>=3.10` | Full validation passed on 3.12.10, 3.13.16, and 3.14.8. A clean pip resolution for 3.14.1 rejects 0.29.1; the project now excludes 3.14.1. A prior locked uv install/import succeeded despite the metadata, but is not valid compatibility evidence. |
| Transformers | 5.20.1 | `Requires-Python >=3.10`; classifiers include Python 3.14 | Model and tokenizer integration passed on the three tested interpreters |
| Pillow | 13.0.0 | `Requires-Python >=3.11`; classifiers include Python 3.14 | Dependency setup and image integration passed on the three tested interpreters |
| Sentence Transformers (integration extra) | 6.1.0 | `Requires-Python >=3.10`; classifiers reviewed extend through Python 3.13 | Official-reference image/text comparisons passed on all three tested interpreters, including 3.14.8 |
| pytest | 9.1.1 | Python 3.10+ metadata and Python 3.14 classifier | 75 unit tests passed on each tested interpreter |
| Ruff | 0.17.1 | Python 3.14 supported by published metadata | `ruff check .` passed on each tested interpreter |
| uv build backend | 0.13.1 | Python 3.14 supported by published metadata | Wheel and sdist builds and artifact inspections passed on each tested interpreter |

Published `Requires-Python` and classifiers describe package metadata, not
proof that every downstream stack is operational on every interpreter. Runtime
results above are limited to the exact versions and environment tested.

## Test results

| CPython | Locked dependency setup | Unit tests | Lint/build/artifacts/clean install | Pinned model integration |
|---|---|---:|---|---|
| 3.12.10 | Passed | 75 passed, 11 skipped | Passed | 7 passed, 2 existing CUDA checks skipped |
| 3.13.16 | Passed | 75 passed, 11 skipped | Passed | 7 passed, 2 existing CUDA checks skipped |
| 3.14.8 | Passed | 75 passed, 11 skipped | Passed | 7 passed, 2 existing CUDA checks skipped |

The integration suite used revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`, CPU FP32 and BF16, and compared
image/text embeddings with Sentence Transformers. Image preprocessing, wrapper
loading, and zero-shot scoring were also exercised. The CUDA checks skipped
because this environment has CPU-only PyTorch and no CUDA device. The newly
added CUDA CPU-comparison/metrics test was run separately on this machine; both
precision cases skipped as expected because CUDA is unavailable.

CPython 3.14.1 was separately checked in a fresh venv with pip 26.2.1. The
resolver rejected torchvision 0.29.1 with
`Requires-Python !=3.14.1,>=3.10` and reported no compatible version under the
project's `torchvision>=0.29.1,<1` constraint. PyPI JSON metadata for version
0.29.1 reports the same `info.requires_python` value. A previous locked uv sync
installed this wheel anyway; fresh pip resolution is the relevant install
check, so `pyproject.toml` now excludes exactly Python 3.14.1. The same pip
26.2.1 dry-run accepted the torchvision 0.29.1 wheel on Python 3.14.8, and
`uv sync --locked` succeeds on 3.14.8 with the project's updated range.

## CUDA validation procedure

The real-GPU test is `tests/integration/test_cuda_integration.py`. Install a
CUDA-enabled torch/torchvision pair matching the host driver via the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/);
the versions must satisfy `pyproject.toml`. Then, from the repository in PowerShell, sync test dependencies, install the
CUDA-enabled torch/torchvision builds into the project environment using the
selector's index URL, and run the test directly through that environment's
Python (to avoid resyncing to the default lock while testing):

```powershell
uv sync --locked --group dev --extra integration
$cudaIndexUrl = "REPLACE_WITH_URL_FROM_PYTORCH_SELECTOR"
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url $cudaIndexUrl
$env:OPENCLIP_GEMMA_RUN_INTEGRATION = "1"
.\.venv\Scripts\python.exe -m pytest tests\integration\test_cuda_integration.py -v
```

Ensure `torch.cuda.is_available()` is true and the pinned Hugging Face model
revision is cached or can be downloaded. The test reports GPU name, model load
time, inference time, peak allocated GPU memory, and CPU-vs-CUDA maximum
absolute error and cosine similarity for image and Japanese text embeddings.
FP32 uses `max_abs_error <= 1e-3` and cosine `>= 0.9999`; BF16 uses
`max_abs_error <= 0.05` and cosine `>= 0.995`. BF16 is skipped when the device
does not support it. This environment has CPU-only PyTorch, so the procedure
has not been exercised on actual CUDA hardware.

## Dependency and tooling references

- [OpenCLIP project and package](https://github.com/mlfoundations/open_clip)
- [EmbeddingGemma 2 pinned model revision](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b)
- [PyTorch package metadata](https://pypi.org/project/torch/)
- [torchvision package metadata](https://pypi.org/project/torchvision/)
- [Transformers package metadata](https://pypi.org/project/transformers/)
- [Pillow package metadata](https://pypi.org/project/pillow/)
- [Sentence Transformers package metadata](https://pypi.org/project/sentence-transformers/)
- [pytest package metadata](https://pypi.org/project/pytest/)
- [Ruff package metadata](https://pypi.org/project/ruff/)
- [uv-build package metadata](https://pypi.org/project/uv-build/)
