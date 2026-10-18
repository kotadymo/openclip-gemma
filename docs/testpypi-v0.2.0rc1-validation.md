# TestPyPI validation record for v0.2.0rc1

## Publication status

The `v0.2.0rc1` release candidate was published to TestPyPI on 2026-10-18.
The tag-triggered workflow completed successfully, including release-source
checks, wheel/sdist builds and inspection, and the TestPyPI Trusted Publishing
job. At the time of this RC publication, the v0.2.0 final release was not
available on production PyPI and v0.1.1 was the production release.

The workflow filename is
`.github/workflows/publish-testpypi-v0.2.0rc1.yml`. Register or update the
TestPyPI Trusted Publisher with:

- Owner: `kotadymo`
- Repository: `openclip-gemma`
- Workflow filename: `publish-testpypi-v0.2.0rc1.yml`
- GitHub environment: `testpypi`

The repository has a GitHub `testpypi` environment, and its existing
`publish-testpypi.yml` uses that environment. TestPyPI's account-side Trusted
Publisher registration is not exposed by the repository; the successful
workflow run confirms that the publisher used for this RC was accepted.

## Safe installation source separation

Use a clean virtual environment. Install a compatible PyTorch/torchvision pair
from the official PyTorch instructions for the target CPU or CUDA environment.
Install the remaining runtime dependencies from PyPI, then install only the
`openclip-gemma` candidate from TestPyPI with dependency resolution disabled:

```bash
python -m pip install --index-url https://pypi.org/simple \
  "numpy>=1.26,<3" "pillow>=13,<14" \
  "transformers>=5.20.1,<6"

# Install torch and torchvision together using the official PyTorch selector.

python -m pip install --index-url https://test.pypi.org/simple \
  --no-deps "openclip-gemma==0.2.0rc1"
python -m pip check
```

Do not combine TestPyPI and PyPI using `--extra-index-url`: pip may choose
packages from either index during dependency resolution. `--no-deps` ensures
the candidate project itself comes from TestPyPI while its dependencies come
from their intended sources. For video tests, install the optional decoder
separately from PyPI:

```bash
python -m pip install --index-url https://pypi.org/simple "av>=19.0.1,<20"
```

## Clean-install and inference results

The following checks were performed in new Python 3.12.10 virtual environments.
The package was installed from TestPyPI with `--no-deps`; runtime dependencies
and PyAV were installed from PyPI. The import paths resolved to each temporary
environment's `site-packages`, not a local editable checkout.

- CPU: PyTorch 2.14.1+cpu and PyAV 19.0.1. Quick Start, Japanese text,
  classification-prompt examples, batch image embeddings, saved image
  index/search, synthetic MP4 frame indexing/search, and playback-relative
  timestamps passed.
- CUDA: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB, driver 610.88, PyTorch
  2.14.1+cu130, torchvision 0.29.1+cu130, CUDA runtime 13.0, and PyAV 19.0.1.
  FP32 embeddings, batch embedding, image index/search, and synthetic MP4 frame
  indexing/search passed. PyTorch peak allocated memory was approximately
  3,135 MiB.
- The base environment imported batch and image-search APIs without PyAV.
  The `video` extra resolved `av==19.0.1`.
- Synthetic-image and synthetic-video results verify execution only; no
  real-world search-quality evaluation was performed. BF16 inference was not
  repeated for this RC validation. Ubuntu/Linux real-model inference and
  macOS were not verified by this RC validation.

Model files are downloaded from Hugging Face on first use unless they are
already cached. A successful install or a synthetic-media test does not by
itself establish real-world retrieval quality. See the
[v0.2.0 release checklist](release-checklist-v0.2.0.md) for the production
release workflow and external release steps.
