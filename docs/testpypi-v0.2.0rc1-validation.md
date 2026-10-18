# TestPyPI validation plan for v0.2.0rc1

## Publication status

The `v0.2.0rc1` release candidate and its tag-triggered workflow are prepared,
but the workflow has not been triggered and the package has not been published
to TestPyPI. Publication requires separate approval, a `v0.2.0rc1` tag that
points to a CI-passing commit reachable from `main`, and a correctly configured
TestPyPI Trusted Publisher.

The workflow filename is
`.github/workflows/publish-testpypi-v0.2.0rc1.yml`. Register or update the
TestPyPI Trusted Publisher with:

- Owner: `kotadymo`
- Repository: `openclip-gemma`
- Workflow filename: `publish-testpypi-v0.2.0rc1.yml`
- GitHub environment: `testpypi`

The repository has a GitHub `testpypi` environment, and its existing
`publish-testpypi.yml` uses that environment. TestPyPI's account-side Trusted
Publisher registration is not exposed by the repository; verify it in the
TestPyPI project settings before creating the release tag.

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

## Post-publication checks

Run these checks in separate clean CPU and CUDA environments where available:

1. Confirm the installed version is `0.2.0rc1`.
2. Import `openclip_gemma.batching.iter_image_embeddings`,
   `openclip_gemma.image_search.build_or_update_image_index`,
   `openclip_gemma.image_search.search_images`,
   `openclip_gemma.video_search.build_or_update_video_index`, and
   `openclip_gemma.video_search.search_video`.
3. Run the v0.1.1 Quick Start calls to `create_model_and_transforms`,
   `get_tokenizer`, `encode_image`, and `encode_text` to check the original API.
4. Embed a small set of real local images, save and reload an image index, and
   search with a Japanese query. Repeat with CUDA only when CUDA is actually
   available.
5. Install `av` from PyPI, index a short local video, reload the video index,
   and search it with a Japanese query. Confirm returned paths and playback-
   relative timestamps. Repeat on CUDA only on a machine with an available GPU.
6. Check that the base installation works without PyAV and that video calls
   report the missing optional dependency clearly.

Model files are downloaded from Hugging Face on first use unless they are
already cached. A successful install or a synthetic-media test does not by
itself establish real-world retrieval quality.
