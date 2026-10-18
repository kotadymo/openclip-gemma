# Video frame search prototype (v0.3.0 development)

## Status and scope

This branch contains an internal prototype for indexing sampled video frames and searching them with Japanese or English text. The implementation is not exported from `openclip_gemma`, is not part of the released `openclip-gemma==0.1.1` package, and does not define a stable public API. It reuses the existing image encoder, preprocessing transform, `SearchQuery` tokenizer prompt, and safe index archive helpers.

The prototype indexes frames independently. It does not recognize motion, event order, or temporal relationships across frames. Search scores are cosine similarities, not probabilities or calibrated confidence.

## Decoder choice and supported inputs

PyAV is used as an optional decoder (`av>=19.0.1,<20`, installed through the `video` extra). It exposes decoded frames' presentation timestamps and time bases and can distribute FFmpeg-backed wheels, avoiding a separate system FFmpeg installation for common environments. This is preferable to deriving timestamps from nominal FPS, especially for variable-frame-rate (VFR) input. OpenCV is not used.

The candidates were compared as follows:

| Candidate | Installation coverage considered | Timing characteristics for this use | Decision |
| --- | --- | --- | --- |
| PyAV | PyAV 19.0.1 publishes CPython stable-ABI wheels for the target Windows x64 and Linux x86_64 environments; the package declares Python 3.12 or newer. The actual prototype was installed and exercised on Windows/Python 3.12 only. | `VideoFrame` exposes PTS and a time base, from which the presentation time can be derived per decoded frame. This gives the implementation an explicit basis for VFR timestamps. | Selected: straightforward per-frame PTS handling and FFmpeg-backed wheels. |
| OpenCV headless | Examined package metadata also showed stable-ABI wheels for Windows and Linux. | OpenCV exposes capture-position properties, but exact decoded-frame timestamp behavior across backends/VFR input was not validated in this work. | Not selected: this prototype requires per-frame presentation timestamps with no FPS-estimated fallback. |

The wheel metadata comparison is not a substitute for install or decode tests on every Python patch and OS. See the [PyAV frame API](https://pyav.org/docs/stable/api/frame.html), [PyAV installation guide](https://pyav.org/docs/stable/overview/installation.html), [PyAV on PyPI](https://pypi.org/project/av/19.0.1/), and [OpenCV video I/O properties](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html).

MP4 is the primary target. The scanner accepts `.mp4`, `.m4v`, `.mov`, `.mkv`, `.webm`, and `.avi` filename suffixes, but a suffix does not guarantee that the bundled FFmpeg build can decode every container/codec combination. Unsupported, corrupt, or empty videos fail with an error; the indexer does not silently omit them. Missing PTS, unusable time bases, and non-monotonic presentation timestamps are also explicit errors. There is no estimated-FPS fallback.

The released `openclip-gemma==0.1.1` does not contain this extra or the video-search module. To use the prototype, install from a development source checkout:

```bash
uv sync --locked --group dev --extra video
```

The optional dependency keeps video decoding out of base installs. The current prototype has been exercised with PyAV 19.0.1 on the development Windows/Python 3.12 environment; PyAV installation and actual video decoding have not been verified across every supported Python patch, Linux, or macOS.

## Sampling and timestamp semantics

The decoder processes frames sequentially and does not retain the whole video or every decoded image. For a positive interval `I`, the first decoded frame establishes the first target; each following target advances by `I` seconds. At each target, the first frame at or after that target is selected. If no frame occurs exactly on a target, the actual selected time can be later. Several targets crossed by one frame produce only one sampled frame rather than duplicate embeddings.

Each sample's time in seconds is computed from the decoder's presentation timestamp and its time base (`pts * time_base`). A frame time base is preferred; the video stream time base is used only when the frame time base is unavailable. Times are not estimated from FPS. VFR streams therefore retain the decoder's actual presentation time, and sampling targets can be unevenly spaced in the selected output. A large interval can miss short-lived scenes.

Frame images are converted to RGB and passed through the existing official image preprocessing transform. Frames are grouped by processed Tensor shape and encoded in bounded batches. `--batch-size` limits the number of image Tensors sent through the model together; decoded-frame and index memory still depend on video duration and index size.

## Indexing and updates

Create or update an index, optionally including subdirectories:

```bash
uv run python examples/video_search.py index \
  --videos ./videos --output ./video-index --interval 1.0 \
  --batch-size 8 --device cpu --recursive
```

Omit `--recursive` to scan only the selected directory. `--device` accepts `cpu` or `cuda`; CPU is the default. Model weights are downloaded from Hugging Face on first use unless already cached. Reduce the batch size if CUDA runs out of memory; the reported error identifies the video and batch being encoded.

The output is a `video-index.npz` archive distinct from the image-search `index.npz`. It contains UTF-8 JSON metadata and an `N x 768` float32 embedding matrix. Metadata records:

- Index kind and format version.
- Model ID and pinned model revision, precision, embedding dimension, and preprocessing identifier.
- Resolved video root, recursion setting, requested interval, and sampling method.
- Each relative video path, file size, and `mtime_ns`.
- Each sampled frame's video path, PTS, time base, and derived timestamp in seconds.
- The embedding matrix checksum.

The loader uses `numpy.load(..., allow_pickle=False)` and validates archive members, metadata, paths, counts, dimensions, finite values, checksum, model revision, precision, preprocessing, and requested sampling interval. Image and video archives are not interchangeable. Writes use a temporary file followed by an atomic replacement, so a failed decode or embedding does not replace the previous complete index.

Re-running the index command reconciles additions, modifications, and removals. A file is considered changed when its relative path is new or its size or `mtime_ns` differs. Rewriting a video while preserving both size and modification time can evade detection; touching or rewriting identical content can trigger an unnecessary full re-embedding. The detector does not hash source videos. Use a new output directory or remove `video-index.npz` for a full rebuild. The full embedding matrix and frame metadata are loaded into system memory for search; the implementation is a brute-force baseline, not an approximate-nearest-neighbor index.

## Text search

Search a saved index:

```bash
uv run python examples/video_search.py search \
  --index ./video-index --query "人がドリルで穴を開けている" \
  --top-k 5 --device cpu
```

The query is encoded with `get_tokenizer(prompt_name="SearchQuery")` and the existing `encode_text()` method. The normalized query is compared against normalized frame embeddings by dot product. Results are sorted by descending cosine similarity; ties are resolved deterministically by relative path, timestamp, and PTS. `top_k` must be positive, and an empty index or blank query is an error.

Output includes the rank, video path, timestamp (`HH:MM:SS.mmm`), and raw cosine score. Use the timestamp to seek in the original video. Scores must not be interpreted as probabilities. This prototype does not create frame previews or clip-level summaries.

## Validation

Model-independent tests mock PyAV and the image encoder to cover PTS/VFR sampling, interval validation, corrupt and empty inputs, archive round trips, incremental changes, compatibility checks, stable ranking, and safe errors.

Opt-in integration tests create a small synthetic MP4 and use the pinned real model to exercise CPU indexing and Japanese text search, plus CUDA/CPU FP32 comparison when CUDA is available:

```powershell
uv sync --locked --group dev --extra video --extra integration
$env:OPENCLIP_GEMMA_RUN_INTEGRATION = "1"
uv run pytest tests/integration/test_video_search_integration.py -v -s
```

The synthetic clip verifies decoder, timestamp, embedding, and query execution paths only; it is not an evaluation of retrieval quality on real-world footage. CUDA tests skip when CUDA is unavailable. Linux and macOS real-model video inference remain unverified until run on those platforms.
