# Video frame search

## Status and scope

The APIs index sampled video frames and search them with Japanese or English text from `openclip_gemma.video_search`. They are not re-exported from the package root. They reuse the image encoder, preprocessing transform, `SearchQuery` tokenizer prompt, and safe index archive helpers. The default is 768 dimensions; v0.4.0 also supports 512, 256, and 128 dimensions through MRL truncation and re-normalization.

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

Install the optional video dependencies with `pip install "openclip-gemma[video]"`. To run the CLI examples, use a source checkout:

```bash
uv sync --locked --group dev --extra video
```

The optional dependency keeps video decoding out of base installs. PyAV 19.0.1 has been exercised on Windows/Python 3.12. PyAV installation and actual video decoding have not been verified across every supported Python patch, Linux, or macOS.

## Sampling and timestamp semantics

The decoder processes frames sequentially and does not retain the whole video or every decoded image. `timestamp_seconds` is relative to the container start time when available, otherwise the video stream start time, otherwise the first decoded frame. Sampling targets start at relative time zero. Frames preceding a declared playback origin are ignored. For a positive interval `I`, the first frame at or after each target is selected; each following target advances by `I` seconds. If no frame occurs exactly on a target, the actual selected time can be later. Several targets crossed by one frame produce only one sampled frame rather than duplicate embeddings.

Each sample retains the original PTS and rational frame time base, along with the exact rational playback origin used to calculate its relative timestamp. A frame time base is preferred; the video stream time base is used only when the frame time base is unavailable. Times are not estimated from FPS. VFR streams therefore retain the decoder's actual presentation time, and sampling targets can be unevenly spaced in the selected output. A large interval can miss short-lived scenes.

Frame images are converted to RGB and passed through the existing official image preprocessing transform. Frames are grouped by processed Tensor shape and encoded in bounded batches. `--batch-size` limits the number of image Tensors sent through the model together; decoded-frame and index memory still depend on video duration and index size.

## Indexing and updates

Create or update an index, optionally including subdirectories:

```bash
uv run python examples/video_search.py index \
  --videos ./videos --output ./video-index --interval 1.0 \
  --batch-size 8 --embedding-dim 256 --device cpu --recursive
```

Omit `--recursive` to scan only the selected directory. `--device` accepts `cpu` or `cuda`; CPU is the default. Model weights are downloaded from Hugging Face on first use unless already cached. Reduce the batch size if CUDA runs out of memory; the reported error identifies the video and batch being encoded.

The output is a version-3 `video-index.npz` archive distinct from the image-search `index.npz`. Version 2 stored playback-relative timestamps and exact timestamp origins and remains readable at 768 dimensions. Version-1 indexes used absolute PTS-derived timestamps and are rejected; rebuild them rather than reusing them. Version 3 stores UTF-8 JSON metadata and an `N x D` float32 embedding matrix, where `D` is 128, 256, 512, or 768. Metadata records:

- Index kind and format version.
- Model ID and pinned model revision, precision, embedding dimension, and preprocessing identifier.
- Resolved video root, recursion setting, requested interval, and sampling method.
- Each relative video path, file size, and `mtime_ns`.
- Each sampled frame's video path, PTS, time base, playback origin, and relative timestamp in seconds.
- The embedding matrix checksum.

New version-3 indexes include a deterministic `embedding_space_id` over the
backend, model and revision, dimension and strategy, normalization,
preprocessing, prompts, precision, storage dtype, and sampling method. Its
SHA-256 input is a compact UTF-8 JSON array of field/value pairs in a fixed
order; null values are explicit and Python's randomized `hash()` is not used.
Search and updates reject same-dimension indexes from a different embedding
space. Existing version-2 and v0.4.0 version-3 indexes remain readable only
when their legacy fields identify the pinned EmbeddingGemma 2 configuration.

The loader preflights ZIP member names, NPY array headers, archive size, expanded member size, and metadata size before loading arrays; NumPy loading also has pickle disabled. It then validates metadata, paths, counts, dimensions, finite values, checksum, model revision, precision, preprocessing, and requested sampling interval. The archive and expanded members are each limited to 512 MiB, and metadata is limited to 64 MiB. Image and video archives are not interchangeable. Writes use a temporary file followed by an atomic replacement, so a failed decode or embedding does not replace the previous complete index.

Re-running the index command reconciles additions, modifications, and removals. A file is considered changed when its relative path is new or its size or `mtime_ns` differs. Rewriting a video while preserving both size and modification time can evade detection; touching or rewriting identical content can trigger an unnecessary full re-embedding. The detector does not hash source videos. Use a new output directory or remove `video-index.npz` for a full rebuild. The full embedding matrix and frame metadata are loaded into system memory for search; the implementation is a brute-force baseline, not an approximate-nearest-neighbor index.

## Text search

Search a saved index:

```bash
uv run python examples/video_search.py search \
  --index ./video-index --query "人がドリルで穴を開けている" \
  --top-k 5 --device cpu
```

The query is encoded with `get_tokenizer(prompt_name="SearchQuery")` and `encode_text()` at the dimension stored in the index. The normalized query is compared against normalized frame embeddings by dot product. Results are sorted by descending cosine similarity; ties are resolved deterministically by relative path, timestamp, and PTS. `top_k` must be positive, and an empty index or blank query is an error.

## Embedding dimensions and migration

Pass `embedding_dim=128`, `256`, `512`, or `768` to
`build_or_update_video_index()` or `--embedding-dim` to the CLI. The default is
768. An existing index cannot be incrementally updated at a different
dimension; build the alternate dimension in a separate directory.

`convert_video_index(source, destination, embedding_dim=...)` converts stored
frame vectors without decoding video or loading the model. The destination
must not exist, and the target dimension cannot exceed the source dimension.
The conversion truncates and re-normalizes every vector, recalculates its
checksum, updates format metadata, and leaves the source index unchanged.
Version-2 768-dimensional indexes can be converted; version-1 timestamp
indexes remain unsupported.

Output includes the rank, video path, timestamp (`HH:MM:SS.mmm`), and raw cosine score. Use the timestamp to seek in the original video. Scores must not be interpreted as probabilities. This prototype does not create frame previews or clip-level summaries.

## Validation

The APIs are `build_or_update_video_index()` and `search_video()` in `openclip_gemma.video_search`. Model-independent tests mock PyAV and the image encoder to cover PTS/VFR sampling, zero and nonzero start times, interval validation, corrupt and empty inputs, archive round trips, incremental changes, compatibility checks, stable ranking, and safe errors.

Opt-in integration tests create a small synthetic MP4 and use the pinned real model to exercise CPU indexing and Japanese text search, plus CUDA/CPU FP32 comparison when CUDA is available:

```powershell
uv sync --locked --group dev --extra video --extra integration
$env:OPENCLIP_GEMMA_RUN_INTEGRATION = "1"
uv run pytest tests/integration/test_video_search_integration.py -v -s
```

The synthetic clip verifies decoder, timestamp, embedding, and query execution paths only; it is not an evaluation of retrieval quality on real-world footage. CUDA tests skip when CUDA is unavailable. Linux and macOS real-model video inference remain unverified until run on those platforms.
