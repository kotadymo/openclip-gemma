# Image indexing and text search

## Status and scope

The image-index API builds a local index and searches it with Japanese or English text. The APIs are available from `openvlclip.image_search`; video and frame search are documented separately. The selected registered backend controls preprocessing, tokenizer routing, native dimension, and supported truncation.

The indexer supports JPEG, PNG, and WebP files and uses the selected model's official preprocessing and image encoder. When `embedding_dim` is omitted, it uses the backend's native dimension: 768 for EmbeddingGemma 2, 2048 for DINOv2 + dino.txt, and 768 for the registered SigLIP 2 models. EmbeddingGemma 2 supports its MRL dimensions 768, 512, 256, and 128. dino.txt and SigLIP 2 do not support truncation, so unsupported dimensions are rejected.

## Indexing and updates

`examples/image_search.py index` scans one directory level by default; pass `--recursive` to include subdirectories. Symlinks are skipped. Supported paths are stored relative to the resolved image root in canonical POSIX form, and duplicate normalized paths are not registered twice.

Running the index command again against the same output directory updates the existing index:

- Added files are embedded.
- Files whose size or `mtime_ns` changed are embedded again.
- Deleted files are removed.
- Unchanged files reuse their saved vectors.

By default, the update detector avoids hashing source files and compares path, size, and `mtime_ns`. A file rewritten with the same size and modification time can therefore be missed; touching identical content can trigger an unnecessary re-embedding. Pass `verify_contents=True` to `build_or_update_image_index()` or `--verify-contents` to the CLI to compare SHA-256 file digests. Strict mode reads each source image to hash it during scanning, then rechecks files before saving a changed index, so it adds disk I/O. Strictly updated records store their content digest. An older index without digests is safely re-embedded once when strict verification is enabled. A directory that has no supported images cannot create a new index; if all previously indexed images are removed, the existing index is updated to an empty index and search reports that it has no entries.

The index operation fails on unreadable or corrupt images. It does not silently omit them. Because the new archive is written only after all changed images have been embedded and checked, a failed update leaves the previous archive intact.

## On-disk format and safety

The output directory contains a single `index.npz` archive with:

- A one-dimensional `uint8` array containing UTF-8 JSON metadata.
- An `N x D` `float32` embedding matrix in the same record order, where `D` is validated by the selected backend.

Format version 2 stores the selected embedding dimension. Existing version-1 indexes remain readable only at 768 dimensions. Metadata includes the format version, absolute image root, recursion setting, model ID and pinned revision, model precision (`fp32` or `bf16`), embedding dimension, preprocessing identifier, entry count, relative image paths, source file sizes and `mtime_ns` values, storage dtype, and a SHA-256 checksum of the embedding matrix.

New indexes also record an EmbeddingSpace schema version, input type, index and
query policies, processing parameters, backend ID, normalization, and
`embedding_space_id`. Schema version 1 retains the historical SHA-256 payload;
schema version 2 includes the extension fields in a separate versioned
signature, so an old index's ID is not rewritten. The schema-1 signature is
SHA-256 over a compact UTF-8 JSON array of field/value pairs in this fixed order:
`backend_id`, `model_id`, `model_revision`, `embedding_dimension`,
`embedding_strategy`, `normalization`, `preprocessing_id`, `index_prompt`,
`query_prompt_name`, `query_prompt`, `model_precision`, `storage_dtype`, and
`sampling_method`. Null prompt and sampling values are included explicitly.
This deterministic signature does not use Python's randomized `hash()`.
Updates and searches reject a different backend, model revision, dimension,
normalization, preprocessing, prompt, or precision even when the vector
dimensions match. Existing format-version-1 and older format-version-2
EmbeddingGemma indexes remain readable when their checkpoint, preprocessing,
strategy, dimension, prompts, and precision identify the compatible space.
New indexes explicitly record `input_type="image"` and processor/query policy;
frame, native-video, patch, and text embeddings are not interchangeable.

The loader checks the ZIP directory and NPY headers before loading arrays, then uses `numpy.load(..., allow_pickle=False)` and validates archive members, metadata, path safety, matrix dimensions and dtype, finite values, checksum, model revision, precision, and preprocessing identifier. It rejects archives larger than 512 MiB or with more than 512 MiB of expanded members, and limits metadata to 64 MiB. These hard limits bound a single archive but mean very large collections must be split across indexes. Incompatible or damaged archives fail explicitly; they are not silently discarded or rebuilt.

Updates are written to a temporary file in the output directory, flushed, and atomically installed with `os.replace`. A process interruption before replacement preserves the previous complete index; an unused temporary file may remain.

The cache is local to its recorded image root and should not be treated as a portable database. The current implementation loads the full embedding matrix into system memory. It bounds image preprocessing and GPU work by `--batch-size` but does not use an approximate-nearest-neighbor library; very large indexes may require substantial RAM and search time.

Changing only the `recursive` setting updates the saved metadata and reuses unchanged vectors where files remain in scope. File updates are detected using path, size, and `mtime_ns`; same-size rewrites that preserve timestamps can be missed.

## Text search

Search reads the index's stored dimension and delegates text encoding to the
selected backend. EmbeddingGemma 2 uses its registered `SearchQuery` prompt;
dino.txt uses its official tokenizer without an EmbeddingGemma task prompt.
Backends that support arbitrary query instructions may use
`query_instruction`; the instruction is part of the index's query policy and
must match at search time. EmbeddingGemma 2's registered task prompts are not
free-form query instructions. Image and query vectors are compared as
normalized float32 vectors by dot product, i.e. cosine similarity.

Results are ordered by descending score. Equal scores are resolved by relative image path (case-insensitive first, then exact path) for deterministic output. `top_k` must be positive; values larger than the index return all entries. Empty indexes and blank queries are errors. Scores are raw cosine similarities, not classification probabilities or calibrated confidence.

Scoring operates on bounded blocks of 4,096 image vectors rather than materializing a query-by-index score matrix. The full embedding matrix is still resident in memory as noted above.

## CLI

From the repository root in a source checkout:

```bash
uv sync --locked --group dev
uv run python examples/image_search.py index \
  --images ./photos --output ./image-index-256 --batch-size 16 \
  --model embeddinggemma-2 --embedding-dim 256 --device cuda --recursive
uv run python examples/image_search.py search \
  --model embeddinggemma-2 --index ./image-index-256 \
  --query "富士山と青空" --top-k 5 --device cuda
```

Omit `--recursive` to index only files directly inside the directory. `--device` accepts `cpu` or `cuda`; CPU is the default. Reduce `--batch-size` if GPU memory is insufficient. Re-running the index command reconciles additions, changes, and deletions.

For strict content verification during indexing, add `--verify-contents`:

```bash
uv run python examples/image_search.py index \
  --images ./photos --output ./image-index --verify-contents
```

## Embedding dimensions and migration

Pass an `embedding_dim` supported by the selected backend to
`build_or_update_image_index()` or `--embedding-dim` to the CLI. The default is
the backend's native dimension. EmbeddingGemma 2 accepts 768, 512, 256, or 128;
dino.txt and SigLIP 2 accept only their native 2048- and 768-dimensional
vectors, respectively. One index must use a single dimension: an update that
requests a different dimension raises `ImageIndexCompatibilityError` instead
of mixing vectors. Build the alternate supported dimension in a separate index
directory.

`convert_image_index(source, destination, embedding_dim=...)` creates a new
index without loading the model or rerunning inference. The destination must
not exist, and the target dimension cannot exceed the source dimension. The
conversion truncates each stored vector to its leading dimensions, re-normalizes
it, updates the checksum and metadata, and leaves the source archive unchanged.
This supports conversion from compatible legacy EmbeddingGemma 2 indexes as
well as current indexes. Conversion is limited by the registered backend's
dimension policy; it cannot create an unsupported MRL representation or convert
vectors between different models or embedding strategies.

## Validation and limitations

Model-independent unit tests cover supported extensions, ordering, incremental updates, safe save/load, corruption and compatibility rejection, Japanese query routing, top-k order, stable ties, and empty/error cases. An opt-in integration test exercises the real model on CPU and CUDA when available:

```powershell
$env:OPENCLIP_GEMMA_RUN_INTEGRATION = "1"
uv run pytest tests/integration/test_image_search_integration.py -v
```

Integration results validate embedding and search execution, not semantic retrieval quality. Synthetic test images must not be treated as evidence of recognition accuracy. Brute-force cosine search is appropriate for an initial correct implementation; FAISS or other indexing acceleration is deferred.

The APIs are `build_or_update_image_index()` and `search_images()` in `openvlclip.image_search`; the batch iterator is in `openvlclip.batching`. They are not re-exported from the package root. No changes are made to the existing OpenCLIP-style API.
