# Image indexing and text search

## Status and scope

The image-index API builds a local index and searches it with Japanese or English text. The APIs are available from `openclip_gemma.image_search`. Video and frame search are documented separately.

The indexer supports JPEG, PNG, and WebP files. It uses the pinned EmbeddingGemma 2 model, the supplied official image preprocessing transform, and the `encode_image()` and `encode_text()` methods. The default remains 768 dimensions; v0.4.0 also supports 512, 256, and 128 dimensions through MRL truncation and re-normalization.

## Indexing and updates

`examples/image_search.py index` scans one directory level by default; pass `--recursive` to include subdirectories. Symlinks are skipped. Supported paths are stored relative to the resolved image root in canonical POSIX form, and duplicate normalized paths are not registered twice.

Running the index command again against the same output directory updates the existing index:

- Added files are embedded.
- Files whose size or `mtime_ns` changed are embedded again.
- Deleted files are removed.
- Unchanged files reuse their saved vectors.

The update detector deliberately avoids hashing every source image. A file rewritten with the same size and modification time can therefore be missed. Conversely, touching a file or rewriting identical image content with a different size or timestamp triggers an unnecessary re-embedding. Use a fresh index directory or remove its `index.npz` file to force a full rebuild. A directory that has no supported images cannot create a new index; if all previously indexed images are removed, the existing index is updated to an empty index and search reports that it has no entries.

The index operation fails on unreadable or corrupt images. It does not silently omit them. Because the new archive is written only after all changed images have been embedded and checked, a failed update leaves the previous archive intact.

## On-disk format and safety

The output directory contains a single `index.npz` archive with:

- A one-dimensional `uint8` array containing UTF-8 JSON metadata.
- An `N x D` `float32` embedding matrix in the same record order, where `D` is 128, 256, 512, or 768.

Format version 2 stores the selected embedding dimension. Existing version-1 indexes remain readable only at 768 dimensions. Metadata includes the format version, absolute image root, recursion setting, model ID and pinned revision, model precision (`fp32` or `bf16`), embedding dimension, preprocessing identifier, entry count, relative image paths, source file sizes and `mtime_ns` values, storage dtype, and a SHA-256 checksum of the embedding matrix.

The loader checks the ZIP directory and NPY headers before loading arrays, then uses `numpy.load(..., allow_pickle=False)` and validates archive members, metadata, path safety, matrix dimensions and dtype, finite values, checksum, model revision, precision, and preprocessing identifier. It rejects archives larger than 512 MiB or with more than 512 MiB of expanded members, and limits metadata to 64 MiB. These hard limits bound a single archive but mean very large collections must be split across indexes. Incompatible or damaged archives fail explicitly; they are not silently discarded or rebuilt.

Updates are written to a temporary file in the output directory, flushed, and atomically installed with `os.replace`. A process interruption before replacement preserves the previous complete index; an unused temporary file may remain.

The cache is local to its recorded image root and should not be treated as a portable database. The current implementation loads the full embedding matrix into system memory. It bounds image preprocessing and GPU work by `--batch-size` but does not use an approximate-nearest-neighbor library; very large indexes may require substantial RAM and search time.

Changing only the `recursive` setting updates the saved metadata and reuses unchanged vectors where files remain in scope. File updates are detected using path, size, and `mtime_ns`; same-size rewrites that preserve timestamps can be missed.

## Text search

Search reads the index's stored dimension, applies `get_tokenizer(prompt_name="SearchQuery")`, and asks `encode_text()` for a query vector of that dimension. Image vectors and query vectors are stored or compared as normalized float32 vectors by dot product, i.e. cosine similarity.

Results are ordered by descending score. Equal scores are resolved by relative image path (case-insensitive first, then exact path) for deterministic output. `top_k` must be positive; values larger than the index return all entries. Empty indexes and blank queries are errors. Scores are raw cosine similarities, not classification probabilities or calibrated confidence.

Scoring operates on bounded blocks of 4,096 image vectors rather than materializing a query-by-index score matrix. The full embedding matrix is still resident in memory as noted above.

## CLI

From the repository root in a v0.2.0 source checkout:

```bash
uv sync --locked --group dev
uv run python examples/image_search.py index \
  --images ./photos --output ./image-index-256 --batch-size 16 \
  --embedding-dim 256 --device cuda --recursive
uv run python examples/image_search.py search \
  --index ./image-index-256 --query "富士山と青空" --top-k 5 --device cuda
```

Omit `--recursive` to index only files directly inside the directory. `--device` accepts `cpu` or `cuda`; CPU is the default. Reduce `--batch-size` if GPU memory is insufficient. Re-running the index command reconciles additions, changes, and deletions.

## Embedding dimensions and migration

Pass `embedding_dim=128`, `256`, `512`, or `768` to
`build_or_update_image_index()` or `--embedding-dim` to the CLI. The default is
768. One index must use a single dimension: an update that requests a different
dimension raises `ImageIndexCompatibilityError` instead of mixing vectors.
Build the alternate dimension in a separate index directory.

`convert_image_index(source, destination, embedding_dim=...)` creates a new
index without loading the model or rerunning inference. The destination must
not exist, and the target dimension cannot exceed the source dimension. The
conversion truncates each stored vector to its leading dimensions, re-normalizes
it, updates the checksum and metadata, and leaves the source archive unchanged.
This supports conversion from legacy version-1 768-dimensional indexes as
well as version-2 indexes.

## Validation and limitations

Model-independent unit tests cover supported extensions, ordering, incremental updates, safe save/load, corruption and compatibility rejection, Japanese query routing, top-k order, stable ties, and empty/error cases. An opt-in integration test exercises the real model on CPU and CUDA when available:

```powershell
$env:OPENCLIP_GEMMA_RUN_INTEGRATION = "1"
uv run pytest tests/integration/test_image_search_integration.py -v
```

Integration results validate embedding and search execution, not semantic retrieval quality. Synthetic test images must not be treated as evidence of recognition accuracy. Brute-force cosine search is appropriate for an initial correct implementation; FAISS or other indexing acceleration is deferred.

The APIs are `build_or_update_image_index()` and `search_images()` in `openclip_gemma.image_search`; the batch iterator is in `openclip_gemma.batching`. They are not re-exported from the package root. No changes are made to the existing OpenCLIP-style API.
