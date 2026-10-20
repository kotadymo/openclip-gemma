# MRL Benchmark Guide

`benchmarks/benchmark_mrl.py` compares 128, 256, 512, and 768-dimensional
image embeddings and indexes using the pinned EmbeddingGemma 2 model. It
measures encoder time, builds real indexes, runs optional text queries, and
records measured storage and memory values.

## Install and run

Install the development dependencies and the optional RSS-measurement
dependency:

```powershell
uv sync --locked --group dev --extra benchmark
uv run python benchmarks/benchmark_mrl.py `
  --images .\photos `
  --device cpu `
  --precision fp32 `
  --warmup 1 `
  --iterations 5 `
  --output .\mrl-benchmark.json
```

`--recursive` includes images in subdirectories. `--limit` selects the first
sorted images for the encoder-latency sample; all supported images are used
when building indexes and measuring query rankings.

For query search latency and ranking agreement, provide a UTF-8 JSON array.
Each entry requires `query` text and may include labeled image paths in
`relevant_images`. Paths are relative to the image root and use `/` separators:

```json
[
  {
    "query": "a red bird",
    "relevant_images": ["birds/red-bird.jpg", "birds/another-red-bird.png"]
  },
  {
    "query": "a mountain landscape"
  }
]
```

Queries without `relevant_images` still contribute search latency and ranking
agreement, but not Recall@K. Run with the manifest as follows:

```powershell
uv run python benchmarks/benchmark_mrl.py `
  --images .\photos `
  --recursive `
  --query-manifest .\queries.json `
  --limit 100 `
  --batch-size 8 `
  --device cpu `
  --precision fp32 `
  --output .\mrl-benchmark.json
```

Use `--format csv` for a one-row-per-dimension summary:

```powershell
uv run python benchmarks/benchmark_mrl.py `
  --images .\photos `
  --query-manifest .\queries.json `
  --format csv `
  --output .\mrl-benchmark.csv
```

Each CSV row includes the model and revision, device and precision, image root,
query manifest, limit/recursion/batch/warmup/iteration settings, and Python,
OS, CPU, PyTorch, CUDA runtime, and device-name metadata.

`--device cuda` and `--precision bf16` can be selected where the installed
PyTorch build and hardware support them. CUDA allocator peak is reported as
`unmeasured` for CPU runs.

## Report interpretation

- Image embedding latency sums model encoder-call time. Image preprocessing,
  host-to-device transfer, and CPU output assembly are excluded.
- Index size and vector bytes are read from each serialized index archive;
  they are measured values, not estimates.
- Query search latency measures the public search operation, including
  tokenization, query encoding, index loading, and ranking.
- Ranking agreement compares each dimension with the 768-dimensional
  reference, reporting positionwise agreement and top-10 set overlap.
- Recall@1, Recall@5, and Recall@10 are calculated only for labeled queries
  and macro-averaged over those queries.
- Process RSS fields are point-in-time measurements after named phases.
  CUDA allocator peak includes the model baseline and reports its increment.
- Metrics that cannot be measured, such as retrieval without a query manifest
  or CUDA allocator use on CPU, are explicitly reported as `unmeasured`.

The backbone still computes its native 768-dimensional representation, so
lower dimensions reduce stored vector size but do not reduce transformer
FLOPs. Use representative, labeled application data before drawing retrieval
quality conclusions. Synthetic images and unlabelled queries exercise the
benchmark but are not evidence of real-world retrieval quality.
