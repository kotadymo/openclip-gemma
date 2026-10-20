# MRL Benchmark Guide

`benchmarks/benchmark_mrl.py` measures encoder latency and float32 index
storage at 768, 512, 256, and 128 dimensions using the pinned EmbeddingGemma 2
model. It can also report paired image-to-text and text-to-image recall when
the user supplies one line-aligned caption per supported image.

## Run the benchmark

Install the development environment, then provide an image directory:

```powershell
uv sync --locked --group dev
uv run python benchmarks/benchmark_mrl.py `
  --images .\photos `
  --device cpu `
  --precision fp32 `
  --warmup 1 `
  --iterations 5 `
  --output .\mrl-benchmark.json
```

For paired retrieval metrics, prepare a UTF-8 text file with exactly one
non-empty caption per supported image. Captions must follow the image indexer's
relative-path ordering (platform-normalized path, then exact path):

```powershell
uv run python benchmarks/benchmark_mrl.py `
  --images .\photos `
  --captions-file .\captions.txt `
  --limit 100 `
  --batch-size 8 `
  --device cpu `
  --precision fp32
```

The optional `--recursive` flag includes subdirectories. `--limit` selects
the first sorted images for timing; the captions file still describes the
complete supported image set. `--device cuda` and `--precision bf16` are
available where the installed PyTorch build and hardware support them.

## Report interpretation

The JSON report records the pinned model revision, environment, latency
statistics, output dimensions, estimated float32 bytes per stored vector,
and storage reduction relative to 768 dimensions. Preprocessing and
tokenization are excluded from latency. The image measurement includes model
encoding and restoring input order; the text measurement includes model
encoding and output concatenation. The backbone still computes its native
representation, so truncation reduces embedding and index storage but does
not reduce transformer FLOPs.

Paired recall uses row-aligned image/caption pairs as ground truth and is
reported only when a captions file is supplied. It is a dataset-specific
measurement, not a universal quality guarantee. Choose representative data,
preserve the caption-to-image alignment, and compare each reduced dimension
with the 768-dimensional baseline before selecting a production index size.

No real-world retrieval-quality measurement is included in this repository's
v0.4.0 implementation results because no application dataset or labeled
caption set was supplied. Unit tests validate supported shapes, finite values,
normalization, safe index conversion, and dimension-matched search; they do
not establish semantic retrieval quality. Benchmark output should not be
presented as a package-wide accuracy claim.
