# OpenVLCLIP v0.6.0 Evaluation and Benchmark Report

## Scope and method

`benchmarks/benchmark_cpu.py` runs the same CPU FP32 smoke procedure for
`embeddinggemma-2` and `dinov2-dinotxt`. It reports model/tokenizer load time,
single/batched image and text encoder latency, process RSS, native embedding
dimensions, a one-image temporary index's serialized/vector size and search
latency, and dino.txt patch/similarity-map latency.

The harness generates one deterministic 256-by-192 RGB image and four fixed
texts. The generated image is only an execution-path fixture. It is not
evaluation data and must not be used to claim accuracy, retrieval quality, or
model superiority. Index bytes are for one synthetic image and are not
representative of a real collection.

Run the two models in separate processes from the v0.6.0 source checkout:

```powershell
uv sync --locked --group dev --extra benchmark --extra dinotxt
uv run --locked python benchmarks/benchmark_cpu.py --model embeddinggemma-2 --warmup 2 --iterations 5 --output gemma-cpu.json
uv run --locked python benchmarks/benchmark_cpu.py --model dinov2-dinotxt --warmup 2 --iterations 5 --output dinotxt-cpu.json
```

The dino.txt command downloads the pinned external weights if they are not
cached. The script records Python/OS/CPU/PyTorch versions, model and revision,
preprocessed tensor shapes, precision, warmup count, and measurement count.
Load time can include network transfers and cache state; compare it only when
those conditions are recorded. Encoder timing excludes preprocessing and
tokenization. CPU peak RSS is sampled and can miss short-lived allocations.

For dataset-based EmbeddingGemma MRL retrieval measurements, use
`benchmarks/benchmark_mrl.py` with a labeled query manifest. Its output is
specific to EmbeddingGemma dimensions; do not compare those MRL dimensions to
dino.txt vectors.

## Results

The harness is a runtime smoke benchmark, not a quality evaluation. Both
commands were run with one warmup, three measured iterations, and four CPU
threads. Model files were already cached.

Environment: Windows 11, Python 3.12.10, PyTorch 2.14.1+cpu, Intel64 Family 6
Model 183 CPU. The same generated 256-by-192 image was used. EmbeddingGemma's
preprocessor produced a 672-by-912 tensor; dino.txt produced a 224-by-224
tensor. Text input shapes were `[4, 9]` and `[4, 77]`, respectively. Because
these model-specific resolutions, token shapes, embedding dimensions, and
architectures differ, the timings below are descriptive measurements for
this host only and are not a fair model-performance ranking.

| Metric | EmbeddingGemma 2 | DINOv2 + dino.txt | Conditions |
| --- | --- | --- | --- |
| Dataset, split, and license | not evaluated | not evaluated | No licensed labeled dataset was supplied. |
| Cosine distribution / ranking | not evaluated | not evaluated | Synthetic input is not evaluation data. |
| Zero-shot top-1 / top-5 | not evaluated | not evaluated | No labeled classification dataset. |
| Retrieval Recall@K | not evaluated | not evaluated | No labeled retrieval manifest. |
| Model + processor load | 8,150 ms | 13,229 ms | Weights were cached; tokenizer load is separate. |
| Tokenizer/config load | 2,178 ms | 96 ms | Model-specific tokenizer/config. |
| Image encode, batch 1 | 4,001 ms | 450 ms | Median; preprocessed tensor shapes differ. |
| Image encode, batch 2 | 9,009 ms | 914 ms | Median; preprocessed tensor shapes differ. |
| Text encode, batch 1 | 48 ms | 255 ms | Median; text token shapes differ. |
| Text encode, batch 4 | 71 ms | 868 ms | Median; text token shapes differ. |
| Native global embedding dimension | 768 | 2048 | Backend metadata. |
| One-image index build | 4,428 ms | 576 ms | Includes preprocessing, model inference, and index write. |
| One-image serialized index / vector bytes | 4,492 / 3,072 | 9,905 / 8,192 | Not representative of a real collection. |
| One-query index search | 2,358 ms | 360 ms | Median; includes query encoding and index loading. |
| Peak process RSS | 4,331,184,128 bytes | 6,294,941,696 bytes | Sampled peak; includes runtime and loaded weights. |
| Patch feature / similarity map | not supported | 585 / 844 ms | dino.txt only; map shape `[1, 16, 16]`, patch shape `[1, 256, 1024]`. |
| CUDA runtime / GPU memory | not evaluated | not evaluated | The current validation environment has no CUDA runtime or available GPU. |

All latency values are medians over three measured iterations after one
warmup; model load and one-image index build were single observations. No
synthetic result is used as evidence of a quality or performance improvement.
Dataset-dependent model comparison remains `not evaluated`.

## Reproducibility and limitations

- The DINOv2 source commit, model artifact URLs, sizes, and SHA-256 hashes are
  listed in [the dino.txt integration reference](dinotxt-v0.6.0.md).
- The benchmark uses FP32, one CPU device, one generated image, a two-image
  encoder batch, and four fixed text strings; it does not measure CUDA,
  BF16, application batch distributions, or a representative index.
- The benchmark reports dino.txt patch and similarity-map execution paths but
  does not compare them with EmbeddingGemma, which does not implement those
  capabilities.
- The current environment's CUDA status and CPU BF16/FP32 official-parity
  results are maintained in the dino.txt reference and the final Phase 4
  validation record.
- Licensing for the dino.txt Vision Head and Text Encoder checkpoints remains
  unresolved. Do not distribute the weights or declare a release ready until
  the rights are clarified.
