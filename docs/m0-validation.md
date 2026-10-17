# M0: Official Inference Path Validation

- Date: 2026-10-17
- Status: **Partially validated; image API contract remains unresolved**
- Scope: Official EmbeddingGemma 2 inference only. No OpenCLIP-compatible API was implemented.

## Environment and method

- Python 3.12.10 on Windows
- PyTorch 2.14.1+cpu
- Transformers 5.20.1
- Sentence Transformers 6.1.0
- Device: CPU; `torch.cuda.is_available()` was `False`
- FP32 model revision: `914f7f89142e33e77833254d9c9b90c3cef7303b`
- Model: `google/embeddinggemma-2`

The experiment uses the official Sentence Transformers model and independently loads
`AutoProcessor` and `AutoModel` from the same model revision. A deterministic 256x192
RGB image (blue background with a red circle) and two fixed text inputs are generated
in code; no external image dataset is needed. Model files are stored in the Hugging
Face cache, not in this repository.

Run with:

```powershell
uv run --locked --group m0 python experiments\m0\validate.py --device cpu --dtype fp32
uv run --locked --group m0 python experiments\m0\validate.py --device cpu --dtype bf16
```

The script prints structured JSON. Use `--output <path>` only when a local report
artifact is useful.

## Results

### Model loading and device/dtype

Both official loading paths succeeded on CPU. FP32 outputs were finite float32
`[B, 768]` tensors on CPU. The image and text vectors produced by the
Sentence Transformers model had L2 norms approximately 1.0. `forward()` and
`get_logits()` were not exercised or implemented.

### Processor image path

For the generated image, `AutoProcessor(images=image, return_tensors="pt")` returned
these tensors:

| Key | Shape | dtype |
|---|---:|---|
| `input_ids` | `[1, 270]` | `int64` |
| `attention_mask` | `[1, 270]` | `int64` |
| `pixel_values` | `[1, 2520, 768]` | `float32` |
| `image_position_ids` | `[1, 2520, 2]` | `int64` |

The processor output is therefore not a plain `[B, 3, H, W]` image tensor; the
model call consumes the complete processor feature mapping, including auxiliary
IDs, mask, and position data.

Passing the PIL-derived processor mapping directly to `AutoModel`, then applying
attention-mask-aware mean pooling and L2 normalization, matched the official
Sentence Transformers image result:

| Comparison | Max absolute error | Cosine similarity | Provisional FP32 thresholds |
|---|---:|---:|---|
| Official image embedding vs. direct `AutoModel` from PIL processor features | `2.7753e-7` | `1.000000` | Pass |

The image processor also accepted a raw float Tensor shaped `[3, 192, 256]` with
values in `[0, 255]`. Its generated token IDs, attention mask, and position IDs
matched the PIL route exactly, but `pixel_values` differed by up to `0.0040472`.
The resulting embedding comparison was:

| Comparison | Max absolute error | Cosine similarity | Provisional FP32 thresholds |
|---|---:|---:|---|
| Official PIL image embedding vs. processor/`AutoModel` from raw image Tensor | `0.00079144` | `0.9999833` | **Fail** max-absolute-error limit `1e-4`; cosine limit passes |

This is an observed numerical difference between processor PIL and Tensor image
paths. It does not establish that all Tensor inputs or image sizes behave the same.

### Text path, pooling, and normalization

For each text configuration, the official Sentence Transformers result was compared
with `AutoProcessor` + `AutoModel`, followed by mean pooling over
`last_hidden_state` using the processor `attention_mask` and L2 normalization.
All outputs were finite CPU float32 tensors of shape `[2, 768]`.

| Text prompt | Effective prefix | Max absolute error | Cosine similarity per row | Provisional FP32 thresholds |
|---|---|---:|---|---|
| None | `""` | `7.9768e-7` | `[1.000000, 1.000000]` | Pass |
| SearchQuery | `task: search result \| query: ` | `9.3132e-7` | `[1.000000, 1.000000]` | Pass |
| Classification | `task: classification \| query: ` | `1.3069e-6` | `[1.000000, 1.000000]` | Pass |

The displayed cosine values are rounded; the script records the full precision.
These checks validate the pooling and normalization path for the sampled text inputs,
not general task quality.

### Tokenizer and attention-mask recovery

For the official tokenizer, the observed `pad_token_id` was `0` and padding side was
`right`. With two different-length texts, `input_ids != pad_token_id` exactly
reconstructed `attention_mask` for all three tested configurations:

| Prompt | `input_ids` shape | Padded positions | Valid PAD-ID positions | Exact mask match |
|---|---:|---:|---:|---|
| None | `[2, 18]` | 15 | 0 | Yes |
| SearchQuery | `[2, 25]` | 15 | 0 | Yes |
| Classification | `[2, 24]` | 15 | 0 | Yes |

This establishes exact recovery for the tested official-tokenizer batches. It does
not prove mask recovery for arbitrary externally supplied token-ID tensors.

### Prompt comparison

Different prompts materially changed text embeddings. Per-row cosine similarities
between prompt choices were:

| Prompt pair | Text 1 | Text 2 |
|---|---:|---:|
| None vs SearchQuery | `0.9252` | `0.9001` |
| None vs Classification | `0.8470` | `0.7889` |
| SearchQuery vs Classification | `0.9361` | `0.9252` |

For this synthetic image, image/text cosine scores were:

| Prompt | Text 1 | Text 2 |
|---|---:|---:|
| None | `0.7245` | `0.6303` |
| SearchQuery | `0.7689` | `0.6739` |
| Classification | `0.7708` | `0.6819` |

This synthetic example is not a task benchmark and must not be used to select a
universal default prompt. Google documents task-specific prefixes for text, while
images are passed without a text prefix.

### Separate CPU BF16 run

The BF16 run completed on CPU. Outputs were finite, on CPU, and had shape `[B, 768]`.
The Sentence Transformers outputs were BF16; the manual pooled outputs were also
BF16 after preserving the model output dtype. Observed high-level row norms ranged
from `1.0002` to `1.0038`; this quantization behavior is why FP32 normalization
tolerances must not be reused for BF16. Observed comparisons:

| Input | Max absolute error | Cosine similarity |
|---|---:|---|
| Image via PIL processor features | `0.0012207` | `0.9999652` |
| Text, no prompt | `0.0019531` | `[0.9999757, 0.9999782]` |
| Text, SearchQuery | `0.0039063` | `[0.9999744, 0.9999839]` |
| Text, Classification | `0.0019531` | `[0.9999722, 0.9999506]` |

No BF16 acceptance thresholds have been established. This is a successful BF16
execution/smoke test, **not** a numerical-parity pass.

## Assessment

### Successful validations

- Sentence Transformers and Transformers loaded the official model on CPU.
- Direct model inference using the complete PIL-derived image processor mapping
  matched the high-level image result within the provisional FP32 thresholds.
- Direct text inference with mean pooling, attention masking, and normalization
  matched the high-level result for no prompt, SearchQuery, and Classification.
- The tested tokenizer batches produced exactly recoverable attention masks from
  token IDs and the known PAD ID.
- A separate CPU BF16 execution completed and produced finite 768-dimensional
  embeddings.

### Failed or incomplete validations

- The raw Tensor image route did not meet the provisional FP32 maximum absolute
  error limit (`0.00079144 > 0.0001`), despite cosine similarity `0.9999833`.
- BF16 parity is not classified because there is no agreed BF16 threshold.
- The prompt experiment does not determine a best prompt; its data is synthetic.
- CUDA was unavailable and was not tested.
- The tokenizer-mask result does not cover arbitrary external token IDs.

## Initial M0 Recommendation (Superseded by M0.1 Below)

The following recommendation was made before the M0.1 cause analysis. Its statement
that the Tensor path failed was true for M0's float `[0,255]` route, but should not
be read as evidence that all OpenCLIP Tensor inputs inherently fail. M0.1 below
demonstrates an exact-order Tensor preprocess route that passes.

## Recommendation for the OpenCLIP-compatible API

1. **Images:** The exact, validated low-level path passes the complete
   `AutoProcessor` feature mapping to `AutoModel`; it cannot be represented as only
   `[B, 3, H, W]` without a conversion or extra metadata. The tested raw Tensor path
   is close in cosine similarity but fails the current provisional maximum-error
   limit. Given the specification's numerical-correctness priority, prefer an
   explicit processor-feature mapping for `preprocess_val`/`encode_image`, accepting
   and documenting the API difference. Do not implement image inference using the
   raw Tensor path unless a follow-up experiment either reproduces the PIL processor
   output or approves a measured tolerance.
2. **Text:** Keep task prompting explicit. Do not silently apply SearchQuery or
   Classification to every `encode_text` call. The no-prompt, SearchQuery, and
   Classification routes all match their respective official Sentence Transformers
   outputs; the experiment does not select a universal default.
3. **Tokenizer:** A Tensor-only tokenizer output is viable for the tested official
   tokenizer batches if the PAD ID is fixed and the attention mask is rebuilt as
   `input_ids != pad_token_id`. Keep validation limited to generated token IDs and
   document the caveat for arbitrary external tensors.
4. **Pooling:** Use the official model output and its mean-pooling/normalization
   behavior. Do not introduce a separate neural network implementation.
5. **Precision:** Retain FP32 as the baseline. Treat BF16 as separately runnable,
   but do not claim numerical parity until thresholds are chosen and validated.

The image input contract is a design decision that must be resolved before M1;
this report does not silently change that contract.

## M0.1: Exact Tensor Preprocessing Diagnosis

- Date: 2026-10-17
- Model revision: `914f7f89142e33e77833254d9c9b90c3cef7303b` for both inference paths
- Device/dtype: CPU FP32
- Versions: PyTorch 2.14.1+cpu, Transformers 5.20.1, Sentence Transformers 6.1.0
- CUDA: unavailable; no GPU run was performed
- Script: `experiments/m0/diagnose_image_tensor.py`

Run using:

```powershell
uv run --locked --group m0 python experiments\m0\diagnose_image_tensor.py
```

It reuses the exact synthetic M0 RGB image and adds its grayscale conversion plus
wide (`384x128`) and tall (`128x384`) RGB versions. The Sentence Transformers model
was explicitly cast to float32, matching the `AutoModel` dtype.

### Verified official image-processing operations

The fixed-revision `Gemma4ImageProcessor` implementation establishes this order:

1. `process_image` converts PIL input to RGB when configured, then uses
   `torchvision` `pil_to_tensor`, producing a CHW `uint8` Tensor.
2. `aspect_ratio_preserving_resize` calculates target dimensions from the source
   aspect ratio, patch budget, patch size, and pooling kernel. It calls torchvision
   resize with antialiasing.
3. `rescale_and_normalize` multiplies by `1/255`; the checkpoint configuration has
   `do_normalize=False`, so it does not apply mean/std normalization.
4. `convert_image_to_patches` splits the resized pixels into row-major 16x16 patches.
   The patch dimension is padded to 2520. The processor constructs
   `image_position_ids` from patch-grid `(x, y)` coordinates and marks padding
   positions with `-1`.
5. The multimodal processor derives image replacement tokens from the count of
   pooled visual tokens and tokenizes the sequence, producing `input_ids` and
   `attention_mask`.

Fixed-revision configuration observed in the run: resample code `3` (PIL/torchvision
BICUBIC), antialias enabled by the official resize method, `patch_size=16`,
`pooling_kernel_size=3`, `max_soft_tokens=280`, and `max_patches=2520`.

### Cause of the original Tensor-path error

The original M0 candidate was a float Tensor containing pixel values in `[0,255]`.
The PIL route stays `uint8` during interpolation and is converted to float only
after resize. The float Tensor route is interpolated as float, without the same
uint8 rounding step. The official processor source and feature comparison support
this explanation:

| Input passed to the default processor | Pixel feature max abs error vs. PIL path | Embedding max abs error | Cosine | Threshold result |
|---|---:|---:|---:|---|
| Raw `uint8` CHW `[0,255]`, RGB | `0` | `2.78e-7` | `1.000000` | Pass |
| Raw float CHW `[0,255]`, M0 route | `0.0040472` | `0.00079144` | `0.9999833` | **Fail** max-error threshold |
| Raw float HWC `[0,255]` | `0.0040472` | `0.00079144` | `0.9999833` | **Fail** max-error threshold |
| Raw float CHW `[0,1]` with default rescaling enabled | `0.960934` | `0.150071` | `0.737283` | **Fail** |

The M0 raw float `[0,255]` path resized and rescaled once; it was not double-scaled.
However, passing an already preprocessed float `[0,1]` Tensor back through the
default processor applies the `1/255` rescale a second time. That accidental
double-rescale produced pixel feature differences of about `0.96` and embedding
max-error differences from `0.1071` to `0.1501` in the tested images. The exact
Tensor path must therefore disable both `do_resize` and `do_rescale` after
preprocessing.

Raw one-channel grayscale Tensors also failed in the processor/model path: the
processor's RGB conversion applies to the PIL input route, but does not expand this
Tensor input to three channels. The PIL-based prototype below converts grayscale
through the official processor `process_image` method and returns RGB Tensor data.

### Exact Tensor preprocessing prototype and rebuilt features

The prototype calls the official `process_image`, `aspect_ratio_preserving_resize`,
and `rescale_and_normalize` methods, returning a float32 CHW Tensor in `[0,1]`.
Passing that Tensor back through the official `AutoProcessor` with
`do_resize=False, do_rescale=False` rebuilt every compared model input exactly:
`pixel_values`, `image_position_ids`, `input_ids`, and `attention_mask` were all
element-wise equal to the PIL-generated mapping for all four test images.

The resulting normalized embeddings matched the Sentence Transformers reference:

| Input image | Prototype Tensor shape | Max absolute error | Cosine similarity | Result |
|---|---:|---:|---:|---|
| M0 RGB, `256x192` | `[3,672,912]` | `2.7753e-7` | `1.000000` | **Pass** |
| Grayscale, `256x192` | `[3,672,912]` | `2.0862e-7` | `1.000000` | **Pass** |
| Wide RGB, `384x128` | `[3,432,1344]` | `3.2224e-7` | `1.000000` | **Pass** |
| Tall RGB, `128x384` | `[3,1344,432]` | `3.0920e-7` | `1.000000` | **Pass** |

The values meet both unchanged limits: max absolute error `<=1e-4` and cosine
similarity `>=0.9999`.

At `256x192`, the resized image is `[3,672,912]`: it contains 2394 real 16x16
patches, 126 padded patch entries, and 266 image tokens. Wide and tall examples
resize to `[3,432,1344]` and `[3,1344,432]` respectively; each has 2268 real and
252 padded patch entries, with 252 image tokens. In all four cases, reconstructed
image positions and token IDs exactly matched the PIL route.

### Tensor-only information and batching

For each tested single image, the returned preprocessed Tensor plus its spatial
shape and the loaded official processor configuration was sufficient to reconstruct
all processor/model features. No additional per-image image-position or token-count
metadata had to be carried separately: the official processor recomputed those
from the Tensor shape and values.

Batch results:

- A real `[2,3,672,912]` Tensor batch (M0 RGB and same-size grayscale) was split into
  per-sample images and passed to the official processor. Its feature mapping was
  exactly equal to the PIL reference. The resulting `[2,768]` embeddings had max
  absolute error `2.7381e-7` and cosine similarity `[1.0,1.0]`; both thresholds pass.
- A list of the four preprocessed CHW Tensors, with differing aspect-ratio shapes,
  was accepted by the official processor when images were grouped per sample.
  Features were element-wise equal to the PIL batch, and embeddings had shape
  `[4,768]`, max absolute error `3.0641e-7`, and cosine similarity `[1.0,1.0,1.0,1.0]`.
- The four variable-sized preprocessed Tensors cannot be stacked into the ordinary
  OpenCLIP-style `[B,C,H,W]` Tensor. `torch.stack` fails because the wide and tall
  output spatial dimensions differ from the M0 RGB dimensions. Thus, exact
  per-image Tensor processing is proven, but unmodified dense Tensor batching for
  arbitrary aspect ratios is **not**.

This is the remaining API limitation for option A/C: either limit dense Tensor
batches to a common preprocessed shape, introduce padding/cropping that changes
official image inputs, or use a per-sample list/mapping path for variable shapes.
The last option no longer satisfies the strict dense OpenCLIP Tensor contract.

### Reapplying the processor and measured cost

Sending the already preprocessed Tensor back through the processor defaults does
not preserve the reference: for these outputs, the processor's resize calculation
leaves the spatial dimensions unchanged, but its rescale step is applied again and
changes the pixels. This fails both numerical criteria (embedding max errors between
`0.1071` and `0.1501` for these examples). The prototype correctly disables both
operations when turning the prepared Tensor into patches.

One five-repeat CPU median sample on the M0 RGB input measured:

| Operation | Median |
|---|---:|
| Official PIL processor to full feature mapping | `5.231 ms` |
| Tensor preprocessing resize/rescale stage | `4.252 ms` |
| Tensor-to-feature mapping stage (patchify/tokenize) | `2.913 ms` |
| Two-stage Tensor preprocessing plus mapping | `7.427 ms` |

These are noisy CPU wall-clock observations, not a performance benchmark; they
exclude model forward. Mapping equality means the model forward itself is the same
for a given device/dtype. For the M0 RGB image, the preprocessed float Tensor occupies
`7,354,368` bytes; the official mapping occupies `7,786,080` Tensor bytes. The
lower-bound live storage when the Tensor is held while constructing the mapping is
`15,140,448` bytes, before model activations and allocator overhead. Variable-ratio
batches can increase aggregate Tensor retention.

### Comparison of implementation options

| Option | API compatibility | Numerical accuracy | Inference speed | Memory use | Maintainability |
|---|---|---|---|---|---|
| **A. OpenCLIP Tensor only** | Highest for `preprocess(...).unsqueeze(0)` and `[B,C,H,W]` at a common spatial size. Exact preprocessing returns aspect-ratio-dependent dimensions, so arbitrary aspect-ratio batch stacking is not compatible. | Proven for a single Tensor and same-shaped Tensor batches with the exact uint8-resize-then-rescale prototype and disabled second processing. | Two-stage preprocess + mapping was `7.427 ms` in the sample, versus `5.231 ms` for full official processing; noisy, but has extra Tensor-to-mapping work. Same model forward afterward. | The example Tensor is `7.35 MB`; Tensor and final mapping coexist during adaptation (`15.14 MB` lower bound). | Simple public surface but requires strict tensor value/range/dtype/spatial-shape rules and custom processor bridging. Aspect-ratio batching remains a significant constraint. |
| **B. Processor feature mapping** | Lower: callers pass the processor result rather than the conventional image Tensor. | Best-established route. Direct PIL mapping and rebuilt Tensor mapping were exact at the feature level and within embedding thresholds. Variable-size image batches work. | One complete processor call was `5.231 ms` in this sample; no second mapping call. Same model forward cost. | Mapping was `7.79 MB` for one example, with output patches padded to the model patch budget. | Closest to the official reference and handles per-sample dimensions, but exposes Transformers-specific structures. |
| **C. Tensor plus EmbeddingGemma input** | Best potential coverage: retain the Tensor path for common-size callers and offer a mapping path for variable-size inputs. | Both routes can be accurate if the Tensor path follows the verified prototype and the mapping path passes through unchanged. | Mapping path is direct. Tensor route has the extra preprocessing/mapping cost; overload dispatch adds small overhead. | Depends on the chosen input; returning/retaining both representations wastes memory. Convert lazily and release intermediate tensors to control peak. | More API branches, input validation, documentation, and tests; avoids forcing one incompatible contract across use cases. |

**Recommendation:** C is the most complete future API only if it is clearly separated
(for example, keep the OpenCLIP-compatible Tensor input for common-shape batches and
provide an explicit EmbeddingGemma processor-feature entry point for variable-shape
batches). Do not return both representations by default. If simplicity wins, B is
the most robust numerical contract. A alone is viable only with a documented
same-spatial-shape batch restriction or an accepted deviation from the official
aspect-ratio-preserving preprocessing.

M0.1 establishes that the original numeric mismatch was not an inherent limitation
of the Tensor contract: matching the official uint8 resize and scaling order makes
single and common-shape batched Tensor input pass the fixed thresholds. It does not
resolve arbitrary aspect-ratio stacking; do not start M1 until the public batch
contract is chosen.

## Specification updates to consider

- Record Python 3.12 as the initial target, `pretrained=None` as an explicit error,
  FP32 as the baseline, BF16 as separately validated, and `forward()` /
  `get_logits()` as out of current scope.
- Resolve the image processor mapping versus `[B, 3, H, W]` Tensor contract before
  implementing the factory or inference wrapper.
- Preserve the observed tokenizer PAD ID/right-padding and exact sampled mask
  recovery, but avoid claiming it for arbitrary external token tensors.
- Keep the task prompt unresolved pending a representative cross-modal benchmark.
- Keep CUDA support unverified until a CUDA-capable run is available.
- M0.1 confirms that a float32 CHW Tensor produced by official PIL conversion,
  uint8 resize, then rescale can reproduce all tested image features when the
  processor does not resize/rescale it again.
- The official aspect-ratio-preserving preprocessing yields different H/W for
  different aspect ratios; a dense `[B,C,H,W]` batch cannot represent those outputs
  without a changed image transform or added per-sample handling.

## Official references

- [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2)
- [Google Sentence Transformers inference guide](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers)
- [Google multimodal inference guide](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
- [Transformers v5.20.1 Gemma4 image processor source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/gemma4/image_processing_gemma4.py)
- [Transformers v5.20.1 EmbeddingGemma2 processor source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/embedding_gemma2/processing_embedding_gemma2.py)
