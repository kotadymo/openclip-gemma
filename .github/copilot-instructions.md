# GitHub Copilot Instructions

## Project Overview

This repository develops the `openclip-gemma` Python package (import name: `openclip_gemma`), an independent library providing an OpenCLIP-compatible inference API for Google's EmbeddingGemma 2.

The primary objective is to enable existing OpenCLIP inference code to run with minimal modifications while preserving the correctness of EmbeddingGemma 2 embeddings.

## Source of Truth

Read `docs/compatibility-spec-v0.1.0.md` before planning or implementing features.

The compatibility specification is the project's primary requirements document.

Verify uncertain behaviors against the official OpenCLIP source code and the official EmbeddingGemma 2 documentation.

If the specification conflicts with the upstream implementation, report the conflict before changing the intended behavior.

Do not silently modify the specification.

## Scope of v0.1.0

- Initial Python support target: Python 3.12.
- FP32 is the numerical baseline. BF16 requires separate validation before it is claimed as supported.
- `pretrained=None` must raise an explicit error; do not silently load official weights or create random weights.
- `forward()` and `get_logits()` are not in the current implementation scope.

Implement:

- `create_model()`
- `create_model_and_transforms()`
- `get_tokenizer()`
- `model.encode_image()`
- `model.encode_text()`
- `model.eval()` and `model.to()`
- Image and text batch inference
- CPU and CUDA inference

Do not implement training, fine-tuning, audio, video, or arbitrary OpenCLIP architectures.

Optional APIs must not delay the required inference functionality.

## Architecture

- Use a standard Python `src/` package layout.
- Separate model loading, image preprocessing, text tokenization, and inference.
- Prefer official Hugging Face Transformers and Sentence Transformers implementations.
- Avoid reimplementing the EmbeddingGemma 2 neural network.
- Preserve PyTorch Tensor input/output contracts wherever OpenCLIP compatibility is claimed.
- Do not decode and re-encode token IDs or reconstruct PIL images from normalized tensors unless the approach has been explicitly validated.
- Do not introduce unnecessary dependencies or abstractions.

## Numerical Correctness

- Treat official EmbeddingGemma 2 outputs as the reference.
- Use FP32 as the baseline; validate BF16 separately before claiming support.
- Do not assume that FP16 is safe.
- Do not invent a learned CLIP `logit_scale`.
- Distinguish API compatibility from numerical equivalence with OpenCLIP.
- Validate preprocessing, prompting, pooling, normalization, and output shapes.
- Preserve batch semantics and input ordering.

## Development Workflow

1. Read the relevant specification and existing code.
2. Explain the implementation plan and affected files.
3. Make the smallest change necessary for the requested feature.
4. Add unit tests and relevant integration tests.
5. Run applicable tests and lint checks.
6. Report what passed, failed, or could not be executed.
7. Summarize changes and known limitations.

Do not proceed to unrelated features without an explicit request.

For the M0 validation phase, only create reference experiments under `experiments/m0/` and record observed results in `docs/m0-validation.md`. Do not implement the OpenCLIP-compatible API until M0 is reviewed.

Do not automatically create commits, push changes, publish packages, or modify repository settings.

## Testing

- Use pytest.
- Mock heavyweight components in unit tests.
- Separate tests requiring actual model weights from lightweight tests.
- Check output shapes, dtypes, devices, finiteness, and normalization.
- Compare actual embeddings against the official implementation where applicable.
- Do not claim numerical correctness from shape-only tests.
- Mark GPU-dependent tests and skip them when CUDA is unavailable.

## Code Quality

- Use Python type hints.
- Write concise docstrings in English.
- Follow Ruff formatting and linting conventions.
- Use clear error messages for unsupported functionality.
- Keep public APIs documented.
- Do not claim compatibility for untested behaviors.

## Communication

Explain implementation decisions and testing results in Japanese.

Keep code identifiers, docstrings, and public documentation in English.

When a design decision is uncertain, state the alternatives and recommend one before proceeding.
