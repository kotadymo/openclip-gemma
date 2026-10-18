# GitHub Copilot Instructions

## Project Overview

This repository develops the `openclip-gemma` Python package (import name: `openclip_gemma`), an independent library providing an OpenCLIP-inspired inference API for Google's EmbeddingGemma 2.

The primary objective is to enable existing OpenCLIP inference code to run with minimal modifications while preserving the correctness of EmbeddingGemma 2 embeddings.

## Source of Truth

Read `docs/openclip-compatibility-v0.3.0.md` before changing the Factory or Model compatibility APIs. Read `docs/compatibility-spec-v0.1.0.md` for the historical core inference contract and its confirmed numerical behavior.

The compatibility specification is the project's primary requirements document.

Verify uncertain behaviors against the official OpenCLIP source code and the official EmbeddingGemma 2 documentation.

If the specification conflicts with the upstream implementation, report the conflict before changing the intended behavior.

Do not silently modify the specification.

## Supported inference API

- Python 3.12-3.14 are supported, excluding Python 3.14.1 as declared in `pyproject.toml`.
- FP32 is the numerical baseline; BF16 has separate CPU and CUDA validation.
- `pretrained=None` must raise an explicit error; do not silently load official weights or create random weights.

Implement:

- `create_model()`
- `create_model_and_transforms()`
- `create_model_from_pretrained()`
- `list_models()` and `list_pretrained()`
- `get_tokenizer()`
- `model.encode_image()`
- `model.encode_text()`
- `model.forward()` and `model.get_logits()` only with explicit `logit_scale_mode="cosine"`; strict mode is the default because the model has no trained OpenCLIP logit scale.
- `model.eval()` and `model.to()`
- Image and text batch inference
- CPU and CUDA inference

Do not implement training, fine-tuning, audio, arbitrary OpenCLIP architectures, or numerical emulation of a trained CLIP `logit_scale`.

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
