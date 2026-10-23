# GitHub Copilot Instructions — OpenVLCLIP

## Repository identity and source of truth

This project is **openvlclip** (Python import: `openvlclip`), not the older `openclip-gemma` project. Preserve the OpenCLIP-inspired multi-backend API and existing released behavior. Before any v0.8.0 work, read **`docs/v0.8.0-copilot-handoff.md`** and **`docs/v0.8.0-scope.md`**. Use the current source code as the authority on what is already implemented; do not reimplement merged work. Consult `docs/openclip-compatibility-v0.3.0.md` and `docs/compatibility-spec-v0.1.0.md` for baseline historical behavior, resolving any discrepancies in favor of currently documented released contracts unless explicitly directed otherwise.

## v0.8.0 guardrails

- In scope: validate three already-registered SigLIP 2 FixRes checkpoints, finish optional PE-Core T/S integration, validate measured NumPy exact top-k search improvement, prepare documentation and package release.
- OUT of scope: DINOv3 and DINOv3+dino.txt, FAISS/ANN, new model families, training, fine-tuning, new CLI/UI, unrelated refactors.
- Existing model/image/video APIs, scoring/ranking/tie ordering, checkpoint identity, EmbeddingSpace validation, and index on-disk formats must remain compatible.
- Preserve pinned Hugging Face commit revisions and load weights only on explicit request; never redistribute model weights or licenses as if they were part of the package.
- Keep PE-Core's `open-clip-torch` optional; importing `openvlclip` or listing models must not require it.
- Validate the **actual** preprocessing pipeline before fixing PE-Core preprocessing identity. Do not claim real-weight numerical parity from mocked tests. Clearly distinguish PE-Core's OpenCLIP-converted checkpoint from the independently packaged Meta originals.
- Python support: respect `pyproject.toml`; FP32 numerical reference, BF16 only when measured.

## Fast development workflow

1. Inspect code and git state, then list only truly unfinished requirements from the handoff.
2. Implement minimal changes with focused commits; avoid unrelated formatting, generated files and new abstractions.
3. Run narrow, deterministic unit/regression checks for changed modules, plus only essential lint/build/smoke checks. Do **not** run full matrices, heavyweight downloads or all-model integration in normal CI.
4. Real-checkpoint comparisons are manual/opt-in release gates; if hardware or network is unavailable, write the exact command and an honest BLOCKED status. Never fabricate parity or performance numbers and never declare the release ready until required checks have actually passed.
5. Benchmark search against previous implementation using representative sizes, tie cases and end-to-end and top-k-only latency; report hardware/software conditions and whether any improvement is real.
6. Report progress and results in Japanese, with changed files, tests executed, pending blockers, and evidence.
7. Do not publish PyPI, create a release or version tag without explicit user authorization. Normal feature commits/PRs and merges are permitted only when explicitly requested for that task.

See `docs/v0.8.0-copilot-handoff.md` for prioritized actions and acceptance criteria.
