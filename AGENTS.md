# FastPLMs

## Purpose and Sources

FastPLMs provides Hugging Face-compatible protein language and structure models under `fastplms/`.

- `docs/architecture.md` and `docs/models.md`: package and model-family contracts
- `docs/embedding_api.md`: shared embedding interface
- `docs/attention_backends.md`: backend-specific behavior
- `docs/testing.md`: per-family images, markers, and parity workflow

## Architectural Invariants

- Model and config classes remain compatible with Transformers auto classes and `trust_remote_code=True`.
- `fastplms/embedding_mixin.py` is the shared sequence embedding API.
- Tokenizer-mode families accept token IDs and attention masks. E1 sequence mode has no tokenizer and must retain its native raw-sequence preparation path.
- `testing/conftest.py` is the authoritative model registry, and `testing/test_parity.py` is the strict parity suite.
- Use per-family Docker images for native dependency parity. Respect the `gpu`, `slow`, `large`, and `structure` markers before running expensive suites.

## Canonical Commands

```bash
git submodule update --init --recursive
./build_images.sh esm2
docker run --rm --gpus all --ipc=host -v ${PWD}:/workspace fastplms-esm2 \
  python -m pytest /workspace/testing/test_parity.py -k esm2 -v
```

Always pass `--ipc=host` to Dockerized PyTorch runs.
