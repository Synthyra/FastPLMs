# FastPLMs

## Purpose and Sources

FastPLMs provides Hugging Face-compatible protein language and structure models under `src/fastplms/`.

- `docs/architecture.md` and `docs/models.md`: package and model-family contracts
- `docs/embedding_api.md`: shared embedding interface
- `docs/attention_backends.md`: backend-specific behavior
- `docs/testing.md`: per-family images, markers, and parity workflow

## Architectural Invariants

- Model and config classes remain compatible with Transformers auto classes and `trust_remote_code=True`.
- `src/fastplms/embeddings/` is the shared sequence embedding API.
- Tokenizer-mode families accept token IDs and attention masks. E1 sequence mode has no tokenizer and must retain its native raw-sequence preparation path.
- `src/fastplms/models.toml` is the authoritative model registry, and `tests/parity/` is the strict parity suite.
- Use the consolidated candidate image and isolated native reference stages. Respect the `gpu`, `slow`, `large`, and `structure` markers before running expensive suites.

## Canonical Commands

```bash
git submodule update --init --recursive
sudo docker buildx bake -f docker/docker-bake.hcl candidate reference-esm2 --load
sudo docker compose -f docker/compose.yaml run --rm candidate \
  python -m pytest tests/parity -k esm2 -v
```

Always pass `--ipc=host` to Dockerized PyTorch runs.
