# FastPLMs documentation

This directory explains the public API, model-family contracts, release
evidence, and contributor workflows for FastPLMs 1.0. The model manifest at
`src/fastplms/models.toml` remains authoritative when prose and generated data
disagree.

## Start here

| Goal | Read |
| --- | --- |
| Understand repository boundaries and loading flow | [Architecture](architecture.md) |
| Find a supported checkpoint or AutoClass | [Models](models.md) and the [generated support matrix](generated/support.md) |
| Embed sequences or FASTA datasets | [Embedding API](embedding_api.md) |
| Select SDPA, Flex Attention, or a pinned FlashAttention kernel | [Attention backends](attention_backends.md) |
| Build and validate an offline Hub artifact | [Artifacts](artifacts.md) |
| Run parity, structure, or release tests | [Testing](testing.md) |
| Measure throughput or memory | [Benchmarking](benchmarking.md) |

## Model and research workflows

- [ESMFold2](esmfold2.md): folding, learned representations, BF16, and
  experimental FP8.
- [Test-time training](ttt.md): opt-in low-rank adaptation and its evidence
  boundary.
- [Binder design](binder_design.md): differentiable ESMFold2 and ESM++ research
  example.
- [Fine-tuning](finetuning.md): Trainer, PEFT, data splits, and reproducibility.
- [Vector benchmark embeddings](vector_embeddings/README.md): reusable
  embedding artifacts for Protify evaluation.

## Maintenance

- [Contributing](contributing.md): adding models, tests, docs, and examples.
- [Licensing](licensing.md): project, source, and checkpoint terms.
- [Migration to 1.0](migration.md): intentional API and repository-layout
  changes.

Generated files carry a marker stating that they come from
`src/fastplms/models.toml`. Edit the typed manifest or renderer, then run:

```bash
PYTHONPATH=src python -m tools.artifacts.generate_docs
PYTHONPATH=src python -m tools.artifacts.generate_docs --check
```

Documentation examples distinguish model output from experimental evidence.
Structure confidence, language-model likelihood, and generated sequences are
prioritization signals unless an independent experiment establishes the
corresponding biological claim.
