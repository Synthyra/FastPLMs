# FastPLMs

## Purpose

FastPLMs provides Hugging Face-compatible protein language and structure models
under `src/fastplms/`. Treat the repository as release software for scientific
models. Changes must preserve biological conventions, Transformers behavior,
reproducibility, legal provenance, and the evidence boundary of each model
family.

Start with [README.md](README.md) for user-facing behavior and
[docs/README.md](docs/README.md) for documentation routing.

## Sources of truth

- `src/fastplms/models.toml`: model IDs, revisions, files, AutoClasses,
  tokenizer modes, state transformations, backends, precision, licenses, and
  release tiers.
- `src/fastplms/registry.py`: typed parsing and validation of the manifest.
- `docs/architecture.md` and `docs/models.md`: package and model-family
  contracts.
- `docs/embedding_api.md`: shared ordered embedding interface and persistence.
- `docs/attention_backends.md`: backend names, dtype constraints, masks, and
  parity boundaries.
- `docs/testing.md`: candidate/reference stages, markers, and parity workflow.
- `tests/parity/`: strict model and tokenizer comparisons.

Do not infer a model contract from an older README, an unpinned Hub card, or an
unused code path when the manifest or current tests say otherwise.

## Repository boundaries

- `src/fastplms/` contains installable runtime code.
- `vendor/upstream/` contains pinned official repositories used as parity
  oracles. Runtime code must not import from this directory.
- `tests/` contains unit, integration, parity, structure, and release checks.
- `tools/` contains artifact, conversion, remote, and maintenance workflows.
- `examples/` contains runnable research and training examples. Keep examples
  directly in this directory rather than creating a tutorial subtree.
- `model_cards/` contains generated checkpoint cards.
- `LICENSES/` contains distributable third-party legal texts and provenance.

Do not place license files, model cards, or READMEs beside runtime model
modules. Do not hand-edit generated model cards or
`docs/generated/support.md`; update the manifest or renderer and regenerate.

## Architectural invariants

- Model and configuration classes remain compatible with Transformers auto
  classes and `trust_remote_code=True`.
- `src/fastplms/embeddings/` is the shared sequence embedding API.
- Tokenizer-mode families accept token IDs and attention masks. E1 has no
  tokenizer and must retain its native raw-sequence preparation path.
- Structure families retain native chain, residue, atom, ligand, nucleic-acid,
  and MSA semantics where applicable.
- A requested attention backend either executes the named implementation or
  raises. Never add a silent fallback.
- Official repositories are isolated references, not build inputs for runtime
  source. Production imports must not change `sys.path`, download code, compile
  a kernel, initialize a model, or mutate global Torch state.
- State transformations are named, deterministic, and covered by exact tests.
- Boltz2 remains provisional until its declared native end-to-end equivalence
  limits pass. Do not broaden its claims from partial contracts.

## Common workflows

Initialize official sources:

```bash
git submodule update --init --recursive
```

Run portable release checks on the declared remote environment:

```bash
python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/key \
  --suite check

python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/key \
  --suite compliance
```

Build the candidate and one isolated reference image:

```bash
sudo docker buildx bake \
  -f docker/docker-bake.hcl \
  candidate reference-esm2 \
  --load
```

Run focused candidate tests only when their required reference results already
exist:

```bash
sudo docker compose -f docker/compose.yaml run --rm candidate \
  python -m pytest tests/parity -k esm2 -v
```

Always pass `--ipc=host` to Dockerized PyTorch runs. Respect the `gpu`, `slow`,
`large`, and `structure` markers before running expensive suites.

Regenerate and check documentation:

```bash
PYTHONPATH=src python -m tools.artifacts.generate_docs
PYTHONPATH=src python -m tools.artifacts.generate_docs --check
python -m pytest tests/release/test_documentation.py \
  tests/release/test_model_card_licenses.py -v
```

Build a local Hub artifact:

```bash
PYTHONPATH=src python -m tools.artifacts.build \
  esm2_150m \
  /cache/fast-snapshot \
  --tokenizer-dir /cache/official-tokenizer-snapshot \
  --output-root dist/hub
```

Preview an add-only Hub update that excludes checkpoint weights:

```bash
PYTHONPATH=src python -m tools.artifacts.publish \
  --files-only esm2_150m \
  --artifact-root dist/hub \
  --dry-run
```

Remove `--dry-run` only after reviewing every planned path. The publisher must
remain manifest-scoped, add-only, protected by the remote parent commit, and
free of token command-line arguments. It must never upload weight-shaped paths,
create repositories, delete remote files, or publish complete-artifact
attestations during a files-only update.

## Change policy

- Inspect the manifest, family code, tests, and current docs before changing a
  biological or model-facing contract.
- Keep changes focused and preserve unrelated work in a dirty tree.
- Add or update tests for public behavior, conversion rules, file identities,
  generated output, and fail-closed paths.
- Use repository-native verification proportional to risk. Do not run large
  GPU or structure suites without the requested environment and authority.
- Record measured results with the environment, dtype, backend, sequence panel,
  and threshold. Do not turn an implementation detail into a parity, speed, or
  biological claim.
- Never inspect or print credential files. Pass credential paths opaquely to
  the trusted command that needs them.
