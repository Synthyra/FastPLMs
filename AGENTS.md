# FastPLMs

## Purpose

FastPLMs maintains the runtime source uploaded to Hugging Face protein language
and structure model repositories. This repository is a source, test, artifact,
and dependency workspace, not an installable Python distribution. Changes must
preserve biological conventions, Transformers behavior, reproducibility, legal
source records, and the evidence boundary of each model family.

Start with [README.md](README.md) for user-facing behavior and
[docs/README.md](docs/README.md) for documentation routing.

## Sources of truth

- `src/fastplms/models.toml`: model IDs, revisions, files, AutoClasses,
  tokenizer modes, state transformations, backends, precision, licenses, and
  release tiers.
- `src/fastplms/registry.py`: typed parsing and validation of the manifest.
- `docs/architecture.md` and `docs/models.md`: runtime-source and model-family
  contracts.
- `docs/embedding_api.md`: shared ordered embedding interface and persistence.
- `docs/attention_backends.md`: backend names, dtype constraints, masks, and
  parity boundaries.
- `docs/testing.md`: candidate/reference stages, markers, and parity workflow.
- `tests/parity/`: strict model and tokenizer comparisons.

Do not infer a model contract from an older README, an unpinned Hub card, or an
unused code path when the manifest or current tests say otherwise.

## Repository boundaries

- `src/fastplms/` contains runtime source copied into Hugging Face artifacts.
- `vendor/upstream/` contains pinned official repositories used as parity
  oracles. Runtime code must not import from this directory.
- `tests/` contains unit, integration, parity, structure, and release checks.
- `tools/` contains artifact, conversion, remote, and maintenance workflows.
- `tools/confidence/` contains the resumable Modal confidence-head pilot. Keep
  its tests and compute remote; do not run intensive confidence work locally.
  The approved run uses two bounded H100 workers in parallel for about ten
  hours each. Cost and duration are tracked, with at most two GPU workers.
  The supplied `rcsb_multimer` copy has downloaded and extracted successfully.
  Split construction passed. Both campaigns were restarted after fixing a
  device mismatch in cache generation, which now reports progress to W&B.
  Both completed caching and passed the overfit checks; main-training optimizer
  updates are verified in W&B. The 300M training stage stopped early under its
  validation rule and passed held-out quality checks. The 600M training stage
  also stopped early and passed its final evaluation. Artifact release
  checks and publication remain pending. Avoid duplicate evaluation calls.
  Quality checks still govern publication. See docs/confidence_training.md for
  current run links. Preserve active runs' recorded training-code hashes.
  The v2 campaign (`ssh.py`, `host.py`, `online_*.py`, `rollouts.py`,
  `target_*.py`, `test_evaluation.py`) trains on a GH200 workstation from
  AtlasFold-Data with online rollouts. Its GPU-hour ledger allows 2 hours of
  smoke checks, 24 hours per model, and 5 hours for the production reference.
  Both v2 heads finished training, the test evaluation and the production
  reference ran once, and the acceptance gates were applied: the 300M head
  passes the pilot and sample-selection gates, the 600M head passes neither,
  and both miss production parity on within-target selection accuracy. Results
  are recorded in `docs/evidence/confidence/esmfold2_{300,600}-v2.json` and
  summarized in the model cards. The test split is now spent, so a further
  evaluation on it is no longer held out. Keep the pilot heads and reports
  unchanged as baselines, and do not publish v2 weights without separate
  approval.
- `examples/` contains runnable research and training examples. Keep examples
  directly in this directory rather than creating a tutorial subtree.
- `model_cards/` contains generated checkpoint cards.
- `LICENSES/` contains distributable third-party legal texts and source records.

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
  raises. Never add a silent fallback. The opt-in `attn_implementation="auto"`
  request follows the family's `attention_auto_order`, ends in a named
  implementation, and records why each skipped candidate was unusable. An order
  that prefers FlashAttention must cite measured evidence.
- Folding progress is disabled by default (`verbose=False`). Progress display must
  preserve outputs and random-number state; confidence-disabled ESMFold2
  exports must retain unknown confidence values.
- Official repositories are isolated references, not build inputs for runtime
  source. Production imports must not change `sys.path`, download code, compile
  a kernel, initialize a model, or mutate global Torch state.
- State transformations are named, deterministic, and covered by exact tests.
- ESMFold2 experimental Fast base300M and base600M variants retain their pinned
  24-block, no-MSA contracts and use backbone dimensions `960 x 30` and
  `1152 x 36`; their disabled confidence heads do not produce pLDDT, pTM, iPTM,
  or PAE. The 300M single-protein comparison passed, but this is not the full
  structure benchmark, which remains pending. The published mirrors are pinned to
  revisions `a38a62ae930d157484b331c2bf4241684573adba` (300M) and
  `71c67d0b2b73dc245ea7c3cc0d0476439a882d08` (600M). The 600M variant has no
  inference validation result.
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

Run confidence-pilot tests and compute through Modal with the launcher described
in [confidence-head training](docs/confidence_training.md). Do not inspect or
print `.secrets.env`; the launcher loads it through the trusted environment
loader and requires W&B initialization for every training run.

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

Compose supplies `ipc: host` for its services. Pass `--ipc=host` to raw
Dockerized PyTorch runs. Respect the `gpu`, `slow`, `large`, and `structure`
markers before running expensive suites.

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
  --files-only \
  --artifact-root dist/hub \
  --dry-run
```

Remove `--dry-run` only after reviewing every planned path. The publisher must
select every manifest model when no positional model IDs are provided; model
IDs restrict the operation to that explicit subset. It must remain
manifest-scoped, add-only, protected by the remote parent commit, and free of
token command-line arguments. It must never upload weight-shaped paths, create
repositories, delete remote files, or publish complete-artifact attestations
during a files-only update.

## Python coding standards

Write direct, readable Python whose data flow and domain intent are visible to
the next maintainer. Preserve correctness, security, public interfaces, data
schemas, tested behavior, and repository tooling before applying these style
preferences. Treat cleanup as behavior-preserving unless the task explicitly
changes behavior. Observable behavior includes exceptions, import side effects,
CLI output, serialization, random-number use, dtype, device, tensor shape, and
documented performance guarantees.

- Prefer narrow, typed functions with domain-specific names and concrete,
  parameterized types. Use `Any` only at a genuinely dynamic boundary.
- Prefer straightforward control flow over cleverness, speculative generality,
  compatibility shims without supported callers, silent fallback cascades, or
  wrappers that merely rename a call.
- Extract a helper only when it names a real concept, clarifies the caller, or
  isolates a separately testable phase. Keep reusable code near the feature
  that owns it; do not create catch-all utility modules.
- Use classes when state and behavior form a meaningful domain object or
  workflow. Prefer functions, typed values, or dataclasses for simple
  transformations and records.
- Keep entry points thin: parse arguments, construct configuration and
  collaborators, then run a short sequence whose names expose the workflow.
- Raise ordinary exceptions for invalid external input. Reserve assertions for
  internal invariants and programmer assumptions. Avoid broad
  `except Exception` handlers unless the boundary genuinely requires one.
- Comment intent, assumptions, units, biological conventions, non-obvious
  mechanics, and design rationale. Remove comments and docstrings that merely
  narrate syntax or repeat precise names and types.
- For NumPy, PyTorch, JAX, and similar numerical code, make tensor or array
  shape transformations traceable at each non-obvious step. State semantic
  dimensions, such as batch, residue, token, atom, or channel, and never invent
  a shape not guaranteed by the contract.
- Give each module one primary responsibility. Split modules by ownership and
  dependency direction, not line count, and keep public surfaces intentional.
- Keep a module docstring first and `from __future__` imports immediately after
  it. Group all direct `import` statements before ordinary `from` imports;
  within each group place standard-library before third-party imports, followed
  by a distinct repository-local section. Preserve guarded, optional,
  registration-sensitive, and initialization-sensitive ordering barriers, and
  defer to enforced formatter output.
- Leave two blank lines after the complete import block. Keep descriptive
  all-caps configuration near the top of runnable modules, and keep parser
  declarations on one line when comfortably readable.

Before structural refactoring, identify and run the smallest existing test set
that characterizes the target. Run the same tests afterward and add focused,
CPU-only characterization coverage when appropriate. For mechanical changes,
use proportional formatter, linter, type-checker, compile, or focused-test
verification. Do not weaken tests, silently fix unrelated bugs, or trigger
downloads, CUDA, large datasets, remote machines, or slow integration suites
without authorization. Record pre-existing failures and review the final diff
for behavior changes, unnecessary movement, stale imports, and unrelated churn.

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
