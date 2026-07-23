# Contributing

Changes should preserve scientific behavior, repository boundaries, and a clear
audit trail. Do not commit or upload model weights as part of a source change.

## Setup

```bash
uv lock --check
uv sync --frozen --no-default-groups \
  --group validation --extra cpu --extra dev --extra structure --extra train
python -m pytest tests/cpu -m cpu_contract -n auto --dist=loadscope
```

The explicit `cpu` extra selects the locked PyTorch CPU index for this uv
development environment. Do not use `--all-extras`: CUDA-only extras are
intentionally incompatible with the CPU environment. Select only the feature
extras needed by the validation lane.

Official submodules are not required for routine CPU work. Initialize them only
for a live compliance run with `git submodule update --init --recursive`. Run
release GPU verification on the configured Linux aarch64 GH200 host through
`tools/remote/run.py`. H100 and H200 are supported Hopper-class devices, but do
not substitute for the current exact-device release evidence. The runner binds
Bake to native `linux/arm64` and records the GPU UUID; never use emulated images
for CUDA evidence.
Candidate PyTorch containers must use `ipc: host`.

## Code rules

- Keep production code under `src/fastplms` and examples under `examples`.
- Do not import `vendor/upstream` from production code.
- Do not download, compile, construct tokenizers, log, or mutate global Torch
  settings at import time.
- Use optional imports only inside the feature that requires them.
- Prefer the shortest clear implementation. Retain complexity only with a
  measured benefit and strict parity coverage.
- Preserve checkpoint keys and aliases. If that is impossible, add a named
  deterministic transform and an exact conversion test.
- Use type annotations for public interfaces and explain non-obvious numerical
  choices near the implementation.

Python identifiers use PEP 8 snake case. In prose and mathematical comments,
scalar quantities and dimensions are lowercase, tensors and matrices use an
uppercase alias, and shapes use parentheses:

```python
# H is the hidden-state tensor with shape (b, l, n, d).
hidden_states = model_output.hidden_states
```

Do not write square-bracket shape signatures or uppercase dimension symbols in
shapes. Run the notation checker before review.

## Adding or changing a model

First freeze the official configuration, tokenizer assets and behavior, state
schema and aliases, representative outputs, source revision, environment, and
licenses. Then:

1. update `src/fastplms/models.toml` with immutable identities and a complete
   conversion record;
2. implement or change package code without importing the official checkout;
3. update a public-API reference adapter in
   `tests/parity/support/reference_adapters`;
4. add exact configuration, tokenizer, state, alias, FP32, BF16, feature, and
   backend cases;
5. build and validate its offline local artifact;
6. regenerate support data and model cards;
7. verify the generated capability-to-evidence row points to a guide, runnable
   offline/local example, and every required test tier;
8. run the required remote tiers.

Never create a family-specific tolerance to make a failing comparison pass. Fix
the implementation or remove the unsupported capability from the manifest and
documentation.

## Documentation

State the input, transformation, output, validation evidence, and limitation.
Avoid unsupported equivalence, performance, or biological claims. Keep
first-party model cards under `model_cards/`, legal texts under `LICENSES/`,
and runnable scripts directly under `examples/`. Generated cards and support
tables must be changed through `src/fastplms/models.toml` or their renderer.

Execute code snippets, validate internal links, and run:

```bash
PYTHONPATH=src python -m tools.artifacts.generate_docs --check
python -m tools.debug.check_notation
python -m pytest tests/release/test_documentation.py \
  tests/release/test_model_card_licenses.py -v
```

## Review scope

Keep unrelated user changes intact. Do not commit, push, upload, delete a live
Hub repository, or open a pull request unless the maintainer explicitly asks.
