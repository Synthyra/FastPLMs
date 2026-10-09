---
id: foundry.secrets_env
kind: library_module
status: active
aliases: [foundry.secrets_env]
---
# foundry.secrets_env

The one loader for a credentials file. It moves `NAME=value` lines from `.secrets.env` into
`os.environ` and hands back the names it set, never a value. Every client after that reads
`os.environ` or lets its own SDK resolve the variable. The module docstring states each rule,
and `tests/tier1_unit/test_secrets_env.py` tests each one.

```python
from foundry.secrets_env import load_secrets_env

load_secrets_env()   # the nearest .secrets.env above foundry: the workspace root, or a clone's root
```

| Name | Does |
|---|---|
| `load_secrets_env(start=None, *, path=None, names=None, override=False)` | Loads `path`, else the file `FOUNDRY_SECRETS` names, else the nearest `.secrets.env` at or above `start`, which defaults to this package. A named file that does not exist loads nothing. A variable already set to a non-empty value wins unless `override`. `names` limits what loads. Returns the names it set |
| `load_secrets(verbose=False)` | `load_secrets_env()` returning how many names it set, the form the Atlas entities, DatasetDev, and serving/api call. `verbose` prints the count and the file's path |
| `find_secrets_file(start=None, *, path=None)` | The file `load_secrets_env` would load, or None |
| `parse_secrets_file(path)` | The file's pairs, values included, so the parsing rules can be tested |
| `SECRETS_FILENAME`, `ALIASES` | The file name, and `WANDB_TOKEN`, `HUGGINGFACE_TOKEN`, and `HUGGING_FACE_HUB_TOKEN` mapped to the names wandb and huggingface_hub read |

foundry sits at the workspace root, and a projection vendors it inside the repository, so the
default search finds the workspace's file here and the repository's file in a clone.

Promoted by [the next promotion](../../docs/decisions/workspace/2026-09-22_next_promotion.md), which records
whose rules were kept. Consumers: `tools/workspace/secrets.py` (`ws doctor`, `ws capture`,
`ws remote run`), atlas, dataset_builders, serving_api,
base_model_distillation, sae_selfies, annotation_vocabulary, property_lowcode, property_oracles,
embedding_translation, contact_esmc6b, folding_service, and gem_integration. A consumer that
publishes a projection lists this module under `github.includes`, so `ws publish` vendors it.

Tests: `tests/tier1_unit/test_secrets_env.py`, with invented names and values under `tmp_path`.
