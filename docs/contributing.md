# Contributing

## Getting Started

1. Fork the repository and clone your fork
2. Install dependencies: `pip install -r requirements.txt`
3. Tests run in Docker only (see below)

## Code Style

- **Python**: PEP 8, type hints in function signatures
- **No unnecessary comments**: Only where logic is not self-evident
- **Hard asserts**: No silent recovery or defensive error handling
- **No `.get()`, `getattr()`, `hasattr()`**: Let missing keys throw `KeyError`/`AttributeError`

### Import Ordering

1. Standard library: `import xyz`
2. Third-party: `from xyz import q`
3. Local/repo: `from models.foo import Bar`

Within each group, `import x` lines come before `from x import y` lines.

### Import Grouping

Never import from the same package on separate lines:

```python
# Bad
from torch import nn
from torch import optim

# Good
from torch import nn, optim

# For many names, use parenthesized form
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    PreTrainedModel,
)
```

## Adding a New Model

### 1. Create the Package

```
fastplms/new_model/
    __init__.py
    modeling_new_model.py    # PreTrainedModel + PretrainedConfig
    get_new_model_weights.py # Weight conversion from official checkpoint
    README.md                # HuggingFace model card README
    LICENSE                  # Model license

testing/official/new_model.py  # Load official model for compliance testing
```

### 2. Implement the Model

Your `modeling_*.py` should:

- Subclass `PreTrainedModel` and `EmbeddingMixin`
- Define a `PretrainedConfig` subclass with `attn_backend` attribute
- Implement the `AttentionBackend` enum and backend resolution
- Implement `_embed(input_ids, attention_mask)` returning last hidden states
- Register in `config.json` via `auto_map`:

```json
{
  "auto_map": {
    "AutoConfig": "modeling_new_model.NewModelConfig",
    "AutoModelForMaskedLM": "modeling_new_model.NewModelForMaskedLM"
  }
}
```

### 3. Add Weight Conversion

`get_*_weights.py` should:
1. Load the official checkpoint
2. Remap parameter names to match your architecture
3. Export `config.json`, `pytorch_model.bin`, and modeling source files
4. The output directory can be pushed to HuggingFace

### 4. Add Compliance Testing

`testing/official/new_model.py` should expose:

```python
def load_official_model(reference_repo_id: str, device: torch.device, dtype: torch.dtype):
    # Load and wrap the official model
    # Return (wrapped_model, tokenizer) where wrapped_model has .logits and .hidden_states outputs
    ...
```

### 5. Register in Test Configuration

Add your model to `testing/conftest.py`:

```python
# In MODEL_REGISTRY (for fast CI, pick the smallest checkpoint)
"new_model": {
    "fast_path": "Synthyra/NewModel-150M",
    "official_path": "org/official-model",
    "load_official": "testing.official.new_model",
    "model_type": "NewModel",
    "uses_tokenizer": True,
},

# In FULL_MODEL_REGISTRY (all checkpoints with size_category)
"new_model_150m": {
    "fast_path": "Synthyra/NewModel-150M",
    "official_path": "org/official-model-150m",
    "load_official": "testing.official.new_model",
    "model_type": "NewModel",
    "uses_tokenizer": True,
    "size_category": "small",
},
```

### 6. Add HuggingFace README

Create `fastplms/new_model/README.md` with the HuggingFace model card content and `fastplms/new_model/LICENSE` with the model license.

### 7. Add a Per-Family Dockerfile

Create `Dockerfile.<family>` that layers on top of `fastplms-base` and installs your model's native reference deps. Add the family to `build_images.sh` so `./build_images.sh` picks it up.

If your model's native package conflicts with another (e.g. transformers version pin, torchtext pin), prefer either:
- Loading the native package from a `sys.path`-injected submodule (see `testing/official/__init__.py` for the ESM++ pattern), or
- Using a HuggingFace `transformers` reference class instead (DPLM uses `EsmForMaskedLM` for this reason).

### 8. Add Parity Tolerances

Add a `ParityTolerances(...)` entry in `FAMILY_TOLERANCES` at the top of `testing/test_parity.py`. Start with the default, then tighten as you investigate failures.

### 9. Update `update_HF.py`

Add entries for pushing your model's files to the Hub.

## Running Tests

All tests must run in Docker. Never run tests natively on Windows (missing Triton, flash-attention, CUDA kernels). Always pass `--ipc=host`.

```bash
# Build base + your family image
./build_images.sh new_model

# Run your model's parity suite (its own image)
docker run --rm --gpus all --ipc=host -v $(pwd):/workspace fastplms-new_model \
    python -m pytest /workspace/testing/test_parity.py -k new_model -v

# Broader smoke tests in the monolithic image
docker build -t fastplms .
docker run --gpus all --ipc=host fastplms python -m pytest /app/testing/ -k new_model -v
```

## Required Passing Tests

Before submitting a PR for a new model, ensure inside the family's Docker image:

1. `test_parity.py::test_tokenizer_parity[<family>]`
2. `test_parity.py::test_weight_parity_fp32[<family>]`
3. `test_parity.py::test_forward_parity_fp32[<family>-{single,uniform,skewed}]` (all three padding scenarios)
4. `test_parity.py::test_forward_parity_bf16[<family>-{single,uniform,skewed}]`
5. `test_parity.py::test_padding_does_not_pollute_valid_positions_fp32[<family>]` (tokenizer-mode families)
6. `test_parity.py::test_backend_consistency_fp32[<family>]`

And in the monolithic image:

7. `test_automodel_loads` and `test_automodel_forward_pass`
8. `test_nan_stability`
9. `test_batch_single_match` (tokenizer-mode models)

## Reporting Issues

Found a bug or have a feature request? Open a [GitHub Issue](https://github.com/Synthyra/FastPLMs/issues).
