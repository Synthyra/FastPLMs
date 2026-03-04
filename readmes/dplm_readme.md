---
library_name: transformers
tags: []
---

# NOTE
The GitHub with the implementation and requirements can be found [here](https://github.com/Synthyra/FastPLMs.git).

# DPLM
Synthyra DPLM checkpoints are HuggingFace AutoModel compatible and include FastPLMs embedding helpers.

## Supported models
```python
model_dict = {
    "Synthyra/DPLM-150M": "airkingbd/dplm_150m",
    "Synthyra/DPLM-650M": "airkingbd/dplm_650m",
    "Synthyra/DPLM-3B": "airkingbd/dplm_3b",
}
```

## Use with transformers
```python
import torch
from transformers import AutoModel, AutoModelForMaskedLM

model_path = "Synthyra/DPLM-150M"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=torch.float16).eval()
tokenizer = model.tokenizer

batch = tokenizer(["MPRTEIN", "MSEQWENCE"], padding=True, return_tensors="pt")
with torch.no_grad():
    hidden = model(**batch).last_hidden_state

mlm = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.float16).eval()
with torch.no_grad():
    logits = mlm(**batch).logits
```

## Attention backends

`sdpa` (PyTorch Scaled Dot Product Attention) is the default.

| Backend | Key | Notes |
| :--- | :--- | :--- |
| PyTorch SDPA | `"sdpa"` | Default. Exact numerics, stable on all hardware. |
| Flash Attention | `"kernels_flash"` | Fastest on Ampere/Hopper GPUs. Requires `pip install kernels` (pre-built — no hours-long compilation). Outputs are not bitwise identical to SDPA due to online softmax reordering; differences are often small but not guaranteed to be inconsequential — use `"sdpa"` if exact numerics matter. |
| Flex Attention | `"flex"` | Skips padding tokens via block mask — faster on variable-length batches. Near-exact numerics. First use compiles a Triton kernel (30–120 s). Best combined with `torch.compile`. |
| Auto | `"auto"` | Picks the best available: `kernels_flash` → `flex` → `sdpa`. |

Set via config before loading, or change on the model after loading (DPLM propagates the change to all attention layers immediately):

```python
from transformers import AutoConfig, AutoModel

# Option 1: set before loading
config = AutoConfig.from_pretrained("Synthyra/DPLM-150M", trust_remote_code=True)
config.attn_backend = "flex"
model = AutoModel.from_pretrained("Synthyra/DPLM-150M", config=config, trust_remote_code=True)

# Option 2: set after loading
model = AutoModel.from_pretrained("Synthyra/DPLM-150M", trust_remote_code=True)
model.attn_backend = "flex"  # propagates to all attention layers in-place
```

## Embed datasets
All DPLM models inherit `EmbeddingMixin`, so you can call `model.embed_dataset(...)` directly.
