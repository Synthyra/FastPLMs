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

## Attention backend
`sdpa` is the default backend. Flex Attention is available by setting `config.attn_backend = "flex"` before loading.

## Embed datasets
All DPLM models inherit `EmbeddingMixin`, so you can call `model.embed_dataset(...)` directly.
