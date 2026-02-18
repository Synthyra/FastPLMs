---
library_name: transformers
tags: []
---

# NOTE
The GitHub with the implementation and requirements can be found [here](https://github.com/Synthyra/FastPLMs.git).

# DPLM2
Synthyra DPLM2 checkpoints are HuggingFace AutoModel compatible and include FastPLMs embedding helpers.

## Supported models
```python
model_dict = {
    "Synthyra/DPLM2-150M": "airkingbd/dplm2_150m",
    "Synthyra/DPLM2-650M": "airkingbd/dplm2_650m",
    "Synthyra/DPLM2-3B": "airkingbd/dplm2_3b",
}
```

## Use with transformers
```python
import torch
from transformers import AutoModel, AutoModelForMaskedLM

model_path = "Synthyra/DPLM2-150M"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=torch.float16).eval()
tokenizer = model.tokenizer

batch = tokenizer(["MPRTEIN", "MSEQWENCE"], padding=True, return_tensors="pt")
with torch.no_grad():
    hidden = model(**batch).last_hidden_state

mlm = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.float16).eval()
with torch.no_grad():
    logits = mlm(**batch).logits
```

## DPLM2 modality types
DPLM2 infers `type_ids` automatically from `input_ids` and `attention_mask` when they are not provided.

## Attention backend
`sdpa` is the default backend. Flex Attention is available by setting `config.attn_backend = "flex"` before loading.

## Embed datasets
All DPLM2 models inherit `EmbeddingMixin`, so you can call `model.embed_dataset(...)` directly.
