---
library_name: transformers
tags: []
---

# FastESM
FastESM is a Huggingface compatible plug in version of ESM2 rewritten with a newer PyTorch attention implementation.

Load any ESM2 models into a FastEsm model to dramatically speed up training and inference without **ANY** cost in performance.

Outputting attention maps (or the contact prediction head) is not natively possible with SDPA. You can still pass ```output_attentions``` to have attention calculated manually and returned.
Various other optimizations also make the base implementation slightly different than the one in transformers.

## Use with 🤗 transformers

### Supported models
```python
model_dict = {
    # Synthyra/ESM2-8M
    'ESM2-8M': 'facebook/esm2_t6_8M_UR50D',
    # Synthyra/ESM2-35M
    'ESM2-35M': 'facebook/esm2_t12_35M_UR50D',
    # Synthyra/ESM2-150M
    'ESM2-150M': 'facebook/esm2_t30_150M_UR50D',
    # Synthyra/ESM2-650M
    'ESM2-650M': 'facebook/esm2_t33_650M_UR50D',
    # Synthyra/ESM2-3B
    'ESM2-3B': 'facebook/esm2_t36_3B_UR50D',
}
```

### For working with embeddings
```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = 'Synthyra/ESM2-8M'
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).eval()
tokenizer = model.tokenizer

sequences = ['MPRTEIN', 'MSEQWENCE']
tokenized = tokenizer(sequences, padding=True, return_tensors='pt')
with torch.no_grad():
    embeddings = model(**tokenized).last_hidden_state

print(embeddings.shape) # (2, 11, 1280)
```

### For working with sequence logits
```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).eval()
with torch.no_grad():
    logits = model(**tokenized).logits

print(logits.shape) # (2, 11, 33)
```

### For working with attention maps
```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).eval()
with torch.no_grad():
    attentions = model(**tokenized, output_attentions).attentions # tuples of (batch_size, num_heads, seq_len, seq_len)

print(attentions[-1].shape) # (2, 20, 11, 11) 
```

## Embed entire datasets with no new code
To embed a list of protein sequences **fast**, just call embed_dataset. Sequences are sorted to reduce padding tokens, so the initial progress bar estimation is usually much longer than the actual time.
```python
embeddings = model.embed_dataset(
    sequences=sequences, # list of protein strings
    batch_size=16, # embedding batch size
    max_len=2048, # truncate to max_len
    full_embeddings=True, # return residue-wise embeddings
    full_precision=False, # store as float32
    pooling_type='mean', # use mean pooling if protein-wise embeddings
    num_workers=0, # data loading num workers
    sql=False, # return dictionary of sequences and embeddings
)

_ = model.embed_dataset(
    sequences=sequences, # list of protein strings
    batch_size=16, # embedding batch size
    max_len=2048, # truncate to max_len
    full_embeddings=True, # return residue-wise embeddings
    full_precision=False, # store as float32
    pooling_type='mean', # use mean pooling if protein-wise embeddings
    num_workers=0, # data loading num workers
    sql=True, # store sequences in local SQL database
    sql_db_path='embeddings.db', # path to .db file of choice
)
```


### Citation
If you use any of this implementation or work please cite it (as well as the [ESM2](https://www.science.org/doi/10.1126/science.ade2574) paper).
```
@misc {FastESM2,
	author       = { Hallee, L. and Bichara, D. and Gleghorn, J, P. },
	title        = { FastESM2 },
	year         = 2024,
	url          = { https://huggingface.co/Synthyra/FastESM2_650 },
	doi          = { 10.57967/hf/3729 },
	publisher    = { Hugging Face }
}
```