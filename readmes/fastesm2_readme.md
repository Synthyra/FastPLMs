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

### Contact prediction
Because we can output attentions using the naive attention implementation, the contact prediction is also supported
```python
with torch.no_grad():
    contact_map = model.predict_contacts(**tokenized).squeeze().cpu().numpy() # (seq_len, seq_len)
```
![image/png](https://cdn-uploads.huggingface.co/production/uploads/62f2bd3bdb7cbd214b658c48/9707OSXZ3Wdgn0Ni-55T-.png)

## Embed entire datasets with no new code
To embed a list of protein sequences **fast**, just call embed_dataset. Sequences are sorted to reduce padding tokens, so the initial progress bar estimation is usually much longer than the actual time it will take.

Example:
```python
embedding_dict = model.embed_dataset(
    sequences=[
        'MALWMRLLPLLALLALWGPDPAAA', ... # list of protein sequences
    ],
    tokenizer=model.tokenizer,
    batch_size=2, # adjust for your GPU memory
    max_len=512, # adjust for your needs
    full_embeddings=False, # if True, no pooling is performed
    embed_dtype=torch.float32, # cast to what dtype you want
    pooling_types=['mean', 'cls'], # more than one pooling type will be concatenated together
    num_workers=0, # if you have many cpu cores, we find that num_workers = 4 is fast for large datasets
    sql=False, # if True, embeddings will be stored in SQLite database
    sql_db_path='embeddings.db',
    save=True, # if True, embeddings will be saved as a .pth file
    save_path='embeddings.pth',
)
# embedding_dict is a dictionary mapping sequences to their embeddings as tensors for .pth or numpy arrays for sql
```

```
model.embed_dataset()
Args:
    sequences: List of protein sequences
    batch_size: Batch size for processing
    max_len: Maximum sequence length
    full_embeddings: Whether to return full residue-wise (True) embeddings or pooled (False)
    pooling_type: Type of pooling ('mean' or 'cls')
    num_workers: Number of workers for data loading, 0 for the main process
    sql: Whether to store embeddings in SQLite database - will be stored in float32
    sql_db_path: Path to SQLite database
    
Returns:
    Dictionary mapping sequences to embeddings, or None if sql=True

Note:
    - If sql=True, embeddings can only be stored in float32
    - sql is ideal if you need to stream a very large dataset for training in real-time
    - save=True is ideal if you can store the entire embedding dictionary in RAM
    - sql will be used if it is True and save is True or False
    - If your sql database or .pth file is already present, they will be scanned first for already embedded sequences
    - Sequences will be truncated to max_len and sorted by length in descending order for faster processing
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