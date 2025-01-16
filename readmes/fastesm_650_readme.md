---
library_name: transformers
tags: []
---

# FastESM
FastESM is a Huggingface compatible plug in version of ESM2 rewritten with a newer PyTorch attention implementation.

Load any ESM2 models into a FastEsm model to dramatically speed up training and inference without **ANY** cost in performance.

Outputting attention maps (or the contact prediction head) is not natively possible with SDPA. You can still pass ```output_attentions``` to have attention calculated manually and returned.
Various other optimizations also make the base implementation slightly different than the one in transformers.

# FastESM2-650

## A faster half-precision version of ESM2-650 with FlashAttention2 and longer context
To enhance the weights with longer context and better fp16 support, we trained ESM2-650 50000 additional steps with a traditional MLM objective (20% masking) in fp16 mixed precision on [OMGprot50](https://huggingface.co/datasets/tattabio/OMG_prot50) up to sequence length of **2048**.

## Use with 🤗 transformers

### For working with embeddings
```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = 'Synthyra/FastESM2_650'
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

## Model probes
We employ linear probing techniques on various PLMs and standard datasets, similar our previous [paper](https://www.biorxiv.org/content/10.1101/2024.07.30.605924v1), to assess the intrinsic correlation between pooled hidden states and valuable properties. FastESM performs very well.

The plot below showcases performance normalized between the negative control (random vector embeddings) and the best performer. Classification task scores are averaged between MCC and F1 (or F1max for multilabel) and regression tasks are averaged between Spearman rho and R2.
![image/png](https://cdn-uploads.huggingface.co/production/uploads/62f2bd3bdb7cbd214b658c48/d1Xi6k1Q4-9By_MtzTvdV.png)

## Comparison of half precisions
Presumabely because we trained in mixed-precision fp16, fp16 has closer outputs to the fp32 weights then bf16. Therefore, we recommend loading in fp16.

When summing the MSE of 1000 sequences vs. the fp32 weights:

Average MSE for FP16: 0.00000140

Average MSE for BF16: 0.00004125

### Inference speed
We look at various ESM models and their throughput on an H100. FastESM is over twice as fast as ESM2-650 with longer sequences. Requires PyTorch 2.5+ for the most savings, see [SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
![image/png](https://cdn-uploads.huggingface.co/production/uploads/62f2bd3bdb7cbd214b658c48/PvaBGfuJXEW2v_WLkt63y.png)

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
