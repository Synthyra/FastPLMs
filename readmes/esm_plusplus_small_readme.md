---
library_name: transformers
tags: []
---

# NOTE
The GitHub with the implementation and requirements.txt can be found [here](https://github.com/Synthyra/FastPLMs.git)

# ESM++
[ESM++](https://github.com/Synthyra/ESMplusplus) is a faithful implementation of [ESMC](https://www.evolutionaryscale.ai/blog/esm-cambrian) ([license](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement)) that allows for batching and standard Huggingface compatibility without requiring the ESM Python package.
The small version corresponds to the 300 million parameter version of ESMC.

## Attention backend defaults
Flex Attention with a block mask that ignores pad tokens is the default attention backend. If Flex Attention is unavailable, ESM++ falls back to native PyTorch attention.

For throughput and memory efficiency, `torch.compile(...)` is heavily recommended, especially when using Flex Attention.


## Use with 🤗 transformers
```python
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True)
tokenizer = model.tokenizer

sequences = ['MPRTEIN', 'MSEQWENCE']
tokenized = tokenizer(sequences, padding=True, return_tensors='pt')

# tokenized['labels'] = tokenized['input_ids'].clone() # correctly mask input_ids and set unmasked instances of labels to -100 for MLM training

output = model(**tokenized) # get all hidden states with output_hidden_states=True
print(output.logits.shape) # language modeling logits, (batch_size, seq_len, vocab_size), (2, 11, 64)
print(output.last_hidden_state.shape) # last hidden state of the model, (batch_size, seq_len, hidden_size), (2, 11, 960)
print(output.loss) # language modeling loss if you passed labels
#print(output.hidden_states) # all hidden states if you passed output_hidden_states=True (in tuple)
```

ESM++ also supports sequence and token level classification tasks like ESM2. Simply pass the number of labels during initialization.

```python
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

model = AutoModelForSequenceClassification.from_pretrained('Synthyra/ESMplusplus_small', num_labels=2, trust_remote_code=True)
logits = model(**tokenized).logits
print(logits.shape) # (batch_size, num_labels), (2, 2)
```

ESM++ weights are fp32 by default. You can load them in fp16 or bf16 like this:
```python
import torch
model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True, dtype=torch.float16) # or torch.bfloat16
```

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

## Fine-tuning with 🤗 peft
```python
model = AutoModelForSequenceClassification.from_pretrained('Synthyra/ESMplusplus_small', num_labels=2, trust_remote_code=True)
# these modules handle ESM++ and ESM2 attention layers
target_modules = ["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense"]

lora_config = LoraConfig(
    r=8, # choose lora parameters to your liking
    lora_alpha=16,
    lora_dropout=0.01,
    bias="none",
    target_modules=target_modules,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Unfreeze the classifier head
for param in model.classifier.parameters():
    param.requires_grad = True
```

For a more thourough example of fine-tuning, check out our example script [here](https://github.com/Synthyra/FastPLMs/blob/main/fine_tuning_example.py).


## Returning attention maps
Flex Attention with a pad-token block mask is used by default for attention calculations, and native PyTorch attention is the fallback. Optimized attention paths do not return attention maps directly.
ESM++ has the option to ```output_attentions```, which will calculate attention manually. This is much slower, so do not use unless you need the attention maps.

```python
output = model(**tokenized, output_attentions=True)
att = output.attentions
len(att) # 30, one for each layer, size (batch_size, num_heads, seq_len, seq_len) each
```

## Comparison across floating-point precision and implementations
We measured the difference of the last hidden states of the fp32 weights vs. fp16 or bf16. We find that the fp16 is closer to the fp32 outputs, so we recommend loading in fp16.
Please note that the ESM package also loads ESMC in fp32 but casts to bf16 by default, which has its share of advantages and disadvantages in inference / training - so load whichever you like for half precision.

Average MSE FP32 vs. FP16: 0.00000003

Average MSE FP32 vs. BF16: 0.00000140

We also measured the difference between the outputs of ESM++ vs. ESMC (both in bfloat16) on 1000 random sequences to ensure compliance with the ESM package.

Average MSE of last hidden state: 7.74e-10

You can load the weights from the ESM package instead of transformers by replacing .from_pretrained(...) to .from_pretrained_esm('esmc_300m')

## Model probes
We employ linear probing techniques on various PLMs and standard datasets, similar our previous [paper](https://www.biorxiv.org/content/10.1101/2024.07.30.605924v1), to assess the intrinsic correlation between pooled hidden states and valuable properties. ESMC (and thus ESM++) perform very well.

The plot below showcases performance normalized between the negative control (random vector embeddings) and the best performer. Classification task scores are averaged between MCC and F1 (or F1max for multilabel) and regression tasks are averaged between Spearman rho and R2.
![image/png](https://cdn-uploads.huggingface.co/production/uploads/62f2bd3bdb7cbd214b658c48/2zyUZeHyOgCR_twvPF2Wy.png)

## Inference speeds
We look at various ESM models and their throughput on an H100. Adding efficient batching between ESMC and ESM++ significantly improves the throughput, although ESM++ is also faster than ESMC for batch size one. ESM++ small is even faster than ESM2-35M with long sequences!
The most gains will be seen with PyTorch > 2.5 on linux machines.
![image/png](https://cdn-uploads.huggingface.co/production/uploads/62f2bd3bdb7cbd214b658c48/RfLRSchFivdsqJrWMh4bo.png)

### Citation
If you use any of this implementation or work please cite it (as well as the ESMC preprint).

```
@misc {FastPLMs,
    author       = { Hallee, Logan and Bichara, David and Gleghorn, Jason P.},
    title        = { FastPLMs: Fast, efficient, protien language model inference from Huggingface AutoModel.},
    year         = {2024},
    url          = { https://huggingface.co/Synthyra/ESMplusplus_small },
    DOI          = { 10.57967/hf/3726 },
    publisher    = { Hugging Face }
}
```