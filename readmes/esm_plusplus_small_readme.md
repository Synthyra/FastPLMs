---
library_name: transformers
tags: []
---

# ESM++
[ESM++](https://github.com/Synthyra/ESMplusplus) is a faithful implementation of [ESMC](https://www.evolutionaryscale.ai/blog/esm-cambrian) ([license](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement)) that allows for batching and standard Huggingface compatibility without requiring the ESM Python package.
The small version corresponds to the 300 million parameter version of ESMC.


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
model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True, torch_dtype=torch.float16) # or torch.bfloat16
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
Usually F.scaled_dot_product_attention is used for the attention calculations, which is much faster than native PyTorch. However, it cannot return attention maps.
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
@misc {ESMPlusPlus,
	author       = { Hallee, L. and Bichara, D. and Gleghorn, J, P. },
	title        = { ESMPlusPlus },
	year         = 2024,
	url          = { https://huggingface.co/Synthyra/ESMplusplus_small },
	doi          = { 10.57967/hf/3725 },
	publisher    = { Hugging Face }
}
```