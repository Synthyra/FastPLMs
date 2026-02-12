---
library_name: transformers
tags: []
---

# NOTE
The GitHub with the implementation and requirements.txt can be found [here](https://github.com/Synthyra/FastPLMs.git)

# Profluent-E1
[Synthyra's version of Profluent-E1](https://github.com/Synthyra/Profluent-E1-300M) is a faithful implementation of Profluent's [E1](https://www.profluent.bio/showcase/e1) models ([license](https://github.com/Profluent-AI/E1/tree/main?tab=License-1-ov-file)) that integrates Huggingface AutoModel compatability and nice embedding functionality.

## Attention backend defaults
Flex Attention with a block mask that ignores pad tokens is the default attention backend. If Flex Attention is unavailable, E1 falls back to native PyTorch attention.

For throughput and memory efficiency, `torch.compile(...)` is heavily recommended, especially when using Flex Attention.


## Use with 🤗 transformers
### Supported models
```python
model_dict = {
    # Synthyra/Profluent-E1-150M
    'Profluent-E1-150M': 'Profluent-Bio/E1-150m',
    # Synthyra/Profluent-E1-150M
    'Profluent-E1-300M': 'Profluent-Bio/E1-300m',
    # Synthyra/Profluent-E1-150M
    'Profluent-E1-600M': 'Profluent-Bio/E1-600m',
}
```

```python
import torch
from transformers import AutoModelForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForMaskedLM.from_pretrained('Synthyra/Profluent-E1-150M', trust_remote_code=True, dtype=torch.bfloat16).eval().to(device)

sequences = ['MPRTEIN', 'MSEQWENCE']
batch = model.prep_tokens.get_batch_kwargs(sequences, device=device)

output = model(**batch) # get all hidden states with output_hidden_states=True
print(output.logits.shape) # language modeling logits, (batch_size, seq_len, vocab_size), (2, 11, 34)
print(output.last_hidden_state.shape) # last hidden state of the model, (batch_size, seq_len, hidden_size), (2, 11, 768)
print(output.loss) # language modeling loss if you passed labels
#print(output.hidden_states) # all hidden states if you passed output_hidden_states=True (in tuple)
#print(outout.attentions) # all attention matrices if you passed output_attentions=True (in tuple)
```

Our E1 implementation also supports sequence and token level classification tasks like ESM2. Simply pass the number of labels during initialization.

```python
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

model = AutoModelForSequenceClassification.from_pretrained('Synthyra/Profluent-E1-150M', num_labels=2, trust_remote_code=True)
logits = model(**batch, labels=labels).logits
print(logits.shape) # (batch_size, num_labels), (2, 2)
```

E1 weights were trained in bf16 and are in bf16 by default. You can load them in the precision of your choosing by leveraging the dtype parameter:
```python
import torch
model = AutoModelForMaskedLM.from_pretrained('Synthyra/Profluent-E1-150M', trust_remote_code=True, dtype=torch.float) # fp32
```

## Embed entire datasets with no new code
To embed a list of protein sequences **fast**, just call embed_dataset. Sequences are sorted to reduce padding tokens, so the initial progress bar estimation is usually much longer than the actual time it will take.

Example:
```python
embedding_dict = model.embed_dataset(
    sequences=[
        'MALWMRLLPLLALLALWGPDPAAA', ... # list of protein sequences
    ],
    batch_size=2, # adjust for your GPU memory
    max_len=512, # adjust for your needs
    full_embeddings=False, # if True, no pooling is performed
    embed_dtype=torch.float32, # cast to what dtype you want
    pooling_types=['mean', 'cls'], # more than one pooling type will be concatenated together
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
model = AutoModelForSequenceClassification.from_pretrained('Synthyra/Profluent-E1-150M', num_labels=2, trust_remote_code=True)
# these modules handle E1 attention layers
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

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


### Citation
If you use any of this implementation or work please cite the following DOI and Profluent's paper.

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

```
 @article{Jain_Beazer_Ruffolo_Bhatnagar_Madani_2025,
    title={E1: Retrieval-Augmented Protein Encoder Models},
    url={https://www.biorxiv.org/content/early/2025/11/13/2025.11.12.688125},
    DOI={10.1101/2025.11.12.688125},
    journal={bioRxiv},
    publisher={Cold Spring Harbor Laboratory},
    author={Jain, Sarthak and Beazer, Joel and Ruffolo, Jeffrey A and Bhatnagar, Aadyot and Madani, Ali},
    year={2025}
}
```