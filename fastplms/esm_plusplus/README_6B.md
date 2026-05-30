---
library_name: transformers
license: mit
tags:
  - biology
  - esm
  - protein
  - protein-language-model
  - masked-language-modeling
---

# ESM++ 6B

[ESM++](https://github.com/Synthyra/FastPLMs) is a Hugging Face compatible implementation of [Biohub ESMC](https://biohub.ai/esm/protein) ([license](https://github.com/Biohub/esm/blob/main/LICENSE.md)).
This checkpoint corresponds to the 6 billion parameter ESMC model released as [`biohub/ESMC-6B`](https://huggingface.co/biohub/ESMC-6B).

This repository includes the Biohub ESM MIT license in `LICENSE`.

The 6B model has 80 transformer layers, hidden size 2560, and 40 attention heads. It is large enough that `dtype=torch.bfloat16` or `torch.float16` plus `device_map="auto"` is usually the practical loading path.

## Attention Backends

`sdpa` is the default backend. Set `config.attn_backend` before loading if you want a different attention implementation.

| Backend | Key | Notes |
| :--- | :--- | :--- |
| PyTorch SDPA | `"sdpa"` | Default. Exact numerics and stable on all hardware. |
| Flash Attention | `"kernels_flash"` | Fastest on Ampere/Hopper GPUs when `kernels` is installed. Outputs are not bitwise identical to SDPA. |
| Flex Attention | `"flex"` | Skips padding tokens via block masks. First use compiles a Triton kernel. |
| Auto | `"auto"` | Picks the best available backend: `kernels_flash`, then `flex`, then `sdpa`. |

```python
import torch
from transformers import AutoConfig, AutoModelForMaskedLM

config = AutoConfig.from_pretrained(
    "Synthyra/ESMplusplus_6B",
    trust_remote_code=True,
)
config.attn_backend = "auto"

model = AutoModelForMaskedLM.from_pretrained(
    "Synthyra/ESMplusplus_6B",
    config=config,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)
```

## Masked Language Modeling

```python
import torch
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained(
    "Synthyra/ESMplusplus_6B",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = model.tokenizer

sequences = ["MPRTEIN", "MSEQWENCE"]
inputs = tokenizer(sequences, padding=True, return_tensors="pt")
inputs = inputs.to(model.device)

with torch.no_grad():
    output = model(**inputs)

print(output.logits.shape)
print(output.last_hidden_state.shape)
```

Pass `output_hidden_states=True` if you need all intermediate hidden states.

## Embed Datasets

All FastPLMs sequence models include `embed_dataset`, which handles batching, length sorting, pooling, FASTA parsing, optional resume from existing outputs, and `.pth` or SQLite storage.

```python
import torch
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained(
    "Synthyra/ESMplusplus_6B",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

embedding_dict = model.embed_dataset(
    sequences=[
        "MALWMRLLPLLALLALWGPDPAAA",
        "MSEQWENCE",
        "MPRTEIN",
    ],
    batch_size=1,
    max_len=1024,
    full_embeddings=False,
    embed_dtype=torch.float32,
    pooling_types=["mean", "cls"],
    num_workers=0,
    save=True,
    save_path="esmplusplus_6b_embeddings.pth",
)

print(embedding_dict["MPRTEIN"].shape)
```

For residue-level embeddings, set `full_embeddings=True`:

```python
residue_embeddings = model.embed_dataset(
    sequences=["MALWMRLLPLLALLALWGPDPAAA"],
    batch_size=1,
    max_len=1024,
    full_embeddings=True,
    embed_dtype=torch.float32,
    save=False,
)
```

For very large datasets, write embeddings directly to SQLite:

```python
model.embed_dataset(
    fasta_path="proteins.fasta",
    batch_size=1,
    max_len=1024,
    pooling_types=["mean"],
    sql=True,
    sql_db_path="esmplusplus_6b_embeddings.db",
    save=False,
)
```

`embed_dataset` returns a dictionary when `sql=False`. With `sql=True`, embeddings are written to the database and loaded as needed.

## Classification Heads

ESM++ supports sequence-level and token-level classification through the standard Transformers auto classes.

```python
import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "Synthyra/ESMplusplus_6B",
    num_labels=2,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

tokenized = model.tokenizer(
    ["MPRTEIN", "MSEQWENCE"],
    padding=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    logits = model(**tokenized).logits

print(logits.shape)
```

## LoRA Fine-Tuning

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "Synthyra/ESMplusplus_6B",
    num_labels=2,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.01,
    bias="none",
    target_modules=[
        "layernorm_qkv.1",
        "out_proj",
        "query",
        "key",
        "value",
        "dense",
    ],
)

model = get_peft_model(model, lora_config)
```

## Attention Maps

Optimized attention backends do not return attention maps directly. ESM++ can compute them manually with `output_attentions=True`, but this is much slower and memory-heavy for the 6B model.

```python
with torch.no_grad():
    output = model(**inputs, output_attentions=True)

attentions = output.attentions
print(len(attentions))
print(attentions[0].shape)
```

## Load Biohub Source Weights

You can also load the Biohub source weights directly through FastPLMs:

```python
from fastplms.esm_plusplus.modeling_esm_plusplus import ESMplusplusForMaskedLM

model = ESMplusplusForMaskedLM.from_pretrained_esm("esmc-6b")
```

The source repository is [`biohub/ESMC-6B`](https://huggingface.co/biohub/ESMC-6B).
The Biohub ESM license is available at https://github.com/Biohub/esm/blob/main/LICENSE.md.

## Citation

```bibtex
@misc{FastPLMs,
  author={Hallee, Logan and Bichara, David and Gleghorn, Jason P.},
  title={FastPLMs: Fast, efficient, protein language model inference from Hugging Face AutoModel.},
  year={2024},
  url={https://huggingface.co/Synthyra/ESMplusplus_6B},
  DOI={10.57967/hf/3726},
  publisher={Hugging Face}
}
```

```bibtex
@misc{candido2026language,
  title  = {Language Modeling Materializes a World Model of Protein Biology},
  author = {Candido, Salvatore and Hayes, Thomas and Derry, Alexander and Rao, Roshan
            and Lin, Zeming and Verkuil, Robert and Wu, Bryan and Lee, Jin Sub
            and Bruguera, Elise S. and Keval, Jehan A. and Kopylov, Mykhailo
            and Pak, John E. and Wu, Wesley and Thomas, Neil and Mataraso, Samson
            and Hsu, Alvin and Trotman-Grant, Ashton C. and Fatras, Kilian
            and dos Santos Costa, Allan and Badkundri, Rohil and Ak{\i}n, Halil
            and Oktay, Deniz and Deaton, Jonathan and Montabana, Elizabeth
            and Sitwala, Hrishita and Yu, Yue and Wiggert, Marius
            and Carlin, Dylan Alexander and Goering, Anthony W. and Blazejewski, Tomasz
            and Sandora, McCullen and Hla, Michael and Jia, Tina Z.
            and Kloker, Leon H. and Sofroniew, Nicholas J. and Uehara, Masatoshi
            and Pannu, Jassi and Bachas, Sharrol and Liu, Daniel S.
            and Sercu, Tom and Rives, Alexander},
  year   = {2026},
  url    = {https://biohub.ai/papers/esm_protein.pdf},
  note   = {Preprint}
}
```
