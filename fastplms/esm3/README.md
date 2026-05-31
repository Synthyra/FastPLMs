---
library_name: transformers
license: mit
tags:
  - biology
  - protein-language-model
  - esm3
  - multimodal-protein-model
---

# FastPLMs ESM3 Small

FastPLMs ESM3 Small is a Hugging Face compatible implementation of Biohub's open ESM3 small model. It loads through `AutoModel`, supports sequence-only inference by default, and exposes ESM3's additional tensor tracks directly through normal keyword arguments.

This repository includes the Biohub ESM MIT license in `LICENSE`.

## Use With Transformers

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Synthyra/ESM3_small",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
).eval()

sequences = ["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"]
tokens = model.tokenize_sequences(sequences, device=model.device)

with torch.inference_mode():
    output = model(**tokens)

print(output.logits.shape) # sequence logits, (batch_size, seq_len, 64)
print(output.last_hidden_state.shape) # ESM3 embeddings, (batch_size, seq_len, hidden_size)
print(output.function_logits.shape) # function logits, (batch_size, seq_len, 8, 260)
```

You can also call sequence inference directly:

```python
output = model.forward_sequence(["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"])
```

Switch between SDPA and Flex Attention after loading:

```python
model.attn_backend = "flex"
output = model.forward_sequence(["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"])
model.attn_backend = "sdpa"
```

## Embed Entire Datasets

To embed a list of protein sequences, call `embed_dataset`. Sequences are deduplicated, sorted by length, optionally truncated, and embedded in batches.

```python
embedding_dict = model.embed_dataset(
    sequences=[
        "MALWMRLLPLLALLALWGPDPAAA",
        "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP",
    ],
    batch_size=2,
    max_len=512,
    full_embeddings=False,
    embed_dtype=torch.float32,
    pooling_types=["mean", "cls"],
    save=True,
    save_path="esm3_embeddings.pth",
)

# embedding_dict maps sequence strings to pooled tensors.
print(embedding_dict["MALWMRLLPLLALLALWGPDPAAA"].shape)
```

Residue-wise embeddings are available by setting `full_embeddings=True`:

```python
residue_embeddings = model.embed_dataset(
    sequences=["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"],
    batch_size=1,
    max_len=512,
    full_embeddings=True,
    save=False,
)

print(residue_embeddings["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"].shape)
```

FASTA input is also supported:

```python
embedding_dict = model.embed_dataset(
    fasta_path="proteins.fasta",
    batch_size=4,
    pooling_types=["mean"],
    save_path="esm3_fasta_embeddings.pth",
)
```

`embed_dataset` currently supports pooled `mean`, `cls`, and `max` embeddings, plus unpooled residue embeddings. It supports `.pth` saves; SQLite streaming is not enabled for the ESM3 wrapper yet.

## Multimodal Track Arguments

The default path is amino acid sequence inference. Additional ESM3 tracks can be supplied directly using the same tensor shapes as Biohub ESM3:

```python
tokens = model.tokenize_sequences(
    ["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"],
    device=model.device,
)

function_tokens = tokens["input_ids"].new_zeros((*tokens["input_ids"].shape, 8))

with torch.inference_mode():
    output = model(
        **tokens,
        function_tokens=function_tokens,
    )

print(output.sequence_logits.shape)
print(output.function_logits.shape)
```

Accepted track arguments include `sequence_tokens`, `structure_tokens`, `ss8_tokens`, `sasa_tokens`, `function_tokens`, `residue_annotation_tokens`, `average_plddt`, `per_res_plddt`, `structure_coords`, `chain_id`, and `sequence_id`. `input_ids` aliases `sequence_tokens`, and `attention_mask` is converted into `sequence_id` if no explicit `sequence_id` is provided.

## Loading Biohub Checkpoints Locally

You can build the FastPLMs wrapper from the Biohub checkpoint directly:

```python
from fastplms.esm3.modeling_esm3 import FastESM3Model

model = FastESM3Model.from_pretrained_esm("esm3-sm-open-v1", device="cuda")
```

This requires Hugging Face access to the gated `biohub/esm3-sm-open-v1` source repo.

## Biohub SDK Compatibility

The core forward path is self-contained. Higher-level Biohub SDK workflows are delegated lazily to the official `esm` submodule when available:

```python
# These methods use Biohub SDK dataclasses and generation configs.
encoded = model.encode(esm_protein)
decoded = model.decode(encoded)
generated = model.generate(esm_protein, generation_config)
```

Available delegated methods include `encode`, `decode`, `generate`, `batch_generate`, `logits`, and `forward_and_sample`.

## Source

- Biohub ESM repository: https://github.com/Biohub/esm
- Biohub ESM license: https://github.com/Biohub/esm/blob/main/LICENSE.md
- Paper: https://biohub.ai/papers/esm_protein.pdf
- Official model source: https://huggingface.co/biohub/esm3-sm-open-v1
