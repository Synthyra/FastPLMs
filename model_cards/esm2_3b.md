---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESM2-3B

This checkpoint packages the FastPLMs `ESM2` implementation.

Accepted inputs are amino-acid sequences tokenized to residue IDs.
Supported Transformers entry points are `AutoConfig`, `AutoModel`,
`AutoModelForMaskedLM`, `AutoModelForSequenceClassification`,
`AutoModelForTokenClassification`.

## Capabilities

| Feature | Status |
| --- | --- |
| Sequence classification | Supported: base weights with an untrained task head |
| Token classification | Supported: base weights with an untrained task head |
| PEFT fine-tuning | Supported pattern: preserve the separately trained `classifier` |
| Embeddings | Supported: shared ordered embedding API |
| Test-time training | Supported: low-rank masked-residue adaptation |
| Attention variants | Supported: `eager`, `sdpa`, `flex_attention`, `flash_attention_2`, `flash_attention_3` |
| Compliance | Declared: exact release evidence is required |

A supported interface is not a pretrained downstream predictor. Classification
heads start untrained, and declared compliance metadata is not a claim that an
arbitrary local build passed its release gate.

## Install and platform requirements

Install the direct dependencies published with this model:

```bash
python -m pip install -r \
  "https://huggingface.co/Synthyra/ESM2-3B/resolve/main/requirements.txt"
```

The FastPLMs implementation itself is embedded in the model repository and loaded
by Transformers through `trust_remote_code=True`.

Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13 are required. The artifact requirements include the direct FlashAttention loader dependency. FlashAttention also requires compatible CUDA hardware and BF16 execution. The Hub quick start below requires network
access on first download. For an air-gapped run, first build the manifest-pinned
local artifact and use the offline form shown in the example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ESM2-3B"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="sdpa",
).eval()
```

For offline validation, replace `model_id` with the manifest-built
`dist/hub/ESM2-3B` path and pass `local_files_only=True`.

## Attention and compliance

The quick start selects `sdpa` explicitly. Declared variants are `eager`, `sdpa`, `flex_attention`, `flash_attention_2`,
`flash_attention_3`. An unavailable requested backend raises instead of
silently switching implementations.
`output_attentions=True` may use the documented, one-call eager fallback solely
to materialize attention tensors; the configured backend remains unchanged.

This family declares the `compliance` tier. Release evidence binds the exact
checkpoint, backend, dtype, hardware, inputs, and reference revision.

## Tokenization and forward inference

Load the tokenizer from the same artifact as the model. Padding is represented
explicitly by the attention mask:

```python
import torch
from transformers import AutoTokenizer

model_id = "Synthyra/ESM2-3B"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
batch = tokenizer(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    padding=True,
    return_tensors="pt",
)

with torch.inference_mode():
    output = model(**batch)

print(output.last_hidden_state.shape)
```

## Dataset embeddings

The shared embedding mixin preserves input order and biological-position
masking. It accepts sequences, identified records, mappings, or a FASTA path:

```python
pooled = model.embed_dataset(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    batch_size=2,
    pooling=("mean", "std"),
)
residues = model.embed_dataset(
    ["MSTNPKPQRKTKRNT"],
    full_embeddings=True,
)
print(pooled[0].tensor.shape)   # (2 * d,)
print(residues[0].tensor.shape) # (l, d)
```

Set `output` and `format="safetensors"` or `"sqlite"` for transactional,
bounded-memory persistence. Resume verifies input order, model state, tokenizer
policy, backend, dtype, and pooling configuration before appending.

## Downstream classification

Both downstream AutoClasses reuse the checkpoint backbone and initialize a new,
untrained `classifier`. Sequence labels have shape `(b,)`; residue labels have
shape `(b, l)` and use `-100` outside biological positions:

```python
import torch
from transformers import AutoTokenizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

model_id = "Synthyra/ESM2-3B"
sequence_model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=2, trust_remote_code=True
).eval()
token_model = AutoModelForTokenClassification.from_pretrained(
    model_id, num_labels=3, trust_remote_code=True
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
sequences = ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"]
batch = tokenizer(sequences, padding=True, return_tensors="pt")
biological = batch["attention_mask"].bool()
for special_id in tokenizer.all_special_ids:
    biological &= batch["input_ids"].ne(special_id)

sequence_labels = torch.zeros(len(sequences), dtype=torch.long)
token_labels = torch.full_like(batch["input_ids"], -100)
token_labels[biological] = 0

with torch.inference_mode():
    sequence_output = sequence_model(**batch, labels=sequence_labels)
    token_output = token_model(**batch, labels=token_labels)
print(sequence_output.logits.shape)  # (b, 2)
print(token_output.logits.shape)     # (b, l, 3)
```

## PEFT fine-tuning

Install the direct training dependencies, then attach LoRA to the loaded checkpoint:

```bash
python -m pip install "datasets>=4.8,<5" "peft>=0.19,<0.20"
```

```python
from peft import LoraConfig, TaskType, get_peft_model

peft_model = get_peft_model(
    sequence_model,
    LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        modules_to_save=["classifier"],
    ),
)
```

This checkpoint advertises a classification head, so the separately trained
`classifier` is saved with the adapter.
All FastPLMs checkpoints follow the Transformers `PreTrainedModel` contract and
can be adapted with PEFT. The ESM2-specific shipped CLI is an example, not a
support boundary. Record the target modules, base revision, data identity, and
trainable parameter scope.

## Test-time training

TTT samples masked views of one protein and updates only injected low-rank
adapters. Base checkpoint weights remain frozen:

```python
from transformers import AutoModelForMaskedLM

ttt_model = AutoModelForMaskedLM.from_pretrained(
    "Synthyra/ESM2-3B",
    trust_remote_code=True,
)
metrics = ttt_model.ttt(
    seq="MSTNPKPQRKTKRNT",
    ttt_config={"steps": 3, "batch_size": 1, "seed": 7},
)
ttt_model.save_pretrained("adapted", safe_serialization=True)
ttt_model.ttt_reset()
print(metrics)
```

Persisted adapters retain their deterministic reset state. TTT adds latency
and memory, can worsen an output, and does not establish biological function.

## Masked language modeling and contacts

Use the masked-language-model AutoClass when logits are required:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "Synthyra/ESM2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
masked_model = AutoModelForMaskedLM.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
batch = tokenizer("MSTNPKPQRKTKRNT", return_tensors="pt")

with torch.inference_mode():
    logits = masked_model(**batch).logits
    contacts = masked_model.predict_contacts(
        batch["input_ids"],
        batch["attention_mask"],
    )

print(logits.shape, contacts.shape)
```

Contact prediction materializes attention maps and should not be enabled in a
high-throughput embedding path unless those maps are required.

Plain `AutoModel` omits the optional ESM pooler because this masked-language-
model checkpoint contains no trained pooler weights. Pass
`add_pooling_layer=True` only when intentionally initializing and training that
head.

## Notes and limitations

The pinned default SDPA BF16 path uses a checkpoint-specific numeric
calibration: relative L2 target/hard limit 0.06/0.07, relative Q99.9 0.15/0.18,
first-percentile residue cosine 0.994/0.992, and pooled cosine 0.998/0.997.
Exact state identity and the global logits-distribution contract remain
required.

## Runtime contract

- Public input: Amino-acid sequences tokenized to residue IDs
- Advertised AutoClasses: `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`, `AutoModelForMaskedLM` = `pretrained`, `AutoModelForSequenceClassification` = `base weights + untrained task head`, `AutoModelForTokenClassification` = `base weights + untrained task head`
- Attention implementations: `eager`, `sdpa`, `flex_attention`, `flash_attention_2`, `flash_attention_3`
- Precision policies: `default`
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `not_applicable`
- Artifact dependency set: `core`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Release record

- FastPLMs weights: `Synthyra/ESM2-3B`
- Runtime revision: recorded separately in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in `provenance.json`
- Official checkpoint: `facebook/esm2_t36_3B_UR50D`
- Artifact source: `fast`
- State transform: `esm2_hf_to_fastplms_v1`
- Pinned upstreams: `fair-esm`
- Release tiers: `check`, `compliance`, `feature`, `artifact`, `benchmark`
- Unresolved required file identities: `0`

`provenance.json` records exact file identities, conversion, source revisions,
legal texts, schema, and attestations. A nonzero unresolved count blocks release.

## Validation boundary

Declared tiers compare applicable configuration, tokenizer behavior, state,
and representative inference with the pinned reference. Metadata alone does
not claim a build passed, a backend is faster, or an output is biologically
valid.

## License

Checkpoint terms: MIT. The Hub model-card identifier is
`mit`. Applicable source licenses, notices, attribution,
and conversion records are distributed with the local artifact. Review them
before use.
