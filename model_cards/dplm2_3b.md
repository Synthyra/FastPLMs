---
library_name: transformers
license: "apache-2.0"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/DPLM2-3B

This checkpoint contains the FastPLMs `DPLM2` implementation.

Accepted inputs are tokenized amino-acid and structure tracks with explicit
modality boundaries.
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
| Attention variants | Supported: `sdpa` |
| Compliance | Declared: exact release evidence is required |

A supported interface is not a pretrained downstream predictor. Classification heads start untrained. Compliance metadata does not show that a local build passed its release gate.

## Install and platform requirements

Install the direct dependencies published with this model:

```bash
python -m pip install -r \
  "https://huggingface.co/Synthyra/DPLM2-3B/resolve/main/requirements.txt"
```

The FastPLMs implementation itself is embedded in the model repository.
Transformers loads it through `trust_remote_code=True`.

This model requires Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13. The CPU gate covers small offline tests. Published checkpoint throughput and parity require the documented device tier. The Hub quick start needs network access for
the first download. For an air-gapped run, build the manifest-pinned local
artifact first and use the offline example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/DPLM2-3B"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="sdpa",
).eval()
```

For offline validation, replace `model_id` with the manifest-built
`dist/hub/DPLM2-3B` path. Pass `local_files_only=True`.

## Attention and compliance

The quick start selects `sdpa` explicitly. Declared variants are `sdpa`. An unavailable requested backend raises. It does
not silently change implementation.
`output_attentions=True` can use the documented one-call eager fallback to
materialize attention tensors. The configured backend does not change.

This family declares the `compliance` tier. Release evidence identifies the
checkpoint, backend, dtype, hardware, inputs, and reference revision.

## Dataset embeddings

The shared embedding mixin keeps input order and biological-position masking.
It accepts sequences, identified records, mappings, or a FASTA path:

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
bounded-memory storage. Resume checks input order, model state, tokenizer
policy, backend, dtype, and pooling configuration before it appends data.

## Downstream classification

Both downstream AutoClasses use the checkpoint backbone and create a new,
untrained `classifier`. Sequence labels have shape `(b,)`. Residue labels have
shape `(b, l)` and use `-100` outside biological positions:

```python
import torch
from transformers import AutoTokenizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

model_id = "Synthyra/DPLM2-3B"
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

Install the training dependencies. Then attach LoRA to the loaded checkpoint:

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

This checkpoint advertises a classification head. Save the separately trained
`classifier` with the adapter.
All FastPLMs checkpoints follow the Transformers `PreTrainedModel` contract and
can use PEFT. The ESM2-specific shipped CLI is an example, not a
support boundary. Record the target modules, base revision, data identity, and
trainable parameter scope.

## Test-time training

TTT samples masked views of one protein and updates only injected low-rank
adapters. Base checkpoint weights stay frozen:

```python
from transformers import AutoModelForMaskedLM

ttt_model = AutoModelForMaskedLM.from_pretrained(
    "Synthyra/DPLM2-3B",
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

Saved adapters retain their deterministic reset state. TTT adds latency and
memory, can worsen an output, and does not show biological function.

## Amino-acid and structure co-generation

DPLM2 uses separate structure and amino-acid tracks. Each track has its own
boundary and mask tokens:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "Synthyra/DPLM2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
generator = AutoModelForMaskedLM.from_pretrained(
    model_id,
    trust_remote_code=True,
).cuda().eval()
vocab = tokenizer.get_vocab()
l = 64
structure = [
    vocab["<cls_struct>"],
    *([vocab["<mask_struct>"]] * l),
    vocab["<eos_struct>"],
]
amino_acids = [
    vocab["<cls_aa>"],
    *([vocab["<mask_aa>"]] * l),
    vocab["<eos_aa>"],
]
input_ids = torch.tensor([structure + amino_acids], device="cuda")

with torch.inference_mode():
    generated = generator.generate(input_ids, max_iter=100)["output_tokens"]
print(generated.shape)
```

Generic `cls_token`, `eos_token`, `mask_token`, and `unk_token` aliases are not
set. Code that creates multimodal tensors must select the amino-acid or structure
token explicitly. Raw amino-acid sequences remain supported by
`model.embed_dataset(...)`.

Plain `AutoModel` omits the optional ESM pooler because this co-generation
checkpoint has no trained pooler weights. Pass `add_pooling_layer=True` only
when you intend to initialize and train that head.

The checkpoint weights use Apache-2.0. The ByteDance [LICENSE](https://github.com/bytedance/dplm/blob/main/LICENSE)
and [README](https://github.com/bytedance/dplm/blob/main/README.md#overview) document the license for pretrained DPLM1 and DPLM2
weights. Complete publication requires all artifact, legal, parity, and
atomic-publication checks.

## Notes and limitations

The pinned official DPLM2-3B sampler fails before generation, so live
generation equivalence cannot be established for this checkpoint. State,
tokenizer, and inference parity remain required.

## Runtime contract

- Public input: Tokenized amino-acid and structure tracks with explicit modality boundaries
- Advertised AutoClasses: `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`, `AutoModelForMaskedLM` = `pretrained`, `AutoModelForSequenceClassification` = `base weights + untrained task head`, `AutoModelForTokenClassification` = `base weights + untrained task head`
- Attention implementations: `sdpa`
- Precision policies: `default`
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `official_unavailable`
- Artifact dependency set: `core`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Release record

- FastPLMs weights: `Synthyra/DPLM2-3B`
- Runtime revision: recorded in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in the source record
- Canonical transformed state SHA-256: `8c46ec09115dbe6cbfb91d94ab5e906369d57e27fe620a7741c6f8cb1b6ca890`
- Conversion equality attestation: recorded in the source record
- Official checkpoint: `airkingbd/dplm2_3b`
- Artifact source: `official`
- State transform: `dplm2_to_fastplms_v1`
- Tokenizer class: `fastplms.models.dplm2.tokenization_dplm2.DPLM2Tokenizer`
- Pinned upstreams: `dplm`
- Release tiers: `check`, `compliance`, `feature`, `artifact`, `benchmark`
- Unresolved required file identities: `0`

The source record records exact file identities, conversion, source revisions,
legal texts, schema, and attestations. A nonzero unresolved count blocks a release.

## Validation boundary

Declared tiers compare configuration, tokenizer behavior, state, and
representative inference with the pinned reference. Metadata does not show that
a build passed, that a backend is faster, or that an output is biologically valid.

## License

Checkpoint terms: Apache-2.0. The Hub model-card identifier is
`apache-2.0`. The local artifact contains applicable source
licenses, notices, attribution, and conversion records. Review them before use.
