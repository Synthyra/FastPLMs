---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESM3_small

This checkpoint packages the FastPLMs `ESM3` implementation.

Accepted inputs are sequence, structure, and function tracks prepared through
the multimodal helpers.
Supported Transformers entry points are `AutoConfig`, `AutoModel`.

## Capabilities

| Feature | Status |
| --- | --- |
| Sequence classification | Unavailable: no advertised AutoClass |
| Token classification | Unavailable: no advertised AutoClass |
| PEFT fine-tuning | Supported pattern: attach LoRA to the pretrained model |
| Embeddings | Supported: shared ordered embedding API |
| Test-time training | Supported: low-rank masked-residue adaptation |
| Attention variants | Supported: `eager`, `sdpa`, `flex_attention` |
| Compliance | Declared: exact release evidence is required |

A supported interface is not a pretrained downstream predictor. Classification
heads start untrained, and declared compliance metadata is not a claim that an
arbitrary local build passed its release gate.

## Install and platform requirements

Install the direct dependencies published with this model:

```bash
python -m pip install -r \
  "https://huggingface.co/Synthyra/ESM3_small/resolve/main/requirements.txt"
```

The FastPLMs implementation itself is embedded in the model repository and loaded
by Transformers through `trust_remote_code=True`.

Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13 are required. The declared CPU gate covers tiny offline contracts; published checkpoint throughput and parity require the documented device tier. The Hub quick start below requires network
access on first download. For an air-gapped run, first build the manifest-pinned
local artifact and use the offline form shown in the example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ESM3_small"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="sdpa",
).eval()
```

For offline validation, replace `model_id` with the manifest-built
`dist/hub/ESM3_small` path and pass `local_files_only=True`.

## Attention and compliance

The quick start selects `sdpa` explicitly. Declared variants are `eager`, `sdpa`, `flex_attention`. An unavailable
requested backend raises instead of silently switching implementations.
`output_attentions=True` may use the documented, one-call eager fallback solely
to materialize attention tensors; the configured backend remains unchanged.

This family declares the `compliance` tier. Release evidence binds the exact
checkpoint, backend, dtype, hardware, inputs, and reference revision.

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

## PEFT fine-tuning

Install the direct training dependencies, then attach LoRA to the loaded checkpoint:

```bash
python -m pip install "datasets>=4.8,<5" "peft>=0.19,<0.20"
```

```python
from peft import LoraConfig, get_peft_model

peft_model = get_peft_model(
    model,
    LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
    ),
)
```

This checkpoint has no advertised classifier. Supply the task-specific
objective and preserve any new head through `modules_to_save`.
All FastPLMs checkpoints follow the Transformers `PreTrainedModel` contract and
can be adapted with PEFT. The ESM2-specific shipped CLI is an example, not a
support boundary. Record the target modules, base revision, data identity, and
trainable parameter scope.

## Test-time training

TTT samples masked views of one protein and updates only injected low-rank
adapters. Base checkpoint weights remain frozen:

```python
from transformers import AutoModel

ttt_model = AutoModel.from_pretrained(
    "Synthyra/ESM3_small",
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

## Sequence inference and masked-sequence generation

ESM3 owns its sequence preparation. This example exercises the sequence track;
the public input contract also supports structure and function tracks through
the multimodal helpers:

```python
import torch

batch = model.tokenize_sequences(
    ["MKTAYIAKQ", "GGGG"],
    device=model.device,
)
with torch.inference_mode():
    output = model(**batch)

print(output.last_hidden_state.shape)
print(output.logits.shape)
print(output.structure_logits.shape)
print(output.function_logits.shape)
```

When `return_dict=False`, ESM3 follows the standard base-model tuple prefix:
`last_hidden_state`, then requested `hidden_states` and `attentions`. Multimodal
logits and extensions follow that prefix. Prefer named fields for individual
tracks.

Generate masked sequence positions with an explicit seed:

```python
from fastplms.models.esm3.modeling_esm3 import FastESM3GenerationConfig

config = FastESM3GenerationConfig(
    num_steps=8,
    temperature=1.0,
    seed=7,
)
generated = model.generate("MK____A", config)
print(generated)
```

Underscores mark positions to generate. Model outputs are predictions over
tracks, not experimental measurements of structure or function.

## Runtime contract

- Public input: Sequence, structure, and function tracks prepared through the multimodal helpers
- Advertised AutoClasses: `AutoConfig`, `AutoModel`
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`
- Attention implementations: `eager`, `sdpa`, `flex_attention`
- Precision policies: `default`
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `not_applicable`
- Artifact dependency set: `core`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Release record

- FastPLMs weights: `Synthyra/ESM3_small`
- Runtime revision: recorded separately in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in `provenance.json`
- Official checkpoint: `biohub/esm3-sm-open-v1`
- Artifact source: `fast`
- State transform: `esm3_to_fastplms_v1`
- Pinned upstreams: `biohub-esm`, `biohub-transformers`
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
