---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/FastESMFold

This checkpoint packages the FastPLMs `ESMFold` implementation.

Accepted inputs are raw amino-acid sequences through folding helpers, or
prepared residue tensors.
Supported Transformers entry points are `AutoConfig`, `AutoModel`.

## Capabilities

| Feature | Status |
| --- | --- |
| Sequence classification | Unavailable: no advertised AutoClass |
| Token classification | Unavailable: no advertised AutoClass |
| PEFT fine-tuning | Supported pattern: attach LoRA to the pretrained model |
| Embeddings | Unavailable for this structure-only checkpoint |
| Test-time training | Unavailable: the checkpoint has no trained MLM head |
| Attention variants | Supported: `eager`, `sdpa`, `flex_attention` |
| Compliance | Declared: exact release evidence is required |

A supported interface is not a pretrained downstream predictor. Classification
heads start untrained, and declared compliance metadata is not a claim that an
arbitrary local build passed its release gate.

## Install and platform requirements

Install the direct dependencies published with this model:

```bash
python -m pip install -r \
  "https://huggingface.co/Synthyra/FastESMFold/resolve/main/requirements.txt"
```

The FastPLMs implementation itself is embedded in the model repository and loaded
by Transformers through `trust_remote_code=True`.

Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13 are required. The artifact requirements include the direct structure dependencies. The published execution contract requires a CUDA device. The current validated release target is the exact NVIDIA GH200 on Linux aarch64; Linux x86-64, CPU-only, Windows, and macOS structure runs are not current release evidence. The Hub quick start below requires network
access on first download. For an air-gapped run, first build the manifest-pinned
local artifact and use the offline form shown in the example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/FastESMFold"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="sdpa",
).eval()
```

For offline validation, replace `model_id` with the manifest-built
`dist/hub/FastESMFold` path and pass `local_files_only=True`.

## Attention and compliance

The quick start selects `sdpa` explicitly. Declared variants are `eager`, `sdpa`, `flex_attention`. An unavailable
requested backend raises instead of silently switching implementations.
`output_attentions=True` may use the documented, one-call eager fallback solely
to materialize attention tensors; the configured backend remains unchanged.

This family declares the `compliance` tier. Release evidence binds the exact
checkpoint, backend, dtype, hardware, inputs, and reference revision.

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

## Protein structure prediction

ESMFold accepts a raw sequence and returns structure tensors and confidence:

```python
import torch

model = model.cuda().eval()
with torch.inference_mode():
    output = model.infer(
        "MKTLLILAVVAAALA",
        num_recycles=4,
    )

print(output["mean_plddt"])

summary = model.fold_protein(
    "MKTLLILAVVAAALA",
    return_pdb_string=True,
)
with open("prediction.pdb", "w", encoding="utf-8") as handle:
    handle.write(summary["pdb_string"])
print(summary["plddt"], summary["ptm"])
```

FastPLMs does not expose ProteinTTT for ESMFold. The pinned folding checkpoint
does not contain a trained masked-language-model head for that objective, so
`ttt()` and TTT folding requests raise explicitly.

## Runtime contract

- Public input: Raw amino-acid sequences through folding helpers, or prepared residue tensors
- Advertised AutoClasses: `AutoConfig`, `AutoModel`
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`
- Attention implementations: `eager`, `sdpa`, `flex_attention`
- Precision policies: `default`
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `not_applicable`
- Artifact dependency set: `core + structure`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Release record

- FastPLMs weights: `Synthyra/FastESMFold`
- Runtime revision: recorded separately in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in `provenance.json`
- Official checkpoint: `facebook/esmfold_v1`
- Artifact source: `fast`
- State transform: `esmfold_meta_to_fastplms_v1`
- Pinned upstreams: `fair-esm`, `openfold`
- Release tiers: `check`, `compliance`, `structure`, `feature`, `artifact`, `benchmark`
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
