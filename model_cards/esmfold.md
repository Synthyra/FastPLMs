---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/FastESMFold

This checkpoint contains the FastPLMs `ESMFold` implementation.

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

A supported interface is not a pretrained downstream predictor. Classification heads start untrained. Compliance metadata does not show that a local build passed its release gate.

## Install and platform requirements

Install the direct dependencies published with this model:

```bash
python -m pip install -r \
  "https://huggingface.co/Synthyra/FastESMFold/resolve/main/requirements.txt"
```

The FastPLMs implementation itself is embedded in the model repository.
Transformers loads it through `trust_remote_code=True`.

This model requires Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13. The artifact requirements include the structure dependencies. The release contract requires a CUDA device. The current validated target is the exact NVIDIA GH200 on Linux aarch64. Linux x86-64, CPU-only, Windows, and macOS structure runs are not release evidence. The Hub quick start needs network access for
the first download. For an air-gapped run, build the manifest-pinned local
artifact first and use the offline example.

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
`dist/hub/FastESMFold` path. Pass `local_files_only=True`.

## Attention and compliance

The quick start selects `sdpa` explicitly. Declared variants are `eager`, `sdpa`, `flex_attention`. An unavailable
requested backend raises. It does not silently change implementation.
`output_attentions=True` can use the documented one-call eager fallback to
materialize attention tensors. The configured backend does not change.

This family declares the `compliance` tier. Release evidence identifies the
checkpoint, backend, dtype, hardware, inputs, and reference revision.

## PEFT fine-tuning

Install the training dependencies. Then attach LoRA to the loaded checkpoint:

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

This checkpoint has no advertised classifier. Supply the task objective and
preserve any new head through `modules_to_save`.
All FastPLMs checkpoints follow the Transformers `PreTrainedModel` contract and
can use PEFT. The ESM2-specific shipped CLI is an example, not a
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
has no trained masked-language-model head for this objective. `ttt()` and TTT
folding requests raise.

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
- Runtime revision: recorded in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in the source record
- Official checkpoint: `facebook/esmfold_v1`
- Artifact source: `fast`
- State transform: `esmfold_meta_to_fastplms_v1`
- Pinned upstreams: `fair-esm`, `openfold`
- Release tiers: `check`, `compliance`, `structure`, `feature`, `artifact`, `benchmark`
- Unresolved required file identities: `0`

The source record records exact file identities, conversion, source revisions,
legal texts, schema, and attestations. A nonzero unresolved count blocks a release.

## Validation boundary

Declared tiers compare configuration, tokenizer behavior, state, and
representative inference with the pinned reference. Metadata does not show that
a build passed, that a backend is faster, or that an output is biologically valid.

## License

Checkpoint terms: MIT. The Hub model-card identifier is
`mit`. The local artifact contains applicable source
licenses, notices, attribution, and conversion records. Review them before use.
