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

## Install and platform requirements

Install the current FastPLMs package:

```bash
python -m pip install \
  "fastplms[structure] @ git+https://github.com/Synthyra/FastPLMs.git"
```

Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13 are required. Structure inference requires the `structure` extra and a CUDA device for the published execution contract. The current validated release target is the exact NVIDIA GH200 on Linux aarch64; Linux x86-64, CPU-only, Windows, and macOS structure runs are not current release evidence. The Hub quick start below requires network
access on first download. For an air-gapped run, first build the manifest-pinned
local artifact and use the offline form shown in the example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/FastESMFold"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/FastESMFold` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default. Supported explicit
choices are `eager`, `sdpa`, `flex_attention`.
Pass the selected name through `attn_implementation`.
When an optimized backend cannot return full attention tensors,
`output_attentions=True` emits one explicit runtime warning and uses a correctly
masked eager implementation for that call only. The warning identifies the
configured backend, effective backend, and reason. Configuration and later
calls are unchanged.
For BF16 execution, this family uses FP32 parameters with CUDA BF16 autocast.

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
- Optional dependency group: `structure`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Release record

- FastPLMs weights: `Synthyra/FastESMFold`
- Runtime revision: recorded separately in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in `provenance.json`
- Generator/schema version and complete/runtime-only attestations: recorded in `provenance.json`
- Official checkpoint: `facebook/esmfold_v1`
- Artifact source: `fast`
- State transform: `esmfold_meta_to_fastplms_v1`
- BF16 execution: `fp32_parameters_autocast`
- Pinned upstreams: `fair-esm`, `openfold`
- Reference container: `reference-esmfold`
- Release tiers: `check`, `compliance`, `structure`, `feature`, `artifact`, `benchmark`
- Unresolved required file identities: `0`

The local artifact records exact file identities, conversion details, source
revisions, and legal texts in `provenance.json`. A nonzero unresolved count is a
release blocker.

## Validation boundary

For tiers declared by the manifest, the release contract compares applicable
semantic configuration, tokenizer behavior, state keys, shapes, dtypes,
values, aliases, and representative inference with the pinned official
implementation. This metadata does not by itself claim that a particular build
passed, that one backend is faster, or that an output has biological or
therapeutic validity.

## License

Checkpoint terms: MIT. The Hub model-card identifier is
`mit`. Applicable source licenses, notices, attribution,
and conversion records are distributed with the local artifact. Review them
before use.
