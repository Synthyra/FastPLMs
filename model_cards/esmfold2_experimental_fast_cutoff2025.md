---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESMFold2-Experimental-Fast-Cutoff2025

This checkpoint contains the FastPLMs `ESMFold2` implementation.

Accepted inputs are raw amino-acid sequences or typed molecular-complex
specifications; low-level forward accepts prepared feature tensors.
Supported Transformers entry points are `AutoConfig`, `AutoModel`,
`AutoModelForSequenceClassification`, `AutoModelForTokenClassification`.

## Capabilities

| Feature | Status |
| --- | --- |
| Sequence classification | Supported: base weights with an untrained task head |
| Token classification | Supported: base weights with an untrained task head |
| PEFT fine-tuning | Supported pattern: preserve the separately trained `classifier` |
| Embeddings | Special: ESMC state mixture to 256-wide residue embeddings |
| Test-time training | Unavailable for this experimental checkpoint |
| Attention variants | Supported: `eager`, `sdpa`, `flex_attention` |
| Compliance | Declared: exact release evidence is required |

A supported interface is not a pretrained downstream predictor. Classification heads start untrained. Compliance metadata does not show that a local build passed its release gate.

## Install and platform requirements

Install the direct dependencies published with this model:

```bash
python -m pip install -r \
  "https://huggingface.co/Synthyra/ESMFold2-Experimental-Fast-Cutoff2025/resolve/main/requirements.txt"
```

The FastPLMs implementation itself is embedded in the model repository.
Transformers loads it through `trust_remote_code=True`.

This model requires Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13. The artifact requirements include the structure dependencies. The release contract requires a CUDA device. The current validated target is the exact NVIDIA GH200 on Linux aarch64. Linux x86-64, CPU-only, Windows, and macOS structure runs are not release evidence. The Hub quick start needs network access for
the first download. For an air-gapped run, build the manifest-pinned local
artifact first and use the offline example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ESMFold2-Experimental-Fast-Cutoff2025"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="sdpa",
).eval()
```

For offline validation, replace `model_id` with the manifest-built
`dist/hub/ESMFold2-Experimental-Fast-Cutoff2025` path. Pass `local_files_only=True`.

## Attention and compliance

The quick start selects `sdpa` explicitly. Declared variants are `eager`, `sdpa`, `flex_attention`. An unavailable
requested backend raises. It does not silently change implementation.
`output_attentions=True` can use the documented one-call eager fallback to
materialize attention tensors. The configured backend does not change.

This family declares the `compliance` tier. Release evidence identifies the
checkpoint, backend, dtype, hardware, inputs, and reference revision.

## Downstream prediction

The sequence and token prediction AutoClasses use the checkpoint backbone and
create a new, untrained `classifier`. Sequence labels have shape `(b,)`.
Residue labels have shape `(b, l)` and use `-100` outside biological positions.
The folding trunk is skipped. The classifier uses the checkpoint's learned pLM
state mixture and projection, followed by one trainable transformer probe.

```python
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

model_id = "Synthyra/ESMFold2-Experimental-Fast-Cutoff2025"
sequence_model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=2, trust_remote_code=True
).eval()
token_model = AutoModelForTokenClassification.from_pretrained(
    model_id, num_labels=3, trust_remote_code=True
).eval()
sequences = ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"]
batch = sequence_model.prepare_classifier_inputs(sequences)
biological = batch["attention_mask"].bool()

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

## Alignment-conditioning contract

This 24-block Fast checkpoint is optimized for single-sequence inference. It
was trained without MSA conditioning. It rejects `ProteinInput.msa` and low-level
MSA-derived features. Typed multichain and multimolecule inputs remain supported
when every protein chain uses `msa=None`. Use the full ESMFold2 checkpoint for
MSA-conditioned inference. This follows the official Biohub architecture
description in [Appendix A.2.1](https://biohub.ai/papers/esm_protein.pdf).


## Protein folding

The single-protein helper returns typed structure and confidence outputs:

```python
result = model.fold_protein(
    "MSTNPKPQRKTKRNT",
    num_loops=1,
    num_sampling_steps=200,
    num_diffusion_samples=1,
    seed=7,
)
pdb_text = model.result_to_pdb(result)
cif_text = model.result_to_cif(result)
print(result.ptm, result.plddt.mean().item())
```

No target structure is required. For complexes, construct the input from the
types exposed by the loaded artifact:

```python
types = model.input_types
complex_input = types.StructurePredictionInput(
    sequences=[
        types.ProteinInput(id="A", sequence="MSTNPKPQRKTKRNT"),
        types.ProteinInput(id="B", sequence="MKTIIALSYIFCLVFA"),
        types.DNAInput(id="C", sequence="ATGC"),
        types.LigandInput(id="L", smiles="O"),
    ]
)
complex_result = model.fold(
    complex_input,
    num_loops=1,
    num_sampling_steps=200,
    seed=7,
)
print(complex_result.ptm, complex_result.plddt.mean().item())
```

The typed interface also supports RNA, modifications, covalent bonds, and
distogram conditioning. Protein MSA inputs are not supported by this Fast
checkpoint; every protein chain must use `msa=None`. The public schema recognizes
`PocketConditioning`, but the pinned official runtime discards it and hard-codes
a zero pocket feature. FastPLMs therefore rejects non-null pocket conditioning
instead of silently ignoring it. Prepared `ref_pos` values are component
reference geometries created during featurization, not target coordinates.
Predicted coordinates and confidence scores are outputs and do not establish
biochemical activity.

## Learned representation and ESMC precision

ESMFold2 applies its learned state mixture and projection as
`H: (b, l, 81, 2560) -> Z: (b, l, 256)`. Retrieve `Z` through the public
embedding API:

```python
representations = model.embed_dataset(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    batch_size=2,
    full_embeddings=True,
)
print(representations[0].tensor.shape)  # (sequence_length, 256)
```

`model.embed_dataset(..., full_embeddings=True)` returns one `(l, 256)` residue
tensor per single-chain input. It rejects complexes, ligands, MSAs,
chain-separated inputs, `cls`, and `parti` in the embedding path.

Set `esmc_precision` to `auto`, `bf16`, `fp32`, or `fp8` when loading.
`auto` always resolves to BF16. Explicit FP8 is experimental, inference-only,
and strict:

```python
model.reload_esmc(precision="fp8", device="cuda:0")
print(model.esmc_precision_status)
```

FP8 raises when the validated CUDA and Transformer Engine path is unavailable.
Canonical BF16 weights are retained, and transient quantization state is never
serialized.

The ESMC backbone uses SDPA as the recommended highest-fidelity path. Flex
Attention is supported and non-experimental but can be numerically divergent;
ESMFold2 does not advertise FlashAttention for the folding interface.

| Backend | Support | Measurement status |
| --- | --- | --- |
| `sdpa` | Recommended fidelity path | Pending release measurement |
| `eager` | Supported | Pending release measurement |
| `flex_attention` | Supported, numerically divergent | Pending release measurement |

Detailed backend measurements, release guardrails, and the GH200 package
compatibility exception are maintained in the
[attention backend guide](https://github.com/Synthyra/FastPLMs/blob/main/docs/attention_backends.md)
and
[release evidence manifest](https://github.com/Synthyra/FastPLMs/blob/main/docs/generated/capability_evidence.md).


## Verified CCD runtime asset

Structure preparation requires `ccd.pkl` from
`biohub/ESMFold2`. The manifest pins its repository, revision, size, content
identity, and MIT terms. This is a trusted-deserialization boundary. FastPLMs
accepts only the pinned snapshot link inside the repository blob directory and
rejects user-supplied asset and `cache_dir` symlinks. The loader verifies a
private temporary snapshot before deserialization. Offline execution requires
the exact cached object and never downloads a replacement.

## Test-time training

This experimental checkpoint does not expose folding TTT. Use the corresponding
standard or Fast checkpoint when you need opt-in ESMC-backbone adaptation.

## Binder-design research example

The FastPLMs binder-design workflow uses the experimental Fast Cutoff2025
checkpoint for differentiable inversion, both experimental Cutoff2025
checkpoints as critics, and ESM++ as the sequence prior:

![FastPLMs EGFR minibinder design](https://raw.githubusercontent.com/Synthyra/FastPLMs/main/docs/assets/egfr_fastplms_binder_design.png)

```bash
python examples/binder_design_fastplms.py \
  --target-name pd-l1 \
  --binder-name minibinder \
  --batch-size 4 \
  --steps 150 \
  --output-dir artifacts/binder-design
```

The workflow ranks candidates by mean iPTM across the approved critics after
the minibinder isoelectric-point filter. These are model-based prioritization
signals, not experimental evidence of affinity or specificity. See the
[complete workflow](https://github.com/Synthyra/FastPLMs/blob/main/docs/binder_design.md).

## Runtime contract

- Public input: Raw amino-acid sequences or typed molecular-complex specifications; low-level forward accepts prepared feature tensors
- Advertised AutoClasses: `AutoConfig`, `AutoModel`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`, `AutoModelForSequenceClassification` = `base weights + untrained task head`, `AutoModelForTokenClassification` = `base weights + untrained task head`
- Attention implementations: `eager`, `sdpa`, `flex_attention`
- Precision policies: `auto`, `fp32`, `bf16`, `fp8` (experimental)
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `not_applicable`
- Artifact dependency set: `core + structure`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Release record

- FastPLMs weights: `Synthyra/ESMFold2-Experimental-Fast-Cutoff2025`
- Runtime revision: recorded separately in the built artifact and published commit
- Runtime source identities: recorded in `source-record.json`
- Official checkpoint: `biohub/ESMFold2-Experimental-Fast-Cutoff2025`
- Artifact source: `fast`
- State transform: `identity`
- Pinned upstreams: `biohub-esm`, `biohub-transformers`, `protein-ttt`
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
