---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# ESMFold2-600

## Quick start

Load the published model, fold two protein chains together, and write an mmCIF
file. This example uses 15 diffusion steps, matching the experimental config.

```python
import torch

from pathlib import Path
from transformers import AutoModel


model = AutoModel.from_pretrained(
    "Synthyra/ESMFold2-600",
    trust_remote_code=True,
    dtype=torch.float32,
    device_map="cuda",
    esmc_precision="bf16",
    attn_implementation="sdpa",
).eval()
model.set_chunk_size(32)

types = model.input_types
complex_input = types.StructurePredictionInput(
    sequences=[
        types.ProteinInput(id="A", sequence="MSTNPKPQRKTKRNT"),
        types.ProteinInput(id="B", sequence="MKTIIALSYIFCLVFA"),
    ]
)
with torch.inference_mode():
    result = model.fold(
        complex_input,
        num_loops=3,
        num_sampling_steps=15,
        num_diffusion_samples=1,
        seed=17,
        verbose=True,
    )
Path("complex.cif").write_text(model.result_to_cif(result), encoding="utf-8")
```

Set `verbose=False` to silence the folding progress display. The confidence fields are unavailable because this experimental variant has a
disabled confidence head.

## Model overview

`Synthyra/ESMFold2-600` packages the
`biohub/ESMFold2-Experimental-Fast-base600M-step1500k` checkpoint with the
FastPLMs runtime for Hugging Face Transformers. It accepts raw amino-acid
sequences or typed molecular-complex specifications; low-level forward accepts
prepared feature tensors.

The repository uses the standard Transformers loading interface with
`trust_remote_code=True`. See Technical details for each registered class and
whether its weights come from the checkpoint.

The sequence- and token-classification classes reuse the pretrained backbone,
but their task heads are newly initialized. Fine-tune those heads before
interpreting their logits as predictions.

## Install and platform requirements

Install the direct dependencies published with this model:

```bash
python -m pip install -r \
  "https://huggingface.co/Synthyra/ESMFold2-600/resolve/main/requirements.txt"
```

The FastPLMs implementation itself is embedded in the model repository.
Transformers loads it through `trust_remote_code=True`.

This model requires Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13.

The artifact requirements include the structure dependencies.

Validation runs in Docker on any compatible CUDA device. Record the container,
hardware, precision, and inputs; no GPU product or workstation is required.

The Hub quick start needs network access for the first download. For an
air-gapped run, build the manifest-pinned local artifact first and use the
offline example.

## Attention backends

The quick start uses `sdpa`.

Available backends are `eager`, `sdpa`, `flex_attention`. Requesting an
unavailable backend raises instead of silently changing implementation.

`output_attentions=True` can use the documented one-call eager fallback to
materialize attention tensors. The configured backend does not change.

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


model_id = "Synthyra/ESMFold2-600"
sequence_model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=2, trust_remote_code=True
).eval()
token_model = AutoModelForTokenClassification.from_pretrained(
    model_id, num_labels=3, trust_remote_code=True
).eval()
sequences = ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"]
batch = sequence_model.prepare_classifier_inputs(sequences)
biological = batch["attention_mask"].bool()  # (b, l)

sequence_labels = torch.zeros(len(sequences), dtype=torch.long)  # (b,)
token_labels = torch.full_like(batch["input_ids"], -100)  # (b, l)
token_labels[biological] = 0  # selected biological positions; labels stay (b, l)

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

## Protein folding

This experimental Fast checkpoint has 24 folding blocks and uses the frozen
`Synthyra/ESMplusplus_large` backbone. The config-declared step-1500000 backbone and the
pinned ESM++ weights are tensor-exact in BF16 after layout conversion.

```python
import torch


model = model.cuda().eval()
with torch.inference_mode():
    output = model.infer_protein(
        "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
        seed=17,
        num_diffusion_samples=1,
    )
print(output.sample_atom_coords.shape)
```

Folding parameters remain FP32 with CUDA BF16 autocast. The backbone uses
BF16; FP8 requests fail. The 15-step sampler and three folding loops remain
the checkpoint defaults. Protein inputs require `msa=None`.
This checkpoint was trained without MSA conditioning. It rejects
`ProteinInput.msa` and MSA-derived features. Typed multichain and multimolecule
inputs remain supported without MSA conditioning.

The confidence head is disabled: pLDDT, pTM, iPTM, and PAE are unavailable.
The 300 and 600 suffixes describe backbone scale, not total model parameters.

## Folding speed settings

Two runtime settings trade memory or exactness for speed on long proteins. They
need no extra package and no compilation, and neither is stored in the
configuration.

```python
model.set_chunk_size(None)            # unchunked pair updates
model.set_atom_attention("windowed")  # the official flash-attn atom window, through PyTorch
```

`set_chunk_size(None)` removes the row chunking of the pair-update blocks, which
costs most of a long fold's time on a data-center GPU and saves little peak
memory; pass a chunk such as 512 when the unchunked fold does not fit.
`set_atom_attention("windowed")` restricts each atom to 64 real neighbors on
each side, as the official model does when flash-attn is installed. It needs
CUDA and changes numerical output, within sampling spread on the measured
panel. The
[ESMFold2 guide](https://github.com/Synthyra/FastPLMs/blob/main/docs/esmfold2.md#measured-folding-cost)
records the conditions, the dense-versus-windowed comparison, and the figure.

| Residues | FastPLMs defaults (s) | Optimized (s) |
| ---: | ---: | ---: |
| 256 | 1.1 | 1.0 |
| 1,024 | 35 | 10 |
| 2,048 | not measured | 40 |

Measured on one NVIDIA H100 80GB HBM3 with PyTorch 2.13.0+cu130: one
fixed pseudo-random protein per length, 3 trunk loops,
50 requested sampling steps under the official noise cap,
1 diffusion sample, BF16 autocast over FP32 folding
parameters, median of end-to-end folds. "Defaults" changes no setting.

## Learned representation and ESMC precision

The learned projection maps `H: (b, l, 37, 1152) -> Z: (b, l, 256)`.
`embed_dataset` returns one `(l, 256)` residue representation per sequence.
The experimental architecture does not expose folding TTT.

## Separately trained confidence head

This checkpoint ships with its confidence head disabled. Separately trained
confidence weights remain unpublished and are not part of this artifact.

The archived correlations, bootstrap intervals, and acceptance gates require
recomputation after correcting tied ranks in Spearman correlation. Raw test
predictions were not recovered from the closed GH200 workstation or W&B, so
the historical numbers are withheld here pending raw prediction recovery.
They do not establish corrected quality or acceptance results.

The [W&B training history audit](https://wandb.ai/lhallee/fastplms-confidence/runs/01c7e12b6d03) inspected all
780 update rows and found zero skipped training targets.
The skipped-target gradient bug therefore did not affect this recorded run's
training weights. This audit does not validate the archived correlations.

The [confidence training guide](https://github.com/Synthyra/FastPLMs/blob/main/docs/confidence_training.md)
preserves the historical results and their review status.

## Notes and limitations

Experimental Fast checkpoint with a frozen 600M ESM++ backbone, tensor-exact in
BF16 with the pinned step-1500000 source, 24 folding blocks, no MSA
conditioning, and no confidence head. BF16 execution uses FP32 folding
parameters with CUDA autocast. FP8 is unsupported. Configuration, weight
identities, and artifact reload are verified; this model is not
inference-validated.

## Technical details

- Inputs: Raw amino-acid sequences or typed molecular-complex specifications; low-level forward accepts prepared feature tensors
- Transformers classes: `AutoConfig`, `AutoModel`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`
- Checkpoint weights: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`, `AutoModelForSequenceClassification` = `base weights + untrained task head`, `AutoModelForTokenClassification` = `base weights + untrained task head`
- Attention backends: `eager`, `sdpa`, `flex_attention`
- Precision: `auto`, `fp32`, `bf16`
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `not_applicable`
- Dependencies: `core + structure`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Validation and sources

FastPLMs pins the checkpoint, upstream source revisions, state transformation,
and required files in `models.toml`. Built artifacts record exact source
identities and conversion details in `source-record.json`.

- FastPLMs checkpoint: `Synthyra/ESMFold2-600`
- Runtime revision: recorded separately in the built artifact and published commit
- Runtime source identities: recorded in `source-record.json`
- Official checkpoint: `biohub/ESMFold2-Experimental-Fast-base600M-step1500k`
- Artifact source: `fast`
- State transform: `identity`
- Pinned upstreams: `biohub-esm`, `biohub-transformers`, `protein-ttt`
- Release tiers: `check`, `compliance`, `structure`, `feature`, `artifact`, `benchmark`
- Unresolved required file identities: `0`

ESMFold2-300 passed a Docker BF16 reference comparison on one compact Protein G
sequence. ESMFold2-600 is not inference-validated. Both have configuration,
weight identity, and artifact loading checks. This is bounded checkpoint
evidence, not a full structure benchmark result.

Declared tiers compare configuration, tokenizer behavior, state, and
representative inference with the pinned reference. A nonzero unresolved count
blocks release. Metadata alone does not show that a build passed, that a backend
is faster, or that an output is biologically valid.

## License

Checkpoint terms: MIT. The Hub model-card identifier is
`mit`. The local artifact contains applicable source
licenses, notices, attribution, and conversion records. Review them before use.
