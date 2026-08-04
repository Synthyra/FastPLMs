---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/Boltz2

This checkpoint contains the FastPLMs `Boltz2` implementation.

Accepted inputs are raw amino-acid sequences through the convenience API, or
prepared model features.
Supported Transformers entry points are `AutoConfig`, `AutoModel`.

## Capabilities

| Feature | Status |
| --- | --- |
| Sequence classification | Unavailable: no advertised AutoClass |
| Token classification | Unavailable: no advertised AutoClass |
| PEFT fine-tuning | Supported pattern: attach LoRA to the pretrained model |
| Embeddings | Unavailable for this structure-only checkpoint |
| Test-time training | Unavailable for this inference-only checkpoint |
| Attention variants | Supported: `eager` |
| Compliance | Unavailable: this provisional family has no compliance tier |

A supported interface is not a pretrained downstream predictor. Classification heads start untrained. Compliance metadata does not show that a local build passed its release gate.

## Install and platform requirements

Install the direct dependencies published with this model:

```bash
python -m pip install -r \
  "https://huggingface.co/Synthyra/Boltz2/resolve/main/requirements.txt"
```

The FastPLMs implementation itself is embedded in the model repository.
Transformers loads it through `trust_remote_code=True`.

This model requires Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13. The artifact requirements include the structure dependencies. The release contract requires a CUDA device. The current validated target is the exact NVIDIA GH200 on Linux aarch64. Linux x86-64, CPU-only, Windows, and macOS structure runs are not release evidence. The Hub quick start needs network access for
the first download. For an air-gapped run, build the manifest-pinned local
artifact first and use the offline example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/Boltz2"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="eager",
).eval()
```

For offline validation, replace `model_id` with the manifest-built
`dist/hub/Boltz2` path. Pass `local_files_only=True`.

## Attention and compliance

The quick start selects `eager` explicitly. Declared variants are `eager`. An unavailable requested backend raises. It does
not silently change implementation.
`output_attentions=True` can use the documented one-call eager fallback to
materialize attention tensors. The configured backend does not change.

This family does not declare the `compliance` tier. Boltz2 remains provisional.
Its structure checks are not parity claims.

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

The high-level helper prepares a protein-only input, runs the declared Boltz2
inference core, and returns coordinates and confidence fields:

```python
import torch

model = model.cuda().eval()
output = model.predict_structure(
    amino_acid_sequence="MSTNPKPQRKTKRNTNRRPQDVKFPGG",
    recycling_steps=3,
    num_sampling_steps=50,
    diffusion_samples=1,
    seed=7,
)
model.save_as_cif(output, "prediction.cif")

print(output.sample_atom_coords.shape)
print(output.plddt, output.ptm, output.iptm)
```

The validation boundary below describes the supported inference subset and its
provisional status. The helper saves and restores Python, NumPy, CPU Torch, and
CUDA RNG state. Parameters and prepared features stay FP32. Supported CUDA
inference runs in BF16 autocast.

## Notes and limitations

Boltz2 is provisional in FastPLMs 1.0. Exact configuration, the declared
inference-core state, feature preparation, and seeded execution remain tested,
but native-environment BF16 end-to-end inference currently exceeds the fixed
numerical-equivalence limits. FastPLMs therefore does not claim official
inference equivalence for this checkpoint yet. Work on that numerical gap
continues independently of the ESM++ and ESMFold2 release gates.

## Runtime contract

- Public input: Raw amino-acid sequences through the convenience API, or prepared model features
- Advertised AutoClasses: `AutoConfig`, `AutoModel`
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`
- Attention implementations: `eager`
- Precision policies: `default`
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `not_applicable`
- Artifact dependency set: `core + structure`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Release record

- FastPLMs weights: `Synthyra/Boltz2`
- Runtime revision: recorded in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in the source record
- Official checkpoint: `boltz-community/boltz-2`
- Artifact source: `fast`
- State transform: `boltz2_inference_core_v1`
- Pinned upstreams: `boltz`
- Release tiers: `structure`, `artifact`, `benchmark`
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
