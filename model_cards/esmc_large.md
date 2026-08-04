---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESMplusplus_large

This checkpoint contains the FastPLMs `ESMC` implementation.

Accepted inputs are amino-acid sequences tokenized to residue IDs.
Supported Transformers entry points are `AutoConfig`, `AutoModel`,
`AutoModelForMaskedLM`.

## Capabilities

| Feature | Status |
| --- | --- |
| Sequence classification | Unavailable: no advertised AutoClass |
| Token classification | Unavailable: no advertised AutoClass |
| PEFT fine-tuning | Supported pattern: attach LoRA to the pretrained model |
| Embeddings | Supported: shared ordered embedding API |
| Test-time training | Supported: low-rank masked-residue adaptation |
| Attention variants | Special: SDPA fidelity path; alternate backends have explicit bands |
| Compliance | Declared: exact release evidence is required |

A supported interface is not a pretrained downstream predictor. Classification heads start untrained. Compliance metadata does not show that a local build passed its release gate.

## Install and platform requirements

Install the direct dependencies published with this model:

```bash
python -m pip install -r \
  "https://huggingface.co/Synthyra/ESMplusplus_large/resolve/main/requirements.txt"
```

The FastPLMs implementation itself is embedded in the model repository.
Transformers loads it through `trust_remote_code=True`.

This model requires Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13. The artifact requirements include the FlashAttention loader dependency. FlashAttention also requires compatible CUDA hardware and BF16 execution. The Hub quick start needs network access for
the first download. For an air-gapped run, build the manifest-pinned local
artifact first and use the offline example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ESMplusplus_large"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="sdpa",
).eval()
```

For offline validation, replace `model_id` with the manifest-built
`dist/hub/ESMplusplus_large` path. Pass `local_files_only=True`.

## Attention and compliance

The quick start selects `sdpa` explicitly. Declared variants are `eager`, `sdpa`, `flex_attention`, `flash_attention_2`,
`flash_attention_3`. An unavailable requested backend raises. It does not
silently change implementation.
`output_attentions=True` can use the documented one-call eager fallback to
materialize attention tensors. The configured backend does not change.

This family declares the `compliance` tier. Release evidence identifies the
checkpoint, backend, dtype, hardware, inputs, and reference revision.

## Tokenization and forward inference

Load the tokenizer from the same artifact as the model. The attention mask
shows padding explicitly:

```python
import torch
from transformers import AutoTokenizer

model_id = "Synthyra/ESMplusplus_large"
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

## Test-time training

TTT samples masked views of one protein and updates only injected low-rank
adapters. Base checkpoint weights stay frozen:

```python
from transformers import AutoModelForMaskedLM

ttt_model = AutoModelForMaskedLM.from_pretrained(
    "Synthyra/ESMplusplus_large",
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

## ESMC behavior

This artifact provides the Biohub ESMC sequence encoder and masked-language-
model head through Transformers. ESMFold2 also uses this language-model family.
SDPA is the default and gives the highest numerical fidelity. Flex Attention and
FlashAttention 3 are supported non-experimental backends. Their BF16 arithmetic
can differ numerically from SDPA. These differences give diagnostic warnings,
not strict-parity failures. Dispatch, masks, finite outputs, shapes, and large
biological disagreements remain hard gates.

The current GH200/aarch64 release environment validates eager, SDPA, and Flex.
Flash requests raise because compatible locked kernels are unavailable on this
platform.

When `sequence_id` is supplied, it controls ESMC attention groups and padding.
`attention_mask` is ignored. Values greater than or equal to zero are valid
sequence-group IDs. `-1` marks padding. Omit `sequence_id` to use
`attention_mask` for padding.

### Hidden-state sparse autoencoders

ESM++ supports hidden-state SAEs from the official
[Biohub ESMC SAE collection](https://huggingface.co/collections/biohub/esmc-saes-for-hidden-states-all-layers).
Select an SAE for this ESMC scale. Load only required layers. Then attach them
to the model:

```python
import torch
from transformers import AutoModel

sae = AutoModel.from_pretrained("biohub/ESMC-600M-sae-layer27-k64-codebook65536", device=model.device)
sae.initialize_layers([27])
model.add_sae_models([sae.layers["27"]])

with torch.inference_mode():
    output = model(**batch, normalize_sae=True)

features = output.sae_outputs["layer27"]
print(features.shape, features.layout)  # (valid_token_count, codebook_dim), sparse COO
```

SAEs run after you attach them. Use `compute_sae=False` to skip SAE work.
Outputs are detached sparse tensors with keys such as `layer{N}`. They omit
padding. The model uses `sequence_id`, then `attention_mask`, for padding.
`normalize_sae=True` uses Biohub `(features / max) * idf` normalization. SAE
computation requires `input_ids`. It rejects mask tokens because Biohub trained
the SAEs with unmasked sequences. This interface supports hidden-state SAEs
only, not MLP-output SAEs. FastPLMs does not copy SAE weights or add SAE
checkpoints to its model manifest.

### Experimental FP8 inference

The default uses checkpoint BF16 behavior. FP8 is an explicit experimental
inference option for every ESM++ scale:

```python
import torch
from transformers import AutoModel

fp8_model = AutoModel.from_pretrained(
    "Synthyra/ESMplusplus_large",
    trust_remote_code=True,
    dtype=torch.bfloat16,
).cuda().eval()
fp8_model.enable_fp8()
print(fp8_model.esmc_precision_status)

with torch.inference_mode():
    fp8_output = fp8_model(**{name: value.cuda() for name, value in batch.items()})
```

FP8 forward calls require `torch.inference_mode()`. The model pads the sequence
dimension to a multiple of 16. Transformer Engine converts supported linear
layers. The call fails if the dependency, compatible CUDA hardware, or complete
conversion set is unavailable. It does not silently use BF16. FP8 does not
claim numerical parity.

| Backend | Support | Measurement status |
| --- | --- | --- |
| `sdpa` | Recommended fidelity path | Pending release measurement |
| `eager` | Supported | Pending release measurement |
| `flash_attention_2` | Supported | Unavailable on current GH200/aarch64 lock |
| `flex_attention` | Supported, numerically divergent | Pending release measurement |
| `flash_attention_3` | Supported, numerically divergent | Unavailable on current GH200/aarch64 lock |

Detailed backend measurements, release guardrails, and the GH200 package
compatibility exception are maintained in the
[attention backend guide](https://github.com/Synthyra/FastPLMs/blob/main/docs/attention_backends.md)
and
[release evidence manifest](https://github.com/Synthyra/FastPLMs/blob/main/docs/generated/capability_evidence.md).


## Runtime contract

- Public input: Amino-acid sequences tokenized to residue IDs
- Advertised AutoClasses: `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`, `AutoModelForMaskedLM` = `pretrained`
- Attention implementations: `eager`, `sdpa`, `flex_attention`, `flash_attention_2`, `flash_attention_3`
- Precision policies: `default`, `fp8` (experimental)
- BF16 execution: `static_parameters`
- Generation contract: `not_applicable`
- Artifact dependency set: `core`
- Weight publication allowed: `true`
- Weight license status: `resolved`
- Redistributable: `true`
- Complete weight publication required: `false`

## Release record

- FastPLMs weights: `Synthyra/ESMplusplus_large`
- Runtime revision: recorded in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in the source record
- Official checkpoint: `biohub/ESMC-600M`
- Artifact source: `fast`
- State transform: `esmc_to_fastplms_v1`
- Pinned upstreams: `biohub-esm`, `biohub-transformers`
- Release tiers: `check`, `compliance`, `feature`, `artifact`, `benchmark`
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
