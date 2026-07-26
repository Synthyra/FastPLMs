---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/ESMFold2-Fast

This checkpoint packages the FastPLMs `ESMFold2` implementation.

Accepted inputs are raw amino-acid sequences or typed molecular-complex
specifications; low-level forward accepts prepared feature tensors.
Supported Transformers entry points are `AutoConfig`, `AutoModel`.

## Capabilities

| Feature | Status |
| --- | --- |
| Sequence classification | Unavailable: no advertised AutoClass |
| Token classification | Unavailable: no advertised AutoClass |
| PEFT fine-tuning | Supported pattern: attach LoRA to the pretrained model |
| Embeddings | Special: ESMC state mixture to 256-wide residue embeddings |
| Test-time training | Special: opt-in folding TTT on the ESMC backbone |
| Attention variants | Supported: `eager`, `sdpa`, `flex_attention` |
| Compliance | Declared: exact release evidence is required |

A supported interface is not a pretrained downstream predictor. Classification
heads start untrained, and declared compliance metadata is not a claim that an
arbitrary local build passed its release gate.

## Install and platform requirements

Install the direct dependencies published with this model:

```bash
python -m pip install -r \
  "https://huggingface.co/Synthyra/ESMFold2-Fast/resolve/main/requirements.txt"
```

The FastPLMs implementation itself is embedded in the model repository and loaded
by Transformers through `trust_remote_code=True`.

Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13 are required. The artifact requirements include the direct structure dependencies. The published execution contract requires a CUDA device. The current validated release target is the exact NVIDIA GH200 on Linux aarch64; Linux x86-64, CPU-only, Windows, and macOS structure runs are not current release evidence. The Hub quick start below requires network
access on first download. For an air-gapped run, first build the manifest-pinned
local artifact and use the offline form shown in the example.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/ESMFold2-Fast"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="sdpa",
).eval()
```

For offline validation, replace `model_id` with the manifest-built
`dist/hub/ESMFold2-Fast` path and pass `local_files_only=True`.

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

## Alignment-conditioning contract

This 24-block Fast checkpoint is inference-optimized for single-sequence
conditioning and was trained without MSA conditioning. It is not
MSA-conditioned and rejects `ProteinInput.msa` and low-level MSA-derived
features. Typed multichain and multimolecule inputs remain supported when
every protein chain uses `msa=None`. Use the corresponding full ESMFold2
checkpoint for MSA-conditioned inference. This follows the official Biohub
architecture description in [Appendix A.2.1](https://biohub.ai/papers/esm_protein.pdf).


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


## Hash-pinned CCD runtime asset

Structure preparation requires `ccd.pkl` from
`biohub/ESMFold2`. The manifest pins
its 417,306,584-byte size and SHA-256
`9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5`
under MIT terms. This is a trusted-deserialization boundary: FastPLMs only
allows the exact manifest repository/revision snapshot link to resolve within
that repository's contained blob directory; user-supplied asset and `cache_dir`
symlinks are rejected. The loader creates a private temporary snapshot, verifies
its size and SHA-256, and unpickles only that loader-owned snapshot, closing
path-replacement and in-place source-write races. Offline execution requires the
exact cache object and never downloads a replacement.

## Optional folding TTT

The standard and Fast checkpoints expose opt-in folding TTT on their ESMC
backbone:

```python
adapted = model.fold_protein_ttt(
    "MSTNPKPQRKTKRNT",
    num_loops=1,
    num_sampling_steps=50,
    seed=7,
    ttt_config={"steps": 3, "batch_size": 1, "seed": 7},
)
print(adapted.ttt_metrics)
```

Entering a gradient-enabled path reloads canonical BF16 ESMC weights. TTT adds
latency and memory, can worsen a prediction, and does not calibrate confidence
or establish biological validity. Folding TTT is result-scoped: its transient
ESMC adapter modules are excluded from checkpoint state, so it is not a generic
`save_pretrained` adapter-persistence path.

## Runtime contract

- Public input: Raw amino-acid sequences or typed molecular-complex specifications; low-level forward accepts prepared feature tensors
- Advertised AutoClasses: `AutoConfig`, `AutoModel`
- AutoClass weight status: `AutoConfig` = `FastPLMs extension`, `AutoModel` = `pretrained`
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

- FastPLMs weights: `Synthyra/ESMFold2-Fast`
- Runtime revision: recorded separately in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in `provenance.json`
- Official checkpoint: `biohub/ESMFold2-Fast`
- Artifact source: `fast`
- State transform: `identity`
- Pinned upstreams: `biohub-esm`, `biohub-transformers`, `protein-ttt`
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
