# FastPLMs

<img width="2816" height="1536" alt="FastPLMs Hero Image" src="https://github.com/user-attachments/assets/ffaf84b6-9970-40fd-aa31-1b314d6ca146" />

FastPLMs provides Hugging Face-compatible protein language and structure
models. It keeps the familiar Transformers interface while making attention,
embedding, generation, folding, and validation behavior explicit.

The runtime package does not import an official model checkout. Each supported
family instead has a pinned upstream source under `vendor/upstream/`, an
immutable checkpoint identity, and a declared state transformation. Release
workflows compare the resulting FastPLMs artifact against that official source.

## Contents

- [Why FastPLMs](#why-fastplms)
- [Supported models](#supported-models)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage examples](#usage-examples)
- [Attention backends](#attention-backends)
- [Design choices](#design-choices)
- [Validation and reproducibility](#validation-and-reproducibility)
- [Files-only Hub publication](#files-only-hub-publication)
- [Documentation](#documentation)
- [Contributing and citation](#contributing-and-citation)

## Why FastPLMs

Protein models are often published with architecture-specific loading code,
tokenization conventions, attention implementations, and output formats.
FastPLMs separates those concerns:

- Transformers auto classes provide a consistent loading interface.
- Model-specific adapters preserve native biological token and structure
  semantics.
- A shared embedding API returns ordered, residue-aware representations.
- Attention backends are explicit capabilities rather than silent fallbacks.
- The model registry records checkpoint, source, conversion, precision,
  license, and test contracts in one typed manifest.
- Official repositories are isolated parity oracles, not production
  dependencies.

A supported model is more than an architecture implementation. Its
configuration, input preparation, checkpoint conversion, representative
inference, artifact contents, and legal inventory are all defined and tested.

## Supported models

The generated [support matrix](docs/generated/support.md) is the authoritative
checkpoint list. It is rendered from
[`src/fastplms/models.toml`](src/fastplms/models.toml), which also defines valid
AutoClasses, attention backends, precision paths, and release tiers.

| Family | Primary use | Typical user input | Important distinction |
| --- | --- | --- | --- |
| ESM2 | Sequence representations and masked language modeling | Amino-acid sequences tokenized to residue IDs | Preserves ESM2 encoder, MLM, contact, and classification contracts |
| ESM++ / ESMC | Sequence representations and masked language modeling | Amino-acid sequences tokenized to residue IDs | Biohub ESMC implementation and ESMFold2 language-model backbone |
| ESM3 | Multimodal protein modeling and generation | Sequence, structure, and function tracks | Retains all three tracks through its multimodal interface |
| E1 | Retrieval-augmented protein encoding | Raw amino-acid sequences | No tokenizer; native E1 preparation is preserved |
| DPLM | Discrete diffusion protein generation | Amino-acid sequences tokenized to masked residue IDs | Confidence-based iterative unmasking |
| DPLM2 | Amino-acid and structure co-generation | Amino-acid and structure token tracks | Separate structure and amino-acid boundary tokens |
| ANKH | T5 protein encoding and sequence-to-sequence modeling | Amino-acid sequences tokenized for encoder or seq2seq use | The 1.0 artifact contract is one full official-compatible encoder-decoder checkpoint with encoder-default embeddings |
| ESMFold | Sequence-to-structure inference | Raw amino-acid sequences | Meta ESMFold contract with FastPLMs ESM2 backbone |
| ESMFold2 | Sequence and complex structure prediction | Raw amino-acid sequences or complex specifications | Full variants have 48 folding blocks and optional MSA conditioning; Fast variants have 24 blocks and no MSA conditioning |
| Boltz2 | Structure prediction | Raw amino-acid sequences or prepared model features | Provisional end-to-end numerical-equivalence status |

The model manifest, not this summary, controls support. A backend or AutoClass
that is valid for one family may be rejected by another.

The Synthyra ANKH repositories contain the complete 1.0 encoder-decoder
checkpoints. `AutoModel` loads the encoder view, and
`AutoModelForSeq2SeqLM` loads the decoder, cross-attention, and language-model
head from the same repository.

## Installation

FastPLMs 1.0 requires Python 3.11 through 3.14 and PyTorch 2.13. The compatible
Transformers requirement is Transformers 5.13. For a development checkout:

```bash
git clone https://github.com/Synthyra/FastPLMs.git
cd FastPLMs
uv sync --extra dev
```

Official reference repositories are not runtime or routine-development
dependencies. Initialize them only for a live release-candidate compliance run,
as described under [Validation and reproducibility](#validation-and-reproducibility).

Install a pinned Git revision when only the runtime package is needed:

```bash
python -m pip install \
  "fastplms @ git+https://github.com/Synthyra/FastPLMs.git@<revision>"
```

Optional dependency groups are isolated by purpose:

| Extra | Purpose |
| --- | --- |
| `cpu` | CPU-only PyTorch 2.13 selection through uv's explicit PyTorch CPU index; incompatible with the CUDA-only `cueq` and `fp8` extras |
| `structure` | ESMFold2 and Boltz2 runtime dependencies, including Accelerate for memory-safe 6B `device_map` loading and OmegaConf for trusted official Lightning checkpoint deserialization |
| `binder` | Bounded AbNumber 0.4.4, ANARCII 2.0.8, pandas, and PyArrow dependencies used only by the binder-design research workflow; combine with `structure` |
| `cueq` | Optional ESMFold2 cuEquivariance kernels on the locked Linux CUDA 13 release stack; installs the separately licensed NVIDIA CUDA runtime |
| `flash` | Pinned precompiled Hugging Face FlashAttention kernels |
| `fp8` | Experimental ESMFold2 ESMC FP8 inference |
| `train` | Trainer, Accelerate, datasets, and PEFT workflows |
| `reporting` | Fine-tuning and research-example plots and statistical reports |
| `dev` | Tests, type checking, linting, and package builds |

Core installation contains Torch, Transformers, Hugging Face Hub, tokenizers,
safetensors, NumPy, einops, and tqdm. Official reference implementations are
not installed into the runtime environment.

## Quick start

Load a model with its standard Transformers auto class:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Synthyra/ESM2-150M",
    trust_remote_code=True,
    attn_implementation="sdpa",
)
model.eval()
```

Use the published Hub identifier for ordinary loading. Pin `revision` when an
immutable model-code snapshot is required. Contributors can instead build the
manifest-pinned artifact under `dist/hub/ESM2-150M` and load it locally before
publication.

Tokenizer-based models use the tokenizer paired with the same artifact:

```python
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Synthyra/ESM2-150M",
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

E1 is intentionally different. It has no tokenizer and retains its native
raw-sequence preparation path. Use `model.embed_dataset(...)` for ordinary
sequence representations or the explicit E1 preparation methods for lower
level token tensors.

## Usage examples

### Ordered sequence embeddings

The package function and model method share the same implementation:

```python
from fastplms import EmbeddingInput, embed_dataset

result = embed_dataset(
    model,
    [
        EmbeddingInput("protein-a", "MSTNPKPQRKTKRNT"),
        EmbeddingInput("protein-a", "MKTIIALSYIFCLVFA"),
    ],
    batch_size=2,
    pooling=("mean", "std"),
    output="embeddings",
)
```

Insertion-ordered mappings are also accepted and preserve their keys as record
identifiers:

```python
result = model.embed_dataset(
    {
        "protein-a": "MSTNPKPQRKTKRNT",
        "protein-b": "MKTIIALSYIFCLVFA",
    },
    batch_size=2,
    pooling="mean",
)
```

`EmbeddingResult` preserves order, duplicate identifiers, and the original
sequence. Each record contains an identifier, sequence, and tensor:

```python
for record in result.records:
    tensor = record.load_tensor()
    print(record.id, record.sequence, tensor.shape)
```

Calling `result.as_dict(key="id")` raises when identifiers repeat unless an
explicit duplicate policy is provided. This avoids silently overwriting FASTA
records.

Safetensors output packs generation-scoped shards across batches and can resume
from the last flushed shard after interruption. An interrupted in-memory shard
is recomputed. Tensor memory is bounded by the configured shard size rather than
the full dataset size. Successful overwrites retain prior immutable generations
so already-open lazy readers remain valid. Stale generations are removed only
through explicit, dry-run-first garbage collection after the caller guarantees
there are no active readers or writers. SQLite remains available when database
transactions and queryable records are preferred.

### FASTA input and in-memory output

Pass a FASTA path directly. Multi-line sequences and record identifiers are
preserved:

```python
result = model.embed_dataset(
    "proteins.fasta",
    batch_size=32,
    batch_window_size=128,
    max_tokens_per_batch=4096,
    max_length=1024,
    pooling=("mean", "max"),
)

for record in result:
    print(record.id, record.tensor.shape)
```

When `output` is omitted, tensors remain in memory. Multiple poolers are
concatenated in request order, and `result.metadata["pool_slices"]` records the
slice belonging to each transformation. FASTA is streamed line by line into an
immutable fingerprinted spool. Length bucketing is bounded by
`batch_window_size`, which defaults to sixteen times `batch_size`; output order
is restored, and `max_length` always counts biological residues rather than
tokenizer-added special tokens. SQLite prefixes commit at completed batch-window
boundaries. Safetensors prefixes publish only when a bounded shard flushes, so
an interruption replays the unflushed shard rather than necessarily one window.
Set the window equal to the batch size when per-batch SQLite checkpoint
boundaries matter.

### Full residue and hidden-state embeddings

Use full embeddings when a downstream task needs one vector per biological
residue:

```python
residue_result = model.embed_dataset(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    batch_size=2,
    full_embeddings=True,
)

for record in residue_result:
    print(record.sequence, record.tensor.shape)
```

Each tensor has shape `(l, d)` after BOS, EOS, padding, chain delimiters, and
other non-biological positions are removed. To retain every returned model
state:

```python
state_result = model.embed_dataset(
    ["MSTNPKPQRKTKRNT"],
    full_embeddings=True,
    store_all_hidden_states=True,
)
print(state_result[0].tensor.shape)
```

The hidden-state tensor has shape `(n, l, d)`, where `n` follows the model's
native hidden-state output order.

ANKH selects the encoder final state by default. Decoder layers require the
full sequence-to-sequence view and explicit decoder inputs. Pass proteins as
raw residue strings without inserted spaces and keep sentinel prompts tight,
for example `M<extra_id_0>`. The model-owned and explicitly supplied tokenizer
paths apply the same ANKH normalization contract.

```python
from transformers import AutoModelForSeq2SeqLM

seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
    "Synthyra/ANKH_base",
    trust_remote_code=True,
).eval()
decoder_result = seq2seq.embed_dataset(
    ["MSTNPKPQRKTKRNT"],
    hidden_state_source="decoder",
    hidden_state_index=-1,
    decoder_inputs=["M<extra_id_0>"],
    full_embeddings=True,
)
```

There is no implicit shifted-source decoder contract. Official ANKH tasks use
task-dependent prompts, sentinels, or generated tokens. Set
`hidden_state_source="encoder"` and `hidden_state_index` for any encoder layer,
or `store_all_hidden_states=True` for every layer in the selected stack.

### Safetensors output and exact resume

Directory output uses sharded safetensors by default:

```python
result = model.embed_dataset(
    "proteins.fasta",
    batch_size=64,
    pooling=("mean",),
    output="artifacts/protein-embeddings",
    format="safetensors",
    resume=True,
)
```

The directory contains tensor and descriptor shards, immutable generation
indexes, the `index.json` convenience pointer, and authoritative `run.json`
commit marker. Generation descriptors record sequence order, identifiers,
shapes, dtypes, hashes, and shard keys. The run manifest binds the input, model
state, tokenizer policy, backend, and pooling configuration. Resume is accepted
only when existing records are the exact ordered prefix of the same run.
`run.json` selects an immutable generation; older generations remain available
to readers opened before a successful overwrite. See
[Embedding API](docs/embedding_api.md#safetensors-storage) for the exclusive
garbage-collection contract.

### SQLite streaming

SQLite commits each completed batch window and is useful for long jobs:

```python
result = model.embed_dataset(
    "proteins.fasta",
    batch_size=64,
    pooling=("mean", "std"),
    output="artifacts/protein-embeddings.sqlite",
    format="sqlite",
    resume=True,
)
```

BF16 tensors are stored as raw bytes with an explicit dtype. Resume rejects a
changed model state, input order, sequence, tokenizer, pooling configuration,
or attention backend.

SQLite readers open results read-only and preserve ordered duplicate filters:

```python
from fastplms.embeddings import load_sqlite_result

selected = load_sqlite_result(
    "artifacts/protein-embeddings.sqlite",
    record_ids=["protein-b", "protein-a", "protein-b"],
)
```

### DPLM sequence generation

DPLM starts from a tokenized sequence whose biological positions define the
requested length. Iterative unmasking fills those positions:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "Synthyra/DPLM-150M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
generator = AutoModelForMaskedLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
).cuda().eval()

input_ids = tokenizer("A" * 64, return_tensors="pt")["input_ids"].cuda()
with torch.inference_mode():
    generated_ids = generator.generate(input_ids, max_iter=100)

sequence = tokenizer.decode(
    generated_ids[0],
    skip_special_tokens=True,
).replace(" ", "")
print(sequence)
```

The official schedule uses 500 steps when `max_iter` is omitted. Reducing the
step count changes the sampling process. DPLM2 uses separate structure and
amino-acid tracks; see the [model guide](docs/models.md) for its explicit
boundary-token example.

### ESM3 sequence generation

ESM3 keeps multimodal helpers on the loaded model:

```python
from fastplms.models.esm3.modeling_esm3 import FastESM3GenerationConfig
from transformers import AutoModel

esm3 = AutoModel.from_pretrained(
    "Synthyra/ESM3_small",
    trust_remote_code=True,
).eval()
config = FastESM3GenerationConfig(num_steps=8, temperature=1.0, seed=7)
generated = esm3.generate("MK____A", config)
print(generated)
```

Underscores mark sequence positions to generate. Sequence-only forward passes
use `esm3.tokenize_sequences(...)`; structure and function tracks remain
available through the model's multimodal interface.

### Boltz2 protein structure prediction

Boltz2 exposes a protein-only convenience path in addition to its prepared
feature interface:

```python
import torch
from transformers import AutoModel

boltz = AutoModel.from_pretrained(
    "Synthyra/Boltz2",
    trust_remote_code=True,
    dtype=torch.float32,
).cuda().eval()
prediction = boltz.predict_structure(
    amino_acid_sequence="MSTNPKPQRKTKRNTNRRPQDVKFPGG",
    recycling_steps=3,
    num_sampling_steps=50,
    diffusion_samples=1,
    seed=7,
)
boltz.save_as_cif(prediction, "prediction.cif")
print(prediction.plddt, prediction.ptm, prediction.iptm)
```

Boltz2 remains provisional in FastPLMs 1.0. This interface is covered for
configuration, declared inference-core state, feature preparation, and seeded
execution, but it is not yet an official end-to-end numerical-equivalence
claim.
The helper restores caller RNG state; FP32 parameters and features execute under
the documented CUDA BF16 autocast policy.

### ESMFold structure prediction

ESMFold accepts raw sequences:

```python
import torch
from transformers import AutoModel

folder = AutoModel.from_pretrained(
    "Synthyra/FastESMFold",
    trust_remote_code=True,
    dtype=torch.float32,
).cuda().eval()

with torch.inference_mode():
    structure = folder.infer("MKTLLILAVVAAALA")

print(structure["mean_plddt"])
```

FastPLMs does not expose ProteinTTT for ESMFold. The pinned folding checkpoint
does not contain a trained masked-language-model head for that objective.

### ESMFold2 folding and learned representations

ESMFold2 accepts amino-acid sequences and typed molecular-complex
specifications. A target structure is not an input. Atomic coordinates and
confidence values are produced by the model.

The two Fast checkpoints are inference-optimized for single-sequence use. They
have 24 folding blocks instead of 48 and were trained without MSA conditioning,
so they reject MSA-derived inputs. The full `ESMFold2` and
`ESMFold2-Experimental-Cutoff2025` checkpoints retain 48 blocks and optional
MSA conditioning. Fast is not necessarily single-chain-only: supported
multichain and multimolecule requests remain available, but every protein chain
uses single-sequence mode. This distinction follows the official model
description in [Appendix A.2.1](https://biohub.ai/papers/esm_protein.pdf). The
quick start below intentionally loads Fast and supplies no MSA.

```python
from transformers import AutoModel

folder = AutoModel.from_pretrained(
    "Synthyra/ESMFold2-Fast",
    trust_remote_code=True,
    device_map={"": "cuda:0"},
    esmc_precision="auto",
).eval()

result = folder.fold_protein(
    "MSTNPKPQRKTKRNT",
    num_loops=1,
    num_sampling_steps=200,
    num_diffusion_samples=1,
    seed=7,
)
pdb_text = folder.result_to_pdb(result)
print(result.ptm, result.plddt.mean().item())
```

Build complexes with the input types exposed by the loaded artifact:

```python
types = folder.input_types
complex_input = types.StructurePredictionInput(
    sequences=[
        types.ProteinInput(id="A", sequence="MSTNPKPQRKTKRNT"),
        types.ProteinInput(id="B", sequence="MKTIIALSYIFCLVFA"),
        types.DNAInput(id="C", sequence="ATGC"),
        types.LigandInput(id="L", smiles="O"),
    ]
)
complex_result = folder.fold(
    complex_input,
    num_loops=1,
    num_sampling_steps=200,
    seed=7,
)
print(complex_result.ptm, complex_result.plddt.mean().item())
```

The typed interface supports RNA, ligands, modifications, covalent bonds, and
distogram conditioning. The Fast checkpoint loaded above rejects an MSA even
when it appears inside an otherwise valid typed request. To attach an MSA to a
protein input, load the full `Synthyra/ESMFold2` or
`Synthyra/ESMFold2-Experimental-Cutoff2025` checkpoint. The schema recognizes
`PocketConditioning`, but the pinned official runtime discards it and hard-codes
a zero pocket feature. FastPLMs rejects non-null pocket conditioning instead of
silently ignoring it. Prepared features contain fields such as `ref_pos`; these
are component reference geometries created during featurization, not a known
target structure.
See the offline
[`structure_preparation.py`](examples/structure_preparation.py) example for the
supported MSA, multimolecule, modification, bond, and distogram paths and the
explicit pocket rejection. Its ESMFold2 MSA branch requires one of the full
checkpoints, not a Fast checkpoint.

Its learned sequence representation combines 81 ordered ESMC hidden states
with the folding checkpoint's projection. Use the public embedding API to
retrieve the resulting residue representation:

```python
representations = folder.embed_dataset(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    batch_size=2,
    full_embeddings=True,
)
print(representations[0].tensor.shape)  # (sequence_length, 256)
```

For lower-level integrations that already hold the ordered ESMC hidden-state
stack, the projection is also exposed directly:

```python
# H: (b, l, 81, 2560)
Z = folder.project_esmc_hidden_states(H)  # Z: (b, l, 256)
```

Here, `H` is the 81-state ESMC representation for a prepared sequence batch,
not a target structure. The dataset embedding API is the higher-level path for
ordinary sequence inputs.

`folder.embed_dataset(..., full_embeddings=True)` returns one `(l, 256)` tensor
per single-chain sequence. The embedding path rejects complexes, ligands, MSAs,
chain-separated inputs, `cls`, and `parti`.

`esmc_precision="auto"` always resolves to BF16. Explicit FP8 is experimental,
inference-only, and strict:

```python
folder.reload_esmc(precision="fp8", device="cuda:0")
print(folder.esmc_precision_status)
```

FP8 raises when the validated CUDA and Transformer Engine path is unavailable.
Gradient-enabled paths reload canonical BF16 ESMC weights.

### Test-time training

Supported masked-language models expose opt-in, low-rank test-time adaptation:

```python
metrics = generator.ttt(
    seq="MSTNPKPQRKTKRNT",
    ttt_config={
        "steps": 3,
        "ags": 1,
        "batch_size": 1,
        "seed": 7,
    },
)
generator.ttt_reset()
```

TTT updates injected adapter parameters, not base checkpoint weights. It adds
latency and memory, can worsen a prediction, and does not establish biological
function. See the [TTT guide](docs/ttt.md) for supported families and folding
behavior.

### Binder design research example

The FastPLMs binder-design example optimizes a soft binder sequence against
ESMFold2 structural objectives and an ESM++ sequence prior:

![FastPLMs EGFR minibinder design](docs/assets/egfr_fastplms_binder_design.png)

Run it with the `structure` extra and the example-only table and antibody
dependencies. The published workflow requires Python 3.11-3.14, PyTorch 2.13,
Transformers 5.13, the verified ESMFold2 runtime assets, and a CUDA device. The
current release evidence target is the exact containerized Linux aarch64
environment on the NVIDIA GH200 workstation. CPU-only, x86-64, Windows, macOS,
H100, and H200 binder runs do not substitute for that evidence.

```bash
uv run --frozen \
  --extra structure \
  --extra binder \
  python examples/binder_design_fastplms.py \
  --target-name pd-l1 \
  --binder-name minibinder \
  --batch-size 4 \
  --steps 150 \
  --output-dir artifacts/binder-design
```

The workflow writes optimization trajectories, ranked sequences, structures,
confidence outputs, and selection tables. These are model outputs for
prioritization, not experimental evidence of binding, specificity, expression,
or therapeutic activity. The output directory must not already exist. The
workflow publishes `run_manifest.json` atomically last, so a missing manifest
marks an incomplete run. See the [binder-design guide](docs/binder_design.md).

### Fine-tuning

FastPLMs follows `PreTrainedModel` conventions for Trainer, Accelerate, and
PEFT workflows. Training needs the `train` extra. Plotting is opt-in; add the
`reporting` extra and `--plot-results` when requested:

```bash
uv run \
  --extra train \
  --extra reporting \
  python examples/fine_tuning.py \
  --task classification \
  --model_path Synthyra/ESM2-8M \
  --model-revision 185ecbd45665d050a8dae326d91886d330c5f9d0 \
  --classification-dataset-source GleghornLab/DL2_reg \
  --classification-dataset-revision 7e18f1b98859b0a3e3da283f63d0a153b774cf1f \
  --attn-backend sdpa \
  --output-dir artifacts/fine-tuning \
  --seed 7 \
  --full-determinism \
  --plot-results
```

For residue-level tasks, align labels to biological residues rather than
assuming tokenizer position equals residue position. Remote model and dataset
inputs require immutable 40-character Hub commits; existing local directory
inputs are identified by a full tree SHA-256 instead. Local datasets must use a
layout accepted by `datasets.load_dataset`; arbitrary `Dataset.save_to_disk()`
trees are not accepted. The example writes
`run_manifest.json` with ordered post-filter hashes of the rows and columns
actually consumed by training. It atomically publishes `final_model`, reloads
it against the same immutable base, and verifies the persisted adapter and
classifier tensor hashes. Omit `--extra reporting` and `--plot-results` for the
default plot-free run. See the
[fine-tuning guide](docs/finetuning.md).

## Attention backends

Select a backend at load time or change it explicitly after loading:

```python
model.set_attn_implementation("sdpa")
```

| Backend | Use when | Main constraint |
| --- | --- | --- |
| `eager` | Attention matrices or the `parti` pooler are required | Materializes attention scores |
| `sdpa` | A stable default or official-parity path is required | Torch selects the underlying kernel |
| `flex_attention` | Padding-aware compiled attention is useful | First use compiles for the requested shape and semantics |
| `flash_attention_2` | A supported ESM2 or ESM++ BF16 CUDA path needs a precompiled kernel | BF16-only and family-limited |
| `flash_attention_3` | A supported ESM2, ESM++, or DPLM BF16 CUDA path needs a precompiled kernel | BF16-only and family-limited |

FastPLMs does not implement an `auto` backend. An unavailable request raises.
When an optimized implementation cannot return attention matrices,
`output_attentions=True` emits one warning naming the configured backend,
effective eager backend, and reason, then runs a correctly masked eager call.
It does not change model configuration or later calls. Leaving
`attn_implementation` unspecified lets
Transformers choose its standard default. Explicit requests either run the
declared implementation or raise.

The `flash` extra installs Hugging Face `kernels`, not the `flash-attn` source
package. FastPLMs resolves immutable FlashAttention 2 and 3 snapshots recorded
in `kernels.lock`, validates the compatible binary, and never compiles a source
fallback. See the [attention guide](docs/attention_backends.md) for exact
family, dtype, padding, and numerical contracts.

For ESMC, SDPA is the recommended highest-fidelity path. Flex Attention and
FlashAttention 3 remain supported, non-experimental backends whose numerical
deviations are diagnostic rather than strict parity failures. The current
frozen-head release report is produced on the exact GH200/aarch64 validation
target and must publish relative L2, Q99.9, residue and pooled cosine, top-1,
and Jensen-Shannon distributions for each backend, dtype, exact hardware, and
sequence panel. H100 and H200 remain supported Hopper-class devices, but their
results are not interchangeable with or accepted as the current GH200 release
evidence. Pending measurements are
labeled pending in every ESMC card; no number is inferred from a threshold or
another checkpoint.

## Design choices

### Manifest-driven model support

[`src/fastplms/models.toml`](src/fastplms/models.toml) is the source of truth
for model IDs, files, revisions, AutoClasses, tokenizer modes, transformations,
attention and precision capabilities, upstreams, licenses, and release tiers.
Support tables and model cards are generated from it.

This avoids three common failure modes: a model appearing in documentation but
not release tooling, a checkpoint conversion with no immutable identity, and a
backend being advertised because it imports rather than because it was tested.

### Native biological preparation

The shared interface does not force every family through one tokenizer. E1
keeps raw-sequence preparation. DPLM2 keeps modality-specific boundaries.
Structure models retain their native protein, nucleic-acid, ligand, and chain
representations. Full ESMFold2 checkpoints additionally retain optional MSA
conditioning; ESMFold2 Fast checkpoints reject MSA-derived inputs. Shared
pooling begins only after each model identifies its biological residue
positions.

### Ordered embedding results

Protein datasets routinely contain duplicate sequences and duplicate FASTA
identifiers. FastPLMs returns ordered records instead of a sequence-keyed
dictionary so those inputs are not silently lost. Persistent formats record
per-tensor hashes and complete run fingerprints for auditing and exact resume.

### Explicit precision and backend policy

Parameter storage, compute dtype, and attention implementation are separate
choices. DPLM and several structure paths retain FP32 parameters while using
CUDA BF16 autocast. ESMFold2 controls the ESMC backbone independently from its
folding trunk. Unsupported combinations fail before inference.

### Official code is a parity oracle

Pinned official repositories live under `vendor/upstream/` and are used only
in isolated reference stages. Production modules cannot import them, modify
`sys.path` to reach them, or download source code at import time. Artifacts are
self-contained and load through Transformers with `trust_remote_code=True`.

### Fail-closed artifacts and licenses

Artifact construction verifies required file identities, canonical legal
texts, conversion details, generated model-card metadata, and offline
loading. A missing or changed required file is a release error. Source licenses
and notices are centralized under [`LICENSES/`](LICENSES/); checkpoint-specific
terms remain distinct from the FastPLMs Apache-2.0 code license.

## Validation and reproducibility

All release validation is containerized. The portable runner accepts the host
and identity at invocation time:

```bash
python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/key \
  --suite check
```

Release tiers cover checks, compliance, structure, features, artifacts, and
benchmarks. Missing required dependencies or declared backends fail rather than
skip. Expensive suites retain explicit `gpu`, `slow`, `large`, and `structure`
markers.

Every pull request also runs a positive, fully offline `tests/cpu/` allowlist on
Python 3.12, CPU-only Torch 2.13, and Transformers 5.13. The required GitHub
status is `cpu-contracts (3.12)` in the `CPU and package contracts` workflow. It
hides CUDA, blocks socket and Hub downloads, rejects skips and xfails, and
targets less than five minutes on four hosted CPU cores. Live official
references are reserved for the release-candidate `compliance` tier; routine
checks consume immutable goldens. Pull-request CI has only one additional
consolidated Python 3.12 quality/package smoke; cross-version, every-extra,
official-reference, and GPU validation run only through explicit workstation
or release suites.

The canonical Docker workflow uses one candidate image and isolated official
reference images:

```bash
git submodule update --init --recursive
sudo docker buildx bake -f docker/docker-bake.hcl candidate reference-esm2 --load
sudo docker compose -f docker/compose.yaml run --rm candidate \
  python -m pytest tests/parity -k esm2 -v
```

Always pass `--ipc=host` to Dockerized PyTorch runs. Benchmarks are separate
from correctness checks and retain raw samples, environment metadata, warm-up,
compile-time, steady-state, padding, and memory measurements.

Boltz2 remains provisional in FastPLMs 1.0. Configuration, declared
inference-core state, feature preparation, and seeded execution are tested, but
native-environment BF16 end-to-end inference does not yet meet the fixed
numerical-equivalence limits. FastPLMs therefore makes no official inference
equivalence claim for Boltz2.

## Files-only Hub publication

After building and validating local artifacts, update Hub runtime files and
model cards without uploading or deleting checkpoint weights:

```bash
PYTHONPATH=src python -m tools.artifacts.publish \
  --files-only \
  --artifact-root dist/hub \
  --dry-run \
  esm2_8m
```

`--artifact-root` is the local directory containing one built artifact
subdirectory per selected model. It tells the publisher which validated files
to upload; it is not a remote Hub path and does not change where models are
published.

Remove `--dry-run` after reviewing the exact add-only file plan. Repository
targets come exclusively from `models.toml`. Files-only publication rejects any
ANKH selection, including the implicit all-model selection, so callers must pass
explicit non-ANKH model IDs. Authentication uses `HF_TOKEN` or the cached
Hugging Face login. See [Local Hub artifacts](docs/artifacts.md) for the full
safety contract.

This files-only workflow is forbidden for the ANKH 1.0 migration. ANKH must
replace its encoder-only contents with the full encoder-decoder state in one
immutable commit containing every weight shard, tokenizer asset, configuration,
runtime source, card, and release record. Validate both `AutoModel` and
`AutoModelForSeq2SeqLM` from that same commit before publication is accepted.
Use the explicit `--complete <ankh-model-id>` dry-run and publication workflow.
It makes one parent-guarded atomic commit containing validated additions and
only the narrowly scoped deletions needed to replace obsolete registry-pinned
files, such as a monolithic ANKH weight file superseded by indexed shards.
DPLM1 and DPLM2 checkpoint weights are Apache-2.0. The maintained ByteDance
[license](https://github.com/bytedance/dplm/blob/main/LICENSE)
is Apache-2.0 and the [README](https://github.com/bytedance/dplm/blob/main/README.md#overview)
defines the repository release as including pretrained DPLM1 and DPLM2 weights.
Validated DPLM artifacts therefore record `weights_license_status="resolved"`
and `redistributable=true`; explicit `--complete` publication is permitted after
the same legal, parity, inventory, parent-commit, and atomic preflight checks.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Models and generated support](docs/models.md)
- [Capability-to-evidence manifest](docs/generated/capability_evidence.md)
- [Embedding API](docs/embedding_api.md)
- [Attention backends](docs/attention_backends.md)
- [ESMFold2](docs/esmfold2.md)
- [Test-time training](docs/ttt.md)
- [Binder design](docs/binder_design.md)
- [Fine-tuning](docs/finetuning.md)
- [Testing and compliance](docs/testing.md)
- [Benchmarking](docs/benchmarking.md)
- [Local Hub artifacts](docs/artifacts.md)
- [Migration to 1.0](docs/migration.md)
- [Licensing](docs/licensing.md)
- [Contributing](docs/contributing.md)
- [Runnable examples](examples/README.md)

FastPLMs 1.0 is an intentional API break. There are no legacy import, backend,
embedding-storage, or command shims. Use the migration guide when moving from
the pre-1.0 repository layout.

## Contributing and citation

Start with [AGENTS.md](AGENTS.md) for repository invariants and
[the contributing guide](docs/contributing.md) for the model-addition and
validation workflow. Generated support tables and model cards must be changed
through the manifest or renderer rather than edited directly.

If FastPLMs supports your work, cite FastPLMs and the paper or model card for
the specific checkpoint family:

```bibtex
@misc{FastPLMs,
  author = {Hallee, Logan and Bichara, David and Gleghorn, Jason P.},
  title = {FastPLMs: Fast, efficient protein language model inference from Hugging Face AutoModel},
  year = {2024},
  url = {https://github.com/Synthyra/FastPLMs},
  doi = {10.57967/hf/3726},
  publisher = {Hugging Face}
}
```
