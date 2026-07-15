# FastPLMs

FastPLMs provides compact, Hugging Face-compatible implementations of protein
language and structure models. Each supported family has a pinned official
repository under `vendor/upstream/` and a declared checkpoint conversion.
Release workflows compare configuration, tokenization, parameters, aliases,
and inference against that official source.

The package is intentionally small at runtime. Official repositories are parity
oracles, not dependencies, and are never copied into runtime images or imported
by production code.

## Install

FastPLMs 1.0 requires Python 3.11 through 3.14, PyTorch 2.13, and Transformers
5.13. Install an editable checkout with the locked development environment:

```bash
git submodule update --init --recursive
uv sync --extra dev
```

Install directly from a Git revision when only the package is needed:

```bash
python -m pip install "fastplms @ git+https://github.com/Synthyra/FastPLMs.git@<revision>"
```

Optional dependency groups are isolated as `structure`, `flash`, `fp8`,
`train`, and `dev`. Core installation contains only Torch, Transformers,
Hugging Face Hub, tokenizers, safetensors, NumPy, einops, and tqdm.

## Load a model

FastPLMs keeps the standard Transformers loading interface:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Synthyra/ESM2-150M",
    trust_remote_code=True,
    attn_implementation="flex_attention",
)
```

The Hub identifier in this example assumes that its generated FastPLMs 1.0
artifact has been published. Before publication, build the manifest-pinned
artifact as described in the [artifact guide](docs/artifacts.md) and pass its
local `dist/hub/<model>` path to the same call.

Leave `attn_implementation` unspecified to let Transformers select its normal
default. A loaded model can be changed explicitly with:

```python
model.set_attn_implementation("sdpa")
```

Backends are model-family capabilities declared in
[`models.toml`](src/fastplms/models.toml). Supported names are `eager`, `sdpa`,
`flex_attention`, `flash_attention_2`, and `flash_attention_3`, limited by the
selected family. An unavailable or unsupported requested backend raises instead
of silently falling back. See the [attention guide](docs/attention_backends.md).

FastPLMs never installs or compiles the `flash-attn` source package. The
isolated `flash` extra installs Hugging Face `kernels` only. FastPLMs resolves
`kernels-community/flash-attn2` at revision `db6b51744f0c` for its PyTorch 2.13,
CUDA 13, C++11 ABI artifact and `kernels-community/flash-attn3` at revision
`43f0bd269777` for its CUDA 13 stable-ABI artifact. The tracked `kernels.lock`
binds those immutable snapshots to exact variant hashes. At first execution,
FastPLMs downloads and validates the compatible binary before importing it;
`kernels download .` can populate the cache ahead of service startup. CPU or
mixed-device Q, K, and V fail before that download. ESM2 and ESM++ advertise both after dense and
mixed-padding H100 parity checks. DPLM advertises FlashAttention 3 only because
FlashAttention 2 misses its engineering relative-L2 target. DPLM2 advertises
SDPA only because all tested alternate backends miss its deep-parity engineering
target. Source compilation is not a fallback.

## Embed sequences

The same ordered embedding API is available as a package function and a model
method:

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

Order and duplicate identifiers are preserved. Pooling includes only biological
residues and supports `mean`, `max`, `norm`, `median`, `std`, `var`, `cls`, and
`parti` where the model has the required semantics. Persisted outputs default to
2 GiB-sharded safetensors with a JSON index and run manifest. SQLite is
available for transactional streaming and exact resume. See the
[embedding guide](docs/embedding_api.md).

## ESMFold2 learned representation and FP8

FastPLMs supports exactly four ESMFold2 checkpoints: standard, fast,
experimental cutoff 2025, and experimental fast cutoff 2025. Their learned
sequence representation combines the 81 ordered ESMC hidden states and projects
them to width 256 before pair features are created:

```python
# H: (b, l, 81, 2560)
Z = model.project_esmc_hidden_states(H)  # Z: (b, l, 256)
```

For datasets, `model.embed_dataset(..., full_embeddings=True)` returns one
residue tensor with shape `(l, 256)` per single-chain sequence. ESMFold2 rejects
`cls`, `parti`, complexes, ligands, MSAs, and chain-separated inputs in this
embedding path.

ESMC serving precision is explicit and reloadable:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Synthyra/ESMFold2-Fast",
    trust_remote_code=True,
    device_map={"": "cuda:0"},
    esmc_precision="auto",
).eval()
model.reload_esmc(precision="auto", device="cuda:0")
print(model.esmc_precision_status)
```

`auto` selects FP8 only when ESMC is loaded directly onto a supported CUDA
device and Transformer Engine reports availability. Otherwise it selects BF16
and records the reason. Explicit `fp8` is strict. The validated path converts
the 80 ESMC attention output projections to Transformer Engine linears and
uses current-scaling FP8 inference while retaining canonical BF16 checkpoint
weights. Runtime quantization state is never serialized. Gradient-enabled and
test-time-training paths reload BF16. See the [ESMFold2 guide](docs/esmfold2.md)
for the measured three-reload compliance evidence.

## Model support and provenance

[`src/fastplms/models.toml`](src/fastplms/models.toml) is the sole model
manifest. It records immutable checkpoint revisions and file identities,
AutoClasses, tokenizer modes, conversion records, attention and precision
paths, upstream source revisions, containers, licenses, and test coverage.
The generated [support matrix](docs/generated/support.md) lists the current
families and checkpoints.

Initialize the exact upstream sources with:

```bash
git submodule update --init --recursive
```

See [`vendor/README.md`](vendor/README.md) for revision and license policy and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution. ANKH weights
retain CC BY-NC-SA 4.0 terms. E1 retains its agreement, attribution, notice, and
launch display requirements. FastPLMs reports these terms but does not enforce
usage restrictions.

## Test and benchmark

All validation is containerized and is designed to run on the declared H100
environment. The portable runner accepts host and identity parameters at
invocation time and does not store workstation details:

```bash
python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/key \
  --suite check
```

The release tiers are `check`, `compliance`, `structure`, `feature`, `artifact`,
and `benchmark`. Missing required dependencies or backends fail rather than
skip. Python 3.12 is the canonical GPU environment; `--suite python-matrix`
validates locked, non-editable core installs on Python 3.11, 3.13, and 3.14.
Benchmarks run outside pytest, use pre-tokenized tensors and CUDA events, and
retain raw samples and environment metadata. See the
[testing](docs/testing.md), [benchmarking](docs/benchmarking.md), and
[artifact](docs/artifacts.md) guides.

## Documentation

- [Architecture](docs/architecture.md)
- [Models and generated support](docs/models.md)
- [Embeddings](docs/embedding_api.md)
- [Attention backends](docs/attention_backends.md)
- [ESMFold2](docs/esmfold2.md)
- [Testing and compliance](docs/testing.md)
- [Benchmarking](docs/benchmarking.md)
- [Local Hub artifacts](docs/artifacts.md)
- [Migration to 1.0](docs/migration.md)
- [Licensing](docs/licensing.md)
- [Test-time training](docs/ttt.md)
- [Binder workflow](docs/binder_design.md)
- [Fine-tuning](docs/finetuning.md)
- [Contributing](docs/contributing.md)

FastPLMs 1.0 is an intentional API break. There are no legacy import or command
shims. Checkpoint keys remain stable where possible; otherwise the manifest
names a deterministic converter with an exact conversion test.
