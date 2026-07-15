# ESMFold2

FastPLMs supports exactly four Biohub ESMFold2 variants:

| Official checkpoint | FastPLMs mirror |
| --- | --- |
| `biohub/ESMFold2` | `Synthyra/ESMFold2` |
| `biohub/ESMFold2-Fast` | `Synthyra/ESMFold2-Fast` |
| `biohub/ESMFold2-Experimental-Cutoff2025` | `Synthyra/ESMFold2-Experimental-Cutoff2025` |
| `biohub/ESMFold2-Experimental-Fast-Cutoff2025` | `Synthyra/ESMFold2-Experimental-Fast-Cutoff2025` |

Other snapshots are not advertised in code, artifacts, tests, or documentation.
This repository does not delete or modify any live Hub repository.

## Loading

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Synthyra/ESMFold2-Fast",
    trust_remote_code=True,
    attn_implementation="flex_attention",
    device_map={"": "cuda:0"},
    esmc_precision="auto",
).eval()
```

The folding model and its ESMC backbone use the same explicitly resolved
attention implementation. Unsupported names raise. The ESMC checkpoint is
loaded directly on the requested CUDA device when a CUDA device is used. For a
declared ESMC-6B Hub identifier, FastPLMs forwards the immutable revision from
`models.toml` to both configuration and weight loading. Other remote ESMC
checkpoints are rejected because they do not satisfy the learned 81-state,
2560-width projection contract. A local checkpoint directory remains a
supported explicit source and has no Hub revision.

The folding checkpoint itself loads with FP32 parameters. Learned projection,
folding trunk, and diffusion computation run under CUDA BF16 autocast. ESMC has
an independent precision policy: its canonical BF16 weights may remain BF16 or
be used to reconstruct the transient FP8 inference path without changing the
folding checkpoint's FP32 storage.

## Learned sequence representation

Biohub ESMC-6B provides the embedding state followed by 80 transformer-layer
states. FastPLMs validates that exact 81-state ordering and width before applying
the folding checkpoint's learned projection:

```text
H: (b, l, 81, 2560)
```

For each state, `base_z_linear` applies layer normalization and a bias-free
linear map to width 256. The softmax of `base_z_combine` gives 81 scalar weights.
The weighted states are combined with the same matrix multiplication order as
Biohub:

```python
Z = model.project_esmc_hidden_states(H, residue_mask=M)
```

The result is:

```text
Z: (b, l, 256)
```

This is the learned sequence summary returned before `base_z_mlp` expands it
into pair features. The refactor retains the checkpoint names
`base_z_linear`, `base_z_combine`, and `base_z_mlp`.

Projection compliance compares identical 81-state inputs. FP32 output must be
exact. The BF16 engineering target for relative L2 error is `5e-4`, with a hard
limit of `1e-3`.

## Dataset embeddings

```python
result = model.embed_dataset(
    "proteins.fasta",
    batch_size=2,
    full_embeddings=True,
)
```

Each output record contains a residue tensor with shape `(l, 256)`. Pooling uses
only real residues. Supported poolers are the residue statistics `mean`, `max`,
`norm`, `median`, `std`, and `var`. ESMFold2 rejects `cls` because the learned
representation has no classification token semantics and rejects `parti`
because it is not an attention-graph representation.

Run metadata records the folding checkpoint identity and the resolved ESMC
repository, immutable revision, and manifest file hashes. These fields are part
of the resume fingerprint, so an embedding run cannot resume after either
checkpoint identity changes.

The dataset path accepts a single protein chain or FASTA records of single
chains. It rejects gaps, chain separators, structured complexes, ligands, MSAs,
and non-protein tokens. Those remain inputs to the folding preparation path,
not the embedding utility.

## ESMC precision policy

The public precision values are `auto`, `bf16`, `fp32`, and `fp8`:

```python
model.reload_esmc(precision="auto", device="cuda")
status = model.esmc_precision_status
print(status.as_dict())
```

The status contains requested and resolved precision, reason, device, and the
installed Transformer Engine version. Resolution is fail-closed:

- `auto` selects FP8 only for direct loading onto a CUDA device when Transformer
  Engine reports FP8 availability;
- `auto` otherwise selects BF16 and records the exact reason;
- explicit `fp8` raises when the device or Transformer Engine path is
  unavailable;
- explicit `bf16` and `fp32` remain supported.

Converting every ESMC linear compounds quantization error across 80 layers.
The validated path instead converts exactly each layer's attention output
projection, for 80 Transformer Engine linears in total. Their canonical
parameters remain BF16. `Float8CurrentScaling` quantizes the GEMMs during the
inference context, and sequence inputs are padded to a multiple of 16 before
ESMC execution.

Three fresh BF16-to-FP8 reload cycles on the locked H100 environment produced
identical metrics in each cycle: projection relative L2 `0.0375936`,
first-percentile residue cosine `0.999091`, and minimum per-sequence pooled
cosine `0.999754`. These satisfy the engineering targets of `0.04`, `0.995`,
and `0.999`, respectively. The runtime loads canonical BF16 weights and rebuilds
Transformer Engine modules on every startup or reload. Transformer Engine
workspaces and quantized caches are transient and are excluded from folding
checkpoints.

The locked H100 image uses Transformer Engine `2.12.0` with its CUDA 13 core.
The `uv` override excludes the package's default CUDA 12 core, so the FP8 image
contains one Transformer Engine runtime rather than two. Transformer Engine
`2.13` through `2.16` are not advertised on this validation stack because their
precompiled CUDA 13 cores require a newer cuBLAS ABI than the pinned CUDA 13.0
toolchain provides.

FastPLMs does not replace CUDA libraries, compile code at import time, or
serialize runtime-quantized state. The FP8 image builds Transformer Engine's
small PyTorch binding once against the locked Torch/CUDA ABI; its CUDA core is
precompiled. Core FastPLMs imports remain independent of Transformer Engine
because the optional runtime is loaded only when the precision policy needs its
capability probe or execution context.

## Validation record

On July 15, 2026, the locked H100 PCIe environment passed all 15 focused
ESMFold2 release tests with no failures, errors, or skips. The run covered all
four variants, official-versus-local BF16 folding, FP8-versus-BF16 folding,
three fresh reload cycles per variant, automatic FP8 selection, strict
unavailable-device behavior, and the CUDA 13 Transformer Engine stack.

Official-versus-local BF16 C-alpha RMSD ranged from `1.65e-6` to `2.86e-6`
angstrom, lDDT-C-alpha was `1.0`, and confidence-output errors were zero. Across
the four FP8 variants, the worst observed values were C-alpha RMSD `0.217190`
angstrom, lDDT-C-alpha `0.994244`, pLDDT MAE `0.004952`, PAE MAE `0.133353`
angstrom, pTM error `0.005495`, and mean probability Jensen-Shannon divergence
`0.000346`. Each value meets its engineering target. This record applies only
to the pinned checkpoint revisions and locked H100 environment documented by
the manifest.

## Gradient-enabled paths

Test-time training and other gradient-enabled ESMC execution use canonical BF16
weights. A future FP8 implementation must retain this boundary.

## Folding compliance

Structure tests hash prepared features and sampled diffusion noise before
comparing implementations. They require exact discrete features and masks,
finite outputs, valid geometry, and no NaNs.

For official versus local BF16, engineering targets are C-alpha RMSD at most
0.10 angstrom, lDDT-C-alpha at least 0.995, pLDDT MAE at most 0.001, PAE MAE at
most 0.10 angstrom, and pTM or ipTM error at most 0.002. Corresponding hard
limits are 0.25 angstrom, 0.99, 0.005, 0.50 angstrom, and 0.005.

For FP8 versus BF16, engineering targets are C-alpha RMSD at most 0.75 angstrom,
lDDT-C-alpha at least 0.97, pLDDT MAE at most 0.01, PAE MAE at most 0.5
angstrom, pTM or ipTM error at most 0.01, and mean probability Jensen-Shannon
divergence at most 0.002. Hard limits are 1.5 angstrom, 0.95, 0.02, 1.0
angstrom, 0.02, and 0.005.
