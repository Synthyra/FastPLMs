# Architecture

FastPLMs separates runtime code, release evidence, and official reference code.
Tests and container build contexts enforce this separation.

## Repository boundaries

```text
src/fastplms/       runtime source copied into Hugging Face artifacts
tests/              unit, integration, parity, structure, and release tests
benchmarks/         standalone, exact-device Hopper/SM90 performance harness
docker/             candidate, runtime, and reference container definitions
examples/           runnable training and protein-design workflows
tools/              artifact, conversion, remote, and debugging commands
vendor/upstream/    pinned official Git submodules
LICENSES/           distributable third-party legal texts
model_cards/        generated checkpoint cards
```

Production modules live only under `src/fastplms`. Their direct Python
dependencies are declared under `requirements/`, but they may not import code
from `vendor`, alter `sys.path` to reach an official checkout, or download code
at import time. Importing runtime source must not create a tokenizer, initialize
a model, compile a kernel, log, change global Torch settings, or access the
network.

Official repositories live under `vendor/upstream` as real Git submodules.
Reference adapters call their public APIs and normalize outputs for comparison.
An adapter may not import FastPLMs, patch an upstream class, use a FastPLMs
loader, or reconstruct an official forward pass.

## Manifest-driven release data

`src/fastplms/models.toml` is the sole source of truth for supported models. Its
typed loader in `fastplms.registry` validates:

- immutable checkpoint and upstream revisions;
- file identities and explicitly unresolved release blockers;
- AutoClass mappings and tokenizer modes;
- state transformations and conversion records;
- attention, dtype, precision, dependency, VRAM, and test contracts;
- code and checkpoint licenses;
- reference containers and documentation state.

Tests derive model cases from this registry. Container validation checks that
every declared reference target exists. Documentation support tables and model
cards are generated from it. The artifact builder selects the declared source,
verifies every pinned input, applies the named transformation, and records the
same source record in the output.

Adding a model only in Python code is insufficient. A release-visible model
must have a complete manifest entry and pass all generated consistency checks.

## Loading and artifact flow

The same model contract is used from source conversion through downstream
inference:

1. the manifest selects an immutable checkpoint and pinned official source;
2. the artifact builder verifies checkpoint, tokenizer, source, and legal file
   identities;
3. the named state transformation produces canonical FastPLMs weights;
4. the builder writes a self-contained runtime bundle and generated model card;
5. Transformers loads the artifact through an advertised AutoClass with
   `trust_remote_code=True`;
6. model-specific preparation identifies biological residues or structure
   entities before shared APIs transform the output;
7. parity and artifact suites compare the same declared behavior against the
   isolated official reference.

This flow does not make the official checkout a runtime dependency. A local
artifact under `dist/hub/<model>` and a published copy use the same
Transformers interface.

## Runtime source

`fastplms.attention` owns backend names, mask construction, and explicit
dispatch. Models use Transformers' `attn_implementation` and
`set_attn_implementation()` contract. Mask builders produce the 4D masks used
by eager and SDPA, the packed 2D token masks used by declared precompiled
Hugging Face kernels, and Flex `BlockMask` objects. Flex functions and masks are
cached only after explicit use, keyed by device, dtype, execution shape, and
mask semantics rather than the exact row-length tuple. FastPLMs exposes bounded
cache cleanup without clearing process-global Torch compiler state. Original
padding masks must have exact `(batch, sequence)` shape before any backend
branch.

`fastplms.embeddings` owns ordered records, biological-residue masks, pooling,
persistence, and resume. Model-specific adapters only prepare the representation
and residue mask. E1 keeps its tokenizer-free raw-sequence adapter. ESMFold2
produces its learned width-256 representation through a dedicated mixin.

`fastplms.models` contains model-family implementations. Parameter names remain
compatible with existing checkpoints where possible. If a schema must change,
`models.toml` names a deterministic converter and the release suite compares the
converted key set, shape, dtype, and values exactly.

`fastplms.runtime` reports source and runtime capabilities without mutating
global state. Optional dependencies are imported only when their feature is
requested.

## Checkpoint and artifact boundary

Hub checkpoint files remain external assets pinned by immutable revision and
hash. `tools/artifacts/build.py` consumes an already downloaded snapshot and
never logs in, downloads, creates a repository, or uploads. It writes a local
artifact under `dist/hub/<model>/` with unchanged runtime source modules,
AutoClass metadata, tokenizer assets, legal files, source records, and deterministic
safetensors shards.

An artifact is valid only if it loads in a fresh offline environment with
FastPLMs absent from `sys.path`, `HF_HUB_OFFLINE=1`, `local_files_only=True`, and
`trust_remote_code=True`. Every advertised AutoClass must load, run, save,
reload, and match the repository-source implementation.

Runtime bundling uses tracked, clean regular files selected by path,
extension, and size allowlists. It rejects untracked inputs, symlinks,
credentials, unknown binaries, and path escapes. Release records separate weight
and runtime revisions, records source-tree and embedded-bundle digests plus
generator/schema version, and provides distinct complete-artifact and
runtime-only attestations. Publication rehashes validated bytes at preflight.

## Container boundary

`docker/Dockerfile` is a digest-pinned multi-stage build. Candidate stages use
the release validation stack. Reference stages install each upstream's native
environment and receive only the corresponding submodule and required legal
files. Runtime stages receive neither submodules nor checkpoint weights.

`docker/docker-bake.hcl` names build targets. `docker/compose.yaml` centralizes
GPU access, `ipc: host`, caches, source mounts, and output mounts.
`tools/remote/run.py` creates an isolated source archive, sends it to a host
specified at invocation time, runs Docker there, and returns JUnit, JSON, and
benchmark outputs. Hostnames, identities, and secrets are never tracked.

## Design rule

Use the shortest clear implementation that meets the exact behavioral contract.
Retain more complex code only when a repeatable benchmark shows a speed or
memory benefit and the strict compliance suite passes.
