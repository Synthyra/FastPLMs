# Testing and compliance

FastPLMs treats official equivalence as a release property. Routine goldens make
development faster, but a release requires live comparison with the pinned
official implementation in its native reference container.

GPU release and benchmark suites run through repository Docker images on the
current exact NVIDIA GH200/aarch64 workstation. H100 and H200 remain
Hopper-class deployment examples, but they are not current release-confirmation
evidence. Portable unit and documentation checks can also run locally. Every
Dockerized PyTorch run uses `--ipc=host` directly or receives `ipc: host` from
Compose. Remote preflight records the GH200 GPU UUID and native OCI platform,
then rejects platform or device drift between preflight and the loaded images.

## Run tiers

| Tier | Purpose |
| --- | --- |
| `cpu_contract` | Required offline CPU confidence gate with tiny models, no checkpoints, network, Docker, skips, or xfails |
| `check` | Candidate-only units, imports, local integration, release checks, and immutable checkpoint goldens; no artifacts, live references, or kernel downloads |
| `gpu-golden-smoke` | Conditional exact-device comparison with checked-in sequence and structure goldens; no live reference build |
| `compliance` | Every checkpoint whose manifest declares the release compliance tier against its live pinned official implementation |
| `structure` | ESMFold, four ESMFold2 variants, provisional Boltz2 diagnostics, feature preparation, export, and seeded stochastic output |
| `feature` | DPLM generation, DPLM2 generation, ESM3 multimodal generation, TTT, E1 sequence and RAG adapters, binder flow, pooling, and conversion |
| `artifact` | Fresh offline remote-code loading and save-reload for every local artifact |
| `benchmark` | Separate GH200/aarch64 latency, throughput, padding, memory, and exact-device regression suite |
| `python-matrix` | Clean-wheel, non-editable core-package smokes on Python 3.11-3.14 |
| `extras-matrix` | Clean resolution-only checks for every public extra against the exact validation contract on Linux x86-64; cuEquivariance and FP8 resolve against CUDA 13.0 |

`extras-matrix` is a package-resolution metadata check only. Its x86-64 runner
does not provide GPU compliance, parity, or benchmark evidence. Current GPU
release confirmation remains restricted to the exact GH200/aarch64 target.

Routine `check` consumes goldens and never builds an official reference image.
Live references are reserved for the frozen exact-head `compliance` release
candidate. Missing expected
dependencies, checkpoints, reference containers, or backends are failures, not
skips.

Boltz2 is intentionally outside the FastPLMs 1.0 `check` and `compliance`
claims while its native-environment BF16 numerical gap remains under
investigation. Its exact state/configuration, feature, seeded-execution,
artifact, and benchmark diagnostics remain in the focused tiers. This is an
explicit provisional boundary, not a relaxed tolerance or silent skip.

## Required offline CPU gate

Every pull request runs this positive allowlist as a required GitHub status:

```bash
python -m pytest tests/cpu -m cpu_contract -n auto --dist=loadscope \
  --durations=25 --junitxml=artifacts/junit/cpu-contract.xml
```

The protected required status is `cpu-contracts (3.12)` from the
`CPU and package contracts` workflow. Branch protection should name that exact
status so a similarly named job cannot satisfy the gate.

The job uses Python 3.12, CPU-only Torch 2.13, Transformers 5.13, four hosted
cores, fixed seeds and thread counts, hidden CUDA, and empty temporary caches.
It sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, guards sockets and Hub
download functions, and fails on a skip or xfail. Tiny models use one layer,
hidden width 8-16, two heads, and a mixed-padding batch of two. The suite budget
is under five minutes and 4 GiB, with an approximately ten-second per-test cap.
The controller samples its complete live Linux process tree from `/proc` every
50 ms and deduplicates PIDs before summing concurrent memory. Complete
`smaps_rollup` proportional-set-size evidence enforces the 4 GiB gate; if any
live process lacks PSS, the gate fails closed to the larger concurrent RSS sum
and records the reason. Per-worker `RUSAGE_SELF` and `RUSAGE_CHILDREN` maxima
remain in telemetry as a temporal upper-bound diagnostic, not as the budget
metric, because those maxima can occur at different times.
Before installation, CI runs `uv lock --check`. It then consumes that exact
lock with `uv sync --frozen`, the `validation` dependency group, and the
explicit `cpu`, `dev`, `structure`, and `train` extras. The `cpu` extra routes
Torch to the locked PyTorch CPU index; `UV_TORCH_BACKEND` does not affect
project-level `uv sync`. CUDA-only `cueq` and `fp8` extras conflict with the
CPU extra so `--all-extras` fails closed instead of silently selecting CPU
Torch for a GPU environment. A stale lock or an environment that would require
fresh dependency resolution fails before pytest starts.

The gate statically covers all 29 checkpoints and executes every advertised
AutoClass once per family, including forward/loss/backward, resize, tuple and
dictionary output, and save/reload. It also covers ANKH stack selection and
state views; backend masks, fallback warnings, fake Flash dispatch, E1 cache and
ANKH concurrency; ESMC diagnostics; sequence-family embeddings and persistence;
bounded disk-spooled generator and FASTA streaming; generation and TTT; PEFT;
injected-core structure and binder flows; publication security; and curated
offline documentation examples.

Lint, typing, generated docs, licenses, model cards, wheel/sdist inspection, and
runtime import closure run in parallel. Clean-wheel smoke jobs cover Python
3.11, 3.12, 3.13, and 3.14, while a clean Python 3.12 source-distribution smoke
guards the independent build and install path. A separate clean resolution-only
matrix verifies every declared public extra: `cpu`, `dev`, `structure`, `binder`,
`cueq`, `reporting`, `flash`, `fp8`, and `train`. It also resolves the published
`structure` plus `binder` workflow as one environment. Every case includes the
exact Torch 2.13.0 and Transformers 5.13.0 validation group without importing GPU
runtimes; the cuEquivariance and FP8 cases use the CUDA 13.0 dependency index.
The runtime-import closure separately requires import-time dependencies in core
source to be core requirements. Manifest-scoped structure modules map to the
`structure` extra, while deferred function or lambda imports and imports protected
by an explicit `ImportError` handler must resolve to one unambiguous intended
extra.

## Cost-controlled schedule

- Every PR: offline CPU contracts and static/package checks only.
- Conditional GH200/aarch64 PR smoke: when a relevant sequence or structure path
  changes, an authorized maintainer dispatches `golden-smoke` from the exact PR
  head and supplies that same full SHA. The protected `h100-validation`
  environment releases SSH material only after approval. Candidate output is
  compared with checked-in goldens; no reference image is built. Its exact-SHA
  status and report are required pre-merge for those relevant changes.
- Nightly: sharded real-checkpoint goldens, eager/SDPA/Flex execution,
  generation, PEFT, structure, artifacts, FP8, and throughput by family. The
  GH200 job does not download, build, or execute FA2/FA3 kernels; FA2 retains
  separate prior focused evidence and FA3 is explicitly unavailable in the
  current linux/arm64 lock.
- Release candidate: every live pinned reference, checkpoint/state/tokenizer
  contract, published artifact, structure panel, and benchmark on one frozen
  exact head.

GPU phases enforce explicit timeouts and cancellation, group all AutoClasses
for one checkpoint in one isolated process, build independent Buildx targets in
parallel, and publish JUnit, duration, cache, environment, and immutable report
telemetry. GH200/aarch64 benchmarking records cold compilation separately from warm
throughput; a missing baseline remains a release blocker rather than a synthetic
placeholder. Scheduled runs use the separately restricted `h100-nightly`
environment so maintainer approval remains mandatory for PR and manual release
dispatches without blocking the unattended nightly tier.

Every remote report contains a structured kernel-capability record. It names
the measured eager, SDPA, and Flex backends, identifies the pinned FA2 revision
as prior-focused-evidence-only, identifies FA3 as unavailable on linux/arm64,
and records that network downloads and source builds were disabled. Asking the
GH200 runner to execute either Flash backend fails before source archiving.

## Frozen ESMC release evidence

The ESMC compliance run writes one schema-v3 JSON record for every combination
of three checkpoints, five BF16 attention backends, and two immutable sequence
panels. A release set therefore contains exactly 30 records under one explicit
directory: 18 measured eager, SDPA, and Flex records, plus 12 structured
FlashAttention 2 and 3 locked-platform unavailable records. All 30 records must
come from the current exact GH200/aarch64
accelerator, repository container images, dependency lock, installed inventory,
and official source attestations. H100 and H200 remain supported Hopper-class
examples, but their measurements are not current release evidence and are not
combined with GH200 results. Candidate and official-reference measurements must
carry the same preflight hardware identity.

Default documentation generation does not inspect the environment or discover
reports. It deliberately renders ESMC measurements as pending. On the frozen
release head, select the completed report directory explicitly:

```bash
PYTHONPATH=src python -m tools.artifacts.generate_docs \
  --source-root . \
  --esmc-report-root artifacts/diagnostics/esmc

PYTHONPATH=src python -m tools.artifacts.generate_docs \
  --source-root . \
  --esmc-report-root artifacts/diagnostics/esmc \
  --check
```

The release-evidence form may instead read
`FASTPLMS_DIAGNOSTIC_REPORTS`, falling back to the same repository-relative
location:

```bash
FASTPLMS_DIAGNOSTIC_REPORTS=/validated/esmc-reports \
PYTHONPATH=src python -m tools.artifacts.generate_docs \
  --source-root . \
  --require-esmc-release-evidence \
  --check
```

Supplying either evidence option is fail closed. Missing or extra files,
duplicate JSON keys, non-finite values, malformed schema, a stale manifest,
checkpoint, weights, source-tree, runtime, kernel, dtype, backend, or panel
identity, either missing or stale official-reference source attestation, an invalid
canonical self-digest, a failed measured-record catastrophe gate, a false Flash
execution claim, and any cross-device record set all stop generation. Successful ingestion records the exact device,
runtime and source identities, official import revision/tree/attestation,
aggregate ranges, and per-case minimum/median/maximum distributions in the
generated capability manifest and applicable checkpoint cards. It never
fabricates a value from a threshold or copies a measurement from another
checkpoint. Flash records contain no numerical metrics. They attest that the
current GH200/aarch64 lock fails closed without dispatch. Prior real
FlashAttention 2 execution was captured in separate workstation JUnit, but its
immutable report and environment attestation are not bundled in this
repository. It is not a current release distribution or numerical claim.

## Portable remote execution

The runner accepts connection data at invocation time, synchronizes an isolated
workspace, preserves external model and container caches, and retrieves test and
benchmark outputs:

```bash
python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/private-key \
  --suite check
```

Workstation names, identity paths, cache credentials, and secrets are not stored
in tracked files. The runner archive excludes Git metadata, known credential
names, private-key extensions, and ignored files. Before synchronization, every
initialized upstream must have a clean tracked tree and a `HEAD` equal to its
parent Git link. The runner records those revisions, the exact tracked-file
inventory, and a tree digest in `.fastplms-source-provenance.json`. Artifact
validation uses this record only in an extracted tree with no Git metadata and
rejects missing, added, or modified upstream files.

Python 3.12 is the canonical GPU validation interpreter. Package compatibility
for Python 3.11, 3.13, and 3.14 is checked separately on the same workstation:

```bash
python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/private-key \
  --suite python-matrix
```

The matrix reuses the candidate image and uv-managed interpreters. For each
version it performs a frozen, core-only, non-editable install, removes the
repository source from the import path, enables offline Hub behavior, disables
CUDA visibility, compiles the installed package, loads `models.toml`, and runs
a small ESM2 CPU forward. It does not select the `flash` extra. Results are
recorded in JSON and JUnit. Python 3.12 remains the only environment used for
the pinned CUDA 13.0, PyTorch 2.13.0, and Transformers 5.13.0 GPU release gates.

For direct execution in an already synchronized checkout:

```bash
sudo docker buildx bake -f docker/docker-bake.hcl candidate-structure --load
sudo docker compose -f docker/compose.yaml run --rm structure \
  python -m pytest tests/unit tests/integration tests/release \
  -m "not gpu and not slow and not structure and not artifact" -v
```

## Manifest-generated cases

`tests/conftest.py` loads `src/fastplms/models.toml`. Each checkpoint contributes
its AutoClasses, tokenizer mode, source revisions, state transform, reference
container, dependencies, backends, dtypes, precision paths, and declared test
tiers. Release tests fail when the manifest, Docker targets, artifact metadata,
support tables, or model cards diverge.

Any manifest `unresolved_files` entry is an explicit release blocker. A test may
report it precisely, but must not guess a hash or silently omit the asset.

## Reference isolation

Each official adapter runs in a reference stage built from its pinned submodule
and native dependency set. The adapter may call official public APIs and
normalize output names and layout. It may not:

- import `fastplms`;
- add the FastPLMs source tree to `sys.path`;
- patch official classes;
- reuse a FastPLMs tokenizer or checkpoint loader;
- duplicate an official forward pass in test code.

Candidate code runs in a separate container. Comparisons exchange serialized
inputs and outputs, not live Python objects.

The Biohub ESM oracle does not resolve the upstream package's mutable
`transformers @ ... @main` dependency. Its reference image derives only the
remaining dependency set, installs ESM without dependency resolution, and
force-installs the manifest-pinned Biohub Transformers checkout last. Both the
image build and every ESMC, ESM3, or ESMFold2 adapter or standalone loader run
verify the exact source revision, complete tracked-tree inventory and digest,
pre-import origin, and package version before the official Transformers API is
imported, then rehash after import. The locally built, non-editable wheel owns
distribution metadata and dependency validation; imports execute from the
separately attested source tree. Native `metadata.json` and ESMFold2 reference
bundles preserve the exact versioned `reference_sources` mapping for both
Biohub ESM and the pinned Biohub Transformers checkout, including each
attestation-file hash and source-relative import identity.

The GH200/SM90 ESMFold reference image makes one documented build-only modification:
`docker/constraints/openfold-sm90.patch` restricts the copied OpenFold
`setup.py` extension architecture list to `sm90`. BuildKit cannot discover the
host GPU, and the upstream fallback list contains architectures rejected by
CUDA 12.1. The patch does not change the pinned submodule, extension source,
model classes, checkpoint data, or the public ESMFold API. Its modified-file
notice is `LICENSES/openfold/MODIFICATIONS.md`.

The same native stage pins PyTorch Lightning `1.9.5`, TorchMetrics `0.11.4`,
Lightning Utilities `0.15.2`, and NVIDIA DLLogger revision
`0478734ff7be75adde8d160e04872664d1c62e5f`. Pinned OpenFold imports those
packages eagerly; they are reference-container dependencies and are excluded
from FastPLMs runtime images and package extras.

## Exact contracts

Every release-gated checkpoint declaring `compliance` must establish:

- exact semantic configuration equality, excluding only declared packaging
  fields;
- exact tokenizer assets, vocabulary, special IDs, normalization, and behavior
  for canonical, ambiguous, lowercase, whitespace, empty, truncated, and padded
  inputs;
- exact state-key sets, tensor shapes, dtypes, and `torch.equal` values after the
  declared transform;
- exact tied-weight and parameter-alias contracts;
- one live BF16 mixed-length inference;
- representative deep FP32 and BF16 comparisons across all layers, public
  outputs, embeddings, skewed padding, and required attention backends.

DPLM2 specifically asserts that input and output embeddings are not aliased and
that the trained `esm.contact_head` keys are present. ANKH covers the official
encoder and sequence-to-sequence heads separately; the named masked-LM extension
is tested as an extension, not as official parity.

## Numerical metrics

Metrics include only valid biological positions. Padding and special positions
cannot improve a score. The suite reports relative L2 error, relative
99.9th-percentile error, first-percentile residue cosine, per-sequence pooled
cosine, confident-position top-1 agreement, and Jensen-Shannon divergence for
probability tensors.

The tables below record the fixed engineering targets and hard limits enforced
by the release tests. Repeated runs and official baselines are diagnostic
evidence, but they do not silently widen those limits. A new implementation
must meet the engineering target, not only the hard limit.

| Contract | Engineering target | Hard limit |
| --- | ---: | ---: |
| FP32 official relative L2 | `2e-6` | `2e-5` |
| FP32 relative Q99.9 error | `1e-5` | `1e-4` |
| BF16 official or backend relative L2, except scoped rows below | `1e-2` | `3e-2` |
| ESM2 optimized-backend BF16 relative L2 | `2e-2` | `3e-2` |
| ESMC eager BF16 relative L2 | `2.9e-2` | `3e-2` |
| BF16 relative Q99.9 error | `2.5e-2` | `5e-2` |
| ESMC eager relative Q99.9 error | `4.9e-2` | `5e-2` |
| BF16 residue cosine, first percentile | `>=0.999` | `>=0.995` |
| ESMC eager residue cosine, first percentile | `>=0.997` | `>=0.995` |
| BF16 pooled cosine, every sequence | `>=0.9995` | `>=0.995` |
| BF16 confident top-1 agreement | `>=99.5%` | `>=99.0%` |
| BF16 Jensen-Shannon divergence | `1e-4` | `1e-3` |
| ESMC eager Jensen-Shannon divergence | `4e-4` | `1e-3` |

The ESM2 row applies to SDPA, Flex Attention, FlashAttention 2, and
FlashAttention 3; eager retains the global contract. The current ESMC row
applies to eager. SDPA remains bit-for-bit exact; Q99.9, pooled-cosine, and
top-1 thresholds remain global. ESMC Flex Attention is supported,
non-experimental, and measured as a diagnostic path. Crossing a published Flex
band emits a warning and writes JSON under `artifacts/diagnostics/esmc/`; it is
not an xfail or release failure. The old FlashAttention 2 threshold is retained
only with its separate historical focused evidence and is not a current GH200
acceptance claim.

Measured Flex still fails on broken dispatch, non-finite output, invalid mask
or shape, and these corruption/catastrophe limits:

| ESMC diagnostic | Catastrophe limit |
| --- | ---: |
| Relative L2 | `<=0.25` |
| Relative Q99.9 | `<=0.50` |
| Residue cosine, first percentile | `>=0.90` |
| Pooled cosine | `>=0.95` |
| Confident-position top-1 | `>=0.80` |
| Jensen-Shannon divergence | `<=0.05` |

These broad limits detect corruption and do not claim parity or biological
quality. Each measured release record must publish full distributions for the
exact checkpoint, backend, dtype, hardware, and locked sequence panel. The
FlashAttention 2 and 3 records contain structured current-platform
unavailability instead. Both remain supported, non-experimental interfaces;
the current locked GH200/aarch64 image raises before dispatch. Model-card cells
remain pending until frozen-head, exact-device GH200/aarch64 evidence exists.

The pinned ESM2-3B SDPA BF16 path has a checkpoint-specific calibration:
relative L2 target/hard limit `0.06`/`0.07`, relative Q99.9 `0.15`/`0.18`,
first-percentile residue cosine `0.994`/`0.992`, and pooled cosine
`0.998`/`0.997`. Its confident top-1 and Jensen-Shannon thresholds remain
global. Exact state identity and perfect confident-token agreement still gate
this checkpoint.

ESMFold2 FP8 is experimental and is not a release numerical-parity gate. Its
smoke coverage on the locked, exact-device GH200/aarch64 stack verifies explicit opt-in, finite
outputs, exactly 80 converted ESMC attention output projections, transient
runtime state, and strict failure when unavailable. `auto` always resolves to
BF16 so model behavior does not change with hardware or optional dependencies.

## ESMFold2 projection and structure

Projection from identical ordered ESMC states is exact in FP32. The BF16
relative L2 target is `5e-4`, with a hard limit of `1e-3`. Experimental FP8
smoke runs once on each of the four variants and performs three fresh
BF16-to-FP8 reload cycles only on the standard variant.

Folding tests hash prepared features and sampled diffusion noise. They require
exact discrete features and masks, valid geometry, and no NaNs. Coordinate and
confidence thresholds are documented in [ESMFold2](esmfold2.md) and encoded once
in the strict metric module. The pinned five-protein, three-seed, four-variant
panel found exact official-versus-candidate BF16 parity in all 60 cases. A
prior FP8 diagnostic passed its historical structure limits in 48 of 60 cases;
that result is retained as evidence, not as a release gate or equivalence claim.

## Goldens

Official-generated goldens use safetensors. Each includes the official source
revision, checkpoint revision and hashes, environment fingerprint, deterministic
generation command, input fingerprint, tensor names and shapes, dtypes, and
output hashes. Goldens are read-only fixtures. They accelerate `check`, but they
never replace live `compliance`.

The manifest declares a required golden only through an `official_golden`
record on a model entry. Both files are SHA-256 pinned and use fixed paths:

```toml
official_golden = { metadata = "tests/goldens/<model>.json=sha256:<digest>", tensors = "tests/goldens/<model>.safetensors=sha256:<digest>" }
```

Absence of this record means that the checkpoint golden is not complete. It
must be reported as release work rather than represented by a placeholder or a
synthetic fixture. Presence makes a missing, modified, or
source-record-inconsistent bundle a `check` failure. Other tiers do not infer a
golden requirement.

The manifest-driven converter consumes only normalized output from an isolated
native reference container. It does not load a model, import an upstream
package, or download a checkpoint:

```bash
python -m tools.goldens \
  --native-root artifacts/reference \
  --output-root tests/goldens \
  --model esm2_8m

python -m tools.goldens \
  --status-only \
  --report-matrix \
  --native-root artifacts/reference \
  --output-root tests/goldens
```

The matrix is the manifest-wide source of truth for all `check` checkpoints. It
reports each native request, reference container, normalized native result,
compact bundle, and declaration state. A targeted native sequence run accepts
one or more explicit model IDs:

```bash
python -m tests.parity.support.native_reference \
  --request-dir artifacts/reference/requests/reference-esm2 \
  --output-dir artifacts/reference/results \
  --model esm2_8m
```

An official public-generation limitation is not generation parity. Only a
checkpoint whose manifest capability is `official_unavailable` may carry a
normalized limitation record, and that record must match the public method,
exception type, and semantic reason exactly. All `required` DPLM and DPLM2
checkpoints fail when generation output is absent. Feature tests retain viable
family representatives even when one checkpoint's pinned official sampler is
unusable.

For sequence models, the input is `metadata.json` plus `bf16.safetensors`. The
converter retains token inputs, the biological-residue mask, the final hidden
state, and logits when the official head returns them. For structure models,
the input is the official `metadata.json` plus `bundle.safetensors`. In both
cases it validates the model ID, checkpoint revision and file identities,
reference environment, normalized tensor contract, and source-result hashes.
It rejects candidate-produced structure bundles and legacy native results that
do not carry an environment record.

The output is a compact safetensors file and JSON sidecar. The sidecar records
upstream revisions, official checkpoint file identities, the native environment
fingerprint, canonical generation command, deterministic input fingerprint,
source-result file hashes, tensor hashes, and the output tensor-file hash. The
converter prints a TOML declaration only when output is written to the canonical
`tests/goldens` directory. It never edits `models.toml`; a reviewer adds the
printed declaration only after validating both generated files. The read-only
validator then verifies every recorded identity, shape, dtype, and hash.

The sequence regression resolves the current package class from the manifest
`auto_map` and loads only the pinned checkpoint weights. Generated remote-code
artifacts have their own offline suite. This separation prevents stale Hub code
from substituting for the package implementation under test.

The pinned Biohub ESMC loader has a standalone reproducer so a native-loader
failure cannot be mistaken for a FastPLMs inference failure:

```bash
python -m tests.parity.support.reference_adapters.biohub_loader_reproducer \
  --repo-id biohub/ESMC-300M \
  --revision a59b831785f907e96e6a246b1d142bfb76df31ee
```

Run it inside `reference-biohub-esm`. It prints the native package versions and
then invokes the pinned public `ESMC.from_pretrained` method unchanged. The
declared ESMC goldens were generated through that native path; the suite does
not patch the loader or replace it with a FastPLMs path.

## Artifacts

Artifact tests build from a pinned local checkpoint snapshot. They create a
fresh environment with FastPLMs absent from `sys.path`, set
`HF_HUB_OFFLINE=1`, pass `local_files_only=True` and `trust_remote_code=True`,
load every advertised AutoClass, run inference, save, reload, and compare with
the package-source implementation. Network access during this tier is a test
failure.

## Test markers

The suite uses `gpu`, `slow`, `large`, and `structure` markers to describe
resource needs. Markers do not authorize skipping a required release case. The
runner chooses the appropriate image and tier, then treats an unmet declared
requirement as a failure.
