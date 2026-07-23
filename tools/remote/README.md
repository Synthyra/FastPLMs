# Remote Hopper/SM90 runner

The runner archives only Git-tracked files plus tracked files from initialized,
pinned upstream submodules, copies them into a unique remote directory, builds
the requested Docker targets, executes the suite, and retrieves `artifacts/`.
Untracked and ignored files, common credential names, private-key extensions,
caches, and `.git` metadata are excluded.

Connection details are required at runtime and are never written into tracked
configuration:

```bash
python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/ssh-key \
  --accept-new-host-key \
  --suite check
```

Available suites include `check`, `gpu-golden-smoke`, `nightly`, `compliance`,
`structure`, `artifact`, `benchmark`, `benchmark-capture`, `release`, and
`python-matrix`, plus the focused `unit`, `integration`, and `feature` suites.
Remote source is removed after artifact retrieval unless `--keep-remote` is
passed. Persistent model and compiler caches are Docker volumes defined by
`docker/compose.yaml`, not part of the synchronized tree.

Every invocation writes `remote-run.json` beside the retrieved outputs. The
report records the source-archive digest, exact suite command graph, pre-build
host-hardware binding, phase durations and timeouts, normalized Docker
cache telemetry, built image IDs, structured no-download kernel availability,
artifact-retrieval result, cleanup result, and a digest over the retrieved
artifact tree. Reports are written atomically into a unique run directory. They
never record the SSH destination, identity path, raw subprocess output, command
exception text, or secret values.

The routine `check` suite builds only `candidate-structure`. It runs portable
units, imports, local integration and release checks, and compares candidates
with the checked-in sequence and structure goldens. It does not build local Hub
artifacts, download attention kernels, or build/import a live official
reference implementation. Artifact construction remains in `artifact`,
`nightly`, and `release`.
The parallel GitHub `CPU and package contracts` workflow is the normal required
pull-request gate. Its required CPU status is `cpu-contracts (3.12)`. The
workflow runs the separate hermetic CPU contract plus lint, bounded strict
typing, generated-document, license, model-card, wheel/sdist, runtime-import,
dependency-resolution, and clean-wheel jobs on Python 3.11 through 3.14. The
dependency-resolution matrix checks every public extra: `cpu`, `dev`, `structure`,
`binder`, `cueq`, `reporting`, `flash`, `fp8`, and `train`. It separately resolves
the published `structure` plus `binder` combination. The CUDA-only `cueq` and
`fp8` cases resolve against Linux CUDA 13 metadata without importing GPU-only
libraries or downloading and compiling attention kernels.

`gpu-golden-smoke` is the conditional Hopper/SM90 tier. The current release run
uses the exact containerized Linux aarch64 environment on the configured GH200,
builds only the structure candidate superset, and compares sequence plus
structure candidates with the checked-in, hash-validated goldens. H100 and H200
remain supported Hopper-class execution devices, but their results do not
substitute for this GH200 release evidence.
It never builds or imports an official reference implementation. Large cases
remain reserved for `nightly`.

`nightly` builds candidate, structure, FP8, and artifact images together. It
exercises the complete checkpoint golden panels, the eager/SDPA/Flex GH200
matrix, generation, TTT, PEFT, binder/structure flows, offline artifact loading,
FP8 reloads, and a descriptive family throughput report. It neither builds live
official references nor downloads/builds Flash kernels. FA2 remains separate
prior focused evidence; FA3 is explicitly unavailable in the current arm64
lock.

The `compliance` suite is the live-reference release-candidate tier. Its build,
reference, and test phases have explicit cancellation timeouts. It compares
every release-gated sequence checkpoint with its pinned official implementation
and runs the complete ESMFold and four-variant ESMFold2 folding gates, including
ESMFold2 FP8 validation. Boltz2 remains provisional and runs only in the focused
`structure`, `artifact`, and `benchmark` tiers.

The Biohub oracle has a platform-specific, fully pinned and hash-attested GH200
lock, including any source-built BioTraj wheel. Before source archiving or
Buildx, every remote suite records `uname -m`, normalized OCI architecture, GPU
name, UUID, driver, and total memory. Bake receives that exact native platform;
the runner rejects an image whose resolved platform or digest does not match the
preflight and also rejects hardware drift during the build. A GH200 therefore
runs `linux/arm64` images directly rather than emulated `linux/amd64` images.
This is exact GH200 evidence, not a claim that Docker erases ABI differences or
that an unvalidated architecture is interchangeable.

Immediately after image inspection and before any reference command, the runner
writes `artifacts/reference/environment/container-images.json` (mounted as
`/exchange/environment/container-images.json`). Schema version 1 contains the
resolved platform, stable Docker server and Buildx identities, and each Bake
target's content digest, image ID, OS, and architecture. It excludes tags,
creation timestamps, hostnames, and other ephemeral fields. Biohub suites also
bind the `biohub-biotraj-wheel` builder image identity.

The focused `structure` suite first produces isolated Meta ESMFold, Boltz2, and
Biohub ESMFold2 reference bundles, then produces candidate bundles from the same
immutable requests before running the metric gates. Reference containers contain
only their pinned upstream sources, the normalization protocol, and required
license notices.

The `feature` suite uses the BF16 structure candidate. It does not install the
FP8 extra because test-time training and all gradient-enabled paths are required
to remain BF16.

`benchmark` is intentionally gated: it requires the tracked immutable
`benchmarks/baselines/h100.json` and fails before remote work if that baseline is
absent. The filename is a legacy automation identifier; the current release
baseline must record the exact GH200 model, Linux aarch64 architecture, and
environment, and regression comparison requires an exact match. No baseline is
synthesized by the runner.
`benchmark-capture` produces an ungated, descriptive candidate report
containing separate cold compilation, first-forward, warmup, and steady-state
measurements. Review that report before adding a baseline in a separate change.
Full release benchmark and ESMC evidence must retain the same preflight hardware
identity as the candidate and official-reference measurements. The GH200 lock
makes that same-host contract available natively on `linux/arm64`; evidence from
another architecture or GPU UUID is not substituted or combined.

Every benchmark-producing invocation is self-contained. The focused benchmark
and capture suites use `tools.artifacts.build_all --benchmark-suite`; the
nightly throughput phase and aggregate `release` suite build their complete
artifact validation set. Each does so inside its own remote
workspace before invoking `benchmarks.suite --artifact-root dist/hub`. The
benchmark therefore loads registry-validated local artifacts and never assumes
that `dist/hub` from another GitHub matrix job or remote invocation is shared.
Official-source artifacts such as ANKH and DPLM2 are revalidated against the
current model registry after construction. The embedded nightly and aggregate
release reports remain descriptive; only the focused `benchmark` suite applies
the checked-in regression baseline.

Remote archives never carry `.git` metadata. Before upload, the runner records
the clean tracked root inventory with portable modes, sizes, symlink targets,
and content digests, then verifies the uploaded archive SHA-256 before
extraction. Artifact construction independently validates that inventory and
rejects missing, extra, linked, sensitive, oversized, or mutated runtime-scope
files. Because an extracted manifest cannot authenticate a Git commit object,
Git-free builds use `source-tree-sha256:<digest>` as `runtime_revision`; the
outer remote report separately records the clean source HEAD and archive
SHA-256. Clean Git worktrees continue to use the exact Git revision directly.

`.github/workflows/h100-validation.yml` maps the validation tiers without
claiming an accelerator is available on a hosted runner. The filename and
`h100-*` environment and secret names are legacy automation identifiers; the
current configured release remote is the GH200 Linux aarch64 workstation. It
uses an Ubuntu orchestrator and requires explicit host, SSH key, and known-host
secrets. Missing
configuration fails the job. A maintainer dispatches conditional PR validation
from the exact PR-head ref, supplies the same full 40-character SHA in the
required `revision` input, and selects `golden-smoke`. The workflow rejects a
revision that differs from `github.sha` or the checked-out `HEAD`, so the status
and immutable report are attached to the exact audited commit. For PRs touching
model, attention, generation, embedding, or structure execution paths, that
exact-SHA golden-smoke status and report are required pre-merge. The protected
`h100-validation` GitHub environment should require an authorized
reviewer and hold the manual-workflow workstation secrets. Scheduled nightly
runs use a separately restricted, non-interactive `h100-nightly` environment
with the same connection contract. No `pull_request` or `pull_request_target`
event receives SSH material. Manual `release-candidate` runs launch
`compliance`, `artifact`, `structure`, and baseline-gated `benchmark` suites for
the same checkout; the scheduled workflow runs `nightly`. The manual
`benchmark-capture` choice produces the descriptive report needed to propose
a baseline without treating it as a passing regression gate. All workflow runs
share one repository-wide Hopper/SM90 concurrency group, so nightly, release,
and manual validation queue rather than contending for accelerator memory.

## Python package-support matrix

Python 3.12 remains the canonical GPU validation environment. Every pull request
builds the wheel independently on Python 3.11, 3.12, 3.13, and 3.14, installs it
into a second clean CPU-only environment, imports all 37 advertised AutoClass
targets, and runs a finite tiny-model forward. The remote `python-matrix` suite
reuses the already-validated canonical 3.12 candidate and runs the non-canonical
3.11, 3.13, and 3.14 members concurrently. It builds one wheel, then installs
that exact wheel with Torch 2.13.0 and Transformers 5.13.0 into a separate
CPU-only environment for each interpreter:

```bash
python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/ssh-key \
  --suite python-matrix
```

Each environment installs the core wheel without optional extras. Its offline,
CPU-only smoke verifies the wheel RECORD, imports every
advertised runtime class, compiles the installed package, parses the model
registry, constructs a small ESM2 encoder, and runs one finite forward. The
suite writes `artifacts/python-matrix.json` and
`artifacts/junit/python-matrix.xml`. Raw installer output is represented only by
byte counts and SHA-256 digests, never copied into reports. A missing interpreter
wheel or dependency wheel is a package-support failure, not a skip.
