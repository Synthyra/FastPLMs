# Remote Hopper/SM90 runner

The runner archives Git-tracked files and tracked files from initialized, pinned
upstream submodules. It copies them to a unique remote directory, builds the
requested Docker targets, runs the suite, and retrieves `artifacts/`. It excludes
untracked and ignored files, common credential names, private-key extensions,
caches, and `.git` metadata.

Connection details are required at run time. They are never written to tracked
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
The remote source is removed after artifact retrieval unless you pass
`--keep-remote`. Persistent model and compiler caches are Docker volumes from
`docker/compose.yaml`. They are not part of the synchronized tree.

Each run writes `remote-run.json` beside retrieved outputs. The report records
the source archive digest, suite command graph, host hardware before build,
phase durations and timeouts, normalized Docker cache telemetry, built image
IDs, no-download kernel availability, artifact-retrieval and cleanup results,
and a digest of the retrieved artifact tree. Reports are written atomically to
a unique run directory. They do not record the SSH destination, identity path,
raw subprocess output, command exception text, or secret values.

The routine `check` suite builds only `candidate-structure`. It runs portable
unit tests, imports, local integration and release checks, and compares
candidates with checked-in sequence and structure goldens. It does not build
local Hub artifacts, download attention kernels, or build or import a live
official reference. Artifact construction is in `artifact`, `nightly`, and
`release`. The repository does not use GitHub Actions. Run CPU, source,
reference, and GPU validation on the workstation before merge or release.

`gpu-golden-smoke` is the conditional Hopper/SM90 tier. The current release run
uses the exact containerized Linux aarch64 environment on the configured GH200.
It builds only the structure candidate superset and compares sequence and
structure candidates with checked-in, hash-validated goldens. H100 and H200
are supported Hopper-class devices, but their results do not replace GH200
release evidence. The suite does not build or import an official reference.
Large cases remain in `nightly`.

`nightly` builds candidate, structure, FP8, and artifact images together. It
runs the complete checkpoint golden panels, the eager/SDPA/Flex GH200 matrix,
generation, TTT, PEFT, binder and structure flows, offline artifact loading,
FP8 reloads, and a descriptive family throughput report. It does not build live
official references or download or build Flash kernels. FA2 remains separate
prior focused evidence. FA3 is unavailable in the current arm64 lock.

The `compliance` suite is the live-reference release-candidate tier. Its build,
reference, and test phases have explicit cancellation timeouts. It compares each
release-gated sequence checkpoint with its pinned official implementation. It
runs the full ESMFold and four-variant ESMFold2 folding gates, including ESMFold2
FP8 validation. Boltz2 remains provisional. It runs only in the focused
`structure`, `artifact`, and `benchmark` tiers.

The Biohub oracle has a platform-specific, fully pinned, hash-attested GH200
lock, including any source-built BioTraj wheel. Before source archive or Buildx,
each remote suite records `uname -m`, normalized OCI architecture, GPU name,
UUID, driver, and total memory. Bake receives this native platform. The runner
rejects an image when its resolved platform or digest differs from preflight. It
also rejects hardware drift during the build. A GH200 runs `linux/arm64` images
directly, not emulated `linux/amd64` images. This is GH200-only evidence. Docker
does not remove ABI differences, and an unvalidated architecture is not equal.

After image inspection and before a reference command, the runner writes
`artifacts/reference/environment/container-images.json` (mounted at
`/exchange/environment/container-images.json`). Schema version 1 contains the
resolved platform, stable Docker server and Buildx identities, and the content
digest, image ID, OS, and architecture of each Bake target. It excludes tags,
creation times, host names, and other temporary fields. Biohub suites also bind
the `biohub-biotraj-wheel` builder-image identity.

The focused `structure` suite first creates isolated Meta ESMFold, Boltz2, and
Biohub ESMFold2 reference bundles. It then creates candidate bundles from the
same immutable requests and runs the metric gates. Reference containers contain
only pinned upstream source, the normalization protocol, and required license
notices.

The `feature` suite uses the BF16 structure candidate. It does not install the
FP8 dependency profile because test-time training and all gradient-enabled paths
must remain BF16.

`benchmark` is gated. It requires the tracked immutable
`benchmarks/baselines/h100.json` and fails before remote work when the baseline
is absent. The file name is a legacy automation identifier. The release baseline
must record the exact GH200 model, Linux aarch64 architecture, and environment.
Regression comparison requires an exact match. The runner does not create a
baseline. `benchmark-capture` makes an ungated, descriptive candidate report
with separate cold compilation, first-forward, warmup, and steady-state
measurements. Review this report before adding a baseline in another change.
Full release benchmark and ESMC evidence must use the same preflight hardware
identity as candidate and official-reference measurements. The GH200 lock
provides this same-host contract on `linux/arm64`. Do not substitute or combine
evidence from another architecture or GPU UUID.

Each benchmark run is self-contained. Focused benchmark and capture suites use
`tools.artifacts.build_all --benchmark-suite`. The nightly throughput phase and
aggregate `release` suite build their complete artifact validation set. Each
does this in its own remote workspace before it calls
`benchmarks.suite --artifact-root dist/hub`. The benchmark loads
registry-validated local artifacts. It does not assume that `dist/hub` is shared
with another GitHub matrix job or remote run. Official-source artifacts such as
ANKH and DPLM2 are checked again against the current model registry after build.
Embedded nightly and aggregate release reports remain descriptive. Only the
focused `benchmark` suite uses the checked-in regression baseline.

Remote archives do not include `.git` metadata. Before upload, the runner
records the clean tracked-root inventory with portable modes, sizes, symlink
targets, and content digests. It verifies the archive SHA-256 before extraction.
Artifact construction validates this inventory again. It rejects missing, extra,
linked, sensitive, oversized, or changed runtime files. An extracted manifest
cannot authenticate a Git commit. Therefore, Git-free builds use
`source-tree-sha256:<digest>` as `runtime_revision`. The outer remote report
records the clean source HEAD and archive SHA-256. Clean Git worktrees use the
exact Git revision.

Run validation tiers with `python -m tools.remote` on the GH200 Linux aarch64
workstation. Bind release evidence to the exact candidate revision. Run only one
accelerator-heavy suite at one time.

## Python source-support matrix

Python 3.12 is the canonical GPU validation environment. Before release, run
the remote `python-matrix` suite for Python 3.11, 3.13, and 3.14. It creates a
separate CPU-only environment for each interpreter. It installs
`requirements/profiles/runtime.in` with the validation constraints:

```bash
python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/ssh-key \
  --suite python-matrix
```

Each environment imports from the explicit repository source root. Its offline,
CPU-only smoke imports each advertised runtime class, compiles runtime source,
parses the model registry, creates a small ESM2 encoder, and runs one finite
forward. The suite writes `artifacts/python-matrix.json` and
`artifacts/junit/python-matrix.xml`. Reports represent raw installer output only
by byte counts and SHA-256 digests. They never copy this output. A missing
interpreter or dependency wheel is a source-support failure, not a skip.
