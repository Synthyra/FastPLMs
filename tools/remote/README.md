# Remote H100 runner

The runner archives the visible Git working tree plus initialized upstream
submodules, copies it into a unique remote directory, builds the requested
Docker targets, executes the suite, and retrieves `artifacts/`. Git-ignored
files, common credential names, private-key extensions, caches, and `.git`
metadata are excluded.

Connection details are required at runtime and are never written into tracked
configuration:

```bash
python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/ssh-key \
  --accept-new-host-key \
  --suite check
```

Available suites are `unit`, `integration`, `check`, `compliance`, `structure`,
`feature`, `artifact`, `benchmark`, `release`, and `python-matrix`. Remote source
is removed after artifact retrieval unless `--keep-remote` is passed. Persistent
model and compiler caches are Docker volumes defined by `docker/compose.yaml`,
not part of the synchronized tree.

Every invocation writes `remote-run.json` beside the retrieved outputs. The
report records the source-archive digest, exact suite command graph, timestamps,
artifact-retrieval result, cleanup result, and a normalized failure phase. It
never records the SSH destination, identity path, command exception text, or
secret values.

The `check` suite runs unit, integration, and release gates, validates every
local Hub artifact offline, and compares each manifest-declared representative
architecture with its live pinned official implementation. Official-generated
goldens cover the remaining checkpoint matrix during this routine gate.

The `compliance` suite has no runtime ceiling. It compares every release-gated
sequence checkpoint with its live pinned official implementation and also runs
the complete ESMFold and four-variant ESMFold2 folding gates, including ESMFold2
FP8 validation. Boltz2 remains provisional and runs only in the focused
`structure`, `artifact`, and `benchmark` tiers.

The focused `structure` suite first produces isolated Meta ESMFold, Boltz2, and
Biohub ESMFold2 reference bundles, then produces candidate bundles from the same
immutable requests before running the metric gates. Reference containers contain
only their pinned upstream sources, the normalization protocol, and required
license notices.

The `feature` suite uses the BF16 structure candidate. It does not install the
FP8 extra because test-time training and all gradient-enabled paths are required
to remain BF16.

## Python package-support matrix

Python 3.12 remains the canonical GPU validation environment. The
`python-matrix` suite reuses the candidate image to install uv-managed Python
3.11, 3.13, and 3.14 environments from `uv.lock`:

```bash
python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/ssh-key \
  --suite python-matrix
```

Each environment installs the core project non-editably without optional
extras. Its offline, CPU-only smoke compiles the installed package, parses the
model registry, constructs a small ESM2 encoder, and runs one finite forward.
The suite writes `artifacts/python-matrix.json` and
`artifacts/junit/python-matrix.xml`. A missing interpreter wheel or dependency
wheel is a package-support failure, not a skip.
