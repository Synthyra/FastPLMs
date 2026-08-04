# Local Hub artifacts

`tools/artifacts/build.py` creates deterministic, offline-loadable Hugging Face
artifacts under `dist/hub/<model>/`. It operates only on an already downloaded,
manifest-pinned checkpoint snapshot. It never authenticates, downloads, creates
a Hub repository, uploads, deletes, commits, pushes, or opens a pull request.

## Dependencies

Artifact tooling uses Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13.
From a normal checkout, without official parity submodules, install the
artifact profile before building:

```bash
uv venv
uv pip install \
  -r requirements/profiles/artifact.in \
  -c requirements/constraints/validation.txt
```

Building an artifact is offline and does not require a GPU. Live compliance is
a separate workflow and is the only stage that requires the official reference
submodules.

## Build

```bash
PYTHONPATH=src python -m tools.artifacts.build \
  esm2_8m \
  /cache/hub/models--Synthyra--ESM2-8M/snapshots/<revision> \
  --tokenizer-dir \
  /cache/hub/models--facebook--esm2_t6_8M_UR50D/snapshots/<official-revision> \
  --output-root dist/hub
```

Tokenizer-mode artifacts built from a FastPLMs checkpoint require
`--tokenizer-dir`. This option must point to the manifest-pinned official
snapshot. The builder copies and records only tokenizer files declared by the
manifest. Artifacts that select the official snapshot can omit this option.

Before writing output, the builder validates:

- the model and every required file identity are resolved in `models.toml`;
- the local snapshot matches the declared immutable revision and file hashes;
- every required upstream submodule is initialized at its pinned commit;
- canonical and distributable license files match their declared hashes;
- the family has a complete mechanism-first conversion record.
- every scoped runtime source is a tracked, clean regular file selected by an
  extension, path, and size allowlist;
- no scoped input is an untracked file, symlink, credential-shaped path,
  unknown binary, or Windows path escape.

An unresolved file or hash mismatch stops the build. The builder does not infer
an identity from a similarly named checkpoint.

## Output

Each artifact contains:

```text
config.json
model-00001-of-000NN.safetensors
model.safetensors.index.json
tokenizer assets, when applicable
modeling_fastplms.py
fastplms_bundle.py
fastplms/...
README.md
source-record.json
artifact-manifest.json
LICENSES/...
THIRD_PARTY_NOTICES.md
```

The builder copies normal runtime source modules unchanged under `fastplms/`.
It also writes the same bytes to a deterministic compressed archive in
`fastplms_bundle.py`. This flat bundle is required because the Transformers
remote-module loader uses flat relative Python imports. It does not import the
copied package tree directly.

`modeling_fastplms.py` imports the flat bundle and verifies its SHA-256 identity
and canonical embedded file inventory. It rejects unsafe or repeated paths,
non-regular archive entries, bytecode, encryption, unexpected compression, and
non-canonical modes before extraction. The bridge extracts with exclusive file
creation into a loader-owned private `TemporaryDirectory`. It then
re-hashes the exact extracted inventory. It
rejects symlinks, bytecode, non-file entries, or any missing, added, or changed
file. Imports run with bytecode writing disabled.
Nothing is extracted into or trusted from the Transformers module cache; only
an ordinary writable temporary directory is required. This step performs no
network access or compilation.

Release validation runs each artifact in a fresh interpreter. Complementary
family bundles from the same FastPLMs release can extend the runtime loaded by
an earlier artifact in one interpreter. Runtime files shared by two bundles
must have identical hashes; an overlapping source conflict fails explicitly
and leaves the first runtime intact. Isolate incompatible releases in separate
Python processes. Production code never imports a submodule checkout.

The release artifact tier selects the checkpoint source declared by the
manifest. This matters for ANKH, where the artifact uses the official
sequence-to-sequence checkpoint so the official decoder and LM head are
present. The named state transform is deterministic.
Weights are written as explicit safetensors shards no larger than 5 GiB with a
sorted index. Trusted legacy `.bin` input is loaded with `weights_only=True` and
is never copied into the output.

The source record separates `weights_revision` from `runtime_revision` and
records source-tree and runtime-bundle SHA-256 digests, generator/schema
version, scope-specific complete and runtime-only attestations, both checkpoint
identities, conversion record, BF16 execution policy, upstream revisions,
runtime assets, legal files, `weights_license_status`, and `redistributable`.
The runtime attestation repeats the two license fields. Every currently
publishable family, including DPLM1 and DPLM2, uses `"resolved"` and `true`.
Synthetic unresolved-license tests retain the fail-closed `"unresolved"` and
`false` path. The model card states the same policy; artifact validation rejects
a disagreement. `artifact-manifest.json` contains a SHA-256 identity for every
artifact file. Upload preflight rehashes selected bytes so a post-validation
mutation cannot cross the time-of-check/time-of-use boundary.

The generated `config.json` also records packaging-only FastPLMs model ID,
selected checkpoint repository and immutable revision, and a deterministic hash
of the checkpoint file identities. Offline embedding metadata uses these fields
when no Hub commit is available. Semantic configuration parity explicitly
removes them.

## Validate

```bash
PYTHONPATH=src python -m tools.artifacts.build \
  esm2_8m /cache/fast-snapshot --tokenizer-dir /cache/official-snapshot \
  --output-root dist/hub --replace
```

The command validates the completed content manifest. The release artifact tier
then creates a fresh environment where FastPLMs is absent from `sys.path` and
sets:

```text
HF_HUB_OFFLINE=1
local_files_only=True
trust_remote_code=True
```

Load the built artifact through the same Transformers API used after
publication:

```python
from transformers import AutoModel

artifact_path = "dist/hub/ESM2-8M"
model = AutoModel.from_pretrained(
    artifact_path,
    local_files_only=True,
    trust_remote_code=True,
)
```

After publication, replace `artifact_path` with the Hub repository ID and pass
the immutable revision of that published FastPLMs 1.0 artifact. The source
checkpoint revision in `models.toml` identifies the input weights; it must not
be reused as the revision of newly generated remote code.

It loads every advertised AutoClass, performs inference, saves, reloads, and
compares configuration, state, and output against repository source. Network access,
an undeclared import, a missing legal text, or a missing conversion record fails
the tier.

## Publish to Hugging Face

Compile and upload source-backed files for every manifest model without
touching its checkpoint weights:

```bash
py -m tools.artifacts.publish --files-only
```

The command reads each family's `runtime_paths` from `models.toml`, packages
those files under `fastplms/`, generates `fastplms_bundle.py` and
`modeling_fastplms.py`, and adds the current model card, requirements, notices,
and license files. Model IDs are optional; omitting them publishes every model.
Use `--dry-run` to print the generated paths without committing them.

Without `--files-only`, files from the prepared local artifact are included in
the same commit, including configuration, tokenizer assets, and weights:

```bash
py -m tools.artifacts.publish esm2_8m
```

Prepared artifacts default to `dist/hub/<repository>`. If one is missing, build
it first with `PYTHONPATH=src python -m tools.artifacts.build_all <model-id> --replace`.
Authentication comes from `HF_TOKEN` or the cached Hugging Face login.

DPLM1 and DPLM2 weights are distributed under Apache-2.0. At the pinned
ByteDance revision, the
[LICENSE](https://github.com/bytedance/dplm/blob/main/LICENSE)
is Apache-2.0 and the [README](https://github.com/bytedance/dplm/blob/main/README.md#overview)
defines the repository release as including pretrained DPLM1 and DPLM2 weights.
Built artifacts carry Hub license `apache-2.0`,
`weights_license_status="resolved"`, and `redistributable=true`, plus the
verbatim license and `LICENSES/dplm/SOURCE_RECORD.md`.

## ESMFold2 runtime asset

The ESMFold2 source record includes `ccd.pkl` from
`biohub/ESMFold2`: 417,306,584 bytes,
SHA-256 `9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5`,
MIT license, and trust kind `hash_pinned_pickle`. Deserialization occurs only
from a private loader-owned temporary snapshot after verifying that snapshot's
size and SHA-256. User-supplied asset and `cache_dir` symlinks are rejected. The
exact manifest repository/revision Hugging Face snapshot link is the sole
exception and must resolve within that repository's contained blob directory.
This prevents path replacement and in-place source mutation across the trust
boundary. Offline execution requires the exact verified cache object and does
not fetch a substitute.

## Generated cards and support data

Run `PYTHONPATH=src python -m tools.artifacts.generate_docs` to render model
cards and the support matrix from `models.toml`. Run the command with `--check`
in CI to reject stale generated files. Generated files state the validation
boundary. They do not turn manifest declarations into unverified performance or
biological claims.
