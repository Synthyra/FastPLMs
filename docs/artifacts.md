# Local Hub artifacts

`tools/artifacts/build.py` creates deterministic, offline-loadable Hugging Face
artifacts under `dist/hub/<model>/`. It operates only on an already downloaded,
manifest-pinned checkpoint snapshot. It never authenticates, downloads, creates
a Hub repository, uploads, deletes, commits, pushes, or opens a pull request.

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
`--tokenizer-dir` pointing to the manifest-pinned official snapshot. The
builder copies and records only the official tokenizer files declared by the
manifest. Artifacts whose selected checkpoint is already the official snapshot
may omit the option.

Before writing output, the builder validates:

- the model and every required file identity are resolved in `models.toml`;
- the local snapshot matches the declared immutable revision and file hashes;
- every required upstream submodule is initialized at its pinned commit;
- canonical and distributable license files match their declared hashes;
- the family has a complete mechanism-first conversion record.

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
provenance.json
artifact-manifest.json
LICENSES/...
THIRD_PARTY_NOTICES.md
```

Normal runtime source modules are copied unchanged under `fastplms/`. The
builder also writes those exact bytes into a deterministic compressed archive
inside `fastplms_bundle.py`. This flat bundle is required because Transformers'
remote-module loader follows flat relative Python imports; it does not import
the copied package tree directly.

`modeling_fastplms.py` imports the flat bundle, verifies its SHA-256 identity,
and extracts it into a hash-named directory in the Transformers module cache.
It then installs that bundled package and exposes the manifest-advertised
classes. This step performs no network access or compilation. The first load
does require a writable module cache for local extraction. Release validation
runs each artifact in a fresh interpreter. Loading a second artifact with a
different runtime-bundle hash in the same interpreter fails explicitly and
leaves the first runtime intact. Isolate different bundles in separate Python
processes. Production code never imports a submodule checkout.

The checkpoint source is selected by the manifest. This matters for ANKH, where
the artifact uses the official sequence-to-sequence checkpoint so the official
decoder and LM head are present. The named state transform is deterministic.
Weights are written as explicit safetensors shards no larger than 5 GiB with a
sorted index. Trusted legacy `.bin` input is loaded with `weights_only=True` and
is never copied into the output.

`provenance.json` records both FastPLMs and official checkpoint identities, the
selected artifact source, conversion record, BF16 execution policy, upstream
revisions, legal files, and FastPLMs version. The model card states the same
BF16 policy; artifact validation rejects a disagreement. `artifact-manifest.json`
contains a SHA-256 identity for every artifact file. Identical inputs produce
identical bytes and hashes.

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
compares configuration, state and output against package source. Network access,
an undeclared import, a missing legal text, or a missing conversion record fails
the tier.

## Generated cards and support data

Run `PYTHONPATH=src python -m tools.artifacts.generate_docs` to render model cards and the
support matrix from `models.toml`. Run the same command with `--check` in CI to
reject stale generated files. Generated files state the validation boundary and
do not turn manifest declarations into unverified performance or biological
claims.
