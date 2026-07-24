# Container environments

`docker/Dockerfile` is the only maintained Dockerfile. Its candidate stages
pin Python 3.12, CUDA 13.0, PyTorch 2.13.0, and Transformers 5.13.0. The runtime
stage contains FastPLMs source but never contains `vendor/upstream/`. Candidate
dependencies are installed with `uv pip install` from the named profiles under
`requirements/profiles/` and the validation constraint. FastPLMs itself is not
installed. Candidate and runtime stages load the copied or mounted source
through `PYTHONPATH`.

The single runtime stage accepts `FASTPLMS_RUNTIME_PROFILE=core` by default or
`FASTPLMS_RUNTIME_PROFILE=esmfold2-fp8` for ESMFold2 serving. Any other value
fails the image build. Bake exposes `runtime` and `runtime-fp8` names that both
point to this stage with the corresponding profile. The FP8 profile and the
`candidate-fp8` stage install Transformer Engine. The
`candidate-structure` and `candidate-fp8` profiles install cuEquivariance so
named ESMFold2 and Boltz2 kernel paths can be validated against the CUDA 13
runtime. The ordinary candidate, structure, and artifact profiles retain the
canonical PyTorch CUDA dependency graph and do not install Transformer Engine.

Reference targets are intentionally isolated because the official projects use
incompatible dependency stacks. They copy only their named submodule from the
target-specific Buildx context and act as parity oracles. The Biohub ESM
reference never resolves
the upstream package's mutable Transformers `@main` direct reference. Its
non-Transformers dependencies are derived with a fail-closed PEP 508 subset
parser, Biohub ESM is installed without dependency resolution, and the
manifest-pinned Biohub Transformers checkout is force-installed last from its
local source. The source checkout is not placed on startup `PYTHONPATH`; the
runtime gate inventories and hashes the complete tree before prioritizing its
package path. The installed wheel supplies dependency and distribution
metadata, while the independently attested source checkout is authoritative
for executed module bytes. That wheel is built from a disposable exact copy so
setuptools cannot add `build/` output to the attested tree. Image construction
runs `pip check` and validates the copied
checkout against its independent checked-in revision and tracked-tree digest.
Every ESMC, ESM3, and ESMFold2 entry path repeats that full-tree,
pre-import-origin, package-version, and post-import attestation. Native result
metadata records the source revision, source-tree and attestation hashes,
package version, and source-relative import identity under the versioned
`reference_source_attestation` field.

Bake does not hard-code a CPU architecture. Direct invocations use the native
builder platform; `tools/remote/run.py` passes its preflight-resolved platform
explicitly and verifies each loaded image's OS, architecture, and content
digest. The GH200 oracle uses the hash-attested `linux/arm64` dependency lock,
including source-built wheels where required. Containers do not erase ABI
boundaries, so evidence is never transferred across platforms.
Biohub suites build and load the tagged `biohub-biotraj-wheel` target alongside
the reference images. Before any oracle runs, remote orchestration persists its
content digest with every other target under
`artifacts/reference/environment/container-images.json`; tags and creation
timestamps are excluded from that canonical identity.

Production code must not import the upstream directories.

Routine candidate checks do not require official submodules. From the repository
root, build the exact image that Compose will run and pass the complete pytest
selection explicitly:

```bash
sudo docker buildx bake -f docker/docker-bake.hcl candidate --load
sudo docker compose -f docker/compose.yaml run --rm candidate \
  python -m pytest tests/unit tests/integration \
  -m "not gpu and not slow and not structure"
```

Live official parity is a compliance workflow, not a bare
`pytest tests/parity` invocation. It must prepare immutable requests, run each
official implementation in its isolated image, build candidate artifacts, and
then consume the normalized results. The remote runner performs that complete
sequence and is the canonical executable command:

```bash
git submodule update --init --recursive
python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/ssh-key \
  --suite compliance
```

Use `docker buildx bake -f docker/docker-bake.hcl references --load` only when
all pinned submodules are initialized and the isolated reference images are
needed for that compliance workflow. Building those images alone does not run
parity.

Compose centralizes GPU access, `ipc: host`, source mounts, and persistent
Hugging Face and Torch caches. Reference build targets use the names recorded
for their model families in `src/fastplms/models.toml`. No image contains
checkpoint weights, SSH keys, Hub tokens, workstation addresses, or other
credentials.
