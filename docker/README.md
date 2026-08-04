# Container environments

`docker/Dockerfile` is the only maintained Dockerfile. Candidate stages use
Python 3.12, CUDA 13.0, PyTorch 2.13.0, and Transformers 5.13.0. The runtime
stage contains FastPLMs source, but not `vendor/upstream/`. Candidate
dependencies are installed with `uv pip install` from the named profiles in
`requirements/profiles/` and the validation constraint. FastPLMs is not
installed. Candidate and runtime stages load copied or mounted source through
`PYTHONPATH`.

The runtime stage uses `FASTPLMS_RUNTIME_PROFILE=core` by default. Use
`FASTPLMS_RUNTIME_PROFILE=esmfold2-fp8` for ESMFold2 serving. Any other value
fails the image build. Bake provides `runtime` and `runtime-fp8`, which use this
stage with the related profile. The FP8 profile and `candidate-fp8` install
Transformer Engine. `candidate-structure` and `candidate-fp8` install
cuEquivariance to validate named ESMFold2 and Boltz2 kernel paths with CUDA 13.
The standard candidate, structure, and artifact profiles use the canonical
PyTorch CUDA dependency graph and do not install Transformer Engine.

Reference targets are isolated because official projects use incompatible
dependency stacks. Each target copies only its named submodule from the
target-specific Buildx context and acts as a parity oracle. The Biohub ESM
reference does not resolve the upstream package's mutable Transformers `@main`
direct reference. A fail-closed PEP 508 subset parser derives its
non-Transformers dependencies. Biohub ESM installs without dependency
resolution, and the manifest-pinned Biohub Transformers checkout is
force-installed last from local source. The source checkout is not on startup
`PYTHONPATH`. Before its package path is used, the runtime gate inventories and
hashes the full tree. The installed wheel provides dependency and distribution
metadata. The separately attested source checkout supplies the executed module
bytes. The wheel builds from a disposable exact copy, so setuptools cannot add
`build/` output to the attested tree. Image construction runs `pip check` and
checks the copied checkout against its independent checked-in revision and
tracked-tree digest. Each ESMC, ESM3, and ESMFold2 entry path repeats the
full-tree, pre-import-origin, package-version, and post-import checks. Native
result metadata records the source revision, source-tree and attestation
hashes, package version, and source-relative import identity in the versioned
`reference_source_attestation` field.

Bake does not set a CPU architecture. Direct calls use the native builder
platform. `tools/remote/run.py` passes the platform found during preflight and
checks the OS, architecture, and content digest of each loaded image. The GH200
oracle uses the hash-attested `linux/arm64` dependency lock, including
source-built wheels when needed. Containers do not remove ABI boundaries. Do
not transfer evidence across platforms. Biohub suites build and load the tagged
`biohub-biotraj-wheel` target with the reference images. Before an oracle runs,
remote orchestration stores its content digest with other targets in
`artifacts/reference/environment/container-images.json`. Tags and creation
times are not part of this canonical identity.

Production code must not import upstream directories.

Routine candidate checks do not need official submodules. From the repository
root, build the image that Compose runs. Then pass the complete pytest selection:

```bash
sudo docker buildx bake -f docker/docker-bake.hcl candidate --load
sudo docker compose -f docker/compose.yaml run --rm candidate \
  python -m pytest tests/unit tests/integration \
  -m "not gpu and not slow and not structure"
```

Live official parity is a compliance workflow. It is not a bare
`pytest tests/parity` call. It prepares immutable requests, runs each official
implementation in an isolated image, builds candidate artifacts, and uses the
normalized results. The remote runner performs this sequence:

```bash
git submodule update --init --recursive
python -m tools.remote \
  --host user@gpu-host \
  --identity /path/to/ssh-key \
  --suite compliance
```

Use `docker buildx bake -f docker/docker-bake.hcl references --load` only when
all pinned submodules are initialized and the compliance workflow needs the
isolated reference images. Building these images alone does not run parity.

Compose sets GPU access, `ipc: host`, source mounts, and persistent Hugging
Face and Torch caches in one place. Reference build targets use names from
`src/fastplms/models.toml`. Images do not contain checkpoint weights, SSH keys,
Hub tokens, workstation addresses, or other credentials.
