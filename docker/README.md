# Container environments

`docker/Dockerfile` is the only maintained Dockerfile. Its candidate stages
pin Python 3.12, CUDA 13.0, PyTorch 2.13.0, and Transformers 5.13.0. The runtime
stage contains FastPLMs source but never contains `vendor/upstream/`. Candidate
dependencies are installed from the checked-in `uv.lock` with `uv sync
--frozen`. Constraint files are reserved for incompatible upstream reference
environments.

The single runtime stage accepts `FASTPLMS_RUNTIME_PROFILE=core` by default or
`FASTPLMS_RUNTIME_PROFILE=esmfold2-fp8` for ESMFold2 serving. Any other value
fails the image build. Bake exposes `runtime` and `runtime-fp8` names that both
point to this stage with the corresponding profile. The FP8 profile and the
`candidate-fp8` stage install the FP8 extra. Ordinary candidate, structure, and
artifact images retain the canonical PyTorch CUDA dependency graph and do not
import Transformer Engine.

Reference targets are intentionally isolated because the official projects use
incompatible dependency stacks. They copy only their named submodule from
`vendor/upstream/` and act as parity oracles. Production code must not import
those directories.

From the repository root:

```bash
docker buildx bake -f docker/docker-bake.hcl check
docker buildx bake -f docker/docker-bake.hcl references
docker compose -f docker/compose.yaml run --rm candidate
docker compose -f docker/compose.yaml run --rm structure
```

Compose centralizes GPU access, `ipc: host`, source mounts, and persistent
Hugging Face and Torch caches. Reference build targets use the names recorded
for their model families in `src/fastplms/models.toml`. No image contains
checkpoint weights, SSH keys, Hub tokens, workstation addresses, or other
credentials.
