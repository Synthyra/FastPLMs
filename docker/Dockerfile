# syntax=docker/dockerfile:1.19@sha256:b6afd42430b15f2d2a4c5a02b919e98a525b785b1aaff16747d2f623364e39b6

# Candidate validation is pinned to CUDA 13.0. The multi-platform digest is
# intentional: Buildx resolves the platform-specific image beneath it.
ARG CUDA13_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:8b2705ea7a8653ad3451b46ab835eced92d77b44e671b9cf3ad4f95fbb2efe5e
ARG CUDA12_LEGACY_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04@sha256:21196d81f56b48dbee70494d5f10322e1a77cc47ffe202a3bf68eab81533c20f

FROM ${CUDA13_IMAGE} AS python312

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CACHE_DIR=/root/.cache/pip \
    TOKENIZERS_PARALLELISM=true \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg \
    FASTPLMS_ARTIFACTS=/workspace/artifacts

# PyTorch 2.13 pins its cuDNN runtime as a wheel. Keep that locked runtime ahead
# of the CUDA base image so Transformer Engine behaves identically regardless
# of whether a BF16 cuDNN kernel ran before the first FP8 reload.

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates git git-lfs ninja-build \
        python3.12 python3.12-dev python3.12-venv && \
    python3.12 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip==26.1.1 uv==0.10.12

WORKDIR /opt/fastplms

FROM python312 AS dependency-files

# Keep dependency-only layers independent from runtime source, tests, docs, and
# legal text. A source edit must not rebuild the CUDA/Python environment.
COPY requirements ./requirements
COPY kernels.lock ./

FROM dependency-files AS source-dependencies

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        --requirement requirements/profiles/runtime.in \
        --constraint requirements/constraints/validation.txt

FROM source-dependencies AS source

COPY README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY LICENSES ./LICENSES
COPY src ./src
ENV PYTHONPATH=/opt/fastplms/src

FROM dependency-files AS runtime-dependencies

ARG FASTPLMS_RUNTIME_PROFILE=core
RUN --mount=type=cache,target=/root/.cache/uv \
    case "${FASTPLMS_RUNTIME_PROFILE}" in \
        core) \
            uv pip install --python /opt/venv/bin/python \
                --requirement requirements/profiles/runtime.in \
                --constraint requirements/constraints/validation.txt ;; \
        esmfold2-fp8) \
            uv pip install --python /opt/venv/bin/python \
                --requirement requirements/profiles/runtime-fp8.in \
                --constraint requirements/constraints/validation.txt \
                --overrides requirements/overrides/cuda.txt ;; \
        *) \
            echo "Unsupported FastPLMs runtime profile: ${FASTPLMS_RUNTIME_PROFILE}" >&2; \
            exit 64 ;; \
    esac

FROM runtime-dependencies AS runtime

COPY README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY LICENSES ./LICENSES
COPY src ./src
ENV PYTHONPATH=/opt/fastplms/src
WORKDIR /workspace
CMD ["python"]

FROM dependency-files AS candidate-dependencies

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        --requirement requirements/profiles/candidate.in \
        --constraint requirements/constraints/validation.txt

FROM candidate-dependencies AS candidate

COPY README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY LICENSES ./LICENSES
COPY src ./src
COPY tests /opt/fastplms/tests
COPY benchmarks /opt/fastplms/benchmarks
COPY tools /opt/fastplms/tools
ENV PYTHONPATH=/workspace/src:/workspace:/opt/fastplms/src:/opt/fastplms
WORKDIR /workspace
CMD ["python", "-m", "pytest", "tests/unit", "tests/integration", "-m", "not gpu and not slow and not structure"]

FROM dependency-files AS candidate-structure-dependencies

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        --requirement requirements/profiles/candidate-structure.in \
        --constraint requirements/constraints/validation.txt

FROM candidate-structure-dependencies AS candidate-structure

COPY README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY LICENSES ./LICENSES
COPY src ./src
COPY tests /opt/fastplms/tests
COPY benchmarks /opt/fastplms/benchmarks
COPY tools /opt/fastplms/tools
ENV PYTHONPATH=/workspace/src:/workspace:/opt/fastplms/src:/opt/fastplms
WORKDIR /workspace
CMD ["python", "-m", "pytest", "tests/structure", "-m", "structure"]

FROM dependency-files AS candidate-fp8-dependencies

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        --requirement requirements/profiles/candidate-fp8.in \
        --constraint requirements/constraints/validation.txt \
        --overrides requirements/overrides/cuda.txt

FROM candidate-fp8-dependencies AS candidate-fp8

COPY README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY LICENSES ./LICENSES
COPY src ./src
COPY tests /opt/fastplms/tests
COPY benchmarks /opt/fastplms/benchmarks
COPY tools /opt/fastplms/tools
ENV PYTHONPATH=/workspace/src:/workspace:/opt/fastplms/src:/opt/fastplms
WORKDIR /workspace
CMD ["python", "-m", "pytest", "tests/structure", "-m", "gpu and structure"]

# The artifact tier intentionally installs dependencies without FastPLMs. A
# mounted `dist/hub/<model>` must therefore carry all remote-code sources it
# needs, and the probe fails if an installed `fastplms` distribution is discoverable.
FROM dependency-files AS candidate-artifact

COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        --requirement requirements/profiles/artifact.in \
        --constraint requirements/constraints/validation.txt
ENV PYTHONPATH=
WORKDIR /workspace
CMD ["python", "-m", "pytest", "tests/release/test_published_automodel.py", "-m", "artifact"]

FROM ${CUDA12_LEGACY_IMAGE} AS python310-reference

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CACHE_DIR=/root/.cache/pip \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg \
    FASTPLMS_ARTIFACTS=/workspace/artifacts

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates git python3.10 python3.10-dev python3.10-venv && \
    python3.10 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip==26.1.1
WORKDIR /workspace

# The native oracle harness is independent of the FastPLMs source. Reference
# stages receive this small protocol plus only their declared upstream source.
FROM scratch AS reference-protocol

COPY tests/__init__.py /opt/oracle/tests/__init__.py
COPY tests/parity/__init__.py /opt/oracle/tests/parity/__init__.py
COPY tests/parity/support/__init__.py /opt/oracle/tests/parity/support/__init__.py
COPY tests/parity/support/state_transforms.py /opt/oracle/tests/parity/support/state_transforms.py
COPY tests/parity/support/semantic_config.py /opt/oracle/tests/parity/support/semantic_config.py
COPY tests/parity/support/native_reference.py /opt/oracle/tests/parity/support/native_reference.py
COPY tests/parity/support/reference_adapters /opt/oracle/tests/parity/support/reference_adapters
COPY tests/structure/__init__.py /opt/oracle/tests/structure/__init__.py
COPY tests/structure/support/__init__.py /opt/oracle/tests/structure/support/__init__.py
COPY tests/structure/support/state_contract.py /opt/oracle/tests/structure/support/state_contract.py
COPY tests/structure/support/boltz2_bundle.py /opt/oracle/tests/structure/support/boltz2_bundle.py
COPY tests/structure/support/esmfold_bundle.py /opt/oracle/tests/structure/support/esmfold_bundle.py
COPY tests/structure/support/esmfold2_bundle.py /opt/oracle/tests/structure/support/esmfold2_bundle.py
COPY tools/source_record.py /opt/oracle/tools/source_record.py
COPY tools/remote/biohub_reference_environment.py /opt/oracle/tools/remote/biohub_reference_environment.py
COPY tools/remote/biohub_reference_lock.py /opt/oracle/tools/remote/biohub_reference_lock.py
COPY tools/remote/biohub_reference_requirements.py /opt/oracle/tools/remote/biohub_reference_requirements.py
COPY tools/remote/reference_source_attestation.py /opt/oracle/tools/remote/reference_source_attestation.py

FROM python310-reference AS reference-ankh

COPY docker/constraints/ankh.txt /tmp/constraints.txt
COPY --from=upstream_ankh --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/ankh
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130 && \
    pip install --constraint /tmp/constraints.txt /opt/oracle/vendor/upstream/ankh \
        pytest==8.4.2 safetensors==0.6.2
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/ankh /licenses/ankh
ENV PYTHONPATH=/opt/oracle/vendor/upstream/ankh/src:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.parity.support.reference_adapters.ankh"]

FROM python310-reference AS reference-dplm

COPY docker/constraints/dplm.txt /tmp/constraints.txt
COPY --from=upstream_dplm --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/dplm
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.2.0 torchtext==0.17.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --constraint /tmp/constraints.txt --only-binary=:all: \
        --find-links https://data.pyg.org/whl/torch-2.2.0+cu121.html \
        torch-scatter==2.1.2+pt22cu121 && \
    pip install --constraint /tmp/constraints.txt -e /opt/oracle/vendor/upstream/dplm \
        safetensors==0.6.2
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/dplm /licenses/dplm
ENV PYTHONPATH=/opt/oracle/vendor/upstream/dplm/src:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.parity.support.reference_adapters.dplm"]

FROM python312 AS reference-e1

COPY docker/constraints/e1.txt /tmp/constraints.txt
COPY --from=upstream_e1 --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/e1
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --constraint /tmp/constraints.txt -e /opt/oracle/vendor/upstream/e1 \
        pytest==8.4.2 safetensors==0.6.2
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/e1 /licenses/e1
ENV PYTHONPATH=/opt/oracle/vendor/upstream/e1/src:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.parity.support.reference_adapters.e1"]

FROM python312 AS reference-biohub-esm

ARG TARGETARCH
RUN test "${TARGETARCH}" = "arm64" && test "$(uname -m)" = "aarch64"
COPY docker/biohub-reference-lock.Dockerfile /opt/lock-root/docker/biohub-reference-lock.Dockerfile
COPY docker/constraints/biohub-reference.in /opt/lock-root/docker/constraints/biohub-reference.in
COPY docker/constraints/biohub-reference.lock.txt /opt/lock-root/docker/constraints/biohub-reference.lock.txt
COPY docker/constraints/biohub-biotraj-build.in /opt/lock-root/docker/constraints/biohub-biotraj-build.in
COPY docker/constraints/biohub-biotraj-build.lock.txt /opt/lock-root/docker/constraints/biohub-biotraj-build.lock.txt
COPY docker/constraints/biohub-reference-lock.json /opt/lock-root/docker/constraints/biohub-reference-lock.json
COPY tools/remote/biohub_reference_lock.py /opt/lock-root/biohub_reference_lock.py
COPY --from=biohub_biotraj_wheel \
    /biohub-reference-lock/biotraj-1.2.2-cp312-cp312-linux_aarch64.whl \
    /opt/wheels/biotraj-1.2.2-cp312-cp312-linux_aarch64.whl
COPY --from=upstream_biohub_transformers --exclude=.git --exclude=.git/** --exclude=**/__pycache__ --exclude=**/__pycache__/** --exclude=**/*.pyc --exclude=**/*.pyo . /opt/oracle/vendor/upstream/biohub-transformers
COPY --from=upstream_biohub_esm --exclude=.git --exclude=.git/** --exclude=**/__pycache__ --exclude=**/__pycache__/** --exclude=**/*.pyc --exclude=**/*.pyo . /opt/oracle/vendor/upstream/biohub-esm
RUN --mount=type=cache,target=/root/.cache/pip \
    python /opt/lock-root/biohub_reference_lock.py verify-contract \
        --root /opt/lock-root \
        --contract /opt/lock-root/docker/constraints/biohub-reference-lock.json && \
    python /opt/lock-root/biohub_reference_lock.py materialize-wheel-lock \
        --root /opt/lock-root \
        --contract /opt/lock-root/docker/constraints/biohub-reference-lock.json \
        --wheel /opt/wheels/biotraj-1.2.2-cp312-cp312-linux_aarch64.whl \
        --wheel-uri file:///opt/wheels/biotraj-1.2.2-cp312-cp312-linux_aarch64.whl \
        --output /tmp/biohub-reference.install.lock.txt && \
    cp -a /opt/oracle/vendor/upstream/biohub-transformers /tmp/biohub-transformers-build && \
    cp -a /opt/oracle/vendor/upstream/biohub-esm /tmp/biohub-esm-build && \
    pip install --require-hashes --only-binary=:all: --no-deps \
        --requirement /tmp/biohub-reference.install.lock.txt && \
    pip install --no-deps --no-build-isolation \
        /tmp/biohub-transformers-build /tmp/biohub-esm-build && \
    python /opt/lock-root/biohub_reference_lock.py verify-pip-check \
        --root /opt/lock-root \
        --contract /opt/lock-root/docker/constraints/biohub-reference-lock.json && \
    python /opt/lock-root/biohub_reference_lock.py verify-inventory \
        --root /opt/lock-root \
        --contract /opt/lock-root/docker/constraints/biohub-reference-lock.json \
        --profile final && \
    rm -rf -- /tmp/biohub-transformers-build /tmp/biohub-esm-build
COPY docker/constraints/biohub-esm-source.json /opt/oracle/biohub-esm-source-contract.json
COPY docker/constraints/biohub-transformers-source.json /opt/oracle/biohub-transformers-source-contract.json
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/biohub-esm /licenses/biohub-esm
COPY LICENSES/biohub-transformers /licenses/biohub-transformers
RUN test ! -e /opt/oracle/tools/remote/__init__.py && \
    PYTHONPATH=/opt/oracle python -c \
        "import importlib.util; import tools.remote.biohub_reference_environment; import tools.remote.biohub_reference_lock; import tools.remote.reference_source_attestation; missing = importlib.util.find_spec('tools.remote.run') is None; raise SystemExit(0 if missing else 'Unexpected tools.remote.run in oracle image.')"
RUN PYTHONPATH=/opt/oracle python -m tools.remote.reference_source_attestation create \
        --source-root /opt/oracle/vendor/upstream/biohub-esm \
        --contract /opt/oracle/biohub-esm-source-contract.json \
        --output /opt/oracle/biohub-esm-source-attestation.json && \
    PYTHONPATH=/opt/oracle python -m tools.remote.reference_source_attestation create \
        --source-root /opt/oracle/vendor/upstream/biohub-transformers \
        --contract /opt/oracle/biohub-transformers-source-contract.json \
        --output /opt/oracle/biohub-transformers-source-attestation.json && \
    PYTHONPATH=/opt/oracle python -m tools.remote.reference_source_attestation verify \
        --source-root /opt/oracle/vendor/upstream/biohub-esm \
        --attestation /opt/oracle/biohub-esm-source-attestation.json \
        --contract /opt/oracle/biohub-esm-source-contract.json \
        --expected-revision 82ee35553d39169d678f784c8d3f8712ffd7d2c4 && \
    PYTHONPATH=/opt/oracle python -m tools.remote.reference_source_attestation verify \
        --source-root /opt/oracle/vendor/upstream/biohub-transformers \
        --attestation /opt/oracle/biohub-transformers-source-attestation.json \
        --contract /opt/oracle/biohub-transformers-source-contract.json \
        --expected-revision 3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf && \
    PYTHONPATH=/opt/oracle python -m tools.remote.biohub_reference_lock verify-inventory \
        --root /opt/lock-root \
        --contract /opt/lock-root/docker/constraints/biohub-reference-lock.json \
        --profile final
ENV FASTPLMS_BIOHUB_ESM_REVISION=82ee35553d39169d678f784c8d3f8712ffd7d2c4 \
    FASTPLMS_BIOHUB_ESM_SOURCE=/opt/oracle/vendor/upstream/biohub-esm \
    FASTPLMS_BIOHUB_ESM_ATTESTATION=/opt/oracle/biohub-esm-source-attestation.json \
    FASTPLMS_BIOHUB_ESM_CONTRACT=/opt/oracle/biohub-esm-source-contract.json \
    FASTPLMS_BIOHUB_TRANSFORMERS_REVISION=3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf \
    FASTPLMS_BIOHUB_TRANSFORMERS_SOURCE=/opt/oracle/vendor/upstream/biohub-transformers \
    FASTPLMS_BIOHUB_TRANSFORMERS_ATTESTATION=/opt/oracle/biohub-transformers-source-attestation.json \
    FASTPLMS_BIOHUB_TRANSFORMERS_CONTRACT=/opt/oracle/biohub-transformers-source-contract.json \
    FASTPLMS_BIOHUB_LOCK_ROOT=/opt/lock-root \
    FASTPLMS_BIOHUB_LOCK_CONTRACT=/opt/lock-root/docker/constraints/biohub-reference-lock.json \
    FASTPLMS_REFERENCE_CONTAINER_IDENTITIES=/exchange/environment/container-images.json \
    FASTPLMS_REFERENCE_CONTAINER_TARGET=reference-biohub-esm \
    PYTHONPATH=/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.parity.support.reference_adapters.esm_plusplus"]

FROM python312 AS reference-esm2

COPY --from=upstream_fair_esm --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/fair-esm
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130 && \
    pip install -e /opt/oracle/vendor/upstream/fair-esm huggingface-hub==0.36.2 \
        numpy==1.26.4 pytest==9.0.2 safetensors==0.6.2
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/fair-esm /licenses/fair-esm
ENV PYTHONPATH=/opt/oracle/vendor/upstream/fair-esm:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.parity.support.reference_adapters.esm2"]

FROM python310-reference AS reference-boltz2

COPY --from=upstream_boltz --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/boltz
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install '/opt/oracle/vendor/upstream/boltz[test]' \
        huggingface-hub==0.36.0 safetensors==0.6.2
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/boltz /licenses/boltz
ENV PYTHONPATH=/opt/oracle/vendor/upstream/boltz/src:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.structure.support.boltz2_bundle", "--help"]

# This diagnostic oracle intentionally matches the candidate numerical
# toolchain. It localizes semantic drift without replacing the upstream-native
# `reference-boltz2` release oracle above. Neither stage installs source-built
# FlashAttention; Boltz2 structure parity uses its eager attention path.
FROM python312 AS reference-boltz2-same-runtime

COPY --from=upstream_boltz --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/boltz
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130 && \
    pip install '/opt/oracle/vendor/upstream/boltz[test]' \
        huggingface-hub==0.36.0 safetensors==0.6.2
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/boltz /licenses/boltz
ENV PYTHONPATH=/opt/oracle/vendor/upstream/boltz/src:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.structure.support.boltz2_bundle", "--help"]

FROM python310-reference AS reference-esmfold

COPY docker/constraints/esmfold.txt /tmp/constraints.txt
COPY docker/constraints/openfold-sm90.patch /tmp/openfold-sm90.patch
COPY --from=upstream_openfold --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/openfold
COPY --from=upstream_fair_esm --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/fair-esm
RUN cd /opt/oracle/vendor/upstream/openfold && \
    git apply --no-index --ignore-space-change --check /tmp/openfold-sm90.patch && \
    git apply --no-index --ignore-space-change /tmp/openfold-sm90.patch && \
    grep -F 'compute_capabilities = {(9, 0)}' setup.py && \
    grep -F -- "'-std=c++17'" setup.py
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --constraint /tmp/constraints.txt ninja wheel && \
    pip install --no-build-isolation --no-deps /opt/oracle/vendor/upstream/openfold && \
    pip install --constraint /tmp/constraints.txt /opt/oracle/vendor/upstream/fair-esm \
        biopython dm-tree einops huggingface-hub lightning-utilities ml-collections \
        omegaconf pytorch-lightning scipy pytest safetensors torchmetrics dllogger
RUN extension="$(python -c 'import torch; import attn_core_inplace_cuda as extension; print(extension.__file__)')" && \
    cuobjdump --list-elf "$extension" | grep -F 'sm_90' && \
    test -f /opt/venv/lib/python3.10/site-packages/dllogger-1.1.0.dist-info/licenses/LICENSE && \
    grep -F 'Apache License' \
        /opt/venv/lib/python3.10/site-packages/dllogger-1.1.0.dist-info/licenses/LICENSE
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/fair-esm /licenses/fair-esm
COPY LICENSES/openfold /licenses/openfold
COPY LICENSES/dllogger/SOURCE_RECORD.md /licenses/dllogger/SOURCE_RECORD.md
ENV PYTHONPATH=/opt/oracle/vendor/upstream/openfold:/opt/oracle/vendor/upstream/fair-esm:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.structure.support.esmfold_bundle", "--help"]

FROM reference-biohub-esm AS reference-esmfold2

ENV FASTPLMS_REFERENCE_CONTAINER_TARGET=reference-esmfold2
CMD ["python", "-c", "from tests.parity.support.reference_adapters.biohub_source import reference_sources; reference_sources(); from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model; print(ESMFold2Model.__name__)"]

FROM python310-reference AS reference-protein-ttt

COPY --from=upstream_protein_ttt --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/protein-ttt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121 && \
    if [ -f /opt/oracle/vendor/upstream/protein-ttt/requirements.txt ]; then \
        pip install -r /opt/oracle/vendor/upstream/protein-ttt/requirements.txt; \
    fi
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/protein-ttt /licenses/protein-ttt
ENV PYTHONPATH=/opt/oracle/vendor/upstream/protein-ttt
CMD ["python", "-c", "import pathlib; print(pathlib.Path('/opt/oracle/vendor/upstream/protein-ttt').resolve())"]
