# syntax=docker/dockerfile:1.19@sha256:b6afd42430b15f2d2a4c5a02b919e98a525b785b1aaff16747d2f623364e39b6

# Candidate validation is pinned to CUDA 13.0. The multi-platform digest is
# intentional: Buildx resolves the platform-specific image beneath it.
ARG CUDA13_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:8b2705ea7a8653ad3451b46ab835eced92d77b44e671b9cf3ad4f95fbb2efe5e
ARG CUDA12_LEGACY_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04@sha256:21196d81f56b48dbee70494d5f10322e1a77cc47ffe202a3bf68eab81533c20f

FROM ${CUDA13_IMAGE} AS python312

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
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

FROM python312 AS source

COPY pyproject.toml uv.lock kernels.lock README.md LICENSE ./
COPY LICENSES ./LICENSES
COPY THIRD_PARTY_NOTICES.md ./THIRD_PARTY_NOTICES.md
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group validation --no-editable

FROM source AS runtime

ARG FASTPLMS_RUNTIME_PROFILE=core
RUN --mount=type=cache,target=/root/.cache/uv \
    case "${FASTPLMS_RUNTIME_PROFILE}" in \
        core) ;; \
        esmfold2-fp8) \
            uv sync --frozen --no-dev --group validation --extra structure --extra fp8 \
                --no-editable ;; \
        *) \
            echo "Unsupported FastPLMs runtime profile: ${FASTPLMS_RUNTIME_PROFILE}" >&2; \
            exit 64 ;; \
    esac
WORKDIR /workspace
CMD ["python"]

FROM source AS candidate

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group validation --extra dev --extra flash \
        --no-editable
COPY tests /opt/fastplms/tests
COPY benchmarks /opt/fastplms/benchmarks
COPY tools /opt/fastplms/tools
ENV PYTHONPATH=/workspace/src:/workspace:/opt/fastplms
WORKDIR /workspace
CMD ["python", "-m", "pytest", "tests/unit", "tests/integration", "-m", "not gpu and not slow and not structure"]

FROM candidate AS candidate-structure

WORKDIR /opt/fastplms
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group validation --extra dev --extra flash --extra structure \
        --no-editable
WORKDIR /workspace
CMD ["python", "-m", "pytest", "tests/structure", "-m", "structure"]

FROM candidate-structure AS candidate-fp8

WORKDIR /opt/fastplms
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group validation --extra dev --extra flash --extra structure \
        --extra fp8 --no-editable
WORKDIR /workspace
CMD ["python", "-m", "pytest", "tests/structure", "-m", "gpu and structure"]

# The artifact tier intentionally installs dependencies without FastPLMs. A
# mounted `dist/hub/<model>` must therefore carry all remote-code sources it
# needs, and the probe fails if an installed `fastplms` package is discoverable.
FROM python312 AS candidate-artifact

COPY pyproject.toml uv.lock kernels.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --group validation --extra dev \
        --extra structure --extra flash
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

# The native oracle harness is independent of the FastPLMs package. Reference
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

FROM python310-reference AS reference-ankh

COPY docker/constraints/ankh.txt /tmp/constraints.txt
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY --from=upstream_ankh --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/ankh
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/ankh /licenses/ankh
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130 && \
    pip install --constraint /tmp/constraints.txt /opt/oracle/vendor/upstream/ankh \
        pytest==8.4.2 safetensors==0.6.2
ENV PYTHONPATH=/opt/oracle/vendor/upstream/ankh/src:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.parity.support.reference_adapters.ankh"]

FROM python310-reference AS reference-dplm

COPY docker/constraints/dplm.txt /tmp/constraints.txt
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY --from=upstream_dplm --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/dplm
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/dplm /licenses/dplm
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.2.0 torchtext==0.17.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --constraint /tmp/constraints.txt --only-binary=:all: \
        --find-links https://data.pyg.org/whl/torch-2.2.0+cu121.html \
        torch-scatter==2.1.2+pt22cu121 && \
    pip install --constraint /tmp/constraints.txt -e /opt/oracle/vendor/upstream/dplm \
        safetensors==0.6.2
ENV PYTHONPATH=/opt/oracle/vendor/upstream/dplm/src:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.parity.support.reference_adapters.dplm"]

FROM python312 AS reference-e1

COPY docker/constraints/e1.txt /tmp/constraints.txt
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY --from=upstream_e1 --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/e1
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/e1 /licenses/e1
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --constraint /tmp/constraints.txt -e /opt/oracle/vendor/upstream/e1 \
        pytest==8.4.2 safetensors==0.6.2
ENV PYTHONPATH=/opt/oracle/vendor/upstream/e1/src:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.parity.support.reference_adapters.e1"]

FROM python312 AS reference-biohub-esm

COPY docker/constraints/biohub.txt /tmp/constraints.txt
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY --from=upstream_biohub_transformers --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/biohub-transformers
COPY --from=upstream_biohub_esm --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/biohub-esm
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/biohub-esm /licenses/biohub-esm
COPY LICENSES/biohub-transformers /licenses/biohub-transformers
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130 && \
    pip install --constraint /tmp/constraints.txt --no-deps \
        -e /opt/oracle/vendor/upstream/biohub-transformers && \
    pip install --constraint /tmp/constraints.txt \
        -e /opt/oracle/vendor/upstream/biohub-esm pytest==9.0.2
ENV PYTHONPATH=/opt/oracle/vendor/upstream/biohub-esm:/opt/oracle/vendor/upstream/biohub-transformers/src:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.parity.support.reference_adapters.esm_plusplus"]

FROM python312 AS reference-esm2

COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY --from=upstream_fair_esm --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/fair-esm
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/fair-esm /licenses/fair-esm
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130 && \
    pip install -e /opt/oracle/vendor/upstream/fair-esm pytest==9.0.2 safetensors==0.6.2
ENV PYTHONPATH=/opt/oracle/vendor/upstream/fair-esm:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.parity.support.reference_adapters.esm2"]

FROM python310-reference AS reference-boltz2

COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY --from=upstream_boltz --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/boltz
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/boltz /licenses/boltz
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install '/opt/oracle/vendor/upstream/boltz[test]' \
        huggingface-hub==0.36.0 safetensors==0.6.2
ENV PYTHONPATH=/opt/oracle/vendor/upstream/boltz/src:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.structure.support.boltz2_bundle", "--help"]

# This diagnostic oracle intentionally matches the candidate numerical
# toolchain. It localizes semantic drift without replacing the upstream-native
# `reference-boltz2` release oracle above. Neither stage installs source-built
# FlashAttention; Boltz2 structure parity uses its eager attention path.
FROM python312 AS reference-boltz2-same-runtime

COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY --from=upstream_boltz --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/boltz
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/boltz /licenses/boltz
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130 && \
    pip install '/opt/oracle/vendor/upstream/boltz[test]' \
        huggingface-hub==0.36.0 safetensors==0.6.2
ENV PYTHONPATH=/opt/oracle/vendor/upstream/boltz/src:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.structure.support.boltz2_bundle", "--help"]

FROM python310-reference AS reference-esmfold

COPY docker/constraints/esmfold.txt /tmp/constraints.txt
COPY docker/constraints/openfold-sm90.patch /tmp/openfold-sm90.patch
COPY --from=reference-protocol /opt/oracle /opt/oracle
COPY --from=upstream_openfold --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/openfold
COPY --from=upstream_fair_esm --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/fair-esm
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/fair-esm /licenses/fair-esm
COPY LICENSES/openfold /licenses/openfold
COPY LICENSES/dllogger/PROVENANCE.md /licenses/dllogger/PROVENANCE.md
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
ENV PYTHONPATH=/opt/oracle/vendor/upstream/openfold:/opt/oracle/vendor/upstream/fair-esm:/opt/oracle
WORKDIR /opt/oracle
CMD ["python", "-m", "tests.structure.support.esmfold_bundle", "--help"]

FROM reference-biohub-esm AS reference-esmfold2

CMD ["python", "-c", "from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model; print(ESMFold2Model.__name__)"]

FROM python310-reference AS reference-protein-ttt

COPY --from=upstream_protein_ttt --exclude=.git --exclude=.git/** . /opt/oracle/vendor/upstream/protein-ttt
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY LICENSES/protein-ttt /licenses/protein-ttt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121 && \
    if [ -f /opt/oracle/vendor/upstream/protein-ttt/requirements.txt ]; then \
        pip install -r /opt/oracle/vendor/upstream/protein-ttt/requirements.txt; \
    fi
ENV PYTHONPATH=/opt/oracle/vendor/upstream/protein-ttt
CMD ["python", "-c", "import pathlib; print(pathlib.Path('/opt/oracle/vendor/upstream/protein-ttt').resolve())"]
