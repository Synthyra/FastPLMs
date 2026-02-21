# syntax=docker/dockerfile:1.7
# Lightweight Docker container for Protein Design Environment
# Provides a standardized Linux environment for testing and inference with torch.compile support

# CUDA / cuDNN base with no Python
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# System prerequisites + Python 3.12
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app:/app/dplm/vendor/openfold \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TOKENIZERS_PARALLELISM=true

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates wget \
        python3.12 python3.12-dev python3.12-venv python3-pip \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
        libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
        ninja-build && \
    python3.12 -m venv /opt/venv && \
    ln -sf /opt/venv/bin/python /usr/local/bin/python && \
    ln -sf /opt/venv/bin/pip /usr/local/bin/pip

# Location of project code (inside image) – NOT shared with host
WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U boltz[cuda]

# Install E1
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 https://github.com/Profluent-AI/E1.git && \
    cd E1 && \
    pip install -e . && \
    cd ..

# Install EvolutionaryScale ESM (keeps the standard 'esm' namespace)
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 https://github.com/evolutionaryscale/esm.git && \
    cd esm && \
    pip install -e . && \
    cd ..

# Vendor Facebook Research FAIR-ESM as 'fair_esm'
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 https://github.com/facebookresearch/esm.git fair-esm-repo && \
    cd fair-esm-repo && \
    # Rename the inner module directory
    mv esm fair_esm && \
    # Comprehensively patch setup.py to rename the package and all subpackages
    sed -i 's/\besm\b/fair_esm/g' setup.py && \
    # Patch all Python files to use the new namespace
    find . -type f -name "*.py" -exec sed -i \
        -e 's/\bfrom esm\b/from fair_esm/g' \
        -e 's/\bimport esm\b/import fair_esm/g' \
        -e 's/\besm\./fair_esm\./g' \
        -e 's/"esm"/"fair_esm"/g' \
        -e "s/'esm'/'fair_esm'/g" {} + && \
    pip install -e . && \
    cd ..

# Install Bytedance DPLM and patch it to use 'fair_esm' instead of 'esm'
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/bytedance/dplm.git && \
    cd dplm && \
    # Patch DPLM imports to point to our vendored fair_esm
    find . -type f -name "*.py" -exec sed -i \
        -e 's/\bfrom esm\b/from fair_esm/g' \
        -e 's/\bimport esm\b/import fair_esm/g' \
        -e 's/\besm\./fair_esm\./g' \
        -e 's/\btransformers\.models\.fair_esm\b/transformers.models.esm/g' {} + && \
    # Avoid eager datamodule imports from byprot/__init__.py (pulls training-only deps like OpenFold)
    sed -i '/^import byprot\.datamodules$/d' src/byprot/__init__.py && \
    # Empty out the requirements file so readlines() returns an empty list
    echo "" > requirements.txt && \
    pip install -e . && \
    cd ..

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U -r requirements.txt && \
    pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    pip install numpy==1.26.4 && \
    pip install "lightning<2.2.0" "pytorch-lightning<2.2.0" "lightning-fabric<2.2.0" "torchmetrics<1.3.0" && \
    pip install "setuptools<81" wheel && \
    python -c "import pkg_resources"

# Inject a zero-dependency local 'imp' shim using modern standard library
RUN printf 'import importlib\n\
import importlib.util\n\
import importlib.machinery\n\
import sys\n\
import types\n\
\n\
def reload(module):\n\
    return importlib.reload(module)\n\
\n\
def new_module(name):\n\
    return types.ModuleType(name)\n\
\n\
def load_source(name, pathname, file=None):\n\
    loader = importlib.machinery.SourceFileLoader(name, pathname)\n\
    spec = importlib.util.spec_from_file_location(name, pathname, loader=loader)\n\
    module = importlib.util.module_from_spec(spec)\n\
    sys.modules[name] = module\n\
    loader.exec_module(module)\n\
    return module\n' > /opt/venv/lib/python3.12/site-packages/imp.py

# Copy the rest of the source
COPY . .

# Change working directory to where the volume will be mounted
WORKDIR /workspace

# ──────────────────────────────────────────────────────────────────────────────
# Single persistent host volume (/workspace) for *all* artefacts & caches
#    Bind-mount it when you run the container:  -v ${PWD}:/workspace
# ──────────────────────────────────────────────────────────────────────────────
ENV PROJECT_ROOT=/workspace \
    PYTHONPATH=/app:/app/dplm/vendor/openfold \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DISABLE_PANDERA_IMPORT_WARNING=True \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache \
    WANDB_DIR=/workspace/logs \
    TQDM_CACHE=/workspace/.cache/tqdm

RUN mkdir -p \
      /workspace/.cache/huggingface \
      /workspace/.cache/torch \
      /workspace/.cache/tqdm \
      /workspace/logs \
      /workspace/data \
      /workspace/results

# Declare the volume so other developers know it's intended to persist
VOLUME ["/workspace"]

# Default command – override in `docker run … python design_proteins.py`
CMD ["bash"]