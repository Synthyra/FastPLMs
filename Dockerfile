# Lightweight Docker container for Protein Design Environment
# Provides a standardized Linux environment for testing and inference with torch.compile support

# CUDA / cuDNN base with no Python
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# System prerequisites + Python 3.12
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.12.7 \
    PATH=/usr/local/bin:$PATH \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TOKENIZERS_PARALLELISM=true

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates wget \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
        libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
        ninja-build && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLO https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION}* && \
    ln -s /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.12    /usr/local/bin/pip

# Location of project code (inside image) – NOT shared with host
WORKDIR /app

# Copy requirements first for layer caching (matching Modal image order)
COPY requirements.txt .

RUN pip install --upgrade pip setuptools
RUN pip install boltz[cuda] -U
RUN git clone https://github.com/Profluent-AI/E1.git && cd E1 && pip install -e . && cd ..
RUN git clone https://github.com/evolutionaryscale/esm.git && cd esm && pip install -e . && cd ..
RUN git clone --recursive https://github.com/bytedance/dplm.git && \
    cd dplm && \
    # Empty out the requirements file so readlines() returns an empty list
    echo "" > requirements.txt && \
    pip install -e . && \
    pip install -e vendor/openfold && \
    cd ..
RUN pip install -r requirements.txt -U
RUN pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128 -U
RUN pip install --force-reinstall numpy==1.26.4

# Copy the rest of the source
COPY . .

# Change working directory to where the volume will be mounted
WORKDIR /workspace

# ──────────────────────────────────────────────────────────────────────────────
# Single persistent host volume (/workspace) for *all* artefacts & caches
#     Bind-mount it when you run the container:  -v ${PWD}:/workspace
# ──────────────────────────────────────────────────────────────────────────────
ENV PROJECT_ROOT=/workspace \
    PYTHONPATH=/app \
    CHAI_DOWNLOADS_DIR=/workspace/models/chai1 \
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