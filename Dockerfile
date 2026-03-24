# syntax=docker/dockerfile:1.7
# NOTE: switch to cudnn-devel if you need to compile CUDA extensions (e.g. flash-attn from source)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TOKENIZERS_PARALLELISM=true \
    PROJECT_ROOT=/workspace \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DISABLE_PANDERA_IMPORT_WARNING=True \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache \
    WANDB_DIR=/workspace/logs \
    TQDM_CACHE=/workspace/.cache/tqdm

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates \
        python3.12 python3.12-dev python3.12-venv \
        ninja-build && \
    python3.12 -m venv /opt/venv && \
    ln -sf /opt/venv/bin/python /usr/local/bin/python && \
    ln -sf /opt/venv/bin/pip /usr/local/bin/pip

WORKDIR /app

COPY requirements.txt .

RUN git submodule update --remote --init --recursive
RUN pip install --upgrade pip setuptools

# Install official repos from submodules for compliance testing.
# EvolutionaryScale's esm is NOT pip installed because it conflicts with
# fair-esm (both use `import esm`). testing/official/esm_plusplus.py
# adds the submodule to sys.path on demand instead.
RUN pip install -e /app/official/e1 && \
    pip install -e /app/official/dplm

RUN pip install -r requirements.txt
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install numpy==1.26.4

COPY . .

WORKDIR /workspace

CMD ["bash"]
