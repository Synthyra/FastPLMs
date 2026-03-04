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

RUN pip install --upgrade pip setuptools
RUN pip install -U -r requirements.txt
RUN pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install numpy==1.26.4

COPY . .

WORKDIR /workspace

CMD ["bash"]
