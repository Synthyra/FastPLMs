"""Pinned settings and resource limits for Modal evidence runs."""

from __future__ import annotations

from tools.confidence.config import GPU_DOLLARS_PER_SECOND, resource_rate


APP_NAME = "fastplms-gpu-evidence"
# The validation stack pins these; a unit test keeps them equal to
# requirements/constraints/validation.txt.
TORCH_VERSION = "2.13.0"
TRANSFORMERS_VERSION = "5.13.0"
PYTHON_VERSION = "3.12"
CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/cu130"
CPU_WHEEL_INDEX = "https://download.pytorch.org/whl/cpu"

# Only GPUs with a recorded per-second rate can be reserved against the cap.
GPU_CHOICES = tuple(GPU_DOLLARS_PER_SECOND)
# L4 is the least expensive recorded GPU that runs BF16 and, at compute
# capability 8.9, can load the locked FlashAttention builds.
DEFAULT_GPU = "L4"
DEFAULT_MAX_DOLLARS = 25.0

CPU_CORES = 8.0
CPU_MEMORY_GIB = 16.0
GPU_CORES = 4.0
GPU_MEMORY_GIB = 32.0
STARTUP_TIMEOUT_SECONDS = 900
ENVIRONMENT_PROBE_TIMEOUT_SECONDS = 300
# Modal fixes a function's timeout at decoration, so each worker class takes
# the longest stage it can host; the stage's own timeout bounds the subprocess.
CPU_WORKER_TIMEOUT_SECONDS = 1_800
GPU_WORKER_TIMEOUT_SECONDS = 3_600
# The folding sweep loads eight models in turn and folds each up to 1,024 residues.
FOLD_WORKER_TIMEOUT_SECONDS = 7_200
OUTPUT_TAIL_CHARACTERS = 60_000


def worker_rate(gpu: str | None) -> float:
    """Dollars per second for one worker, from its requested allocation."""
    if gpu is None:
        return resource_rate(None, cpu=CPU_CORES, memory_gib=CPU_MEMORY_GIB)
    return resource_rate(gpu, cpu=GPU_CORES, memory_gib=GPU_MEMORY_GIB)
