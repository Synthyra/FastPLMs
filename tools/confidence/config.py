"""Fixed scientific settings and resource limits for the confidence pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass


MODEL_IDS = ("esmfold2_300", "esmfold2_600")
DONOR_REPO = "biohub/ESMFold2-Experimental-Fast-Cutoff2025"
DONOR_REVISION = "74b88548bf19688b8727432db0d698cb2e1d8783"
DONOR_WEIGHT_SHA256 = "4e903b740ad6ad704ec60881bfd593e0d6c874a630ffa0f0838276e0b665088f"
DONOR_TENSOR_COUNT = 93
DONOR_TENSOR_BYTES = 31_071_252
VOLUME_NAME = "fastplms-confidence-pilot"
WANDB_PROJECT = "fastplms-confidence"
STAGES = ("prepare", "cache", "train", "evaluate", "campaign", "package", "release")
GPU_DOLLARS_PER_SECOND = {"L4": 0.000222, "L40S": 0.000542, "H100": 0.001097}
CPU_DOLLARS_PER_CORE_SECOND = 0.0000131
MEMORY_DOLLARS_PER_GIB_SECOND = 0.00000222
MAX_PARALLEL_WORKERS = 2
TRAIN_TIMEOUT_SECONDS = 36_600
TRAINING_MAXIMUM_SECONDS = 36_000
CAMPAIGN_TIMEOUT_SECONDS = 46_800
BENCHMARK_TIMEOUT_SECONDS = 600
CPU_STAGE_TIMEOUT_SECONDS = 1_800
GPU_STARTUP_TIMEOUT_SECONDS = 900


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 17
    learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    accumulation_steps: int = 16
    maximum_updates: int = 1000
    warmup_updates: int = 100
    validation_interval: int = 100
    early_stopping_patience: int = 3
    gradient_clip: float = 1.0
    num_loops: int = 3
    num_sampling_steps: int = 15
    num_diffusion_samples: int = 1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def resource_rate(gpu: str | None, cpu: float = 4.0, memory_gib: float = 32.0) -> float:
    """Conservative per-second rate using requested resource allocations."""
    gpu_rate = GPU_DOLLARS_PER_SECOND[gpu] if gpu else 0.0
    return gpu_rate + cpu * CPU_DOLLARS_PER_CORE_SECOND + memory_gib * MEMORY_DOLLARS_PER_GIB_SECOND
