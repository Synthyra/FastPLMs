import contextlib
import random

import pytest
import torch

from fastplms.registry import ModelSpec, get_model_registry


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "slow: loads two models simultaneously (compliance tests)")
    config.addinivalue_line("markers", "large: requires 24+ GB VRAM (3B parameter models)")
    config.addinivalue_line(
        "markers", "structure: structure prediction models (Boltz2, ESMFold, ESMFold2)"
    )


CANONICAL_AAS = "ACDEFGHIKLMNPQRSTVWY"
SEED = 42
DEFAULT_BATCH_SIZE = 4
MAX_EMBED_LEN = 128


@contextlib.contextmanager
def strict_fp32_matmul():
    """Temporarily disable TF32 for fp32 numerical parity checks."""
    try:
        old_fp32_precision = torch.backends.fp32_precision
        old_matmul_precision = torch.backends.cuda.matmul.fp32_precision
        old_cudnn_precision = torch.backends.cudnn.fp32_precision
    except AttributeError:
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_matmul_precision = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.set_float32_matmul_precision(old_matmul_precision)
        return

    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.fp32_precision = old_fp32_precision
        torch.backends.cuda.matmul.fp32_precision = old_matmul_precision
        torch.backends.cudnn.fp32_precision = old_cudnn_precision


# The package manifest is the sole checkpoint registry. These dictionaries keep
# the legacy helper surface while deriving every value from typed ModelSpec data.
def _legacy_entry(model: ModelSpec) -> dict:
    return {
        "fast_path": model.fast.repo_id,
        "fast_revision": model.fast.revision,
        "official_path": model.official.repo_id,
        "official_revision": model.official.revision,
        "load_official": model.family.reference_adapter,
        "model_type": model.family.architecture,
        "uses_tokenizer": model.family.tokenizer_mode == "tokenizer",
        "size_category": model.size_category,
        "attention": model.family.attention,
        "dtypes": model.family.dtypes,
        "precisions": model.family.precisions,
        "state_transform": model.family.state_transform,
        "reference_container": model.family.reference_container,
    }


_TYPED_REGISTRY = get_model_registry()
FULL_MODEL_REGISTRY: dict[str, dict] = {
    model.id: _legacy_entry(model)
    for model in _TYPED_REGISTRY.values()
    if model.family.tokenizer_mode != "structure"
}
STRUCTURE_MODEL_REGISTRY: dict[str, dict] = {
    model.id: _legacy_entry(model)
    for model in _TYPED_REGISTRY.values()
    if model.family.tokenizer_mode == "structure"
}
MODEL_REGISTRY: dict[str, dict] = {
    model.family.architecture.lower(): _legacy_entry(model)
    for model in _TYPED_REGISTRY.values()
    if model.is_deep_reference and model.family.tokenizer_mode != "structure"
}
BACKENDS = (
    "sdpa",
    "flex_attention",
    "flash_attention_2",
    "flash_attention_3",
)


def get_models_by_size(*categories: str) -> dict[str, dict]:
    return {k: v for k, v in FULL_MODEL_REGISTRY.items() if v["size_category"] in categories}


# Pre-built key lists by size category
SMALL_MODEL_KEYS = list(get_models_by_size("small").keys())
MEDIUM_MODEL_KEYS = list(get_models_by_size("small", "medium").keys())
LARGE_MODEL_KEYS = list(get_models_by_size("large").keys())
XLARGE_MODEL_KEYS = list(get_models_by_size("xlarge").keys())
ALL_FULL_MODEL_KEYS = list(FULL_MODEL_REGISTRY.keys())
SEQUENCE_MODEL_KEYS = [
    k for k in ALL_FULL_MODEL_KEYS if FULL_MODEL_REGISTRY[k]["size_category"] != "structure"
]
STRUCTURE_MODEL_KEYS = list(STRUCTURE_MODEL_REGISTRY.keys())


def mark_by_size(
    keys: list[str],
    registry: dict[str, dict],
    extra_marks: list | None = None,
) -> list:
    """Return pytest.param list with appropriate markers based on size_category."""
    params = []
    for k in keys:
        marks = list(extra_marks or [])
        if registry[k]["size_category"] == "xlarge":
            marks.append(pytest.mark.large)
        elif registry[k]["size_category"] in ("large", "medium"):
            marks.append(pytest.mark.slow)
        params.append(pytest.param(k, marks=marks))
    return params


def tokenize_batch(
    model,
    model_key: str,
    sequences: list[str],
    device: torch.device,
    registry: dict[str, dict] | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of sequences, handling E1's sequence mode.

    Shared helper used across multiple test files to avoid duplication.
    """
    if registry is None:
        registry = FULL_MODEL_REGISTRY
    config = registry[model_key] if model_key in registry else MODEL_REGISTRY[model_key]

    if config["model_type"] == "E1":
        batch = model.model.prep_tokens.get_batch_kwargs(sequences, device=device)
        return {
            "input_ids": batch["input_ids"],
            "within_seq_position_ids": batch["within_seq_position_ids"],
            "global_position_ids": batch["global_position_ids"],
            "sequence_ids": batch["sequence_ids"],
            "attention_mask": (batch["sequence_ids"] != -1).long(),
        }
    tokenizer = model.tokenizer
    tokenized = tokenizer(sequences, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in tokenized.items()}


def add_model_specific_inputs(
    model_inputs: dict[str, torch.Tensor],
    model_type: str,
) -> dict[str, torch.Tensor]:
    """Add model-specific extra inputs (e.g. sequence_id for ESMC)."""
    if model_type == "ESMC":
        model_inputs["sequence_id"] = model_inputs["attention_mask"].to(dtype=torch.bool)
    return model_inputs


def random_sequences(n: int, min_len: int = 8, max_len: int = 64) -> list[str]:
    return [
        "M" + "".join(random.choices(CANONICAL_AAS, k=random.randint(min_len, max_len)))
        for _ in range(n)
    ]


def random_sequences_fixed_len(n: int, length: int = 64) -> list[str]:
    return ["M" + "".join(random.choices(CANONICAL_AAS, k=length - 1)) for _ in range(n)]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
