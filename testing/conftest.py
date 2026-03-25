import os
import random
from typing import Dict, List, Tuple

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "slow: loads two models simultaneously (compliance tests)")
    config.addinivalue_line("markers", "large: requires 24+ GB VRAM (3B parameter models)")
    config.addinivalue_line("markers", "structure: structure prediction models (Boltz2, ESMFold)")


# Standalone scripts that are not pytest tests
collect_ignore = [
    os.path.join(os.path.dirname(__file__), "test_contact_maps.py"),
    os.path.join(os.path.dirname(__file__), "compliance.py"),
    os.path.join(os.path.dirname(__file__), "throughput.py"),
    os.path.join(os.path.dirname(__file__), "run_boltz2_compliance.py"),
]

CANONICAL_AAS = "ACDEFGHIKLMNPQRSTVWY"
SEED = 42
DEFAULT_BATCH_SIZE = 4
MAX_EMBED_LEN = 128

# Default registry: one small model per family for fast CI
MODEL_REGISTRY: Dict[str, Dict] = {
    "esm2": {
        "fast_path": "Synthyra/ESM2-8M",
        "official_path": "facebook/esm2_t6_8M_UR50D",
        "load_official": "testing.official.esm2",
        "model_type": "ESM2",
        "uses_tokenizer": True,
    },
    "esmc": {
        "fast_path": "Synthyra/ESMplusplus_small",
        "official_path": "esmc-300",
        "load_official": "testing.official.esm_plusplus",
        "model_type": "ESMC",
        "uses_tokenizer": True,
    },
    "e1": {
        "fast_path": "Synthyra/Profluent-E1-150M",
        "official_path": "Profluent-Bio/E1-150m",
        "load_official": "testing.official.e1",
        "model_type": "E1",
        "uses_tokenizer": False,
    },
    "dplm": {
        "fast_path": "Synthyra/DPLM-150M",
        "official_path": "airkingbd/dplm_150m",
        "load_official": "testing.official.dplm",
        "model_type": "DPLM",
        "uses_tokenizer": True,
    },
    "dplm2": {
        "fast_path": "Synthyra/DPLM2-150M",
        "official_path": "airkingbd/dplm2_150m",
        "load_official": "testing.official.dplm2",
        "model_type": "DPLM2",
        "uses_tokenizer": True,
    },
    "ankh": {
        "fast_path": "Synthyra/ANKH_base",
        "official_path": "ElnaggarLab/ankh-base",
        "load_official": "testing.official.ankh",
        "model_type": "ANKH",
        "uses_tokenizer": True,
    },
}

# Full registry: every checkpoint across all model families
FULL_MODEL_REGISTRY: Dict[str, Dict] = {
    # ESM2 family
    "esm2_8m": {
        "fast_path": "Synthyra/ESM2-8M",
        "official_path": "facebook/esm2_t6_8M_UR50D",
        "load_official": "testing.official.esm2",
        "model_type": "ESM2",
        "uses_tokenizer": True,
        "size_category": "small",
    },
    "esm2_35m": {
        "fast_path": "Synthyra/ESM2-35M",
        "official_path": "facebook/esm2_t12_35M_UR50D",
        "load_official": "testing.official.esm2",
        "model_type": "ESM2",
        "uses_tokenizer": True,
        "size_category": "small",
    },
    "esm2_150m": {
        "fast_path": "Synthyra/ESM2-150M",
        "official_path": "facebook/esm2_t30_150M_UR50D",
        "load_official": "testing.official.esm2",
        "model_type": "ESM2",
        "uses_tokenizer": True,
        "size_category": "medium",
    },
    "esm2_650m": {
        "fast_path": "Synthyra/ESM2-650M",
        "official_path": "facebook/esm2_t33_650M_UR50D",
        "load_official": "testing.official.esm2",
        "model_type": "ESM2",
        "uses_tokenizer": True,
        "size_category": "large",
    },
    "esm2_3b": {
        "fast_path": "Synthyra/ESM2-3B",
        "official_path": "facebook/esm2_t36_3B_UR50D",
        "load_official": "testing.official.esm2",
        "model_type": "ESM2",
        "uses_tokenizer": True,
        "size_category": "xlarge",
    },
    # ESM++ family
    "esmc_small": {
        "fast_path": "Synthyra/ESMplusplus_small",
        "official_path": "esmc-300",
        "load_official": "testing.official.esm_plusplus",
        "model_type": "ESMC",
        "uses_tokenizer": True,
        "size_category": "medium",
    },
    "esmc_large": {
        "fast_path": "Synthyra/ESMplusplus_large",
        "official_path": "esmc-600",
        "load_official": "testing.official.esm_plusplus",
        "model_type": "ESMC",
        "uses_tokenizer": True,
        "size_category": "large",
    },
    # E1 family
    "e1_150m": {
        "fast_path": "Synthyra/Profluent-E1-150M",
        "official_path": "Profluent-Bio/E1-150m",
        "load_official": "testing.official.e1",
        "model_type": "E1",
        "uses_tokenizer": False,
        "size_category": "small",
    },
    "e1_300m": {
        "fast_path": "Synthyra/Profluent-E1-300M",
        "official_path": "Profluent-Bio/E1-300m",
        "load_official": "testing.official.e1",
        "model_type": "E1",
        "uses_tokenizer": False,
        "size_category": "medium",
    },
    "e1_600m": {
        "fast_path": "Synthyra/Profluent-E1-600M",
        "official_path": "Profluent-Bio/E1-600m",
        "load_official": "testing.official.e1",
        "model_type": "E1",
        "uses_tokenizer": False,
        "size_category": "large",
    },
    # DPLM family
    "dplm_150m": {
        "fast_path": "Synthyra/DPLM-150M",
        "official_path": "airkingbd/dplm_150m",
        "load_official": "testing.official.dplm",
        "model_type": "DPLM",
        "uses_tokenizer": True,
        "size_category": "small",
    },
    "dplm_650m": {
        "fast_path": "Synthyra/DPLM-650M",
        "official_path": "airkingbd/dplm_650m",
        "load_official": "testing.official.dplm",
        "model_type": "DPLM",
        "uses_tokenizer": True,
        "size_category": "large",
    },
    "dplm_3b": {
        "fast_path": "Synthyra/DPLM-3B",
        "official_path": "airkingbd/dplm_3b",
        "load_official": "testing.official.dplm",
        "model_type": "DPLM",
        "uses_tokenizer": True,
        "size_category": "xlarge",
    },
    # DPLM2 family
    "dplm2_150m": {
        "fast_path": "Synthyra/DPLM2-150M",
        "official_path": "airkingbd/dplm2_150m",
        "load_official": "testing.official.dplm2",
        "model_type": "DPLM2",
        "uses_tokenizer": True,
        "size_category": "small",
    },
    "dplm2_650m": {
        "fast_path": "Synthyra/DPLM2-650M",
        "official_path": "airkingbd/dplm2_650m",
        "load_official": "testing.official.dplm2",
        "model_type": "DPLM2",
        "uses_tokenizer": True,
        "size_category": "large",
    },
    "dplm2_3b": {
        "fast_path": "Synthyra/DPLM2-3B",
        "official_path": "airkingbd/dplm2_3b",
        "load_official": "testing.official.dplm2",
        "model_type": "DPLM2",
        "uses_tokenizer": True,
        "size_category": "xlarge",
    },
    # ANKH family
    "ankh_base": {
        "fast_path": "Synthyra/ANKH_base",
        "official_path": "ElnaggarLab/ankh-base",
        "load_official": "testing.official.ankh",
        "model_type": "ANKH",
        "uses_tokenizer": True,
        "size_category": "medium",
    },
    "ankh_large": {
        "fast_path": "Synthyra/ANKH_large",
        "official_path": "ElnaggarLab/ankh-large",
        "load_official": "testing.official.ankh",
        "model_type": "ANKH",
        "uses_tokenizer": True,
        "size_category": "large",
    },
    "ankh2_large": {
        "fast_path": "Synthyra/ANKH2_large",
        "official_path": "ElnaggarLab/ankh2-ext2",
        "load_official": "testing.official.ankh",
        "model_type": "ANKH",
        "uses_tokenizer": True,
        "size_category": "large",
    },
    "ankh3_large": {
        "fast_path": "Synthyra/ANKH3_large",
        "official_path": "ElnaggarLab/ankh3-large",
        "load_official": "testing.official.ankh",
        "model_type": "ANKH",
        "uses_tokenizer": True,
        "size_category": "large",
    },
    "ankh3_xl": {
        "fast_path": "Synthyra/ANKH3_xl",
        "official_path": "ElnaggarLab/ankh3-xl",
        "load_official": "testing.official.ankh",
        "model_type": "ANKH",
        "uses_tokenizer": True,
        "size_category": "xlarge",
    },
}

# Structure prediction models (separate API, not MaskedLM)
STRUCTURE_MODEL_REGISTRY: Dict[str, Dict] = {
    "boltz2": {
        "fast_path": "Synthyra/Boltz2",
        "model_type": "Boltz2",
        "size_category": "structure",
    },
    "esmfold": {
        "fast_path": "Synthyra/FastESMFold",
        "model_type": "ESMFold",
        "size_category": "structure",
    },
}

BACKENDS = ("sdpa", "flex", "kernels_flash")


def get_models_by_size(*categories: str) -> Dict[str, Dict]:
    return {k: v for k, v in FULL_MODEL_REGISTRY.items() if v["size_category"] in categories}


# Pre-built key lists by size category
SMALL_MODEL_KEYS = list(get_models_by_size("small").keys())
MEDIUM_MODEL_KEYS = list(get_models_by_size("small", "medium").keys())
LARGE_MODEL_KEYS = list(get_models_by_size("large").keys())
XLARGE_MODEL_KEYS = list(get_models_by_size("xlarge").keys())
ALL_FULL_MODEL_KEYS = list(FULL_MODEL_REGISTRY.keys())
SEQUENCE_MODEL_KEYS = [k for k in ALL_FULL_MODEL_KEYS if FULL_MODEL_REGISTRY[k]["size_category"] != "structure"]
STRUCTURE_MODEL_KEYS = list(STRUCTURE_MODEL_REGISTRY.keys())


def mark_by_size(keys: List[str], registry: Dict[str, Dict], extra_marks: List = None) -> List:
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
    sequences: List[str],
    device: torch.device,
    registry: Dict[str, Dict] = None,
) -> Dict[str, torch.Tensor]:
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
    model_inputs: Dict[str, torch.Tensor],
    model_type: str,
) -> Dict[str, torch.Tensor]:
    """Add model-specific extra inputs (e.g. sequence_id for ESMC)."""
    if model_type == "ESMC":
        model_inputs["sequence_id"] = model_inputs["attention_mask"].to(dtype=torch.bool)
    return model_inputs


def random_sequences(n: int, min_len: int = 8, max_len: int = 64) -> List[str]:
    return [
        "M" + "".join(random.choices(CANONICAL_AAS, k=random.randint(min_len, max_len)))
        for _ in range(n)
    ]


def random_sequences_fixed_len(n: int, length: int = 64) -> List[str]:
    return [
        "M" + "".join(random.choices(CANONICAL_AAS, k=length - 1))
        for _ in range(n)
    ]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
