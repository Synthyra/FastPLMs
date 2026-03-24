import os
import random
from typing import Dict, List

import torch

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

MODEL_REGISTRY: Dict[str, Dict] = {
    "esm2": {
        "fast_path": "Synthyra/ESM2-8M",
        "official_path": "facebook/esm2_t6_8M_UR50D",
        "load_official": "esm2.load_official",
        "model_type": "ESM2",
        "uses_tokenizer": True,
    },
    "esmc": {
        "fast_path": "Synthyra/ESMplusplus_small",
        "official_path": "esmc-300",
        "load_official": "esm_plusplus.load_official",
        "model_type": "ESMC",
        "uses_tokenizer": True,
    },
    "e1": {
        "fast_path": "Synthyra/Profluent-E1-150M",
        "official_path": "Profluent-Bio/E1-150m",
        "load_official": "e1_fastplms.load_official",
        "model_type": "E1",
        "uses_tokenizer": False,
    },
    "dplm": {
        "fast_path": "Synthyra/DPLM-150M",
        "official_path": "airkingbd/dplm_150m",
        "load_official": "dplm_fastplms.load_official",
        "model_type": "DPLM",
        "uses_tokenizer": True,
    },
    "dplm2": {
        "fast_path": "Synthyra/DPLM2-150M",
        "official_path": "airkingbd/dplm2_150m",
        "load_official": "dplm2_fastplms.load_official",
        "model_type": "DPLM2",
        "uses_tokenizer": True,
    },
}

BACKENDS = ("sdpa", "flex", "kernels_flash")


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
