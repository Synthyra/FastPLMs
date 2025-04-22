import os
import torch
from functools import cache
from pathlib import Path
from huggingface_hub import snapshot_download
from modeling_esm_plusplus import ESMplusplusForMaskedLM, ESMplusplusConfig


@staticmethod
@cache
def data_root(model: str):
    if "INFRA_PROVIDER" in os.environ:
        return Path("")
    # Try to download from hugginface if it doesn't exist
    if model.startswith("esmc-300"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12"))
    elif model.startswith("esmc-600"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-600m-2024-12"))
    else:
        raise ValueError(f"{model=} is an invalid model name.")
    return path


def ESMplusplus_300M(device: torch.device | str = "cpu"):
    with torch.device(device):
        config = ESMplusplusConfig(
            hidden_size=960,
            num_attention_heads=15,
            num_hidden_layers=30,
        )
        model = ESMplusplusForMaskedLM(config)
    state_dict = torch.load(
        data_root("esmc-300") / "data/weights/esmc_300m_2024_12_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


def ESMplusplus_600M(device: torch.device | str = "cpu"):
    with torch.device(device):
        config = ESMplusplusConfig(
            hidden_size=1152,
            num_attention_heads=18,
            num_hidden_layers=36,
        )
        model = ESMplusplusForMaskedLM(config)
    state_dict = torch.load(
        data_root("esmc-600") / "data/weights/esmc_600m_2024_12_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


model_dict = {
    # Synthyra/ESM++small
    'Synthyra/ESMplusplus_small': ESMplusplus_300M,
    # Synthyra/ESM++large
    'Synthyra/ESMplusplus_large': ESMplusplus_600M,
}


for model_path, model_fn in model_dict.items():
    model = model_fn()
    model.config.auto_map = {
        "AutoConfig": "modeling_esm_plusplus.ESMplusplusConfig",
        "AutoModel": "modeling_esm_plusplus.ESMplusplusModel",
        "AutoModelForMaskedLM": "modeling_esm_plusplus.ESMplusplusForMaskedLM",
        "AutoModelForSequenceClassification": "modeling_esm_plusplus.ESMplusplusForSequenceClassification",
        "AutoModelForTokenClassification": "modeling_esm_plusplus.ESMplusplusForTokenClassification"
    }
    model.push_to_hub(model_path, safe_serialization=False)
