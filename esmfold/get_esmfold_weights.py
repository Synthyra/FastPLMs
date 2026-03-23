"""Convert Synthyra/ESMFold-v1 weights to FastESMFold format and push to HuggingFace.

Usage:
    py -m esmfold.get_esmfold_weights
    py -m esmfold.get_esmfold_weights --dry_run
    py -m esmfold.get_esmfold_weights --hf_token <token>
"""
import os
import sys

import torch

from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, EsmConfig

from esmfold.modeling_fast_esmfold import FastEsmFoldConfig, FastEsmForProteinFolding

SOURCE_REPO = "Synthyra/ESMFold-v1"
TARGET_REPO = "Synthyra/FastESMFold"
SHARD_SIZE = "5GB"


def convert_and_push(
    hf_token: str = None,
    dry_run: bool = False,
    attn_backend: str = "sdpa",
) -> None:
    if hf_token is not None:
        login(token=hf_token)

    api = HfApi()
    script_root = os.path.dirname(os.path.abspath(__file__))

    print(f"Loading source model from {SOURCE_REPO}...")
    source_config = EsmConfig.from_pretrained(SOURCE_REPO)

    # Build FastEsmFoldConfig from the source ESMFold config
    config_dict = source_config.to_dict()
    config_dict["attn_backend"] = attn_backend
    config_dict["ttt_config"] = {
        "lr": 4e-4,
        "steps": 30,
        "ags": 4,
        "batch_size": 4,
        "mask_ratio": 0.15,
        "lora_rank": 8,
        "lora_alpha": 32.0,
    }
    config_dict["auto_map"] = {
        "AutoConfig": "modeling_fast_esmfold.FastEsmFoldConfig",
        "AutoModel": "modeling_fast_esmfold.FastEsmForProteinFolding",
    }
    config = FastEsmFoldConfig(**config_dict)

    print("Creating FastEsmForProteinFolding model...")
    model = FastEsmForProteinFolding(config)

    # Load source weights (EsmFoldWithLMHead = EsmForProteinFolding + mlm_head)
    print(f"Loading weights from {SOURCE_REPO}...")
    source_state_dict = torch.hub.load_state_dict_from_url(
        f"https://huggingface.co/{SOURCE_REPO}/resolve/main/model.safetensors",
        map_location="cpu",
    ) if False else None

    # Use from_pretrained to load weights into a temporary model, then transfer
    from transformers import EsmForProteinFolding as HFEsmFold

    # Load the source model state dict
    source_model = HFEsmFold.from_pretrained(
        SOURCE_REPO,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    source_sd = source_model.state_dict()

    # Load what we can (backbone weights map directly since FastEsmBackbone
    # has the same module structure as EsmModel)
    missing, unexpected = model.load_state_dict(source_sd, strict=False)

    print(f"Missing keys: {len(missing)}")
    for k in sorted(missing):
        print(f"  {k}")
    print(f"Unexpected keys: {len(unexpected)}")
    for k in sorted(unexpected):
        print(f"  {k}")

    if dry_run:
        print(f"[dry_run] Validated weight transfer for {TARGET_REPO} <- {SOURCE_REPO}")
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        return

    print(f"Pushing model to {TARGET_REPO}...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    tokenizer.push_to_hub(TARGET_REPO)
    model.push_to_hub(TARGET_REPO, max_shard_size=SHARD_SIZE)

    # Upload modeling file
    api.upload_file(
        path_or_fileobj=os.path.join(script_root, "modeling_fast_esmfold.py"),
        path_in_repo="modeling_fast_esmfold.py",
        repo_id=TARGET_REPO,
        repo_type="model",
    )

    print(f"Done. Model available at https://huggingface.co/{TARGET_REPO}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
    )
    parser.add_argument(
        "--attn_backend",
        type=str,
        default="sdpa",
    )
    args = parser.parse_args()

    convert_and_push(
        hf_token=args.hf_token,
        dry_run=args.dry_run,
        attn_backend=args.attn_backend,
    )
