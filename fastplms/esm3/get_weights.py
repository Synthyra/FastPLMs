import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import HfApi, login
from transformers import AutoModel

from fastplms.esm3.modeling_esm3 import (
    ESM3_OPEN_SMALL,
    FastESM3Config,
    FastESM3Model,
    _ESM3_CHECKPOINT_SPECS,
    _resolve_esm3_checkpoint_key,
)
from fastplms.weight_parity_utils import assert_model_parameters_fp32, assert_state_dict_equal


MODEL_DICT: Dict[str, Tuple[str, str]] = {
    "Synthyra/ESM3_small": ("esm3-sm-open-v1", "README.md"),
}

HUB_AUTO_MAP = {
    "AutoConfig": "modeling_esm3.FastESM3Config",
    "AutoModel": "modeling_esm3.FastESM3Model",
    "AutoModelForMaskedLM": "modeling_esm3.FastESM3Model",
}


def _build_config(model_name: str) -> FastESM3Config:
    key = _resolve_esm3_checkpoint_key(model_name)
    spec = _ESM3_CHECKPOINT_SPECS[key]
    config = FastESM3Config(
        hidden_size=spec["hidden_size"],
        num_attention_heads=spec["num_attention_heads"],
        num_vector_heads=spec["num_vector_heads"],
        num_hidden_layers=spec["num_hidden_layers"],
        model_name=key,
    )
    config.architectures = ["FastESM3Model"]
    config.auto_map = HUB_AUTO_MAP
    config.tie_word_embeddings = False
    return config


def _upload_repo_files(api: HfApi, repo_id: str, script_root: Path, readme_name: str) -> None:
    readme_path = script_root / readme_name
    assert readme_path.exists(), f"Missing model card: {readme_path}"
    api.upload_file(
        path_or_fileobj=str(script_root / "modeling_esm3.py"),
        path_in_repo="modeling_esm3.py",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    license_path = script_root / "LICENSE"
    assert license_path.exists(), f"Missing license: {license_path}"
    api.upload_file(
        path_or_fileobj=str(license_path),
        path_in_repo="LICENSE",
        repo_id=repo_id,
        repo_type="model",
    )


def _resolve_repo_items(repo_ids: Optional[List[str]]) -> List[Tuple[str, str, str]]:
    if repo_ids is None or len(repo_ids) == 0:
        return [
            (repo_id, esm3_model_key, readme_name)
            for repo_id, (esm3_model_key, readme_name) in MODEL_DICT.items()
        ]

    selected_items: List[Tuple[str, str, str]] = []
    for repo_id in repo_ids:
        assert repo_id in MODEL_DICT, (
            f"Unknown repo_id {repo_id}. "
            f"Valid options: {sorted(MODEL_DICT.keys())}"
        )
        esm3_model_key, readme_name = MODEL_DICT[repo_id]
        selected_items.append((repo_id, esm3_model_key, readme_name))
    return selected_items


def _token_from_environment() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if key in os.environ and len(os.environ[key]) > 0:
            return os.environ[key]
    return None


def _login_if_requested(args: argparse.Namespace) -> None:
    token = args.hf_token
    if token is None:
        token = _token_from_environment()
    if token is not None:
        assert len(token) > 0, "HF token cannot be empty."
        login(token=token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Deprecated. Prefer HF_TOKEN in the environment so tokens are not in shell history.",
    )
    parser.add_argument("--repo_ids", nargs="*", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent
    _login_if_requested(args)
    api = HfApi()

    for repo_id, esm3_model_key, readme_name in _resolve_repo_items(args.repo_ids):
        config = _build_config(esm3_model_key)
        if args.skip_weights:
            if args.dry_run:
                print(f"[skip-weights][dry-run] validated config metadata for {repo_id}")
                continue
            config.push_to_hub(repo_id)
            _upload_repo_files(api, repo_id, script_root, readme_name)
            print(f"[skip-weights] uploaded config for {repo_id}")
            continue

        model = FastESM3Model.from_pretrained_esm(
            ESM3_OPEN_SMALL,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        model.config.architectures = ["FastESM3Model"]
        model.config.auto_map = HUB_AUTO_MAP
        model.config.tie_word_embeddings = False
        tokenizer = model.tokenizer

        assert_model_parameters_fp32(
            model=model,
            model_name=f"mapped ESM3 model ({esm3_model_key})",
        )

        if args.dry_run:
            print(f"[dry_run] validated ESM3 conversion for {repo_id} <- {esm3_model_key}")
            continue

        tokenizer.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
        _upload_repo_files(api, repo_id, script_root, readme_name)
        downloaded_model = AutoModel.from_pretrained(
            repo_id,
            dtype=torch.float32,
            device_map="cpu",
            force_download=True,
            trust_remote_code=True,
        )
        assert_state_dict_equal(
            reference_state_dict=model.state_dict(),
            candidate_state_dict=downloaded_model.state_dict(),
            context=f"ESM3 weight parity post-download ({repo_id})",
        )
