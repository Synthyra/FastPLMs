import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import HfApi, login
from transformers import AutoModelForMaskedLM

from fastplms.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    _ESMC_CHECKPOINT_SPECS,
    _resolve_esmc_checkpoint_key,
)
from fastplms.weight_parity_utils import assert_state_dict_equal, assert_model_parameters_fp32


MODEL_DICT: Dict[str, Tuple[str, str]] = {
    "Synthyra/ESMplusplus_small": ("biohub/ESMC-300M", "README_small.md"),
    "Synthyra/ESMplusplus_large": ("biohub/ESMC-600M", "README_large.md"),
    "Synthyra/ESMplusplus_6B": ("biohub/ESMC-6B", "README_6B.md"),
}

LICENSE_DICT: Dict[str, str] = {
    "Synthyra/ESMplusplus_small": "LICENSE_small",
    "Synthyra/ESMplusplus_large": "LICENSE_large",
    "Synthyra/ESMplusplus_6B": "LICENSE_6B",
}

HUB_AUTO_MAP = {
    "AutoConfig": "modeling_esm_plusplus.ESMplusplusConfig",
    "AutoModel": "modeling_esm_plusplus.ESMplusplusModel",
    "AutoModelForMaskedLM": "modeling_esm_plusplus.ESMplusplusForMaskedLM",
    "AutoModelForSequenceClassification": "modeling_esm_plusplus.ESMplusplusForSequenceClassification",
    "AutoModelForTokenClassification": "modeling_esm_plusplus.ESMplusplusForTokenClassification",
}


def _build_config(esmc_model_key: str) -> ESMplusplusConfig:
    spec = _ESMC_CHECKPOINT_SPECS[_resolve_esmc_checkpoint_key(esmc_model_key)]
    config = ESMplusplusConfig(
        hidden_size=spec["hidden_size"],
        num_attention_heads=spec["num_attention_heads"],
        num_hidden_layers=spec["num_hidden_layers"],
    )
    config.architectures = ["ESMplusplusForMaskedLM"]
    config.auto_map = HUB_AUTO_MAP
    config.tie_word_embeddings = False
    return config


def _upload_repo_files(api: HfApi, repo_id: str, script_root: str, readme_name: str) -> None:
    readme_path = os.path.join(script_root, readme_name)
    assert os.path.exists(readme_path), f"Missing model card: {readme_path}"
    from update_HF import build_composite

    composite_code = build_composite(
        "fastplms/esm_plusplus/modeling_esm_plusplus.py",
        include_embedding_mixin=True,
    )
    compile(composite_code, "modeling_esm_plusplus.py", "exec")
    with tempfile.TemporaryDirectory() as tmpdir:
        composite_path = Path(tmpdir) / "modeling_esm_plusplus.py"
        composite_path.write_text(composite_code, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(composite_path),
            path_in_repo="modeling_esm_plusplus.py",
            repo_id=repo_id,
            repo_type="model",
        )
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    license_path = os.path.join(script_root, LICENSE_DICT[repo_id])
    assert os.path.exists(license_path), f"Missing license: {license_path}"
    api.upload_file(
        path_or_fileobj=license_path,
        path_in_repo="LICENSE",
        repo_id=repo_id,
        repo_type="model",
    )


def _resolve_repo_items(repo_ids: Optional[List[str]]) -> List[Tuple[str, str, str]]:
    if repo_ids is None or len(repo_ids) == 0:
        return [
            (repo_id, esmc_model_key, readme_name)
            for repo_id, (esmc_model_key, readme_name) in MODEL_DICT.items()
        ]

    selected_items: List[Tuple[str, str, str]] = []
    for repo_id in repo_ids:
        assert repo_id in MODEL_DICT, (
            f"Unknown repo_id {repo_id}. "
            f"Valid options: {sorted(MODEL_DICT.keys())}"
        )
        esmc_model_key, readme_name = MODEL_DICT[repo_id]
        selected_items.append((repo_id, esmc_model_key, readme_name))
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
    # py -m fastplms.esm_plusplus.get_weights
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
    _login_if_requested(args)
    api = HfApi()

    script_root = os.path.dirname(os.path.abspath(__file__))

    for repo_id, esmc_model_key, readme_name in _resolve_repo_items(args.repo_ids):
        config = _build_config(esmc_model_key)
        if args.skip_weights:
            if args.dry_run:
                print(f"[skip-weights][dry-run] validated config metadata for {repo_id}")
                continue
            config.push_to_hub(repo_id)
            _upload_repo_files(api, repo_id, script_root, readme_name)
            print(f"[skip-weights] uploaded config for {repo_id}")
            continue

        model = ESMplusplusForMaskedLM.from_pretrained_esm(
            esmc_model_key,
            device=torch.device("cpu"),
        ).eval().cpu().to(torch.float32)
        model.config.architectures = ["ESMplusplusForMaskedLM"]
        model.config.auto_map = HUB_AUTO_MAP
        model.config.tie_word_embeddings = False
        tokenizer = model.tokenizer

        assert_model_parameters_fp32(
            model=model,
            model_name=f"mapped ESM++ model ({esmc_model_key})",
        )

        if args.dry_run:
            print(f"[dry_run] validated ESM++ conversion for {repo_id} <- {esmc_model_key}")
            continue

        tokenizer.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
        _upload_repo_files(api, repo_id, script_root, readme_name)
        downloaded_model = AutoModelForMaskedLM.from_pretrained(
            repo_id,
            dtype=torch.float32,
            device_map="cpu",
            force_download=True,
            trust_remote_code=True,
        )
        assert_state_dict_equal(
            reference_state_dict=model.state_dict(),
            candidate_state_dict=downloaded_model.state_dict(),
            context=f"ESMC/ESM++ weight parity post-download ({repo_id})",
        )
