"""Copy Biohub ESMFold2 checkpoints into FastPLMs AutoModel repos.

Usage:
    python -m fastplms.esmfold2.get_weights
    python -m fastplms.esmfold2.get_weights --skip-weights
    python -m fastplms.esmfold2.get_weights --repo_ids Synthyra/ESMFold2-Fast
"""

import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, login

from fastplms.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.esmfold2.modeling_esmfold2 import ESMFold2Model

SOURCE_REPOS = {
    "Synthyra/ESMFold2": "biohub/ESMFold2",
    "Synthyra/ESMFold2-Fast": "biohub/ESMFold2-Fast",
}
SHARD_SIZE = "5GB"
AUTO_MAP = {
    "AutoConfig": "configuration_esmfold2.ESMFold2Config",
    "AutoModel": "modeling_esmfold2.ESMFold2Model",
}
IGNORE_PATTERNS = [
    "__pycache__/*",
    "*.pyc",
    "get_weights.py",
]


def _prepare_config(source_repo: str) -> ESMFold2Config:
    config = ESMFold2Config.from_pretrained(source_repo)
    config.auto_map = AUTO_MAP
    config.architectures = ["ESMFold2Model"]
    return config


def _upload_code(api: HfApi, repo_id: str, package_dir: Path) -> None:
    api.upload_folder(
        folder_path=str(package_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=IGNORE_PATTERNS,
    )


def convert_and_push(
    repo_ids: list[str] | None = None,
    hf_token: str | None = None,
    dry_run: bool = False,
    skip_weights: bool = False,
) -> None:
    if hf_token is not None:
        login(token=hf_token)

    api = HfApi()
    package_dir = Path(__file__).resolve().parent
    targets = repo_ids if repo_ids is not None else list(SOURCE_REPOS)

    for target_repo in targets:
        assert target_repo in SOURCE_REPOS, (
            f"Unknown repo_id {target_repo}. Expected one of {sorted(SOURCE_REPOS)}."
        )
        source_repo = SOURCE_REPOS[target_repo]
        config = _prepare_config(source_repo)

        if dry_run:
            print(f"[dry-run] validated config and code for {target_repo} <- {source_repo}")
            continue

        if skip_weights:
            config.push_to_hub(target_repo)
            _upload_code(api, target_repo, package_dir)
            print(f"[skip-weights] uploaded config/code for {target_repo}")
            continue

        print(f"Loading {source_repo} with FastPLMs ESMFold2 code...")
        model = ESMFold2Model.from_pretrained(
            source_repo,
            config=config,
            load_esmc=False,
            dtype=torch.float32,
        )
        print(f"Pushing {target_repo}...")
        model.push_to_hub(target_repo, max_shard_size=SHARD_SIZE)
        _upload_code(api, target_repo, package_dir)
        print(f"Done. Model available at https://huggingface.co/{target_repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_ids",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ["HF_TOKEN"] if "HF_TOKEN" in os.environ else None,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
    )
    args = parser.parse_args()

    convert_and_push(
        repo_ids=args.repo_ids,
        hf_token=args.hf_token,
        dry_run=args.dry_run,
        skip_weights=args.skip_weights,
    )
