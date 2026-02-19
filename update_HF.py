"""
Data-driven HuggingFace upload script for all FastPLMs models.

Runs weight conversion scripts for each family, then uploads
modeling code, embedding_mixin, entrypoint_setup, readmes, and licenses
to each HF repo.

Usage:
    py -m update_HF
    py -m update_HF --hf_token YOUR_TOKEN
    py -m update_HF --families esm2 dplm
    py -m update_HF --skip-weights
"""

import argparse
import platform
import subprocess

from huggingface_hub import HfApi, login


MODEL_REGISTRY = [
    {
        "family": "esm2",
        "repo_ids": [
            "Synthyra/ESM2-8M",
            "Synthyra/ESM2-35M",
            "Synthyra/ESM2-150M",
            "Synthyra/ESM2-650M",
            "Synthyra/ESM2-3B",
            "Synthyra/FastESM2_650",
        ],
        "files": {
            "esm2/modeling_fastesm.py": "modeling_fastesm.py",
        },
        "readme_map": {
            "Synthyra/ESM2-8M": "readmes/fastesm2_readme.md",
            "Synthyra/ESM2-35M": "readmes/fastesm2_readme.md",
            "Synthyra/ESM2-150M": "readmes/fastesm2_readme.md",
            "Synthyra/ESM2-650M": "readmes/fastesm2_readme.md",
            "Synthyra/ESM2-3B": "readmes/fastesm2_readme.md",
            "Synthyra/FastESM2_650": "readmes/fastesm_650_readme.md",
        },
        "license": "LICENSE",
        "weight_module": "esm2.get_esm2_weights",
    },
    {
        "family": "esmplusplus",
        "repo_ids": [
            "Synthyra/ESMplusplus_small",
            "Synthyra/ESMplusplus_large",
        ],
        "files": {
            "esm_plusplus/modeling_esm_plusplus.py": "modeling_esm_plusplus.py",
        },
        "readme_map": {
            "Synthyra/ESMplusplus_small": "readmes/esm_plusplus_small_readme.md",
            "Synthyra/ESMplusplus_large": "readmes/esm_plusplus_large_readme.md",
        },
        "license": "LICENSE",
        "weight_module": "esm_plusplus.get_esmc_weights",
    },
    {
        "family": "e1",
        "repo_ids": [
            "Synthyra/Profluent-E1-150M",
            "Synthyra/Profluent-E1-300M",
            "Synthyra/Profluent-E1-600M",
        ],
        "files": {
            "e1_fastplms/modeling_e1.py": "modeling_e1.py",
            "e1_fastplms/tokenizer.json": "tokenizer.json",
        },
        "readme_map": {
            "Synthyra/Profluent-E1-150M": "readmes/e1_readme.md",
            "Synthyra/Profluent-E1-300M": "readmes/e1_readme.md",
            "Synthyra/Profluent-E1-600M": "readmes/e1_readme.md",
        },
        "license": "LICENSE",
        "weight_module": "e1_fastplms.get_e1_weights",
    },
    {
        "family": "dplm",
        "repo_ids": [
            "Synthyra/DPLM-150M",
            "Synthyra/DPLM-650M",
            "Synthyra/DPLM-3B",
        ],
        "files": {
            "dplm_fastplms/modeling_dplm.py": "modeling_dplm.py",
            "dplm_fastplms/base_tokenizer.py": "base_tokenizer.py",
        },
        "readme_map": {
            "Synthyra/DPLM-150M": "readmes/dplm_readme.md",
            "Synthyra/DPLM-650M": "readmes/dplm_readme.md",
            "Synthyra/DPLM-3B": "readmes/dplm_readme.md",
        },
        "license": "LICENSE",
        "weight_module": "dplm_fastplms.get_dplm_weights",
    },
    {
        "family": "dplm2",
        "repo_ids": [
            "Synthyra/DPLM2-150M",
            "Synthyra/DPLM2-650M",
            "Synthyra/DPLM2-3B",
        ],
        "files": {
            "dplm2_fastplms/modeling_dplm2.py": "modeling_dplm2.py",
            "dplm2_fastplms/base_tokenizer.py": "base_tokenizer.py",
        },
        "readme_map": {
            "Synthyra/DPLM2-150M": "readmes/dplm2_readme.md",
            "Synthyra/DPLM2-650M": "readmes/dplm2_readme.md",
            "Synthyra/DPLM2-3B": "readmes/dplm2_readme.md",
        },
        "license": "LICENSE",
        "weight_module": "dplm2_fastplms.get_dplm2_weights",
    },
    {
        "family": "ankh",
        "repo_ids": [
            "Synthyra/ANKH_base",
            "Synthyra/ANKH_large",
            "Synthyra/ANKH2_large",
        ],
        "files": {},
        "readme_map": {},
        "license": "LICENSE",
        "weight_module": None,
    },
    {
        "family": "boltz",
        "repo_ids": [
            "Synthyra/Boltz2",
        ],
        "files": {},
        "readme_map": {},
        "license": "LICENSE",
        "weight_module": "boltz_fastplms.get_boltz2_weights",
    },
]

SHARED_FILES = {
    "embedding_mixin.py": "embedding_mixin.py",
    "entrypoint_setup.py": "entrypoint_setup.py",
}


def _run_weight_scripts(families: list[str] | None, hf_token: str | None) -> None:
    python_cmd = "python" if platform.system().lower() == "linux" else "py"
    for entry in MODEL_REGISTRY:
        if families is not None and entry["family"] not in families:
            continue
        module = entry["weight_module"]
        if module is None:
            continue
        command = [python_cmd, "-m", module]
        if hf_token is not None:
            command.extend(["--hf_token", hf_token])
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)


def _upload_files(api: HfApi, families: list[str] | None) -> None:
    for entry in MODEL_REGISTRY:
        if families is not None and entry["family"] not in families:
            continue

        for repo_id in entry["repo_ids"]:
            print(f"\nUploading to {repo_id}")

            for local_path, repo_path in entry["files"].items():
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model",
                )

            for local_path, repo_path in SHARED_FILES.items():
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model",
                )

            if entry["license"]:
                api.upload_file(
                    path_or_fileobj=entry["license"],
                    path_in_repo="LICENSE",
                    repo_id=repo_id,
                    repo_type="model",
                )

            readme_path = entry["readme_map"].get(repo_id)
            if readme_path:
                api.upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload FastPLMs models to HuggingFace")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--skip-weights", action="store_true", help="Skip weight conversion scripts")
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    if not args.skip_weights:
        _run_weight_scripts(args.families, args.hf_token)

    api = HfApi()
    _upload_files(api, args.families)
    print("\nDone.")
