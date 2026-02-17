import argparse
import subprocess

from huggingface_hub import HfApi, login


FAST_ESM_MODELS = [
    'Synthyra/ESM2-8M',
    'Synthyra/ESM2-35M',
    'Synthyra/ESM2-150M',
    'Synthyra/ESM2-650M',
    'Synthyra/ESM2-3B',
    'Synthyra/FastESM2_650'
]

ESM_PLUSPLUS_MODELS = [
    'Synthyra/ESMplusplus_small',
    'Synthyra/ESMplusplus_large',
]

E1_MODELS = [
    'Synthyra/Profluent-E1-150M',
    'Synthyra/Profluent-E1-300M',
    'Synthyra/Profluent-E1-600M',
]

ANKH_MODELS = [
    'Synthyra/ANKH_base',
    'Synthyra/ANKH_large',
    'Synthyra/ANKH2_large'
]

BOLTZ_MODELS = [
    'Synthyra/Boltz2',
]


def _run_get_weights_scripts(hf_token: str | None) -> None:
    import platform
    python_cmd = "python" if platform.system().lower() == "linux" else "py"
    modules = [
        "boltz_fastplms.get_boltz2_weights",
        "e1_fastplms.get_e1_weights",
        "esm_plusplus.get_esmc_weights",
        "esm2.get_esm2_weights",
    ]
    for module in modules:
        command = [python_cmd, "-m", module]
        if hf_token is not None:
            command.extend(["--hf_token", hf_token])
        print(f"Running {' '.join(command)}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    # py -m update_HF
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_token', type=str, default=None)
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    _run_get_weights_scripts(hf_token=args.hf_token)

    api = HfApi()

    for path in FAST_ESM_MODELS:
        print(path.lower())
        api.upload_file(
            path_or_fileobj="esm2/modeling_fastesm.py",
            path_in_repo="modeling_fastesm.py",
            repo_id=path,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="LICENSE",
            path_in_repo="LICENSE",
            repo_id=path,
            repo_type="model",
        )
        if 'esm2' in path.lower():
            api.upload_file(
                path_or_fileobj="readmes/fastesm2_readme.md",
                path_in_repo="README.md",
                repo_id=path,
                repo_type="model",
            )

        if 'fastesm' in path.lower():
            api.upload_file(
                path_or_fileobj="readmes/fastesm_650_readme.md",
                path_in_repo="README.md",
                repo_id=path,
                repo_type="model",
            )

    for path in ESM_PLUSPLUS_MODELS:
        print(path)
        api.upload_file(
            path_or_fileobj="esm_plusplus/modeling_esm_plusplus.py",
            path_in_repo="modeling_esm_plusplus.py",
            repo_id=path,
            repo_type="model",
        )
        if path == 'Synthyra/ESMplusplus_small':
            api.upload_file(
                path_or_fileobj="readmes/esm_plusplus_small_readme.md",
                path_in_repo="README.md",
                repo_id=path,
                repo_type="model",
            )
            api.upload_file(
                path_or_fileobj="LICENSE",
                path_in_repo="LICENSE",
                repo_id=path,
                repo_type="model",
            )
        
        if path == 'Synthyra/ESMplusplus_large':
            api.upload_file(
                path_or_fileobj="readmes/esm_plusplus_large_readme.md",
                path_in_repo="README.md",
                repo_id=path,
                repo_type="model",
            )
            api.upload_file(
                path_or_fileobj="LICENSE",
                path_in_repo="LICENSE",
                repo_id=path,
                repo_type="model",
            )

    for path in E1_MODELS:
        print(path)
        api.upload_file(
            path_or_fileobj="LICENSE",
            path_in_repo="LICENSE",
            repo_id=path,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="readmes/e1_readme.md",
            path_in_repo="README.md",
            repo_id=path,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="e1/tokenizer.json",
            path_in_repo="tokenizer.json",
            repo_id=path,
            repo_type="model",
        )

    for path in ANKH_MODELS:
        print(path)
        api.upload_file(
            path_or_fileobj="LICENSE",
            path_in_repo="LICENSE",
            repo_id=path,
            repo_type="model",
        )

    for path in BOLTZ_MODELS:
        print(path)
        api.upload_file(
            path_or_fileobj="LICENSE",
            path_in_repo="LICENSE",
            repo_id=path,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="readmes/boltz2_readme.md",
            path_in_repo="README.md",
            repo_id=path,
            repo_type="model",
        )
