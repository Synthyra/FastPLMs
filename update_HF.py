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

ANKH_MODELS = [
    'Synthyra/ANKH_base',
    'Synthyra/ANKH_large',
    'Synthyra/ANKH2_large'
]


if __name__ == "__main__":
    # py -m update_HF
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    api = HfApi()

    for path in FAST_ESM_MODELS:
        print(path.lower())
        api.upload_file(
            path_or_fileobj="modeling_fastesm.py",
            path_in_repo="modeling_fastesm.py",
            repo_id=path,
            repo_type="model",
        )
        # Upload license file for FastESM models
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
            path_or_fileobj="modeling_esm_plusplus.py",
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
            # Upload license file for ESM++ small model
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
            # Upload license file for ESM++ large model
            api.upload_file(
                path_or_fileobj="LICENSE",
                path_in_repo="LICENSE",
                repo_id=path,
                repo_type="model",
            )

    # Add code to upload files for ANKH models
    for path in ANKH_MODELS:
        print(path)
        # Upload license file for ANKH models
        api.upload_file(
            path_or_fileobj="LICENSE",
            path_in_repo="LICENSE",
            repo_id=path,
            repo_type="model",
        )
        