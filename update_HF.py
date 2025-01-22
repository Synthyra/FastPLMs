from huggingface_hub import HfApi


fast_esm_models = [
    'Synthyra/ESM2-8M',
    'Synthyra/ESM2-35M',
    'Synthyra/ESM2-150M',
    'Synthyra/ESM2-650M',
    'Synthyra/ESM2-3B',
    'Synthyra/FastESM2_650'
]


esm_plusplus_models = [
    'Synthyra/ESMplusplus_small',
    'Synthyra/ESMplusplus_large',
]


api = HfApi()

for path in fast_esm_models:
    print(path.lower())
    api.upload_file(
        path_or_fileobj="modeling_fastesm.py",
        path_in_repo="modeling_fastesm.py",
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


for path in esm_plusplus_models:
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
    
    if path == 'Synthyra/ESMplusplus_large':
        api.upload_file(
            path_or_fileobj="readmes/esm_plusplus_large_readme.md",
            path_in_repo="README.md",
            repo_id=path,
            repo_type="model",
        )