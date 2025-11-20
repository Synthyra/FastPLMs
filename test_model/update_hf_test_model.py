from huggingface_hub import HfApi, login


TEST_MODELS = [
    'lhallee/test_auto_model'
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


    for path in TEST_MODELS:
        print(path)
        api.upload_file(
            path_or_fileobj="test_model.py",
            path_in_repo="test_model.py",
            repo_id=path,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj='pooler.py',
            path_in_repo='pooler.py',
            repo_id=path,
            repo_type="model",
        )