import torch
from huggingface_hub import HfApi, login

from dplm.dplm import DPLMConfig
from dplm.dplm import DPLMForMaskedLM


MODEL_DICT = {
    "Synthyra/DPLM-150M": "airkingbd/dplm_150m",
    "Synthyra/DPLM-650M": "airkingbd/dplm_650m",
    "Synthyra/DPLM-3B": "airkingbd/dplm_3b",
}


if __name__ == "__main__":
    # py -m dplm.get_dplm_weights
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()
    api = HfApi()

    if args.hf_token is not None:
        assert len(args.hf_token) > 0, "--hf_token cannot be empty."
        login(token=args.hf_token)

    for repo_id, source_repo in MODEL_DICT.items():
        config = DPLMConfig.from_pretrained(source_repo)
        config.auto_map = {
            "AutoConfig": "dplm.DPLMConfig",
            "AutoModel": "dplm.DPLMModel",
            "AutoModelForMaskedLM": "dplm.DPLMForMaskedLM",
            "AutoModelForSequenceClassification": "dplm.DPLMForSequenceClassification",
            "AutoModelForTokenClassification": "dplm.DPLMForTokenClassification",
        }
        config.tie_word_embeddings = False
        model = DPLMForMaskedLM.from_pretrained(source_repo, config=config, dtype=torch.bfloat16)
        tokenizer = model.tokenizer
        tokenizer.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj="dplm/dplm.py",
            path_in_repo="dplm.py",
            repo_id=repo_id,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="dplm/base_tokenizer.py",
            path_in_repo="base_tokenizer.py",
            repo_id=repo_id,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="embedding_mixin.py",
            path_in_repo="embedding_mixin.py",
            repo_id=repo_id,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="entrypoint_setup.py",
            path_in_repo="entrypoint_setup.py",
            repo_id=repo_id,
            repo_type="model",
        )
