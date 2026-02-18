import torch
from huggingface_hub import HfApi, login

from dplm_fastplms.dplm2 import DPLM2Config
from dplm_fastplms.dplm2 import DPLM2ForMaskedLM


MODEL_DICT = {
    "Synthyra/DPLM2-150M": "airkingbd/dplm2_150m",
    "Synthyra/DPLM2-650M": "airkingbd/dplm2_650m",
    "Synthyra/DPLM2-3B": "airkingbd/dplm2_3b",
}


if __name__ == "__main__":
    # py -m dplm_fastplms.get_dplm2_weights
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()
    api = HfApi()

    if args.hf_token is not None:
        assert len(args.hf_token) > 0, "--hf_token cannot be empty."
        login(token=args.hf_token)

    for repo_id, source_repo in MODEL_DICT.items():
        config = DPLM2Config.from_pretrained(source_repo)
        config.auto_map = {
            "AutoConfig": "dplm2.DPLM2Config",
            "AutoModel": "dplm2.DPLM2Model",
            "AutoModelForMaskedLM": "dplm2.DPLM2ForMaskedLM",
            "AutoModelForSequenceClassification": "dplm2.DPLM2ForSequenceClassification",
            "AutoModelForTokenClassification": "dplm2.DPLM2ForTokenClassification",
        }
        config.tie_word_embeddings = False
        model = DPLM2ForMaskedLM.from_pretrained(source_repo, config=config, dtype=torch.bfloat16)
        tokenizer = model.tokenizer
        tokenizer.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj="dplm_fastplms/dplm2.py",
            path_in_repo="dplm2.py",
            repo_id=repo_id,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="dplm_fastplms/base_tokenizer.py",
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
