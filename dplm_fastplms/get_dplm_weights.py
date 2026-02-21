import copy
import torch
from huggingface_hub import HfApi, login
from transformers import AutoModelForMaskedLM

from dplm_fastplms.modeling_dplm import DPLMForMaskedLM
from weight_parity_utils import assert_state_dict_equal, assert_model_parameters_fp32

from byprot.models.dplm.dplm import DiffusionProteinLanguageModel, DPLMConfig


MODEL_DICT = {
    "Synthyra/DPLM-150M": "airkingbd/dplm_150m",
    "Synthyra/DPLM-650M": "airkingbd/dplm_650m",
    "Synthyra/DPLM-3B": "airkingbd/dplm_3b",
}

if __name__ == "__main__":
    # py -m dplm_fastplms.get_dplm_weights
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--repo_ids", nargs="*", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    api = HfApi()

    if args.hf_token is not None:
        assert len(args.hf_token) > 0, "--hf_token cannot be empty."
        login(token=args.hf_token)

    for repo_id, source_repo in MODEL_DICT.items():
        official_model = DiffusionProteinLanguageModel.from_pretrained(
            source_repo,
            device_map="cpu",
            dtype=torch.float32,
        ).eval().cpu().to(torch.float32)
        official_state_dict = official_model.net.state_dict()

        config = DPLMConfig.from_pretrained(source_repo)
        config.auto_map = {
            "AutoConfig": "modeling_dplm.DPLMConfig",
            "AutoModel": "modeling_dplm.DPLMModel",
            "AutoModelForMaskedLM": "modeling_dplm.DPLMForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_dplm.DPLMForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_dplm.DPLMForTokenClassification",
        }
        config.tie_word_embeddings = False
        model = DPLMForMaskedLM(config=config).eval().cpu().to(torch.float32)

        model.tokenizer = official_model.net.tokenizer
        load_result = model.load_state_dict(official_state_dict, strict=True)
        model.lm_head.dense.weight = copy.deepcopy(official_model.net.lm_head.dense.weight)
        model.lm_head.dense.bias = copy.deepcopy(official_model.net.lm_head.dense.bias)
        model.lm_head.decoder.weight = copy.deepcopy(official_model.net.lm_head.decoder.weight)
        model.lm_head.decoder.bias = copy.deepcopy(official_model.net.lm_head.decoder.bias)
        model.lm_head.layer_norm.weight = copy.deepcopy(official_model.net.lm_head.layer_norm.weight)
        model.lm_head.layer_norm.bias = copy.deepcopy(official_model.net.lm_head.layer_norm.bias)
        assert_model_parameters_fp32(
            model=model,
            model_name=f"mapped DPLM model ({source_repo})",
        )
        assert_model_parameters_fp32(
            model=official_model,
            model_name=f"official DPLM model ({source_repo})",
        )
        assert_state_dict_equal(
            reference_state_dict=official_state_dict,
            candidate_state_dict=model.state_dict(),
            context=f"DPLM weight parity ({source_repo})",
        )

        if args.dry_run:
            print(f"[dry_run] validated DPLM parity for {repo_id} <- {source_repo}")
            continue

        tokenizer = model.tokenizer
        tokenizer.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj="dplm_fastplms/modeling_dplm.py",
            path_in_repo="modeling_dplm.py",
            repo_id=repo_id,
            repo_type="model",
        )
        downloaded_model = AutoModelForMaskedLM.from_pretrained(
            repo_id,
            dtype=torch.float32,
            device_map="cpu",
            force_download=True,
            trust_remote_code=True,
        )
        assert_model_parameters_fp32(
            model=downloaded_model,
            model_name=f"downloaded DPLM model ({repo_id})",
        )
        assert_state_dict_equal(
            reference_state_dict=official_state_dict,
            candidate_state_dict=downloaded_model.state_dict(),
            context=f"DPLM weight parity post-download ({repo_id})",
        )
