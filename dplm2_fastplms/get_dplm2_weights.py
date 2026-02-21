import copy
import os
import sys
import torch
from huggingface_hub import HfApi, login
from transformers import AutoModelForMaskedLM

from dplm2_fastplms.modeling_dplm2 import DPLM2Config as FastDPLM2Config, DPLM2ForMaskedLM
from weight_parity_utils import assert_state_dict_equal, assert_model_parameters_fp32

from byprot.models.dplm2.dplm2 import MultimodalDiffusionProteinLanguageModel


MODEL_DICT = {
    "Synthyra/DPLM2-150M": "airkingbd/dplm2_150m",
    "Synthyra/DPLM2-650M": "airkingbd/dplm2_650m",
    "Synthyra/DPLM2-3B": "airkingbd/dplm2_3b",
}

def _resolve_repo_items(repo_ids: list[str] | None) -> list[tuple[str, str]]:
    if repo_ids is None or len(repo_ids) == 0:
        return list(MODEL_DICT.items())

    selected_items: list[tuple[str, str]] = []
    for repo_id in repo_ids:
        assert repo_id in MODEL_DICT, (
            f"Unknown repo_id {repo_id}. "
            f"Valid options: {sorted(MODEL_DICT.keys())}"
        )
        selected_items.append((repo_id, MODEL_DICT[repo_id]))
    return selected_items

if __name__ == "__main__":
    # py -m dplm2_fastplms.get_dplm2_weights
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

    for repo_id, source_repo in _resolve_repo_items(args.repo_ids):
        official_model = MultimodalDiffusionProteinLanguageModel.from_pretrained(
            source_repo,
        ).eval().cpu().to(torch.float32)
        official_state_dict = official_model.state_dict()
        
        config = FastDPLM2Config.from_pretrained(source_repo)
        config.auto_map = {
            "AutoConfig": "modeling_dplm2.DPLM2Config",
            "AutoModel": "modeling_dplm2.DPLM2Model",
            "AutoModelForMaskedLM": "modeling_dplm2.DPLM2ForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_dplm2.DPLM2ForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_dplm2.DPLM2ForTokenClassification",
        }
        config.tie_word_embeddings = False
        model = DPLM2ForMaskedLM(config=config).eval().cpu().to(torch.float32)

        model.tokenizer = official_model.tokenizer
        load_result = model.load_state_dict(official_state_dict, strict=True)
        
        # Manually load LM head to prevent weight tying issues
        model.lm_head.dense.weight = copy.deepcopy(official_model.net.lm_head.dense.weight)
        model.lm_head.dense.bias = copy.deepcopy(official_model.net.lm_head.dense.bias)
        model.lm_head.decoder.weight = copy.deepcopy(official_model.net.lm_head.decoder.weight)
        model.lm_head.decoder.bias = copy.deepcopy(official_model.net.lm_head.decoder.bias)
        model.lm_head.layer_norm.weight = copy.deepcopy(official_model.net.lm_head.layer_norm.weight)
        model.lm_head.layer_norm.bias = copy.deepcopy(official_model.net.lm_head.layer_norm.bias)
        
        assert_model_parameters_fp32(
            model=model,
            model_name=f"mapped DPLM2 model ({source_repo})",
        )
        assert_model_parameters_fp32(
            model=official_model,
            model_name=f"official DPLM2 model ({source_repo})",
        )
        assert_state_dict_equal(
            reference_state_dict=official_state_dict,
            candidate_state_dict=model.state_dict(),
            context=f"DPLM2 weight parity ({source_repo})",
        )

        if args.dry_run:
            print(f"[dry_run] validated DPLM2 parity for {repo_id} <- {source_repo}")
            continue

        tokenizer = model.tokenizer
        tokenizer.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj="dplm2_fastplms/modeling_dplm2.py",
            path_in_repo="modeling_dplm2.py",
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
            model_name=f"downloaded DPLM2 model ({repo_id})",
        )
        assert_state_dict_equal(
            reference_state_dict=official_state_dict,
            candidate_state_dict=downloaded_model.state_dict(),
            context=f"DPLM2 weight parity post-download ({repo_id})",
        )
