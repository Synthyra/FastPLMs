import copy
import torch
from typing import List, Optional, Tuple

from huggingface_hub import HfApi, login
from transformers import AutoModelForMaskedLM, EsmTokenizer

from dplm2_fastplms.modeling_dplm2 import DPLM2Config as FastDPLM2Config, DPLM2ForMaskedLM
from weight_parity_utils import assert_model_parameters_fp32


MODEL_DICT = {
    "Synthyra/DPLM2-150M": "airkingbd/dplm2_150m",
    "Synthyra/DPLM2-650M": "airkingbd/dplm2_650m",
    "Synthyra/DPLM2-3B": "airkingbd/dplm2_3b",
}
SHARDED_REPO_IDS = {"Synthyra/DPLM2-3B"}
SHARD_SIZE = "5GB"


def _delete_legacy_unsharded_weights_if_present(api: HfApi, repo_id: str) -> None:
    if repo_id not in SHARDED_REPO_IDS:
        return
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    if "model.safetensors" in repo_files:
        print(f"Deleting legacy unified model.safetensors from {repo_id}")
        api.delete_file(
            path_in_repo="model.safetensors",
            repo_id=repo_id,
            repo_type="model",
        )


def _assert_repo_has_sharded_weights(api: HfApi, repo_id: str) -> None:
    if repo_id not in SHARDED_REPO_IDS:
        return
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    has_index_file = "model.safetensors.index.json" in repo_files
    has_shard_file = any(
        repo_file.startswith("model-") and repo_file.endswith(".safetensors")
        for repo_file in repo_files
    )
    assert has_index_file, f"{repo_id} is missing model.safetensors.index.json."
    assert has_shard_file, f"{repo_id} has no model shard files."
    assert "model.safetensors" not in repo_files, f"{repo_id} still has unified model.safetensors."


def _push_model_with_expected_format(model: DPLM2ForMaskedLM, api: HfApi, repo_id: str) -> None:
    if repo_id in SHARDED_REPO_IDS:
        print(f"Pushing sharded weights for {repo_id} with max_shard_size={SHARD_SIZE}")
        model.push_to_hub(repo_id, max_shard_size=SHARD_SIZE)
        _delete_legacy_unsharded_weights_if_present(api, repo_id)
        _assert_repo_has_sharded_weights(api, repo_id)
        return
    model.push_to_hub(repo_id)


def _resolve_repo_items(repo_ids: Optional[List[str]]) -> List[Tuple[str, str]]:
    if repo_ids is None or len(repo_ids) == 0:
        return list(MODEL_DICT.items())

    selected_items: List[Tuple[str, str]] = []
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
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()
    api = HfApi()

    if args.hf_token is not None:
        assert len(args.hf_token) > 0, "--hf_token cannot be empty."
        login(token=args.hf_token)

    for repo_id, source_repo in _resolve_repo_items(args.repo_ids):
        config = FastDPLM2Config.from_pretrained(source_repo)
        config.auto_map = {
            "AutoConfig": "modeling_dplm2.DPLM2Config",
            "AutoModel": "modeling_dplm2.DPLM2Model",
            "AutoModelForMaskedLM": "modeling_dplm2.DPLM2ForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_dplm2.DPLM2ForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_dplm2.DPLM2ForTokenClassification",
        }
        config.tie_word_embeddings = False
        if args.skip_weights:
            tokenizer = EsmTokenizer.from_pretrained(source_repo)
            config.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"[skip-weights] uploaded config+tokenizer for {repo_id}")
            continue
        model = DPLM2ForMaskedLM.from_pretrained(source_repo, config=config).eval().cpu().to(torch.float32)
        model.tokenizer = EsmTokenizer.from_pretrained(source_repo)

        # Break any potential embedding/LM-head parameter aliasing before export.
        model.lm_head.dense.weight = copy.deepcopy(model.lm_head.dense.weight)
        model.lm_head.dense.bias = copy.deepcopy(model.lm_head.dense.bias)
        model.lm_head.decoder.weight = copy.deepcopy(model.lm_head.decoder.weight)
        model.lm_head.decoder.bias = copy.deepcopy(model.lm_head.decoder.bias)
        model.lm_head.layer_norm.weight = copy.deepcopy(model.lm_head.layer_norm.weight)
        model.lm_head.layer_norm.bias = copy.deepcopy(model.lm_head.layer_norm.bias)

        assert_model_parameters_fp32(
            model=model,
            model_name=f"DPLM2 model ({source_repo})",
        )

        tokenizer = model.tokenizer
        tokenizer.push_to_hub(repo_id)
        _push_model_with_expected_format(model, api, repo_id)
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
