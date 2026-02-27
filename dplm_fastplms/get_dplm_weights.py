import copy
import torch
from huggingface_hub import HfApi, login
from transformers import AutoModelForMaskedLM, EsmTokenizer

from dplm_fastplms.modeling_dplm import DPLMConfig as FastDPLMConfig, DPLMForMaskedLM
from weight_parity_utils import assert_model_parameters_fp32


MODEL_DICT = {
    "Synthyra/DPLM-150M": "airkingbd/dplm_150m",
    "Synthyra/DPLM-650M": "airkingbd/dplm_650m",
    "Synthyra/DPLM-3B": "airkingbd/dplm_3b",
}
SHARDED_REPO_IDS = {"Synthyra/DPLM-3B"}
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


def _push_model_with_expected_format(model: DPLMForMaskedLM, api: HfApi, repo_id: str) -> None:
    if repo_id in SHARDED_REPO_IDS:
        print(f"Pushing sharded weights for {repo_id} with max_shard_size={SHARD_SIZE}")
        model.push_to_hub(repo_id, max_shard_size=SHARD_SIZE)
        _delete_legacy_unsharded_weights_if_present(api, repo_id)
        _assert_repo_has_sharded_weights(api, repo_id)
        return
    model.push_to_hub(repo_id)


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
    # py -m dplm_fastplms.get_dplm_weights
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--repo_ids", nargs="*", type=str, default=None)
    args = parser.parse_args()
    api = HfApi()

    if args.hf_token is not None:
        assert len(args.hf_token) > 0, "--hf_token cannot be empty."
        login(token=args.hf_token)

    for repo_id, source_repo in _resolve_repo_items(args.repo_ids):
        config = FastDPLMConfig.from_pretrained(source_repo)
        config.auto_map = {
            "AutoConfig": "modeling_dplm.DPLMConfig",
            "AutoModel": "modeling_dplm.DPLMModel",
            "AutoModelForMaskedLM": "modeling_dplm.DPLMForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_dplm.DPLMForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_dplm.DPLMForTokenClassification",
        }
        config.tie_word_embeddings = False
        model = DPLMForMaskedLM.from_pretrained(source_repo, config=config).eval().cpu().to(torch.float32)
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
            model_name=f"DPLM model ({source_repo})",
        )

        tokenizer = model.tokenizer
        tokenizer.push_to_hub(repo_id)
        _push_model_with_expected_format(model, api, repo_id)
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
