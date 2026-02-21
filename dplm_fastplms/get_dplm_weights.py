import json
from pathlib import Path
import torch
from huggingface_hub import HfApi, hf_hub_download, login
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.modeling_utils import load_state_dict

from dplm_fastplms.modeling_dplm import DPLMConfig as FastDPLMConfig, DPLMForMaskedLM
from weight_parity_utils import assert_state_dict_equal, assert_model_parameters_fp32


MODEL_DICT = {
    "Synthyra/DPLM-150M": "airkingbd/dplm_150m",
    "Synthyra/DPLM-650M": "airkingbd/dplm_650m",
    "Synthyra/DPLM-3B": "airkingbd/dplm_3b",
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


def _load_repo_state_dict(api: HfApi, source_repo: str) -> dict[str, torch.Tensor]:
    repo_files = set(api.list_repo_files(repo_id=source_repo, repo_type="model"))
    shard_filenames: list[str]

    if "model.safetensors.index.json" in repo_files:
        index_path = hf_hub_download(repo_id=source_repo, filename="model.safetensors.index.json", repo_type="model")
        index_data = json.loads(Path(index_path).read_text())
        shard_filenames = sorted(set(index_data["weight_map"].values()))
    elif "pytorch_model.bin.index.json" in repo_files:
        index_path = hf_hub_download(repo_id=source_repo, filename="pytorch_model.bin.index.json", repo_type="model")
        index_data = json.loads(Path(index_path).read_text())
        shard_filenames = sorted(set(index_data["weight_map"].values()))
    elif "model.safetensors" in repo_files:
        shard_filenames = ["model.safetensors"]
    elif "pytorch_model.bin" in repo_files:
        shard_filenames = ["pytorch_model.bin"]
    else:
        raise AssertionError(
            f"Could not find model weights in {source_repo}. "
            "Expected model.safetensors(.index.json) or pytorch_model.bin(.index.json)."
        )

    source_state_dict: dict[str, torch.Tensor] = {}
    for filename in shard_filenames:
        shard_path = hf_hub_download(repo_id=source_repo, filename=filename, repo_type="model")
        shard_state_dict = load_state_dict(shard_path)
        source_state_dict.update(shard_state_dict)
    return source_state_dict


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if prefix == "":
        return dict(state_dict)
    stripped_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped_state_dict[key[len(prefix):]] = value
        else:
            stripped_state_dict[key] = value
    return stripped_state_dict


def _align_state_dict_to_model(
    source_state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
    source_repo: str,
) -> dict[str, torch.Tensor]:
    target_keys = set(model_state_dict.keys())
    prefixes = ["", "net.", "model.", "module.", "net.model."]
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_overlap = -1

    for prefix in prefixes:
        candidate_state_dict = _strip_prefix(source_state_dict, prefix)
        overlap = sum(1 for key in target_keys if key in candidate_state_dict)
        if overlap > best_overlap:
            best_overlap = overlap
            best_state_dict = candidate_state_dict

    assert best_state_dict is not None, f"Unable to align source state_dict for {source_repo}."
    assert best_overlap == len(target_keys), (
        f"Failed to align all model keys for {source_repo}. "
        f"Matched {best_overlap}/{len(target_keys)} keys."
    )

    aligned_state_dict: dict[str, torch.Tensor] = {}
    for key in model_state_dict.keys():
        aligned_state_dict[key] = best_state_dict[key]
    return aligned_state_dict


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
        model = DPLMForMaskedLM(config=config).eval().cpu().to(torch.float32)
        source_state_dict = _load_repo_state_dict(api=api, source_repo=source_repo)
        aligned_source_state_dict = _align_state_dict_to_model(
            source_state_dict=source_state_dict,
            model_state_dict=model.state_dict(),
            source_repo=source_repo,
        )

        model.tokenizer = AutoTokenizer.from_pretrained(source_repo)
        model.load_state_dict(aligned_source_state_dict, strict=True)
        assert_model_parameters_fp32(
            model=model,
            model_name=f"mapped DPLM model ({source_repo})",
        )
        assert_state_dict_equal(
            reference_state_dict=aligned_source_state_dict,
            candidate_state_dict=model.state_dict(),
            context=f"DPLM weight parity ({source_repo})",
        )

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
            reference_state_dict=aligned_source_state_dict,
            candidate_state_dict=downloaded_model.state_dict(),
            context=f"DPLM weight parity post-download ({repo_id})",
        )
