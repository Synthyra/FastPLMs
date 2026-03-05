import torch
import copy
import os
from huggingface_hub import HfApi, login
from transformers import EsmConfig, EsmForMaskedLM, AutoModelForMaskedLM, AutoTokenizer

from esm2.modeling_fastesm import FastEsmConfig, FastEsmForMaskedLM
from weight_parity_utils import assert_state_dict_equal, assert_model_parameters_fp32


MODEL_DICT = {
    # Synthyra/ESM2-8M
    "ESM2-8M": "facebook/esm2_t6_8M_UR50D",
    # Synthyra/ESM2-35M
    "ESM2-35M": "facebook/esm2_t12_35M_UR50D",
    # Synthyra/ESM2-150M
    "ESM2-150M": "facebook/esm2_t30_150M_UR50D",
    # Synthyra/ESM2-650M
    "ESM2-650M": "facebook/esm2_t33_650M_UR50D",
    # Synthyra/ESM2-3B
    "ESM2-3B": "facebook/esm2_t36_3B_UR50D",
}
SHARDED_REPO_IDS = {"Synthyra/ESM2-3B"}
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


def _push_model_with_expected_format(model: FastEsmForMaskedLM, api: HfApi, repo_id: str) -> None:
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
        # Check if repo_id is a key in MODEL_DICT
        if repo_id in MODEL_DICT:
            selected_items.append((repo_id, MODEL_DICT[repo_id]))
        else:
            assert repo_id in MODEL_DICT, (
                f"Unknown model name {repo_id}. "
                f"Valid options: {sorted(MODEL_DICT.keys())}"
            )
    return selected_items


if __name__ == "__main__":
    # py -m esm2.get_esm2_weights
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--repo_ids", nargs="*", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()
    api = HfApi()

    if args.hf_token is not None:
        assert len(args.hf_token) > 0, "--hf_token cannot be empty."
        login(token=args.hf_token)

    script_root = os.path.dirname(os.path.abspath(__file__))

    for model_name, source_repo in _resolve_repo_items(args.repo_ids):
        official_config = EsmConfig.from_pretrained(source_repo)
        # Makes sure the esm2 word and lm head are correctly loaded
        official_config.tie_word_embeddings = True

        official_model = EsmForMaskedLM.from_pretrained(
            source_repo,
            config=official_config,
            dtype=torch.float32,
            device_map="cpu",
            force_download=True
        )

        config = FastEsmConfig.from_pretrained(source_repo)
        config.auto_map = {
            "AutoConfig": "modeling_fastesm.FastEsmConfig",
            "AutoModel": "modeling_fastesm.FastEsmModel",
            "AutoModelForMaskedLM": "modeling_fastesm.FastEsmForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_fastesm.FastEsmForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_fastesm.FastEsmForTokenClassification",
        }
        config.tie_word_embeddings = False
        if args.skip_weights:
            if args.dry_run:
                print(f"[skip-weights][dry-run] validated config for {repo_id} <- {source_repo}")
                continue
            tokenizer = AutoTokenizer.from_pretrained(source_repo)
            config.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"[skip-weights] uploaded config+tokenizer for {repo_id}")
            continue
        model = FastEsmForMaskedLM.from_pretrained(
            source_repo,
            config=config,
            dtype=torch.float32,
            device_map="cpu",
        )
        model.load_state_dict(official_model.state_dict(), strict=True)
        
        # Manually load LM head to prevent weight tying issues
        model.lm_head.dense.weight = copy.deepcopy(official_model.lm_head.dense.weight)
        model.lm_head.dense.bias = copy.deepcopy(official_model.lm_head.dense.bias)
        model.lm_head.decoder.weight = copy.deepcopy(official_model.lm_head.decoder.weight)
        model.lm_head.decoder.bias = copy.deepcopy(official_model.lm_head.decoder.bias)
        model.lm_head.layer_norm.weight = copy.deepcopy(official_model.lm_head.layer_norm.weight)
        model.lm_head.layer_norm.bias = copy.deepcopy(official_model.lm_head.layer_norm.bias)

        assert_model_parameters_fp32(
            model=official_model,
            model_name=f"official ESM2 model ({source_repo})",
        )
        assert_model_parameters_fp32(
            model=model,
            model_name=f"mapped ESM2 model ({source_repo})",
        )
        assert_state_dict_equal(
            reference_state_dict=official_model.state_dict(),
            candidate_state_dict=model.state_dict(),
            context=f"ESM2 weight parity ({source_repo})",
        )

        repo_id = "Synthyra/" + model_name
        if args.dry_run:
            print(f"[dry_run] validated ESM2 parity for {repo_id} <- {source_repo}")
            continue

        tokenizer = model.tokenizer
        tokenizer.push_to_hub(repo_id)

        _push_model_with_expected_format(model, api, repo_id)

        api.upload_file(
            path_or_fileobj=os.path.join(script_root, "modeling_fastesm.py"),
            path_in_repo="modeling_fastesm.py",
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
        assert_state_dict_equal(
            reference_state_dict=official_model.state_dict(),
            candidate_state_dict=downloaded_model.state_dict(),
            context=f"ESM2 weight parity post-download ({repo_id})",
        )

 