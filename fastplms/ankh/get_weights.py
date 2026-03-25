import copy
import os

import torch
from typing import Dict, List, Optional, Tuple

from huggingface_hub import HfApi, login
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer, AutoModel

from fastplms.ankh.modeling_ankh import FastAnkhConfig, FastAnkhForMaskedLM
from fastplms.weight_parity_utils import assert_model_parameters_fp32


MODEL_DICT = {
    "Synthyra/ANKH_base": "ElnaggarLab/ankh-base",
    "Synthyra/ANKH_large": "ElnaggarLab/ankh-large",
    "Synthyra/ANKH2_large": "ElnaggarLab/ankh2-large",
    "Synthyra/ANKH3_large": "ElnaggarLab/ankh3-large",
    "Synthyra/ANKH3_xl": "ElnaggarLab/ankh3-xl",
}
SHARDED_REPO_IDS = {"Synthyra/ANKH3_xl"}
SHARD_SIZE = "5GB"


def _map_encoder_state_dict(official_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map T5ForConditionalGeneration state dict to FastAnkh format (encoder only)."""
    new_sd = {}
    for key, value in official_sd.items():
        if not key.startswith("encoder."):
            continue

        new_key = key
        # encoder.embed_tokens.weight -> ankh.embed_tokens.weight
        new_key = new_key.replace("encoder.embed_tokens.", "ankh.embed_tokens.")
        # encoder.block.{i}.layer.0.SelfAttention.* -> ankh.encoder.layer.{i}.attention.*
        new_key = new_key.replace("encoder.block.", "ankh.encoder.layer.")
        new_key = new_key.replace(".layer.0.SelfAttention.", ".attention.")
        new_key = new_key.replace(".layer.0.layer_norm.", ".attention_norm.")
        # encoder.block.{i}.layer.1.DenseReluDense.* -> ankh.encoder.layer.{i}.ffn.*
        new_key = new_key.replace(".layer.1.DenseReluDense.", ".ffn.")
        new_key = new_key.replace(".layer.1.layer_norm.", ".ffn_norm.")
        # encoder.final_layer_norm.* -> ankh.encoder.final_layer_norm.*
        new_key = new_key.replace("encoder.final_layer_norm.", "ankh.encoder.final_layer_norm.")

        new_sd[new_key] = value.clone()

    # LM head: copy from shared embedding (not tied)
    assert "ankh.embed_tokens.weight" in new_sd, "Missing embed_tokens in mapped state dict"
    new_sd["lm_head.weight"] = new_sd["ankh.embed_tokens.weight"].clone()

    return new_sd


def _build_config(source_repo: str) -> FastAnkhConfig:
    """Build FastAnkhConfig from official T5 config."""
    t5_config = T5Config.from_pretrained(source_repo)

    # Determine activation function
    act_info = t5_config.feed_forward_proj.split("-")
    if t5_config.feed_forward_proj == "gated-gelu":
        dense_act_fn = "gelu_new"
    elif len(act_info) > 1:
        dense_act_fn = act_info[-1]
    else:
        dense_act_fn = act_info[0]

    config = FastAnkhConfig(
        vocab_size=t5_config.vocab_size,
        d_model=t5_config.d_model,
        d_kv=t5_config.d_kv,
        d_ff=t5_config.d_ff,
        num_heads=t5_config.num_heads,
        num_layers=t5_config.num_layers,
        relative_attention_num_buckets=t5_config.relative_attention_num_buckets,
        relative_attention_max_distance=t5_config.relative_attention_max_distance,
        dense_act_fn=dense_act_fn,
        layer_norm_epsilon=t5_config.layer_norm_epsilon,
        initializer_factor=t5_config.initializer_factor,
        pad_token_id=t5_config.pad_token_id or 0,
        eos_token_id=t5_config.eos_token_id or 1,
    )
    config.auto_map = {
        "AutoConfig": "modeling_ankh.FastAnkhConfig",
        "AutoModel": "modeling_ankh.FastAnkhModel",
        "AutoModelForMaskedLM": "modeling_ankh.FastAnkhForMaskedLM",
        "AutoModelForSequenceClassification": "modeling_ankh.FastAnkhForSequenceClassification",
        "AutoModelForTokenClassification": "modeling_ankh.FastAnkhForTokenClassification",
    }
    config.tie_word_embeddings = False
    return config


def _delete_legacy_unsharded_weights_if_present(api: HfApi, repo_id: str) -> None:
    if repo_id not in SHARDED_REPO_IDS:
        return
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    if "model.safetensors" in repo_files:
        print(f"Deleting legacy unified model.safetensors from {repo_id}")
        api.delete_file(path_in_repo="model.safetensors", repo_id=repo_id, repo_type="model")


def _assert_repo_has_sharded_weights(api: HfApi, repo_id: str) -> None:
    if repo_id not in SHARDED_REPO_IDS:
        return
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    assert "model.safetensors.index.json" in repo_files, f"{repo_id} missing index file."
    has_shards = any(f.startswith("model-") and f.endswith(".safetensors") for f in repo_files)
    assert has_shards, f"{repo_id} has no shard files."
    assert "model.safetensors" not in repo_files, f"{repo_id} still has unified weights."


def _push_model_with_expected_format(model: FastAnkhForMaskedLM, api: HfApi, repo_id: str) -> None:
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
    selected = []
    for repo_id in repo_ids:
        assert repo_id in MODEL_DICT, (
            f"Unknown repo {repo_id}. Valid: {sorted(MODEL_DICT.keys())}"
        )
        selected.append((repo_id, MODEL_DICT[repo_id]))
    return selected


if __name__ == "__main__":
    # py -m fastplms.ankh.get_weights
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--repo_ids", nargs="*", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()
    api = HfApi()

    if args.hf_token is not None:
        assert len(args.hf_token) > 0
        login(token=args.hf_token)

    script_root = os.path.dirname(os.path.abspath(__file__))

    for repo_id, source_repo in _resolve_repo_items(args.repo_ids):
        print(f"\n{'='*60}")
        print(f"Processing {repo_id} <- {source_repo}")
        print(f"{'='*60}")

        config = _build_config(source_repo)

        if args.skip_weights:
            if args.dry_run:
                print(f"[skip-weights][dry-run] validated config for {repo_id}")
                continue
            tokenizer = AutoTokenizer.from_pretrained(source_repo)
            config.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"[skip-weights] uploaded config+tokenizer for {repo_id}")
            continue

        # Load official T5 encoder-decoder
        print(f"Loading official T5ForConditionalGeneration from {source_repo}...")
        official_model = T5ForConditionalGeneration.from_pretrained(
            source_repo, torch_dtype=torch.float32, device_map="cpu",
        )

        # Map encoder weights to FastAnkh format
        mapped_sd = _map_encoder_state_dict(official_model.state_dict())
        print(f"Mapped {len(mapped_sd)} parameters from encoder")

        # Create FastAnkh model and load mapped weights
        model = FastAnkhForMaskedLM(config)
        model = model.to(dtype=torch.float32)

        # Verify all expected keys are present
        model_keys = set(model.state_dict().keys())
        mapped_keys = set(mapped_sd.keys())
        missing = model_keys - mapped_keys
        unexpected = mapped_keys - model_keys
        assert not missing, f"Missing keys in mapped state dict:\n{missing}"
        assert not unexpected, f"Unexpected keys in mapped state dict:\n{unexpected}"

        model.load_state_dict(mapped_sd, strict=True)

        assert_model_parameters_fp32(model=model, model_name=f"FastAnkh ({repo_id})")

        # Verify encoder weight parity (compare mapped values directly)
        print("Verifying encoder weight parity...")
        for key, mapped_val in mapped_sd.items():
            if key == "lm_head.weight":
                continue  # LM head is a copy, not from official encoder
            model_val = model.state_dict()[key]
            mse = (model_val.float() - mapped_val.float()).pow(2).mean().item()
            assert mse == 0.0, f"Weight mismatch at {key}: MSE={mse}"
        print("Weight parity verified (MSE=0.0 for all encoder parameters)")

        if args.dry_run:
            print(f"[dry_run] validated ANKH parity for {repo_id} <- {source_repo}")
            continue

        # Push tokenizer
        tokenizer = AutoTokenizer.from_pretrained(source_repo)
        tokenizer.push_to_hub(repo_id)

        # Push model
        _push_model_with_expected_format(model, api, repo_id)

        # Upload modeling file
        api.upload_file(
            path_or_fileobj=os.path.join(script_root, "modeling_ankh.py"),
            path_in_repo="modeling_ankh.py",
            repo_id=repo_id,
            repo_type="model",
        )

        # Verify download
        print(f"Verifying download from {repo_id}...")
        downloaded_model = AutoModel.from_pretrained(
            repo_id, torch_dtype=torch.float32, device_map="cpu",
            force_download=True, trust_remote_code=True,
        )
        for key in model.ankh.state_dict():
            orig = model.ankh.state_dict()[key]
            dl = downloaded_model.state_dict()[key]
            mse = (orig.float() - dl.float()).pow(2).mean().item()
            assert mse == 0.0, f"Post-download mismatch at {key}: MSE={mse}"
        print(f"Download verification passed for {repo_id}")
