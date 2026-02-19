import torch
import copy
from huggingface_hub import HfApi, login
from transformers import EsmConfig, EsmForMaskedLM, AutoModelForMaskedLM

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


def _resolve_model_items(model_names: list[str] | None) -> list[tuple[str, str]]:
    if model_names is None:
        return list(MODEL_DICT.items())

    selected_items: list[tuple[str, str]] = []
    for model_name in model_names:
        assert model_name in MODEL_DICT, (
            f"Unknown model name {model_name}. "
            f"Valid options: {sorted(MODEL_DICT.keys())}"
        )
        selected_items.append((model_name, MODEL_DICT[model_name]))
    return selected_items


if __name__ == "__main__":
    # py -m esm2.get_esm2_weights
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--model_names", nargs="*", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    api = HfApi()

    if args.hf_token is not None:
        assert len(args.hf_token) > 0, "--hf_token cannot be empty."
        login(token=args.hf_token)

    for model_name, source_repo in _resolve_model_items(args.model_names):
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
        model = FastEsmForMaskedLM.from_pretrained(
            source_repo,
            config=config,
            dtype=torch.float32,
            device_map="cpu",
        )
        model.load_state_dict(official_model.state_dict(), strict=True)
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

        model.push_to_hub(repo_id)

        api.upload_file(
            path_or_fileobj="esm2/modeling_fastesm.py",
            path_in_repo="modeling_fastesm.py",
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

 