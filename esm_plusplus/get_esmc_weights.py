import copy
import torch
from huggingface_hub import HfApi, login
from transformers import AutoModelForMaskedLM

from esm_plusplus.load_official import load_official_model
from esm_plusplus.modeling_esm_plusplus import ESMplusplusForMaskedLM
from weight_parity_utils import assert_state_dict_equal, assert_model_parameters_fp32


MODEL_DICT = {
    "Synthyra/ESMplusplus_small": "esmc-300",
    "Synthyra/ESMplusplus_large": "esmc-600",
}

if __name__ == "__main__":
    # py -m esm_plusplus.get_esmc_weights
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--model_paths", nargs="*", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    api = HfApi()

    if args.hf_token is not None:
        assert len(args.hf_token) > 0, "--hf_token cannot be empty."
        login(token=args.hf_token)

    for model_path, esmc_model_key in MODEL_DICT.items():
        official_model, tokenizer = load_official_model(esmc_model_key, device=torch.device("cpu"), dtype=torch.float32)
        config = copy.deepcopy(official_model.config)
        config.auto_map = {
            "AutoConfig": "modeling_esm_plusplus.ESMplusplusConfig",
            "AutoModel": "modeling_esm_plusplus.ESMplusplusModel",
            "AutoModelForMaskedLM": "modeling_esm_plusplus.ESMplusplusForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_esm_plusplus.ESMplusplusForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_esm_plusplus.ESMplusplusForTokenClassification",
        }
        config.tie_word_embeddings = False
        model = ESMplusplusForMaskedLM(config=config).eval().cpu().to(torch.float32)
        load_result = model.load_state_dict(official_model.state_dict(), strict=True)

        model.sequence_head[0].weight = copy.deepcopy(official_model.sequence_head[0].weight)
        model.sequence_head[0].bias = copy.deepcopy(official_model.sequence_head[0].bias)
        model.sequence_head[2].weight = copy.deepcopy(official_model.sequence_head[2].weight)
        model.sequence_head[2].bias = copy.deepcopy(official_model.sequence_head[2].bias)
        model.sequence_head[3].weight = copy.deepcopy(official_model.sequence_head[3].weight)
        model.sequence_head[3].bias = copy.deepcopy(official_model.sequence_head[3].bias)
        
        assert_model_parameters_fp32(
            model=model,
            model_name=f"mapped ESM++ model ({esmc_model_key})",
        )
        assert_model_parameters_fp32(
            model=official_model,
            model_name=f"official ESM++ model ({esmc_model_key})",
        )
        assert_state_dict_equal(
            reference_state_dict=official_model.state_dict(),
            candidate_state_dict=model.state_dict(),
            context=f"ESMC/ESM++ weight parity ({esmc_model_key})",
        )

        if args.dry_run:
            print(f"[dry_run] validated ESM++ parity for {model_path} <- {esmc_model_key}")
            continue

        tokenizer = model.tokenizer
        tokenizer.push_to_hub(model_path)
        model.push_to_hub(model_path)
        api.upload_file(
            path_or_fileobj="esm_plusplus/modeling_esm_plusplus.py",
            path_in_repo="modeling_esm_plusplus.py",
            repo_id=model_path,
            repo_type="model",
        )
        downloaded_model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map="cpu",
            force_download=True,
            trust_remote_code=True,
        )
        assert_state_dict_equal(
            reference_state_dict=official_model.state_dict(),
            candidate_state_dict=downloaded_model.state_dict(),
            context=f"ESMC/ESM++ weight parity post-download ({model_path})",
        )
