import copy
import os
import torch
from huggingface_hub import HfApi, login
from transformers import AutoModelForMaskedLM

from e1_fastplms.load_official import load_official_model
from e1_fastplms.modeling_e1 import E1Config, E1ForMaskedLM
from weight_parity_utils import assert_state_dict_equal, assert_model_parameters_fp32


MODEL_DICT = {
    "Synthyra/Profluent-E1-150M": "Profluent-Bio/E1-150m",
    "Synthyra/Profluent-E1-300M": "Profluent-Bio/E1-300m",
    "Synthyra/Profluent-E1-600M": "Profluent-Bio/E1-600m",
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
    # py -m e1_fastplms.get_e1_weights
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

    script_root = os.path.dirname(os.path.abspath(__file__))

    for repo_id, source_repo in _resolve_repo_items(args.repo_ids):
        official_model, batch_preparer = load_official_model(source_repo, device=torch.device("cpu"), dtype=torch.float32)
        config = E1Config.from_pretrained(source_repo)
        config.auto_map = {
            "AutoConfig": "modeling_e1.E1Config",
            "AutoModel": "modeling_e1.E1Model",
            "AutoModelForMaskedLM": "modeling_e1.E1ForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_e1.E1ForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_e1.E1ForTokenClassification",
        }
        config.tie_word_embeddings = False
        model = E1ForMaskedLM(config=config).eval().cpu().to(torch.float32)
        load_result = model.load_state_dict(official_model.model.state_dict(), strict=True)

        # Manually load MLM head to prevent weight tying issues
        model.mlm_head[0].weight = copy.deepcopy(official_model.model.mlm_head[0].weight)
        model.mlm_head[0].bias = copy.deepcopy(official_model.model.mlm_head[0].bias)
        model.mlm_head[2].weight = copy.deepcopy(official_model.model.mlm_head[2].weight)
        model.mlm_head[2].bias = copy.deepcopy(official_model.model.mlm_head[2].bias)
        model.mlm_head[3].weight = copy.deepcopy(official_model.model.mlm_head[3].weight)
        model.mlm_head[3].bias = copy.deepcopy(official_model.model.mlm_head[3].bias)
        
        assert_model_parameters_fp32(
            model=official_model.model,
            model_name=f"official E1 model ({source_repo})",
        )
        assert_model_parameters_fp32(
            model=model,
            model_name=f"mapped E1 model ({source_repo})",
        )
        assert_state_dict_equal(
            reference_state_dict=official_model.model.state_dict(),
            candidate_state_dict=model.state_dict(),
            context=f"E1 weight parity ({source_repo})",
        )

        if args.dry_run:
            print(f"[dry_run] validated E1 parity for {repo_id} <- {source_repo}")
            continue

        model.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj=os.path.join(script_root, "modeling_e1.py"),
            path_in_repo="modeling_e1.py",
            repo_id=repo_id,
            repo_type="model",
        )
        
        # Push tokenizer
        batch_preparer.tokenizer.save(os.path.join(script_root, "tokenizer.json"))
        api.upload_file(
            path_or_fileobj=os.path.join(script_root, "tokenizer.json"),
            path_in_repo="tokenizer.json",
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
            model_name=f"downloaded E1 model ({repo_id})",
        )
        assert_state_dict_equal(
            reference_state_dict=official_model.model.state_dict(),
            candidate_state_dict=downloaded_model.state_dict(),
            context=f"E1 weight parity post-download ({repo_id})",
        )
