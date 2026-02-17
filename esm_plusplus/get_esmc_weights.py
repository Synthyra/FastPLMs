import torch
from huggingface_hub import HfApi
from huggingface_hub import login

from esm_plusplus.modeling_esm_plusplus import ESMplusplus_300M
from esm_plusplus.modeling_esm_plusplus import ESMplusplus_600M
from esm_plusplus.modeling_esm_plusplus import get_esmc_checkpoint_path


def _load_reference_state_dict(esmc_model_key: str) -> dict[str, torch.Tensor]:
    checkpoint_path = get_esmc_checkpoint_path(esmc_model_key)
    return torch.load(checkpoint_path, map_location="cpu")


def _assert_fp32_state_dict_equal(reference_state_dict: dict[str, torch.Tensor], candidate_model: torch.nn.Module) -> None:
    candidate_state_dict = candidate_model.state_dict()
    reference_keys = set(reference_state_dict.keys())
    candidate_keys = set(candidate_state_dict.keys())
    only_in_reference = sorted(reference_keys - candidate_keys)
    only_in_candidate = sorted(candidate_keys - reference_keys)
    shape_mismatches = []
    differing_tensors = []
    max_abs_diff = 0.0
    max_abs_diff_param = ""

    for name in sorted(reference_keys & candidate_keys):
        reference_tensor = reference_state_dict[name].detach().cpu().to(torch.float32)
        candidate_tensor = candidate_state_dict[name].detach().cpu().to(torch.float32)
        if reference_tensor.shape != candidate_tensor.shape:
            shape_mismatches.append(
                {
                    "name": name,
                    "reference_shape": list(reference_tensor.shape),
                    "candidate_shape": list(candidate_tensor.shape),
                }
            )
            continue
        if torch.equal(reference_tensor, candidate_tensor):
            continue
        abs_diff = torch.abs(reference_tensor - candidate_tensor)
        param_max_abs_diff = float(torch.max(abs_diff).item())
        param_mean_abs_diff = float(torch.mean(abs_diff).item())
        differing_tensors.append(
            {
                "name": name,
                "max_abs_diff": param_max_abs_diff,
                "mean_abs_diff": param_mean_abs_diff,
            }
        )
        if param_max_abs_diff > max_abs_diff:
            max_abs_diff = param_max_abs_diff
            max_abs_diff_param = name

    assert len(only_in_reference) == 0 and len(only_in_candidate) == 0 and len(shape_mismatches) == 0 and len(differing_tensors) == 0, (
        "Strict ESMplusplus reference requires float32-identical checkpoints. "
        f"diff_param_count={len(differing_tensors)} "
        f"max_abs_diff={max_abs_diff} "
        f"max_abs_diff_param={max_abs_diff_param} "
        f"only_in_reference={only_in_reference[:5]} "
        f"only_in_candidate={only_in_candidate[:5]} "
        f"shape_mismatches={shape_mismatches[:5]} "
        f"diff_params_sample={differing_tensors[:5]}"
    )


if __name__ == "__main__":
    # py -m esm_plusplus.get_esmc_weights
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()
    api = HfApi()

    if args.hf_token is not None:
        assert len(args.hf_token) > 0, "--hf_token cannot be empty."
        login(token=args.hf_token)

    model_dict = {
        "Synthyra/ESMplusplus_small": (ESMplusplus_300M, "esmc-300"),
        "Synthyra/ESMplusplus_large": (ESMplusplus_600M, "esmc-600"),
    }

    for model_path, model_spec in model_dict.items():
        model_fn, esmc_model_key = model_spec
        reference_state_dict = _load_reference_state_dict(esmc_model_key)
        model = model_fn(device="cpu")
        _assert_fp32_state_dict_equal(reference_state_dict, model)
        model = model.eval().cpu().to(torch.float32)
        _assert_fp32_state_dict_equal(reference_state_dict, model)
        model.config.auto_map = {
            "AutoConfig": "modeling_esm_plusplus.ESMplusplusConfig",
            "AutoModel": "modeling_esm_plusplus.ESMplusplusModel",
            "AutoModelForMaskedLM": "modeling_esm_plusplus.ESMplusplusForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_esm_plusplus.ESMplusplusForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_esm_plusplus.ESMplusplusForTokenClassification",
        }
        tokenizer = model.tokenizer
        tokenizer.push_to_hub(model_path)
        model.push_to_hub(model_path)
        api.upload_file(
            path_or_fileobj="esm_plusplus/modeling_esm_plusplus.py",
            path_in_repo="modeling_esm_plusplus.py",
            repo_id=model_path,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="embedding_mixin.py",
            path_in_repo="embedding_mixin.py",
            repo_id=model_path,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="entrypoint_setup.py",
            path_in_repo="entrypoint_setup.py",
            repo_id=model_path,
            repo_type="model",
        )
