import importlib

import torch
from huggingface_hub import HfApi, login
from test_scripts.common import (
    LOAD_DTYPE,
    _ensure_local_dplm_module_on_path,
    _import_byprot_module_with_dataclass_patch,
    _install_byprot_package_shims,
    _patch_lightning_fabric_fsdp_for_byprot,
    _patch_transformers_esm_star_exports_for_byprot,
)

from dplm_fastplms.dplm2 import DPLM2Config, DPLM2ForMaskedLM
from weight_parity_utils import assert_fp32_state_dict_equal, assert_model_parameters_fp32


MODEL_DICT = {
    "Synthyra/DPLM2-150M": "airkingbd/dplm2_150m",
    "Synthyra/DPLM2-650M": "airkingbd/dplm2_650m",
    "Synthyra/DPLM2-3B": "airkingbd/dplm2_3b",
}


def _resolve_model_items(repo_ids: list[str] | None) -> list[tuple[str, str]]:
    if repo_ids is None:
        return list(MODEL_DICT.items())

    selected_items: list[tuple[str, str]] = []
    for repo_id in repo_ids:
        assert repo_id in MODEL_DICT, (
            f"Unknown repo id {repo_id}. "
            f"Valid options: {sorted(MODEL_DICT.keys())}"
        )
        selected_items.append((repo_id, MODEL_DICT[repo_id]))
    return selected_items


def _load_official_dplm2_source_model(source_repo: str) -> torch.nn.Module:
    dplm_src_path = _ensure_local_dplm_module_on_path()
    _patch_lightning_fabric_fsdp_for_byprot()
    _patch_transformers_esm_star_exports_for_byprot()
    _install_byprot_package_shims(dplm_src_path)

    dplm2_module = _import_byprot_module_with_dataclass_patch(
        module_name="byprot.models.dplm2.dplm2",
        purge_prefixes=[
            "byprot.models.utils",
            "byprot.models.dplm2",
        ],
    )
    dplm2_modeling_module = importlib.import_module(
        "byprot.models.dplm2.modules.dplm2_modeling_esm"
    )
    EsmForDPLM2 = dplm2_modeling_module.EsmForDPLM2
    EsmForDPLM2.all_tied_weights_keys = {}
    MultimodalDiffusionProteinLanguageModel = (
        dplm2_module.MultimodalDiffusionProteinLanguageModel
    )
    official_model = MultimodalDiffusionProteinLanguageModel.from_pretrained(
        source_repo
    ).to(
        device=torch.device("cpu"),
        dtype=LOAD_DTYPE,
    ).eval()
    official_model = official_model.eval().cpu().to(torch.float32)
    assert_model_parameters_fp32(
        model=official_model.net,
        model_name=f"official DPLM2 net ({source_repo})",
    )
    return official_model


if __name__ == "__main__":
    # py -m dplm_fastplms.get_dplm2_weights
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

    for repo_id, source_repo in _resolve_model_items(args.repo_ids):
        official_model = _load_official_dplm2_source_model(source_repo)
        official_state_dict = official_model.net.state_dict()
        excluded_official_keys = {
            "esm.contact_head.regression.weight",
            "esm.contact_head.regression.bias",
        }
        filtered_official_state_dict: dict[str, torch.Tensor] = {}
        for tensor_name in official_state_dict:
            if tensor_name in excluded_official_keys:
                continue
            filtered_official_state_dict[tensor_name] = official_state_dict[tensor_name]
        dropped_official_keys = sorted(
            set(official_state_dict.keys()) - set(filtered_official_state_dict.keys())
        )
        assert dropped_official_keys == sorted(excluded_official_keys), (
            f"Unexpected excluded official keys for {source_repo}: {dropped_official_keys}"
        )

        config = DPLM2Config.from_pretrained(source_repo)
        config.auto_map = {
            "AutoConfig": "dplm2.DPLM2Config",
            "AutoModel": "dplm2.DPLM2Model",
            "AutoModelForMaskedLM": "dplm2.DPLM2ForMaskedLM",
            "AutoModelForSequenceClassification": "dplm2.DPLM2ForSequenceClassification",
            "AutoModelForTokenClassification": "dplm2.DPLM2ForTokenClassification",
        }
        config.tie_word_embeddings = False
        model = DPLM2ForMaskedLM(config=config).eval().cpu().to(torch.float32)
        load_result = model.load_state_dict(filtered_official_state_dict, strict=False)
        assert len(load_result.missing_keys) == 0, (
            f"Missing keys while mapping official DPLM2 weights for {source_repo}: "
            f"{load_result.missing_keys[:20]}"
        )
        assert len(load_result.unexpected_keys) == 0, (
            f"Unexpected keys while mapping official DPLM2 weights for {source_repo}: "
            f"{load_result.unexpected_keys[:20]}"
        )
        assert_model_parameters_fp32(
            model=model,
            model_name=f"mapped DPLM2 model ({source_repo})",
        )
        assert_fp32_state_dict_equal(
            reference_state_dict=filtered_official_state_dict,
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
            path_or_fileobj="dplm_fastplms/dplm2.py",
            path_in_repo="dplm2.py",
            repo_id=repo_id,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="dplm_fastplms/base_tokenizer.py",
            path_in_repo="base_tokenizer.py",
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
