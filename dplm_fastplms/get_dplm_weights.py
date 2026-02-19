import copy
import importlib
import importlib.util
import sys
import types

import torch
from huggingface_hub import HfApi, login
from transformers import AutoModelForMaskedLM
from testing.common import (
    LOAD_DTYPE,
    _ensure_local_dplm_module_on_path,
    _import_byprot_module_with_dataclass_patch,
    _install_byprot_package_shims,
    _patch_lightning_fabric_fsdp_for_byprot,
    _patch_transformers_esm_star_exports_for_byprot,
)

from dplm_fastplms.modeling_dplm import DPLMConfig, DPLMForMaskedLM
from weight_parity_utils import assert_state_dict_equal, assert_model_parameters_fp32


MODEL_DICT = {
    "Synthyra/DPLM-150M": "airkingbd/dplm_150m",
    "Synthyra/DPLM-650M": "airkingbd/dplm_650m",
    "Synthyra/DPLM-3B": "airkingbd/dplm_3b",
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


def _load_official_dplm_source_model(source_repo: str) -> torch.nn.Module:
    _ensure_imp_module_stub()
    dplm_src_path = _ensure_local_dplm_module_on_path()
    _patch_lightning_fabric_fsdp_for_byprot()
    _patch_transformers_esm_star_exports_for_byprot()
    _install_byprot_package_shims(dplm_src_path)

    dplm_module = _import_byprot_module_with_dataclass_patch(
        module_name="byprot.models.dplm.dplm",
        purge_prefixes=[
            "byprot.models.utils",
            "byprot.models.dplm",
        ],
    )
    dplm_modeling_module = importlib.import_module(
        "byprot.models.dplm.modules.dplm_modeling_esm"
    )
    EsmForDPLM = dplm_modeling_module.EsmForDPLM
    EsmForDPLM.all_tied_weights_keys = {}
    DiffusionProteinLanguageModel = dplm_module.DiffusionProteinLanguageModel
    official_model = DiffusionProteinLanguageModel.from_pretrained(source_repo).to(
        device=torch.device("cpu"),
        dtype=LOAD_DTYPE,
    ).eval()
    official_model = official_model.eval().cpu().to(torch.float32)
    assert_model_parameters_fp32(
        model=official_model.net,
        model_name=f"official DPLM net ({source_repo})",
    )
    return official_model


def _ensure_imp_module_stub() -> None:
    try:
        import imp  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    if "imp" in sys.modules:
        return

    imp_module = types.ModuleType("imp")

    def _new_module(name: str):
        return types.ModuleType(name)

    def _reload(module):
        return importlib.reload(module)

    def _find_module(name, path=None):
        spec = importlib.util.find_spec(name)
        if spec is None:
            raise ImportError(f"Cannot find module {name}")
        return None, spec.origin, (None, None, None)

    def _load_module(name, file=None, filename=None, details=None):  # noqa: ARG001
        return importlib.import_module(name)

    def _load_source(name, pathname, file=None):  # noqa: ARG001
        spec = importlib.util.spec_from_file_location(name, pathname)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load source for module {name} from {pathname}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    imp_module.new_module = _new_module
    imp_module.reload = _reload
    imp_module.find_module = _find_module
    imp_module.load_module = _load_module
    imp_module.load_source = _load_source
    sys.modules["imp"] = imp_module


if __name__ == "__main__":
    # py -m dplm_fastplms.get_dplm_weights
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
        official_model = _load_official_dplm_source_model(source_repo)
        official_state_dict = official_model.net.state_dict()

        config = DPLMConfig.from_pretrained(source_repo)
        config.auto_map = {
            "AutoConfig": "modeling_dplm.DPLMConfig",
            "AutoModel": "modeling_dplm.DPLMModel",
            "AutoModelForMaskedLM": "modeling_dplm.DPLMForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_dplm.DPLMForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_dplm.DPLMForTokenClassification",
        }
        config.tie_word_embeddings = False
        model = DPLMForMaskedLM(config=config).eval().cpu().to(torch.float32)
        load_result = model.load_state_dict(official_state_dict, strict=False)
        assert len(load_result.missing_keys) == 0, (
            f"Missing keys while mapping official DPLM weights for {source_repo}: "
            f"{load_result.missing_keys[:20]}"
        )
        assert len(load_result.unexpected_keys) == 0, (
            f"Unexpected keys while mapping official DPLM weights for {source_repo}: "
            f"{load_result.unexpected_keys[:20]}"
        )
        model.lm_head.dense.weight = copy.deepcopy(official_model.net.lm_head.dense.weight)
        model.lm_head.dense.bias = copy.deepcopy(official_model.net.lm_head.dense.bias)
        model.lm_head.decoder.weight = copy.deepcopy(official_model.net.lm_head.decoder.weight)
        model.lm_head.decoder.bias = copy.deepcopy(official_model.net.lm_head.decoder.bias)
        model.lm_head.layer_norm.weight = copy.deepcopy(official_model.net.lm_head.layer_norm.weight)
        model.lm_head.layer_norm.bias = copy.deepcopy(official_model.net.lm_head.layer_norm.bias)
        assert_model_parameters_fp32(
            model=model,
            model_name=f"mapped DPLM model ({source_repo})",
        )
        assert_state_dict_equal(
            reference_state_dict=official_state_dict,
            candidate_state_dict=model.state_dict(),
            context=f"DPLM weight parity ({source_repo})",
        )

        if args.dry_run:
            print(f"[dry_run] validated DPLM parity for {repo_id} <- {source_repo}")
            continue

        tokenizer = model.tokenizer
        tokenizer.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj="dplm_fastplms/modeling_dplm.py",
            path_in_repo="modeling_dplm.py",
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
        downloaded_model = AutoModelForMaskedLM.from_pretrained(
            repo_id,
            dtype=torch.float32,
            device_map="cpu",
            force_download=True,
            trust_remote_code=True,
        )
        assert_state_dict_equal(
            reference_state_dict=official_state_dict,
            candidate_state_dict=downloaded_model.state_dict(),
            context=f"DPLM weight parity post-download ({repo_id})",
        )
