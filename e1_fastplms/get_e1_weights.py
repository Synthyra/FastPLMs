import copy
import importlib
import importlib.util
import pathlib
import shutil
import sys
import types

import torch
from huggingface_hub import HfApi, login
from transformers import AutoModelForMaskedLM

from e1_fastplms.modeling_e1 import E1Config, E1ForMaskedLM
from weight_parity_utils import assert_state_dict_equal, assert_model_parameters_fp32


MODEL_DICT = {
    "Profluent-E1-150M": "Profluent-Bio/E1-150m",
    "Profluent-E1-300M": "Profluent-Bio/E1-300m",
    "Profluent-E1-600M": "Profluent-Bio/E1-600m",
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


def _ensure_local_e1_module_on_path() -> pathlib.Path:
    script_root = pathlib.Path(__file__).resolve().parents[1]
    candidates = [script_root / "e1" / "src", script_root / "E1" / "src"]

    cwd = pathlib.Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / "e1" / "src")
        candidates.append(parent / "E1" / "src")

    deduplicated_candidates: list[pathlib.Path] = []
    seen = set()
    for candidate in candidates:
        candidate_resolved = candidate.resolve()
        candidate_key = str(candidate_resolved)
        if candidate_key not in seen:
            seen.add(candidate_key)
            deduplicated_candidates.append(candidate_resolved)

    for candidate in deduplicated_candidates:
        package_marker = candidate / "E1" / "__init__.py"
        if package_marker.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate

    raise FileNotFoundError(
        "Unable to locate local E1 submodule. "
        f"Checked: {', '.join([str(path) for path in deduplicated_candidates])}"
    )


def _ensure_local_e1_tokenizer_json() -> None:
    e1_spec = importlib.util.find_spec("E1")
    assert e1_spec is not None, "Unable to find E1 package after path setup."
    assert e1_spec.origin is not None, "E1 package origin is required."
    e1_package_dir = pathlib.Path(e1_spec.origin).resolve().parent
    tokenizer_path = e1_package_dir / "tokenizer.json"
    if tokenizer_path.exists():
        return

    script_root = pathlib.Path(__file__).resolve().parents[1]
    fallback_tokenizer_path = script_root / "e1_fastplms" / "tokenizer.json"
    assert fallback_tokenizer_path.exists(), (
        f"Missing fallback tokenizer at {fallback_tokenizer_path}."
    )
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fallback_tokenizer_path, tokenizer_path)
    assert tokenizer_path.exists(), f"Failed to create tokenizer at {tokenizer_path}."


def _ensure_kernels_module_stub() -> None:
    if "kernels" in sys.modules:
        return
    kernels_spec = importlib.util.find_spec("kernels")
    if kernels_spec is not None:
        return

    kernels_module = types.ModuleType("kernels")

    def _missing_get_kernel(kernel_name: str):
        raise ModuleNotFoundError(
            f"No module named 'kernels' while requesting kernel '{kernel_name}'."
        )

    kernels_module.get_kernel = _missing_get_kernel
    sys.modules["kernels"] = kernels_module


def _load_official_e1_model(source_repo: str) -> torch.nn.Module:
    _ensure_local_e1_module_on_path()
    _ensure_local_e1_tokenizer_json()
    _ensure_kernels_module_stub()
    modeling_module = importlib.import_module("E1.modeling")
    OfficialE1ForMaskedLM = modeling_module.E1ForMaskedLM
    official_model = (
        OfficialE1ForMaskedLM.from_pretrained(source_repo)
        .eval()
        .cpu()
        .to(torch.float32)
    )
    assert_model_parameters_fp32(
        model=official_model,
        model_name=f"official E1 model ({source_repo})",
    )
    return official_model


if __name__ == "__main__":
    # py -m e1_fastplms.get_e1_weights
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
        official_model = _load_official_e1_model(source_repo)
        config = E1Config.from_pretrained(source_repo)
        config.auto_map = {
            "AutoConfig": "modeling_e1.E1Config",
            "AutoModel": "modeling_e1.E1Model",
            "AutoModelForMaskedLM": "modeling_e1.E1ForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_e1.E1ForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_e1.E1ForTokenClassification",
        }
        model = E1ForMaskedLM(config=config).eval().cpu().to(torch.float32)
        load_result = model.load_state_dict(official_model.state_dict(), strict=False)
        assert len(load_result.missing_keys) == 0, (
            f"Missing keys while mapping official E1 weights for {source_repo}: "
            f"{load_result.missing_keys[:20]}"
        )
        assert len(load_result.unexpected_keys) == 0, (
            f"Unexpected keys while mapping official E1 weights for {source_repo}: "
            f"{load_result.unexpected_keys[:20]}"
        )
        model.mlm_head[0].weight = copy.deepcopy(official_model.mlm_head[0].weight)
        model.mlm_head[0].bias = copy.deepcopy(official_model.mlm_head[0].bias)
        model.mlm_head[2].weight = copy.deepcopy(official_model.mlm_head[2].weight)
        model.mlm_head[2].bias = copy.deepcopy(official_model.mlm_head[2].bias)
        model.mlm_head[3].weight = copy.deepcopy(official_model.mlm_head[3].weight)
        model.mlm_head[3].bias = copy.deepcopy(official_model.mlm_head[3].bias)
        assert_model_parameters_fp32(
            model=model,
            model_name=f"mapped E1 model ({source_repo})",
        )
        assert_state_dict_equal(
            reference_state_dict=official_model.state_dict(),
            candidate_state_dict=model.state_dict(),
            context=f"E1 weight parity ({source_repo})",
        )

        repo_id = "Synthyra/" + model_name
        if args.dry_run:
            print(f"[dry_run] validated E1 parity for {repo_id} <- {source_repo}")
            continue

        model.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj="e1_fastplms/modeling_e1.py",
            path_in_repo="modeling_e1.py",
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
            path_or_fileobj="e1_fastplms/tokenizer.json",
            path_in_repo="tokenizer.json",
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
            context=f"E1 weight parity post-download ({repo_id})",
        )
