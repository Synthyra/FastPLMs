import copy
import importlib.util
import pathlib
import sys
import types

import torch
from huggingface_hub import HfApi, login
from transformers import AutoModelForMaskedLM

from esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig, ESMplusplusForMaskedLM
from weight_parity_utils import assert_state_dict_equal, assert_model_parameters_fp32


MODEL_DICT = {
    "Synthyra/ESMplusplus_small": "esmc-300",
    "Synthyra/ESMplusplus_large": "esmc-600",
}


def _resolve_model_items(model_paths: list[str] | None) -> list[tuple[str, str]]:
    if model_paths is None:
        return list(MODEL_DICT.items())

    selected_items: list[tuple[str, str]] = []
    for model_path in model_paths:
        assert model_path in MODEL_DICT, (
            f"Unknown model path {model_path}. "
            f"Valid options: {sorted(MODEL_DICT.keys())}"
        )
        selected_items.append((model_path, MODEL_DICT[model_path]))
    return selected_items


def _ensure_local_esm_module_on_path() -> pathlib.Path:
    script_root = pathlib.Path(__file__).resolve().parents[1]
    candidates = [script_root / "esm"]

    cwd = pathlib.Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / "esm")

    deduplicated_candidates: list[pathlib.Path] = []
    seen = set()
    for candidate in candidates:
        candidate_resolved = candidate.resolve()
        candidate_key = str(candidate_resolved)
        if candidate_key not in seen:
            seen.add(candidate_key)
            deduplicated_candidates.append(candidate_resolved)

    for candidate in deduplicated_candidates:
        package_marker = candidate / "esm" / "__init__.py"
        if package_marker.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate

    raise FileNotFoundError(
        "Unable to locate local esm submodule. "
        f"Checked: {', '.join([str(path) for path in deduplicated_candidates])}"
    )


def _load_official_esmc_model(esmc_model_key: str) -> torch.nn.Module:
    _ensure_local_esm_module_on_path()
    _ensure_zstd_module_stub()
    from esm.pretrained import ESMC_300M_202412, ESMC_600M_202412

    if esmc_model_key == "esmc-300":
        official_model = ESMC_300M_202412(device="cpu", use_flash_attn=False)
    elif esmc_model_key == "esmc-600":
        official_model = ESMC_600M_202412(device="cpu", use_flash_attn=False)
    else:
        raise ValueError(f"Unsupported official ESMC model key: {esmc_model_key}")

    official_model = official_model.eval().cpu().to(torch.float32)
    assert_model_parameters_fp32(
        model=official_model,
        model_name=f"official ESMC model ({esmc_model_key})",
    )
    return official_model


def _ensure_zstd_module_stub() -> None:
    if "zstd" in sys.modules:
        return
    try:
        zstd_spec = importlib.util.find_spec("zstd")
    except ValueError:
        zstd_spec = None
    if zstd_spec is not None:
        return

    zstd_module = types.ModuleType("zstd")

    def _missing_zstd_uncompress(data: bytes) -> bytes:
        raise ModuleNotFoundError(
            "No module named 'zstd'. Install zstd if compressed tensor "
            "deserialization is required."
        )

    zstd_module.ZSTD_uncompress = _missing_zstd_uncompress
    sys.modules["zstd"] = zstd_module


def _build_local_esmplusplus_model(esmc_model_key: str) -> ESMplusplusForMaskedLM:
    if esmc_model_key == "esmc-300":
        config = ESMplusplusConfig(
            hidden_size=960,
            num_attention_heads=15,
            num_hidden_layers=30,
        )
    elif esmc_model_key == "esmc-600":
        config = ESMplusplusConfig(
            hidden_size=1152,
            num_attention_heads=18,
            num_hidden_layers=36,
        )
    else:
        raise ValueError(f"Unsupported local ESM++ model key: {esmc_model_key}")

    model = ESMplusplusForMaskedLM(config).eval().cpu().to(torch.float32)
    return model


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

    for model_path, esmc_model_key in _resolve_model_items(args.model_paths):
        official_model = _load_official_esmc_model(esmc_model_key)
        model = _build_local_esmplusplus_model(esmc_model_key)
        model.config.auto_map = {
            "AutoConfig": "modeling_esm_plusplus.ESMplusplusConfig",
            "AutoModel": "modeling_esm_plusplus.ESMplusplusModel",
            "AutoModelForMaskedLM": "modeling_esm_plusplus.ESMplusplusForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_esm_plusplus.ESMplusplusForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_esm_plusplus.ESMplusplusForTokenClassification",
        }
        load_result = model.load_state_dict(official_model.state_dict(), strict=False)
        assert len(load_result.missing_keys) == 0, (
            f"Missing keys while mapping official ESMC weights for {esmc_model_key}: "
            f"{load_result.missing_keys[:20]}"
        )
        assert len(load_result.unexpected_keys) == 0, (
            f"Unexpected keys while mapping official ESMC weights for {esmc_model_key}: "
            f"{load_result.unexpected_keys[:20]}"
        )
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
