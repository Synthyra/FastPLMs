import argparse
import copy
import shutil
import sys
import urllib.request
from pathlib import Path

import torch
from huggingface_hub import HfApi, login

from boltz_fastplms.modeling_boltz2 import (
    Boltz2Model,
    _filtered_kwargs,
    _state_dict_without_wrappers,
    _to_plain_python,
)
from weight_parity_utils import assert_fp32_state_dict_equal, assert_model_parameters_fp32


BOLTZ2_CKPT_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt"


def _download_checkpoint_if_needed(checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        urllib.request.urlretrieve(BOLTZ2_CKPT_URL, str(checkpoint_path))  # noqa: S310
    return checkpoint_path


def _copy_runtime_package(output_dir: Path) -> None:
    source_pkg = Path(__file__).resolve().parent
    project_root = source_pkg.parent
    runtime_files = [
        "__init__.py",
        "modeling_boltz2.py",
        "minimal_featurizer.py",
        "minimal_structures.py",
        "cif_writer.py",
    ]
    for filename in runtime_files:
        shutil.copyfile(source_pkg / filename, output_dir / filename)
    shutil.copyfile(project_root / "entrypoint_setup.py", output_dir / "entrypoint_setup.py")
    for flat_module in source_pkg.glob("vb_*.py"):
        shutil.copyfile(flat_module, output_dir / flat_module.name)


def _ensure_local_boltz_module_on_path() -> Path:
    script_root = Path(__file__).resolve().parents[1]
    candidates = [script_root / "boltz" / "src"]

    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / "boltz" / "src")

    deduplicated_candidates: list[Path] = []
    seen = set()
    for candidate in candidates:
        candidate_resolved = candidate.resolve()
        candidate_key = str(candidate_resolved)
        if candidate_key not in seen:
            seen.add(candidate_key)
            deduplicated_candidates.append(candidate_resolved)

    for candidate in deduplicated_candidates:
        package_marker = candidate / "boltz" / "__init__.py"
        if package_marker.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate

    raise FileNotFoundError(
        "Unable to locate local boltz submodule. "
        f"Checked: {', '.join([str(path) for path in deduplicated_candidates])}"
    )


def _load_official_boltz2_model(
    checkpoint_path: Path,
    use_kernels: bool,
) -> torch.nn.Module:
    _ensure_local_boltz_module_on_path()
    from boltz.model.models.boltz2 import Boltz2 as OfficialBoltz2
    from boltz.model.modules.diffusionv2 import AtomDiffusion

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    assert isinstance(checkpoint, dict), "Checkpoint must deserialize to a dictionary."
    assert "hyper_parameters" in checkpoint, "Checkpoint missing 'hyper_parameters'."
    assert "state_dict" in checkpoint, "Checkpoint missing 'state_dict'."
    hyper_parameters = checkpoint["hyper_parameters"]
    state_dict = checkpoint["state_dict"]
    assert isinstance(hyper_parameters, dict), "Checkpoint hyper_parameters must be a dictionary."
    assert isinstance(state_dict, dict), "Checkpoint state_dict must be a dictionary."

    init_kwargs = _filtered_kwargs(
        target=OfficialBoltz2,
        kwargs=_to_plain_python(copy.deepcopy(hyper_parameters)),
    )
    if "use_kernels" in init_kwargs:
        init_kwargs["use_kernels"] = use_kernels
    assert "pairformer_args" in init_kwargs, (
        "Checkpoint hyperparameters missing pairformer_args for official Boltz2."
    )
    raw_pairformer_args = init_kwargs["pairformer_args"]
    assert isinstance(raw_pairformer_args, dict), "Expected pairformer_args to be a dictionary."
    pairformer_args = _to_plain_python(copy.deepcopy(raw_pairformer_args))
    assert isinstance(pairformer_args, dict), "Expected normalized pairformer_args to be a dictionary."
    pairformer_args["v2"] = True
    init_kwargs["pairformer_args"] = pairformer_args
    assert "diffusion_process_args" in init_kwargs, (
        "Checkpoint hyperparameters missing diffusion_process_args for official Boltz2."
    )
    raw_diffusion_process_args = init_kwargs["diffusion_process_args"]
    assert isinstance(raw_diffusion_process_args, dict), (
        "Expected diffusion_process_args to be a dictionary."
    )
    filtered_diffusion_process_args = _filtered_kwargs(
        target=AtomDiffusion,
        kwargs=raw_diffusion_process_args,
    )
    sanitized_diffusion_process_args: dict[str, object] = {}
    for key in filtered_diffusion_process_args:
        if key == "score_model_args":
            continue
        sanitized_diffusion_process_args[key] = filtered_diffusion_process_args[key]
    init_kwargs["diffusion_process_args"] = sanitized_diffusion_process_args
    official_model = OfficialBoltz2(**init_kwargs)

    cleaned_state_dict = _state_dict_without_wrappers(state_dict)
    target_keys = set(official_model.state_dict().keys())
    filtered_state_dict: dict[str, torch.Tensor] = {}
    for key in cleaned_state_dict:
        if key in target_keys:
            filtered_state_dict[key] = cleaned_state_dict[key]
    missing_keys = sorted(target_keys.difference(filtered_state_dict.keys()))
    assert len(missing_keys) == 0, (
        "Official Boltz2 model is missing required checkpoint keys. "
        f"Missing keys (first 20): {missing_keys[:20]}"
    )
    load_result = official_model.load_state_dict(filtered_state_dict, strict=False)
    assert len(load_result.missing_keys) == 0, (
        "Missing keys while loading official Boltz2 checkpoint. "
        f"Missing keys (first 20): {load_result.missing_keys[:20]}"
    )
    assert len(load_result.unexpected_keys) == 0, (
        "Unexpected keys while loading official Boltz2 checkpoint. "
        f"Unexpected keys (first 20): {load_result.unexpected_keys[:20]}"
    )

    official_model = official_model.eval().cpu().to(torch.float32)
    assert_model_parameters_fp32(
        model=official_model,
        model_name="official Boltz2 model",
    )
    return official_model


if __name__ == "__main__":
    # py -m boltz_fastplms.get_boltz2_weights
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="boltz_fastplms/weights/boltz2_conf.ckpt")
    parser.add_argument("--output_dir", type=str, default="boltz2_automodel_export")
    parser.add_argument("--repo_id", type=str, default="Synthyra/Boltz2")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--use_kernels", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    checkpoint_path = _download_checkpoint_if_needed(Path(args.checkpoint_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    official_model = _load_official_boltz2_model(
        checkpoint_path=checkpoint_path,
        use_kernels=args.use_kernels,
    )
    model = Boltz2Model.from_boltz_checkpoint(
        checkpoint_path=str(checkpoint_path),
        use_kernels=args.use_kernels,
    )
    model = model.eval().cpu().to(torch.float32)
    assert_model_parameters_fp32(
        model=model.core,
        model_name="mapped Boltz2 inference core",
    )
    official_state_dict = official_model.state_dict()
    candidate_state_dict = model.core.state_dict()
    official_keys = set(official_state_dict.keys())
    candidate_keys = set(candidate_state_dict.keys())
    missing_official_keys = sorted(candidate_keys - official_keys)
    assert len(missing_official_keys) == 0, (
        "Official Boltz2 model is missing inference-core keys required by FastPLMs. "
        f"Missing keys (first 20): {missing_official_keys[:20]}"
    )
    excluded_official_keys = sorted(official_keys - candidate_keys)
    allowed_excluded_prefixes = (
        "template_module.",
        "bfactor_module.",
    )
    unexpected_excluded_official_keys: list[str] = []
    for key in excluded_official_keys:
        is_allowed = False
        for prefix in allowed_excluded_prefixes:
            if key.startswith(prefix):
                is_allowed = True
                break
        if is_allowed is False:
            unexpected_excluded_official_keys.append(key)
    assert len(unexpected_excluded_official_keys) == 0, (
        "Unexpected official Boltz2 keys not present in FastPLMs inference core. "
        f"Unexpected keys (first 20): {unexpected_excluded_official_keys[:20]}"
    )
    filtered_official_state_dict: dict[str, torch.Tensor] = {}
    for key in candidate_state_dict:
        filtered_official_state_dict[key] = official_state_dict[key]
    assert_fp32_state_dict_equal(
        reference_state_dict=filtered_official_state_dict,
        candidate_state_dict=candidate_state_dict,
        context="Boltz2 weight parity",
    )

    model.config.auto_map = {
        "AutoConfig": "modeling_boltz2.Boltz2Config",
        "AutoModel": "modeling_boltz2.Boltz2Model",
    }
    if args.dry_run:
        print(f"[dry_run] validated Boltz2 parity for checkpoint {checkpoint_path}")
    else:
        model.save_pretrained(str(output_dir))
        _copy_runtime_package(output_dir=output_dir)

    if args.repo_id is not None and args.dry_run is False:
        if args.hf_token is not None:
            login(token=args.hf_token)
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.repo_id,
            repo_type="model",
        )
