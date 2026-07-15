"""Produce isolated ESMFold2 folding bundles for release compliance.

The reference command imports only the pinned Biohub environment. The candidate
command imports FastPLMs only after command dispatch. Both write the same
safetensors and JSON schema so the release gate can compare them without ever
loading both implementations in one Python process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file

from tests.structure.support.state_contract import (
    exact_state_contract,
    semantic_config_contract,
    validate_exact_state_contract,
    validate_semantic_config_contract,
)

schema_version = 1
reference_container = "reference-esmfold2"
supported_model_ids = (
    "esmfold2",
    "esmfold2_fast",
    "esmfold2_experimental_cutoff2025",
    "esmfold2_experimental_fast_cutoff2025",
)
# Protein G B1 is a compact, experimentally characterized single-chain fold.
fold_sequence = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
fold_seed = 17
# The checkpoints declare 14 inference steps. Shorter diagnostic schedules can
# leave even the official model outside physically valid C-alpha geometry.
fold_sampling_steps = 14

_required_outputs = (
    "atom_pad_mask",
    "distogram_logits",
    "iptm",
    "pae",
    "pae_logits",
    "plddt",
    "plddt_logits",
    "ptm",
    "sample_atom_coords",
)
_optional_outputs = ("pde_logits",)
_semantic_config_fields = (
    "confidence_head",
    "d_pair",
    "d_single",
    "disable_msa_features",
    "folding_trunk",
    "force_lm_dropout_during_inference",
    "inputs",
    "lm_d_model",
    "lm_dropout",
    "lm_encoder",
    "lm_num_layers",
    "model_type",
    "msa_encoder",
    "msa_encoder_overwrite",
    "n_relative_chain_bins",
    "n_relative_residx_bins",
    "num_diffusion_samples",
    "num_loops",
    "parcae",
    "structure_head",
    "type",
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _request_fingerprint(request: Mapping[str, Any]) -> str:
    payload = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Return the content digest of one tensor without dtype coercion."""

    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def tensor_set_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    """Hash names, dtypes, shapes, and values in deterministic key order."""

    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    return {
        "repo_id": checkpoint.repo_id,
        "revision": checkpoint.revision,
        "files": [
            {
                "path": item.path,
                "algorithm": item.algorithm,
                "digest": item.digest,
            }
            for item in checkpoint.files
        ],
    }


def prepare_requests(
    exchange_root: Path,
    *,
    model_ids: Sequence[str] = supported_model_ids,
) -> tuple[Path, ...]:
    """Write manifest-derived requests for the isolated reference service."""

    from fastplms.registry import get_model_registry

    registry = get_model_registry()
    actual_ids = tuple(spec.id for spec in registry.by_family("esmfold2"))
    if actual_ids != supported_model_ids:
        raise RuntimeError(
            "The ESMFold2 bundle schema supports exactly the four release variants; "
            f"manifest contains {actual_ids}."
        )

    request_root = exchange_root / "structure" / "requests" / reference_container
    request_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model_id in model_ids:
        if model_id not in supported_model_ids:
            raise ValueError(f"Unsupported ESMFold2 model ID: {model_id!r}")
        spec = registry[model_id]
        request = {
            "schema_version": schema_version,
            "model_id": spec.id,
            "architecture": spec.family.architecture,
            "official": _checkpoint_metadata(spec.official),
            "candidate": _checkpoint_metadata(spec.fast),
            "candidate_auto_model": spec.auto_map["AutoModel"],
            "state_transform": spec.family.state_transform,
            "backbone_model": spec.family.backbone_model,
            "attention_backend": "sdpa",
            "deterministic_algorithms": True,
            "sequence": fold_sequence,
            "seed": fold_seed,
            "sampling_steps": fold_sampling_steps,
        }
        request["request_sha256"] = _request_fingerprint(request)
        path = request_root / f"{model_id}.json"
        _atomic_write_text(path, _canonical_json(request))
        paths.append(path)
    return tuple(paths)


def _validate_request(request: Mapping[str, Any]) -> None:
    if request.get("schema_version") != schema_version:
        raise ValueError("Unsupported ESMFold2 structure-bundle schema.")
    model_id = request.get("model_id")
    if model_id not in supported_model_ids:
        raise ValueError(f"Unsupported ESMFold2 model ID: {model_id!r}")
    expected = dict(request)
    observed_fingerprint = expected.pop("request_sha256", None)
    expected_fingerprint = _request_fingerprint(expected)
    if observed_fingerprint != expected_fingerprint:
        raise ValueError(
            f"{model_id}: request fingerprint mismatch "
            f"{observed_fingerprint!r} != {expected_fingerprint!r}."
        )
    for source_name in ("official", "candidate"):
        source = request.get(source_name)
        if not isinstance(source, Mapping):
            raise ValueError(f"{model_id}: missing {source_name} checkpoint metadata.")
        revision = source.get("revision")
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError(f"{model_id}: {source_name} revision is not immutable.")
    if request.get("state_transform") != "identity":
        raise ValueError(f"{model_id}: ESMFold2 requires the identity state transform.")
    backbone_model = request.get("backbone_model")
    if not isinstance(backbone_model, str) or not backbone_model:
        raise ValueError(f"{model_id}: ESMFold2 requires a logical backbone model ID.")
    if request.get("attention_backend") != "sdpa":
        raise ValueError(f"{model_id}: ESMFold2 folding compliance requires SDPA.")
    if request.get("deterministic_algorithms") is not True:
        raise ValueError(f"{model_id}: ESMFold2 folding compliance requires determinism.")


def load_request(path: Path) -> dict[str, Any]:
    """Load and validate one manifest-derived folding request."""

    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise TypeError(f"ESMFold2 request must be a JSON object: {path}")
    _validate_request(request)
    return request


def _is_experimental(request: Mapping[str, Any]) -> bool:
    return "experimental" in str(request["model_id"])


def _model_package(model: torch.nn.Module) -> str:
    return model.__class__.__module__.rsplit(".", maxsplit=1)[0]


@contextmanager
def _stable_cuda_numerics():
    """Use deterministic CUDA algorithms with TF32 and autotuning disabled."""

    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_deterministic = torch.backends.cudnn.deterministic
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_deterministic
        torch.use_deterministic_algorithms(previous_algorithms)


def _run_fold(
    model: torch.nn.Module,
    request: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    feature_module = importlib.import_module(f"{_model_package(model)}.protein_utils")
    common_module = importlib.import_module(f"{_model_package(model)}.modeling_esmfold2_common")
    cpu_features = feature_module.prepare_protein_features(request["sequence"])
    if not isinstance(cpu_features, Mapping) or not all(
        torch.is_tensor(value) for value in cpu_features.values()
    ):
        raise TypeError("Official feature preparation did not return a tensor mapping.")
    device = next(model.parameters()).device
    device_features = {name: tensor.to(device=device) for name, tensor in cpu_features.items()}

    captured_noise: list[torch.Tensor] = []
    original_randn = torch.randn

    def recording_randn(*args: Any, **kwargs: Any) -> torch.Tensor:
        tensor = original_randn(*args, **kwargs)
        if not captured_noise and tensor.ndim == 3 and tensor.shape[-1] == 3:
            captured_noise.append(tensor.detach().cpu().contiguous().clone())
        return tensor

    forward_kwargs: dict[str, Any] = {
        "num_loops": 1,
        "num_sampling_steps": int(request["sampling_steps"]),
        "num_diffusion_samples": 1,
    }
    if _is_experimental(request):
        forward_kwargs.update({"calculate_confidence": True, "seed": int(request["seed"])})
    else:
        forward_kwargs.update(
            {
                "msa_column_mask_rate": 0.0,
                "msa_subsample_at_inference": False,
            }
        )

    torch.randn = recording_randn
    try:
        with (
            _stable_cuda_numerics(),
            common_module._seed_context(int(request["seed"])),
            torch.inference_mode(),
        ):
            output = model(**device_features, **forward_kwargs)
    finally:
        torch.randn = original_randn

    if len(captured_noise) != 1:
        raise RuntimeError(
            f"Expected one initial diffusion-noise tensor, captured {len(captured_noise)}."
        )
    if not isinstance(output, Mapping):
        raise TypeError("ESMFold2 forward did not return a tensor mapping.")
    missing_outputs = sorted(set(_required_outputs).difference(output))
    if missing_outputs:
        raise RuntimeError(f"ESMFold2 fold omitted required outputs: {missing_outputs}")

    tensors = {
        f"feature__{name}": tensor.detach().cpu().contiguous().clone()
        for name, tensor in cpu_features.items()
    }
    tensors["noise__initial_standard_normal"] = captured_noise[0]
    for name in (*_required_outputs, *_optional_outputs):
        value = output.get(name)
        if torch.is_tensor(value):
            tensors[f"output__{name}"] = value.detach().cpu().contiguous().clone()
    return tensors


def _environment_metadata() -> dict[str, Any]:
    import transformers

    transformer_engine_version = None
    try:
        import transformer_engine

        transformer_engine_version = transformer_engine.__version__
    except ImportError:
        pass
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": device_name,
        "transformer_engine": transformer_engine_version,
    }


def _precision_status(value: object) -> dict[str, Any]:
    if is_dataclass(value):
        status = asdict(value)
    else:
        status = {
            name: getattr(value, name)
            for name in (
                "requested",
                "resolved",
                "reason",
                "device",
                "transformer_engine_version",
            )
            if hasattr(value, name)
        }
    return {
        name: str(item) if isinstance(item, torch.device) else item for name, item in status.items()
    }


def _esmfold2_semantic_config(
    config: object,
    *,
    backbone_model: str,
) -> dict[str, Any]:
    """Normalize official and mirrored configs to inference semantics.

    Transformers generation defaults, candidate attention/precision policy,
    and the official-versus-mirror Hub identifier are packaging or runtime
    concerns. The logical backbone identity comes from the typed manifest.
    """

    if isinstance(config, Mapping):
        raw = dict(config)
    else:
        to_dict = getattr(config, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("ESMFold2 semantic config must be a mapping or expose to_dict().")
        raw = to_dict()
    missing = [name for name in _semantic_config_fields if name not in raw]
    if missing:
        raise ValueError(f"ESMFold2 semantic config omits required fields: {missing}.")
    fields = {name: raw[name] for name in _semantic_config_fields}
    fields["backbone_model"] = backbone_model
    return semantic_config_contract(fields)


def _load_reference_model(
    request: Mapping[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from transformers.models.esmfold2.modeling_esmfold2_experimental import (
        ESMFold2ExperimentalModel,
    )

    source = request["official"]
    config = ESMFold2Config.from_pretrained(
        source["repo_id"],
        revision=source["revision"],
    )
    model_class = ESMFold2ExperimentalModel if config.type == "experimental" else ESMFold2Model
    model = model_class.from_pretrained(
        source["repo_id"],
        revision=source["revision"],
        config=config,
        load_esmc=False,
        dtype=torch.float32,
    )
    model = model.eval().to(device=device, dtype=torch.float32)
    if config.type == "experimental":
        model.load_esmc(model.config.esmc_id)
    else:
        model.load_esmc(model.config.esmc_id, precision="bf16")
    return model


def _load_candidate_model(
    request: Mapping[str, Any],
    device: torch.device,
    precision: Literal["bf16", "fp8"],
) -> torch.nn.Module:
    from fastplms.registry import get_model_registry

    spec = get_model_registry()[request["model_id"]]
    source = request["candidate"]
    if source["repo_id"] != spec.fast.repo_id or source["revision"] != spec.fast.revision:
        raise RuntimeError(f"{spec.id}: candidate request disagrees with models.toml.")
    if request["candidate_auto_model"] != spec.auto_map["AutoModel"]:
        raise RuntimeError(f"{spec.id}: candidate AutoModel request disagrees with models.toml.")
    module_name, class_name = spec.auto_map["AutoModel"].rsplit(".", maxsplit=1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    model = model_class.from_pretrained(
        source["repo_id"],
        revision=source["revision"],
        load_esmc=False,
        dtype=torch.float32,
    )
    model = model.eval().to(device=device)
    model.reload_esmc(precision=precision, device=device)
    return model


def _base_metadata(
    request: Mapping[str, Any],
    *,
    producer: Literal["reference", "candidate"],
    requested_precision: str,
    resolved_precision: str,
    precision_status: Mapping[str, Any],
    model: torch.nn.Module,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "producer": producer,
        "model_id": request["model_id"],
        "request_sha256": request["request_sha256"],
        "official": request["official"],
        "candidate": request["candidate"],
        "sequence": request["sequence"],
        "seed": request["seed"],
        "sampling_steps": request["sampling_steps"],
        "state_transform": request["state_transform"],
        "attention_backend": request["attention_backend"],
        "deterministic_algorithms": request["deterministic_algorithms"],
        "semantic_config": _esmfold2_semantic_config(
            model.config,
            backbone_model=str(request["backbone_model"]),
        ),
        "state": exact_state_contract(model, excluded_prefixes=("_esmc.",)),
        "requested_precision": requested_precision,
        "resolved_precision": resolved_precision,
        "precision_status": dict(precision_status),
        "environment": _environment_metadata(),
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_bundle(
    output_dir: Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> None:
    """Atomically publish one normalized structure bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = {
        name: tensor.detach().cpu().contiguous().clone() for name, tensor in sorted(tensors.items())
    }
    feature_tensors = {
        name.removeprefix("feature__"): tensor
        for name, tensor in normalized.items()
        if name.startswith("feature__")
    }
    complete_metadata = dict(metadata)
    complete_metadata.update(
        {
            "feature_sha256": tensor_set_sha256(feature_tensors),
            "diffusion_noise_sha256": tensor_sha256(normalized["noise__initial_standard_normal"]),
            "tensor_hashes": {name: tensor_sha256(tensor) for name, tensor in normalized.items()},
            "tensor_keys": sorted(normalized),
        }
    )

    bundle_path = output_dir / "bundle.safetensors"
    handle, temporary_name = tempfile.mkstemp(
        dir=output_dir,
        prefix=".bundle.",
        suffix=".safetensors.tmp",
    )
    os.close(handle)
    try:
        save_file(normalized, temporary_name)
        os.replace(temporary_name, bundle_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    _atomic_write_text(output_dir / "metadata.json", _canonical_json(complete_metadata))


def load_bundle(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a bundle and verify every declared tensor digest."""

    tensor_path = path / "bundle.safetensors"
    metadata_path = path / "metadata.json"
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing ESMFold2 structure bundle under {path}. Run native producers first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != schema_version:
        raise ValueError(f"Unsupported ESMFold2 bundle schema under {path}.")
    tensors = load_file(tensor_path, device="cpu")
    if sorted(tensors) != metadata.get("tensor_keys"):
        raise ValueError(f"Tensor-key mismatch in ESMFold2 bundle {path}.")
    expected_hashes = metadata.get("tensor_hashes")
    observed_hashes = {name: tensor_sha256(tensor) for name, tensor in tensors.items()}
    if observed_hashes != expected_hashes:
        raise ValueError(f"Tensor hash mismatch in ESMFold2 bundle {path}.")
    features = {
        name.removeprefix("feature__"): tensor
        for name, tensor in tensors.items()
        if name.startswith("feature__")
    }
    if tensor_set_sha256(features) != metadata.get("feature_sha256"):
        raise ValueError(f"Feature hash mismatch in ESMFold2 bundle {path}.")
    if tensor_sha256(tensors["noise__initial_standard_normal"]) != metadata.get(
        "diffusion_noise_sha256"
    ):
        raise ValueError(f"Diffusion-noise hash mismatch in ESMFold2 bundle {path}.")
    validate_exact_state_contract(metadata.get("state"))
    validate_semantic_config_contract(metadata.get("semantic_config"))
    return tensors, metadata


def produce_reference(request_path: Path, output_dir: Path) -> None:
    """Produce an official BF16 bundle in the native Biohub service."""

    request = load_request(request_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Official ESMFold2 structure bundles require CUDA.")
    device = torch.device("cuda")
    model = _load_reference_model(request, device)
    try:
        metadata = _base_metadata(
            request,
            producer="reference",
            requested_precision="bf16",
            resolved_precision="bf16",
            precision_status={
                "requested": "bf16",
                "resolved": "bf16",
                "reason": (
                    "Pinned Biohub config retains FP32 folding weights; the public CUDA "
                    "forward applies internal BF16 autocast and loads ESMC in BF16."
                ),
                "device": str(device),
                "transformer_engine_version": None,
                "weight_dtype": "float32",
                "compute_dtype": "bfloat16",
                "execution": "fp32_parameters_internal_bf16_autocast",
            },
            model=model,
        )
        tensors = _run_fold(model, request)
        write_bundle(output_dir, tensors, metadata)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def produce_candidate(
    request_path: Path,
    output_dir: Path,
    *,
    precision: Literal["bf16", "fp8"],
) -> None:
    """Produce one FastPLMs BF16 or FP8 bundle in the candidate service."""

    request = load_request(request_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Candidate ESMFold2 structure bundles require CUDA.")
    device = torch.device("cuda")
    model = _load_candidate_model(request, device, precision)
    try:
        status = _precision_status(model.esmc_precision_status)
        if status.get("requested") != precision or status.get("resolved") != precision:
            raise RuntimeError(
                f"{request['model_id']}: requested {precision}, resolved status is {status}."
            )
        metadata = _base_metadata(
            request,
            producer="candidate",
            requested_precision=precision,
            resolved_precision=str(status["resolved"]),
            precision_status=status,
            model=model,
        )
        tensors = _run_fold(model, request)
        write_bundle(output_dir, tensors, metadata)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _default_request(exchange_root: Path, model_id: str) -> Path:
    return exchange_root / "structure" / "requests" / reference_container / f"{model_id}.json"


def _all_prepared_requests(exchange_root: Path) -> tuple[Path, ...]:
    """Return the exact four prepared requests in the release-schema order."""

    request_root = exchange_root / "structure" / "requests" / reference_container
    available = {path.stem: path for path in sorted(request_root.glob("*.json"))}
    missing = sorted(set(supported_model_ids).difference(available))
    unexpected = sorted(set(available).difference(supported_model_ids))
    if missing or unexpected:
        raise FileNotFoundError(
            "ESMFold2 prepared-request inventory differs from the release schema: "
            f"missing={missing}, unexpected={unexpected}."
        )
    paths = tuple(available[model_id] for model_id in supported_model_ids)
    for model_id, path in zip(supported_model_ids, paths, strict=True):
        request = load_request(path)
        if request["model_id"] != model_id:
            raise ValueError(f"ESMFold2 request filename and model ID differ: {path}")
    return paths


def _default_output(
    exchange_root: Path,
    model_id: str,
    *,
    producer: Literal["reference", "candidate"],
    precision: str | None = None,
) -> Path:
    path = exchange_root / "structure" / "results" / producer / model_id
    return path if precision is None else path / precision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--exchange-root", type=Path, required=True)
    prepare.add_argument("--model-id", action="append", choices=supported_model_ids)

    for command in ("produce-reference", "produce-candidate"):
        produce = subparsers.add_parser(command)
        produce.add_argument("--exchange-root", type=Path, required=True)
        selection = produce.add_mutually_exclusive_group(required=True)
        selection.add_argument("--model-id", choices=supported_model_ids)
        selection.add_argument(
            "--all",
            action="store_true",
            help="Produce every request in the validated release inventory.",
        )
        produce.add_argument("--request", type=Path)
        produce.add_argument("--output", type=Path)
        if command == "produce-candidate":
            produce.add_argument("--precision", choices=("bf16", "fp8"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run request preparation or one isolated bundle producer."""

    args = _parser().parse_args(argv)
    if args.command == "prepare":
        model_ids = tuple(args.model_id or supported_model_ids)
        for path in prepare_requests(args.exchange_root, model_ids=model_ids):
            print(path)
        return 0

    if args.all:
        if args.request is not None or args.output is not None:
            raise ValueError("--request and --output require a single --model-id")
        request_paths = _all_prepared_requests(args.exchange_root)
    else:
        request_paths = (args.request or _default_request(args.exchange_root, args.model_id),)

    for request_path in request_paths:
        request = load_request(request_path)
        model_id = request["model_id"]
        if args.command == "produce-reference":
            output_dir = args.output or _default_output(
                args.exchange_root,
                model_id,
                producer="reference",
            )
            produce_reference(request_path, output_dir)
        else:
            output_dir = args.output or _default_output(
                args.exchange_root,
                model_id,
                producer="candidate",
                precision=args.precision,
            )
            produce_candidate(
                request_path,
                output_dir,
                precision=args.precision,
            )
        print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "fold_sampling_steps",
    "fold_seed",
    "fold_sequence",
    "load_bundle",
    "load_request",
    "main",
    "prepare_requests",
    "produce_candidate",
    "produce_reference",
    "reference_container",
    "schema_version",
    "supported_model_ids",
    "tensor_set_sha256",
    "tensor_sha256",
    "write_bundle",
]
