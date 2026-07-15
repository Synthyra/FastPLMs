"""Produce isolated Meta ESMFold v1 bundles for structure compliance.

The reference path imports only the pinned fair-esm and OpenFold sources through
the manifest adapter. The candidate path imports FastPLMs only after command
dispatch. Both paths use the same raw sequence and normalize the public
``infer`` output into a hash-verified safetensors bundle.
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
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file

from tests.parity.support.state_transforms import transform_parameter_names
from tests.structure.support.state_contract import (
    exact_state_contract,
    semantic_config_contract,
    validate_exact_state_contract,
    validate_semantic_config_contract,
)

schema_version = 1
model_id = "esmfold"
reference_container = "reference-esmfold"
fold_sequence = "MSTNPKPQRKTKRNTNR"
fold_seed = 17
fold_recycles = 1
fold_backend = "eager"
fold_deterministic_algorithms = True
supported_precisions = ("fp32", "bf16")
Precision = Literal["fp32", "bf16"]

_required_outputs = (
    "aatype",
    "aligned_confidence_probs",
    "atom14_atom_exists",
    "atom37_atom_exists",
    "chain_index",
    "distogram_logits",
    "lm_logits",
    "mean_plddt",
    "plddt",
    "positions",
    "predicted_aligned_error",
    "ptm",
    "ptm_logits",
    "residue_index",
)
_exact_outputs = (
    "aatype",
    "atom14_atom_exists",
    "atom37_atom_exists",
    "chain_index",
    "residue_index",
)
_derived_state_buffers = frozenset(
    {
        "positional_encoding._float_tensor",
        "trunk.structure_module.atom_mask",
        "trunk.structure_module.default_frames",
        "trunk.structure_module.group_idx",
        "trunk.structure_module.lit_positions",
    }
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
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Return the exact byte digest for one tensor."""

    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


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


def _upstream_metadata(upstream: Any) -> dict[str, Any]:
    return {
        "id": upstream.id,
        "path": upstream.path,
        "url": upstream.url,
        "revision": upstream.revision,
        "license_expression": upstream.license_expression,
    }


def _oracle_asset_metadata(asset: Any) -> dict[str, Any]:
    return asdict(asset)


def prepare_request(exchange_root: Path) -> Path:
    """Write the manifest-derived ESMFold v1 reference request."""

    from fastplms.registry import get_model_registry

    registry = get_model_registry()
    spec = registry[model_id]
    if spec.family.reference_container != reference_container:
        raise RuntimeError("ESMFold reference container disagrees with models.toml.")
    expected_upstreams = ("fair-esm", "openfold")
    if not set(expected_upstreams).issubset(spec.family.upstreams):
        raise RuntimeError("ESMFold is missing its fair-esm or OpenFold provenance.")
    request = {
        "schema_version": schema_version,
        "model_id": model_id,
        "architecture": spec.family.architecture,
        "adapter": spec.family.reference_adapter,
        "reference_container": spec.family.reference_container,
        "official": _checkpoint_metadata(spec.official),
        "candidate": _checkpoint_metadata(spec.fast),
        "candidate_auto_model": spec.auto_map["AutoModel"],
        "upstreams": [_upstream_metadata(registry.upstreams[name]) for name in expected_upstreams],
        "oracle_assets": [_oracle_asset_metadata(asset) for asset in spec.oracle_assets],
        "state_transform": spec.family.state_transform,
        "sequence": fold_sequence,
        "seed": fold_seed,
        "recycles": fold_recycles,
        "attention_backend": fold_backend,
        "deterministic_algorithms": fold_deterministic_algorithms,
        "parameter_dtype": "float32",
        "compute_dtypes": list(supported_precisions),
    }
    request["request_sha256"] = _request_fingerprint(request)
    path = exchange_root / "structure" / "requests" / reference_container / f"{model_id}.json"
    _atomic_write_text(path, _canonical_json(request))
    return path


def _validate_checkpoint(source: object, *, label: str) -> None:
    if not isinstance(source, Mapping):
        raise ValueError(f"ESMFold request omits {label} checkpoint metadata.")
    revision = source.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError(f"ESMFold {label} revision is not immutable.")
    files = source.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"ESMFold {label} checkpoint has no pinned files.")


def _validate_request(request: Mapping[str, Any]) -> None:
    if request.get("schema_version") != schema_version:
        raise ValueError("Unsupported ESMFold structure-bundle schema.")
    if request.get("model_id") != model_id:
        raise ValueError(f"Unsupported ESMFold model ID: {request.get('model_id')!r}")
    if request.get("reference_container") != reference_container:
        raise ValueError("ESMFold request names the wrong reference container.")
    if request.get("attention_backend") != fold_backend:
        raise ValueError("ESMFold official parity requires the eager backend.")
    if request.get("deterministic_algorithms") is not fold_deterministic_algorithms:
        raise ValueError("ESMFold official parity requires deterministic CUDA algorithms.")
    if request.get("parameter_dtype") != "float32":
        raise ValueError("ESMFold structure parity requires canonical FP32 parameters.")
    if request.get("compute_dtypes") != list(supported_precisions):
        raise ValueError("ESMFold structure parity requires FP32 and BF16 compute gates.")
    expected = dict(request)
    observed_fingerprint = expected.pop("request_sha256", None)
    if observed_fingerprint != _request_fingerprint(expected):
        raise ValueError("ESMFold request fingerprint mismatch.")
    _validate_checkpoint(request.get("official"), label="official")
    _validate_checkpoint(request.get("candidate"), label="candidate")
    upstreams = request.get("upstreams")
    if not isinstance(upstreams, list) or {
        item.get("id") for item in upstreams if isinstance(item, Mapping)
    } != {"fair-esm", "openfold"}:
        raise ValueError("ESMFold request must pin fair-esm and OpenFold.")
    for upstream in upstreams:
        revision = upstream.get("revision")
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError("ESMFold upstream revision is not immutable.")
    assets = request.get("oracle_assets")
    if not isinstance(assets, list) or len(assets) != 1:
        raise ValueError("ESMFold request must contain its native weights asset.")
    asset = assets[0]
    if asset.get("role") != "weights" or len(str(asset.get("sha256", ""))) != 64:
        raise ValueError("ESMFold native weights asset is not hash-pinned.")


def load_request(path: Path) -> dict[str, Any]:
    """Load and validate one manifest-derived ESMFold request."""

    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise TypeError(f"ESMFold request must be a JSON object: {path}")
    _validate_request(request)
    return request


@contextmanager
def _stable_cuda_numerics():
    """Use deterministic IEEE CUDA numerics for one official or candidate fold."""

    old_algorithms = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)

    try:
        old_fp32_precision = torch.backends.fp32_precision
        old_matmul_precision = torch.backends.cuda.matmul.fp32_precision
        old_cudnn_precision = torch.backends.cudnn.fp32_precision
    except AttributeError:
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_benchmark = torch.backends.cudnn.benchmark
        old_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.backends.cudnn.benchmark = old_benchmark
            torch.backends.cudnn.deterministic = old_deterministic
            torch.use_deterministic_algorithms(old_algorithms, warn_only=old_warn_only)
        return
    old_benchmark = torch.backends.cudnn.benchmark
    old_deterministic = torch.backends.cudnn.deterministic
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.fp32_precision = old_fp32_precision
        torch.backends.cuda.matmul.fp32_precision = old_matmul_precision
        torch.backends.cudnn.fp32_precision = old_cudnn_precision
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_deterministic
        torch.use_deterministic_algorithms(old_algorithms, warn_only=old_warn_only)


def _normalize_output(output: object) -> dict[str, torch.Tensor]:
    if not isinstance(output, Mapping):
        if hasattr(output, "items"):
            output = dict(output.items())
        else:
            raise TypeError("ESMFold infer did not return a tensor mapping.")
    missing = sorted(set(_required_outputs).difference(output))
    if missing:
        raise RuntimeError(f"ESMFold infer omitted required outputs: {missing}")
    tensors: dict[str, torch.Tensor] = {}
    for name in _required_outputs:
        value = output[name]
        if not torch.is_tensor(value):
            raise TypeError(f"ESMFold output {name!r} is not a tensor.")
        tensors[f"output__{name}"] = value.detach().cpu().contiguous().clone()
    return tensors


def _run_infer(
    model: torch.nn.Module,
    request: Mapping[str, Any],
    precision: Precision,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(request["seed"]))
    torch.cuda.manual_seed_all(int(request["seed"]))
    with (
        torch.inference_mode(),
        _stable_cuda_numerics(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"),
    ):
        output = model.infer(
            request["sequence"],
            num_recycles=int(request["recycles"]),
        )
    return _normalize_output(output)


def _environment_metadata() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("transformers", "esm", "openfold"):
        try:
            module = importlib.import_module(package)
        except ImportError:
            versions[package] = None
        else:
            versions[package] = str(getattr(module, "__version__", "unknown"))
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": versions,
    }


def _load_reference_model(
    request: Mapping[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    adapter_name = request["adapter"]
    if adapter_name != "tests.parity.support.reference_adapters.esmfold":
        raise ValueError(f"Unexpected ESMFold adapter: {adapter_name!r}")
    adapter = importlib.import_module(adapter_name)
    source = request["official"]
    model, tokenizer = adapter.load_official_model(
        reference_repo_id=source["repo_id"],
        reference_revision=source["revision"],
        device=device,
        dtype=torch.float32,
        oracle_assets=request["oracle_assets"],
    )
    if tokenizer is not None:
        raise RuntimeError("Meta ESMFold public API must not return a separate tokenizer.")
    return model.eval()


def _load_candidate_model(
    request: Mapping[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    from fastplms.registry import get_model_registry

    spec = get_model_registry()[model_id]
    source = request["candidate"]
    if source["repo_id"] != spec.fast.repo_id or source["revision"] != spec.fast.revision:
        raise RuntimeError("ESMFold candidate request disagrees with models.toml.")
    auto_model = spec.auto_map["AutoModel"]
    if request["candidate_auto_model"] != auto_model:
        raise RuntimeError("ESMFold candidate AutoModel request disagrees with models.toml.")
    module_name, class_name = auto_model.rsplit(".", maxsplit=1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    model = model_class.from_pretrained(
        source["repo_id"],
        revision=source["revision"],
        dtype=torch.float32,
        attn_implementation=request["attention_backend"],
    )
    return model.eval().to(device=device)


def _esmfold_semantic_config(model: torch.nn.Module) -> dict[str, Any]:
    """Return the shared native/local architecture configuration."""

    esm = model.esm
    esm_config = getattr(esm, "config", None)
    hidden_size = getattr(esm, "embed_dim", None)
    n_layers = getattr(esm, "num_layers", None)
    n_heads = getattr(esm, "attention_heads", None)
    if esm_config is not None:
        hidden_size = hidden_size or esm_config.hidden_size
        n_layers = n_layers or esm_config.num_hidden_layers
        n_heads = n_heads or esm_config.num_attention_heads
    if hidden_size is None or n_layers is None or n_heads is None:
        raise RuntimeError("ESMFold language-model configuration is incomplete.")
    token_embedding = getattr(esm, "embed_tokens", None)
    if token_embedding is None:
        token_embedding = esm.embeddings.word_embeddings
    fields = {
        "architecture": "ESMFold",
        "distogram_bins": int(model.distogram_head.out_features),
        "esm_attention_heads": int(n_heads),
        "esm_hidden_size": int(hidden_size),
        "esm_layers": int(n_layers),
        "esm_state_count": int(model.esm_s_combine.numel()),
        "folding_blocks": len(model.trunk.blocks),
        "esm_vocab_size": int(token_embedding.num_embeddings),
        "pairwise_state_dim": int(model.distogram_head.in_features),
        "sequence_output_tokens": int(model.lm_head.out_features),
        "sequence_state_dim": int(model.lm_head.in_features),
    }
    return semantic_config_contract(fields)


def _metadata(
    request: Mapping[str, Any],
    *,
    producer: Literal["reference", "candidate"],
    model: torch.nn.Module,
    precision: Precision,
) -> dict[str, Any]:
    transform_name = str(request["state_transform"])
    if producer == "reference":

        def name_transform(name: str) -> tuple[str, ...]:
            return transform_parameter_names(transform_name, name)

    else:

        def name_transform(name: str) -> tuple[str, ...]:
            if name in _derived_state_buffers or name.startswith(
                ("mlm_head.", "esm.contact_head.")
            ):
                return ()
            return (name,)

    return {
        "schema_version": schema_version,
        "producer": producer,
        "model_id": model_id,
        "request_sha256": request["request_sha256"],
        "official": request["official"],
        "candidate": request["candidate"],
        "upstreams": request["upstreams"],
        "oracle_assets": request["oracle_assets"],
        "sequence": request["sequence"],
        "seed": request["seed"],
        "recycles": request["recycles"],
        "attention_backend": request["attention_backend"],
        "deterministic_algorithms": request["deterministic_algorithms"],
        "parameter_dtype": request["parameter_dtype"],
        "compute_dtype": precision,
        "execution": (
            "fp32_parameters_cuda_bf16_autocast" if precision == "bf16" else "fp32_parameters"
        ),
        "esm_parameter_dtypes": sorted(
            {
                str(parameter.dtype).removeprefix("torch.")
                for parameter in model.esm.parameters()
                if parameter.is_floating_point()
            }
        ),
        "state_transform": transform_name,
        "semantic_config": _esmfold_semantic_config(model),
        "state": exact_state_contract(
            model,
            name_transform=name_transform,
        ),
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
    """Atomically publish one normalized ESMFold structure bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = {
        name: tensor.detach().cpu().contiguous().clone() for name, tensor in sorted(tensors.items())
    }
    complete_metadata = dict(metadata)
    complete_metadata.update(
        {
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
    """Load one ESMFold bundle and verify all declared tensor hashes."""

    tensor_path = path / "bundle.safetensors"
    metadata_path = path / "metadata.json"
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing ESMFold structure bundle under {path}. Run native producers first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != schema_version:
        raise ValueError(f"Unsupported ESMFold bundle schema under {path}.")
    tensors = load_file(tensor_path, device="cpu")
    if sorted(tensors) != metadata.get("tensor_keys"):
        raise ValueError(f"Tensor-key mismatch in ESMFold bundle {path}.")
    observed_hashes = {name: tensor_sha256(tensor) for name, tensor in tensors.items()}
    if observed_hashes != metadata.get("tensor_hashes"):
        raise ValueError(f"Tensor hash mismatch in ESMFold bundle {path}.")
    validate_exact_state_contract(metadata.get("state"))
    validate_semantic_config_contract(metadata.get("semantic_config"))
    return tensors, metadata


def _require_fp32_parameters(model: torch.nn.Module) -> None:
    unexpected = sorted(
        {
            str(parameter.dtype)
            for parameter in model.parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        }
    )
    if unexpected:
        raise RuntimeError(f"ESMFold requires FP32 checkpoint parameters, found {unexpected}.")


def produce_reference(
    request_path: Path,
    output_dir: Path,
    *,
    precision: Precision,
) -> None:
    """Run Meta's pinned public ESMFold v1 constructor and ``infer`` API."""

    request = load_request(request_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Official ESMFold structure bundles require CUDA.")
    model = _load_reference_model(request, torch.device("cuda"))
    try:
        _require_fp32_parameters(model)
        metadata = _metadata(request, producer="reference", model=model, precision=precision)
        tensors = _run_infer(model, request, precision)
        write_bundle(
            output_dir,
            tensors,
            metadata,
        )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def produce_candidate(
    request_path: Path,
    output_dir: Path,
    *,
    precision: Precision,
) -> None:
    """Run the pinned FastPLMs ESMFold package class through public ``infer``."""

    request = load_request(request_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Candidate ESMFold structure bundles require CUDA.")
    model = _load_candidate_model(request, torch.device("cuda"))
    try:
        _require_fp32_parameters(model)
        metadata = _metadata(request, producer="candidate", model=model, precision=precision)
        tensors = _run_infer(model, request, precision)
        write_bundle(
            output_dir,
            tensors,
            metadata,
        )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _default_request(exchange_root: Path) -> Path:
    return exchange_root / "structure" / "requests" / reference_container / f"{model_id}.json"


def _default_output(
    exchange_root: Path,
    producer: Literal["reference", "candidate"],
    precision: Precision,
) -> Path:
    return exchange_root / "structure" / "results" / producer / model_id / precision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--exchange-root", type=Path, required=True)
    for name in ("produce-reference", "produce-candidate"):
        producer = subparsers.add_parser(name)
        producer.add_argument("--exchange-root", type=Path, required=True)
        producer.add_argument("--precision", choices=supported_precisions, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        print(prepare_request(arguments.exchange_root))
        return
    request_path = _default_request(arguments.exchange_root)
    precision = arguments.precision
    if arguments.command == "produce-reference":
        output = _default_output(arguments.exchange_root, "reference", precision)
        produce_reference(request_path, output, precision=precision)
    else:
        output = _default_output(arguments.exchange_root, "candidate", precision)
        produce_candidate(request_path, output, precision=precision)
    print(output)


if __name__ == "__main__":
    main()


__all__ = [
    "_exact_outputs",
    "fold_backend",
    "fold_deterministic_algorithms",
    "fold_recycles",
    "fold_seed",
    "fold_sequence",
    "load_bundle",
    "load_request",
    "model_id",
    "prepare_request",
    "produce_candidate",
    "produce_reference",
    "reference_container",
    "schema_version",
    "supported_precisions",
    "tensor_sha256",
]
