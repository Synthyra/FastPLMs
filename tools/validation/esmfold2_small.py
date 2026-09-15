"""Run and compare the small, isolated ESMFold2-300 parity case.

The ``produce`` command runs in one Docker service and writes a portable
tensor bundle.  Reference and candidate services must be separate processes:
the reference path imports pinned Transformers and native ESMC, while the
candidate path imports FastPLMs and ESM++.  The ``compare`` command consumes
the two bundles and emits a JSON report without importing either model.
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
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


MODEL_ID = "esmfold2-300"
SEQUENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
SEED = 17
NUM_LOOPS = 3
NUM_SAMPLES = 1
NUM_SAMPLING_STEPS = 15
CHUNK_SIZE = 32
SCHEMA_VERSION = 1

_CORE_FEATURES = (
    "token_index",
    "residue_index",
    "asym_id",
    "sym_id",
    "entity_id",
    "mol_type",
    "res_type",
    "token_bonds",
    "token_attention_mask",
    "ref_pos",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_attention_mask",
    "atom_to_token",
    "distogram_atom_idx",
    "input_ids",
)
_CONFIDENCE_NAMES = {"plddt", "plddt_logits", "pae", "pae_logits", "ptm", "iptm", "pde_logits"}
_ALLOWED_CANDIDATE_OUTPUTS = {
    "output__last_hidden_state",
    "output__representative_atom_coords",
}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    # value: (...)
    value = tensor.detach().cpu().contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash one tensor with its dtype, shape, and raw values."""

    value = tensor.detach().cpu().contiguous()  # (...)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _state_sha256(state: Mapping[str, torch.Tensor], *, include_names: bool) -> str:
    digest = hashlib.sha256()
    if include_names:
        for name in sorted(state):
            value = state[name].detach().cpu().contiguous()  # (...)
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(repr(tuple(value.shape)).encode("ascii"))
            digest.update(_tensor_bytes(value))
        return digest.hexdigest()

    # A producer-specific key namespace must not change the canonical value
    # fingerprint. Retain only one small digest per tensor, rather than every
    # checkpoint tensor, while the model remains resident on the GPU.
    tensor_digests: list[bytes] = []
    for tensor in state.values():
        value = tensor.detach().cpu().contiguous()  # (...)
        tensor_digest = hashlib.sha256()
        tensor_digest.update(str(value.dtype).encode("ascii"))
        tensor_digest.update(repr(tuple(value.shape)).encode("ascii"))
        tensor_digest.update(_tensor_bytes(value))
        tensor_digests.append(tensor_digest.digest())
    for tensor_digest in sorted(tensor_digests):
        digest.update(tensor_digest)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _snapshot_files(path: Path) -> list[dict[str, Any]]:
    if not path.is_dir():
        raise FileNotFoundError(f"Snapshot directory does not exist: {path}")
    files: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            files.append(
                {"path": item.relative_to(path).as_posix(), "size_bytes": item.stat().st_size}
            )
    if not files:
        raise ValueError(f"Snapshot directory is empty: {path}")
    return files


def _environment_metadata() -> dict[str, Any]:
    import transformers

    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": properties.name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_total_memory": int(properties.total_memory),
        "peak_memory_allocated": int(torch.cuda.max_memory_allocated()),
    }


@contextmanager
def _deterministic_cuda() -> Any:
    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            deterministic,
        ) = previous
        torch.use_deterministic_algorithms(deterministic)


def _finish_model_load(model: torch.nn.Module) -> torch.nn.Module:
    confidence_config = getattr(model.config, "confidence_head", None)
    confidence_enabled = (
        confidence_config.get("enabled")
        if isinstance(confidence_config, Mapping)
        else getattr(confidence_config, "enabled", None)
    )
    if confidence_enabled is not False:
        raise RuntimeError("ESMFold2-300 validation requires the confidence head to be disabled.")
    model.set_chunk_size(CHUNK_SIZE)
    if hasattr(model, "set_kernel_backend"):
        model.set_kernel_backend(None)
    return model


def _validate_backbone_keys(snapshot: Path, model: torch.nn.Module) -> None:
    """Fail closed when a local backbone loader drops or invents weights."""

    checkpoint_path = snapshot / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Backbone snapshot lacks model.safetensors: {snapshot}")
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        snapshot_keys = set(map(_canonical_backbone_name, handle.keys()))
    backbone = getattr(model, "_esmc", None)
    if backbone is None:
        raise RuntimeError("Backbone loader did not retain an ESMC module.")
    loaded_keys = {_canonical_backbone_name(name) for name in backbone.state_dict()}
    missing = sorted(snapshot_keys - loaded_keys)
    unexpected = sorted(loaded_keys - snapshot_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Backbone state-key validation failed: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )


def _canonical_backbone_name(name: str) -> str:
    """Remove the candidate adapter prefix from a backbone state key."""

    return name.removeprefix("model.")


def _is_encoder_backbone_name(name: str) -> bool:
    return not _canonical_backbone_name(name).startswith("sequence_head.")


def _load_reference_model(fold_snapshot: Path, backbone_snapshot: Path) -> torch.nn.Module:
    """Load the pinned native Transformers model without candidate imports."""

    device = torch.device("cuda")
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2_experimental import (
        ESMFold2ExperimentalModel,
    )

    config = ESMFold2Config.from_pretrained(str(fold_snapshot), local_files_only=True)
    config.attn_implementation = "sdpa"
    model = ESMFold2ExperimentalModel.from_pretrained(
        str(fold_snapshot),
        config=config,
        load_esmc=False,
        dtype=torch.float32,
        local_files_only=True,
    )
    model = model.to(device=device, dtype=torch.float32).eval()
    model.load_esmc(str(backbone_snapshot))
    _validate_backbone_keys(backbone_snapshot, model)
    return _finish_model_load(model)


def _load_candidate_model(
    fold_snapshot: Path,
    backbone_snapshot: Path,
    *,
    artifact: bool = False,
) -> torch.nn.Module:
    """Load the FastPLMs model and its local ESM++ backbone."""

    device = torch.device("cuda")
    if artifact:
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(
            str(fold_snapshot),
            local_files_only=True,
            trust_remote_code=True,
        )
        model_loader = AutoModel
    else:
        from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
        from fastplms.models.esmfold2.modeling_esmfold2_experimental import (
            ESMFold2ExperimentalModel,
        )

        config = ESMFold2Config.from_pretrained(str(fold_snapshot), local_files_only=True)
        model_loader = ESMFold2ExperimentalModel
    config.attn_implementation = "sdpa"
    config.esmc_attn_backend = "sdpa"
    load_kwargs: dict[str, Any] = {
        "config": config,
        "load_esmc": False,
        "dtype": torch.float32,
        "local_files_only": True,
    }
    if artifact:
        load_kwargs["trust_remote_code"] = True
    model = model_loader.from_pretrained(str(fold_snapshot), **load_kwargs)
    model = model.to(device=device, dtype=torch.float32).eval()
    model.load_esmc(
        str(backbone_snapshot),
        precision="bf16",
        device=device,
        local_files_only=True,
    )
    _validate_backbone_keys(backbone_snapshot, model)
    return _finish_model_load(model)


def _load_model(
    producer: Literal["reference", "candidate"],
    fold_snapshot: Path,
    backbone_snapshot: Path,
    *,
    artifact: bool = False,
) -> torch.nn.Module:
    if producer == "reference":
        return _load_reference_model(fold_snapshot, backbone_snapshot)
    return _load_candidate_model(fold_snapshot, backbone_snapshot, artifact=artifact)


def _feature_module_for_model(model: torch.nn.Module) -> Any:
    """Find protein features through an artifact wrapper's concrete base class."""

    attempted: list[str] = []
    for runtime_class in type(model).__mro__:
        module_name = runtime_class.__module__
        if "." not in module_name:
            continue
        package = module_name.rsplit(".", maxsplit=1)[0]
        feature_module_name = f"{package}.protein_utils"
        attempted.append(feature_module_name)
        try:
            return importlib.import_module(feature_module_name)
        except ModuleNotFoundError as error:
            if error.name is None or not feature_module_name.startswith(error.name):
                raise
    raise RuntimeError(
        f"Could not locate ESMFold2 protein features through the model MRO; attempted {attempted}."
    )


def _prepare_features(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    feature_module = _feature_module_for_model(model)
    features = feature_module.prepare_protein_features(SEQUENCE)
    if not isinstance(features, Mapping) or not all(
        torch.is_tensor(value) for value in features.values()
    ):
        raise TypeError("Protein feature preparation must return a tensor mapping.")
    return {name: value.to(device="cuda") for name, value in features.items()}


def _capture_initial_noise(
    model: torch.nn.Module,
    forward_features: Mapping[str, torch.Tensor],
    hidden_states: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    atom_count = int(forward_features["atom_attention_mask"].shape[-1])
    expected_shape = (1, atom_count, 3)
    noise: list[torch.Tensor] = []
    original_randn = torch.randn

    def recording_randn(*args: Any, **kwargs: Any) -> torch.Tensor:
        value = original_randn(*args, **kwargs)  # (b, a, 3) for initial diffusion noise
        if not noise and tuple(value.shape) == expected_shape:
            noise.append(value.detach().cpu().contiguous().clone())
        return value

    projection: list[torch.Tensor] = []  # each tensor: (b, l, d_pair)

    def capture_projection(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        if len(args) != 1 or not torch.is_tensor(args[0]):
            raise RuntimeError("base_z_mlp received an unexpected input.")
        projection.append(args[0].detach().cpu().contiguous().clone())

    hook = model.language_model.base_z_mlp.register_forward_pre_hook(capture_projection)
    torch.randn = recording_randn
    try:
        with _deterministic_cuda(), torch.inference_mode():
            output = model(
                **forward_features,
                lm_hidden_states=hidden_states,
                num_loops=NUM_LOOPS,
                num_diffusion_samples=NUM_SAMPLES,
                num_sampling_steps=NUM_SAMPLING_STEPS,
                seed=SEED,
                calculate_confidence=False,
            )
    finally:
        torch.randn = original_randn
        hook.remove()
    if len(noise) != 1:
        raise RuntimeError(f"Expected one initial diffusion noise tensor, captured {len(noise)}.")
    if len(projection) != 1:
        raise RuntimeError(f"Expected one base_z_mlp call, captured {len(projection)}.")
    if not isinstance(output, Mapping):
        output = output.to_dict() if hasattr(output, "to_dict") else vars(output)
    outputs = {
        f"output__{name}": value.detach().cpu().contiguous().clone()
        for name, value in output.items()
        if torch.is_tensor(value)
    }
    if any(name.removeprefix("output__") in _CONFIDENCE_NAMES for name in outputs):
        raise RuntimeError("Confidence tensors were emitted by a confidence-disabled model.")
    return outputs, noise[0], projection[0]


def _produce(
    producer: Literal["reference", "candidate"],
    fold_snapshot: Path,
    backbone_snapshot: Path,
    output: Path,
    *,
    artifact: bool = False,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("ESMFold2-300 validation requires CUDA.")
    torch.cuda.reset_peak_memory_stats()
    if artifact and producer != "candidate":
        raise ValueError("--artifact is supported only for the candidate producer.")
    model = _load_model(producer, fold_snapshot, backbone_snapshot, artifact=artifact)
    try:
        features = _prepare_features(model)
        forward_features = {name: features[name] for name in _CORE_FEATURES}
        hidden_states = model._compute_lm_hidden_states(
            features["input_ids"],
            features["asym_id"],
            features["residue_index"],
            features["mol_type"],
            features["token_attention_mask"],
        )
        hidden_cpu = hidden_states.detach().cpu().contiguous().clone()  # (b, l, n_states, d_lm)
        outputs, noise, projection = _capture_initial_noise(model, forward_features, hidden_states)
        tensors = {
            **{
                f"feature__{name}": value.detach().cpu().contiguous().clone()
                for name, value in features.items()
            },
            "hidden__lm": hidden_cpu,
            "projection__base_z_mlp_input": projection,
            "noise__initial_standard_normal": noise,
            **outputs,
        }
        fold_state = {
            name: value
            for name, value in model.state_dict().items()
            if not name.startswith("_esmc.")
        }
        backbone = getattr(model, "_esmc", None)
        if backbone is None:
            raise RuntimeError("Model did not retain its loaded ESMC backbone.")
        backbone_state = {
            name: value
            for name, value in backbone.state_dict().items()
            if _is_encoder_backbone_name(name)
        }
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "producer": producer,
            "model_id": MODEL_ID,
            "sequence": SEQUENCE,
            "seed": SEED,
            "num_loops": NUM_LOOPS,
            "num_diffusion_samples": NUM_SAMPLES,
            "num_sampling_steps": NUM_SAMPLING_STEPS,
            "chunk_size": CHUNK_SIZE,
            "attention_backend": "sdpa",
            "folding_parameter_dtype": "float32",
            "backbone_compute_dtype": "bfloat16",
            "confidence_head_enabled": False,
            "fold_snapshot": str(fold_snapshot.resolve()),
            "backbone_snapshot": str(backbone_snapshot.resolve()),
            "source_files": {
                "fold": _snapshot_files(fold_snapshot),
                "backbone": _snapshot_files(backbone_snapshot),
            },
            "config": _jsonable(model.config.to_dict()),
            "state_identity": {
                "fold": {
                    "tensor_count": len(fold_state),
                    "sha256": _state_sha256(fold_state, include_names=True),
                },
                "backbone": {
                    "tensor_count": len(backbone_state),
                    "sha256": _state_sha256(backbone_state, include_names=False),
                },
            },
            "environment": _environment_metadata(),
        }
        output.mkdir(parents=True, exist_ok=True)
        normalized = {
            name: tensor.detach().cpu().contiguous() for name, tensor in sorted(tensors.items())
        }
        metadata["tensor_keys"] = sorted(normalized)
        metadata["tensor_hashes"] = {
            name: tensor_sha256(value) for name, value in normalized.items()
        }
        handle, temporary_name = tempfile.mkstemp(
            dir=output, prefix=".bundle.", suffix=".safetensors.tmp"
        )
        os.close(handle)
        try:
            save_file(normalized, temporary_name)
            os.replace(temporary_name, output / "bundle.safetensors")
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        _write_json(output / "metadata.json", metadata)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _load_bundle(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported ESMFold2-300 bundle schema: {path}")
    tensors = load_file(str(path / "bundle.safetensors"), device="cpu")
    if sorted(tensors) != metadata.get("tensor_keys"):
        raise ValueError(f"Tensor keys differ from metadata: {path}")
    observed = {name: tensor_sha256(value) for name, value in tensors.items()}
    if observed != metadata.get("tensor_hashes"):
        raise ValueError(f"Tensor hashes differ from metadata: {path}")
    return tensors, metadata


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise ValueError(f"Tensor shapes differ: {tuple(actual.shape)} != {tuple(expected.shape)}")
    difference = actual.float() - expected.float()  # (...)
    denominator = expected.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "relative_l2": float(difference.norm() / denominator),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
    }


def _esmc_metric(
    actual: torch.Tensor,
    expected: torch.Tensor,
    residue_mask: torch.Tensor,
) -> dict[str, float]:
    if actual.shape != expected.shape or actual.ndim < 3:
        raise ValueError("ESMC tensors must have equal shape and at least three dimensions")
    if residue_mask.shape != actual.shape[:2]:
        raise ValueError("ESMC residue mask does not match tensor batch and residue axes")
    actual_residues = actual.float().reshape(actual.shape[0], actual.shape[1], -1)[
        residue_mask
    ]  # (n, d)
    expected_residues = expected.float().reshape(expected.shape[0], expected.shape[1], -1)[
        residue_mask
    ]  # (n, d)
    difference = actual_residues - expected_residues  # (n, d)
    denominator = expected_residues.norm().clamp_min(torch.finfo(torch.float32).tiny)
    reference_q999 = torch.quantile(expected_residues.abs().reshape(-1), 0.999)
    residue_cosines = torch.nn.functional.cosine_similarity(
        actual_residues, expected_residues, dim=-1
    )
    mask = residue_mask.unsqueeze(-1)
    actual_full = actual.float().reshape(actual.shape[0], actual.shape[1], -1)  # (b, l, d)
    expected_full = expected.float().reshape(expected.shape[0], expected.shape[1], -1)  # (b, l, d)
    count = mask.sum(dim=1).clamp_min(1)
    actual_pooled = torch.where(mask, actual_full, 0.0).sum(dim=1) / count  # (b, d)
    expected_pooled = torch.where(mask, expected_full, 0.0).sum(dim=1) / count  # (b, d)
    pooled_cosines = torch.nn.functional.cosine_similarity(actual_pooled, expected_pooled, dim=-1)
    return {
        "relative_l2": float(difference.norm() / denominator),
        "relative_q999": float(
            difference.abs().quantile(0.999)
            / reference_q999.clamp_min(torch.finfo(torch.float32).tiny)
        ),
        "residue_cosine_p01": float(torch.quantile(residue_cosines, 0.01)),
        "pooled_cosine_min": float(pooled_cosines.min()),
    }


def _ca_coordinates(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    names = tensors["feature__ref_atom_name_chars"][0]
    atom_mask = tensors["feature__atom_attention_mask"][0].bool()
    ca_mask = names.eq(torch.tensor([35, 33, 0, 0])).all(dim=-1) & atom_mask
    coords = tensors["output__sample_atom_coords"].float()
    if coords.ndim == 4:
        coords = coords[0, 0]
    elif coords.ndim == 3:
        coords = coords[0]
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"Unexpected sample_atom_coords shape: {tuple(coords.shape)}")
    selected = coords[ca_mask]
    if selected.shape != (len(SEQUENCE), 3):
        raise ValueError(
            f"Expected {len(SEQUENCE)} C-alpha coordinates, got {tuple(selected.shape)}"
        )
    if not torch.isfinite(selected).all():
        raise ValueError("C-alpha coordinates contain NaN or infinity")
    return selected


def _aligned_ca_rmsd(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[-1] != 3:
        raise ValueError("C-alpha coordinate shapes differ")
    actual_centered = actual - actual.mean(dim=0, keepdim=True)
    expected_centered = expected - expected.mean(dim=0, keepdim=True)
    covariance = actual_centered.T @ expected_centered
    left, _, right = torch.linalg.svd(covariance)
    correction = torch.eye(3)
    correction[-1, -1] = torch.sign(torch.det(left @ right))
    rotation = left @ correction @ right
    aligned = actual_centered @ rotation
    value = torch.sqrt(torch.mean(torch.sum((aligned - expected_centered) ** 2, dim=-1)))
    if not torch.isfinite(value):
        raise ValueError("C-alpha RMSD is not finite")
    return float(value)


def _lddt_ca(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_distances = torch.cdist(actual, actual)
    expected_distances = torch.cdist(expected, expected)
    pair_mask = expected_distances.lt(15.0)
    pair_mask.fill_diagonal_(False)
    if not pair_mask.any():
        raise ValueError("No valid C-alpha pairs for lDDT")
    errors = (actual_distances - expected_distances).abs()
    score = torch.stack([errors.lt(limit).float() for limit in (0.5, 1.0, 2.0, 4.0)]).mean(dim=0)
    value = score[pair_mask].mean()
    if not torch.isfinite(value):
        raise ValueError("C-alpha lDDT is not finite")
    return float(value)


def _validate_candidate_outputs(tensors: Mapping[str, torch.Tensor]) -> list[str]:
    """Validate the two candidate-only output extensions when present."""

    failures: list[str] = []
    last_hidden = tensors.get("output__last_hidden_state")
    if last_hidden is not None:
        # Pair representation: (b, l, l, d_pair).
        if last_hidden.ndim != 4 or tuple(last_hidden.shape[:3]) != (
            1,
            len(SEQUENCE),
            len(SEQUENCE),
        ):
            failures.append("last_hidden_state has an invalid pair-representation shape")
        elif not torch.isfinite(last_hidden).all():
            failures.append("last_hidden_state contains NaN or infinity")
    representative = tensors.get("output__representative_atom_coords")
    if representative is not None:
        # Representative coordinates: (b, l, 3).
        if representative.ndim != 3 or tuple(representative.shape) != (1, len(SEQUENCE), 3):
            failures.append("representative_atom_coords has an invalid shape")
        elif not torch.isfinite(representative).all():
            failures.append("representative_atom_coords contains NaN or infinity")
        elif "output__sample_atom_coords" in tensors and "feature__distogram_atom_idx" in tensors:
            sample_coords = tensors["output__sample_atom_coords"].float()
            if sample_coords.ndim == 4:
                sample_coords = sample_coords[:, 0]
            indices = tensors["feature__distogram_atom_idx"].long()
            expected = torch.gather(sample_coords, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
            if not torch.equal(representative.float(), expected):
                failures.append(
                    "representative_atom_coords is not gathered from sample_atom_coords"
                )
    return failures


def _validate_bundle_geometry(tensors: Mapping[str, torch.Tensor], label: str) -> None:
    atom_mask = tensors["feature__atom_attention_mask"].bool()
    coordinates = tensors["output__sample_atom_coords"].float()
    if coordinates.ndim == 4:
        coordinates = coordinates[:, 0]
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(f"{label}: unexpected coordinate shape {tuple(coordinates.shape)}")
    if not torch.isfinite(coordinates[atom_mask]).all():
        raise ValueError(f"{label}: atom coordinates contain NaN or infinity")
    atom_pad_mask = tensors.get("output__atom_pad_mask")
    if atom_pad_mask is not None and not torch.equal(
        atom_pad_mask.bool().reshape_as(atom_mask), atom_mask
    ):
        raise ValueError(f"{label}: atom_pad_mask differs from atom_attention_mask")
    ca = _ca_coordinates(tensors)
    distances = torch.linalg.vector_norm(ca[1:] - ca[:-1], dim=-1)
    if (
        not torch.isfinite(distances).all()
        or not distances.gt(2.0).all()
        or not distances.lt(5.0).all()
    ):
        raise ValueError(f"{label}: consecutive C-alpha distances are outside (2, 5) Angstrom")
    for name, value in tensors.items():
        if (
            name.startswith("output__")
            and value.is_floating_point()
            and not torch.isfinite(value).all()
        ):
            raise ValueError(f"{label}: {name} contains NaN or infinity")


def compare_bundles(reference: Path, candidate: Path) -> dict[str, Any]:
    """Compare two bundles and return a report that always states pass/fail."""

    failures: list[str] = []
    try:
        reference_tensors, reference_metadata = _load_bundle(reference)
        candidate_tensors, candidate_metadata = _load_bundle(candidate)
    except Exception as error:
        return {"schema_version": SCHEMA_VERSION, "status": "failed", "failures": [str(error)]}
    for name, expected in (
        ("reference", "reference"),
        ("candidate", "candidate"),
    ):
        observed = (reference_metadata if name == "reference" else candidate_metadata).get(
            "producer"
        )
        if observed != expected:
            failures.append(f"{name} producer metadata is {observed!r}")
    for key, expected in {
        "model_id": MODEL_ID,
        "sequence": SEQUENCE,
        "seed": SEED,
        "num_loops": NUM_LOOPS,
        "num_diffusion_samples": NUM_SAMPLES,
        "num_sampling_steps": NUM_SAMPLING_STEPS,
        "chunk_size": CHUNK_SIZE,
        "attention_backend": "sdpa",
        "folding_parameter_dtype": "float32",
        "backbone_compute_dtype": "bfloat16",
        "confidence_head_enabled": False,
    }.items():
        if reference_metadata.get(key) != expected or candidate_metadata.get(key) != expected:
            failures.append(f"metadata field {key!r} does not match the ESMFold2-300 contract")
    for key in ("fold", "backbone"):
        left = reference_metadata.get("state_identity", {}).get(key, {})
        right = candidate_metadata.get("state_identity", {}).get(key, {})
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            failures.append(f"state identity metadata is missing for {key}")
        elif any(field not in left or field not in right for field in ("tensor_count", "sha256")):
            failures.append(f"state identity metadata is incomplete for {key}")
        elif left.get("tensor_count") != right.get("tensor_count") or left.get(
            "sha256"
        ) != right.get("sha256"):
            failures.append(f"state identity hash differs for {key}")
    missing_candidate = set(reference_tensors) - set(candidate_tensors)
    unexpected_candidate = (
        set(candidate_tensors) - set(reference_tensors) - _ALLOWED_CANDIDATE_OUTPUTS
    )
    if missing_candidate:
        failures.append(f"candidate is missing tensors: {sorted(missing_candidate)}")
    if unexpected_candidate:
        failures.append(f"candidate has unsupported tensors: {sorted(unexpected_candidate)}")
    failures.extend(_validate_candidate_outputs(candidate_tensors))
    exact_names = sorted(
        name
        for name in set(reference_tensors) | set(candidate_tensors)
        if name.startswith("feature__") or name.startswith("noise__")
    )
    exact_report: dict[str, str] = {}
    for name in exact_names:
        if name not in candidate_tensors or not torch.equal(
            reference_tensors[name], candidate_tensors[name]
        ):
            failures.append(f"exact tensor mismatch: {name}")
        else:
            exact_report[name] = "equal"
    numeric: dict[str, Any] = {}
    residue_mask = reference_tensors["feature__token_attention_mask"].bool()
    for name in ("hidden__lm", "projection__base_z_mlp_input"):
        try:
            numeric[name] = _esmc_metric(
                candidate_tensors[name], reference_tensors[name], residue_mask
            )
            if numeric[name]["relative_l2"] > 0.03:
                failures.append(f"{name} exceeds BF16 relative L2 limit")
            if numeric[name]["relative_q999"] > 0.05:
                failures.append(f"{name} exceeds BF16 relative q999 limit")
            if numeric[name]["residue_cosine_p01"] < 0.995:
                failures.append(f"{name} falls below BF16 residue cosine limit")
            if numeric[name]["pooled_cosine_min"] < 0.995:
                failures.append(f"{name} falls below BF16 pooled cosine limit")
        except (KeyError, ValueError) as error:
            failures.append(str(error))
    for name in sorted(set(reference_tensors) & set(candidate_tensors)):
        if name.startswith("output__"):
            try:
                numeric[name] = _metric(candidate_tensors[name], reference_tensors[name])
            except ValueError as error:
                failures.append(f"{name}: {error}")
    try:
        _validate_bundle_geometry(reference_tensors, "reference")
        _validate_bundle_geometry(candidate_tensors, "candidate")
        reference_ca = _ca_coordinates(reference_tensors)
        candidate_ca = _ca_coordinates(candidate_tensors)
        geometry = {
            "ca_rmsd": _aligned_ca_rmsd(candidate_ca, reference_ca),
            "lddt_ca": _lddt_ca(candidate_ca, reference_ca),
        }
        if geometry["ca_rmsd"] > 0.25:
            failures.append("C-alpha RMSD exceeds 0.25 hard limit")
        if geometry["lddt_ca"] < 0.99:
            failures.append("C-alpha lDDT is below 0.99 hard limit")
    except (KeyError, ValueError, RuntimeError) as error:
        geometry = {}
        failures.append(f"geometry check failed: {error}")
    forbidden = sorted(
        name
        for name in set(reference_tensors) | set(candidate_tensors)
        if name.removeprefix("output__") in _CONFIDENCE_NAMES
    )
    if forbidden:
        failures.append(f"confidence outputs are forbidden: {forbidden}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "exact_tensors": exact_report,
        "numeric": numeric,
        "geometry": geometry,
        "reference": str(reference),
        "candidate": str(candidate),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser("produce", help="produce one isolated model bundle")
    produce.add_argument("--producer", choices=("reference", "candidate"), required=True)
    produce.add_argument("--fold-snapshot", type=Path, required=True)
    produce.add_argument("--backbone-snapshot", type=Path, required=True)
    produce.add_argument("--output", type=Path, required=True)
    produce.add_argument(
        "--artifact",
        action="store_true",
        help="load the candidate fold through AutoModel with trust_remote_code",
    )
    compare = commands.add_parser("compare", help="compare isolated model bundles")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "produce":
        _produce(
            args.producer,
            args.fold_snapshot,
            args.backbone_snapshot,
            args.output,
            artifact=args.artifact,
        )
        return 0
    report = compare_bundles(args.reference, args.candidate)
    _write_json(args.output, report)
    print(report["status"])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
