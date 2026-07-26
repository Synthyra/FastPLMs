"""Compact exact checkpoint contracts for isolated structure-model oracles."""

from __future__ import annotations

import hashlib
import json
import torch
from collections.abc import Callable, Mapping
from typing import Any


NameTransform = Callable[[str], tuple[str, ...]]

_PACKAGING_CONFIG_FIELDS = frozenset(
    {
        "_commit_hash",
        "_name_or_path",
        "architectures",
        "auto_map",
        "fastplms_checkpoint_hash",
        "fastplms_checkpoint_repo_id",
        "fastplms_checkpoint_revision",
        "fastplms_model_id",
        "fastplms_runtime_bundle_sha256",
        "fastplms_runtime_revision",
        "fastplms_source_tree_sha256",
        "fastplms_weights_revision",
        "dtype",
        "name_or_path",
        "torch_dtype",
        "transformers_version",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def tensor_sha256(X: torch.Tensor) -> str:
    """Hash one tensor exactly, including scalar tensors."""

    # X: (...)
    # value: (-1,)
    value = X.detach().to(device="cpu").contiguous().reshape(-1)
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _included(name: str, excluded_prefixes: tuple[str, ...]) -> bool:
    return not any(name.startswith(prefix) for prefix in excluded_prefixes)


def exact_state_contract(
    model: torch.nn.Module,
    *,
    name_transform: NameTransform | None = None,
    excluded_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return exact state tensor and parameter-alias metadata without tensor payloads."""

    transform = name_transform or (lambda name: (name,))
    tensors: dict[str, dict[str, object]] = {}
    for source_name, X in sorted(model.state_dict().items()):
        if not _included(source_name, excluded_prefixes):
            continue
        targets = transform(source_name)
        for name in targets:
            if name in tensors:
                raise RuntimeError(f"State-contract key collision for {name!r}.")
            tensors[name] = {
                "dtype": str(X.dtype).removeprefix("torch."),
                "shape": list(X.shape),
                "sha256": tensor_sha256(X),
            }
    if not tensors:
        raise RuntimeError("A structure checkpoint state contract cannot be empty.")

    by_parameter: dict[int, set[str]] = {}
    for source_name, parameter in model.named_parameters(remove_duplicate=False):
        if not _included(source_name, excluded_prefixes):
            continue
        by_parameter.setdefault(id(parameter), set()).update(transform(source_name))
    aliases = sorted(sorted(names) for names in by_parameter.values() if len(names) > 1)
    payload = {"aliases": aliases, "tensors": dict(sorted(tensors.items()))}
    return {
        **payload,
        "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
    }


def semantic_config_contract(config: object) -> dict[str, Any]:
    """Normalize a Transformers configuration after removing packaging fields."""

    if hasattr(config, "to_dict"):
        raw = config.to_dict()
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raise TypeError(f"Unsupported semantic configuration: {type(config)!r}")

    def normalize(value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items())
                if str(key) not in _PACKAGING_CONFIG_FIELDS
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, torch.dtype):
            return str(value).removeprefix("torch.")
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    normalized = normalize(raw)
    assert isinstance(normalized, dict)
    return {
        "fields": normalized,
        "sha256": hashlib.sha256(_canonical_json(normalized)).hexdigest(),
    }


def validate_exact_state_contract(contract: object) -> None:
    """Reject malformed or modified compact state metadata."""

    if not isinstance(contract, Mapping):
        raise ValueError("Structure state contract must be a mapping.")
    tensors = contract.get("tensors")
    aliases = contract.get("aliases")
    if not isinstance(tensors, Mapping) or not tensors or not isinstance(aliases, list):
        raise ValueError("Structure state contract is incomplete.")
    tensor_names: set[str] = set()
    for name, metadata in tensors.items():
        if not isinstance(name, str) or not name or not isinstance(metadata, Mapping):
            raise ValueError("Structure state tensor metadata is malformed.")
        if set(metadata) != {"dtype", "shape", "sha256"}:
            raise ValueError(f"Structure state tensor {name!r} has an invalid schema.")
        dtype = metadata["dtype"]
        shape = metadata["shape"]
        digest = metadata["sha256"]
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"Structure state tensor {name!r} has an invalid dtype.")
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
            for dimension in shape
        ):
            raise ValueError(f"Structure state tensor {name!r} has an invalid shape.")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Structure state tensor {name!r} has an invalid digest.")
        tensor_names.add(name)

    normalized_aliases: list[list[str]] = []
    for group in aliases:
        if not isinstance(group, list) or len(group) < 2:
            raise ValueError("Structure state aliases are malformed.")
        if any(not isinstance(name, str) or name not in tensor_names for name in group):
            raise ValueError("Structure state aliases name an unknown tensor.")
        normalized = sorted(set(group))
        if len(normalized) != len(group):
            raise ValueError("Structure state aliases contain duplicate names.")
        normalized_aliases.append(normalized)
    if aliases != sorted(normalized_aliases):
        raise ValueError("Structure state aliases are not canonical.")

    payload = {"aliases": aliases, "tensors": dict(sorted(tensors.items()))}
    expected = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if contract.get("sha256") != expected:
        raise ValueError("Structure state contract digest mismatch.")


def validate_semantic_config_contract(contract: object) -> None:
    """Reject malformed or modified compact semantic configuration metadata."""

    if not isinstance(contract, Mapping) or not isinstance(contract.get("fields"), Mapping):
        raise ValueError("Structure semantic configuration contract is incomplete.")
    fields = contract["fields"]

    def reject_packaging_fields(value: object) -> None:
        if isinstance(value, Mapping):
            if any(str(key) in _PACKAGING_CONFIG_FIELDS for key in value):
                raise ValueError("Structure semantic configuration contains packaging fields.")
            for item in value.values():
                reject_packaging_fields(item)
        elif isinstance(value, list):
            for item in value:
                reject_packaging_fields(item)

    reject_packaging_fields(fields)
    expected = hashlib.sha256(_canonical_json(fields)).hexdigest()
    if contract.get("sha256") != expected:
        raise ValueError("Structure semantic configuration digest mismatch.")


__all__ = [
    "exact_state_contract",
    "semantic_config_contract",
    "tensor_sha256",
    "validate_exact_state_contract",
    "validate_semantic_config_contract",
]
