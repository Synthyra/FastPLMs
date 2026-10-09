"""Compact exact checkpoint contracts and the digest and file helpers every isolated structure oracle shares.

docker/Dockerfile copies this file and the three bundle producers beside it into the oracle image, where FastPLMs is
absent, so none of them may import FastPLMs or any other tools module. ``tools/tensor_digests.py`` holds the same
tensor digests for host code, and ``tests/unit/test_tensor_digests.py`` fails when the two differ.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import torch

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal


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


def _tensor_bytes(X: torch.Tensor) -> bytes:
    # X: (...) the named state entry's shape; flattened only for byte hashing.
    value = X.detach().to(device="cpu").contiguous().reshape(-1)  # (X.numel(),)
    return value.view(torch.uint8).numpy().tobytes()


def tensor_sha256(X: torch.Tensor) -> str:
    """Hash one tensor exactly, including scalar tensors."""

    # X: (...)
    return hashlib.sha256(_tensor_bytes(X)).hexdigest()


def tensor_set_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and values in stable name order."""

    # tensors: (...) one tensor per name, any shape
    digest = hashlib.sha256()
    for name in sorted(tensors):
        X = tensors[name]  # (...)
        digest.update(name.encode("utf-8"))
        digest.update(str(X.dtype).encode("ascii"))
        digest.update(repr(tuple(X.shape)).encode("ascii"))
        digest.update(_tensor_bytes(X))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file's bytes, read in blocks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stored_json_text(value: Mapping[str, Any]) -> str:
    """Return the two-space-indented, key-sorted JSON form of a stored bundle file, with one trailing newline."""

    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def request_fingerprint(request: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a request in compact, key-sorted JSON."""

    payload = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    """Replace ``path`` with ``content`` in one step, creating its parent directories."""

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


def load_request_object(path: Path, label: str) -> dict[str, Any]:
    """Read the JSON object a request file holds; ``label`` names the model in the error of any other JSON value."""

    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise TypeError(f"{label} request must be a JSON object: {path}")
    return request


def result_directory(
    exchange_root: Path,
    model_id: str,
    *,
    producer: Literal["reference", "candidate"],
    precision: str | None = None,
) -> Path:
    """Return where ``producer`` writes the bundle of ``model_id`` under the exchange root, below ``precision`` if given."""

    path = exchange_root / "structure" / "results" / producer / model_id
    return path if precision is None else path / precision


def checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    """Return the repository, revision, and file digests a request records for one checkpoint."""

    return {
        "repo_id": checkpoint.repo_id,
        "revision": checkpoint.revision,
        "files": [
            {"path": item.path, "algorithm": item.algorithm, "digest": item.digest}
            for item in checkpoint.files
        ],
    }


def upstream_metadata(upstream: Any) -> dict[str, Any]:
    """Return the pinned upstream record a request carries."""

    return {
        "id": upstream.id,
        "path": upstream.path,
        "url": upstream.url,
        "revision": upstream.revision,
        "license_expression": upstream.license_expression,
    }


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
    "atomic_write_text",
    "checkpoint_metadata",
    "exact_state_contract",
    "file_sha256",
    "load_request_object",
    "request_fingerprint",
    "result_directory",
    "semantic_config_contract",
    "stored_json_text",
    "tensor_set_sha256",
    "tensor_sha256",
    "upstream_metadata",
    "validate_exact_state_contract",
    "validate_semantic_config_contract",
]
