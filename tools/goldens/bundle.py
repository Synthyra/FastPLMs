"""Safetensors official-golden bundles with strict provenance metadata.

This module does not load models or download checkpoints. The producer accepts
already generated tensors and records the immutable manifest provenance. The
validator is read-only and verifies every declared digest and tensor.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastplms.registry import ModelRegistry, ModelSpec, OfficialGolden


_SCHEMA_VERSION = 1
_HEX = frozenset("0123456789abcdef")


class GoldenError(RuntimeError):
    """Raised when an official golden cannot be produced or verified exactly."""


@dataclass(frozen=True, slots=True)
class GoldenBundleRecord:
    """Content identities returned by golden production or validation."""

    metadata_sha256: str
    tensors_sha256: str
    tensor_hashes: Mapping[str, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX)
    )


def _tensor_hash(tensor: Any) -> str:
    import torch

    if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
        raise GoldenError("Official golden entries must be strided Torch tensors.")
    # Golden tensors may have any rank; their exact shapes are metadata.
    T = tensor.detach().to(device="cpu").contiguous()  # (...)
    shape = list(T.shape)
    dtype = str(T.dtype).removeprefix("torch.")
    raw = T.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": dtype, "shape": shape}))
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _normalize_tensors(tensors: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    if not tensors:
        raise GoldenError("An official golden must contain at least one tensor.")
    normalized: dict[str, Any] = {}
    for name in sorted(tensors):
        if not isinstance(name, str) or not name:
            raise GoldenError(f"Invalid official golden tensor name: {name!r}.")
        tensor = tensors[name]  # (...)
        if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
            raise GoldenError(f"Official golden entry {name!r} is not a strided tensor.")
        normalized[name] = tensor.detach().to(device="cpu").contiguous().clone()  # (...)
    return normalized


def _validate_output_paths(metadata_path: Path, tensors_path: Path) -> None:
    if metadata_path.suffix != ".json" or tensors_path.suffix != ".safetensors":
        raise GoldenError("Official goldens require one .json and one .safetensors file.")
    if metadata_path.parent.resolve() != tensors_path.parent.resolve():
        raise GoldenError("Official golden metadata and tensors must share one directory.")
    if metadata_path.stem != tensors_path.stem:
        raise GoldenError("Official golden metadata and tensor basenames must match.")


def _environment_record(environment: Mapping[str, str]) -> dict[str, object]:
    if not environment or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in environment.items()
    ):
        raise GoldenError("Environment details must be a non-empty string mapping.")
    details = {key: environment[key] for key in sorted(environment)}
    return {
        "details": details,
        "fingerprint": hashlib.sha256(_canonical_json(details)).hexdigest(),
    }


def _source_file_record(source_files: Mapping[str, str] | None) -> dict[str, str]:
    if source_files is None:
        return {}
    result: dict[str, str] = {}
    if not isinstance(source_files, Mapping):
        raise GoldenError("Official-golden source files must be a mapping.")
    for name, digest in source_files.items():
        if not isinstance(name, str):
            raise GoldenError(f"Invalid official-golden source file record: {name!r}.")
        path = Path(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or not _is_sha256(digest)
        ):
            raise GoldenError(f"Invalid official-golden source file record: {name!r}.")
        result[name] = digest
    return dict(sorted(result.items()))


def _limitation_records(
    limitations: Sequence[Mapping[str, str]] | None,
) -> list[dict[str, str]]:
    """Validate portable, fail-closed official capability limitations."""

    if limitations is None:
        return []
    required = {
        "capability",
        "status",
        "public_method",
        "exception_type",
        "reason",
    }
    result: list[dict[str, str]] = []
    for limitation in limitations:
        if not isinstance(limitation, Mapping) or set(limitation) != required:
            raise GoldenError("Official-golden capability limitation schema is invalid.")
        record = {key: limitation[key] for key in sorted(required)}
        if (
            record["capability"] != "generation"
            or record["status"] != "official_unavailable"
            or any(not isinstance(value, str) or not value for value in record.values())
        ):
            raise GoldenError("Official-golden capability limitation is invalid.")
        result.append(record)
    capabilities = [record["capability"] for record in result]
    if len(capabilities) != len(set(capabilities)):
        raise GoldenError("Official-golden capability limitations contain duplicates.")
    return sorted(result, key=lambda record: record["capability"])


def _source_records(spec: ModelSpec, registry: ModelRegistry) -> list[dict[str, str]]:
    return [
        {
            "id": source_id,
            "revision": registry.upstreams[source_id].revision,
            "url": registry.upstreams[source_id].url,
        }
        for source_id in spec.family.upstreams
    ]


def _checkpoint_record(spec: ModelSpec) -> dict[str, object]:
    return {
        "repo_id": spec.official.repo_id,
        "revision": spec.official.revision,
        "files": {
            item.path: item.encoded
            for item in sorted(spec.official.files, key=lambda value: value.path)
        },
    }


def write_golden_bundle(
    spec: ModelSpec,
    registry: ModelRegistry,
    tensors: Mapping[str, Any],
    *,
    metadata_path: Path,
    tensors_path: Path,
    generation_command: Sequence[str],
    environment: Mapping[str, str],
    input_fingerprint: str,
    source_files: Mapping[str, str] | None = None,
    limitations: Sequence[Mapping[str, str]] | None = None,
    replace: bool = False,
) -> GoldenBundleRecord:
    """Persist supplied official outputs without performing model generation."""

    from safetensors.torch import save_file

    metadata_path = metadata_path.resolve()
    tensors_path = tensors_path.resolve()
    _validate_output_paths(metadata_path, tensors_path)
    if isinstance(generation_command, (str, bytes)) or not generation_command or any(
        not isinstance(argument, str) or not argument for argument in generation_command
    ):
        raise GoldenError("generation_command must be a non-empty string argument sequence.")
    if not _is_sha256(input_fingerprint):
        raise GoldenError("input_fingerprint must be a lowercase SHA-256 digest.")
    if not replace and (metadata_path.exists() or tensors_path.exists()):
        raise GoldenError("Official golden output already exists; pass replace=True explicitly.")

    normalized = _normalize_tensors(tensors)  # values: (...)
    tensor_metadata = {
        name: {
            "dtype": str(T.dtype).removeprefix("torch."),
            "shape": list(T.shape),
            "sha256": _tensor_hash(T),
        }
        for name, T in normalized.items()  # T: (...)
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_tensors = tensors_path.with_name(f".{tensors_path.name}.{os.getpid()}.tmp")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    try:
        save_file(
            normalized,
            temporary_tensors,
            # The JSON sidecar owns the schema. A single stable safetensors
            # metadata entry avoids implementation-dependent map ordering.
            metadata={"format": "pt"},
        )
        tensors_sha256 = _sha256_file(temporary_tensors)
        metadata = {
            "schema_version": _SCHEMA_VERSION,
            "model_id": spec.id,
            "sources": _source_records(spec, registry),
            "checkpoint": _checkpoint_record(spec),
            "environment": _environment_record(environment),
            "generation_command": list(generation_command),
            "input_fingerprint": input_fingerprint,
            "source_files": _source_file_record(source_files),
            "tensor_file": {
                "path": tensors_path.name,
                "sha256": tensors_sha256,
            },
            "tensors": tensor_metadata,
        }
        normalized_limitations = _limitation_records(limitations)
        if normalized_limitations:
            metadata["limitations"] = normalized_limitations
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        metadata_sha256 = _sha256_file(temporary_metadata)
        os.replace(temporary_tensors, tensors_path)
        os.replace(temporary_metadata, metadata_path)
    except BaseException:
        temporary_tensors.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        raise
    return GoldenBundleRecord(
        metadata_sha256=metadata_sha256,
        tensors_sha256=tensors_sha256,
        tensor_hashes={name: value["sha256"] for name, value in tensor_metadata.items()},
    )


def _read_metadata(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GoldenError(f"Unable to read official golden metadata: {path}.") from error
    if not isinstance(value, dict):
        raise GoldenError("Official golden metadata must contain a JSON object.")
    expected = {
        "schema_version",
        "model_id",
        "sources",
        "checkpoint",
        "environment",
        "generation_command",
        "input_fingerprint",
        "source_files",
        "tensor_file",
        "tensors",
    }
    observed = frozenset(value)
    if (
        observed not in {frozenset(expected), frozenset((*expected, "limitations"))}
        or value.get("schema_version") != _SCHEMA_VERSION
    ):
        raise GoldenError("Official golden metadata schema is invalid.")
    return value


def validate_golden_bundle(
    spec: ModelSpec,
    registry: ModelRegistry,
    *,
    metadata_path: Path,
    tensors_path: Path,
    declaration: OfficialGolden | None = None,
) -> GoldenBundleRecord:
    """Read and exactly validate one golden bundle without changing any file."""

    from safetensors.torch import load_file

    metadata_path = metadata_path.resolve()
    tensors_path = tensors_path.resolve()
    _validate_output_paths(metadata_path, tensors_path)
    if not metadata_path.is_file() or not tensors_path.is_file():
        raise GoldenError(f"Missing required official golden for check tier: {spec.id}.")
    metadata_sha256 = _sha256_file(metadata_path)
    tensors_sha256 = _sha256_file(tensors_path)
    if declaration is not None:
        if metadata_sha256 != declaration.metadata.digest:
            raise GoldenError(f"Official golden metadata digest mismatch for {spec.id}.")
        if tensors_sha256 != declaration.tensors.digest:
            raise GoldenError(f"Official golden tensor-file digest mismatch for {spec.id}.")

    metadata = _read_metadata(metadata_path)
    if metadata["model_id"] != spec.id:
        raise GoldenError(f"Official golden model identity mismatch for {spec.id}.")
    if metadata["sources"] != _source_records(spec, registry):
        raise GoldenError(f"Official golden source revisions mismatch for {spec.id}.")
    if metadata["checkpoint"] != _checkpoint_record(spec):
        raise GoldenError(f"Official golden checkpoint provenance mismatch for {spec.id}.")
    if "limitations" in metadata:
        try:
            limitations = _limitation_records(metadata["limitations"])
        except GoldenError as error:
            raise GoldenError(
                f"Official golden capability limitations are invalid for {spec.id}."
            ) from error
        if metadata["limitations"] != limitations:
            raise GoldenError(
                f"Official golden capability limitations are invalid for {spec.id}."
            )

    environment = metadata["environment"]
    if not isinstance(environment, dict) or set(environment) != {"details", "fingerprint"}:
        raise GoldenError(f"Official golden environment record is invalid for {spec.id}.")
    details = environment["details"]
    if not isinstance(details, dict) or not details or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in details.items()
    ):
        raise GoldenError(f"Official golden environment details are invalid for {spec.id}.")
    expected_environment = hashlib.sha256(_canonical_json(details)).hexdigest()
    if environment["fingerprint"] != expected_environment:
        raise GoldenError(f"Official golden environment fingerprint mismatch for {spec.id}.")

    command = metadata["generation_command"]
    if not isinstance(command, list) or not command or any(
        not isinstance(argument, str) or not argument for argument in command
    ):
        raise GoldenError(f"Official golden generation command is invalid for {spec.id}.")
    if not _is_sha256(metadata["input_fingerprint"]):
        raise GoldenError(f"Official golden input fingerprint is invalid for {spec.id}.")
    source_files = metadata["source_files"]
    if not isinstance(source_files, dict):
        raise GoldenError(f"Official golden source-file records are invalid for {spec.id}.")
    try:
        normalized_source_files = _source_file_record(source_files)
    except GoldenError as error:
        raise GoldenError(
            f"Official golden source-file records are invalid for {spec.id}."
        ) from error
    if normalized_source_files != source_files:
        raise GoldenError(f"Official golden source-file records are invalid for {spec.id}.")
    tensor_file = metadata["tensor_file"]
    if (
        not isinstance(tensor_file, dict)
        or set(tensor_file) != {"path", "sha256"}
        or tensor_file["path"] != tensors_path.name
        or tensor_file["sha256"] != tensors_sha256
    ):
        raise GoldenError(f"Official golden tensor-file record mismatch for {spec.id}.")

    tensor_records = metadata["tensors"]
    if not isinstance(tensor_records, dict) or not tensor_records:
        raise GoldenError(f"Official golden tensor records are invalid for {spec.id}.")
    try:
        tensors = load_file(tensors_path, device="cpu")  # values: (...)
    except Exception as error:
        raise GoldenError(f"Unable to load official golden tensors for {spec.id}.") from error
    if set(tensors) != set(tensor_records):
        raise GoldenError(f"Official golden tensor names mismatch for {spec.id}.")
    tensor_hashes: dict[str, str] = {}
    for name, T in tensors.items():
        # T: (...)
        record = tensor_records[name]
        if (
            not isinstance(record, dict)
            or set(record) != {"dtype", "shape", "sha256"}
            or record["dtype"] != str(T.dtype).removeprefix("torch.")
            or record["shape"] != list(T.shape)
            or not _is_sha256(record["sha256"])
        ):
            raise GoldenError(f"Official golden tensor metadata mismatch for {spec.id}:{name}.")
        actual_hash = _tensor_hash(T)
        if record["sha256"] != actual_hash:
            raise GoldenError(f"Official golden tensor hash mismatch for {spec.id}:{name}.")
        tensor_hashes[name] = actual_hash
    return GoldenBundleRecord(
        metadata_sha256=metadata_sha256,
        tensors_sha256=tensors_sha256,
        tensor_hashes=tensor_hashes,
    )


def _declared_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise GoldenError(f"Official golden path escapes the repository: {relative!r}.")
    return path


def require_declared_goldens(
    root: Path,
    registry: ModelRegistry,
    *,
    tier: str = "check",
) -> tuple[GoldenBundleRecord, ...]:
    """Validate manifest-declared goldens only when running the check tier."""

    if tier != "check":
        return ()
    records: list[GoldenBundleRecord] = []
    for spec in registry.values():
        declaration = spec.official_golden
        if declaration is None:
            continue
        records.append(
            validate_golden_bundle(
                spec,
                registry,
                metadata_path=_declared_path(root, declaration.metadata.path),
                tensors_path=_declared_path(root, declaration.tensors.path),
                declaration=declaration,
            )
        )
    return tuple(records)
