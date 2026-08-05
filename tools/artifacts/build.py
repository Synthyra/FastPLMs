"""Build deterministic, offline-loadable local model artifacts.

This tool only reads an already downloaded checkpoint snapshot. It never logs
in, downloads weights, creates a Hub repository, or uploads files.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

from fastplms import __version__
from fastplms.registry import (
    CheckpointSource,
    FileDigest,
    ModelRegistry,
    ModelSpec,
    RegistryError,
    _portable_relative_path,
    get_model_registry,
)
from tools.artifacts.license_metadata import (
    parse_hub_license_metadata,
    validate_hub_license_metadata,
)
from tools.conversion import StateTransformError, apply_state_transform
from tools.source_record import (
    ARCHIVE_PROVENANCE_NAME,
    SourceProvenanceError,
    validate_archived_root,
    validate_archived_submodule,
)


_MAX_SHARD_BYTES = 5 * 1024**3
_IGNORED_PARTS = frozenset({".cache", ".git", "__pycache__"})
_WEIGHT_SUFFIXES = frozenset({".bin", ".ckpt", ".pt", ".pth", ".safetensors"})
_WEIGHT_INDEX = "model.safetensors.index.json"
_SHARD_NAME_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
_BF16_EXECUTION_POLICIES = frozenset({"static_parameters", "fp32_parameters_autocast"})
_PROVENANCE_SCHEMA_VERSION = 4
_ARTIFACT_GENERATOR_VERSION = 4
_CANONICAL_STATE_SCHEMA_VERSION = 1
_CANONICAL_STATE_DOMAIN = b"fastplms-canonical-state-v1\0"
_CANONICAL_TENSOR_DOMAIN = b"fastplms-canonical-tensor-v1\0"
_CONVERSION_ATTESTATION_SCHEMA_VERSION = 1
_RUNTIME_ATTESTATION_SCHEMA_VERSION = 2
_RUNTIME_ATTESTATION_NAME = "runtime-attestation.json"
_MODEL_CARD_RUNTIME_REVISION_PLACEHOLDER = "<runtime-revision>"
_MODEL_CARD_RUNTIME_PROVENANCE = (
    "- Runtime revision: recorded separately in the built artifact and published commit"
)
_MODEL_CARD_DIGEST_PROVENANCE = (
    "- Runtime source identities: recorded in `source-record.json`"
)
_ARTIFACT_REQUIREMENT_INPUTS = (
    "requirements/core.in",
    "requirements/features/flash.in",
    "requirements/features/structure.in",
)
_RELEASE_TOOL_SCOPE_PATHS = (
    *_ARTIFACT_REQUIREMENT_INPUTS,
    "src/fastplms/__init__.py",
    "src/fastplms/models.toml",
    "src/fastplms/registry.py",
    "tools/artifacts/__init__.py",
    "tools/artifacts/build.py",
    "tools/artifacts/build_all.py",
    "tools/artifacts/generate_docs.py",
    "tools/artifacts/license_metadata.py",
    "tools/artifacts/offline_probe.py",
    "tools/artifacts/publish.py",
    "tools/artifacts/resolve_fair_esm_assets.py",
    "tools/artifacts/resolve_manifest_hashes.py",
    "tools/conversion/__init__.py",
    "tools/conversion/extract_esmfold2_geometry.py",
    "tools/conversion/state_transforms.py",
    "tools/conversion/state_validation.py",
    "tools/remote/biohub_reference_environment.py",
    "tools/source_record.py",
)
_RELEASE_TOOL_SCOPE_ROOTS = (
    *_ARTIFACT_REQUIREMENT_INPUTS,
    "src/fastplms/__init__.py",
    "src/fastplms/models.toml",
    "src/fastplms/registry.py",
    "tools/artifacts",
    "tools/conversion",
    "tools/remote/biohub_reference_environment.py",
    "tools/source_record.py",
)
_RELEASE_TOOL_DIGEST_DOMAIN = b"fastplms-release-tools-v1\0"
_GENERATED_RUNTIME_UPDATE_PATHS = frozenset(
    {
        "README.md",
        "config.json",
        "fastplms_bundle.py",
        "modeling_fastplms.py",
        "requirements.txt",
        "THIRD_PARTY_NOTICES.md",
        _RUNTIME_ATTESTATION_NAME,
    }
)
_RUNTIME_SOURCE_SUFFIXES = frozenset({".json", ".lock", ".py", ".toml"})
_MAX_RUNTIME_SOURCE_BYTES = 8 * 1024**2
_MAX_RUNTIME_ARCHIVE_BYTES = 128 * 1024**2
_MAX_RUNTIME_ARCHIVE_EXPANDED_BYTES = 256 * 1024**2
_MAX_RUNTIME_ARCHIVE_MEMBERS = 4096
_SENSITIVE_SOURCE_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        "credentials",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "token",
        "token.txt",
    }
)
_SENSITIVE_SOURCE_SUFFIXES = frozenset({".key", ".p12", ".pfx", ".pem"})
_SENSITIVE_SOURCE_STEMS = frozenset(
    {"credential", "credentials", "secret", "secrets", "token"}
)
_TOKENIZER_FILE_NAMES = frozenset(
    {
        "added_tokens.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
)


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be built or validated safely."""


def _update_length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _canonical_tensor_leaf(name: str, tensor: Any) -> bytes:
    """Hash one logical tensor independent of safetensors sharding."""

    try:
        import torch
    except ImportError as error:
        raise ArtifactError("Canonical state hashing requires torch.") from error
    if sys.byteorder != "little":
        raise ArtifactError("Canonical state hashing requires a little-endian host.")
    if not isinstance(name, str) or not name or not torch.is_tensor(tensor):
        raise ArtifactError(f"Canonical state contains an invalid tensor entry: {name!r}.")
    if tensor.layout != torch.strided:
        raise ArtifactError(f"Canonical state tensor {name!r} is not strided.")
    canonical = tensor.detach().to(device="cpu").contiguous()
    raw = canonical.reshape(-1).view(torch.uint8).numpy().tobytes()
    leaf = hashlib.sha256()
    leaf.update(_CANONICAL_TENSOR_DOMAIN)
    _update_length_prefixed(leaf, name.encode("utf-8"))
    _update_length_prefixed(leaf, str(canonical.dtype).removeprefix("torch.").encode("ascii"))
    _update_length_prefixed(
        leaf,
        json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"),
    )
    _update_length_prefixed(leaf, raw)
    return leaf.digest()


def _canonical_state_sha256(state: Mapping[str, Any]) -> str:
    """Return a deterministic digest of tensor names, metadata, and values."""

    if not state:
        raise ArtifactError("Canonical state cannot be empty.")
    leaves = {name: _canonical_tensor_leaf(name, state[name]) for name in sorted(state)}
    digest = hashlib.sha256()
    digest.update(_CANONICAL_STATE_DOMAIN)
    digest.update(_CANONICAL_STATE_SCHEMA_VERSION.to_bytes(4, "big"))
    digest.update(len(leaves).to_bytes(8, "big"))
    for name in sorted(leaves):
        _update_length_prefixed(digest, name.encode("utf-8"))
        _update_length_prefixed(digest, leaves[name])
    return digest.hexdigest()


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    """Return a normal SHA-256 or Git-blob SHA-1 digest for one file."""

    if algorithm == "sha256":
        digest = hashlib.sha256()
    elif algorithm == "git-sha1":
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    else:
        raise ArtifactError(f"Unsupported digest algorithm: {algorithm!r}")
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_legal_bytes(path: Path) -> bytes:
    """Return UTF-8 legal text with Git-canonical LF line endings."""

    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ArtifactError(f"Legal text must be readable UTF-8: {path}") from error
    if "\x00" in text:
        raise ArtifactError(f"Legal text contains a NUL byte: {path}")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _hash_canonical_legal_file(path: Path, algorithm: str) -> str:
    if algorithm != "sha256":
        raise ArtifactError(f"Legal texts require SHA-256, received {algorithm!r}.")
    return hashlib.sha256(_canonical_legal_bytes(path)).hexdigest()


def verify_checkpoint(snapshot: Path, source: CheckpointSource) -> None:
    """Verify every manifest-pinned file in a local checkpoint snapshot."""

    snapshot = snapshot.resolve()
    if not snapshot.is_dir():
        raise ArtifactError(f"Checkpoint snapshot does not exist: {snapshot}")
    failures: list[str] = []
    for expected in source.files:
        path = snapshot.joinpath(*PurePosixPath(expected.path).parts)
        if not path.is_file():
            failures.append(f"missing {expected.path}")
            continue
        actual = hash_file(path, expected.algorithm)
        if actual != expected.digest:
            failures.append(
                f"{expected.path}: expected {expected.encoded}, "
                f"received {expected.algorithm}:{actual}"
            )
    if failures:
        detail = "\n  - ".join(failures)
        raise ArtifactError(f"Checkpoint verification failed for {source.repo_id}:\n  - {detail}")


def _is_weight_file(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in _WEIGHT_SUFFIXES


def _is_runtime_update_path(path: str) -> bool:
    relative = PurePosixPath(path)
    return (
        path in _GENERATED_RUNTIME_UPDATE_PATHS
        or (relative.parts and relative.parts[0] in {"fastplms", "LICENSES"})
        or (len(relative.parts) == 1 and relative.name in _TOKENIZER_FILE_NAMES)
    )


def _copy_checkpoint_assets(
    snapshot: Path,
    destination: Path,
    source: CheckpointSource,
) -> None:
    """Copy only pinned, non-weight checkpoint assets into an artifact."""

    for expected in source.files:
        if _is_weight_file(expected.path):
            continue
        relative = PurePosixPath(expected.path)
        _copy_verified_checkpoint_file(
            snapshot.joinpath(*relative.parts),
            destination.joinpath(*relative.parts),
            expected,
        )


def _copy_official_tokenizer_assets(
    snapshot: Path,
    destination: Path,
    source: CheckpointSource,
) -> None:
    """Copy byte-exact tokenizer files from the pinned official snapshot."""

    selected = [
        item for item in source.files if PurePosixPath(item.path).name in _TOKENIZER_FILE_NAMES
    ]
    if not selected:
        raise ArtifactError(f"Official checkpoint {source.repo_id} declares no tokenizer files.")
    for expected in selected:
        relative = PurePosixPath(expected.path)
        source_path = snapshot.joinpath(*relative.parts)
        _copy_verified_checkpoint_file(
            source_path,
            destination.joinpath(*relative.parts),
            expected,
        )


def _validated_weight_snapshot(
    snapshot: Path,
    source: CheckpointSource,
    destination: Path,
) -> Path:
    """Copy pinned weight bytes into a private builder-owned snapshot."""

    selected = tuple(item for item in source.files if _is_weight_file(item.path))
    if not selected:
        raise ArtifactError(f"Checkpoint {source.repo_id} declares no weight files.")
    for expected in selected:
        relative = PurePosixPath(expected.path)
        _copy_verified_checkpoint_file(
            snapshot.joinpath(*relative.parts),
            destination.joinpath(*relative.parts),
            expected,
        )
    return destination


def _load_checkpoint_state(snapshot: Path, source: CheckpointSource) -> dict[str, Any]:
    """Load a builder-owned, hash-verified state without unrestricted pickle."""

    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as error:
        raise ArtifactError(
            "Checkpoint canonicalization requires the core torch and safetensors dependencies."
        ) from error

    weight_files = sorted(
        (item for item in source.files if _is_weight_file(item.path)),
        key=lambda item: item.path,
    )
    if not weight_files:
        raise ArtifactError(f"Checkpoint {source.repo_id} does not declare any weight files.")

    state: dict[str, Any] = {}
    for expected in weight_files:
        path = snapshot.joinpath(*PurePosixPath(expected.path).parts)
        suffix = path.suffix.lower()
        try:
            if suffix == ".safetensors":
                loaded: object = load_file(path, device="cpu")
            elif suffix == ".bin":
                loaded = torch.load(path, map_location="cpu", weights_only=True)
            else:
                raise ArtifactError(
                    f"Artifact conversion does not accept {suffix!r} weight files. "
                    "Convert the trusted source to safetensors or a hash-pinned .bin first."
                )
        except ArtifactError:
            raise
        except Exception as error:
            raise ArtifactError(f"Unable to load verified weight file: {path}") from error

        if isinstance(loaded, Mapping) and set(loaded) == {"state_dict"}:
            loaded = loaded["state_dict"]
        if not isinstance(loaded, Mapping):
            raise ArtifactError(f"Weight file does not contain a state dictionary: {path}")

        loaded_keys = list(loaded)
        if any(not isinstance(name, str) or not name for name in loaded_keys):
            raise ArtifactError(f"Weight file contains an invalid parameter key: {path}")
        for name in sorted(loaded_keys):
            tensor = loaded[name]
            if name in state:
                raise ArtifactError(f"Duplicate parameter {name!r} across checkpoint shards.")
            if not torch.is_tensor(tensor):
                raise ArtifactError(f"State entry {name!r} in {path.name} is not a tensor.")
            if tensor.layout != torch.strided:
                raise ArtifactError(
                    f"State entry {name!r} uses unsupported layout {tensor.layout}."
                )
            # Break shared storage deterministically because safetensors stores
            # each state key independently. Parameter alias contracts are tested
            # after model loading rather than encoded as shared file storage.
            state[name] = tensor.detach().to(device="cpu").contiguous().clone()
        del loaded
    if not state:
        raise ArtifactError(f"Checkpoint {source.repo_id} contains an empty state dictionary.")
    return state


def canonicalize_checkpoint_weights(
    snapshot: Path,
    source: CheckpointSource,
    destination: Path,
    *,
    state_transform: str = "identity",
    source_is_canonical: bool = False,
    max_shard_bytes: int = _MAX_SHARD_BYTES,
) -> dict[str, Any]:
    """Transform weights, then write deterministic safetensors shards and an index."""

    if max_shard_bytes <= 256:
        raise ArtifactError("max_shard_bytes must exceed 256 bytes.")
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise ArtifactError("Writing artifacts requires safetensors.") from error

    snapshot = snapshot.resolve()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".fastplms-validated-checkpoint-",
        dir=destination.parent,
    ) as directory:
        owned_snapshot = _validated_weight_snapshot(snapshot, source, Path(directory))
        state = _load_checkpoint_state(owned_snapshot, source)
    try:
        state = apply_state_transform(
            state_transform,
            state,
            expected_keys=state if source_is_canonical else None,
        )
    except StateTransformError as error:
        raise ArtifactError(
            f"Unable to apply declared state transform {state_transform!r} "
            f"for {source.repo_id}: {error}"
        ) from error
    state_sha256 = _canonical_state_sha256(state)
    margin = min(16 * 1024**2, max(128, max_shard_bytes // 100))
    payload_limit = max_shard_bytes - margin
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    total_size = 0
    for name in sorted(state):
        tensor = state[name]
        tensor_bytes = tensor.numel() * tensor.element_size()
        total_size += tensor_bytes
        if tensor_bytes > payload_limit:
            raise ArtifactError(
                f"Parameter {name!r} requires {tensor_bytes} bytes and cannot fit in a "
                f"{max_shard_bytes}-byte safetensors shard."
            )
        if current and current_bytes + tensor_bytes > payload_limit:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += tensor_bytes
    if current:
        groups.append(current)
    if not groups or len(groups) > 99_999:
        raise ArtifactError(f"Invalid canonical shard count: {len(groups)}")

    destination.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    shard_hashes: dict[str, str] = {}
    shard_count = len(groups)
    for index, names in enumerate(groups, start=1):
        shard_name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        shard_path = destination / shard_name
        shard_state = {name: state[name] for name in names}
        try:
            save_file(shard_state, shard_path, metadata={"format": "pt"})
        except Exception as error:
            raise ArtifactError(f"Unable to write canonical shard: {shard_path}") from error
        if shard_path.stat().st_size > max_shard_bytes:
            raise ArtifactError(f"Generated shard {shard_name} exceeds {max_shard_bytes} bytes.")
        for name in names:
            weight_map[name] = shard_name
        shard_hashes[shard_name] = f"sha256:{hash_file(shard_path)}"

    index_path = destination / _WEIGHT_INDEX
    _write_json(
        index_path,
        {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        },
    )
    validate_weight_artifact(
        destination,
        max_shard_bytes=max_shard_bytes,
        expected_state_sha256=state_sha256,
    )
    return {
        "format": "safetensors",
        "index": _WEIGHT_INDEX,
        "index_digest": f"sha256:{hash_file(index_path)}",
        "max_shard_bytes": max_shard_bytes,
        "shards": shard_hashes,
        "source_schema": "canonical" if source_is_canonical else "official",
        "state_transform": state_transform,
        "state_digest": {
            "schema_version": _CANONICAL_STATE_SCHEMA_VERSION,
            "algorithm": "sha256",
            "sha256": state_sha256,
        },
        "tensor_count": len(weight_map),
        "total_size": total_size,
    }


def validate_weight_artifact(
    path: Path,
    *,
    max_shard_bytes: int = _MAX_SHARD_BYTES,
    expected_state_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an explicit safetensors shard set and its index."""

    path = path.resolve()
    index_path = path / _WEIGHT_INDEX
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Unable to read weight index: {index_path}") from error
    if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
        raise ArtifactError("Weight index must contain exactly metadata and weight_map.")
    metadata = index["metadata"]
    weight_map = index["weight_map"]
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("total_size"), int)
        or metadata["total_size"] < 0
        or not isinstance(weight_map, dict)
        or not weight_map
        or any(not isinstance(key, str) or not key for key in weight_map)
        or any(not isinstance(value, str) for value in weight_map.values())
    ):
        raise ArtifactError("Weight index metadata or weight_map is invalid.")

    shard_names = sorted(set(weight_map.values()))
    parsed_names: list[tuple[int, int, str]] = []
    for shard_name in shard_names:
        match = _SHARD_NAME_RE.fullmatch(shard_name)
        if match is None:
            raise ArtifactError(f"Invalid explicit shard name: {shard_name!r}")
        parsed_names.append((int(match.group(1)), int(match.group(2)), shard_name))
    expected_count = len(shard_names)
    if {index for index, _, _ in parsed_names} != set(range(1, expected_count + 1)) or {
        count for _, count, _ in parsed_names
    } != {expected_count}:
        raise ArtifactError("Weight index does not describe a complete shard sequence.")

    actual_shards = {item.name for item in path.glob("*.safetensors") if item.is_file()}
    if actual_shards != set(shard_names):
        raise ArtifactError(
            "Weight index and artifact shard files differ: "
            f"index={shard_names}, files={sorted(actual_shards)}"
        )
    legacy_weights = sorted(
        item.name
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".bin", ".ckpt", ".pt", ".pth"}
    )
    if legacy_weights:
        raise ArtifactError(f"Artifact contains non-safetensors weights: {legacy_weights}")

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ArtifactError("Validating artifact weights requires safetensors.") from error
    observed_keys: set[str] = set()
    observed_total_size = 0
    tensor_leaves: dict[str, bytes] = {}
    for shard_name in shard_names:
        shard_path = path / shard_name
        if not shard_path.is_file() or shard_path.stat().st_size > max_shard_bytes:
            raise ArtifactError(
                f"Shard {shard_name} is missing or exceeds {max_shard_bytes} bytes."
            )
        expected_keys = {key for key, value in weight_map.items() if value == shard_name}
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
                if actual_keys != expected_keys:
                    raise ArtifactError(f"Shard {shard_name} keys differ from the weight index.")
                for key in sorted(actual_keys):
                    tensor = handle.get_tensor(key)
                    observed_total_size += tensor.numel() * tensor.element_size()
                    tensor_leaves[key] = _canonical_tensor_leaf(key, tensor)
                    del tensor
        except ArtifactError:
            raise
        except Exception as error:
            raise ArtifactError(f"Unable to validate safetensors shard: {shard_name}") from error
        if observed_keys.intersection(actual_keys):
            raise ArtifactError(f"Duplicate tensor keys found in shard {shard_name}.")
        observed_keys.update(actual_keys)
    if observed_keys != set(weight_map):
        raise ArtifactError("Weight index does not cover every stored tensor.")
    if observed_total_size != metadata["total_size"]:
        raise ArtifactError(
            "Weight index total_size differs from the stored tensors: "
            f"expected {metadata['total_size']}, received {observed_total_size}."
        )
    state_digest = hashlib.sha256()
    state_digest.update(_CANONICAL_STATE_DOMAIN)
    state_digest.update(_CANONICAL_STATE_SCHEMA_VERSION.to_bytes(4, "big"))
    state_digest.update(len(tensor_leaves).to_bytes(8, "big"))
    for name in sorted(tensor_leaves):
        _update_length_prefixed(state_digest, name.encode("utf-8"))
        _update_length_prefixed(state_digest, tensor_leaves[name])
    actual_state_sha256 = state_digest.hexdigest()
    if expected_state_sha256 is not None and actual_state_sha256 != expected_state_sha256:
        raise ArtifactError(
            "Canonical state digest differs from the trusted conversion commitment: "
            f"expected sha256:{expected_state_sha256}, "
            f"received sha256:{actual_state_sha256}."
        )
    return index


def _verify_expected_file(path: Path, expected: FileDigest, label: str) -> None:
    if not path.is_file():
        raise ArtifactError(f"Missing required {label}: {path}")
    actual = hash_file(path, expected.algorithm)
    if actual != expected.digest:
        raise ArtifactError(
            f"Required {label} differs from the manifest: {path}; "
            f"expected {expected.encoded}, received {expected.algorithm}:{actual}"
        )


def _verify_expected_legal_file(path: Path, expected: FileDigest, label: str) -> None:
    if not path.is_file():
        raise ArtifactError(f"Missing required {label}: {path}")
    actual = _hash_canonical_legal_file(path, expected.algorithm)
    if actual != expected.digest:
        raise ArtifactError(
            f"Required {label} differs from the manifest after canonical LF normalization: "
            f"{path}; expected {expected.encoded}, received {expected.algorithm}:{actual}"
        )


def validate_repository_legal_inventory(
    source_root: Path,
    registry: ModelRegistry,
    spec: ModelSpec | None = None,
) -> None:
    """Verify legal texts and conversion records before release packaging."""

    source_root = source_root.resolve()
    for expected in registry.legal_files:
        path = source_root.joinpath(*PurePosixPath(expected.path).parts)
        _verify_expected_legal_file(path, expected, "repository legal file")

    source_ids = spec.family.upstreams if spec is not None else tuple(registry.upstreams)
    for source_id in source_ids:
        source = registry.upstreams[source_id]
        for expected in source.license_digests:
            path = source_root / source.path
            path = path.joinpath(*PurePosixPath(expected.path).parts)
            _verify_expected_legal_file(path, expected, f"{source_id} canonical legal file")
        for expected in source.distribution_files:
            path = source_root / "LICENSES" / source_id
            path = path.joinpath(*PurePosixPath(expected.path).parts)
            _verify_expected_legal_file(path, expected, f"{source_id} distribution legal file")

    families = (spec.family,) if spec is not None else tuple(registry.families.values())
    for family in families:
        required_sections = ("Input:", "Transformation:", "Output:", "Validation:", "Limitation:")
        if (
            not family.state_transform
            or family.state_transform not in family.conversion_provenance
            or any(section not in family.conversion_provenance for section in required_sections)
        ):
            raise ArtifactError(f"Model family {family.id!r} is missing conversion provenance.")


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ArtifactError(f"Symlinks are not allowed in artifacts: {path}")
        if path.is_file() and path.suffix not in {".pyc", ".pyo"}:
            yield path


def _iter_runtime_source_files(root: Path) -> Iterable[Path]:
    """Yield only bounded runtime sources from one artifact scope."""

    if root.is_symlink():
        raise ArtifactError(f"Symlinks are not allowed in runtime sources: {root}")
    candidates = (root,) if root.is_file() else root.rglob("*")
    for path in sorted(candidates, key=lambda item: item.as_posix()):
        if path == root and root.is_dir():
            continue
        relative = path.relative_to(root) if path != root else Path(path.name)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ArtifactError(f"Symlinks are not allowed in runtime sources: {path}")
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        lowered_parts = tuple(part.lower() for part in path.parts)
        if (
            any(part in _SENSITIVE_SOURCE_NAMES for part in lowered_parts)
            or any(
                PurePosixPath(part).stem in _SENSITIVE_SOURCE_STEMS
                for part in lowered_parts
            )
            or suffix in _SENSITIVE_SOURCE_SUFFIXES
        ):
            raise ArtifactError(f"Runtime source contains a sensitive path: {path}")
        if suffix not in _RUNTIME_SOURCE_SUFFIXES:
            raise ArtifactError(
                f"Runtime source has an unapproved extension {suffix!r}: {path}"
            )
        size = path.stat().st_size
        if size > _MAX_RUNTIME_SOURCE_BYTES:
            raise ArtifactError(
                f"Runtime source exceeds {_MAX_RUNTIME_SOURCE_BYTES} bytes: {path}"
            )
        yield path


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _copy_verified_checkpoint_file(
    source: Path,
    destination: Path,
    expected: FileDigest,
) -> None:
    """Preserve one mutable source file, then validate the owned bytes."""

    if source.is_symlink():
        # Hugging Face snapshots intentionally use links into their blob cache.
        # The copied destination is a regular builder-owned file whose complete
        # content is checked below, so source-link identity is not trusted.
        if not source.is_file():
            raise ArtifactError(f"Pinned checkpoint link is unavailable: {source}")
    elif not source.is_file():
        raise ArtifactError(f"Pinned checkpoint file is missing: {source}")
    try:
        _copy_file(source, destination)
        actual = hash_file(destination, expected.algorithm)
    except OSError as error:
        raise ArtifactError(f"Unable to preserve pinned checkpoint file: {source}") from error
    if actual != expected.digest:
        destination.unlink(missing_ok=True)
        raise ArtifactError(
            "Preserved checkpoint bytes differ from the pinned source after copy: "
            f"{expected.path}; expected {expected.encoded}, "
            f"received {expected.algorithm}:{actual}."
        )


def _copy_canonical_legal_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_legal_bytes(source))


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise ArtifactError(f"Required artifact source path does not exist: {source}")
    copied = False
    for path in _iter_runtime_source_files(source):
        target = destination if source.is_file() else destination / path.relative_to(source)
        _copy_file(path, target)
        copied = True
    if not copied:
        raise ArtifactError(f"Runtime source scope contains no distributable files: {source}")


def _git_runtime_revision(
    source_root: Path,
    scopes: Iterable[Path],
    entries: Iterable[tuple[Path, PurePosixPath]],
) -> str | None:
    """Return HEAD after proving every selected repository source is tracked and clean."""

    git_metadata = source_root / ".git"
    if not (git_metadata.exists() or git_metadata.is_symlink()):
        return None
    selected_scopes = tuple(scopes)
    relative_scopes: list[str] = []
    for scope in selected_scopes:
        try:
            relative = scope.resolve().relative_to(source_root)
        except ValueError as error:
            raise ArtifactError(f"Release source escapes the repository: {scope}") from error
        relative_scopes.append(relative.as_posix())
    command_prefix = [
        "git",
        "-c",
        f"safe.directory={source_root.as_posix()}",
    ]
    try:
        status = subprocess.run(
            [
                *command_prefix,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *relative_scopes,
            ],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise ArtifactError(
                "Artifact runtime inputs must be tracked and clean; scoped Git status: "
                + status.replace("\n", "; ")
            )
        tracked = subprocess.run(
            [*command_prefix, "ls-files", "-z", "--", *relative_scopes],
            cwd=source_root,
            check=True,
            capture_output=True,
        ).stdout
        tracked_names = {
            raw.decode("utf-8") for raw in tracked.split(b"\0") if raw
        }
        selected_names = {
            path.resolve().relative_to(source_root).as_posix() for path, _ in entries
        }
        missing = sorted(selected_names.difference(tracked_names))
        if missing:
            raise ArtifactError(
                f"Artifact runtime inputs contain untracked files: {missing[:10]}"
            )
        revision = subprocess.run(
            [*command_prefix, "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
        raise ArtifactError("Unable to validate tracked artifact runtime sources.") from error
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ArtifactError(f"Git returned an invalid runtime revision: {revision!r}")
    return revision


def _release_tool_scope_digest(payloads: Mapping[str, bytes]) -> str:
    """Hash path-bound release-tool bytes without trusting filesystem metadata."""

    digest = hashlib.sha256()
    digest.update(_RELEASE_TOOL_DIGEST_DOMAIN)
    for relative_name, payload in sorted(payloads.items()):
        encoded_name = relative_name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _git_archived_regular_files(
    source_root: Path,
    revision: str,
    relative_names: Iterable[str],
    *,
    label: str,
) -> dict[str, bytes]:
    """Read an exact regular-file allowlist from one immutable Git tree."""

    names = tuple(sorted(relative_names))
    command_prefix = ["git", "-c", f"safe.directory={source_root.as_posix()}"]
    try:
        archive = subprocess.run(
            [*command_prefix, "archive", "--format=tar", revision, "--", *names],
            cwd=source_root,
            check=True,
            capture_output=True,
        ).stdout
        payloads: dict[str, bytes] = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
            for member in handle.getmembers():
                if member.isdir():
                    continue
                name = PurePosixPath(member.name).as_posix()
                if not member.isfile() or name in payloads:
                    raise ArtifactError(
                        f"Tracked {label} is not a unique regular file: {member.name}"
                    )
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise ArtifactError(f"Unable to read tracked {label}: {member.name}")
                payloads[name] = extracted.read(_MAX_RUNTIME_SOURCE_BYTES + 1)
    except ArtifactError:
        raise
    except (OSError, subprocess.CalledProcessError, tarfile.TarError) as error:
        raise ArtifactError(f"Unable to archive immutable {label} bytes.") from error
    if set(payloads) != set(names):
        raise ArtifactError(f"Tracked {label} archive differs from its exact allowlist.")
    oversized = [
        name for name, payload in payloads.items() if len(payload) > _MAX_RUNTIME_SOURCE_BYTES
    ]
    if oversized:
        raise ArtifactError(f"Tracked {label} files exceed the size limit: {oversized[:10]}")
    return payloads


def _validated_release_tool_snapshot(
    source_root: Path,
    *,
    _allow_untracked_for_tests: bool = False,
) -> tuple[str, str, dict[str, bytes]]:
    """Return immutable bytes and identity for every artifact release tool.

    A Git worktree must have an exact, clean tracked inventory. A portable
    Git-free runner must carry a validated root archive attestation. The test
    escape hatch is private and deliberately produces a content-only identity.
    """

    source_root = source_root.resolve()
    payloads: dict[str, bytes]
    if _allow_untracked_for_tests:
        payloads = {
            relative_name: source_root.joinpath(
                *PurePosixPath(relative_name).parts
            ).read_bytes()
            for relative_name in _ARTIFACT_REQUIREMENT_INPUTS
        }
        payloads["test-only-release-tool-scope"] = b"untracked test fixture"
        tool_digest = _release_tool_scope_digest(payloads)
        return f"release-tools-sha256:{tool_digest}", tool_digest, payloads
    git_metadata = source_root / ".git"
    if git_metadata.exists() or git_metadata.is_symlink():
        command_prefix = ["git", "-c", f"safe.directory={source_root.as_posix()}"]
        try:
            revision = subprocess.run(
                [*command_prefix, "rev-parse", "HEAD"],
                cwd=source_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                [
                    *command_prefix,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--",
                    *_RELEASE_TOOL_SCOPE_ROOTS,
                ],
                cwd=source_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if status:
                raise ArtifactError(
                    "Artifact release tools must be tracked and clean; scoped Git status: "
                    + status.replace("\n", "; ")
                )
            tracked = subprocess.run(
                [
                    *command_prefix,
                    "ls-files",
                    "-z",
                    "--",
                    *_RELEASE_TOOL_SCOPE_ROOTS,
                ],
                cwd=source_root,
                check=True,
                capture_output=True,
            ).stdout
        except ArtifactError:
            raise
        except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
            raise ArtifactError("Unable to validate tracked artifact release tools.") from error
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ArtifactError(f"Git returned an invalid release-tool revision: {revision!r}")
        try:
            tracked_names = {
                raw.decode("utf-8") for raw in tracked.split(b"\0") if raw
            }
        except UnicodeDecodeError as error:
            raise ArtifactError("Git returned a non-UTF-8 release-tool path.") from error
        if tracked_names != set(_RELEASE_TOOL_SCOPE_PATHS):
            missing = sorted(set(_RELEASE_TOOL_SCOPE_PATHS).difference(tracked_names))
            extra = sorted(tracked_names.difference(_RELEASE_TOOL_SCOPE_PATHS))
            raise ArtifactError(
                "Tracked artifact release-tool inventory differs from the exact allowlist; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        payloads = _git_archived_regular_files(
            source_root,
            revision,
            _RELEASE_TOOL_SCOPE_PATHS,
            label="artifact release tool",
        )
        tool_digest = _release_tool_scope_digest(payloads)
        return revision, tool_digest, payloads

    archive_marker = source_root / ARCHIVE_PROVENANCE_NAME
    if archive_marker.exists() or archive_marker.is_symlink():
        try:
            _diagnostic_head, inventory = validate_archived_root(source_root)
        except SourceProvenanceError as error:
            raise ArtifactError(
                f"Git-free release-tool attestation is invalid: {error}"
            ) from error
        scoped_names = {
            relative_name
            for relative_name in inventory
            if any(
                relative_name == scope
                or relative_name.startswith(scope.rstrip("/") + "/")
                for scope in _RELEASE_TOOL_SCOPE_ROOTS
            )
        }
        if scoped_names != set(_RELEASE_TOOL_SCOPE_PATHS):
            missing = sorted(set(_RELEASE_TOOL_SCOPE_PATHS).difference(scoped_names))
            extra = sorted(scoped_names.difference(_RELEASE_TOOL_SCOPE_PATHS))
            raise ArtifactError(
                "Attested artifact release-tool inventory differs from the exact allowlist; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        payloads = {}
        for relative_name in _RELEASE_TOOL_SCOPE_PATHS:
            path = source_root.joinpath(*PurePosixPath(relative_name).parts)
            record = inventory[relative_name]
            if path.is_symlink() or not path.is_file() or record.get("mode") not in {
                "100644",
                "100755",
            }:
                raise ArtifactError(
                    f"Attested artifact release tool is not a regular file: {path}"
                )
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise ArtifactError(f"Unable to read artifact release tool: {path}") from error
            if (
                record.get("size") != len(payload)
                or record.get("sha256") != hashlib.sha256(payload).hexdigest()
            ):
                raise ArtifactError(
                    f"Attested artifact release tool mutated during snapshot: {relative_name}"
                )
            payloads[relative_name] = payload
        tool_digest = _release_tool_scope_digest(payloads)
        return f"release-tools-sha256:{tool_digest}", tool_digest, payloads

    raise ArtifactError(
        "Artifact release tools require either a clean verifiable Git worktree "
        "or a content-attested tracked remote source archive."
    )


def _requirement_lines(payload: bytes, source_name: str) -> tuple[str, ...]:
    """Parse one direct dependency declaration used by Hub artifacts."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactError(f"Artifact dependency input is not UTF-8: {source_name}") from error

    requirements: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        requirement = raw_line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        if requirement.startswith(("-", "--")):
            raise ArtifactError(
                f"Artifact dependency input must contain direct requirements only: "
                f"{source_name}:{line_number}"
            )
        requirements.append(requirement)
    if not requirements:
        raise ArtifactError(f"Artifact dependency input is empty: {source_name}")
    return tuple(requirements)


def _artifact_requirement_paths(spec: ModelSpec) -> tuple[str, ...]:
    paths = ["requirements/core.in"]
    if spec.family.extra == "structure":
        paths.append("requirements/features/structure.in")
    if any(name.startswith("flash_attention_") for name in spec.family.attention):
        paths.append("requirements/features/flash.in")
    return tuple(paths)


def _render_artifact_requirements(
    spec: ModelSpec,
    release_tool_payloads: Mapping[str, bytes],
) -> str:
    """Render the direct dependencies shipped beside one Hub model."""

    requirements: list[str] = []
    for source_name in _artifact_requirement_paths(spec):
        try:
            payload = release_tool_payloads[source_name]
        except KeyError as error:
            raise ArtifactError(
                f"Release-tool snapshot is missing artifact dependencies: {source_name}"
            ) from error
        for requirement in _requirement_lines(payload, source_name):
            if requirement not in requirements:
                requirements.append(requirement)

    return "\n".join(
        (
            f"# Direct runtime dependencies for {spec.fast.repo_id}.",
            "# FastPLMs source is embedded in this model repository.",
            *requirements,
            "",
        )
    )


def _runtime_source_entries(
    source_root: Path,
    spec: ModelSpec,
) -> tuple[tuple[Path, PurePosixPath], ...]:
    """Map every approved runtime input to its packaged relative path."""

    package_source = source_root / "src" / "fastplms"
    entries: dict[str, Path] = {}
    for relative_name in spec.family.runtime_paths:
        target_root = PurePosixPath(relative_name)
        source = package_source.joinpath(*target_root.parts)
        selected = tuple(_iter_runtime_source_files(source))
        if not selected:
            raise ArtifactError(
                f"Runtime source scope contains no distributable files: {source}"
            )
        for path in selected:
            target = (
                target_root
                if source.is_file()
                else target_root / path.relative_to(source).as_posix()
            )
            target_name = target.as_posix()
            if target_name in entries:
                raise ArtifactError(
                    f"Runtime source scopes overlap at packaged path {target_name!r}."
                )
            entries[target_name] = path
    if any(name.startswith("flash_attention_") for name in spec.family.attention):
        kernel_source = source_root / "kernels.lock"
        if tuple(_iter_runtime_source_files(kernel_source)) != (kernel_source,):
            raise ArtifactError(f"Runtime kernel lock is missing or invalid: {kernel_source}")
        entries["kernels.lock"] = kernel_source
    return tuple(
        (entries[name], PurePosixPath(name))
        for name in sorted(entries)
    )


def _snapshot_runtime_sources(
    source_root: Path,
    entries: Iterable[tuple[Path, PurePosixPath]],
    revision: str | None,
) -> dict[str, bytes]:
    """Retain runtime bytes, using immutable Git blobs whenever available."""

    selected = tuple(entries)
    source_names = {
        source.resolve().relative_to(source_root).as_posix(): target.as_posix()
        for source, target in selected
    }
    if len(source_names) != len(selected):
        raise ArtifactError("Runtime source inputs contain duplicate tracked paths.")
    if revision is None:
        try:
            payloads = {
                target.as_posix(): source.read_bytes()
                for source, target in selected
            }
        except OSError as error:
            raise ArtifactError("Unable to snapshot runtime source inputs.") from error
    else:
        try:
            archive = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={source_root.as_posix()}",
                    "archive",
                    "--format=tar",
                    revision,
                    "--",
                    *source_names,
                ],
                cwd=source_root,
                check=True,
                capture_output=True,
            ).stdout
            archived: dict[str, bytes] = {}
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
                for member in handle.getmembers():
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ArtifactError(
                            f"Tracked runtime source is not a regular file: {member.name}"
                        )
                    extracted = handle.extractfile(member)
                    if extracted is None:
                        raise ArtifactError(
                            f"Unable to read tracked runtime blob: {member.name}"
                        )
                    archived[PurePosixPath(member.name).as_posix()] = extracted.read()
        except ArtifactError:
            raise
        except (OSError, subprocess.CalledProcessError, tarfile.TarError) as error:
            raise ArtifactError(
                f"Unable to materialize tracked runtime blobs at {revision}."
            ) from error
        if set(archived) != set(source_names):
            raise ArtifactError(
                "Tracked runtime archive differs from the validated source allowlist."
            )
        payloads = {
            target_name: archived[source_name]
            for source_name, target_name in source_names.items()
        }
    oversized = [
        name for name, payload in payloads.items() if len(payload) > _MAX_RUNTIME_SOURCE_BYTES
    ]
    if oversized:
        raise ArtifactError(f"Tracked runtime sources exceed the size limit: {oversized[:10]}")
    return payloads


def _write_runtime_snapshot(destination: Path, payloads: Mapping[str, bytes]) -> None:
    for relative_name, payload in sorted(payloads.items()):
        target = destination.joinpath(*PurePosixPath(relative_name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def _runtime_payload_tree_sha256(payloads: Mapping[str, bytes]) -> str:
    inventory = {
        name: f"sha256:{hashlib.sha256(payload).hexdigest()}"
        for name, payload in payloads.items()
    }
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _archived_runtime_payloads(
    source_root: Path,
    spec: ModelSpec,
    entries: Iterable[tuple[Path, PurePosixPath]],
) -> dict[str, bytes]:
    """Validate and snapshot runtime bytes from a content-attested Git-free archive."""

    selected = tuple(entries)
    try:
        _diagnostic_head, inventory = validate_archived_root(source_root)
    except SourceProvenanceError as error:
        raise ArtifactError(f"Git-free runtime source attestation is invalid: {error}") from error

    package_source = source_root / "src" / "fastplms"
    runtime_scopes = [
        package_source.joinpath(*PurePosixPath(relative_name).parts)
        for relative_name in spec.family.runtime_paths
    ]
    if any(name.startswith("flash_attention_") for name in spec.family.attention):
        runtime_scopes.append(source_root / "kernels.lock")

    scope_names: list[str] = []
    actual_scope_files: set[str] = set()
    for scope in runtime_scopes:
        try:
            relative_scope = scope.relative_to(source_root).as_posix()
        except ValueError as error:
            raise ArtifactError(
                f"Archived runtime scope escapes the source root: {scope}"
            ) from error
        scope_names.append(relative_scope)
        if scope.is_symlink():
            raise ArtifactError(f"Archived runtime scope is a symlink: {scope}")
        if not scope.exists():
            raise ArtifactError(f"Archived runtime scope is missing: {scope}")
        candidates = (scope,) if scope.is_file() else scope.rglob("*")
        for path in candidates:
            if path.is_symlink():
                raise ArtifactError(f"Archived runtime source is a symlink: {path}")
            if path.is_file():
                actual_scope_files.add(path.relative_to(source_root).as_posix())

    expected_scope_files = {
        relative_name
        for relative_name in inventory
        if any(
            relative_name == scope_name
            or relative_name.startswith(scope_name.rstrip("/") + "/")
            for scope_name in scope_names
        )
    }
    if actual_scope_files != expected_scope_files:
        missing = sorted(expected_scope_files.difference(actual_scope_files))
        extra = sorted(actual_scope_files.difference(expected_scope_files))
        raise ArtifactError(
            "Git-free runtime source inventory differs from the tracked archive; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    source_names: dict[str, str] = {}
    for source, target in selected:
        try:
            source_name = source.relative_to(source_root).as_posix()
        except ValueError as error:
            raise ArtifactError(
                f"Archived runtime source escapes the source root: {source}"
            ) from error
        if source_name in source_names:
            raise ArtifactError(f"Archived runtime source is selected twice: {source_name}")
        source_names[source_name] = target.as_posix()
    if set(source_names) != actual_scope_files:
        missing = sorted(actual_scope_files.difference(source_names))
        extra = sorted(set(source_names).difference(actual_scope_files))
        raise ArtifactError(
            "Approved runtime allowlist differs from the archived tracked scope; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    payloads = _snapshot_runtime_sources(source_root, selected, None)
    for source_name, target_name in source_names.items():
        record = inventory.get(source_name)
        payload = payloads[target_name]
        if (
            not isinstance(record, Mapping)
            or record.get("mode") not in {"100644", "100755"}
            or record.get("size") != len(payload)
            or record.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise ArtifactError(
                f"Archived runtime source mutated during snapshot: {source_name}"
            )
    return payloads


def _validated_runtime_snapshot(
    source_root: Path,
    registry: ModelRegistry,
    spec: ModelSpec,
    *,
    _allow_untracked_for_tests: bool = False,
) -> tuple[str, dict[str, bytes], str]:
    """Return clean, tracked runtime bytes and their immutable identities."""

    package_source = source_root / "src" / "fastplms"
    runtime_scopes = [
        package_source.joinpath(*PurePosixPath(relative_name).parts)
        for relative_name in spec.family.runtime_paths
    ]
    if any(name.startswith("flash_attention_") for name in spec.family.attention):
        runtime_scopes.append(source_root / "kernels.lock")
    entries = _runtime_source_entries(source_root, spec)
    git_revision = _git_runtime_revision(source_root, runtime_scopes, entries)
    if git_revision is None:
        archive_marker = source_root / ARCHIVE_PROVENANCE_NAME
        if archive_marker.exists() or archive_marker.is_symlink():
            payloads = _archived_runtime_payloads(source_root, spec, entries)
        elif _allow_untracked_for_tests:
            payloads = _snapshot_runtime_sources(source_root, entries, None)
        else:
            raise ArtifactError(
                "Artifact runtime sources require either a clean verifiable Git worktree "
                "or a content-attested tracked remote source archive."
            )
    else:
        payloads = _snapshot_runtime_sources(
            source_root,
            entries,
            git_revision,
        )
    _validate_attention_kernel_lock(payloads.get("kernels.lock"), registry, spec)
    source_tree_sha256 = _runtime_payload_tree_sha256(payloads)
    runtime_revision = git_revision or f"source-tree-sha256:{source_tree_sha256}"
    return runtime_revision, payloads, source_tree_sha256


def _tree_sha256(root: Path) -> str:
    """Hash a portable path/digest inventory rather than filesystem metadata."""

    inventory = {
        path.relative_to(root).as_posix(): f"sha256:{hash_file(path)}"
        for path in _iter_files(root)
    }
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_attention_kernel_lock(
    payload: bytes | None,
    registry: ModelRegistry,
    spec: ModelSpec,
) -> None:
    """Validate the snapshotted kernel lock for advertised FlashAttention backends."""

    implementations = tuple(
        name for name in spec.family.attention if name.startswith("flash_attention_")
    )
    if not implementations:
        return
    if payload is None:
        raise ArtifactError("The runtime snapshot is missing kernels.lock.")
    try:
        entries = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError("Unable to read the snapshotted kernel lock.") from error
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ArtifactError("kernels.lock must contain a list of JSON objects.")
    locked: dict[str, str] = {}
    for entry in entries:
        repo_id = entry.get("repo_id")
        revision = entry.get("sha")
        if (
            not isinstance(repo_id, str)
            or not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or repo_id in locked
        ):
            raise ArtifactError("kernels.lock contains an invalid or duplicate identity.")
        locked[repo_id] = revision
    for implementation in implementations:
        kernel = registry.attention_kernels[implementation]
        if locked.get(kernel.repository) != kernel.revision:
            raise ArtifactError(
                f"kernels.lock does not match {implementation!r}: expected "
                f"{kernel.repository}@{kernel.revision}."
            )


def _copy_attention_kernel_lock(
    source_root: Path,
    package_target: Path,
    registry: ModelRegistry,
    spec: ModelSpec,
) -> None:
    """Compatibility helper for focused tests; builds use the retained snapshot."""

    implementations = tuple(
        name for name in spec.family.attention if name.startswith("flash_attention_")
    )
    if not implementations:
        return
    source = source_root / "kernels.lock"
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise ArtifactError(f"Unable to read the repository kernel lock: {source}") from error
    _validate_attention_kernel_lock(payload, registry, spec)
    destination = package_target / "kernels.lock"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _artifact_auto_map(spec: ModelSpec) -> dict[str, str]:
    return {
        auto_class: f"modeling_fastplms.{class_path.rsplit('.', maxsplit=1)[1]}"
        for auto_class, class_path in spec.auto_map.items()
    }


def _apply_artifact_config_contract(spec: ModelSpec, config: dict[str, Any]) -> None:
    """Materialize runtime-only config invariants without rewriting source provenance."""

    if spec.family.id == "dplm2":
        config.update(
            {
                "is_decoder": False,
                "add_cross_attention": False,
                "use_cache": False,
            }
        )
    if spec.family.id == "esmfold2":
        if spec.msa_conditioning is None:
            raise ArtifactError(
                f"ESMFold2 checkpoint {spec.id!r} has no MSA-conditioning contract."
            )
        msa_encoder = config.get("msa_encoder")
        if not isinstance(msa_encoder, dict):
            raise ArtifactError("ESMFold2 config.msa_encoder must be an object.")
        config_msa_conditioning = msa_encoder.get("enabled")
        if not isinstance(config_msa_conditioning, bool):
            raise ArtifactError("ESMFold2 config.msa_encoder.enabled must be a boolean.")
        if config_msa_conditioning != spec.msa_conditioning:
            raise ArtifactError(
                f"ESMFold2 config.msa_encoder.enabled={config_msa_conditioning!r} "
                f"differs from models.toml msa_conditioning={spec.msa_conditioning!r}."
            )
        declared = config.get("msa_conditioning")
        if "msa_conditioning" in config and (
            not isinstance(declared, bool) or declared != spec.msa_conditioning
        ):
            raise ArtifactError(
                "ESMFold2 config.msa_conditioning differs from the models.toml contract."
            )
        config["msa_conditioning"] = spec.msa_conditioning


def _configure_custom_tokenizer(path: Path, spec: ModelSpec) -> None:
    """Point a manifest-declared custom tokenizer at the flat artifact bridge."""

    class_path = spec.family.tokenizer_class
    if class_path is None:
        return
    config_path = path / "tokenizer_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(
            f"Custom tokenizer {class_path!r} requires a valid tokenizer_config.json."
        ) from error
    if not isinstance(config, dict):
        raise ArtifactError("tokenizer_config.json must contain a JSON object.")
    raw_auto_map = config.get("auto_map")
    if raw_auto_map is None or isinstance(raw_auto_map, (list, tuple)):
        auto_map: dict[str, Any] = {}
    elif isinstance(raw_auto_map, dict):
        auto_map = dict(raw_auto_map)
    else:
        raise ArtifactError("tokenizer_config.json auto_map must be an object or legacy list.")
    class_name = class_path.rsplit(".", maxsplit=1)[1]
    auto_map["AutoTokenizer"] = [f"modeling_fastplms.{class_name}", None]
    config["auto_map"] = auto_map
    _write_json(config_path, config)


def _runtime_archive_files(package_root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    expanded_size = 0
    for path in _iter_files(package_root):
        relative_name = path.relative_to(package_root).as_posix()
        try:
            relative = _portable_relative_path(relative_name, "Runtime archive path")
        except RegistryError as error:
            raise ArtifactError(f"Runtime archive path is invalid: {relative_name!r}") from error
        if relative.suffix.lower() not in _RUNTIME_SOURCE_SUFFIXES:
            raise ArtifactError(
                f"Runtime archive path has an unapproved extension: {relative_name!r}"
            )
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ArtifactError(f"Unable to inspect runtime archive source: {path}") from error
        if size > _MAX_RUNTIME_SOURCE_BYTES:
            raise ArtifactError(f"Runtime archive source exceeds its size limit: {path}")
        if len(files) >= _MAX_RUNTIME_ARCHIVE_MEMBERS:
            raise ArtifactError("The artifact runtime archive contains too many source files.")
        archive_name = (PurePosixPath("fastplms") / relative).as_posix()
        try:
            with path.open("rb") as handle:
                payload = handle.read(_MAX_RUNTIME_SOURCE_BYTES + 1)
        except OSError as error:
            raise ArtifactError(f"Unable to read runtime archive source: {path}") from error
        if len(payload) > _MAX_RUNTIME_SOURCE_BYTES or len(payload) != size:
            raise ArtifactError(f"Runtime archive source changed or exceeded its limit: {path}")
        expanded_size += len(payload)
        if expanded_size > _MAX_RUNTIME_ARCHIVE_EXPANDED_BYTES:
            raise ArtifactError("The expanded artifact runtime archive exceeds its size limit.")
        files[archive_name] = payload
    if not files:
        raise ArtifactError("The artifact runtime source archive would be empty.")
    return files


def _build_runtime_archive(package_root: Path) -> bytes:
    """Return a deterministic archive of unchanged package runtime sources."""

    files = _runtime_archive_files(package_root)
    if len(files) > _MAX_RUNTIME_ARCHIVE_MEMBERS:
        raise ArtifactError("The artifact runtime archive contains too many source files.")
    if sum(map(len, files.values())) > _MAX_RUNTIME_ARCHIVE_EXPANDED_BYTES:
        raise ArtifactError("The expanded artifact runtime archive exceeds its size limit.")
    buffer = io.BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for archive_path, contents in sorted(files.items()):
            info = ZipInfo(archive_path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, contents, compress_type=ZIP_DEFLATED, compresslevel=9)
    payload = buffer.getvalue()
    if len(payload) > _MAX_RUNTIME_ARCHIVE_BYTES:
        raise ArtifactError("The compressed artifact runtime archive exceeds its size limit.")
    return payload


def _render_runtime_bundle(package_root: Path) -> tuple[str, bytes]:
    archive = _build_runtime_archive(package_root)
    archive_hash = hashlib.sha256(archive).hexdigest()
    encoded = base64.b85encode(archive).decode("ascii")
    chunks = (encoded[index : index + 100] for index in range(0, len(encoded), 100))
    lines = [
        '"""Generated deterministic archive of unchanged FastPLMs runtime sources."""',
        "",
        f'RUNTIME_HASH = "{archive_hash}"',
        "RUNTIME_DATA = (",
        *(f"    {chunk!r}" for chunk in chunks),
        ")",
        "",
    ]
    return archive_hash, "\n".join(lines).encode("utf-8")


def _write_runtime_bundle(path: Path, package_root: Path) -> str:
    """Write the flat source bundle consumed by Transformers remote code."""

    archive_hash, payload = _render_runtime_bundle(package_root)
    path.write_bytes(payload)
    return archive_hash


def _decode_runtime_bundle(path: Path) -> tuple[str, bytes]:
    """Decode a data-only generated bundle without executing artifact code."""

    try:
        source = path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise ArtifactError(f"Runtime bundle is missing or invalid: {path}") from error
    if len(module.body) != 3 or not (
        isinstance(module.body[0], ast.Expr)
        and isinstance(module.body[0].value, ast.Constant)
        and isinstance(module.body[0].value.value, str)
    ):
        raise ArtifactError("Runtime bundle must contain only its docstring and data assignments.")
    assignments: dict[str, Any] = {}
    for statement in module.body[1:]:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in {"RUNTIME_HASH", "RUNTIME_DATA"}
        ):
            raise ArtifactError("Runtime bundle contains executable or unknown statements.")
        name = statement.targets[0].id
        if name in assignments:
            raise ArtifactError(f"Runtime bundle repeats {name}.")
        try:
            assignments[name] = ast.literal_eval(statement.value)
        except (ValueError, TypeError) as error:
            raise ArtifactError(f"Runtime bundle {name} is not literal data.") from error
    runtime_hash = assignments.get("RUNTIME_HASH")
    encoded = assignments.get("RUNTIME_DATA")
    if (
        not isinstance(runtime_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", runtime_hash) is None
        or not isinstance(encoded, str)
        or not encoded
    ):
        raise ArtifactError("Runtime bundle data identities are missing or invalid.")
    try:
        archive = base64.b85decode(encoded.encode("ascii"))
    except (UnicodeEncodeError, ValueError) as error:
        raise ArtifactError("Runtime bundle contains invalid base85 archive data.") from error
    if hashlib.sha256(archive).hexdigest() != runtime_hash:
        raise ArtifactError("Runtime bundle archive bytes differ from RUNTIME_HASH.")
    if len(archive) > _MAX_RUNTIME_ARCHIVE_BYTES:
        raise ArtifactError("Runtime bundle archive exceeds its compressed size limit.")
    return runtime_hash, archive


def _validate_runtime_bundle(
    path: Path,
    package_root: Path,
    expected_hash: str,
) -> None:
    """Bind the executable bundle archive to the validated package-source bytes."""

    runtime_hash, archive = _decode_runtime_bundle(path)
    if runtime_hash != expected_hash:
        raise ArtifactError("Runtime bundle identity differs from provenance.")
    expected = _runtime_archive_files(package_root)
    try:
        with ZipFile(io.BytesIO(archive)) as bundle:
            members = bundle.infolist()
            if not members or len(members) > _MAX_RUNTIME_ARCHIVE_MEMBERS:
                raise ArtifactError("Runtime bundle archive has an invalid member count.")
            member_names = [member.filename for member in members]
            if len(member_names) != len(set(member_names)):
                raise ArtifactError("Runtime bundle archive contains duplicate paths.")
            if set(member_names) != set(expected):
                raise ArtifactError(
                    "Runtime bundle archive inventory differs from packaged runtime sources."
                )
            for member in members:
                mode = member.external_attr >> 16
                expected_payload = expected[member.filename]
                if (
                    member.is_dir()
                    or member.flag_bits & 0x1
                    or member.compress_type != ZIP_DEFLATED
                    or mode != 0o100644
                    or member.file_size != len(expected_payload)
                    or member.file_size > _MAX_RUNTIME_SOURCE_BYTES
                ):
                    raise ArtifactError(
                        f"Runtime bundle archive member is not canonical: {member.filename!r}."
                    )
                if bundle.read(member) != expected_payload:
                    raise ArtifactError(
                        f"Runtime bundle archive differs at {member.filename!r}."
                    )
    except ArtifactError:
        raise
    except (BadZipFile, KeyError, RuntimeError, OSError) as error:
        raise ArtifactError("Runtime bundle archive is invalid.") from error


def _render_bootstrap(spec: ModelSpec, runtime_hash: str) -> str:
    """Render the flat Transformers bridge to the bundled unchanged sources."""

    grouped: dict[str, list[str]] = {}
    class_paths = list(spec.auto_map.values())
    if spec.family.tokenizer_class is not None:
        class_paths.append(spec.family.tokenizer_class)
    for class_path in class_paths:
        module_name, class_name = class_path.rsplit(".", maxsplit=1)
        grouped.setdefault(module_name, []).append(class_name)
    lines = [
        '"""Generated bridge to the embedded FastPLMs runtime sources."""',
        "",
        "import base64",
        "import hashlib",
        "import importlib",
        "import importlib.util",
        "import sys",
        "import tempfile",
        "from io import BytesIO",
        "from pathlib import Path",
        "from typing import ClassVar",
        "from zipfile import ZIP_DEFLATED, ZipFile",
        "",
        "from .fastplms_bundle import RUNTIME_DATA, RUNTIME_HASH",
        "",
        f'if RUNTIME_HASH != "{runtime_hash}":',
        '    raise RuntimeError("FastPLMs runtime identity differs from the bridge.")',
        "",
        "_RUNTIME_TEMPORARIES = []",
        "",
        "def _archive_runtime_hashes(payload):",
        "    result = {}",
        "    with ZipFile(BytesIO(payload)) as archive:",
        "        for member in archive.infolist():",
        "            name = member.filename",
        "            parts = Path(name).parts",
        "            if (",
        "                member.is_dir()",
        '                or "\\\\" in name',
        "                or not parts",
        '                or parts[0] != "fastplms"',
        "                or len(parts) < 2",
        '                or any(part in {"", ".", ".."} for part in parts)',
        '                or Path(name).suffix in {".pyc", ".pyo"}',
        "                or member.flag_bits & 0x1",
        "                or member.compress_type != ZIP_DEFLATED",
        "                or member.external_attr >> 16 != 0o100644",
        "            ):",
        '                raise RuntimeError("Embedded FastPLMs archive has an unsafe path.")',
        "            relative = Path(*parts[1:]).as_posix()",
        "            if relative in result:",
        '                raise RuntimeError("Embedded FastPLMs archive repeats a path.")',
        "            result[relative] = hashlib.sha256(archive.read(member)).hexdigest()",
        "    return result",
        "",
        "def _ensure_runtime():",
        '    payload = base64.b85decode("".join(RUNTIME_DATA))',
        "    if hashlib.sha256(payload).hexdigest() != RUNTIME_HASH:",
        '        raise RuntimeError("Embedded FastPLMs runtime hash mismatch.")',
        "    expected = _archive_runtime_hashes(payload)",
        '    temporary = tempfile.TemporaryDirectory(prefix="fastplms-artifact-runtime-")',
        "    try:",
        "        runtime_root = Path(temporary.name)",
        "        with ZipFile(BytesIO(payload)) as archive:",
        "            for member in archive.infolist():",
        "                target = runtime_root.joinpath(*Path(member.filename).parts)",
        "                target.parent.mkdir(parents=True, exist_ok=True)",
        '                with target.open("xb") as handle:',
        "                    handle.write(archive.read(member))",
        '        package_root = runtime_root / "fastplms"',
        "        if _runtime_file_hashes(package_root) != expected:",
        "            raise RuntimeError(",
        '                "Private FastPLMs runtime differs from the embedded archive."',
        "            )",
        "    except BaseException:",
        "        temporary.cleanup()",
        "        raise",
        "    _RUNTIME_TEMPORARIES.append(temporary)",
        "    return package_root",
        "",
        "def _runtime_file_hashes(package_root):",
        "    result = {}",
        '    for path in sorted(package_root.rglob("*")):',
        "        relative = path.relative_to(package_root)",
        "        if path.is_symlink():",
        '            raise RuntimeError("Private FastPLMs runtime contains a symlink.")',
        "        if path.is_dir():",
        "            continue",
        '        if path.suffix in {".pyc", ".pyo"}:',
        '            raise RuntimeError("Private FastPLMs runtime contains bytecode.")',
        "        if not path.is_file():",
        '            raise RuntimeError("Private FastPLMs runtime contains a non-file entry.")',
        "        result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()",
        "    return result",
        "",
        "def _extend_loaded_package_paths(package_root):",
        "    for name, module in list(sys.modules.items()):",
        '        if name != "fastplms" and not name.startswith("fastplms."):',
        "            continue",
        '        paths = getattr(module, "__path__", None)',
        "        if paths is None:",
        "            continue",
        '        relative = name.split(".")[1:]',
        "        candidate = package_root.joinpath(*relative)",
        "        candidate_text = str(candidate)",
        "        if candidate.is_dir() and candidate_text not in paths:",
        "            paths.append(candidate_text)",
        "",
        "def _merge_runtime(package, package_root):",
        "    incoming = _runtime_file_hashes(package_root)",
        '    known = getattr(package, "__fastplms_artifact_runtime_files__", None)',
        "    if not isinstance(known, dict):",
        "        raise RuntimeError(",
        '            "A non-artifact fastplms module is already loaded. Load the Hub artifact "',
        '            "in a separate Python process."',
        "        )",
        "    conflicts = sorted(",
        "        relative",
        "        for relative, digest in incoming.items()",
        "        if relative in known and known[relative] != digest",
        "    )",
        "    if conflicts:",
        "        raise RuntimeError(",
        '            "FastPLMs artifacts contain incompatible runtime sources at "',
        '            + ", ".join(repr(path) for path in conflicts[:5])',
        '            + ". Load incompatible releases in separate Python processes."',
        "        )",
        "    known = dict(known)",
        "    known.update(incoming)",
        '    package.__fastplms_artifact_runtime_files__ = known',
        '    roots = list(getattr(package, "__fastplms_artifact_runtime_roots__", ()))',
        "    if str(package_root) not in roots:",
        "        roots.append(str(package_root))",
        '    package.__fastplms_artifact_runtime_roots__ = tuple(roots)',
        "    temporaries = list(",
        '        getattr(package, "__fastplms_artifact_runtime_temporaries__", ())',
        "    )",
        "    for temporary in _RUNTIME_TEMPORARIES:",
        "        if temporary not in temporaries:",
        "            temporaries.append(temporary)",
        '    package.__fastplms_artifact_runtime_temporaries__ = tuple(temporaries)',
        '    hashes = set(getattr(package, "__fastplms_artifact_runtime_hashes__", ()))',
        "    hashes.add(RUNTIME_HASH)",
        '    package.__fastplms_artifact_runtime_hashes__ = frozenset(hashes)',
        "    _extend_loaded_package_paths(package_root)",
        "    return package",
        "",
        "def _import_without_bytecode(module_name):",
        "    previous = sys.dont_write_bytecode",
        "    sys.dont_write_bytecode = True",
        "    try:",
        "        return importlib.import_module(module_name)",
        "    finally:",
        "        sys.dont_write_bytecode = previous",
        "",
        "def _install_runtime():",
        '    package = sys.modules.get("fastplms")',
        '    hashes = getattr(package, "__fastplms_artifact_runtime_hashes__", ())',
        "    if RUNTIME_HASH in hashes:",
        "        return package",
        "    package_root = _ensure_runtime()",
        "    if package is not None:",
        "        return _merge_runtime(package, package_root)",
        "    spec = importlib.util.spec_from_file_location(",
        '        "fastplms",',
        '        package_root / "__init__.py",',
        "        submodule_search_locations=[str(package_root)],",
        "    )",
        "    if spec is None or spec.loader is None:",
        '        raise ImportError("Unable to load the embedded FastPLMs runtime.")',
        "    package = importlib.util.module_from_spec(spec)",
        "    package.__fastplms_artifact_runtime_hash__ = RUNTIME_HASH",
        "    package.__fastplms_artifact_runtime_hashes__ = frozenset({RUNTIME_HASH})",
        "    package.__fastplms_artifact_runtime_files__ = _runtime_file_hashes(package_root)",
        "    package.__fastplms_artifact_runtime_roots__ = (str(package_root),)",
        "    package.__fastplms_artifact_runtime_temporaries__ = tuple(",
        "        _RUNTIME_TEMPORARIES",
        "    )",
        '    sys.modules["fastplms"] = package',
        "    previous = sys.dont_write_bytecode",
        "    sys.dont_write_bytecode = True",
        "    try:",
        "        try:",
        "            spec.loader.exec_module(package)",
        "        except BaseException:",
        '            sys.modules.pop("fastplms", None)',
        "            raise",
        "    finally:",
        "        sys.dont_write_bytecode = previous",
        "    return package",
        "",
        "_install_runtime()",
    ]
    for module_name in sorted(grouped):
        variable = f"_module_{len(lines)}"
        lines.append(f'{variable} = _import_without_bytecode("{module_name}")')
        for class_name in sorted(set(grouped[module_name])):
            lines.extend(
                (
                    f"{class_name} = {variable}.{class_name}",
                    f"{class_name}.__module__ = __name__",
                )
            )
    lines.append("")
    return "\n".join(lines)


def _write_bootstrap(path: Path, spec: ModelSpec, runtime_hash: str) -> None:
    """Write a flat Transformers bridge to the bundled unchanged sources."""

    path.write_text(
        _render_bootstrap(spec, runtime_hash),
        encoding="utf-8",
        newline="\n",
    )


def _validate_bootstrap(path: Path, spec: ModelSpec, runtime_hash: str) -> None:
    try:
        actual = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ArtifactError(f"Artifact bootstrap is missing or invalid: {path}") from error
    if actual != _render_bootstrap(spec, runtime_hash):
        raise ArtifactError("Artifact bootstrap differs from the current deterministic generator.")


def render_model_card(spec: ModelSpec) -> str:
    """Render the canonical generated card used by documentation and artifacts."""

    from tools.artifacts.generate_docs import render_model_card as render_canonical_card

    return render_canonical_card(spec, allow_generic_family=True)


def _validated_model_card_template(
    source_root: Path,
    spec: ModelSpec,
    *,
    release_tool_revision: str,
    _allow_untracked_for_tests: bool = False,
) -> str:
    """Read one immutable tracked card template, or render with immutable tools."""

    source_root = source_root.resolve()
    card_relative = f"model_cards/{spec.id}.md"
    card_source = source_root.joinpath(*PurePosixPath(card_relative).parts)
    if _allow_untracked_for_tests:
        if card_source.is_file() and not card_source.is_symlink():
            return card_source.read_text(encoding="utf-8")
        return render_model_card(spec)
    git_metadata = source_root / ".git"
    if git_metadata.exists() or git_metadata.is_symlink():
        if re.fullmatch(r"[0-9a-f]{40}", release_tool_revision) is None:
            raise ArtifactError("Git model-card templates require a Git release-tool revision.")
        command_prefix = ["git", "-c", f"safe.directory={source_root.as_posix()}"]
        try:
            status = subprocess.run(
                [
                    *command_prefix,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--",
                    card_relative,
                ],
                cwd=source_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if status:
                raise ArtifactError(
                    "Artifact model-card template must be tracked and clean; scoped Git "
                    f"status: {status.replace(chr(10), '; ')}"
                )
            tracked = subprocess.run(
                [*command_prefix, "ls-files", "-z", "--", card_relative],
                cwd=source_root,
                check=True,
                capture_output=True,
            ).stdout
        except ArtifactError:
            raise
        except (OSError, subprocess.CalledProcessError) as error:
            raise ArtifactError("Unable to validate the tracked model-card template.") from error
        tracked_names = {
            raw.decode("utf-8") for raw in tracked.split(b"\0") if raw
        }
        if tracked_names:
            if tracked_names != {card_relative}:
                raise ArtifactError("Git returned an ambiguous model-card template.")
            payload = _git_archived_regular_files(
                source_root,
                release_tool_revision,
                (card_relative,),
                label="model-card template",
            )[card_relative]
            try:
                return payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ArtifactError("Tracked model-card template is not UTF-8.") from error
        if card_source.exists() or card_source.is_symlink():
            raise ArtifactError(f"Artifact model-card template is untracked: {card_source}")
        return render_model_card(spec)

    archive_marker = source_root / ARCHIVE_PROVENANCE_NAME
    if archive_marker.exists() or archive_marker.is_symlink():
        try:
            _diagnostic_head, inventory = validate_archived_root(source_root)
        except SourceProvenanceError as error:
            raise ArtifactError(
                f"Git-free model-card template attestation is invalid: {error}"
            ) from error
        record = inventory.get(card_relative)
        if record is None:
            if card_source.exists() or card_source.is_symlink():
                raise ArtifactError(
                    f"Artifact model-card template is absent from archive provenance: {card_source}"
                )
            return render_model_card(spec)
        if (
            card_source.is_symlink()
            or not card_source.is_file()
            or record.get("mode") not in {"100644", "100755"}
        ):
            raise ArtifactError(
                f"Attested model-card template is not a regular file: {card_source}"
            )
        try:
            payload = card_source.read_bytes()
        except OSError as error:
            raise ArtifactError(
                f"Unable to read attested model-card template: {card_source}"
            ) from error
        if (
            record.get("size") != len(payload)
            or record.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise ArtifactError("Attested model-card template mutated during snapshot.")
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ArtifactError("Attested model-card template is not UTF-8.") from error

    raise ArtifactError(
        "Artifact model-card templates require either a clean verifiable Git worktree "
        "or a content-attested tracked remote source archive."
    )


def _materialize_model_card(
    template: str,
    *,
    runtime_revision: str,
    source_tree_sha256: str,
    runtime_bundle_sha256: str,
) -> str:
    """Bind an immutable card template to one exact packaged runtime identity."""

    if re.fullmatch(
        r"(?:[0-9a-f]{40}|source-tree-sha256:[0-9a-f]{64})",
        runtime_revision,
    ) is None:
        raise ArtifactError(f"Invalid model-card runtime revision: {runtime_revision!r}")
    for label, digest in (
        ("source-tree", source_tree_sha256),
        ("runtime-bundle", runtime_bundle_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ArtifactError(f"Invalid model-card {label} SHA-256: {digest!r}")
    if _MODEL_CARD_RUNTIME_REVISION_PLACEHOLDER in template:
        raise ArtifactError("Model-card template retains a runtime-revision placeholder.")
    if template.count(_MODEL_CARD_RUNTIME_PROVENANCE) != 1:
        raise ArtifactError("Model-card template lacks the canonical runtime provenance line.")
    if template.count(_MODEL_CARD_DIGEST_PROVENANCE) != 1:
        raise ArtifactError("Model-card template lacks the canonical runtime-digest line.")
    materialized = template
    if _MODEL_CARD_RUNTIME_REVISION_PLACEHOLDER in materialized:
        raise ArtifactError("Materialized model card retains a runtime placeholder.")
    return materialized.rstrip() + "\n"


def _validate_vendor_revisions(source_root: Path, registry: ModelRegistry, spec: ModelSpec) -> None:
    for source_id in spec.family.upstreams:
        source = registry.upstreams[source_id]
        checkout = source_root / source.path
        if not checkout.is_dir():
            raise ArtifactError(
                f"Official source {source_id!r} is not initialized. Run "
                "'git submodule update --init --recursive'."
            )
        checkout_git_metadata = checkout / ".git"
        checkout_has_git = checkout_git_metadata.exists() or checkout_git_metadata.is_symlink()
        source_git_metadata = source_root / ".git"
        source_has_git = source_git_metadata.exists() or source_git_metadata.is_symlink()
        if not checkout_has_git:
            if source_has_git:
                raise ArtifactError(
                    f"Official source {source_id!r} is not initialized. Run "
                    "'git submodule update --init --recursive'."
                )
            try:
                validate_archived_submodule(
                    source_root,
                    relative_path=source.path,
                    expected_revision=source.revision,
                )
            except SourceProvenanceError as error:
                raise ArtifactError(
                    f"Official source {source_id!r} has invalid archive provenance: {error}"
                ) from error
            continue
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={checkout.as_posix()}",
                "-C",
                str(checkout),
                "rev-parse",
                "HEAD",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        revision = result.stdout.strip()
        if result.returncode != 0 or revision != source.revision:
            raise ArtifactError(
                f"Official source {source_id!r} must be at {source.revision}; "
                f"received {revision or result.stderr.strip()!r}."
            )
        status = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={checkout.as_posix()}",
                "-C",
                str(checkout),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        dirty_paths = status.stdout.strip()
        if status.returncode != 0 or dirty_paths:
            detail = dirty_paths or status.stderr.strip() or "unable to inspect worktree"
            raise ArtifactError(
                f"Official source {source_id!r} must have a clean worktree at "
                f"{source.revision}; received {detail!r}."
            )


def _copy_licenses(
    temporary: Path,
    source_root: Path,
    registry: ModelRegistry,
    spec: ModelSpec,
) -> None:
    for source_id in spec.family.upstreams:
        source = registry.upstreams[source_id]
        for expected in source.distribution_files:
            relative_path = PurePosixPath(expected.path)
            license_source = (source_root / "LICENSES" / source_id).joinpath(*relative_path.parts)
            license_target = (temporary / "LICENSES" / source_id).joinpath(*relative_path.parts)
            _copy_canonical_legal_file(license_source, license_target)
    _copy_canonical_legal_file(
        source_root / "LICENSE",
        temporary / "LICENSES" / "FastPLMs-Apache-2.0.txt",
    )
    _copy_canonical_legal_file(
        source_root / "THIRD_PARTY_NOTICES.md",
        temporary / "THIRD_PARTY_NOTICES.md",
    )


def _checkpoint_provenance(source: CheckpointSource) -> dict[str, Any]:
    return {
        "repo_id": source.repo_id,
        "revision": source.revision,
        "files": {item.path: item.encoded for item in source.files},
        "unresolved_files": list(source.unresolved_files),
    }


def _upstream_provenance(
    registry: ModelRegistry,
    spec: ModelSpec,
) -> list[dict[str, Any]]:
    return [
        {
            "id": source.id,
            "license": source.license_expression,
            "canonical_license_files": {
                item.path: item.encoded for item in source.license_digests
            },
            "distribution_files": {
                item.path: item.encoded for item in source.distribution_files
            },
            "path": source.path,
            "revision": source.revision,
            "url": source.url,
        }
        for source_id in spec.family.upstreams
        for source in (registry.upstreams[source_id],)
    ]


def _runtime_asset_provenance(
    registry: ModelRegistry,
    spec: ModelSpec,
) -> list[dict[str, Any]]:
    return [
        {
            "id": asset.id,
            "repository": asset.repository,
            "revision": asset.revision,
            "path": asset.path,
            "sha256": asset.sha256,
            "size": asset.size,
            "license": asset.license_expression,
            "consumer_family": asset.consumer_family,
            "trust_kind": asset.trust_kind,
            "offline_behavior": asset.offline_behavior,
            "cache_identity": hashlib.sha256(
                (
                    f"{asset.repository}@{asset.revision}:{asset.path}:"
                    f"{asset.sha256}:{asset.size}"
                ).encode()
            ).hexdigest(),
        }
        for asset in registry.runtime_assets.values()
        if asset.consumer_family == spec.family.id
    ]


def _expected_registry_provenance(
    registry: ModelRegistry,
    spec: ModelSpec,
) -> dict[str, Any]:
    selected_checkpoint = spec.artifact_checkpoint
    tokenizer_checkpoint = _tokenizer_checkpoint(registry, spec)
    return {
        "schema_version": _PROVENANCE_SCHEMA_VERSION,
        "generator": {
            "name": "tools.artifacts.build",
            "version": _ARTIFACT_GENERATOR_VERSION,
        },
        "fastplms_version": __version__,
        "model_id": spec.id,
        "architecture": spec.family.architecture,
        "auto_map": dict(spec.auto_map),
        "tokenizer_class": spec.family.tokenizer_class,
        "tokenizer_auto_map": (
            [
                "modeling_fastplms."
                + spec.family.tokenizer_class.rsplit(".", maxsplit=1)[1],
                None,
            ]
            if spec.family.tokenizer_class is not None
            else None
        ),
        "bf16_execution": spec.family.bf16_execution,
        **(
            {"msa_conditioning": spec.msa_conditioning}
            if spec.family.id == "esmfold2"
            else {}
        ),
        "checkpoint_license": spec.family.checkpoint_license,
        "weights_license_status": (
            "resolved" if spec.family.weights_publication_allowed else "unresolved"
        ),
        "redistributable": spec.family.weights_publication_allowed,
        "hub_license_metadata": dict(spec.family.hub_license_metadata),
        "legal_files": {item.path: item.encoded for item in registry.legal_files},
        "artifact_source": spec.artifact_source,
        "artifact_checkpoint": _checkpoint_provenance(selected_checkpoint),
        "weights_revision": selected_checkpoint.revision,
        "fast_checkpoint": _checkpoint_provenance(spec.fast),
        "official_checkpoint": _checkpoint_provenance(spec.official),
        "tokenizer_checkpoint": (
            {
                "repo_id": tokenizer_checkpoint.repo_id,
                "revision": tokenizer_checkpoint.revision,
                "files": {
                    item.path: item.encoded
                    for item in tokenizer_checkpoint.files
                    if PurePosixPath(item.path).name in _TOKENIZER_FILE_NAMES
                },
                "unresolved_files": [
                    path
                    for path in tokenizer_checkpoint.unresolved_files
                    if PurePosixPath(path).name in _TOKENIZER_FILE_NAMES
                ],
            }
            if spec.family.tokenizer_mode == "tokenizer"
            else None
        ),
        "oracle_assets": [
            {
                "role": asset.role,
                "path": asset.path,
                "url": asset.url,
                "sha256": asset.sha256,
                "size": asset.size,
            }
            for asset in spec.oracle_assets
        ],
        "runtime_assets": _runtime_asset_provenance(registry, spec),
        "state_transform": spec.family.state_transform,
        "conversion": {
            "id": spec.family.state_transform,
            "record": spec.family.conversion_provenance,
        },
        "conversion_equality_attestation": _conversion_equality_attestation(spec),
        "upstreams": _upstream_provenance(registry, spec),
    }


def _validate_registry_provenance(
    provenance: Mapping[str, Any],
    registry: ModelRegistry,
    spec: ModelSpec,
) -> None:
    try:
        registered = registry[spec.id]
    except KeyError as error:
        raise ArtifactError(f"Artifact model {spec.id!r} is absent from the registry.") from error
    if registered != spec:
        raise ArtifactError(f"Artifact model {spec.id!r} differs from the supplied registry.")
    expected = _expected_registry_provenance(registry, spec)
    mismatches = [
        name for name, value in expected.items() if provenance.get(name) != value
    ]
    if mismatches:
        raise ArtifactError(
            "Artifact provenance differs from the current registry for fields: "
            + ", ".join(sorted(mismatches))
        )


def _provenance(
    registry: ModelRegistry,
    spec: ModelSpec,
    canonical_weights: Mapping[str, Any],
    *,
    runtime_revision: str,
    source_tree_sha256: str,
    runtime_bundle_sha256: str,
    release_tool_revision: str,
    release_tool_sha256: str,
) -> dict[str, Any]:
    selected_checkpoint = spec.artifact_checkpoint
    return {
        **_expected_registry_provenance(registry, spec),
        "runtime_revision": runtime_revision,
        "source_tree_sha256": source_tree_sha256,
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "release_tool_revision": release_tool_revision,
        "release_tool_sha256": release_tool_sha256,
        "attestations": {
            "complete_artifact": {
                "scope": "weights+runtime",
                "weights_revision": selected_checkpoint.revision,
                "runtime_revision": runtime_revision,
                "release_tool_revision": release_tool_revision,
                "release_tool_sha256": release_tool_sha256,
                "weights_license_status": (
                    "resolved"
                    if spec.family.weights_publication_allowed
                    else "unresolved"
                ),
                "redistributable": spec.family.weights_publication_allowed,
            },
            "runtime_update": {
                "path": _RUNTIME_ATTESTATION_NAME,
                "scope": "runtime-only",
                "weights_repo_id": spec.fast.repo_id,
                "weights_revision": spec.fast.revision,
                "release_tool_revision": release_tool_revision,
                "release_tool_sha256": release_tool_sha256,
                "weights_license_status": (
                    "resolved"
                    if spec.family.weights_publication_allowed
                    else "unresolved"
                ),
                "redistributable": spec.family.weights_publication_allowed,
            },
        },
        "canonical_weights": dict(canonical_weights),
    }


def _tokenizer_checkpoint(
    registry: ModelRegistry,
    spec: ModelSpec,
) -> CheckpointSource:
    """Resolve the manifest-declared official tokenizer identity for a model."""

    if spec.tokenizer_source_id is None:
        return spec.official
    try:
        return registry[spec.tokenizer_source_id].official
    except KeyError as error:
        raise ArtifactError(
            f"Unknown tokenizer source {spec.tokenizer_source_id!r} for {spec.id}."
        ) from error


def _checkpoint_identity_hash_fields(
    repo_id: str,
    revision: str,
    files: Mapping[str, str],
) -> str:
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "files": [{"path": path, "digest": digest} for path, digest in sorted(files.items())],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _checkpoint_identity_hash(source: CheckpointSource) -> str:
    """Return a deterministic digest of one immutable checkpoint identity."""

    return _checkpoint_identity_hash_fields(
        source.repo_id,
        source.revision,
        {item.path: item.encoded for item in source.files},
    )


def _conversion_equality_attestation(spec: ModelSpec) -> dict[str, Any] | None:
    """Return the registry-owned conversion commitment for official-source state."""

    if spec.artifact_source != "official":
        return None
    expected_state_sha256 = spec.canonical_state_sha256
    if (
        not isinstance(expected_state_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_state_sha256) is None
    ):
        raise ArtifactError(
            f"Official-source model {spec.id!r} lacks a canonical-state commitment."
        )
    source = spec.artifact_checkpoint
    payload: dict[str, Any] = {
        "schema_version": _CONVERSION_ATTESTATION_SCHEMA_VERSION,
        "model_id": spec.id,
        "source_checkpoint": {
            "repo_id": source.repo_id,
            "revision": source.revision,
            "identity_sha256": _checkpoint_identity_hash(source),
        },
        "state_transform": spec.family.state_transform,
        "conversion_record_sha256": hashlib.sha256(
            spec.family.conversion_provenance.encode("utf-8")
        ).hexdigest(),
        "canonical_state": {
            "schema_version": _CANONICAL_STATE_SCHEMA_VERSION,
            "algorithm": "sha256",
            "sha256": expected_state_sha256,
        },
    }
    payload["attestation_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _content_manifest(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in _iter_files(root):
        relative = path.relative_to(root).as_posix()
        if relative != "artifact-manifest.json":
            result[relative] = f"sha256:{hash_file(path)}"
    return result


def _runtime_attestation(
    root: Path,
    spec: ModelSpec,
    *,
    weights_revision: str,
    runtime_revision: str,
    source_tree_sha256: str,
    runtime_bundle_sha256: str,
    release_tool_revision: str,
    release_tool_sha256: str,
) -> dict[str, Any]:
    """Create a deliberately runtime-scoped attestation for files-only updates."""

    canonical_weights = _load_json_object_for_build(root / "source-record.json").get(
        "canonical_weights"
    )
    if not isinstance(canonical_weights, Mapping):
        raise ArtifactError("Artifact provenance is missing canonical weight metadata.")
    index = canonical_weights.get("index")
    shards = canonical_weights.get("shards")
    if not isinstance(index, str) or not isinstance(shards, Mapping):
        raise ArtifactError("Artifact provenance has invalid canonical weight metadata.")
    excluded = {
        "artifact-manifest.json",
        "source-record.json",
        _RUNTIME_ATTESTATION_NAME,
        index,
        *(str(name) for name in shards),
    }
    files = {
        relative_name: encoded
        for relative_name, encoded in _content_manifest(root).items()
        if (
            relative_name not in excluded
            and not _is_weight_file(relative_name)
            and _is_runtime_update_path(relative_name)
        )
    }
    if not files or not any(name.startswith("fastplms/") for name in files):
        raise ArtifactError("Runtime attestation would contain no packaged runtime sources.")
    return {
        "schema_version": _RUNTIME_ATTESTATION_SCHEMA_VERSION,
        "scope": "runtime-only",
        "model_id": spec.id,
        "weights": {
            "repo_id": spec.fast.repo_id,
            "revision": weights_revision,
        },
        "runtime_revision": runtime_revision,
        "source_tree_sha256": source_tree_sha256,
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "release_tool_revision": release_tool_revision,
        "release_tool_sha256": release_tool_sha256,
        "weights_license_status": (
            "resolved" if spec.family.weights_publication_allowed else "unresolved"
        ),
        "redistributable": spec.family.weights_publication_allowed,
        "files": files,
    }


def _load_json_object_for_build(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Unable to read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON document must contain an object: {path}")
    return value


def _resolve_artifact_manifest_path(root: Path, relative_name: str) -> Path:
    """Resolve one portable manifest path while keeping it inside the artifact."""

    try:
        relative = _portable_relative_path(relative_name, "Artifact manifest path")
    except RegistryError as error:
        raise ArtifactError(
            f"invalid artifact manifest path: {relative_name!r}"
        ) from error
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ArtifactError(
            f"artifact manifest path escapes the artifact root: {relative_name!r}"
        ) from error
    return resolved


def _artifact_build_slots(
    output_root: Path,
    repository_name: str,
) -> tuple[Path, Path, Path]:
    """Return the only three sibling paths an artifact build may mutate."""

    try:
        relative = _portable_relative_path(repository_name, "Artifact repository name")
    except RegistryError as error:
        raise ArtifactError(f"Invalid artifact repository name: {repository_name!r}") from error
    if (
        len(relative.parts) != 1
        or relative.name != repository_name
        or repository_name.startswith(".")
    ):
        raise ArtifactError(f"Invalid artifact repository name: {repository_name!r}")
    slots = (
        output_root / repository_name,
        output_root / f".{repository_name}.tmp",
        output_root / f".{repository_name}.backup",
    )
    for slot in slots:
        if slot.parent != output_root:
            raise ArtifactError(f"Artifact build path escapes the output root: {slot}")
    return slots


def _is_artifact_path_link(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(callable(is_junction) and is_junction())


def _assert_artifact_directory_slot(path: Path, output_root: Path, label: str) -> None:
    """Reject escaped, linked, or non-directory artifact transaction slots."""

    if path.parent != output_root:
        raise ArtifactError(f"{label} escapes the resolved output root: {path}")
    if _is_artifact_path_link(path):
        raise ArtifactError(f"{label} must not be a symlink or junction: {path}")
    if path.exists() and not path.is_dir():
        raise ArtifactError(f"{label} must be a directory: {path}")


def _remove_artifact_directory(path: Path, output_root: Path, label: str) -> None:
    """Remove one checked sibling transaction directory without following links."""

    _assert_artifact_directory_slot(path, output_root, label)
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as error:
        raise ArtifactError(f"Unable to remove {label}: {path}") from error


def _atomic_artifact_rename(
    source: Path,
    destination: Path,
    output_root: Path,
) -> None:
    """Atomically rename one artifact slot to another on the same filesystem."""

    _assert_artifact_directory_slot(source, output_root, "Artifact rename source")
    _assert_artifact_directory_slot(destination, output_root, "Artifact rename destination")
    if not source.is_dir():
        raise ArtifactError(f"Artifact rename source is missing: {source}")
    if destination.exists():
        raise ArtifactError(f"Artifact rename destination already exists: {destination}")
    try:
        source.rename(destination)
    except OSError as error:
        raise ArtifactError(
            f"Unable to atomically rename artifact slot {source} to {destination}."
        ) from error


def _validate_artifact_slot(
    path: Path,
    output_root: Path,
    spec: ModelSpec,
    registry: ModelRegistry,
    label: str,
) -> None:
    _assert_artifact_directory_slot(path, output_root, label)
    if not path.is_dir():
        raise ArtifactError(f"{label} is missing: {path}")
    try:
        validate_artifact(path, spec=spec, registry=registry)
    except ArtifactError as error:
        raise ArtifactError(f"{label} is not a valid recoverable artifact: {path}") from error


def _recover_artifact_transaction(
    destination: Path,
    temporary: Path,
    backup: Path,
    output_root: Path,
    spec: ModelSpec,
    registry: ModelRegistry,
) -> None:
    """Recover or clean a transaction left by an interrupted artifact build."""

    for path, label in (
        (destination, "Artifact destination"),
        (temporary, "Artifact temporary directory"),
        (backup, "Artifact rollback backup"),
    ):
        _assert_artifact_directory_slot(path, output_root, label)

    destination_validated = False
    if backup.exists():
        if destination.exists():
            _validate_artifact_slot(
                destination,
                output_root,
                spec,
                registry,
                "Artifact destination beside a stale rollback backup",
            )
            destination_validated = True
            _remove_artifact_directory(
                backup,
                output_root,
                "stale artifact rollback backup",
            )
        else:
            _validate_artifact_slot(
                backup,
                output_root,
                spec,
                registry,
                "Interrupted artifact rollback backup",
            )
            _atomic_artifact_rename(backup, destination, output_root)
            destination_validated = True

    if temporary.exists():
        if not destination.exists():
            raise ArtifactError(
                "An incomplete artifact temporary directory exists without a valid destination "
                f"or rollback backup: {temporary}"
            )
        if not destination_validated:
            _validate_artifact_slot(
                destination,
                output_root,
                spec,
                registry,
                "Artifact destination beside a stale temporary directory",
            )
        _remove_artifact_directory(
            temporary,
            output_root,
            "stale artifact temporary directory",
        )


def _restore_artifact_backup(
    destination: Path,
    temporary: Path,
    backup: Path,
    output_root: Path,
    spec: ModelSpec,
    registry: ModelRegistry,
) -> None:
    """Restore the prior artifact while retaining every valid copy until validation."""

    if not backup.exists():
        raise ArtifactError(f"Artifact rollback backup is missing: {backup}")
    if destination.exists():
        if temporary.exists():
            raise ArtifactError(
                "Cannot quarantine a failed replacement because the temporary slot exists."
            )
        _atomic_artifact_rename(destination, temporary, output_root)
    _atomic_artifact_rename(backup, destination, output_root)
    _validate_artifact_slot(
        destination,
        output_root,
        spec,
        registry,
        "Restored artifact rollback backup",
    )
    if temporary.exists():
        _remove_artifact_directory(
            temporary,
            output_root,
            "failed artifact replacement",
        )


def _commit_artifact_transaction(
    destination: Path,
    temporary: Path,
    backup: Path,
    output_root: Path,
    spec: ModelSpec,
    registry: ModelRegistry,
) -> None:
    """Install a validated temporary artifact with rollback-safe directory swaps."""

    for path, label in (
        (destination, "Artifact destination"),
        (temporary, "Validated artifact temporary directory"),
        (backup, "Artifact rollback backup"),
    ):
        _assert_artifact_directory_slot(path, output_root, label)
    if not temporary.is_dir():
        raise ArtifactError(f"Validated artifact temporary directory is missing: {temporary}")
    if backup.exists():
        raise ArtifactError(f"Artifact rollback backup was not recovered: {backup}")

    if not destination.exists():
        _atomic_artifact_rename(temporary, destination, output_root)
        return

    try:
        _atomic_artifact_rename(destination, backup, output_root)
    except BaseException as error:
        try:
            if backup.exists() and not destination.exists():
                _restore_artifact_backup(
                    destination,
                    temporary,
                    backup,
                    output_root,
                    spec,
                    registry,
                )
            elif destination.exists():
                _validate_artifact_slot(
                    destination,
                    output_root,
                    spec,
                    registry,
                    "Artifact destination after a failed backup rename",
                )
                _remove_artifact_directory(
                    temporary,
                    output_root,
                    "aborted artifact replacement",
                )
        except BaseException as rollback_error:
            raise ArtifactError(
                "Unable to move the prior artifact into its rollback backup, and automatic "
                "recovery did not complete. The transaction slots were retained."
            ) from rollback_error
        raise ArtifactError(
            "Unable to move the prior artifact into its rollback backup; the prior artifact "
            "was restored."
        ) from error

    try:
        _atomic_artifact_rename(temporary, destination, output_root)
    except BaseException as error:
        try:
            _restore_artifact_backup(
                destination,
                temporary,
                backup,
                output_root,
                spec,
                registry,
            )
        except BaseException as rollback_error:
            raise ArtifactError(
                "Artifact replacement failed and automatic rollback did not complete. "
                "The transaction slots were retained for deterministic recovery."
            ) from rollback_error
        raise ArtifactError(
            "Artifact replacement failed; the prior validated artifact was restored."
        ) from error

    try:
        _remove_artifact_directory(
            backup,
            output_root,
            "completed artifact rollback backup",
        )
    except ArtifactError as error:
        raise ArtifactError(
            "The new artifact was installed, but its rollback backup could not be removed. "
            "The next invocation will validate the destination before cleanup."
        ) from error


def build_artifact(
    spec: ModelSpec,
    registry: ModelRegistry,
    checkpoint_dir: Path,
    output_root: Path,
    source_root: Path,
    *,
    tokenizer_dir: Path | None = None,
    replace: bool = False,
    _allow_untracked_runtime_for_tests: bool = False,
) -> Path:
    """Build one local artifact from an already verified checkpoint snapshot."""

    try:
        registered_spec = registry[spec.id]
    except KeyError as error:
        raise ArtifactError(f"Model {spec.id!r} is absent from the supplied registry.") from error
    if registered_spec != spec:
        raise ArtifactError(
            f"Model {spec.id!r} differs from the current supplied registry contract."
        )
    try:
        registry.require_resolved(spec.id)
    except (KeyError, RegistryError) as error:
        raise ArtifactError(str(error)) from error
    checkpoint_dir = checkpoint_dir.resolve()
    source_root = source_root.resolve()
    validate_repository_legal_inventory(source_root, registry, spec)
    selected_checkpoint = spec.artifact_checkpoint
    tokenizer_checkpoint = _tokenizer_checkpoint(registry, spec)
    resolved_tokenizer_dir: Path | None = None
    if spec.family.tokenizer_mode == "tokenizer":
        if tokenizer_dir is not None:
            resolved_tokenizer_dir = tokenizer_dir.resolve()
        elif selected_checkpoint == tokenizer_checkpoint:
            resolved_tokenizer_dir = checkpoint_dir
        else:
            raise ArtifactError(
                f"{spec.id} packages a FastPLMs checkpoint and requires the pinned official "
                "tokenizer snapshot via tokenizer_dir/--tokenizer-dir."
            )
    if _is_artifact_path_link(output_root):
        raise ArtifactError(
            f"Artifact output root must not be a symlink or junction: {output_root}"
        )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    repository_parts = spec.fast.repo_id.split("/", maxsplit=1)
    if len(repository_parts) != 2:
        raise ArtifactError(f"Invalid artifact repository id: {spec.fast.repo_id!r}")
    repository_name = repository_parts[1]
    destination, temporary, backup = _artifact_build_slots(output_root, repository_name)
    _recover_artifact_transaction(
        destination,
        temporary,
        backup,
        output_root,
        spec,
        registry,
    )
    if destination.exists() and not replace:
        raise ArtifactError(f"Artifact already exists: {destination}")
    temporary.mkdir(parents=True)
    try:
        _copy_checkpoint_assets(checkpoint_dir, temporary, selected_checkpoint)
        if resolved_tokenizer_dir is not None:
            _copy_official_tokenizer_assets(
                resolved_tokenizer_dir,
                temporary,
                tokenizer_checkpoint,
            )
        _configure_custom_tokenizer(temporary, spec)
        package_target = temporary / "fastplms"
        runtime_revision, runtime_payloads, expected_source_tree_sha256 = (
            _validated_runtime_snapshot(
                source_root,
                registry,
                spec,
                _allow_untracked_for_tests=_allow_untracked_runtime_for_tests,
            )
        )
        release_tool_revision, release_tool_sha256, release_tool_payloads = (
            _validated_release_tool_snapshot(
                source_root,
                _allow_untracked_for_tests=_allow_untracked_runtime_for_tests,
            )
        )
        canonical_weights = canonicalize_checkpoint_weights(
            checkpoint_dir,
            selected_checkpoint,
            temporary,
            state_transform=spec.family.state_transform,
            source_is_canonical=spec.artifact_source == "fast",
        )
        conversion_attestation = _conversion_equality_attestation(spec)
        if conversion_attestation is not None and canonical_weights.get("state_digest") != (
            conversion_attestation["canonical_state"]
        ):
            raise ArtifactError(
                f"Canonical state for {spec.id} differs from the registry-owned "
                "conversion equality commitment."
            )
        _write_runtime_snapshot(package_target, runtime_payloads)
        source_tree_sha256 = _tree_sha256(package_target)
        if source_tree_sha256 != expected_source_tree_sha256:
            raise ArtifactError(
                "Packaged runtime sources differ from the validated tracked snapshot."
            )

        config_path = temporary / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(f"Unable to read checkpoint config: {config_path}") from error
        if not isinstance(config, dict):
            raise ArtifactError("Checkpoint config.json must contain a JSON object.")
        _apply_artifact_config_contract(spec, config)
        config["auto_map"] = _artifact_auto_map(spec)
        config["fastplms_model_id"] = spec.id
        config["fastplms_checkpoint_repo_id"] = selected_checkpoint.repo_id
        config["fastplms_checkpoint_revision"] = selected_checkpoint.revision
        config["fastplms_checkpoint_hash"] = _checkpoint_identity_hash(selected_checkpoint)
        config["fastplms_weights_revision"] = selected_checkpoint.revision
        config["fastplms_runtime_revision"] = runtime_revision
        config["fastplms_source_tree_sha256"] = source_tree_sha256
        config["fastplms_release_tool_revision"] = release_tool_revision
        config["fastplms_release_tool_sha256"] = release_tool_sha256
        _write_json(config_path, config)
        runtime_hash = _write_runtime_bundle(
            temporary / "fastplms_bundle.py",
            package_target,
        )
        config["fastplms_runtime_bundle_sha256"] = runtime_hash
        _write_json(config_path, config)
        _write_bootstrap(temporary / "modeling_fastplms.py", spec, runtime_hash)

        card_template = _validated_model_card_template(
            source_root,
            spec,
            release_tool_revision=release_tool_revision,
            _allow_untracked_for_tests=_allow_untracked_runtime_for_tests,
        )
        try:
            card_license = parse_hub_license_metadata(card_template)
        except ValueError as error:
            raise ArtifactError(f"Model-card template metadata is invalid: {error}") from error
        if card_license != dict(spec.family.hub_license_metadata):
            raise ArtifactError(
                "Model-card template license metadata differs from models.toml."
            )
        card_text = _materialize_model_card(
            card_template,
            runtime_revision=runtime_revision,
            source_tree_sha256=source_tree_sha256,
            runtime_bundle_sha256=runtime_hash,
        )
        (temporary / "requirements.txt").write_text(
            _render_artifact_requirements(spec, release_tool_payloads),
            encoding="utf-8",
            newline="\n",
        )
        (temporary / "README.md").write_text(
            card_text,
            encoding="utf-8",
            newline="\n",
        )
        _copy_licenses(temporary, source_root, registry, spec)
        _write_json(
            temporary / "source-record.json",
            _provenance(
                registry,
                spec,
                canonical_weights,
                runtime_revision=runtime_revision,
                source_tree_sha256=source_tree_sha256,
                runtime_bundle_sha256=runtime_hash,
                release_tool_revision=release_tool_revision,
                release_tool_sha256=release_tool_sha256,
            ),
        )
        _write_json(
            temporary / _RUNTIME_ATTESTATION_NAME,
            _runtime_attestation(
                temporary,
                spec,
                weights_revision=spec.fast.revision,
                runtime_revision=runtime_revision,
                source_tree_sha256=source_tree_sha256,
                runtime_bundle_sha256=runtime_hash,
                release_tool_revision=release_tool_revision,
                release_tool_sha256=release_tool_sha256,
            ),
        )
        _write_json(temporary / "artifact-manifest.json", _content_manifest(temporary))
        validate_artifact(temporary, spec=spec, registry=registry)
    except BaseException:
        if temporary.exists():
            _remove_artifact_directory(
                temporary,
                output_root,
                "failed artifact temporary directory",
            )
        raise
    _commit_artifact_transaction(
        destination,
        temporary,
        backup,
        output_root,
        spec,
        registry,
    )
    return destination


def build_local_artifact(
    model_id: str,
    checkpoint_dir: Path,
    output_root: Path = Path("dist/hub"),
    source_root: Path | None = None,
    *,
    tokenizer_dir: Path | None = None,
    replace: bool = False,
) -> Path:
    """Validate provenance and build one manifest-selected local artifact."""

    registry = get_model_registry()
    try:
        spec = registry[model_id]
    except KeyError as error:
        raise ArtifactError(f"Unknown model ID: {model_id!r}") from error
    root = source_root or Path(__file__).resolve().parents[2]
    _validate_vendor_revisions(root.resolve(), registry, spec)
    return build_artifact(
        spec=spec,
        registry=registry,
        checkpoint_dir=checkpoint_dir,
        output_root=output_root,
        source_root=root,
        tokenizer_dir=tokenizer_dir,
        replace=replace,
    )


def validate_artifact(
    path: Path,
    *,
    spec: ModelSpec | None = None,
    registry: ModelRegistry | None = None,
) -> None:
    """Verify an artifact and optionally bind it to a current registry model."""

    if registry is not None and spec is None:
        raise ArtifactError("Artifact registry validation requires a selected model spec.")
    path = path.resolve()
    manifest_path = path / "artifact-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Unable to read artifact manifest: {manifest_path}") from error
    if not isinstance(manifest, dict) or not manifest:
        raise ArtifactError("artifact-manifest.json must contain a non-empty object.")
    failures: list[str] = []
    required_paths = {
        "README.md",
        "config.json",
        "source-record.json",
        "requirements.txt",
        _RUNTIME_ATTESTATION_NAME,
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/FastPLMs-Apache-2.0.txt",
        _WEIGHT_INDEX,
    }
    missing_required = sorted(required_paths.difference(manifest))
    if missing_required:
        failures.append(f"missing required artifact entries: {', '.join(missing_required)}")
    weight_validation_attempted = False
    for relative_name, encoded_digest in sorted(manifest.items()):
        if not isinstance(relative_name, str) or not isinstance(encoded_digest, str):
            failures.append("manifest keys and values must be strings")
            continue
        try:
            algorithm, expected = encoded_digest.split(":", maxsplit=1)
        except ValueError:
            failures.append(f"invalid digest entry for {relative_name}")
            continue
        try:
            artifact_file = _resolve_artifact_manifest_path(path, relative_name)
        except ArtifactError as error:
            failures.append(str(error))
            continue
        if not artifact_file.is_file():
            failures.append(f"missing {relative_name}")
            continue
        actual = hash_file(artifact_file, algorithm)
        if actual != expected:
            failures.append(f"digest mismatch for {relative_name}")
    unlisted = sorted(set(_content_manifest(path)).difference(manifest))
    if unlisted:
        failures.append(f"unlisted files: {', '.join(unlisted)}")
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        config = None
        failures.append("config.json is missing or invalid")
    if not isinstance(config, dict):
        config = None
        failures.append("config.json must contain an object")
    provenance_path = path / "source-record.json"
    try:
        card_text = (path / "README.md").read_text(encoding="utf-8")
        card_license = parse_hub_license_metadata(card_text)
    except (OSError, ValueError) as error:
        card_text = None
        card_license = None
        failures.append(f"README.md has invalid Hub license metadata: {error}")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        provenance = None
        failures.append("source-record.json is missing or invalid")
    if isinstance(provenance, dict):
        if spec is not None and registry is not None:
            try:
                _validate_registry_provenance(provenance, registry, spec)
            except ArtifactError as error:
                failures.append(str(error))
        if provenance.get("schema_version") != _PROVENANCE_SCHEMA_VERSION:
            failures.append("provenance schema version is missing or unsupported")
        if provenance.get("generator") != {
            "name": "tools.artifacts.build",
            "version": _ARTIFACT_GENERATOR_VERSION,
        }:
            failures.append("artifact generator identity is missing or unsupported")
        if spec is not None and provenance.get("model_id") != spec.id:
            failures.append("artifact model identity differs from the current registry")
        weights_license_status = provenance.get("weights_license_status")
        redistributable = provenance.get("redistributable")
        if (
            weights_license_status not in {"resolved", "unresolved"}
            or not isinstance(redistributable, bool)
            or redistributable != (weights_license_status == "resolved")
        ):
            failures.append("weight-license publication status is missing or invalid")
        if spec is not None and redistributable != spec.family.weights_publication_allowed:
            failures.append("weight-license publication status differs from the registry")
        raw_hub_license = provenance.get("hub_license_metadata")
        try:
            if not isinstance(raw_hub_license, Mapping):
                raise TypeError("Hub license metadata must be a mapping")
            provenance_license = validate_hub_license_metadata(raw_hub_license)
        except (TypeError, ValueError) as error:
            provenance_license = None
            failures.append(f"Hub license provenance is missing or invalid: {error}")
        if card_license is not None and provenance_license != card_license:
            failures.append("README Hub license metadata differs from provenance")
        bf16_execution = provenance.get("bf16_execution")
        if bf16_execution not in _BF16_EXECUTION_POLICIES:
            failures.append("BF16 execution provenance is missing or invalid")
        elif card_text is not None and f"`{bf16_execution}`" not in card_text:
            failures.append("README BF16 execution policy differs from provenance")
        artifact_source = provenance.get("artifact_source")
        artifact_checkpoint = provenance.get("artifact_checkpoint")
        if artifact_source not in {"fast", "official"}:
            failures.append("artifact_source is missing or invalid")
        if (
            not isinstance(artifact_checkpoint, dict)
            or not isinstance(artifact_checkpoint.get("repo_id"), str)
            or not artifact_checkpoint.get("repo_id")
            or not isinstance(artifact_checkpoint.get("revision"), str)
            or re.fullmatch(
                r"[0-9a-f]{40}", artifact_checkpoint.get("revision", "")
            )
            is None
        ):
            failures.append("selected artifact checkpoint provenance is missing")
        elif config is not None:
            checkpoint_files = artifact_checkpoint.get("files")
            if not isinstance(checkpoint_files, dict) or any(
                not isinstance(name, str) or not isinstance(digest, str)
                for name, digest in checkpoint_files.items()
            ):
                failures.append("selected artifact checkpoint file identities are invalid")
            else:
                expected_identity = _checkpoint_identity_hash_fields(
                    artifact_checkpoint["repo_id"],
                    artifact_checkpoint["revision"],
                    checkpoint_files,
                )
                expected_config_identity = {
                    "fastplms_model_id": provenance.get("model_id"),
                    "fastplms_checkpoint_repo_id": artifact_checkpoint["repo_id"],
                    "fastplms_checkpoint_revision": artifact_checkpoint["revision"],
                    "fastplms_checkpoint_hash": expected_identity,
                    "fastplms_weights_revision": artifact_checkpoint["revision"],
                    "fastplms_runtime_revision": provenance.get("runtime_revision"),
                    "fastplms_source_tree_sha256": provenance.get("source_tree_sha256"),
                    "fastplms_runtime_bundle_sha256": provenance.get(
                        "runtime_bundle_sha256"
                    ),
                    "fastplms_release_tool_revision": provenance.get(
                        "release_tool_revision"
                    ),
                    "fastplms_release_tool_sha256": provenance.get(
                        "release_tool_sha256"
                    ),
                }
                if spec is not None and spec.family.id == "esmfold2":
                    expected_config_identity["msa_conditioning"] = spec.msa_conditioning
                    msa_encoder = config.get("msa_encoder")
                    if not isinstance(msa_encoder, dict) or (
                        msa_encoder.get("enabled") != spec.msa_conditioning
                    ):
                        failures.append(
                            "config MSA encoder policy differs from the current registry"
                        )
                if any(
                    config.get(name) != expected
                    for name, expected in expected_config_identity.items()
                ):
                    failures.append("config packaging identity differs from checkpoint provenance")
        if provenance.get("weights_revision") != (
            artifact_checkpoint.get("revision")
            if isinstance(artifact_checkpoint, dict)
            else None
        ):
            failures.append("weights revision differs from checkpoint provenance")
        runtime_revision = provenance.get("runtime_revision")
        source_tree_sha256 = provenance.get("source_tree_sha256")
        runtime_bundle_sha256 = provenance.get("runtime_bundle_sha256")
        release_tool_revision = provenance.get("release_tool_revision")
        release_tool_sha256 = provenance.get("release_tool_sha256")
        if not isinstance(runtime_revision, str) or re.fullmatch(
            r"(?:[0-9a-f]{40}|source-tree-sha256:[0-9a-f]{64})",
            runtime_revision,
        ) is None:
            failures.append("runtime revision provenance is missing")
        if not isinstance(source_tree_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", source_tree_sha256
        ) is None:
            failures.append("runtime source-tree digest is missing or invalid")
        else:
            try:
                actual_source_tree_sha256 = _tree_sha256(path / "fastplms")
            except ArtifactError:
                actual_source_tree_sha256 = None
            if actual_source_tree_sha256 != source_tree_sha256:
                failures.append("runtime source-tree digest differs from packaged sources")
            if (
                isinstance(runtime_revision, str)
                and runtime_revision.startswith("source-tree-sha256:")
                and runtime_revision != f"source-tree-sha256:{source_tree_sha256}"
            ):
                failures.append("content-addressed runtime revision differs from source tree")
        if not isinstance(runtime_bundle_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", runtime_bundle_sha256
        ) is None:
            failures.append("runtime bundle digest is missing or invalid")
        else:
            try:
                _validate_runtime_bundle(
                    path / "fastplms_bundle.py",
                    path / "fastplms",
                    runtime_bundle_sha256,
                )
                if spec is not None:
                    _validate_bootstrap(
                        path / "modeling_fastplms.py",
                        spec,
                        runtime_bundle_sha256,
                    )
            except ArtifactError as error:
                failures.append(str(error))
        if not isinstance(release_tool_revision, str) or re.fullmatch(
            r"(?:[0-9a-f]{40}|release-tools-sha256:[0-9a-f]{64})",
            release_tool_revision,
        ) is None:
            failures.append("release-tool revision provenance is missing or invalid")
        if not isinstance(release_tool_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}",
            release_tool_sha256,
        ) is None:
            failures.append("release-tool digest provenance is missing or invalid")
        elif (
            isinstance(release_tool_revision, str)
            and release_tool_revision.startswith("release-tools-sha256:")
            and release_tool_revision != f"release-tools-sha256:{release_tool_sha256}"
        ):
            failures.append("content-addressed release-tool revision differs from its digest")
        if card_text is not None:
            if _MODEL_CARD_RUNTIME_REVISION_PLACEHOLDER in card_text:
                failures.append("README retains an unresolved runtime-revision placeholder")
            expected_card_lines = (
                _MODEL_CARD_RUNTIME_PROVENANCE,
                _MODEL_CARD_DIGEST_PROVENANCE,
            )
            if any(line not in card_text for line in expected_card_lines):
                failures.append("README runtime identity differs from provenance")
        attestations = provenance.get("attestations")
        complete_attestation = (
            attestations.get("complete_artifact") if isinstance(attestations, dict) else None
        )
        runtime_attestation_record = (
            attestations.get("runtime_update") if isinstance(attestations, dict) else None
        )
        fast_checkpoint = provenance.get("fast_checkpoint")
        expected_attestations = {
            "complete_artifact": {
                "scope": "weights+runtime",
                "weights_revision": provenance.get("weights_revision"),
                "runtime_revision": runtime_revision,
                "release_tool_revision": release_tool_revision,
                "release_tool_sha256": release_tool_sha256,
                "weights_license_status": weights_license_status,
                "redistributable": redistributable,
            },
            "runtime_update": {
                "path": _RUNTIME_ATTESTATION_NAME,
                "scope": "runtime-only",
                "weights_repo_id": (
                    fast_checkpoint.get("repo_id")
                    if isinstance(fast_checkpoint, dict)
                    else None
                ),
                "weights_revision": (
                    fast_checkpoint.get("revision")
                    if isinstance(fast_checkpoint, dict)
                    else None
                ),
                "release_tool_revision": release_tool_revision,
                "release_tool_sha256": release_tool_sha256,
                "weights_license_status": weights_license_status,
                "redistributable": redistributable,
            },
        }
        if (
            not isinstance(complete_attestation, dict)
            or not isinstance(runtime_attestation_record, dict)
            or attestations != expected_attestations
        ):
            failures.append("scoped artifact attestations are missing or invalid")
        if spec is not None:
            expected_checkpoint = (
                spec.fast if provenance.get("artifact_source") == "fast" else spec.official
            )
            expected_record = {
                "repo_id": expected_checkpoint.repo_id,
                "revision": expected_checkpoint.revision,
                "files": {item.path: item.encoded for item in expected_checkpoint.files},
                "unresolved_files": list(expected_checkpoint.unresolved_files),
            }
            if artifact_checkpoint != expected_record:
                failures.append("artifact checkpoint differs from the current registry")
        canonical_weights = provenance.get("canonical_weights")
        canonical_state = (
            canonical_weights.get("state_digest")
            if isinstance(canonical_weights, dict)
            else None
        )
        if (
            not isinstance(canonical_weights, dict)
            or canonical_weights.get("format") != "safetensors"
            or canonical_weights.get("index") != _WEIGHT_INDEX
            or canonical_weights.get("source_schema") not in {"canonical", "official"}
            or not isinstance(canonical_weights.get("state_transform"), str)
            or not canonical_weights.get("state_transform")
            or not isinstance(canonical_weights.get("shards"), dict)
            or not canonical_weights.get("shards")
            or not isinstance(canonical_state, dict)
            or canonical_state.get("schema_version") != _CANONICAL_STATE_SCHEMA_VERSION
            or canonical_state.get("algorithm") != "sha256"
            or not isinstance(canonical_state.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", canonical_state.get("sha256", "")) is None
        ):
            failures.append("canonical weight provenance is missing")
        else:
            expected_index_digest = canonical_weights.get("index_digest")
            expected_shards = cast(dict[str, str], canonical_weights["shards"])
            if manifest.get(_WEIGHT_INDEX) != expected_index_digest:
                failures.append("canonical weight index digest differs from artifact manifest")
            if any(manifest.get(name) != digest for name, digest in expected_shards.items()):
                failures.append("canonical shard digests differ from artifact manifest")
            weight_validation_attempted = True
            try:
                validate_weight_artifact(
                    path,
                    expected_state_sha256=canonical_state["sha256"],
                )
            except ArtifactError as error:
                failures.append(str(error))
        conversion_attestation = provenance.get("conversion_equality_attestation")
        if artifact_source == "official":
            if spec is None or registry is None:
                failures.append(
                    "official-source artifact validation requires a current registry commitment"
                )
            else:
                try:
                    expected_attestation = _conversion_equality_attestation(spec)
                except ArtifactError as error:
                    failures.append(str(error))
                else:
                    if expected_attestation is None:
                        failures.append(
                            "official-source artifact has no registry conversion commitment"
                        )
                    elif conversion_attestation != expected_attestation:
                        failures.append(
                            "conversion equality attestation differs from the current registry"
                        )
                    elif isinstance(canonical_state, dict) and (
                        canonical_state != expected_attestation["canonical_state"]
                    ):
                        failures.append(
                            "canonical state differs from the registry conversion commitment"
                        )
        elif conversion_attestation is not None:
            failures.append("canonical-source artifact has an unexpected conversion attestation")
        selected_record = provenance.get(f"{artifact_source}_checkpoint")
        if isinstance(artifact_checkpoint, dict) and artifact_checkpoint != selected_record:
            failures.append("selected artifact checkpoint differs from source record")
        conversion = provenance.get("conversion")
        if (
            not isinstance(conversion, dict)
            or not isinstance(conversion.get("id"), str)
            or not conversion.get("id")
            or not isinstance(conversion.get("record"), str)
            or not conversion.get("record")
        ):
            failures.append("conversion provenance is missing")
        elif isinstance(canonical_weights, dict) and (
            canonical_weights.get("state_transform") != conversion.get("id")
        ):
            failures.append("canonical weights did not use the declared conversion")
        oracle_assets = provenance.get("oracle_assets")
        if not isinstance(oracle_assets, list):
            failures.append("oracle asset provenance is missing")
        else:
            required_oracle_fields = {"role", "path", "url", "sha256", "size"}
            for asset in oracle_assets:
                if not isinstance(asset, dict) or set(asset) != required_oracle_fields:
                    failures.append("invalid oracle asset provenance entry")
                    continue
                if (
                    not isinstance(asset["role"], str)
                    or not isinstance(asset["path"], str)
                    or not isinstance(asset["url"], str)
                    or not isinstance(asset["sha256"], str)
                    or not isinstance(asset["size"], int)
                ):
                    failures.append("invalid oracle asset provenance value")
        runtime_assets = provenance.get("runtime_assets")
        if not isinstance(runtime_assets, list):
            failures.append("runtime asset provenance is missing")
        else:
            required_runtime_asset_fields = {
                "id",
                "repository",
                "revision",
                "path",
                "sha256",
                "size",
                "license",
                "consumer_family",
                "trust_kind",
                "offline_behavior",
                "cache_identity",
            }
            for asset in runtime_assets:
                if not isinstance(asset, dict) or set(asset) != required_runtime_asset_fields:
                    failures.append("invalid runtime asset provenance entry")
                    continue
                cache_material = (
                    f"{asset['repository']}@{asset['revision']}:{asset['path']}:"
                    f"{asset['sha256']}:{asset['size']}"
                ).encode()
                if (
                    not isinstance(asset["id"], str)
                    or not isinstance(asset["repository"], str)
                    or re.fullmatch(r"[0-9a-f]{40}", str(asset["revision"])) is None
                    or not isinstance(asset["path"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", str(asset["sha256"])) is None
                    or isinstance(asset["size"], bool)
                    or not isinstance(asset["size"], int)
                    or asset["size"] <= 0
                    or not isinstance(asset["license"], str)
                    or not asset["license"]
                    or asset["trust_kind"] != "hash_pinned_pickle"
                    or asset["offline_behavior"] != "requires_cached_verified_file"
                    or asset["cache_identity"] != hashlib.sha256(cache_material).hexdigest()
                ):
                    failures.append("invalid runtime asset provenance value")
        tokenizer_checkpoint = provenance.get("tokenizer_checkpoint")
        tokenizer_auto_map = provenance.get("tokenizer_auto_map")
        if tokenizer_auto_map is not None:
            try:
                tokenizer_config = json.loads(
                    (path / "tokenizer_config.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                tokenizer_config = None
            configured_auto_map = (
                tokenizer_config.get("auto_map", {}).get("AutoTokenizer")
                if isinstance(tokenizer_config, dict)
                and isinstance(tokenizer_config.get("auto_map"), dict)
                else None
            )
            if configured_auto_map != tokenizer_auto_map:
                failures.append("custom tokenizer AutoTokenizer mapping differs from provenance")
        if tokenizer_checkpoint is not None:
            tokenizer_files = (
                tokenizer_checkpoint.get("files")
                if isinstance(tokenizer_checkpoint, dict)
                else None
            )
            if (
                not isinstance(tokenizer_checkpoint, dict)
                or not isinstance(tokenizer_checkpoint.get("repo_id"), str)
                or not isinstance(tokenizer_checkpoint.get("revision"), str)
                or not isinstance(tokenizer_files, dict)
                or not tokenizer_files
            ):
                failures.append("official tokenizer provenance is missing or invalid")
            else:
                for relative_name, encoded_digest in tokenizer_files.items():
                    if PurePosixPath(relative_name).name not in _TOKENIZER_FILE_NAMES:
                        failures.append(
                            f"tokenizer provenance contains a non-tokenizer file: {relative_name}"
                        )
                        continue
                    if (
                        PurePosixPath(relative_name).name == "tokenizer_config.json"
                        and tokenizer_auto_map is not None
                    ):
                        # This file is intentionally rewritten to point at the
                        # artifact-local bridge. Its final digest is enforced by
                        # artifact-manifest.json and the mapping above.
                        continue
                    try:
                        algorithm, expected = encoded_digest.split(":", maxsplit=1)
                        actual = hash_file(path / relative_name, algorithm)
                    except (AttributeError, ArtifactError, OSError, ValueError):
                        failures.append(f"invalid tokenizer provenance for {relative_name}")
                        continue
                    if actual != expected:
                        failures.append(f"tokenizer digest mismatch for {relative_name}")
        upstreams = provenance.get("upstreams")
        if not isinstance(upstreams, list) or not upstreams:
            failures.append("upstream legal provenance is missing")
        else:
            for upstream in upstreams:
                if not isinstance(upstream, dict):
                    failures.append("invalid upstream provenance entry")
                    continue
                source_id = upstream.get("id")
                distributed = upstream.get("distribution_files")
                if not isinstance(source_id, str) or not isinstance(distributed, dict):
                    failures.append("invalid upstream distribution record")
                    continue
                for relative_name, encoded_digest in distributed.items():
                    artifact_name = f"LICENSES/{source_id}/{relative_name}"
                    if artifact_name not in manifest:
                        failures.append(f"missing required legal artifact {artifact_name}")
                        continue
                    if not isinstance(encoded_digest, str):
                        failures.append(f"invalid legal digest for {artifact_name}")
                        continue
                    try:
                        algorithm, expected = encoded_digest.split(":", maxsplit=1)
                        actual = hash_file(path / artifact_name, algorithm)
                    except (ArtifactError, OSError, ValueError):
                        failures.append(f"invalid legal digest for {artifact_name}")
                        continue
                    if actual != expected:
                        failures.append(f"legal digest mismatch for {artifact_name}")
        try:
            runtime_attestation = _load_json_object_for_build(
                path / _RUNTIME_ATTESTATION_NAME
            )
        except ArtifactError as error:
            runtime_attestation = None
            failures.append(str(error))
        if isinstance(runtime_attestation, dict):
            canonical = provenance.get("canonical_weights")
            raw_shards = canonical.get("shards") if isinstance(canonical, dict) else None
            weight_paths = {
                canonical.get("index") if isinstance(canonical, dict) else None,
                *(raw_shards if isinstance(raw_shards, dict) else ()),
            }
            excluded = {
                None,
                "artifact-manifest.json",
                "source-record.json",
                _RUNTIME_ATTESTATION_NAME,
                *weight_paths,
            }
            expected_runtime_files = {
                name: digest
                for name, digest in manifest.items()
                if (
                    name not in excluded
                    and not _is_weight_file(name)
                    and _is_runtime_update_path(name)
                )
            }
            expected_attestation_fields = {
                "schema_version": _RUNTIME_ATTESTATION_SCHEMA_VERSION,
                "scope": "runtime-only",
                "model_id": provenance.get("model_id"),
                "weights": {
                    "repo_id": (
                        provenance.get("fast_checkpoint", {}).get("repo_id")
                        if isinstance(provenance.get("fast_checkpoint"), dict)
                        else None
                    ),
                    "revision": (
                        provenance.get("fast_checkpoint", {}).get("revision")
                        if isinstance(provenance.get("fast_checkpoint"), dict)
                        else None
                    ),
                },
                "runtime_revision": provenance.get("runtime_revision"),
                "source_tree_sha256": provenance.get("source_tree_sha256"),
                "runtime_bundle_sha256": provenance.get("runtime_bundle_sha256"),
                "release_tool_revision": provenance.get("release_tool_revision"),
                "release_tool_sha256": provenance.get("release_tool_sha256"),
                "weights_license_status": provenance.get("weights_license_status"),
                "redistributable": provenance.get("redistributable"),
                "files": expected_runtime_files,
            }
            if runtime_attestation != expected_attestation_fields:
                failures.append(
                    "runtime-only attestation differs from artifact contents or provenance"
                )
    if not weight_validation_attempted:
        try:
            validate_weight_artifact(path)
        except ArtifactError as error:
            failures.append(str(error))
    if failures:
        raise ArtifactError("Artifact validation failed: " + "; ".join(failures))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="Stable model ID from src/fastplms/models.toml")
    parser.add_argument("checkpoint_dir", type=Path, help="Pinned local Hub snapshot")
    parser.add_argument("--output-root", type=Path, default=Path("dist/hub"))
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Pinned official tokenizer snapshot (required for FastPLMs tokenizer checkpoints)",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    registry = get_model_registry()
    try:
        spec = registry[args.model_id]
    except KeyError as error:
        raise ArtifactError(f"Unknown model ID: {args.model_id}") from error
    destination = build_local_artifact(
        model_id=args.model_id,
        checkpoint_dir=args.checkpoint_dir,
        output_root=args.output_root,
        source_root=args.source_root,
        tokenizer_dir=args.tokenizer_dir,
        replace=args.replace,
    )
    validate_artifact(destination, spec=spec, registry=registry)
    print(destination)


if __name__ == "__main__":
    main()


__all__ = [
    "ArtifactError",
    "build_artifact",
    "build_local_artifact",
    "canonicalize_checkpoint_weights",
    "hash_file",
    "main",
    "render_model_card",
    "validate_artifact",
    "validate_repository_legal_inventory",
    "validate_weight_artifact",
    "verify_checkpoint",
]
