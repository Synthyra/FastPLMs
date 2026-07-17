"""Build deterministic, offline-loadable local model artifacts.

This tool only reads an already downloaded checkpoint snapshot. It never logs
in, downloads weights, creates a Hub repository, or uploads files.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from fastplms import __version__
from fastplms.registry import (
    CheckpointSource,
    FileDigest,
    ModelRegistry,
    ModelSpec,
    RegistryError,
    UpstreamSource,
    get_model_registry,
)
from tools.artifacts.license_metadata import (
    parse_hub_license_metadata,
    render_checkpoint_terms,
    render_hub_license_yaml,
    validate_hub_license_metadata,
)
from tools.conversion import StateTransformError, apply_state_transform
from tools.source_provenance import SourceProvenanceError, validate_archived_submodule

_MAX_SHARD_BYTES = 5 * 1024**3
_IGNORED_PARTS = frozenset({".cache", ".git", "__pycache__"})
_WEIGHT_SUFFIXES = frozenset({".bin", ".ckpt", ".pt", ".pth", ".safetensors"})
_WEIGHT_INDEX = "model.safetensors.index.json"
_SHARD_NAME_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
_BF16_EXECUTION_POLICIES = frozenset({"static_parameters", "fp32_parameters_autocast"})
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
        _copy_file(
            snapshot.joinpath(*relative.parts),
            destination.joinpath(*relative.parts),
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
        if not source_path.is_file():
            raise ArtifactError(f"Official tokenizer file is missing: {source_path}")
        if hash_file(source_path, expected.algorithm) != expected.digest:
            raise ArtifactError(f"Official tokenizer digest differs for {expected.path}")
        _copy_file(source_path, destination.joinpath(*relative.parts))


def _load_checkpoint_state(snapshot: Path, source: CheckpointSource) -> dict[str, Any]:
    """Load a hash-verified state dictionary without unrestricted pickle."""

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
    verify_checkpoint(snapshot, source)
    state = _load_checkpoint_state(snapshot, source)
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
    validate_weight_artifact(destination, max_shard_bytes=max_shard_bytes)
    return {
        "format": "safetensors",
        "index": _WEIGHT_INDEX,
        "index_digest": f"sha256:{hash_file(index_path)}",
        "max_shard_bytes": max_shard_bytes,
        "shards": shard_hashes,
        "source_schema": "canonical" if source_is_canonical else "official",
        "state_transform": state_transform,
        "tensor_count": len(weight_map),
        "total_size": total_size,
    }


def validate_weight_artifact(
    path: Path,
    *,
    max_shard_bytes: int = _MAX_SHARD_BYTES,
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


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _copy_canonical_legal_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_legal_bytes(source))


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise ArtifactError(f"Required artifact source path does not exist: {source}")
    if source.is_file():
        _copy_file(source, destination)
        return
    for path in _iter_files(source):
        _copy_file(path, destination / path.relative_to(source))


def _copy_attention_kernel_lock(
    source_root: Path,
    package_target: Path,
    registry: ModelRegistry,
    spec: ModelSpec,
) -> None:
    """Embed the validated kernel lock when an artifact advertises FlashAttention."""
    implementations = tuple(
        name for name in spec.family.attention if name.startswith("flash_attention_")
    )
    if not implementations:
        return
    source = source_root / "kernels.lock"
    try:
        entries = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Unable to read the repository kernel lock: {source}") from error
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ArtifactError("kernels.lock must contain a list of JSON objects.")
    locked = {entry.get("repo_id"): entry.get("sha") for entry in entries}
    for implementation in implementations:
        kernel = registry.attention_kernels[implementation]
        if locked.get(kernel.repository) != kernel.revision:
            raise ArtifactError(
                f"kernels.lock does not match {implementation!r}: expected "
                f"{kernel.repository}@{kernel.revision}."
            )
    _copy_file(source, package_target / "kernels.lock")


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


def _bf16_execution_description(policy: str) -> str:
    if policy == "static_parameters":
        return "parameters are loaded directly in BF16"
    if policy == "fp32_parameters_autocast":
        return "parameters remain FP32 and the forward pass uses CUDA BF16 autocast"
    raise ArtifactError(f"Unsupported BF16 execution policy: {policy!r}")


def _build_runtime_archive(package_root: Path) -> bytes:
    """Return a deterministic archive of unchanged package runtime sources."""

    files = {
        (PurePosixPath("fastplms") / path.relative_to(package_root).as_posix()).as_posix(): (
            path.read_bytes()
        )
        for path in _iter_files(package_root)
    }
    if not files:
        raise ArtifactError("The artifact runtime source archive would be empty.")
    buffer = io.BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for archive_path, contents in sorted(files.items()):
            info = ZipInfo(archive_path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, contents, compress_type=ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def _write_runtime_bundle(path: Path, package_root: Path) -> str:
    """Write the flat source bundle consumed by Transformers remote code."""

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
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return archive_hash


def _write_bootstrap(path: Path, spec: ModelSpec, runtime_hash: str) -> None:
    """Write a flat Transformers bridge to the bundled unchanged sources."""

    grouped: dict[str, list[str]] = {}
    class_paths = list(spec.auto_map.values())
    if spec.family.tokenizer_class is not None:
        class_paths.append(spec.family.tokenizer_class)
    for class_path in class_paths:
        module_name, class_name = class_path.rsplit(".", maxsplit=1)
        grouped.setdefault(module_name, []).append(class_name)
    lines = [
        '"""Generated bridge to the unchanged FastPLMs package sources."""',
        "",
        "import base64",
        "import hashlib",
        "import importlib",
        "import importlib.util",
        "import shutil",
        "import sys",
        "import tempfile",
        "from io import BytesIO",
        "from pathlib import Path",
        "from zipfile import ZipFile",
        "",
        "from .fastplms_bundle import RUNTIME_DATA, RUNTIME_HASH",
        "",
        f'if RUNTIME_HASH != "{runtime_hash}":',
        '    raise RuntimeError("FastPLMs runtime identity differs from the bridge.")',
        "",
        "def _ensure_runtime():",
        '    payload = base64.b85decode("".join(RUNTIME_DATA))',
        "    if hashlib.sha256(payload).hexdigest() != RUNTIME_HASH:",
        '        raise RuntimeError("Embedded FastPLMs runtime hash mismatch.")',
        "    module_root = Path(__file__).resolve().parent",
        '    runtime_root = module_root / f"_fastplms_runtime_{RUNTIME_HASH[:16]}"',
        '    package_root = runtime_root / "fastplms"',
        '    marker = package_root / "__init__.py"',
        "    if not marker.is_file():",
        "        temporary = Path(",
        '            tempfile.mkdtemp(prefix=f".{runtime_root.name}.", dir=module_root)',
        "        )",
        "        try:",
        "            with ZipFile(BytesIO(payload)) as archive:",
        "                archive.extractall(temporary)",
        "            try:",
        "                temporary.rename(runtime_root)",
        "            except OSError:",
        "                if not marker.is_file():",
        "                    raise",
        "        finally:",
        "            if temporary.exists():",
        "                shutil.rmtree(temporary)",
        "    return package_root",
        "",
        "def _install_runtime():",
        '    installed = sys.modules.get("fastplms")',
        '    if getattr(installed, "__fastplms_artifact_runtime_hash__", None) == RUNTIME_HASH:',
        "        return installed",
        "    loaded = sorted(",
        "        name for name in sys.modules",
        '        if name == "fastplms" or name.startswith("fastplms.")',
        "    )",
        "    if loaded:",
        "        raise RuntimeError(",
        '            "A different FastPLMs runtime is already loaded. Artifacts with "',
        '            "different runtime hashes must run in separate Python processes."',
        "        )",
        "    package_root = _ensure_runtime()",
        "    spec = importlib.util.spec_from_file_location(",
        '        "fastplms",',
        '        package_root / "__init__.py",',
        "        submodule_search_locations=[str(package_root)],",
        "    )",
        "    if spec is None or spec.loader is None:",
        '        raise ImportError("Unable to load the embedded FastPLMs runtime.")',
        "    package = importlib.util.module_from_spec(spec)",
        "    package.__fastplms_artifact_runtime_hash__ = RUNTIME_HASH",
        '    sys.modules["fastplms"] = package',
        "    try:",
        "        spec.loader.exec_module(package)",
        "    except BaseException:",
        '        sys.modules.pop("fastplms", None)',
        "        raise",
        "    return package",
        "",
        "_install_runtime()",
    ]
    for module_name in sorted(grouped):
        variable = f"_module_{len(lines)}"
        lines.append(f'{variable} = importlib.import_module("{module_name}")')
        for class_name in sorted(set(grouped[module_name])):
            lines.extend(
                (
                    f"{class_name} = {variable}.{class_name}",
                    f"{class_name}.__module__ = __name__",
                )
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def render_model_card(spec: ModelSpec) -> str:
    """Render a concise card whose claims do not exceed manifest evidence."""

    auto_classes = ", ".join(f"`{name}`" for name in sorted(spec.auto_map))
    attention = ", ".join(f"`{name}`" for name in spec.family.attention)
    experimental_precisions = set(spec.family.experimental_precisions)
    precision = ", ".join(
        f"`{name}` (experimental)" if name in experimental_precisions else f"`{name}`"
        for name in spec.family.precisions
    )
    bf16_execution = _bf16_execution_description(spec.family.bf16_execution)
    license_yaml = render_hub_license_yaml(spec.family)
    checkpoint_terms = render_checkpoint_terms(spec.family)
    notes = ""
    if spec.notes:
        notes = f"""\
## Notes and limitations

{spec.notes}

"""
    return f"""---
library_name: transformers
{license_yaml}
---

# {spec.fast.repo_id}

This artifact packages the FastPLMs `{spec.family.architecture}` implementation
with one immutable checkpoint snapshot. It accepts the input mode documented for
the model family, applies the local FastPLMs model code, and returns the
corresponding Transformers model output.

## Load

```python
from transformers import AutoModel

artifact_path = "."
model = AutoModel.from_pretrained(
    artifact_path,
    local_files_only=True,
    trust_remote_code=True,
)
```

After publication, replace `artifact_path` with the Hub repository ID and pass
the immutable revision of the published FastPLMs 1.0 artifact. The checkpoint
revision in `provenance.json` identifies the source weights, not a future
artifact commit.

Advertised AutoClasses: {auto_classes}.

## Runtime contract

- Input mode: `{spec.family.tokenizer_mode}`
- Attention implementations: {attention}
- Precision policies: {precision}
- BF16 execution: `{spec.family.bf16_execution}` ({bf16_execution})
- Generation contract: `{spec.generation_contract}`
- Optional dependency group: `{spec.family.extra}`

{notes}## Provenance and validation boundary

The official comparison source is `{spec.official.repo_id}` at revision
`{spec.official.revision}`. The source implementation is pinned through
`{spec.family.upstreams[0]}`. Exact file identities and the state transformation
`{spec.family.state_transform}` are recorded in `provenance.json`.
The artifact weight source is `{spec.artifact_source}` checkpoint
`{spec.artifact_checkpoint.repo_id}` at revision
`{spec.artifact_checkpoint.revision}`.

The artifact metadata identifies the intended compliance contract. It is not a
claim of experimental validation, biological activity, or therapeutic effect.

## License

Checkpoint terms: {checkpoint_terms}. The Hub model-card identifier is
`{spec.family.hub_license}`. Applicable upstream texts and notices are included
under `LICENSES/` and in `THIRD_PARTY_NOTICES.md`.
"""


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


def _provenance(
    registry: ModelRegistry,
    spec: ModelSpec,
    canonical_weights: Mapping[str, Any],
) -> dict[str, Any]:
    upstreams: list[dict[str, Any]] = []
    for source_id in spec.family.upstreams:
        source: UpstreamSource = registry.upstreams[source_id]
        upstreams.append(
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
        )

    def checkpoint_record(source: CheckpointSource) -> dict[str, Any]:
        return {
            "repo_id": source.repo_id,
            "revision": source.revision,
            "files": {item.path: item.encoded for item in source.files},
            "unresolved_files": list(source.unresolved_files),
        }

    selected_checkpoint = spec.artifact_checkpoint
    tokenizer_checkpoint = _tokenizer_checkpoint(registry, spec)
    return {
        "schema_version": 1,
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
        "checkpoint_license": spec.family.checkpoint_license,
        "hub_license_metadata": dict(spec.family.hub_license_metadata),
        "legal_files": {item.path: item.encoded for item in registry.legal_files},
        "artifact_source": spec.artifact_source,
        "artifact_checkpoint": checkpoint_record(selected_checkpoint),
        "canonical_weights": dict(canonical_weights),
        "fast_checkpoint": checkpoint_record(spec.fast),
        "official_checkpoint": checkpoint_record(spec.official),
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
        "state_transform": spec.family.state_transform,
        "conversion": {
            "id": spec.family.state_transform,
            "record": spec.family.conversion_provenance,
        },
        "upstreams": upstreams,
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


def _content_manifest(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in _iter_files(root):
        relative = path.relative_to(root).as_posix()
        if relative != "artifact-manifest.json":
            result[relative] = f"sha256:{hash_file(path)}"
    return result


def _resolve_artifact_manifest_path(root: Path, relative_name: str) -> Path:
    """Resolve one portable manifest path while keeping it inside the artifact."""

    relative = PurePosixPath(relative_name)
    windows_relative = PureWindowsPath(relative_name)
    if (
        not relative_name
        or relative.is_absolute()
        or windows_relative.is_absolute()
        or windows_relative.drive
        or "\\" in relative_name
        or ".." in relative.parts
        or "." in relative.parts
        or relative_name != relative.as_posix()
    ):
        raise ArtifactError(f"invalid artifact manifest path: {relative_name!r}")
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ArtifactError(
            f"artifact manifest path escapes the artifact root: {relative_name!r}"
        ) from error
    return resolved


def build_artifact(
    spec: ModelSpec,
    registry: ModelRegistry,
    checkpoint_dir: Path,
    output_root: Path,
    source_root: Path,
    *,
    tokenizer_dir: Path | None = None,
    replace: bool = False,
) -> Path:
    """Build one local artifact from an already verified checkpoint snapshot."""

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
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    repository_name = spec.fast.repo_id.split("/", maxsplit=1)[1]
    destination = output_root / repository_name
    temporary = output_root / f".{repository_name}.tmp"
    if destination.exists() and not replace:
        raise ArtifactError(f"Artifact already exists: {destination}")
    if temporary.exists():
        if not replace:
            raise ArtifactError(f"Incomplete temporary artifact exists: {temporary}")
        shutil.rmtree(temporary)
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
        canonical_weights = canonicalize_checkpoint_weights(
            checkpoint_dir,
            selected_checkpoint,
            temporary,
            state_transform=spec.family.state_transform,
            source_is_canonical=spec.artifact_source == "fast",
        )
        package_source = source_root / "src" / "fastplms"
        package_target = temporary / "fastplms"
        for relative_name in spec.family.runtime_paths:
            relative_path = PurePosixPath(relative_name)
            _copy_tree(
                package_source.joinpath(*relative_path.parts),
                package_target.joinpath(*relative_path.parts),
            )
        _copy_attention_kernel_lock(source_root, package_target, registry, spec)

        config_path = temporary / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(f"Unable to read checkpoint config: {config_path}") from error
        if not isinstance(config, dict):
            raise ArtifactError("Checkpoint config.json must contain a JSON object.")
        config["auto_map"] = _artifact_auto_map(spec)
        config["fastplms_model_id"] = spec.id
        config["fastplms_checkpoint_repo_id"] = selected_checkpoint.repo_id
        config["fastplms_checkpoint_revision"] = selected_checkpoint.revision
        config["fastplms_checkpoint_hash"] = _checkpoint_identity_hash(selected_checkpoint)
        _write_json(config_path, config)
        runtime_hash = _write_runtime_bundle(
            temporary / "fastplms_bundle.py",
            package_target,
        )
        _write_bootstrap(temporary / "modeling_fastplms.py", spec, runtime_hash)

        card_source = source_root / "model_cards" / f"{spec.id}.md"
        if card_source.is_file():
            card_text = card_source.read_text(encoding="utf-8")
            try:
                card_license = parse_hub_license_metadata(card_text)
            except ValueError as error:
                raise ArtifactError(
                    f"Model card {card_source} has invalid Hub license metadata: {error}"
                ) from error
            if card_license != dict(spec.family.hub_license_metadata):
                raise ArtifactError(
                    f"Model card {card_source} license metadata differs from models.toml."
                )
            _copy_file(card_source, temporary / "README.md")
        else:
            card_text = render_model_card(spec)
            try:
                card_license = parse_hub_license_metadata(card_text)
            except ValueError as error:
                raise ArtifactError(f"Generated model-card metadata is invalid: {error}") from error
            if card_license != dict(spec.family.hub_license_metadata):
                raise ArtifactError(
                    "Generated model-card license metadata differs from models.toml."
                )
            (temporary / "README.md").write_text(card_text, encoding="utf-8", newline="\n")
        _copy_licenses(temporary, source_root, registry, spec)
        _write_json(
            temporary / "provenance.json",
            _provenance(registry, spec, canonical_weights),
        )
        _write_json(temporary / "artifact-manifest.json", _content_manifest(temporary))
        validate_artifact(temporary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
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


def validate_artifact(path: Path) -> None:
    """Verify the deterministic content manifest of an existing artifact."""

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
        "provenance.json",
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/FastPLMs-Apache-2.0.txt",
        _WEIGHT_INDEX,
    }
    missing_required = sorted(required_paths.difference(manifest))
    if missing_required:
        failures.append(f"missing required artifact entries: {', '.join(missing_required)}")
    try:
        validate_weight_artifact(path)
    except ArtifactError as error:
        failures.append(str(error))
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
    provenance_path = path / "provenance.json"
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
        failures.append("provenance.json is missing or invalid")
    if isinstance(provenance, dict):
        raw_hub_license = provenance.get("hub_license_metadata")
        try:
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
            or len(artifact_checkpoint.get("revision", "")) != 40
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
                }
                if any(
                    config.get(name) != expected
                    for name, expected in expected_config_identity.items()
                ):
                    failures.append("config packaging identity differs from checkpoint provenance")
        canonical_weights = provenance.get("canonical_weights")
        if (
            not isinstance(canonical_weights, dict)
            or canonical_weights.get("format") != "safetensors"
            or canonical_weights.get("index") != _WEIGHT_INDEX
            or canonical_weights.get("source_schema") not in {"canonical", "official"}
            or not isinstance(canonical_weights.get("state_transform"), str)
            or not canonical_weights.get("state_transform")
            or not isinstance(canonical_weights.get("shards"), dict)
            or not canonical_weights.get("shards")
        ):
            failures.append("canonical weight provenance is missing")
        else:
            expected_index_digest = canonical_weights.get("index_digest")
            expected_shards = canonical_weights.get("shards")
            if manifest.get(_WEIGHT_INDEX) != expected_index_digest:
                failures.append("canonical weight index digest differs from artifact manifest")
            if any(manifest.get(name) != digest for name, digest in expected_shards.items()):
                failures.append("canonical shard digests differ from artifact manifest")
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
    destination = build_local_artifact(
        model_id=args.model_id,
        checkpoint_dir=args.checkpoint_dir,
        output_root=args.output_root,
        source_root=args.source_root,
        tokenizer_dir=args.tokenizer_dir,
        replace=args.replace,
    )
    validate_artifact(destination)
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
