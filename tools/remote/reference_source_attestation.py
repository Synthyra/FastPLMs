"""Create and verify immutable source attestations for Git-free reference images."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from tools.source_provenance import actual_tree_paths, tracked_tree_digest


_SCHEMA_VERSION = 1
REFERENCE_SOURCE_EVIDENCE_SCHEMA_VERSION = 1
_HEX_REVISION_LENGTH = 40
_HEX_DIGEST_LENGTH = 64


class ReferenceSourceAttestationError(RuntimeError):
    """A reference source tree or imported package differs from its pinned contract."""


@dataclass(frozen=True)
class ReferenceSourceContract:
    """Immutable identity and import contract for one copied reference source tree."""

    schema_version: int
    source_revision: str
    tree_sha256: str
    import_name: str
    import_root: str
    package_version: str


def _load_json_bytes(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        serialized = path.read_bytes()
        raw: Any = json.loads(serialized.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceSourceAttestationError(f"Unable to read source contract: {path}") from error
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ReferenceSourceAttestationError(f"Source contract is not a JSON object: {path}")
    return {str(key): value for key, value in raw.items()}, serialized


def _load_json(path: Path) -> dict[str, object]:
    return _load_json_bytes(path)[0]


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _portable_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReferenceSourceAttestationError(f"{field} must be a non-empty string.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ReferenceSourceAttestationError(f"{field} is not a portable relative path: {value!r}")
    return value


def _validate_contract(raw: Mapping[str, object]) -> ReferenceSourceContract:
    required = {
        "schema_version",
        "source_revision",
        "tree_sha256",
        "import_name",
        "import_root",
        "package_version",
    }
    if set(raw) != required:
        raise ReferenceSourceAttestationError(
            "Reference source contract fields differ: "
            f"missing={sorted(required.difference(raw))}, "
            f"extra={sorted(set(raw).difference(required))}"
        )
    if raw["schema_version"] != _SCHEMA_VERSION:
        raise ReferenceSourceAttestationError("Unsupported reference source schema version.")
    source_revision = raw["source_revision"]
    tree_sha256 = raw["tree_sha256"]
    if not isinstance(source_revision, str) or not _is_lower_hex(
        source_revision, _HEX_REVISION_LENGTH
    ):
        raise ReferenceSourceAttestationError("Reference source revision must be 40 lowercase hex.")
    if not isinstance(tree_sha256, str) or not _is_lower_hex(
        tree_sha256, _HEX_DIGEST_LENGTH
    ):
        raise ReferenceSourceAttestationError("Reference tree digest must be 64 lowercase hex.")
    import_name = raw["import_name"]
    package_version = raw["package_version"]
    if not isinstance(import_name, str) or re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", import_name
    ) is None:
        raise ReferenceSourceAttestationError(
            "Reference import_name must name one top-level package."
        )
    if not isinstance(package_version, str) or not package_version:
        raise ReferenceSourceAttestationError("Reference package_version must be non-empty.")
    return ReferenceSourceContract(
        schema_version=_SCHEMA_VERSION,
        source_revision=source_revision,
        tree_sha256=tree_sha256,
        import_name=import_name,
        import_root=_portable_relative_path(raw["import_root"], field="import_root"),
        package_version=package_version,
    )


def load_reference_source_contract(path: Path) -> ReferenceSourceContract:
    """Load and validate one immutable source contract."""

    return _validate_contract(_load_json(path))


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_reference_source_attestation(
    source_root: Path,
    contract_path: Path,
    output: Path,
) -> dict[str, object]:
    """Verify a Git-free copied tree and persist its tracked runtime proof."""

    source_root = source_root.resolve()
    contract = load_reference_source_contract(contract_path)
    tracked_files = actual_tree_paths(source_root)
    actual_digest = tracked_tree_digest(source_root, tracked_files)
    if actual_digest != contract.tree_sha256:
        raise ReferenceSourceAttestationError(
            "Copied reference source tree differs from its pinned revision: "
            f"expected {contract.tree_sha256}, received {actual_digest}."
        )
    payload: dict[str, object] = {
        **asdict(contract),
        "file_count": len(tracked_files),
        "tracked_files": list(tracked_files),
    }
    _atomic_json(output, payload)
    return payload


def _load_attestation(
    path: Path,
) -> tuple[ReferenceSourceContract, tuple[str, ...], str]:
    raw, serialized = _load_json_bytes(path)
    extra_fields = {"file_count", "tracked_files"}
    contract_fields = set(ReferenceSourceContract.__dataclass_fields__)
    if set(raw) != contract_fields | extra_fields:
        raise ReferenceSourceAttestationError("Reference source attestation fields differ.")
    contract = _validate_contract({field: raw[field] for field in contract_fields})
    tracked_files = raw["tracked_files"]
    file_count = raw["file_count"]
    if not isinstance(tracked_files, list) or not all(
        isinstance(relative_name, str) and relative_name for relative_name in tracked_files
    ):
        raise ReferenceSourceAttestationError("Reference source attestation file list is invalid.")
    normalized = tuple(
        _portable_relative_path(relative_name, field="tracked_files entry")
        for relative_name in tracked_files
    )
    if normalized != tuple(sorted(normalized)) or len(normalized) != len(set(normalized)):
        raise ReferenceSourceAttestationError(
            "Reference source attestation file list is not unique and sorted."
        )
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count != len(normalized)
    ):
        raise ReferenceSourceAttestationError("Reference source attestation file count differs.")
    return contract, normalized, hashlib.sha256(serialized).hexdigest()


def validate_reference_source_evidence(value: object) -> dict[str, object]:
    """Validate the stable, portable provenance record written to native results."""

    expected_fields = {
        "schema_version",
        "source_revision",
        "tree_sha256",
        "attestation_sha256",
        "file_count",
        "import_name",
        "import_root",
        "import_file",
        "package_version",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ReferenceSourceAttestationError(
            "Reference source evidence fields differ from schema v1."
        )
    if value["schema_version"] != REFERENCE_SOURCE_EVIDENCE_SCHEMA_VERSION:
        raise ReferenceSourceAttestationError(
            "Unsupported reference source evidence schema version."
        )
    if not _is_lower_hex(value["source_revision"], _HEX_REVISION_LENGTH):
        raise ReferenceSourceAttestationError("Evidence source revision is invalid.")
    for field in ("tree_sha256", "attestation_sha256"):
        if not _is_lower_hex(value[field], _HEX_DIGEST_LENGTH):
            raise ReferenceSourceAttestationError(f"Evidence {field} is invalid.")
    file_count = value["file_count"]
    if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count <= 0:
        raise ReferenceSourceAttestationError("Evidence file_count must be positive.")
    import_name = value["import_name"]
    if not isinstance(import_name, str) or re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", import_name
    ) is None:
        raise ReferenceSourceAttestationError("Evidence import_name is invalid.")
    import_root = _portable_relative_path(value["import_root"], field="import_root")
    import_file = _portable_relative_path(value["import_file"], field="import_file")
    if not import_file.startswith(import_root + "/"):
        raise ReferenceSourceAttestationError(
            "Evidence import_file is outside the declared import_root."
        )
    package_version = value["package_version"]
    if not isinstance(package_version, str) or not package_version.strip():
        raise ReferenceSourceAttestationError("Evidence package_version is invalid.")
    return {field: value[field] for field in sorted(expected_fields)}


def validate_reference_sources_evidence(
    value: object,
    *,
    required_sources: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Validate a named, exact set of source attestations for one reference."""

    required = tuple(required_sources)
    if (
        not required
        or len(required) != len(set(required))
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
            for name in required
        )
    ):
        raise ReferenceSourceAttestationError(
            "Required reference source names must be unique lowercase slugs."
        )
    if not isinstance(value, Mapping) or set(value) != set(required):
        observed = sorted(str(name) for name in value) if isinstance(value, Mapping) else []
        raise ReferenceSourceAttestationError(
            "Reference source evidence names differ: "
            f"expected={sorted(required)}, observed={observed}."
        )
    return {
        name: validate_reference_source_evidence(value[name])
        for name in sorted(required)
    }


def _assert_exact_tree_inventory(
    source_root: Path,
    tracked_files: tuple[str, ...],
    *,
    stage: str,
) -> None:
    actual_files = actual_tree_paths(source_root)
    if actual_files == tracked_files:
        return
    missing = sorted(set(tracked_files).difference(actual_files))
    extra = sorted(set(actual_files).difference(tracked_files))
    raise ReferenceSourceAttestationError(
        f"Reference source inventory differs {stage}; "
        f"missing={missing[:10]}, extra={extra[:10]}."
    )


def _assert_source_file(
    raw_module_file: object,
    import_root: Path,
    *,
    context: str,
) -> Path:
    if not isinstance(raw_module_file, str):
        raise ReferenceSourceAttestationError(
            f"{context} reference package has no source file."
        )
    module_file = Path(raw_module_file).resolve()
    try:
        module_file.relative_to(import_root)
    except ValueError as error:
        raise ReferenceSourceAttestationError(
            f"{context} reference package resolves outside the pinned source: "
            f"{module_file}"
        ) from error
    return module_file


def _module_source_file(module: object, import_root: Path, *, context: str) -> Path:
    return _assert_source_file(
        getattr(module, "__file__", None),
        import_root,
        context=context,
    )


def _prioritize_import_parent(import_root: Path) -> None:
    import_parent = import_root.parent.resolve()
    retained: list[str] = []
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == import_parent:
                continue
        except OSError:
            pass
        retained.append(entry)
    sys.path[:] = [str(import_parent), *retained]


def _validate_cached_package_modules(
    import_name: str,
    import_root: Path,
) -> object | None:
    package_prefix = import_name + "."
    cached_top_level: object | None = None
    for module_name, module in tuple(sys.modules.items()):
        if module_name != import_name and not module_name.startswith(package_prefix):
            continue
        if module is None:
            raise ReferenceSourceAttestationError(
                f"Cached reference module {module_name!r} has no module object."
            )
        _module_source_file(module, import_root, context=f"Cached {module_name!r}")
        if module_name == import_name:
            cached_top_level = module
    return cached_top_level


def verify_reference_source(
    source_root: Path,
    attestation_path: Path,
    contract_path: Path,
    *,
    expected_revision: str,
) -> dict[str, object]:
    """Rehash the pinned tree and prove that the imported package comes from it."""

    if not _is_lower_hex(expected_revision, _HEX_REVISION_LENGTH):
        raise ReferenceSourceAttestationError(
            "Expected reference source revision must be 40 lowercase hex."
        )
    source_root = source_root.resolve()
    pinned_contract = load_reference_source_contract(contract_path)
    contract, tracked_files, attestation_sha256 = _load_attestation(attestation_path)
    if contract != pinned_contract:
        raise ReferenceSourceAttestationError(
            "Runtime source attestation differs from the checked-in contract."
        )
    if contract.source_revision != expected_revision:
        raise ReferenceSourceAttestationError(
            "Reference source revision differs from the runtime expectation: "
            f"expected {expected_revision}, received {contract.source_revision}."
        )
    _assert_exact_tree_inventory(source_root, tracked_files, stage="before import")
    actual_digest = tracked_tree_digest(source_root, tracked_files)
    if actual_digest != contract.tree_sha256:
        raise ReferenceSourceAttestationError(
            "Reference source tree changed after image construction: "
            f"expected {contract.tree_sha256}, received {actual_digest}."
        )

    import_root = source_root.joinpath(*PurePosixPath(contract.import_root).parts).resolve()
    try:
        import_root.relative_to(source_root)
    except ValueError as error:
        raise ReferenceSourceAttestationError(
            "Reference import root escapes its attested source tree."
        ) from error
    _prioritize_import_parent(import_root)

    cached_module = _validate_cached_package_modules(contract.import_name, import_root)
    if cached_module is not None:
        _module_source_file(cached_module, import_root, context="Cached top-level")
    else:
        spec = importlib.util.find_spec(contract.import_name)
        if spec is None or not isinstance(spec.origin, str):
            raise ReferenceSourceAttestationError(
                f"Pinned reference package {contract.import_name!r} is not importable."
            )
        _assert_source_file(
            spec.origin,
            import_root,
            context="Preflight",
        )

    previous_dont_write_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        module = importlib.import_module(contract.import_name)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    module_file = _module_source_file(module, import_root, context="Imported")
    if cached_module is not None and module is not cached_module:
        raise ReferenceSourceAttestationError(
            "Reference import cache identity changed during attestation."
        )
    _validate_cached_package_modules(contract.import_name, import_root)
    imported_version = getattr(module, "__version__", None)
    if imported_version != contract.package_version:
        raise ReferenceSourceAttestationError(
            f"Imported {contract.import_name!r} version {imported_version!r}, "
            f"expected {contract.package_version!r}."
        )
    _assert_exact_tree_inventory(source_root, tracked_files, stage="after import")
    post_import_digest = tracked_tree_digest(source_root, tracked_files)
    if post_import_digest != actual_digest:
        raise ReferenceSourceAttestationError(
            "Reference source tree changed while importing the pinned package."
        )
    import_file = module_file.relative_to(source_root).as_posix()
    return validate_reference_source_evidence({
        "schema_version": REFERENCE_SOURCE_EVIDENCE_SCHEMA_VERSION,
        "source_revision": contract.source_revision,
        "tree_sha256": actual_digest,
        "attestation_sha256": attestation_sha256,
        "file_count": len(tracked_files),
        "import_name": contract.import_name,
        "import_root": contract.import_root,
        "import_file": import_file,
        "package_version": imported_version,
    })


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--source-root", type=Path, required=True)
    create.add_argument("--contract", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--attestation", type=Path, required=True)
    verify.add_argument("--contract", type=Path, required=True)
    verify.add_argument("--expected-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Create or verify a reference-source attestation."""

    arguments = _parser().parse_args(argv)
    if arguments.command == "create":
        payload = create_reference_source_attestation(
            arguments.source_root,
            arguments.contract,
            arguments.output,
        )
    else:
        payload = verify_reference_source(
            arguments.source_root,
            arguments.attestation,
            arguments.contract,
            expected_revision=arguments.expected_revision,
        )
    summary = {
        key: payload[key]
        for key in (
            "schema_version",
            "source_revision",
            "tree_sha256",
            "attestation_sha256",
            "file_count",
            "import_name",
            "import_root",
            "import_file",
            "package_version",
        )
        if key in payload
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
