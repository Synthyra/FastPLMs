"""Prepare local, loadable ESMFold2 small checkpoint artifacts.

This command is deliberately narrower than ``tools.artifacts.build``. It uses
the current checkout to compile runtime files, but it does not require a clean
tracked worktree and it does not emit a complete artifact compliance
attestation. Checkpoint files remain byte-identical to the verified official
snapshots.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from fastplms.models.esmfold2.configuration_esmfold2 import normalize_esmc_id
from fastplms.registry import ModelSpec, get_model_registry
from tools.artifacts.build import (
    ArtifactError,
    _apply_artifact_config_contract,
    _artifact_auto_map,
    _checkpoint_identity_hash,
    _portable_relative_path,
    hash_file,
    verify_checkpoint,
)
from tools.artifacts.publish import compile_model_files


_MODEL_IDS = ("esmfold2_300", "esmfold2_600")
_SNAPSHOT_DIRECTORIES = {
    "esmfold2_300": "fold300",
    "esmfold2_600": "fold600",
}
_WEIGHT_FILE = "model.safetensors"
_RUNTIME_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Unable to read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON document must contain an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _runtime_bundle_hash(payload: bytes) -> str:
    """Read the literal runtime identity from generated bundle source."""

    try:
        module = ast.parse(payload.decode("utf-8"), filename="fastplms_bundle.py")
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ArtifactError("Compiled runtime bundle is not valid Python source.") from error
    for statement in module.body:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "RUNTIME_HASH"
        ):
            continue
        try:
            value = ast.literal_eval(statement.value)
        except (ValueError, TypeError) as error:
            raise ArtifactError("Compiled runtime bundle has an invalid RUNTIME_HASH.") from error
        if isinstance(value, str) and _RUNTIME_HASH_PATTERN.fullmatch(value):
            return value
        raise ArtifactError("Compiled runtime bundle has an invalid RUNTIME_HASH.")
    raise ArtifactError("Compiled runtime bundle does not declare RUNTIME_HASH.")


def _file_record(payload: bytes, declared: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    if declared is not None:
        record["declared"] = declared
    return record


def _source_payload_records(snapshot: Path, spec: ModelSpec) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for expected in spec.official.files:
        source = snapshot.joinpath(*PurePosixPath(expected.path).parts)
        try:
            payload = source.read_bytes()
        except OSError as error:
            raise ArtifactError(f"Unable to read verified checkpoint file: {source}") from error
        actual = hash_file(source, expected.algorithm)
        if actual != expected.digest:
            raise ArtifactError(
                f"Official checkpoint file changed while recording: {expected.path}"
            )
        record = _file_record(payload, expected.encoded)
        record["verified"] = expected.encoded
        records[expected.path] = record
    return records


def _materialize_config(
    source_config: Mapping[str, Any],
    spec: ModelSpec,
    runtime_hash: str,
) -> dict[str, Any]:
    """Apply only runtime bridge fields to an official biological config."""

    config = dict(source_config)
    _apply_artifact_config_contract(spec, config)
    raw_esmc_id = config.get("esmc_id")
    if not isinstance(raw_esmc_id, str) or not raw_esmc_id:
        raise ArtifactError("ESMFold2 checkpoint config.esmc_id must be a non-empty string.")
    config["esmc_id"] = normalize_esmc_id(raw_esmc_id)
    config["auto_map"] = _artifact_auto_map(spec)
    config["fastplms_model_id"] = spec.id
    config["fastplms_checkpoint_repo_id"] = spec.official.repo_id
    config["fastplms_checkpoint_revision"] = spec.official.revision
    config["fastplms_checkpoint_hash"] = _checkpoint_identity_hash(spec.official)
    config["fastplms_weights_revision"] = spec.official.revision
    config["fastplms_runtime_bundle_sha256"] = runtime_hash
    return config


def _safe_output_path(root: Path, relative_name: str) -> Path:
    try:
        relative = _portable_relative_path(relative_name, "Prepared artifact path")
    except ValueError as error:
        raise ArtifactError(f"Invalid prepared artifact path: {relative_name!r}") from error
    destination = root.joinpath(*relative.parts).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as error:
        raise ArtifactError(
            f"Prepared artifact path escapes its root: {relative_name!r}"
        ) from error
    return destination


def _prepare_one(
    spec: ModelSpec,
    snapshot: Path,
    destination: Path,
    source_root: Path,
) -> Path:
    verify_checkpoint(snapshot, spec.official)
    source_records = _source_payload_records(snapshot, spec)
    compiled_files = compile_model_files(spec, source_root)
    try:
        runtime_hash = _runtime_bundle_hash(compiled_files["fastplms_bundle.py"])
    except KeyError as error:
        raise ArtifactError("Compiled Hub files are missing fastplms_bundle.py.") from error

    config_payload = _read_json_object(snapshot / "config.json")
    config = _materialize_config(config_payload, spec, runtime_hash)
    output_files = dict(compiled_files)
    output_files["config.json"] = (
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    weight_source = snapshot / _WEIGHT_FILE
    if not weight_source.is_file():
        raise ArtifactError(f"Pinned checkpoint has no {_WEIGHT_FILE}: {weight_source}")
    output_files[_WEIGHT_FILE] = weight_source.read_bytes()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.prepare-",
        dir=destination.parent,
    ) as temp:
        temporary = Path(temp)
        for relative_name, payload in sorted(output_files.items()):
            target = _safe_output_path(temporary, relative_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

        preparation_record = {
            "schema_version": 1,
            "scope": "local-preparation",
            "complete_artifact_compliance": False,
            "checkpoint_publication_status": spec.publication_status,
            "model_id": spec.id,
            "target_repository": spec.fast.repo_id,
            "official_checkpoint": {
                "repo_id": spec.official.repo_id,
                "revision": spec.official.revision,
                "files": source_records,
            },
            "runtime_bundle_sha256": runtime_hash,
            "source_payloads": {
                relative_name: _file_record(payload)
                for relative_name, payload in sorted(compiled_files.items())
            },
            "prepared_payloads": {
                relative_name: _file_record(payload)
                for relative_name, payload in sorted(output_files.items())
            },
            "config_contract": {
                "esmc_id": config["esmc_id"],
                "msa_conditioning": config["msa_conditioning"],
                "auto_map": config["auto_map"],
            },
        }
        _write_json(temporary / "preparation-record.json", preparation_record)
        if destination.exists() or destination.is_symlink():
            raise ArtifactError(
                f"Prepared artifact already exists: {destination}. Use --replace to replace it."
            )
        temporary.rename(destination)
    return destination


def prepare_esmfold2_small(
    *,
    snapshot_root: Path,
    output_root: Path,
    source_root: Path,
    replace: bool = False,
) -> tuple[Path, ...]:
    """Prepare exactly the manifest-scoped 300M and 600M ESMFold2 artifacts."""

    registry = get_model_registry()
    snapshot_root = snapshot_root.resolve()
    output_root = output_root.resolve()
    source_root = source_root.resolve()
    destinations: list[Path] = []
    for model_id in _MODEL_IDS:
        spec = registry[model_id]
        snapshot = snapshot_root / _SNAPSHOT_DIRECTORIES[model_id]
        repository_name = spec.fast.repo_id.rsplit("/", maxsplit=1)[1]
        destination = output_root / repository_name
        if not replace:
            destinations.append(_prepare_one(spec, snapshot, destination, source_root))
            continue
        staging = output_root / f".{repository_name}.prepared"
        if staging.exists() or staging.is_symlink():
            if staging.is_symlink() or not staging.is_dir():
                raise ArtifactError(f"Prepared artifact staging path is unsafe: {staging}")
            shutil.rmtree(staging)
        _prepare_one(spec, snapshot, staging, source_root)
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_dir():
                raise ArtifactError(f"Prepared artifact destination is unsafe: {destination}")
            shutil.rmtree(destination)
        staging.rename(destination)
        destinations.append(destination)
    return tuple(destinations)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, default=Path("artifacts/esmfold2-small"))
    parser.add_argument("--output-root", type=Path, default=Path("dist/hub"))
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    try:
        destinations = prepare_esmfold2_small(
            snapshot_root=arguments.snapshot_root,
            output_root=arguments.output_root,
            source_root=arguments.source_root,
            replace=arguments.replace,
        )
    except (ArtifactError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from error
    for destination in destinations:
        print(destination)


if __name__ == "__main__":
    main()


__all__ = ["main", "prepare_esmfold2_small"]
