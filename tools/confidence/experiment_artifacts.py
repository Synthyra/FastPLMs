"""Immutable confidence evaluation inputs, completion records, and public exports.

Checkpoint snapshots remain local. Exporting copies verified evaluation records and their
identities to a new directory; it never uploads files or publishes trained weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import uuid

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

from tools.execution.source import excluded_from_upload, require_regular_source


SCHEMA_VERSION = 1
RESULT_FILES = frozenset({"request.json", "records.json", "skipped.json", "summary.json"})
ENVIRONMENT_PACKAGES = (
    "torch",
    "transformers",
    "numpy",
    "scipy",
    "safetensors",
    "DockQ",
    "tmtools",
)


def new_evaluation_id() -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def validate_evaluation_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", value):
        raise ValueError("Evaluation IDs must be 1-100 letters, digits, underscores, or hyphens")
    return value


def write_new_json(path: Path, payload: object) -> None:
    """Reserve a JSON file exclusively; a partial write cannot replace earlier evidence."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


@dataclass(frozen=True)
class FileIdentity:
    size: int
    sha256: str


def file_identity(path: Path) -> FileIdentity:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Artifact must be a regular file: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return FileIdentity(path.stat().st_size, digest)


def _artifact_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe artifact path: {name}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes its directory: {name}")
    return path


def source_identity(root: Path) -> dict[str, dict[str, int | str]]:
    """Record the actual runtime and confidence source bytes, including uncommitted fixes."""
    paths = [root / "src" / "fastplms" / "models.toml"]
    for directory in (
        root / "src" / "fastplms",
        root / "tools" / "confidence",
        root / "tools" / "execution",
    ):
        paths.extend(sorted(directory.rglob("*.py")))
    return {
        path.relative_to(root).as_posix(): asdict(
            file_identity(require_regular_source(root, path.relative_to(root)))
        )
        for path in paths
        if not excluded_from_upload(path.relative_to(root))
    }


def environment_identity() -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in ENVIRONMENT_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


@dataclass(frozen=True)
class EvaluationArtifacts:
    """One reserved evaluation directory whose original inputs are never replaced."""

    directory: Path

    @classmethod
    def create(
        cls,
        directory: Path,
        *,
        evaluation_id: str,
        model_id: str,
        split: str,
        targets: Sequence[Mapping[str, object]],
        head_files: Mapping[str, Path | None],
        metadata: Mapping[str, object],
        input_files: Mapping[str, Path],
    ) -> EvaluationArtifacts:
        validate_evaluation_id(evaluation_id)
        validate_evaluation_id(model_id)
        if split not in {"test", "validation"}:
            raise ValueError("Evaluations require the test or validation split")
        samples = metadata.get("inference", {}).get("samples")
        if type(samples) is not int or samples <= 0:
            raise ValueError("Evaluation metadata requires a positive integer sample count")
        if not targets:
            raise ValueError("An evaluation requires at least one target")
        if len({str(target["target_id"]) for target in targets}) != len(targets):
            raise ValueError("Evaluation target IDs must be unique")
        directory.mkdir(parents=True, exist_ok=False)
        run = cls(directory)
        try:
            public_files: dict[str, object] = {}
            private_files: dict[str, object] = {}
            checkpoint_inputs: dict[str, object] = {}
            for name, original in head_files.items():
                validate_evaluation_id(name)
                if original is None:
                    checkpoint_inputs[name] = {"source": "pinned_donor"}
                    continue
                relative = f"checkpoints/{name}.safetensors"
                snapshot = directory / relative
                snapshot.parent.mkdir(exist_ok=True)
                # Load the copied bytes later, so replacement of a training checkpoint cannot
                # change the checkpoint being evaluated after its identity is recorded.
                with original.open("rb") as source, snapshot.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
                identity = asdict(file_identity(snapshot))
                private_files[relative] = identity
                checkpoint_inputs[name] = {
                    "source_name": original.name,
                    "snapshot": relative,
                    **identity,
                }
            for name, original in input_files.items():
                if PurePosixPath(name).suffix != ".json":
                    raise ValueError("Public evaluation inputs must be JSON metadata")
                destination = _artifact_path(directory / "inputs", name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with original.open("rb") as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output)
                public_files[destination.relative_to(directory).as_posix()] = asdict(
                    file_identity(destination)
                )
            write_new_json(directory / "targets.json", list(targets))
            public_files["targets.json"] = asdict(file_identity(directory / "targets.json"))
            write_new_json(
                directory / "request.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "evaluation_id": evaluation_id,
                    "model_id": model_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "split": split,
                    "test_set_status": "spent" if split == "test" else "validation_dry_run",
                    "new_heldout_evaluation": False,
                    "requested_targets": len(targets),
                    "checkpoint_inputs": checkpoint_inputs,
                    "metadata": dict(metadata),
                    "environment": environment_identity(),
                    "public_inputs": public_files,
                    "private_inputs": private_files,
                },
            )
        except BaseException as error:
            run.fail(error)
            raise
        return run

    def head_files(self) -> dict[str, Path | None]:
        request = json.loads((self.directory / "request.json").read_text(encoding="utf-8"))
        _verify_files(self.directory, request["private_inputs"])
        return {
            name: _artifact_path(self.directory, value["snapshot"]) if "snapshot" in value else None
            for name, value in request["checkpoint_inputs"].items()
        }

    def complete(self) -> dict[str, object]:
        if (self.directory / "failure.json").exists():
            raise ValueError("A failed evaluation cannot be marked complete")
        request = json.loads((self.directory / "request.json").read_text(encoding="utf-8"))
        _verify_files(self.directory, request["public_inputs"])
        _verify_files(self.directory, request["private_inputs"])
        _verify_record_coverage(self.directory, request)
        public_files = dict(request["public_inputs"])
        for name in sorted(RESULT_FILES):
            public_files[name] = asdict(file_identity(self.directory / name))
        completion = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "completed_at": datetime.now(UTC).isoformat(),
            "evaluation_id": request["evaluation_id"],
            "model_id": request["model_id"],
            "split": request["split"],
            "public_files": public_files,
            "private_files": request["private_inputs"],
        }
        write_new_json(self.directory / "completion.json", completion)
        return completion

    def fail(self, error: BaseException) -> None:
        """Append a failure marker without altering inputs or any completed result."""
        path = self.directory / "failure.json"
        if not (self.directory / "completion.json").exists() and not path.exists():
            write_new_json(path, {"status": "failed", "error_type": type(error).__name__})


def _verify_files(root: Path, files: Mapping[str, object]) -> None:
    for name, expected in files.items():
        if asdict(file_identity(_artifact_path(root, name))) != expected:
            raise ValueError(f"Artifact integrity check failed: {name}")


def _verify_record_coverage(root: Path, request: Mapping[str, object]) -> None:
    """Every requested target must have all diffusion samples or an explicit skip record."""
    targets = json.loads((root / "targets.json").read_text(encoding="utf-8"))
    records = json.loads((root / "records.json").read_text(encoding="utf-8"))
    skipped = json.loads((root / "skipped.json").read_text(encoding="utf-8"))
    requested = {target["target_id"] for target in targets}
    if len(requested) != len(targets) or request.get("requested_targets") != len(targets):
        raise ValueError("Requested target count or uniqueness differs from the target inventory")
    skipped_ids = set(skipped)
    if len(skipped_ids) != len(skipped) or not skipped_ids <= requested:
        raise ValueError("Skipped target IDs must be unique members of the requested targets")
    samples = request["metadata"]["inference"]["samples"]
    if type(samples) is not int or samples <= 0:
        raise ValueError("Evaluation metadata requires a positive integer sample count")
    expected = {(target, sample) for target in requested - skipped_ids for sample in range(samples)}
    actual = {(record["target_id"], record["sample"]) for record in records}
    if not records or actual != expected or len(actual) != len(records):
        raise ValueError(
            "Evaluation records do not contain every expected target/sample exactly once"
        )
    expected_heads = set(request["checkpoint_inputs"]) or {"production"}
    if any(set(record["predictions"]) != expected_heads for record in records):
        raise ValueError("Evaluation prediction heads differ from the checkpoint inputs")


def verify_evaluation(directory: Path, *, require_checkpoints: bool = True) -> dict[str, object]:
    if (directory / "failure.json").exists():
        raise ValueError("Failed evaluations cannot be exported as complete evidence")
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    if completion.get("schema_version") != SCHEMA_VERSION or completion.get("status") != "complete":
        raise ValueError("Unsupported or incomplete evaluation manifest")
    public_files = completion.get("public_files")
    if (
        not isinstance(public_files, dict)
        or not RESULT_FILES | {"targets.json"} <= public_files.keys()
    ):
        raise ValueError("Completion manifest omits required public evaluation files")
    _verify_files(directory, public_files)
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported evaluation request schema")
    for name in ("evaluation_id", "model_id"):
        if not isinstance(request.get(name), str):
            raise ValueError(f"Evaluation request requires a string {name}")
        validate_evaluation_id(request[name])
        if request[name] != completion.get(name):
            raise ValueError(f"Request and completion {name} differ")
    split = request.get("split")
    if split not in {"test", "validation"} or completion.get("split") != split:
        raise ValueError("Request and completion split differ or are invalid")
    expected_status = "spent" if split == "test" else "validation_dry_run"
    if (
        request.get("test_set_status") != expected_status
        or request.get("new_heldout_evaluation") is not False
    ):
        raise ValueError("Evaluation request misidentifies test-set reuse")
    inputs = request.get("public_inputs")
    if not isinstance(inputs, dict) or "targets.json" not in inputs:
        raise ValueError("Evaluation request omits its target inventory")
    if set(public_files) != set(inputs) | RESULT_FILES:
        raise ValueError("Completion public inventory differs from the evaluation request")
    for name, identity in inputs.items():
        if public_files[name] != identity:
            raise ValueError(f"Request and completion input identity differ: {name}")
    private_inputs = request.get("private_inputs")
    if not isinstance(private_inputs, dict) or completion.get("private_files") != private_inputs:
        raise ValueError("Request and completion checkpoint inventory differ")
    checkpoint_inputs = request.get("checkpoint_inputs")
    if not isinstance(checkpoint_inputs, dict):
        raise ValueError("Evaluation request omits checkpoint identities")
    snapshots = {
        value["snapshot"]: {"size": value["size"], "sha256": value["sha256"]}
        for value in checkpoint_inputs.values()
        if "snapshot" in value
    }
    if private_inputs != snapshots:
        raise ValueError("Checkpoint input identities differ from the private inventory")
    _verify_record_coverage(directory, request)
    if require_checkpoints:
        _verify_files(directory, private_inputs)
    return completion


def _biological_targets(directory: Path) -> dict[str, tuple[object, ...]]:
    targets = json.loads((directory / "targets.json").read_text(encoding="utf-8"))
    identities: dict[str, tuple[object, ...]] = {}
    for target in targets:
        sequences = target.get("sequences")
        if (
            not isinstance(sequences, list)
            or not sequences
            or any(not isinstance(sequence, str) or not sequence for sequence in sequences)
        ):
            raise ValueError("Paired evaluation targets require ordered chain sequences")
        if target.get("num_tokens") != sum(map(len, sequences)) or target.get("num_chains") != len(
            sequences
        ):
            raise ValueError("Paired target token and chain counts disagree with its sequences")
        if not isinstance(target.get("stratum"), str):
            raise ValueError("Paired evaluation targets require a stratum")
        identities[target["target_id"]] = (
            tuple(sequences),
            target["num_tokens"],
            target["num_chains"],
            target["stratum"],
        )
    return identities


def _native_coordinate_identities(
    directory: Path, targets: Mapping[str, tuple[object, ...]]
) -> dict[str, dict[str, object]]:
    records = json.loads((directory / "records.json").read_text(encoding="utf-8"))
    identities: dict[str, dict[str, object]] = {}
    for record in records:
        target_id = record["target_id"]
        _, num_tokens, num_chains, stratum = targets[target_id]
        if (
            record.get("stratum") != stratum
            or record.get("num_chains") != num_chains
            or record.get("num_tokens", num_tokens) != num_tokens
        ):
            raise ValueError(f"Record biological fields differ from its target: {target_id}")
        identity = record.get("target_positions")
        if (
            not isinstance(identity, dict)
            or not isinstance(identity.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"])
            or identity.get("shape") != [targets[target_id][1], 14, 3]
            or identity.get("dtype") != "float32"
        ):
            raise ValueError(f"Missing or invalid native coordinate identity: {target_id}")
        if target_id in identities and identities[target_id] != identity:
            raise ValueError(f"Native coordinate identity changes between samples: {target_id}")
        identities[target_id] = identity
    return identities


def verify_evaluation_group(
    directories: Mapping[str, Path], *, split: str = "test"
) -> dict[str, dict[str, object]]:
    """Verify the biological inputs shared by independently folded model evaluations.

    Different requested or retained target subsets are allowed. Every overlapping target must
    retain the same ordered sequences and stratum; every jointly retained target must also
    use identical experimental coordinates. Predicted structures may differ across models.
    """
    if len(directories) < 2:
        raise ValueError("A paired evaluation group requires at least two models")
    completions: dict[str, dict[str, object]] = {}
    seen_targets: dict[str, tuple[object, ...]] = {}
    seen_coordinates: dict[str, dict[str, object]] = {}
    evaluation_id: str | None = None
    for model_id, directory in directories.items():
        completion = verify_evaluation(directory, require_checkpoints=False)
        if completion["model_id"] != model_id or completion["split"] != split:
            raise ValueError("Evaluation group model identity or requested split differs")
        if evaluation_id is None:
            evaluation_id = completion["evaluation_id"]
        elif completion["evaluation_id"] != evaluation_id:
            raise ValueError("Evaluations belong to different result groups")
        targets = _biological_targets(directory)
        coordinates = _native_coordinate_identities(directory, targets)
        for target_id in seen_targets.keys() & targets.keys():
            if seen_targets[target_id] != targets[target_id]:
                raise ValueError(f"Biological identity differs across evaluations: {target_id}")
        for target_id in seen_coordinates.keys() & coordinates.keys():
            if seen_coordinates[target_id] != coordinates[target_id]:
                raise ValueError(f"Native coordinates differ across evaluations: {target_id}")
        seen_targets.update(targets)
        seen_coordinates.update(coordinates)
        completions[model_id] = completion
    return completions


def export_evaluation(directory: Path, output_dir: Path) -> dict[str, object]:
    """Copy a completed evaluation into a new, checksummed bundle without weight files."""
    if output_dir.exists():
        raise FileExistsError(f"Export output must be a new directory: {output_dir}")
    completion = verify_evaluation(directory)
    names = [*completion["public_files"], "completion.json"]
    if any(PurePosixPath(name).suffix != ".json" for name in names):
        raise ValueError("Public evaluation exports may contain only JSON records and metadata")
    output_dir.mkdir(parents=True, exist_ok=False)
    exported: dict[str, object] = {}
    for name in names:
        source = _artifact_path(directory, name)
        destination = _artifact_path(output_dir, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as stream, destination.open("xb") as output:
            shutil.copyfileobj(stream, output)
        exported[name] = asdict(file_identity(destination))
    verify_evaluation(output_dir, require_checkpoints=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "operation": "public_evaluation_export",
        "checkpoint_weights_included": False,
        "files": exported,
    }
    write_new_json(output_dir / "export.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("operation", choices=("verify", "export"))
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="New public export directory")
    args = parser.parse_args()
    if args.operation == "export":
        if args.output_dir is None:
            parser.error("export requires --output-dir")
        report = export_evaluation(args.evaluation_dir, args.output_dir)
    else:
        export_path = args.evaluation_dir / "export.json"
        if export_path.exists():
            exported = json.loads(export_path.read_text(encoding="utf-8"))
            _verify_files(args.evaluation_dir, exported["files"])
        report = verify_evaluation(
            args.evaluation_dir, require_checkpoints=not export_path.exists()
        )
    print(json.dumps({"status": report["status"], "operation": args.operation}))


if __name__ == "__main__":
    main()
