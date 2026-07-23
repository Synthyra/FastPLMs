"""Run the manifest-declared Hopper/SM90 benchmark matrix outside pytest."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastplms.registry import ModelRegistry, ModelSpec, get_model_registry

from .regression import compare_reports
from .run import (
    _load_model,
    _require_torch,
    environment_fingerprint,
    run_case,
    validate_hopper_sm90_environment,
)

PADDING_LENGTHS = (1024, 512, 256, 128, 64, 64, 32, 32)
FIXED_SHAPES = (
    (1, 512, ()),
    (8, 1024, ()),
    (8, 1024, PADDING_LENGTHS),
)
EXHAUSTIVE_BATCH_SIZES = (1, 2, 4, 8)
EXHAUSTIVE_SEQUENCE_LENGTHS = (128, 256, 512, 1024)

SEQUENCE_FORWARD_PROFILE = "sequence_forward"
ESMFOLD2_REPRESENTATION_PROFILE = "esmfold2_representation"
STRUCTURE_STARTUP_PROFILE = "structure_startup"
STRUCTURE_DEDICATED_MODE = "structure"
ESMFOLD2_DEDICATED_MODE = "representation"
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RUNTIME_REVISION_PATTERN = re.compile(
    r"(?:[0-9a-f]{40}|source-tree-sha256:[0-9a-f]{64})"
)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read benchmark artifact metadata: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark artifact metadata must be a JSON object: {path}")
    return value


def _artifact_repository_name(spec: ModelSpec) -> str:
    parts = spec.fast.repo_id.split("/")
    if len(parts) != 2 or not all(parts) or parts[1] in {".", ".."}:
        raise ValueError(f"Invalid registry repository ID for {spec.id}: {spec.fast.repo_id!r}")
    return parts[1]


def _is_link_like(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction is not None and is_junction())


def _validate_built_artifact(
    path: Path,
    spec: ModelSpec,
    registry: ModelRegistry,
) -> None:
    """Apply the complete Hub-artifact validator without weakening its errors."""

    from tools.artifacts.build import ArtifactError, validate_artifact

    try:
        validate_artifact(path, spec=spec, registry=registry)
    except ArtifactError as error:
        raise ValueError(f"Invalid benchmark artifact for {spec.id}: {error}") from error


def _frozen_runtime_identity(
    source_root: Path,
    spec: ModelSpec,
    registry: ModelRegistry,
) -> tuple[str, str]:
    """Return the clean tracked runtime revision and digest for one frozen source tree."""

    from tools.artifacts.build import ArtifactError, _validated_runtime_snapshot

    try:
        runtime_revision, _payloads, source_tree_sha256 = _validated_runtime_snapshot(
            source_root,
            registry,
            spec,
        )
    except ArtifactError as error:
        raise ValueError(
            f"Unable to validate frozen benchmark source for {spec.id}: {error}"
        ) from error
    return runtime_revision, source_tree_sha256


def _require_identity(
    value: object,
    *,
    name: str,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not value or (
        pattern is not None and pattern.fullmatch(value) is None
    ):
        raise ValueError(f"Benchmark artifact has invalid {name}: {value!r}")
    return value


def _artifact_path(root: Path, spec: ModelSpec) -> Path:
    candidate = root / _artifact_repository_name(spec)
    if _is_link_like(candidate):
        raise ValueError(f"Benchmark artifact for {spec.id} may not be a link or junction")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"Benchmark artifact is missing for {spec.id}") from error
    if resolved.parent != root or not resolved.is_dir():
        raise ValueError(f"Benchmark artifact is not a contained directory for {spec.id}")
    return resolved


def _artifact_identity(
    path: Path,
    spec: ModelSpec,
    registry: ModelRegistry,
    *,
    source_root: Path,
) -> dict[str, Any]:
    """Validate and return path-free identities for one locally built Hub artifact."""

    registry.require_resolved(spec.id)
    _validate_built_artifact(path, spec, registry)
    expected_runtime_revision, expected_source_sha256 = _frozen_runtime_identity(
        source_root,
        spec,
        registry,
    )
    config = _load_json_object(path / "config.json")
    provenance = _load_json_object(path / "provenance.json")

    expected_config: dict[str, object] = {
        "fastplms_model_id": spec.id,
        "fastplms_checkpoint_repo_id": spec.artifact_checkpoint.repo_id,
        "fastplms_checkpoint_revision": spec.artifact_checkpoint.revision,
        "fastplms_weights_revision": spec.artifact_checkpoint.revision,
        "fastplms_runtime_revision": expected_runtime_revision,
        "fastplms_source_tree_sha256": expected_source_sha256,
    }
    config_mismatches = sorted(
        name for name, expected in expected_config.items() if config.get(name) != expected
    )
    if config_mismatches:
        raise ValueError(
            f"Benchmark artifact config for {spec.id} differs from the registry/frozen source: "
            + ", ".join(config_mismatches)
        )

    expected_provenance: dict[str, object] = {
        "model_id": spec.id,
        "weights_revision": spec.artifact_checkpoint.revision,
        "runtime_revision": expected_runtime_revision,
        "source_tree_sha256": expected_source_sha256,
    }
    provenance_mismatches = sorted(
        name
        for name, expected in expected_provenance.items()
        if provenance.get(name) != expected
    )
    if provenance_mismatches:
        raise ValueError(
            f"Benchmark artifact provenance for {spec.id} differs from the "
            "registry/frozen source: "
            + ", ".join(provenance_mismatches)
        )

    checkpoint = provenance.get("artifact_checkpoint")
    if not isinstance(checkpoint, Mapping) or (
        checkpoint.get("repo_id") != spec.artifact_checkpoint.repo_id
        or checkpoint.get("revision") != spec.artifact_checkpoint.revision
    ):
        raise ValueError(f"Benchmark artifact checkpoint identity differs for {spec.id}")

    runtime_revision = _require_identity(
        provenance.get("runtime_revision"),
        name="runtime revision",
        pattern=_RUNTIME_REVISION_PATTERN,
    )
    source_tree_sha256 = _require_identity(
        provenance.get("source_tree_sha256"),
        name="source-tree SHA-256",
        pattern=_SHA256_PATTERN,
    )
    runtime_bundle_sha256 = _require_identity(
        provenance.get("runtime_bundle_sha256"),
        name="runtime-bundle SHA-256",
        pattern=_SHA256_PATTERN,
    )
    if config.get("fastplms_runtime_bundle_sha256") != runtime_bundle_sha256:
        raise ValueError(f"Benchmark artifact runtime-bundle identity differs for {spec.id}")

    canonical_weights = provenance.get("canonical_weights")
    state_digest = (
        canonical_weights.get("state_digest")
        if isinstance(canonical_weights, Mapping)
        else None
    )
    if not isinstance(state_digest, Mapping) or (
        state_digest.get("algorithm") != "sha256"
        or state_digest.get("schema_version") != 1
    ):
        raise ValueError(f"Benchmark artifact canonical-state identity is invalid for {spec.id}")
    canonical_state_sha256 = _require_identity(
        state_digest.get("sha256"),
        name="canonical-state SHA-256",
        pattern=_SHA256_PATTERN,
    )
    try:
        manifest_sha256 = hashlib.sha256(
            (path / "artifact-manifest.json").read_bytes()
        ).hexdigest()
    except OSError as error:
        raise ValueError(f"Benchmark artifact manifest is unavailable for {spec.id}") from error

    return {
        "model_id": spec.id,
        "registry_repo_id": spec.fast.repo_id,
        "registry_revision": _require_identity(
            spec.fast.revision,
            name="registry revision",
            pattern=_COMMIT_PATTERN,
        ),
        "checkpoint_repo_id": spec.artifact_checkpoint.repo_id,
        "weights_revision": _require_identity(
            provenance.get("weights_revision"),
            name="weights revision",
            pattern=_COMMIT_PATTERN,
        ),
        "runtime_revision": runtime_revision,
        "source_tree_sha256": source_tree_sha256,
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "canonical_state_sha256": canonical_state_sha256,
        "artifact_manifest_sha256": manifest_sha256,
    }


def bind_local_artifacts(
    cases: Sequence[SimpleNamespace],
    artifact_root: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Prevalidate local artifacts and bind private load-only paths to benchmark cases."""

    if _is_link_like(artifact_root):
        raise ValueError("--artifact-root may not be a link or junction")
    try:
        root = artifact_root.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"--artifact-root does not exist: {artifact_root}") from error
    if not root.is_dir():
        raise ValueError(f"--artifact-root is not a directory: {artifact_root}")

    registry = get_model_registry()
    by_report_identity = {
        (spec.fast.repo_id, spec.fast.revision): spec for spec in registry.values()
    }
    selected: dict[str, ModelSpec] = {}
    for case in cases:
        spec = by_report_identity.get((str(case.model), str(case.revision)))
        if spec is None:
            raise ValueError(
                "Local benchmark artifacts require an exact registry model/revision identity; "
                f"got {(case.model, case.revision)!r}"
            )
        selected[spec.id] = spec
        if spec.family.id == "esmfold2":
            backbone_id = spec.family.backbone_model
            if backbone_id is None:
                raise ValueError(f"ESMFold2 benchmark model {spec.id} has no registry backbone")
            selected[backbone_id] = registry[backbone_id]

    artifact_paths: dict[str, Path] = {}
    missing: list[str] = []
    for spec in selected.values():
        try:
            artifact_paths[spec.id] = _artifact_path(root, spec)
        except ValueError:
            missing.append(f"{spec.id} ({spec.fast.repo_id})")
    if missing:
        raise ValueError("Missing or invalid selected benchmark artifacts: " + ", ".join(missing))

    path_owners: dict[Path, str] = {}
    for model_id, path in artifact_paths.items():
        previous = path_owners.setdefault(path, model_id)
        if previous != model_id:
            raise ValueError(
                f"Benchmark artifact path collision between {previous!r} and {model_id!r}"
            )

    frozen_root = (source_root or Path(__file__).resolve().parents[1]).resolve()
    identities = {
        model_id: _artifact_identity(
            artifact_paths[model_id],
            selected[model_id],
            registry,
            source_root=frozen_root,
        )
        for model_id in sorted(selected)
    }
    for case in cases:
        spec = by_report_identity[(str(case.model), str(case.revision))]
        case.load_model = artifact_paths[spec.id]
        case.load_revision = None
        case.local_files_only = True
        case.artifact_identity = identities[spec.id]
        if spec.family.id == "esmfold2":
            backbone_id = spec.family.backbone_model
            if backbone_id is None:
                raise ValueError(f"ESMFold2 benchmark model {spec.id} has no registry backbone")
            case.esmc_load_model = artifact_paths[backbone_id]
            case.artifact_dependencies = {"esmc": identities[backbone_id]}
        else:
            case.esmc_load_model = None
            case.artifact_dependencies = {}
    return identities


def benchmark_auto_class(spec: ModelSpec) -> str:
    """Select the manifest-advertised head measured for one architecture."""

    advertised = set(spec.auto_map)
    if (
        spec.family.id == "ankh"
        or spec.family.tokenizer_mode == "structure"
        or "AutoModelForMaskedLM" not in advertised
    ):
        selected = "AutoModel"
    else:
        selected = "AutoModelForMaskedLM"
    if selected not in advertised:
        raise ValueError(f"{spec.id} does not advertise required benchmark class {selected}")
    return selected


def benchmark_model_key(arguments: SimpleNamespace) -> tuple[str, str, str, str, str, str]:
    """Return the checkpoint identity that can share one in-memory model."""

    artifact_identity = getattr(arguments, "artifact_identity", None)
    artifact_manifest = (
        artifact_identity.get("artifact_manifest_sha256")
        if isinstance(artifact_identity, Mapping)
        else ""
    )
    return (
        str(arguments.model),
        str(arguments.revision),
        str(arguments.auto_class),
        str(arguments.precision),
        str(arguments.bf16_execution),
        str(artifact_manifest),
    )


def _default_backend(spec: ModelSpec) -> str:
    if "sdpa" in spec.family.attention:
        return "sdpa"
    if not spec.family.attention:
        raise ValueError(f"{spec.id} does not declare a benchmark backend")
    return spec.family.attention[0]


def _selected_backends(
    spec: ModelSpec,
    requested_backends: Sequence[str] | None,
) -> tuple[str, ...]:
    """Select a declared backend subset while preserving manifest order."""

    if requested_backends is None:
        return spec.family.attention
    requested = set(requested_backends)
    selected = tuple(
        backend for backend in spec.family.attention if backend in requested
    )
    if not selected:
        raise ValueError(
            f"Requested benchmark backends do not apply to {spec.id}: "
            + ", ".join(requested_backends)
        )
    return selected


def _arguments(
    spec: ModelSpec,
    *,
    backend: str,
    mode: str,
    batch_size: int,
    sequence_length: int,
    lengths: tuple[int, ...] = (),
    precision: str = "bf16",
    local_files_only: bool,
    profile: str,
    dedicated_mode: str | None = None,
    claim_eligible: bool | None = None,
    matrix_kind: str = "fixed",
) -> SimpleNamespace:
    if claim_eligible is None:
        claim_eligible = mode in {
            "compile",
            "steady",
            "projection",
            "esmc_projection",
        }
    return SimpleNamespace(
        model=spec.fast.repo_id,
        revision=spec.fast.revision,
        auto_class=benchmark_auto_class(spec),
        backend=backend,
        precision=precision,
        bf16_execution=spec.family.bf16_execution,
        mode=mode,
        batch_size=batch_size,
        sequence_length=sequence_length,
        lengths=lengths,
        local_files_only=local_files_only,
        seed=42,
        output=None,
        suite_profile=profile,
        dedicated_mode=dedicated_mode,
        claim_eligible=claim_eligible,
        matrix_kind=matrix_kind,
    )


def _representative_specs(family: str | None) -> list[ModelSpec]:
    registry = get_model_registry()
    specs = [
        spec
        for spec in registry.values()
        if (spec.is_deep_reference or spec.family.id == "esmfold2")
        and "benchmark" in spec.family.test_tiers
    ]
    if family is not None:
        specs = [spec for spec in specs if spec.family.id == family]
    if not specs:
        raise ValueError(f"No benchmark representative matches family={family!r}")
    return specs


def benchmark_artifact_model_ids() -> tuple[str, ...]:
    """Return the registry models needed by the complete fixed benchmark matrix."""

    registry = get_model_registry()
    selected: dict[str, None] = {}
    for spec in _representative_specs(None):
        selected[spec.id] = None
        if spec.family.id == "esmfold2":
            backbone_id = spec.family.backbone_model
            if backbone_id is None:
                raise ValueError(f"ESMFold2 benchmark model {spec.id} has no registry backbone")
            selected[backbone_id] = None
    # Registry order is the release order and keeps artifact construction stable.
    return tuple(model_id for model_id in registry if model_id in selected)


def _axis(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def benchmark_cases(
    *,
    family: str | None,
    quick: bool,
    local_files_only: bool,
    backends: Sequence[str] | None = None,
) -> Iterator[SimpleNamespace]:
    """Yield the fixed benchmark matrix derived from ``models.toml``."""

    specs = _representative_specs(family)

    if quick:
        spec = specs[0]
        if spec.family.id == "esmfold2":
            yield _arguments(
                spec,
                backend=_default_backend(spec),
                mode="projection",
                batch_size=1,
                sequence_length=16,
                local_files_only=local_files_only,
                profile=ESMFOLD2_REPRESENTATION_PROFILE,
                dedicated_mode=ESMFOLD2_DEDICATED_MODE,
            )
            return
        if spec.family.tokenizer_mode == "structure":
            yield _arguments(
                spec,
                backend=_default_backend(spec),
                mode="startup",
                batch_size=1,
                sequence_length=1,
                local_files_only=local_files_only,
                profile=STRUCTURE_STARTUP_PROFILE,
                dedicated_mode=STRUCTURE_DEDICATED_MODE,
                claim_eligible=False,
            )
            return
        yield _arguments(
            spec,
            backend=_default_backend(spec),
            mode="steady",
            batch_size=1,
            sequence_length=128,
            local_files_only=local_files_only,
            profile=SEQUENCE_FORWARD_PROFILE,
        )
        return

    for spec in specs:
        if spec.family.id == "esmfold2":
            for batch_size, sequence_length, lengths in FIXED_SHAPES:
                yield _arguments(
                    spec,
                    backend=_default_backend(spec),
                    mode="projection",
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    lengths=lengths,
                    precision="bf16",
                    local_files_only=local_files_only,
                    profile=ESMFOLD2_REPRESENTATION_PROFILE,
                    dedicated_mode=ESMFOLD2_DEDICATED_MODE,
                )
            for precision in ("bf16", "fp8"):
                for backend in _selected_backends(spec, backends):
                    for batch_size, sequence_length, lengths in FIXED_SHAPES:
                        yield _arguments(
                            spec,
                            backend=backend,
                            mode="esmc_projection",
                            batch_size=batch_size,
                            sequence_length=sequence_length,
                            lengths=lengths,
                            precision=precision,
                            local_files_only=local_files_only,
                            profile=ESMFOLD2_REPRESENTATION_PROFILE,
                            dedicated_mode=ESMFOLD2_DEDICATED_MODE,
                        )
                yield _arguments(
                    spec,
                    backend=_default_backend(spec),
                    mode="esmfold2_embed",
                    batch_size=1,
                    sequence_length=512,
                    precision=precision,
                    local_files_only=local_files_only,
                    profile=ESMFOLD2_REPRESENTATION_PROFILE,
                    dedicated_mode=ESMFOLD2_DEDICATED_MODE,
                    claim_eligible=False,
                )
            continue
        if spec.family.tokenizer_mode == "structure":
            yield _arguments(
                spec,
                backend=_default_backend(spec),
                mode="startup",
                batch_size=1,
                sequence_length=1,
                local_files_only=local_files_only,
                profile=STRUCTURE_STARTUP_PROFILE,
                dedicated_mode=STRUCTURE_DEDICATED_MODE,
                claim_eligible=False,
            )
            continue

        yield _arguments(
            spec,
            backend=_default_backend(spec),
            mode="startup",
            batch_size=1,
            sequence_length=512,
            local_files_only=local_files_only,
            profile=SEQUENCE_FORWARD_PROFILE,
            claim_eligible=False,
        )
        yield _arguments(
            spec,
            backend=_default_backend(spec),
            mode="embed",
            batch_size=1,
            sequence_length=512,
            local_files_only=local_files_only,
            profile=SEQUENCE_FORWARD_PROFILE,
            claim_eligible=False,
        )
        for backend in _selected_backends(spec, backends):
            yield _arguments(
                spec,
                backend=backend,
                mode="compile",
                batch_size=1,
                sequence_length=512,
                local_files_only=local_files_only,
                profile=SEQUENCE_FORWARD_PROFILE,
            )
            yield _arguments(
                spec,
                backend=backend,
                mode="steady",
                batch_size=1,
                sequence_length=512,
                local_files_only=local_files_only,
                profile=SEQUENCE_FORWARD_PROFILE,
            )
            yield _arguments(
                spec,
                backend=backend,
                mode="steady",
                batch_size=8,
                sequence_length=1024,
                local_files_only=local_files_only,
                profile=SEQUENCE_FORWARD_PROFILE,
            )
            yield _arguments(
                spec,
                backend=backend,
                mode="steady",
                batch_size=8,
                sequence_length=1024,
                lengths=PADDING_LENGTHS,
                local_files_only=local_files_only,
                profile=SEQUENCE_FORWARD_PROFILE,
            )


def exhaustive_benchmark_cases(
    *,
    family: str | None,
    batch_sizes: Iterable[int] = EXHAUSTIVE_BATCH_SIZES,
    sequence_lengths: Iterable[int] = EXHAUSTIVE_SEQUENCE_LENGTHS,
    local_files_only: bool,
    backends: Sequence[str] | None = None,
) -> Iterator[SimpleNamespace]:
    """Yield a descriptive all-checkpoint sweep that is never claim-eligible."""

    batches = _axis(batch_sizes, "batch_sizes")
    lengths = _axis(sequence_lengths, "sequence_lengths")
    registry = get_model_registry()
    specs = [
        spec
        for spec in registry.values()
        if "benchmark" in spec.family.test_tiers and (family is None or spec.family.id == family)
    ]
    if not specs:
        raise ValueError(f"No exhaustive benchmark checkpoint matches family={family!r}")

    for spec in specs:
        if spec.family.id == "esmfold2":
            for batch_size in batches:
                for sequence_length in lengths:
                    yield _arguments(
                        spec,
                        backend=_default_backend(spec),
                        mode="projection",
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        local_files_only=local_files_only,
                        profile=ESMFOLD2_REPRESENTATION_PROFILE,
                        dedicated_mode=ESMFOLD2_DEDICATED_MODE,
                        claim_eligible=False,
                        matrix_kind="exhaustive",
                    )
                    for precision in ("bf16", "fp8"):
                        for backend in _selected_backends(spec, backends):
                            yield _arguments(
                                spec,
                                backend=backend,
                                mode="esmc_projection",
                                batch_size=batch_size,
                                sequence_length=sequence_length,
                                precision=precision,
                                local_files_only=local_files_only,
                                profile=ESMFOLD2_REPRESENTATION_PROFILE,
                                dedicated_mode=ESMFOLD2_DEDICATED_MODE,
                                claim_eligible=False,
                                matrix_kind="exhaustive",
                            )
            continue
        if spec.family.tokenizer_mode == "structure":
            yield _arguments(
                spec,
                backend=_default_backend(spec),
                mode="startup",
                batch_size=1,
                sequence_length=1,
                local_files_only=local_files_only,
                profile=STRUCTURE_STARTUP_PROFILE,
                dedicated_mode=STRUCTURE_DEDICATED_MODE,
                claim_eligible=False,
                matrix_kind="exhaustive",
            )
            continue
        for backend in _selected_backends(spec, backends):
            for batch_size in batches:
                for sequence_length in lengths:
                    yield _arguments(
                        spec,
                        backend=backend,
                        mode="steady",
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        local_files_only=local_files_only,
                        profile=SEQUENCE_FORWARD_PROFILE,
                        claim_eligible=False,
                        matrix_kind="exhaustive",
                    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_junit(
    path: Path,
    *,
    suite_name: str,
    failures: Sequence[str] = (),
) -> None:
    """Write one atomic, dependency-free JUnit summary for orchestration."""

    root = ET.Element(
        "testsuite",
        {
            "name": suite_name,
            "tests": "1",
            "failures": "1" if failures else "0",
            "errors": "0",
            "skipped": "0",
        },
    )
    case = ET.SubElement(root, "testcase", {"classname": "benchmarks", "name": suite_name})
    if failures:
        failure = ET.SubElement(case, "failure", {"message": failures[0]})
        failure.text = "\n".join(failures)
    ET.indent(root, space="  ")
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _gate_failures(gate: Any) -> tuple[str, ...]:
    failures = [
        *gate.report_mismatches,
        *gate.environment_mismatches,
        *gate.artifact_mismatches,
        *(f"unmatched current case: {case}" for case in gate.unmatched_current),
        *(f"unmatched baseline case: {case}" for case in gate.unmatched_baseline),
    ]
    failures.extend(
        f"{case.case}: {reason}"
        for case in gate.cases
        for reason in case.reasons
    )
    if not gate.passed and not failures:
        failures.append("Benchmark gate did not contain any comparable throughput cases")
    return tuple(failures)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--gate-output", type=Path)
    parser.add_argument("--junit-output", type=Path)
    parser.add_argument("--family")
    matrix = parser.add_mutually_exclusive_group()
    matrix.add_argument("--quick", action="store_true")
    matrix.add_argument("--exhaustive", action="store_true")
    parser.add_argument(
        "--exhaustive-batch-sizes",
        nargs="+",
        type=int,
        default=EXHAUSTIVE_BATCH_SIZES,
    )
    parser.add_argument(
        "--exhaustive-sequence-lengths",
        nargs="+",
        type=int,
        default=EXHAUSTIVE_SEQUENCE_LENGTHS,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=(
            "eager",
            "sdpa",
            "flex_attention",
            "flash_attention_2",
            "flash_attention_3",
        ),
        help=(
            "Restrict the matrix to this explicit backend subset. The GH200 release "
            "runner passes eager, SDPA, and Flex and never downloads Flash kernels."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "Load the selected registry checkpoints from validated locally built Hub "
            "artifacts while retaining registry repo/revision report identities."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.junit_output is not None:
        _write_junit(
            arguments.junit_output,
            suite_name="benchmark-incomplete",
            failures=("Benchmark process did not complete",),
        )
    if arguments.backends is not None and len(set(arguments.backends)) != len(
        arguments.backends
    ):
        raise ValueError("--backends may not contain duplicates")
    local_files_only = arguments.local_files_only or arguments.artifact_root is not None
    if arguments.exhaustive:
        if arguments.baseline is not None:
            raise ValueError("Exhaustive sweeps are descriptive and cannot gate a baseline")
        cases = list(
            exhaustive_benchmark_cases(
                family=arguments.family,
                batch_sizes=arguments.exhaustive_batch_sizes,
                sequence_lengths=arguments.exhaustive_sequence_lengths,
                local_files_only=local_files_only,
                backends=arguments.backends,
            )
        )
    else:
        cases = list(
            benchmark_cases(
                family=arguments.family,
                quick=arguments.quick,
                local_files_only=local_files_only,
                backends=arguments.backends,
            )
        )
    artifact_identities: dict[str, dict[str, Any]] = {}
    if arguments.artifact_root is not None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        artifact_identities = bind_local_artifacts(cases, arguments.artifact_root)

    torch = _require_torch()
    environment = environment_fingerprint(torch)
    if not arguments.quick and not arguments.exhaustive:
        validate_hopper_sm90_environment(environment)
    report: dict[str, Any] = {
        "schema_version": 3,
        "status": "running",
        "environment": environment,
        "matrix_kind": (
            "exhaustive" if arguments.exhaustive else "quick" if arguments.quick else "fixed"
        ),
        "claim_scope": (
            "descriptive_only"
            if arguments.exhaustive
            else "smoke_only"
            if arguments.quick
            else "validated_hopper_sm90_exact_device"
        ),
        "artifact_load_mode": (
            "validated_local_build" if arguments.artifact_root is not None else "hub"
        ),
        "artifacts": artifact_identities,
        "backend_policy": {
            "requested": list(arguments.backends) if arguments.backends is not None else None,
            "selection": (
                "explicit_subset" if arguments.backends is not None else "manifest_all"
            ),
            "external_kernel_downloads": False,
            "external_kernel_builds": False,
        },
        "timing_contract": {
            "cold_compile_field": "results[].compile_ms",
            "first_forward_field": "results[].first_forward_ms",
            "warmup_field": "results[].warmup_samples_ms",
            "warm_throughput_field": "results[].blocks",
            "compile_amortized_into_throughput": False,
        },
        "baseline_promotion_contract": {
            "report_is_complete_when": "all cases are present and the process exits zero",
            "legacy_baseline_path": "benchmarks/baselines/h100.json",
            "requires_exact_environment_match": True,
            "requires_exact_artifact_inventory_match": True,
        },
        "expected_case_count": len(cases),
        "completed_case_count": 0,
        "results": [],
    }
    cached_key: tuple[str, str, str, str, str, str] | None = None
    cached_model: Any | None = None
    for case in cases:
        key = benchmark_model_key(case)
        reused = cached_model is not None and key == cached_key
        load_ms: float | None = None
        if not reused:
            if cached_model is not None:
                del cached_model
                gc.collect()
                torch.cuda.empty_cache()
            cached_model, load_ms = _load_model(case, torch)
            cached_key = key
        result = run_case(
            case,
            model=cached_model,
            load_ms=load_ms,
            model_reused=reused,
        )
        result.update(
            {
                "suite_profile": case.suite_profile,
                "dedicated_mode": case.dedicated_mode,
                "claim_eligible": case.claim_eligible,
                "matrix_kind": case.matrix_kind,
                "artifact": getattr(case, "artifact_identity", None),
                "artifact_dependencies": getattr(case, "artifact_dependencies", {}),
            }
        )
        report["results"].append(result)
        report["completed_case_count"] = len(report["results"])
        _write_report(arguments.output, report)
        gc.collect()
        torch.cuda.empty_cache()

    report["status"] = "complete"
    _write_report(arguments.output, report)
    if arguments.baseline is None:
        if arguments.junit_output is not None:
            _write_junit(arguments.junit_output, suite_name="benchmark-capture")
        return 0
    gate = compare_reports(report, _load_report(arguments.baseline))
    gate_output = arguments.gate_output or arguments.output.with_suffix(".gate.json")
    _write_report(gate_output, gate.to_dict())
    if arguments.junit_output is not None:
        _write_junit(
            arguments.junit_output,
            suite_name="benchmark-regression",
            failures=_gate_failures(gate),
        )
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
