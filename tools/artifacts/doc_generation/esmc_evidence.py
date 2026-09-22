"""Load and validate complete, identity-bound ESMC release report sets."""

from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fastplms.registry import ModelRegistry, ModelSpec
from tools.remote.biohub_reference_environment import (
    BiohubReferenceEnvironmentError,
    validate_biohub_reference_environment_evidence,
)


class EsmcReportError(ValueError):
    """Raised when frozen ESMC release evidence is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class EsmcRuntimeIdentity:
    """Artifact-derived runtime identity required by every ESMC report."""

    runtime_revision: str
    source_tree_sha256: str
    runtime_bundle_sha256: str


@dataclass(frozen=True, slots=True)
class EsmcReportSet:
    """One complete, cross-device-consistent 30-record ESMC evidence set."""

    reports: tuple[dict[str, object], ...]
    runtime_identity: EsmcRuntimeIdentity
    candidate_environment: dict[str, object]
    reference_environment: dict[str, object]

    def select(self, model_id: str) -> tuple[dict[str, object], ...]:
        """Return the ten backend/panel reports for one checkpoint."""

        return tuple(report for report in self.reports if report["model_id"] == model_id)

    def get(self, model_id: str, backend: str, panel: str) -> dict[str, object]:
        """Return one uniquely keyed report from the complete evidence set."""

        matches = tuple(
            report
            for report in self.reports
            if report["model_id"] == model_id
            and report["configured_backend"] == backend
            and isinstance(report["panel"], Mapping)
            and report["panel"]["kind"] == panel
        )
        if len(matches) != 1:
            raise EsmcReportError(
                f"ESMC evidence key {(model_id, backend, panel)!r} resolved to "
                f"{len(matches)} reports"
            )
        return matches[0]


ESMC_DIAGNOSTIC_SCHEMA_VERSION = 3


ESMC_MODEL_IDS = ("esmc_small", "esmc_large", "esmc_6b")


ESMC_PANEL_KINDS = ("generated_kernel_boundary", "real_biological_holdout")


ESMC_REFERENCE_SOURCE_NAMES = ("biohub-esm", "biohub-transformers")


ESMC_BACKENDS = (
    "eager",
    "sdpa",
    "flex_attention",
    "flash_attention_2",
    "flash_attention_3",
)


ESMC_MEASURED_BACKENDS = ("eager", "sdpa", "flex_attention")


ESMC_UNAVAILABLE_BACKENDS = ("flash_attention_2", "flash_attention_3")


ESMC_REPORT_COUNT = len(ESMC_MODEL_IDS) * len(ESMC_BACKENDS) * len(ESMC_PANEL_KINDS)


ESMC_REPORT_MAX_BYTES = 16 * 1024 * 1024


ESMC_RELEASE_GATE_MODES = {
    "sdpa": "exact",
    "eager": "strict_numeric",
    "flex_attention": "diagnostic_with_catastrophe_gate",
}


ESMC_CATASTROPHE_UPPER = {
    "relative_l2": 0.25,
    "relative_q999": 0.50,
}


ESMC_CATASTROPHE_LOWER = {
    "residue_cosine_p01": 0.90,
    "pooled_cosine_min": 0.95,
}


ESMC_TOP_LEVEL_FIELDS = {
    "schema_version",
    "model_id",
    "candidate",
    "reference",
    "record_status",
    "unavailability",
    "configured_backend",
    "effective_backend",
    "dtype",
    "panel",
    "environment",
    "kernel",
    "panel_tensor_metrics",
    "panel_logits_metrics",
    "cases",
    "published_band_violations",
    "catastrophic_gate",
    "release_gate",
    "report_sha256",
}


def _esmc_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EsmcReportError(f"ESMC JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _esmc_reject_json_constant(value: str) -> object:
    raise EsmcReportError(f"ESMC JSON contains non-finite constant {value!r}")


def _esmc_decode_json(encoded: str, *, context: str) -> dict[str, object]:
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_esmc_json_object,
            parse_constant=_esmc_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise EsmcReportError(f"{context} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise EsmcReportError(f"{context} must contain one JSON object")
    return payload


def _esmc_read_json(path: Path) -> dict[str, object]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise EsmcReportError(f"Unable to stat ESMC evidence file {path}: {error}") from error
    if size <= 0 or size > ESMC_REPORT_MAX_BYTES:
        raise EsmcReportError(
            f"ESMC evidence file {path.name!r} has invalid size {size}; "
            f"maximum is {ESMC_REPORT_MAX_BYTES} bytes"
        )
    try:
        encoded = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EsmcReportError(f"Unable to read ESMC evidence file {path}: {error}") from error
    return _esmc_decode_json(encoded, context=f"ESMC evidence file {path.name!r}")


def _esmc_require_mapping(
    value: object,
    fields: set[str],
    *,
    context: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise EsmcReportError(f"{context} fields differ from schema v3")
    return value


def _esmc_require_object(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EsmcReportError(f"{context} must be a JSON object")
    return value


def _esmc_require_list(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise EsmcReportError(f"{context} must be a JSON array")
    return value


def _esmc_require_text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EsmcReportError(f"{context} must be a nonempty string")
    return value


def _esmc_require_sha256(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EsmcReportError(f"{context} must be a canonical lowercase SHA-256 digest")
    return value


def _esmc_require_finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EsmcReportError(f"{context} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise EsmcReportError(f"{context} must be a finite number")
    return numeric


def _esmc_require_gpu_capability(
    value: object,
    *,
    context: str,
) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise EsmcReportError(f"{context} must contain exactly two integers")
    major, minor = value
    if (
        isinstance(major, bool)
        or not isinstance(major, int)
        or major < 0
        or isinstance(minor, bool)
        or not isinstance(minor, int)
        or minor < 0
    ):
        raise EsmcReportError(f"{context} must contain exactly two non-negative integers")
    return major, minor


def _esmc_report_sha256(payload: Mapping[str, object]) -> str:
    digest_payload = dict(payload)
    digest_payload.pop("report_sha256", None)
    encoded = json.dumps(
        digest_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _esmc_public_case(case: Mapping[str, object]) -> dict[str, object]:
    return {
        "case_id": case["case_id"],
        "sequence_length": case["sequence_length"],
        "sequence_sha256": case["sequence_sha256"],
        "source": case.get("source"),
        "source_sha256": case.get("source_sha256"),
    }


def _esmc_panel_identity(kind: str, cases: list[dict[str, object]]) -> dict[str, object]:
    definition = {
        "schema_version": 1,
        "kind": kind,
        "seed": 42,
        "cases": cases,
    }
    definition_sha256 = hashlib.sha256(
        json.dumps(definition, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "kind": kind,
        "seed": 42,
        "definition_sha256": definition_sha256,
        "cases": [_esmc_public_case(case) for case in cases],
    }


def _expected_esmc_panels(source_root: Path) -> dict[str, dict[str, object]]:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    generator = random.Random(42)
    generated_cases: list[dict[str, object]] = []
    for length in (13, 15, 16, 17, 29, 31, 32, 33, 61, 127, 128, 129):
        sequence = "M" + "".join(generator.choices(alphabet, k=length - 1))
        generated_cases.append(
            {
                "case_id": f"generated-boundary-{length}",
                "sequence": sequence,
                "sequence_length": length,
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            }
        )

    fixture_path = source_root / "tests" / "parity" / "fixtures" / "esmc_biological_holdout.json"
    try:
        fixture_text = fixture_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EsmcReportError(f"Unable to read immutable ESMC panel fixture: {error}") from error
    fixture = _esmc_decode_json(fixture_text, context="ESMC biological holdout fixture")
    if set(fixture) != {"schema_version", "cases"} or fixture["schema_version"] != 1:
        raise EsmcReportError("ESMC biological holdout fixture fields differ from schema v1")
    raw_cases = fixture["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EsmcReportError("ESMC biological holdout fixture has no ordered cases")
    biological_cases: list[dict[str, object]] = []
    for index, raw_case in enumerate(raw_cases):
        case = _esmc_require_mapping(
            raw_case,
            {"case_id", "sequence", "sequence_sha256", "source", "source_sha256"},
            context=f"ESMC biological holdout case {index}",
        )
        sequence = _esmc_require_text(
            case["sequence"], context=f"ESMC biological holdout case {index} sequence"
        )
        if not sequence.isupper() or not set(sequence).issubset(set(alphabet)):
            raise EsmcReportError(
                f"ESMC biological holdout case {index} is not canonical uppercase protein"
            )
        sequence_sha256 = _esmc_require_sha256(
            case["sequence_sha256"], context=f"ESMC biological holdout case {index} sequence"
        )
        if sequence_sha256 != hashlib.sha256(sequence.encode("ascii")).hexdigest():
            raise EsmcReportError(f"ESMC biological holdout case {index} sequence digest drifted")
        _esmc_require_sha256(
            case["source_sha256"], context=f"ESMC biological holdout case {index} source"
        )
        biological_cases.append({**case, "sequence_length": len(sequence)})

    return {
        "generated_kernel_boundary": _esmc_panel_identity(
            "generated_kernel_boundary", generated_cases
        ),
        "real_biological_holdout": _esmc_panel_identity(
            "real_biological_holdout", biological_cases
        ),
    }


def _expected_biohub_source_contracts(
    source_root: Path,
) -> dict[str, dict[str, object]]:
    expected_fields = {
        "import_name",
        "import_root",
        "package_version",
        "schema_version",
        "source_revision",
        "tree_sha256",
    }
    file_names = {
        "biohub-esm": "biohub-esm-source.json",
        "biohub-transformers": "biohub-transformers-source.json",
    }
    contracts: dict[str, dict[str, object]] = {}
    for source_name in ESMC_REFERENCE_SOURCE_NAMES:
        path = source_root / "docker" / "constraints" / file_names[source_name]
        try:
            encoded = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise EsmcReportError(
                f"Unable to read pinned {source_name} source contract: {error}"
            ) from error
        contract = _esmc_decode_json(encoded, context=f"{source_name} source contract")
        if set(contract) != expected_fields or contract["schema_version"] != 1:
            raise EsmcReportError(f"{source_name} source contract differs from schema v1")
        _esmc_require_text(contract["import_name"], context=f"{source_name} import name")
        _esmc_require_text(contract["import_root"], context=f"{source_name} import root")
        _esmc_require_text(contract["package_version"], context=f"{source_name} package version")
        revision = contract["source_revision"]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or revision != revision.lower()
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise EsmcReportError(
                f"{source_name} source revision is not canonical lowercase 40-hex"
            )
        _esmc_require_sha256(contract["tree_sha256"], context=f"{source_name} tree")
        contracts[source_name] = contract
    return contracts


def _validate_esmc_reference_sources(
    value: object,
    expected_contracts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    sources = _esmc_require_mapping(
        value,
        set(ESMC_REFERENCE_SOURCE_NAMES),
        context="ESMC reference sources",
    )
    source_fields = {
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
    for source_name in ESMC_REFERENCE_SOURCE_NAMES:
        source = _esmc_require_mapping(
            sources[source_name],
            source_fields,
            context=f"ESMC reference source {source_name}",
        )
        expected_contract = expected_contracts[source_name]
        if source["schema_version"] != 1:
            raise EsmcReportError(f"ESMC reference source {source_name} schema is unsupported")
        for name in (
            "source_revision",
            "tree_sha256",
            "import_name",
            "import_root",
            "package_version",
        ):
            if source[name] != expected_contract[name]:
                raise EsmcReportError(
                    f"ESMC reference source {source_name} {name} differs from the pin"
                )
        _esmc_require_sha256(
            source["attestation_sha256"],
            context=f"ESMC reference source {source_name} attestation",
        )
        file_count = source["file_count"]
        if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count <= 0:
            raise EsmcReportError(f"ESMC reference source {source_name} file count is invalid")
        expected_import_file = f"{expected_contract['import_root']}/__init__.py"
        if source["import_file"] != expected_import_file:
            raise EsmcReportError(
                f"ESMC reference source {source_name} import file differs from its root"
            )
    return sources


def _esmc_runtime_identity_from_source(
    source_root: Path,
    registry: ModelRegistry,
) -> EsmcRuntimeIdentity:
    try:
        from tools.artifacts.build import (
            _render_runtime_bundle,
            _validated_runtime_snapshot,
            _write_runtime_snapshot,
        )

        identities: set[tuple[str, str, str]] = set()
        with tempfile.TemporaryDirectory(prefix="fastplms-esmc-runtime-") as directory:
            temporary_root = Path(directory)
            for spec in (registry[model_id] for model_id in ESMC_MODEL_IDS):
                runtime_revision, payloads, source_tree_sha256 = _validated_runtime_snapshot(
                    source_root,
                    registry,
                    spec,
                )
                package_root = temporary_root / spec.id / "fastplms"
                _write_runtime_snapshot(package_root, payloads)
                runtime_bundle_sha256, _ = _render_runtime_bundle(package_root)
                identities.add((runtime_revision, source_tree_sha256, runtime_bundle_sha256))
    except Exception as error:
        raise EsmcReportError(
            "Unable to derive the clean tracked ESMC runtime identity required for release "
            f"evidence: {error}"
        ) from error
    if len(identities) != 1:
        raise EsmcReportError(
            "ESMC checkpoints do not resolve to one shared runtime/source/bundle identity"
        )
    runtime_revision, source_tree_sha256, runtime_bundle_sha256 = identities.pop()
    return EsmcRuntimeIdentity(
        runtime_revision=runtime_revision,
        source_tree_sha256=source_tree_sha256,
        runtime_bundle_sha256=runtime_bundle_sha256,
    )


def _validate_esmc_runtime_identity(identity: EsmcRuntimeIdentity) -> None:
    source_digest = _esmc_require_sha256(
        identity.source_tree_sha256, context="ESMC expected source tree"
    )
    _esmc_require_sha256(identity.runtime_bundle_sha256, context="ESMC expected runtime bundle")
    revision = identity.runtime_revision
    is_git_revision = (
        isinstance(revision, str)
        and len(revision) == 40
        and revision == revision.lower()
        and all(character in "0123456789abcdef" for character in revision)
    )
    if not is_git_revision and revision != f"source-tree-sha256:{source_digest}":
        raise EsmcReportError(
            "ESMC runtime revision must be a clean 40-hex Git revision or the exact "
            "source-tree-sha256 fallback"
        )


def _esmc_runtime_platform_identity(
    reference_environment: Mapping[str, object],
) -> tuple[str, str]:
    runtime = _esmc_require_object(
        reference_environment.get("runtime"),
        context="ESMC locked reference runtime",
    )
    operating_system = _esmc_require_text(
        runtime.get("operating_system"), context="ESMC locked operating system"
    )
    architecture = _esmc_require_text(
        runtime.get("architecture"), context="ESMC locked architecture"
    )
    gpu = _esmc_require_object(runtime.get("gpu"), context="ESMC locked GPU identity")
    gpu_name = _esmc_require_text(gpu.get("name"), context="ESMC locked GPU name")
    capability = gpu.get("capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in capability
        )
    ):
        raise EsmcReportError("ESMC locked GPU capability is malformed")
    return (
        f"{operating_system.lower()}/{architecture.lower()}",
        f"{gpu_name}/SM{capability[0]}{capability[1]}",
    )


def _esmc_unavailability_identity(
    backend: str,
    reference_environment: Mapping[str, object],
) -> dict[str, str]:
    platform_identity, accelerator_identity = _esmc_runtime_platform_identity(reference_environment)
    if backend == "flash_attention_2":
        historical_evidence = "separate_historical_focused_evidence_only"
        reason = (
            f"The locked {platform_identity} {accelerator_identity} release environment "
            "has no validated FlashAttention 2 "
            "kernel. Prior focused execution evidence is historical and is not part of "
            "the current ESMC release distribution."
        )
    elif backend == "flash_attention_3":
        historical_evidence = "none"
        reason = (
            "The manifest-pinned FlashAttention 3 kernel has no validated artifact for "
            f"the locked {platform_identity} {accelerator_identity} release environment."
        )
    else:
        raise EsmcReportError(f"ESMC backend {backend!r} is not a structured unavailable backend")
    return {
        "code": "locked_platform_kernel_unavailable",
        "platform": platform_identity,
        "accelerator": accelerator_identity,
        "dispatch_contract": "fail_closed_without_dispatch",
        "historical_evidence": historical_evidence,
        "reason": reason,
    }


def _validate_esmc_candidate_environment(value: object) -> dict[str, object]:
    environment = _esmc_require_mapping(
        value,
        {
            "python",
            "torch",
            "transformers",
            "cuda_runtime",
            "cuda_driver",
            "gpu",
            "packages",
        },
        context="ESMC candidate environment",
    )
    for name in ("python", "torch", "transformers", "cuda_runtime", "cuda_driver"):
        _esmc_require_text(environment[name], context=f"ESMC candidate environment {name}")
    packages = _esmc_require_mapping(
        environment["packages"],
        {
            "fastplms",
            "huggingface-hub",
            "kernels",
            "tokenizers",
            "transformer-engine",
            "transformer-engine-torch",
        },
        context="ESMC candidate package inventory",
    )
    for name, version in packages.items():
        if version is not None:
            _esmc_require_text(version, context=f"ESMC candidate package {name}")
    for required in ("fastplms", "huggingface-hub", "kernels", "tokenizers"):
        if packages[required] is None:
            raise EsmcReportError(f"ESMC candidate package {required!r} is unavailable")
    gpu = _esmc_require_mapping(
        environment["gpu"],
        {"name", "capability", "total_memory_bytes"},
        context="ESMC candidate GPU identity",
    )
    _esmc_require_text(gpu["name"], context="ESMC candidate GPU name")
    capability = gpu["capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in capability
        )
    ):
        raise EsmcReportError("ESMC candidate GPU capability is invalid")
    memory = gpu["total_memory_bytes"]
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
        raise EsmcReportError("ESMC candidate GPU memory identity is invalid")
    return environment


def _validate_esmc_reference_environment(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EsmcReportError("ESMC reference environment is missing")
    required = {
        "cuda_device",
        "cuda_device_capability",
        "cuda_total_memory",
        "cuda_runtime",
        "packages",
        "python",
        "torch",
    }
    if not required.issubset(value):
        raise EsmcReportError("ESMC reference environment fields are incomplete")
    for name in ("cuda_device", "cuda_runtime", "python", "torch"):
        _esmc_require_text(value[name], context=f"ESMC reference environment {name}")
    _esmc_require_text(value["cuda_device"], context="ESMC reference environment CUDA device")
    capability = value["cuda_device_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in capability
        )
    ):
        raise EsmcReportError("ESMC reference CUDA capability is invalid")
    memory = value["cuda_total_memory"]
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
        raise EsmcReportError("ESMC reference GPU memory identity is invalid")
    packages_text = _esmc_require_text(
        value["packages"], context="ESMC reference package inventory"
    )
    packages = _esmc_decode_json(packages_text, context="ESMC reference package inventory")
    if not isinstance(packages, dict):
        raise EsmcReportError("ESMC reference package inventory is not an object")
    return value


def _validate_locked_esmc_reference_environment(
    value: object,
    *,
    source_root: Path,
) -> dict[str, object]:
    try:
        return validate_biohub_reference_environment_evidence(
            value,
            repository_root=source_root,
            contract_path=source_root / "docker/constraints/biohub-reference-lock.json",
        )
    except BiohubReferenceEnvironmentError as error:
        raise EsmcReportError(f"ESMC locked reference environment is invalid: {error}") from error


def _validate_esmc_environment_binding(
    candidate_environment: Mapping[str, object],
    dynamic_reference_environment: Mapping[str, object],
    locked_reference_environment: Mapping[str, object],
) -> None:
    candidate_gpu = _esmc_require_object(
        candidate_environment.get("gpu"), context="ESMC candidate GPU binding"
    )
    locked_runtime = _esmc_require_object(
        locked_reference_environment.get("runtime"),
        context="ESMC locked reference runtime binding",
    )
    locked_gpu = _esmc_require_object(
        locked_runtime.get("gpu"), context="ESMC locked reference GPU binding"
    )
    candidate_identity = {
        "python": candidate_environment.get("python"),
        "torch": candidate_environment.get("torch"),
        "cuda_runtime": candidate_environment.get("cuda_runtime"),
        "cuda_driver": candidate_environment.get("cuda_driver"),
        "gpu": dict(candidate_gpu),
    }
    dynamic_reference_identity = {
        "python": dynamic_reference_environment.get("python"),
        "torch": dynamic_reference_environment.get("torch"),
        "cuda_runtime": dynamic_reference_environment.get("cuda_runtime"),
        "cuda_driver": candidate_environment.get("cuda_driver"),
        "gpu": {
            "name": dynamic_reference_environment.get("cuda_device"),
            "capability": dynamic_reference_environment.get("cuda_device_capability"),
            "total_memory_bytes": dynamic_reference_environment.get("cuda_total_memory"),
        },
    }
    locked_identity = {
        "python": locked_runtime.get("python_version"),
        "torch": locked_runtime.get("torch"),
        "cuda_runtime": locked_runtime.get("cuda_runtime"),
        "cuda_driver": locked_runtime.get("cuda_driver"),
        "gpu": dict(locked_gpu),
    }
    if candidate_identity != dynamic_reference_identity:
        raise EsmcReportError("ESMC candidate and native reference environments differ")
    if candidate_identity != locked_identity:
        raise EsmcReportError(
            "ESMC candidate environment differs from the locked reference runtime"
        )


def _validate_esmc_kernel(
    value: object,
    backend: str,
    environment: Mapping[str, object],
    registry: ModelRegistry,
) -> None:
    kernel_spec = registry.attention_kernels.get(backend)
    if kernel_spec is None:
        expected = {
            "implementation": backend,
            "provider": "torch",
            "torch_version": environment["torch"],
        }
    else:
        packages = _esmc_require_object(
            environment["packages"], context="ESMC candidate package inventory"
        )
        expected = {
            "implementation": backend,
            "provider": "huggingface_kernels",
            "repository": kernel_spec.repository,
            "revision": kernel_spec.revision,
            "version": kernel_spec.version,
            "expected_variant": kernel_spec.expected_variant,
            "supported_dtypes": list(kernel_spec.dtypes),
            "kernels_package_version": packages["kernels"],
        }
    if value != expected:
        raise EsmcReportError(
            f"ESMC {backend} kernel identity differs from the manifest/runtime contract"
        )


def _validate_esmc_logits_metrics(value: object, *, context: str) -> None:
    metrics = _esmc_require_mapping(
        value,
        {"confident_top1_agreement", "mean_jsd"},
        context=f"{context} logits metrics",
    )
    agreement = _esmc_require_finite(
        metrics["confident_top1_agreement"], context=f"{context} top-1 agreement"
    )
    mean_jsd = _esmc_require_finite(metrics["mean_jsd"], context=f"{context} mean JSD")
    if not 0.80 <= agreement <= 1.000001:
        raise EsmcReportError(f"{context} top-1 agreement fails the catastrophe gate")
    if not -1e-7 <= mean_jsd <= 0.05:
        raise EsmcReportError(f"{context} mean JSD fails the catastrophe gate")


def _validate_esmc_tensor_metrics(
    value: object,
    *,
    context: str,
    expected_metric_context: str,
) -> tuple[tuple[str, int | None], ...]:
    if not isinstance(value, list) or not value:
        raise EsmcReportError(f"{context} tensor metrics are missing")
    layout: list[tuple[str, int | None]] = []
    hidden_layers: list[int] = []
    output_counts = {"last_hidden_state": 0, "logits": 0}
    fields = {
        "context",
        "output",
        "layer_index",
        "relative_l2",
        "relative_q999",
        "residue_cosine_p01",
        "pooled_cosine_min",
    }
    for index, raw_metric in enumerate(value):
        metric = _esmc_require_mapping(
            raw_metric, fields, context=f"{context} tensor metric {index}"
        )
        if metric["context"] != expected_metric_context:
            raise EsmcReportError(f"{context} tensor metric context is stale or misaligned")
        output = metric["output"]
        layer_index = metric["layer_index"]
        if output == "hidden_state":
            if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
                raise EsmcReportError(f"{context} hidden-state layer index is invalid")
            hidden_layers.append(layer_index)
        elif output in output_counts:
            if layer_index is not None:
                raise EsmcReportError(f"{context} {output} layer index must be null")
            output_counts[output] += 1
        else:
            raise EsmcReportError(f"{context} tensor output {output!r} is unsupported")
        for metric_name, upper in ESMC_CATASTROPHE_UPPER.items():
            numeric = _esmc_require_finite(metric[metric_name], context=f"{context} {metric_name}")
            if not 0 <= numeric <= upper:
                raise EsmcReportError(f"{context} {metric_name} fails the catastrophe gate")
        for metric_name, lower in ESMC_CATASTROPHE_LOWER.items():
            numeric = _esmc_require_finite(metric[metric_name], context=f"{context} {metric_name}")
            if not lower <= numeric <= 1.000001:
                raise EsmcReportError(f"{context} {metric_name} fails the catastrophe gate")
        if not isinstance(output, str) or not (layer_index is None or isinstance(layer_index, int)):
            raise EsmcReportError(f"{context} tensor metric layout is invalid")
        layout.append((output, layer_index))
    if hidden_layers != list(range(len(hidden_layers))):
        raise EsmcReportError(f"{context} hidden-state layers are incomplete or unordered")
    if output_counts != {"last_hidden_state": 1, "logits": 1}:
        raise EsmcReportError(
            f"{context} must contain exactly one last-hidden-state and one logits metric"
        )
    return tuple(layout)


def _validate_esmc_report(
    payload: dict[str, object],
    *,
    spec: ModelSpec,
    backend: str,
    panel: str,
    expected_panel: Mapping[str, object],
    expected_reference_sources: Mapping[str, Mapping[str, object]],
    runtime_identity: EsmcRuntimeIdentity,
    registry: ModelRegistry,
    source_root: Path,
) -> None:
    if set(payload) != ESMC_TOP_LEVEL_FIELDS or (
        payload.get("schema_version") != ESMC_DIAGNOSTIC_SCHEMA_VERSION
    ):
        raise EsmcReportError("ESMC diagnostic fields differ from schema v3")
    report_sha256 = _esmc_require_sha256(
        payload["report_sha256"], context="ESMC report self-digest"
    )
    if report_sha256 != _esmc_report_sha256(payload):
        raise EsmcReportError("ESMC report self-digest does not match its canonical payload")
    if payload["model_id"] != spec.id or payload["dtype"] != "bfloat16":
        raise EsmcReportError("ESMC report model or dtype identity is stale")
    if payload["configured_backend"] != backend:
        raise EsmcReportError("ESMC configured backend identity is invalid")
    record_status = payload["record_status"]
    if record_status not in {"measured", "unavailable"}:
        raise EsmcReportError("ESMC record status is invalid")

    candidate = _esmc_require_mapping(
        payload["candidate"],
        {
            "repo_id",
            "manifest_revision",
            "resolved_commit",
            "checkpoint_repo_id",
            "checkpoint_revision",
            "weights_revision",
            "runtime_revision",
            "source_tree_sha256",
            "runtime_bundle_sha256",
        },
        context="ESMC candidate identity",
    )
    expected_candidate = {
        "repo_id": spec.fast.repo_id,
        "manifest_revision": spec.fast.revision,
        "checkpoint_repo_id": spec.artifact_checkpoint.repo_id,
        "checkpoint_revision": spec.artifact_checkpoint.revision,
        "weights_revision": spec.artifact_checkpoint.revision,
        "runtime_revision": runtime_identity.runtime_revision,
        "source_tree_sha256": runtime_identity.source_tree_sha256,
        "runtime_bundle_sha256": runtime_identity.runtime_bundle_sha256,
    }
    for name, expected in expected_candidate.items():
        if candidate[name] != expected:
            raise EsmcReportError(f"ESMC candidate {name} differs from frozen release identity")
    if candidate["resolved_commit"] not in {None, spec.fast.revision}:
        raise EsmcReportError("ESMC candidate resolved Hub commit is stale")

    reference = _esmc_require_mapping(
        payload["reference"],
        {
            "repo_id",
            "revision",
            "state_transform",
            "environment",
            "reference_environment",
            "reference_sources",
        },
        context="ESMC reference identity",
    )
    if (
        reference["repo_id"] != spec.official.repo_id
        or reference["revision"] != spec.official.revision
        or reference["state_transform"] != spec.family.state_transform
    ):
        raise EsmcReportError("ESMC reference identity differs from the pinned manifest")
    _validate_esmc_reference_sources(
        reference["reference_sources"],
        expected_reference_sources,
    )
    dynamic_reference_environment = _validate_esmc_reference_environment(reference["environment"])
    locked_reference_environment = _validate_locked_esmc_reference_environment(
        reference["reference_environment"], source_root=source_root
    )
    candidate_environment = _validate_esmc_candidate_environment(payload["environment"])
    _validate_esmc_environment_binding(
        candidate_environment,
        dynamic_reference_environment,
        locked_reference_environment,
    )
    _validate_esmc_kernel(payload["kernel"], backend, candidate_environment, registry)

    report_panel = payload["panel"]
    if report_panel != expected_panel:
        raise EsmcReportError(f"ESMC panel {panel!r} differs from its immutable definition")
    report_panel = _esmc_require_object(report_panel, context="ESMC panel identity")
    panel_cases = report_panel["cases"]
    cases = payload["cases"]
    if (
        not isinstance(panel_cases, list)
        or not isinstance(cases, list)
        or (len(cases) != len(panel_cases))
    ):
        raise EsmcReportError("ESMC panel and per-case metrics are not aligned")
    identity_fields = {
        "case_id",
        "sequence_length",
        "sequence_sha256",
        "source",
        "source_sha256",
    }
    violations = payload["published_band_violations"]
    if not isinstance(violations, list) or any(
        not isinstance(item, str) or not item.strip() for item in violations
    ):
        raise EsmcReportError("ESMC published-band violations must be a string list")
    release_gate = _esmc_require_mapping(
        payload["release_gate"], {"mode", "status"}, context="ESMC release gate"
    )
    if record_status == "unavailable":
        if backend not in ESMC_UNAVAILABLE_BACKENDS:
            raise EsmcReportError("Only locked Flash backends may be unavailable")
        if payload["effective_backend"] is not None:
            raise EsmcReportError("Unavailable ESMC records must not claim effective dispatch")
        if payload["unavailability"] != _esmc_unavailability_identity(
            backend, locked_reference_environment
        ):
            raise EsmcReportError("ESMC structured unavailability identity is invalid")
        if payload["catastrophic_gate"] != "not_run":
            raise EsmcReportError("Unavailable ESMC catastrophe gate must be not run")
        if release_gate != {"mode": "availability", "status": "unavailable"}:
            raise EsmcReportError("Unavailable ESMC release-gate identity is invalid")
        if (
            payload["panel_tensor_metrics"] is not None
            or payload["panel_logits_metrics"] is not None
            or violations
        ):
            raise EsmcReportError("Unavailable ESMC records must not contain measurements")
        if cases != panel_cases:
            raise EsmcReportError("Unavailable ESMC cases must be immutable panel identities only")
        return

    if backend not in ESMC_MEASURED_BACKENDS:
        raise EsmcReportError("Current frozen measurements are limited to eager, SDPA, and Flex")
    if payload["effective_backend"] != backend:
        raise EsmcReportError("ESMC effective backend identity is invalid or fell back")
    if payload["unavailability"] is not None:
        raise EsmcReportError("Measured ESMC records carry unavailability metadata")
    if payload["catastrophic_gate"] != "passed":
        raise EsmcReportError("Measured ESMC report catastrophe gate did not pass")
    if release_gate != {"mode": ESMC_RELEASE_GATE_MODES[backend], "status": "passed"}:
        raise EsmcReportError("Measured ESMC release-gate identity is invalid")

    metric_context = f"{spec.id}:bf16:{backend}:{panel}"
    panel_layout = _validate_esmc_tensor_metrics(
        payload["panel_tensor_metrics"],
        context="ESMC panel",
        expected_metric_context=metric_context,
    )
    _validate_esmc_logits_metrics(payload["panel_logits_metrics"], context="ESMC panel")
    for index, (panel_case, raw_case) in enumerate(zip(panel_cases, cases, strict=True)):
        case = _esmc_require_mapping(
            raw_case,
            identity_fields | {"tensor_metrics", "logits_metrics"},
            context=f"ESMC case {index}",
        )
        if not isinstance(panel_case, Mapping) or any(
            case[name] != panel_case[name] for name in identity_fields
        ):
            raise EsmcReportError(f"ESMC case {index} identity is misaligned with its panel")
        _esmc_require_sha256(case["sequence_sha256"], context=f"ESMC case {index} sequence")
        if case["source_sha256"] is not None:
            _esmc_require_sha256(case["source_sha256"], context=f"ESMC case {index} source")
        case_id = _esmc_require_text(case["case_id"], context=f"ESMC case {index} ID")
        case_layout = _validate_esmc_tensor_metrics(
            case["tensor_metrics"],
            context=f"ESMC case {case_id}",
            expected_metric_context=f"{metric_context}:case={case_id}",
        )
        if case_layout != panel_layout:
            raise EsmcReportError(f"ESMC case {case_id} metric layout differs from its panel")
        _validate_esmc_logits_metrics(case["logits_metrics"], context=f"ESMC case {case_id}")
    if backend in {"sdpa", "eager"} and violations:
        raise EsmcReportError(f"ESMC strict backend {backend} has published-band violations")


def load_esmc_report_set(
    report_root: Path,
    registry: ModelRegistry,
    *,
    source_root: Path | None = None,
    expected_runtime_identity: EsmcRuntimeIdentity | None = None,
) -> EsmcReportSet:
    """Load exactly 30 immutable schema-v3 records and fail closed on any drift."""

    source_root = (source_root or Path(__file__).resolve().parents[3]).resolve()
    expected_specs = tuple(spec.id for spec in registry.by_family("esm_plusplus"))
    if expected_specs != ESMC_MODEL_IDS:
        raise EsmcReportError(
            f"ESMC manifest inventory {expected_specs!r} differs from {ESMC_MODEL_IDS!r}"
        )
    family = registry.families["esm_plusplus"]
    supported_backends = tuple(
        backend
        for backend in family.attention
        if "bfloat16" in registry.supported_attention_dtypes(family.id, backend)
    )
    if supported_backends != ESMC_BACKENDS:
        raise EsmcReportError(
            f"ESMC BF16 backend inventory {supported_backends!r} differs from {ESMC_BACKENDS!r}"
        )
    runtime_identity = expected_runtime_identity or _esmc_runtime_identity_from_source(
        source_root, registry
    )
    _validate_esmc_runtime_identity(runtime_identity)
    panels = _expected_esmc_panels(source_root)
    expected_reference_sources = _expected_biohub_source_contracts(source_root)
    expected_names = {
        f"{model_id}-{backend}-{panel}.json"
        for model_id in ESMC_MODEL_IDS
        for backend in ESMC_BACKENDS
        for panel in ESMC_PANEL_KINDS
    }
    if report_root.is_symlink():
        raise EsmcReportError(f"ESMC report root must not be a symlink: {report_root}")
    report_root = report_root.resolve()
    if not report_root.exists() or not report_root.is_dir():
        raise EsmcReportError(f"ESMC report root is not a real directory: {report_root}")
    entries = tuple(report_root.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise EsmcReportError("ESMC report root contains a symlink or non-file entry")
    observed_names = {entry.name for entry in entries}
    if len(observed_names) != len(entries):
        raise EsmcReportError("ESMC report root contains duplicate path identities")
    if observed_names != expected_names:
        missing = sorted(expected_names.difference(observed_names))
        unexpected = sorted(observed_names.difference(expected_names))
        raise EsmcReportError(
            "ESMC release evidence must contain exactly 30 records; "
            f"missing={missing}, unexpected={unexpected}"
        )

    reports: list[dict[str, object]] = []
    for model_id in ESMC_MODEL_IDS:
        spec = registry[model_id]
        for backend in ESMC_BACKENDS:
            for panel in ESMC_PANEL_KINDS:
                path = report_root / f"{model_id}-{backend}-{panel}.json"
                payload = _esmc_read_json(path)
                _validate_esmc_report(
                    payload,
                    spec=spec,
                    backend=backend,
                    panel=panel,
                    expected_panel=panels[panel],
                    expected_reference_sources=expected_reference_sources,
                    runtime_identity=runtime_identity,
                    registry=registry,
                    source_root=source_root,
                )
                reports.append(payload)
    if len(reports) != ESMC_REPORT_COUNT:
        raise EsmcReportError(
            f"ESMC release evidence contains {len(reports)} validated records, expected 30"
        )
    measured_count = sum(report["record_status"] == "measured" for report in reports)
    unavailable_count = sum(report["record_status"] == "unavailable" for report in reports)
    if measured_count != 18 or unavailable_count != 12:
        raise EsmcReportError(
            "ESMC release evidence must contain exactly 18 measured and 12 structured "
            "unavailable records"
        )

    candidate_environments = {
        json.dumps(report["environment"], sort_keys=True) for report in reports
    }
    reference_environments = {
        json.dumps(report["reference"]["environment"], sort_keys=True)
        for report in reports
        if isinstance(report["reference"], Mapping)
    }
    locked_reference_environments = {
        json.dumps(report["reference"]["reference_environment"], sort_keys=True)
        for report in reports
        if isinstance(report["reference"], Mapping)
    }
    reference_sources = {
        json.dumps(report["reference"]["reference_sources"], sort_keys=True)
        for report in reports
        if isinstance(report["reference"], Mapping)
    }
    if (
        len(candidate_environments) != 1
        or len(reference_environments) != 1
        or len(locked_reference_environments) != 1
        or len(reference_sources) != 1
    ):
        raise EsmcReportError(
            "ESMC release evidence crosses candidate/reference devices, software "
            "environments, or source attestations"
        )
    candidate_environment = reports[0]["environment"]
    reference = reports[0]["reference"]
    if not isinstance(candidate_environment, dict):
        raise EsmcReportError("Validated ESMC candidate environment is not an object")
    if not isinstance(reference, dict):
        raise EsmcReportError("Validated ESMC reference identity is not an object")
    reference_environment = reference["reference_environment"]
    if not isinstance(reference_environment, dict):
        raise EsmcReportError("Validated ESMC reference environment is not an object")
    return EsmcReportSet(
        reports=tuple(reports),
        runtime_identity=runtime_identity,
        candidate_environment=candidate_environment,
        reference_environment=reference_environment,
    )
