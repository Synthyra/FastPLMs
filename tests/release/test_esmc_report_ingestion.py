"""Fail-closed schema-v3 ESMC documentation evidence contracts."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from fastplms.registry import ModelRegistry, ModelSpec, get_model_registry
from tests.unit.test_biohub_reference_lock import _reference_environment_payload
from tools.artifacts import generate_docs
from tools.artifacts.generate_docs import (
    ESMC_BACKENDS,
    ESMC_MODEL_IDS,
    ESMC_PANEL_KINDS,
    EsmcReportError,
    EsmcRuntimeIdentity,
    load_esmc_report_set,
    render_capability_evidence,
    render_model_card,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = get_model_registry()
RUNTIME_IDENTITY = EsmcRuntimeIdentity(
    runtime_revision=f"source-tree-sha256:{'1' * 64}",
    source_tree_sha256="1" * 64,
    runtime_bundle_sha256="2" * 64,
)
LOCKED_REFERENCE_ENVIRONMENT = _reference_environment_payload()
LOCKED_RUNTIME = LOCKED_REFERENCE_ENVIRONMENT["runtime"]
if not isinstance(LOCKED_RUNTIME, dict) or not isinstance(LOCKED_RUNTIME.get("gpu"), dict):
    raise RuntimeError("Synthetic locked Biohub environment fixture is malformed")
GPU = copy.deepcopy(LOCKED_RUNTIME["gpu"])
REFERENCE_SOURCES: dict[str, dict[str, object]] = {
    "biohub-esm": {
        "attestation_sha256": "a" * 64,
        "file_count": 412,
        "import_file": "esm/__init__.py",
        "import_name": "esm",
        "import_root": "esm",
        "package_version": "3.3.0",
        "schema_version": 1,
        "source_revision": "82ee35553d39169d678f784c8d3f8712ffd7d2c4",
        "tree_sha256": "c5489f1fc58de200978803de2c38e1a78f769cb183a2ee90be833f0f4a0212e8",
    },
    "biohub-transformers": {
        "attestation_sha256": "b" * 64,
        "file_count": 5218,
        "import_file": "src/transformers/__init__.py",
        "import_name": "transformers",
        "import_root": "src/transformers",
        "package_version": "4.57.6",
        "schema_version": 1,
        "source_revision": "3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf",
        "tree_sha256": "28b910cc18b821870db2fb6d1c50376c2d14287ae18485080699e03fa4ba4f43",
    },
}


def _candidate_environment() -> dict[str, object]:
    return {
        "python": LOCKED_RUNTIME["python_version"],
        "torch": LOCKED_RUNTIME["torch"],
        "transformers": "5.13.0",
        "cuda_runtime": LOCKED_RUNTIME["cuda_runtime"],
        "cuda_driver": LOCKED_RUNTIME["cuda_driver"],
        "gpu": copy.deepcopy(GPU),
        "packages": {
            "fastplms": "1.0.0",
            "huggingface-hub": "1.4.0",
            "kernels": "0.12.2",
            "tokenizers": "0.22.2",
            "transformer-engine": None,
            "transformer-engine-torch": None,
        },
    }


def _reference_environment() -> dict[str, object]:
    return {
        "cuda_device": GPU["name"],
        "cuda_device_capability": copy.deepcopy(GPU["capability"]),
        "cuda_total_memory": GPU["total_memory_bytes"],
        "cuda_runtime": LOCKED_RUNTIME["cuda_runtime"],
        "packages": json.dumps(
            {
                "python": LOCKED_RUNTIME["python_version"],
                "torch": LOCKED_RUNTIME["torch"],
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        "python": LOCKED_RUNTIME["python_version"],
        "torch": LOCKED_RUNTIME["torch"],
    }


def _kernel(registry: ModelRegistry, backend: str) -> dict[str, object]:
    kernel = registry.attention_kernels.get(backend)
    if kernel is None:
        return {
            "implementation": backend,
            "provider": "torch",
            "torch_version": LOCKED_RUNTIME["torch"],
        }
    return {
        "implementation": backend,
        "provider": "huggingface_kernels",
        "repository": kernel.repository,
        "revision": kernel.revision,
        "version": kernel.version,
        "expected_variant": kernel.expected_variant,
        "supported_dtypes": list(kernel.dtypes),
        "kernels_package_version": "0.12.2",
    }


def _tensor_metrics(context: str, base: float) -> list[dict[str, object]]:
    result = []
    for output, layer_index, offset in (
        ("hidden_state", 0, 0.0),
        ("last_hidden_state", None, 0.00001),
        ("logits", None, 0.00002),
    ):
        value = base + offset
        result.append(
            {
                "context": context,
                "output": output,
                "layer_index": layer_index,
                "relative_l2": value,
                "relative_q999": value * 2,
                "residue_cosine_p01": 1.0 - value,
                "pooled_cosine_min": 1.0 - value / 2,
            }
        )
    return result


def _logits_metrics(base: float) -> dict[str, float]:
    return {
        "confident_top1_agreement": 1.0 - base,
        "mean_jsd": base / 10,
    }


def _report(
    registry: ModelRegistry,
    spec: ModelSpec,
    backend: str,
    panel: Mapping[str, object],
) -> dict[str, object]:
    panel_kind = str(panel["kind"])
    is_unavailable = backend in generate_docs.ESMC_UNAVAILABLE_BACKENDS
    model_offset = ESMC_MODEL_IDS.index(spec.id) * 0.001
    backend_offset = ESMC_BACKENDS.index(backend) * 0.0001
    panel_offset = ESMC_PANEL_KINDS.index(panel_kind) * 0.00001
    base = 0.001 + model_offset + backend_offset + panel_offset
    context = f"{spec.id}:bf16:{backend}:{panel_kind}"
    panel_cases = panel["cases"]
    assert isinstance(panel_cases, list)
    measured_cases = []
    for index, panel_case in enumerate(panel_cases):
        assert isinstance(panel_case, Mapping)
        case_base = base + index * 0.000001
        case_id = str(panel_case["case_id"])
        measured_cases.append(
            {
                **panel_case,
                "tensor_metrics": _tensor_metrics(f"{context}:case={case_id}", case_base),
                "logits_metrics": _logits_metrics(case_base),
            }
        )
    release_modes = {
        "sdpa": "exact",
        "eager": "strict_numeric",
        "flex_attention": "diagnostic_with_catastrophe_gate",
    }
    payload: dict[str, object] = {
        "schema_version": 3,
        "model_id": spec.id,
        "candidate": {
            "repo_id": spec.fast.repo_id,
            "manifest_revision": spec.fast.revision,
            "resolved_commit": spec.fast.revision,
            "checkpoint_repo_id": spec.artifact_checkpoint.repo_id,
            "checkpoint_revision": spec.artifact_checkpoint.revision,
            "weights_revision": spec.artifact_checkpoint.revision,
            "runtime_revision": RUNTIME_IDENTITY.runtime_revision,
            "source_tree_sha256": RUNTIME_IDENTITY.source_tree_sha256,
            "runtime_bundle_sha256": RUNTIME_IDENTITY.runtime_bundle_sha256,
        },
        "reference": {
            "repo_id": spec.official.repo_id,
            "revision": spec.official.revision,
            "state_transform": spec.family.state_transform,
            "environment": _reference_environment(),
            "reference_environment": copy.deepcopy(LOCKED_REFERENCE_ENVIRONMENT),
            "reference_sources": copy.deepcopy(REFERENCE_SOURCES),
        },
        "record_status": "unavailable" if is_unavailable else "measured",
        "unavailability": (
            generate_docs._esmc_unavailability_identity(backend, LOCKED_REFERENCE_ENVIRONMENT)
            if is_unavailable
            else None
        ),
        "configured_backend": backend,
        "effective_backend": None if is_unavailable else backend,
        "dtype": "bfloat16",
        "panel": copy.deepcopy(panel),
        "environment": _candidate_environment(),
        "kernel": _kernel(registry, backend),
        "panel_tensor_metrics": None if is_unavailable else _tensor_metrics(context, base),
        "panel_logits_metrics": None if is_unavailable else _logits_metrics(base),
        "cases": copy.deepcopy(panel_cases) if is_unavailable else measured_cases,
        "published_band_violations": [],
        "catastrophic_gate": "not_run" if is_unavailable else "passed",
        "release_gate": (
            {"mode": "availability", "status": "unavailable"}
            if is_unavailable
            else {"mode": release_modes[backend], "status": "passed"}
        ),
    }
    payload["report_sha256"] = generate_docs._esmc_report_sha256(payload)
    return payload


@pytest.fixture
def complete_report_root(tmp_path: Path) -> Path:
    report_root = tmp_path / "reports"
    report_root.mkdir()
    panels = generate_docs._expected_esmc_panels(ROOT)
    for model_id in ESMC_MODEL_IDS:
        spec = REGISTRY[model_id]
        for backend in ESMC_BACKENDS:
            for panel_kind in ESMC_PANEL_KINDS:
                payload = _report(REGISTRY, spec, backend, panels[panel_kind])
                path = report_root / f"{model_id}-{backend}-{panel_kind}.json"
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    return report_root


def _rewrite_report(
    path: Path,
    mutate: Callable[[dict[str, object]], None],
    *,
    rehash: bool = True,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    if rehash:
        payload["report_sha256"] = generate_docs._esmc_report_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(report_root: Path):
    return load_esmc_report_set(
        report_root,
        REGISTRY,
        source_root=ROOT,
        expected_runtime_identity=RUNTIME_IDENTITY,
    )


@pytest.mark.parametrize(
    ("gpu_name", "architecture", "memory"),
    (
        ("NVIDIA H100 80GB HBM3", "x86_64", 80 * 1024**3),
        ("NVIDIA H200", "x86_64", 141 * 1024**3),
        ("NVIDIA GH200 480GB", "aarch64", 480_000_000_000),
    ),
)
def test_generator_dynamic_environment_schema_is_hardware_neutral(
    gpu_name: str,
    architecture: str,
    memory: int,
) -> None:
    candidate = _candidate_environment()
    gpu = {
        "name": gpu_name,
        "capability": [9, 0],
        "total_memory_bytes": memory,
    }
    candidate["gpu"] = copy.deepcopy(gpu)
    dynamic_reference = _reference_environment()
    dynamic_reference["cuda_device"] = gpu_name
    dynamic_reference["cuda_device_capability"] = [9, 0]
    dynamic_reference["cuda_total_memory"] = memory
    locked_reference = {
        "runtime": {
            "operating_system": "linux",
            "architecture": architecture,
            "python_version": candidate["python"],
            "torch": candidate["torch"],
            "cuda_runtime": candidate["cuda_runtime"],
            "cuda_driver": candidate["cuda_driver"],
            "gpu": copy.deepcopy(gpu),
        }
    }

    validated_candidate = generate_docs._validate_esmc_candidate_environment(candidate)
    validated_reference = generate_docs._validate_esmc_reference_environment(dynamic_reference)
    generate_docs._validate_esmc_environment_binding(
        validated_candidate,
        validated_reference,
        locked_reference,
    )
    unavailable = generate_docs._esmc_unavailability_identity("flash_attention_2", locked_reference)
    assert unavailable["platform"] == f"linux/{architecture}"
    assert unavailable["accelerator"] == f"{gpu_name}/SM90"


def test_complete_esmc_report_set_renders_checkpoint_specific_measurements(
    complete_report_root: Path,
) -> None:
    evidence = _load(complete_report_root)

    assert len(evidence.reports) == 30
    assert sum(report["record_status"] == "measured" for report in evidence.reports) == 18
    assert sum(report["record_status"] == "unavailable" for report in evidence.reports) == 12
    assert len(evidence.select("esmc_small")) == 10
    small = render_model_card(REGISTRY["esmc_small"], esmc_evidence=evidence)
    large = render_model_card(REGISTRY["esmc_large"], esmc_evidence=evidence)
    six_b = render_model_card(REGISTRY["esmc_6b"], esmc_evidence=evidence)
    manifest = render_capability_evidence(REGISTRY, esmc_evidence=evidence)

    assert "validated complete set (30/30 records)" in manifest
    assert "NVIDIA GH200 480GB" in small
    assert REFERENCE_SOURCES["biohub-esm"]["attestation_sha256"] in small
    assert REFERENCE_SOURCES["biohub-transformers"]["attestation_sha256"] in small
    assert "Locked-platform unavailable backends" in small
    assert "no validated artifact" in small
    assert "Per-case distributions" in small
    assert "0.001 to 0.00102" in small
    assert "0.002 to 0.00202" in large
    assert "0.003 to 0.00302" in six_b
    assert "ESMC-6B Flex Attention exceeds" not in small
    assert "ESMC-6B Flex Attention exceeds" not in large


def test_default_generation_stays_explicitly_pending_and_ignores_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FASTPLMS_DIAGNOSTIC_REPORTS", str(tmp_path / "partial"))

    card = render_model_card(REGISTRY["esmc_small"])
    manifest = render_capability_evidence(REGISTRY)

    assert "Pending release measurement" in card
    assert "Status: pending" in manifest
    assert "ESMC-6B Flex Attention exceeds" not in card


def test_missing_esmc_report_fails_closed(complete_report_root: Path) -> None:
    (complete_report_root / "esmc_6b-flash_attention_3-real_biological_holdout.json").unlink()

    with pytest.raises(EsmcReportError, match="exactly 30 records"):
        _load(complete_report_root)


@pytest.mark.parametrize(
    ("injected", "message"),
    (
        ('"schema_version": 3,', "duplicate key"),
        ('"nonfinite": NaN,', "non-finite constant"),
    ),
)
def test_non_strict_esmc_json_fails_before_schema_validation(
    complete_report_root: Path,
    injected: str,
    message: str,
) -> None:
    path = complete_report_root / "esmc_small-eager-generated_kernel_boundary.json"
    encoded = path.read_text(encoding="utf-8")
    path.write_text("{\n  " + injected + encoded[1:], encoding="utf-8")

    with pytest.raises(EsmcReportError, match=message):
        _load(complete_report_root)


def test_esmc_schema_validation_remains_fail_closed_under_python_optimized_mode(
    complete_report_root: Path,
) -> None:
    path = complete_report_root / "esmc_small-sdpa-generated_kernel_boundary.json"
    _rewrite_report(
        path,
        lambda report: report.__setitem__("panel_tensor_metrics", {"invalid": "mapping"}),
    )
    script = f"""
from pathlib import Path
from fastplms.registry import get_model_registry
from tools.artifacts.generate_docs import EsmcReportError, EsmcRuntimeIdentity, load_esmc_report_set

try:
    load_esmc_report_set(
        Path({str(complete_report_root)!r}),
        get_model_registry(),
        source_root=Path({str(ROOT)!r}),
        expected_runtime_identity=EsmcRuntimeIdentity(
            runtime_revision={RUNTIME_IDENTITY.runtime_revision!r},
            source_tree_sha256={RUNTIME_IDENTITY.source_tree_sha256!r},
            runtime_bundle_sha256={RUNTIME_IDENTITY.runtime_bundle_sha256!r},
        ),
    )
except EsmcReportError as error:
    if "tensor metrics are missing" not in str(error):
        raise
else:
    raise SystemExit("optimized-mode validation accepted malformed ESMC evidence")
"""
    environment = os.environ.copy()
    import_roots = (str(ROOT), str(ROOT / "src"))
    environment["PYTHONPATH"] = os.pathsep.join(
        (*import_roots, environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize(
    ("mutate", "rehash", "message"),
    (
        (
            lambda report: report["candidate"].__setitem__("weights_revision", "0" * 40),
            True,
            "weights_revision",
        ),
        (
            lambda report: report["candidate"].__setitem__("source_tree_sha256", "3" * 64),
            True,
            "source_tree_sha256",
        ),
        (
            lambda report: report["candidate"].__setitem__("runtime_revision", "4" * 40),
            True,
            "runtime_revision",
        ),
        (
            lambda report: report.__setitem__("effective_backend", "sdpa"),
            True,
            "backend identity",
        ),
        (
            lambda report: report.__setitem__("dtype", "float32"),
            True,
            "dtype identity",
        ),
        (
            lambda report: report["panel"].__setitem__("definition_sha256", "5" * 64),
            True,
            "immutable definition",
        ),
        (
            lambda report: report.__setitem__("unexpected", True),
            True,
            "schema v3",
        ),
        (
            lambda report: report["reference"]["reference_sources"][
                "biohub-transformers"
            ].__setitem__("tree_sha256", "6" * 64),
            True,
            "reference source biohub-transformers tree_sha256",
        ),
        (
            lambda report: report["reference"]["reference_environment"]["runtime"][
                "gpu"
            ].__setitem__("name", "forged accelerator"),
            True,
            "locked reference environment is invalid",
        ),
        (
            lambda report: report["panel_logits_metrics"].__setitem__("mean_jsd", 0.06),
            True,
            "catastrophe gate",
        ),
        (
            lambda report: report.__setitem__("catastrophic_gate", "failed"),
            True,
            "catastrophe gate",
        ),
        (
            lambda report: report.__setitem__("report_sha256", "f" * 64),
            False,
            "self-digest",
        ),
    ),
)
def test_stale_malformed_or_tampered_esmc_report_fails_closed(
    complete_report_root: Path,
    mutate: Callable[[dict[str, object]], None],
    rehash: bool,
    message: str,
) -> None:
    path = complete_report_root / "esmc_small-flex_attention-generated_kernel_boundary.json"
    _rewrite_report(path, mutate, rehash=rehash)

    with pytest.raises(EsmcReportError, match=message):
        _load(complete_report_root)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda report: report.__setitem__("effective_backend", "flash_attention_3"),
            "must not claim effective dispatch",
        ),
        (
            lambda report: report.__setitem__("panel_tensor_metrics", []),
            "must not contain measurements",
        ),
        (
            lambda report: report["unavailability"].__setitem__(
                "dispatch_contract", "silent_fallback"
            ),
            "unavailability identity",
        ),
    ),
)
def test_flash_unavailability_records_fail_closed_on_false_execution_claims(
    complete_report_root: Path,
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    path = complete_report_root / "esmc_small-flash_attention_3-generated_kernel_boundary.json"
    _rewrite_report(path, mutate)

    with pytest.raises(EsmcReportError, match=message):
        _load(complete_report_root)


def test_cross_device_esmc_report_set_fails_closed(complete_report_root: Path) -> None:
    path = complete_report_root / "esmc_large-sdpa-real_biological_holdout.json"

    def mutate(report: dict[str, object]) -> None:
        environment = report["environment"]
        reference = report["reference"]
        assert isinstance(environment, dict)
        assert isinstance(reference, dict)
        gpu = environment["gpu"]
        reference_environment = reference["environment"]
        assert isinstance(gpu, dict)
        assert isinstance(reference_environment, dict)
        gpu["name"] = "NVIDIA GH200 96GB"
        gpu["total_memory_bytes"] = 95 * 1024**3
        reference_environment["cuda_device"] = gpu["name"]
        reference_environment["cuda_total_memory"] = gpu["total_memory_bytes"]

    _rewrite_report(path, mutate)

    with pytest.raises(EsmcReportError, match="locked reference runtime"):
        _load(complete_report_root)


def test_cli_explicit_report_root_renders_only_after_complete_validation(
    complete_report_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "rendered"
    fixture = source_root / "tests" / "parity" / "fixtures" / "esmc_biological_holdout.json"
    fixture.parent.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "tests" / "parity" / "fixtures" / "esmc_biological_holdout.json",
        fixture,
    )
    constraints = source_root / "docker" / "constraints"
    constraints.mkdir(parents=True)
    for name in (
        "biohub-esm-source.json",
        "biohub-transformers-source.json",
        "biohub-reference-lock.json",
        "biohub-reference.in",
        "biohub-reference.lock.txt",
        "biohub-biotraj-build.in",
        "biohub-biotraj-build.lock.txt",
    ):
        shutil.copyfile(ROOT / "docker" / "constraints" / name, constraints / name)
    shutil.copyfile(
        ROOT / "docker" / "biohub-reference-lock.Dockerfile",
        source_root / "docker" / "biohub-reference-lock.Dockerfile",
    )
    monkeypatch.setattr(
        generate_docs,
        "_esmc_runtime_identity_from_source",
        lambda source, registry: RUNTIME_IDENTITY,
    )

    result = generate_docs.main(
        (
            "--source-root",
            str(source_root),
            "--esmc-report-root",
            str(complete_report_root),
        )
    )

    assert result == 0
    card = (source_root / "model_cards" / "esmc_small.md").read_text(encoding="utf-8")
    assert "NVIDIA GH200 480GB" in card
    assert "Per-case distributions" in card


@pytest.mark.parametrize("use_environment", (False, True))
def test_cli_release_evidence_options_fail_closed_on_missing_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    use_environment: bool,
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setattr(
        generate_docs,
        "_esmc_runtime_identity_from_source",
        lambda source, registry: RUNTIME_IDENTITY,
    )
    arguments = ["--source-root", str(ROOT)]
    if use_environment:
        monkeypatch.setenv("FASTPLMS_DIAGNOSTIC_REPORTS", str(missing))
        arguments.append("--require-esmc-release-evidence")
    else:
        arguments.extend(("--esmc-report-root", str(missing)))

    assert generate_docs.main(arguments) == 1
    assert "invalid ESMC release evidence" in capsys.readouterr().out
