"""Strict schema-v3 ESMC evidence with warning-only backend bands."""

from __future__ import annotations

import copy
import json
import pytest
import torch
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

from fastplms.registry import get_model_registry
from tests.parity import test_native_results as diagnostics
from tests.parity.support.esmc_calibration import (
    ESMC_CALIBRATION_SEED,
    esmc_calibration_batches,
    validate_esmc_calibration_batch,
)
from tests.unit.test_biohub_reference_lock import _reference_environment_payload
from tools.remote.prepare_references import _esmc_calibration_batches


SPEC = get_model_registry()["esmc_small"]
ADVERTISED_BF16_BACKENDS = tuple(
    backend
    for backend in SPEC.family.attention
    if "bfloat16"
    in get_model_registry().supported_attention_dtypes(
        SPEC.family.id,
        backend,
    )
)
PANEL_KINDS = ("generated_kernel_boundary", "real_biological_holdout")
SOURCE_TREE_SHA256 = "1" * 64
RUNTIME_BUNDLE_SHA256 = "2" * 64
REFERENCE_SOURCES: dict[str, dict[str, object]] = {
    "biohub-esm": {
        "attestation_sha256": "a" * 64,
        "file_count": 412,
        "import_file": "esm/__init__.py",
        "import_name": "esm",
        "import_root": "esm",
        "package_version": "3.3.0",
        "schema_version": 1,
        "source_revision": diagnostics.BIOHUB_ESM_REVISION,
        "tree_sha256": diagnostics.BIOHUB_ESM_TREE_SHA256,
    },
    "biohub-transformers": {
        "attestation_sha256": "b" * 64,
        "file_count": 5218,
        "import_file": "src/transformers/__init__.py",
        "import_name": "transformers",
        "import_root": "src/transformers",
        "package_version": "4.57.6",
        "schema_version": 1,
        "source_revision": diagnostics.BIOHUB_TRANSFORMERS_REVISION,
        "tree_sha256": diagnostics.BIOHUB_TRANSFORMERS_TREE_SHA256,
    },
}


def _output(hidden: torch.Tensor) -> SimpleNamespace:
    signal = hidden[..., :1] * 8
    logits = torch.cat((signal, -signal), dim=-1)
    return SimpleNamespace(
        hidden_states=(hidden * 0.5, hidden),
        last_hidden_state=hidden,
        logits=logits,
    )


def _batch(kind: str) -> dict[str, object]:
    return copy.deepcopy(
        next(batch for batch in esmc_calibration_batches() if batch["kind"] == kind)
    )


def _panel_tensors(
    kind: str,
    *,
    candidate_scale: float,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = _batch(kind)
    cases = batch["cases"]
    assert isinstance(cases, list)
    lengths = torch.tensor([int(case["sequence_length"]) for case in cases])  # (b,)
    maximum = int(lengths.max().item())
    residue_mask = torch.arange(maximum).unsqueeze(0) < lengths.unsqueeze(1)  # (b, l)
    reference = torch.ones(len(cases), maximum, 4)  # (b, l, d=4)
    candidate = reference * candidate_scale
    return batch, candidate, reference, residue_mask


def _candidate_model(
    runtime_revision: str = f"source-tree-sha256:{SOURCE_TREE_SHA256}",
) -> SimpleNamespace:
    config = SimpleNamespace(
        fastplms_model_id=SPEC.id,
        fastplms_checkpoint_repo_id=SPEC.artifact_checkpoint.repo_id,
        fastplms_checkpoint_revision=SPEC.artifact_checkpoint.revision,
        fastplms_weights_revision=SPEC.artifact_checkpoint.revision,
        fastplms_runtime_revision=runtime_revision,
        fastplms_source_tree_sha256=SOURCE_TREE_SHA256,
        fastplms_runtime_bundle_sha256=RUNTIME_BUNDLE_SHA256,
        _commit_hash=SPEC.fast.revision,
    )
    return SimpleNamespace(config=config)


def _reference_metadata() -> dict[str, object]:
    locked_environment = _reference_environment_payload()
    runtime = locked_environment["runtime"]
    if not isinstance(runtime, dict):
        raise AssertionError("Synthetic locked reference runtime is malformed")
    gpu = runtime["gpu"]
    if not isinstance(gpu, dict):
        raise AssertionError("Synthetic locked reference GPU is malformed")
    return {
        "reference_repo_id": SPEC.official.repo_id,
        "reference_revision": SPEC.official.revision,
        "state_transform": SPEC.family.state_transform,
        "reference_sources": copy.deepcopy(REFERENCE_SOURCES),
        "reference_environment": locked_environment,
        "environment": {
            "cuda_device": gpu["name"],
            "cuda_device_capability": copy.deepcopy(gpu["capability"]),
            "cuda_total_memory": gpu["total_memory_bytes"],
            "cuda_runtime": runtime["cuda_runtime"],
            "packages": json.dumps(
                {"python": runtime["python_version"], "torch": runtime["torch"]},
                separators=(",", ":"),
                sort_keys=True,
            ),
            "python": runtime["python_version"],
            "torch": runtime["torch"],
        },
    }


def _candidate_environment() -> dict[str, object]:
    locked_environment = _reference_environment_payload()
    runtime = locked_environment["runtime"]
    if not isinstance(runtime, dict):
        raise AssertionError("Synthetic locked reference runtime is malformed")
    return {
        "python": runtime["python_version"],
        "torch": runtime["torch"],
        "transformers": "5.13.0",
        "cuda_runtime": runtime["cuda_runtime"],
        "cuda_driver": runtime["cuda_driver"],
        "gpu": copy.deepcopy(runtime["gpu"]),
        "packages": {
            "fastplms": "1.0.0",
            "huggingface-hub": "1.4.0",
            "kernels": "0.12.2",
            "tokenizers": "0.22.2",
            "transformer-engine": None,
            "transformer-engine-torch": None,
        },
    }


def _patch_candidate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = _candidate_environment()
    monkeypatch.setattr(
        diagnostics,
        "_candidate_environment_identity",
        lambda: copy.deepcopy(environment),
    )


def _build_report(
    *,
    backend: str,
    kind: str,
    candidate_scale: float = 1.0,
    runtime_revision: str = f"source-tree-sha256:{SOURCE_TREE_SHA256}",
) -> dict[str, object]:
    batch, candidate, reference, residue_mask = _panel_tensors(
        kind,
        candidate_scale=candidate_scale,
    )
    return diagnostics._build_esmc_diagnostic_report(
        SPEC,
        _output(candidate),
        _output(reference),
        residue_mask,
        backend=backend,
        effective_backend=backend,
        context=f"{SPEC.id}:bf16:{backend}:{kind}",
        calibration_batch=batch,
        model=_candidate_model(runtime_revision),
        reference_metadata=_reference_metadata(),
    )


def test_esmc_schema_v3_partitions_every_advertised_bf16_backend() -> None:
    assert (*diagnostics.ESMC_MEASURED_BACKENDS, *diagnostics.ESMC_UNAVAILABLE_BACKENDS) == (
        ADVERTISED_BF16_BACKENDS
    )


@pytest.mark.parametrize(
    ("gpu_name", "architecture", "memory"),
    (
        ("NVIDIA H100 80GB HBM3", "x86_64", 80 * 1024**3),
        ("NVIDIA H200", "x86_64", 141 * 1024**3),
        ("NVIDIA GH200 480GB", "aarch64", 480_000_000_000),
    ),
)
def test_esmc_dynamic_environment_schema_is_hardware_neutral_and_exactly_bound(
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setattr(
        diagnostics,
        "_candidate_environment_identity",
        lambda: copy.deepcopy(candidate),
    )
    diagnostics._validate_candidate_environment(candidate)

    dynamic_reference = {
        "cuda_device": gpu_name,
        "cuda_device_capability": [9, 0],
        "cuda_total_memory": memory,
        "cuda_runtime": candidate["cuda_runtime"],
        "packages": json.dumps({"torch": candidate["torch"]}),
        "python": candidate["python"],
        "torch": candidate["torch"],
    }
    diagnostics._validate_reference_environment(dynamic_reference)
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
    diagnostics._validate_esmc_environment_binding(
        candidate,
        dynamic_reference,
        locked_reference,
    )
    unavailable = diagnostics._esmc_unavailability_identity("flash_attention_3", locked_reference)
    assert unavailable["platform"] == f"linux/{architecture}"
    assert unavailable["accelerator"] == f"{gpu_name}/SM90"


def test_esmc_environment_binding_rejects_mismatch_and_malformed_capability() -> None:
    candidate = _candidate_environment()
    dynamic_reference = _reference_metadata()["environment"]
    locked_reference = _reference_environment_payload()
    if not isinstance(dynamic_reference, dict):
        raise AssertionError("Synthetic dynamic reference environment is malformed")
    diagnostics._validate_esmc_environment_binding(
        candidate,
        dynamic_reference,
        locked_reference,
    )

    mismatched = copy.deepcopy(dynamic_reference)
    mismatched["cuda_device"] = "NVIDIA H100 80GB HBM3"
    with pytest.raises(ValueError, match="native reference environments differ"):
        diagnostics._validate_esmc_environment_binding(
            candidate,
            mismatched,
            locked_reference,
        )

    malformed = copy.deepcopy(candidate)
    malformed_gpu = malformed["gpu"]
    if not isinstance(malformed_gpu, dict):
        raise AssertionError("Synthetic candidate GPU is malformed")
    malformed_gpu["capability"] = [True, 0]
    with pytest.raises(ValueError, match="GPU identity is malformed"):
        diagnostics._validate_candidate_environment(malformed)


@pytest.mark.parametrize("backend", diagnostics.ESMC_MEASURED_BACKENDS)
@pytest.mark.parametrize("kind", PANEL_KINDS)
def test_esmc_schema_v3_covers_every_measured_bf16_backend_and_panel(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    kind: str,
) -> None:
    _patch_candidate_environment(monkeypatch)
    report = _build_report(backend=backend, kind=kind)

    assert report["schema_version"] == 3
    assert report["model_id"] == SPEC.id
    assert report["configured_backend"] == backend
    assert report["effective_backend"] == backend
    assert report["record_status"] == "measured"
    assert report["unavailability"] is None
    assert report["dtype"] == "bfloat16"
    assert report["catastrophic_gate"] == "passed"
    assert report["release_gate"] == {
        "mode": {
            "sdpa": "exact",
            "eager": "strict_numeric",
            "flex_attention": "diagnostic_with_catastrophe_gate",
        }[backend],
        "status": "passed",
    }
    assert report["environment"] == _candidate_environment()
    assert report["candidate"] == {
        "repo_id": SPEC.fast.repo_id,
        "manifest_revision": SPEC.fast.revision,
        "resolved_commit": SPEC.fast.revision,
        "checkpoint_repo_id": SPEC.artifact_checkpoint.repo_id,
        "checkpoint_revision": SPEC.artifact_checkpoint.revision,
        "weights_revision": SPEC.artifact_checkpoint.revision,
        "runtime_revision": f"source-tree-sha256:{SOURCE_TREE_SHA256}",
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "runtime_bundle_sha256": RUNTIME_BUNDLE_SHA256,
    }
    assert report["reference"] == diagnostics._reference_identity(
        SPEC,
        _reference_metadata(),
    )
    assert report["reference"]["reference_sources"] == REFERENCE_SOURCES
    panel = report["panel"]
    assert isinstance(panel, Mapping)
    assert panel["kind"] == kind
    assert panel["seed"] == ESMC_CALIBRATION_SEED
    assert len(str(panel["definition_sha256"])) == 64
    cases = report["cases"]
    panel_cases = panel["cases"]
    assert isinstance(cases, list)
    assert isinstance(panel_cases, list)
    assert len(cases) == len(panel_cases)
    assert report["published_band_violations"] == []
    assert report["report_sha256"] == diagnostics._report_sha256(report)
    diagnostics._validate_esmc_diagnostic_report(report, SPEC)

    panel_metrics = report["panel_tensor_metrics"]
    assert isinstance(panel_metrics, list)
    assert [(metric["output"], metric["layer_index"]) for metric in panel_metrics] == [
        ("hidden_state", 0),
        ("hidden_state", 1),
        ("last_hidden_state", None),
        ("logits", None),
    ]
    for case, panel_case in zip(cases, panel_cases, strict=True):
        assert {
            name: case[name]
            for name in (
                "case_id",
                "sequence_length",
                "sequence_sha256",
                "source",
                "source_sha256",
            )
        } == panel_case
        assert len(case["sequence_sha256"]) == 64
        assert [(metric["output"], metric["layer_index"]) for metric in case["tensor_metrics"]] == [
            ("hidden_state", 0),
            ("hidden_state", 1),
            ("last_hidden_state", None),
            ("logits", None),
        ]
        assert set(case["logits_metrics"]) == {
            "confident_top1_agreement",
            "mean_jsd",
        }


@pytest.mark.parametrize("backend", diagnostics.ESMC_UNAVAILABLE_BACKENDS)
@pytest.mark.parametrize("kind", PANEL_KINDS)
def test_esmc_schema_v3_records_locked_flash_unavailability_without_metrics(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    kind: str,
) -> None:
    _patch_candidate_environment(monkeypatch)
    report = diagnostics._build_esmc_unavailable_report(
        SPEC,
        backend=backend,
        calibration_batch=_batch(kind),
        model=_candidate_model(),
        reference_metadata=_reference_metadata(),
    )

    assert report["record_status"] == "unavailable"
    assert report["configured_backend"] == backend
    assert report["effective_backend"] is None
    assert report["panel_tensor_metrics"] is None
    assert report["panel_logits_metrics"] is None
    assert report["catastrophic_gate"] == "not_run"
    assert report["release_gate"] == {"mode": "availability", "status": "unavailable"}
    reference = report["reference"]
    assert isinstance(reference, Mapping)
    locked_environment = reference["reference_environment"]
    assert isinstance(locked_environment, Mapping)
    assert report["unavailability"] == diagnostics._esmc_unavailability_identity(
        backend, locked_environment
    )
    assert report["cases"] == report["panel"]["cases"]
    diagnostics._validate_esmc_diagnostic_report(report, SPEC)


@pytest.mark.parametrize(
    "runtime_revision",
    (
        "a" * 40,
        f"source-tree-sha256:{SOURCE_TREE_SHA256}",
    ),
    ids=("clean-git-revision", "content-addressed-fallback"),
)
def test_esmc_accepts_both_artifact_runtime_revision_forms(
    monkeypatch: pytest.MonkeyPatch,
    runtime_revision: str,
) -> None:
    _patch_candidate_environment(monkeypatch)
    report = _build_report(
        backend="sdpa",
        kind="generated_kernel_boundary",
        runtime_revision=runtime_revision,
    )

    candidate = report["candidate"]
    assert isinstance(candidate, Mapping)
    assert candidate["runtime_revision"] == runtime_revision
    diagnostics._validate_esmc_diagnostic_report(
        report,
        SPEC,
        expected_candidate=candidate,
    )


@pytest.mark.parametrize(
    "runtime_revision",
    (
        "main",
        "A" * 40,
        "source-tree-sha256:" + "f" * 64,
    ),
    ids=("symbolic-ref", "noncanonical-git", "wrong-source-digest"),
)
def test_esmc_rejects_runtime_revision_not_emitted_by_artifact_builder(
    runtime_revision: str,
) -> None:
    with pytest.raises(ValueError, match="clean Git revision or the exact source-tree"):
        diagnostics._candidate_identity(
            SPEC,
            _candidate_model(runtime_revision),
        )


def test_esmc_report_must_match_validated_artifact_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_candidate_environment(monkeypatch)
    report = _build_report(
        backend="sdpa",
        kind="real_biological_holdout",
        runtime_revision="a" * 40,
    )
    expected_candidate = copy.deepcopy(report["candidate"])
    report["candidate"]["runtime_revision"] = "b" * 40
    report["report_sha256"] = diagnostics._report_sha256(report)

    with pytest.raises(ValueError, match="validated artifact identity"):
        diagnostics._validate_esmc_diagnostic_report(
            report,
            SPEC,
            expected_candidate=expected_candidate,
        )


def test_native_result_reference_identity_raises_explicitly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / SPEC.id
    result_dir.mkdir()
    metadata = _reference_metadata()
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setenv("FASTPLMS_REFERENCE_RESULTS", str(tmp_path))

    observed, directory = diagnostics._result(SPEC)
    assert observed == metadata
    assert directory == result_dir

    metadata["reference_repo_id"] = "untrusted/reference"
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reference_repo_id"):
        diagnostics._result(SPEC)

    metadata["reference_repo_id"] = SPEC.official.repo_id
    metadata["reference_sources"]["biohub-transformers"]["tree_sha256"] = "0" * 64
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reference source evidence tree_sha256"):
        diagnostics._result(SPEC)


@pytest.mark.parametrize(
    ("backend", "kind"),
    (("flex_attention", "generated_kernel_boundary"),),
)
def test_esmc_supported_backend_deviation_warns_and_writes_complete_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    kind: str,
) -> None:
    _patch_candidate_environment(monkeypatch)
    monkeypatch.setenv("FASTPLMS_DIAGNOSTIC_REPORTS", str(tmp_path))
    batch, candidate, reference, residue_mask = _panel_tensors(
        kind,
        candidate_scale=1.04,
    )

    with pytest.warns(
        UserWarning,
        match=(
            rf"configured backend={backend}, effective backend={backend}.*"
            r"outside the published ESMC backend bands"
        ),
    ) as diagnostic_warnings:
        diagnostics._assert_and_record_esmc_diagnostic(
            SPEC,
            _output(candidate),
            _output(reference),
            residue_mask,
            backend=backend,
            effective_backend=backend,
            context=f"{SPEC.id}:bf16:{backend}:{kind}",
            calibration_batch=batch,
            model=_candidate_model(),
            reference_metadata=_reference_metadata(),
        )

    assert len(diagnostic_warnings) == 1
    report_path = tmp_path / f"{SPEC.id}-{backend}-{kind}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    diagnostics._validate_esmc_diagnostic_report(report, SPEC)
    assert report["published_band_violations"]
    assert any("case=" in violation for violation in report["published_band_violations"])
    assert report["kernel"]["implementation"] == backend
    assert report["kernel"]["provider"] == (
        "torch" if backend == "flex_attention" else "huggingface_kernels"
    )


def test_esmc_catastrophic_disagreement_remains_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_candidate_environment(monkeypatch)
    monkeypatch.setenv("FASTPLMS_DIAGNOSTIC_REPORTS", str(tmp_path))
    kind = "real_biological_holdout"
    batch, _, reference, residue_mask = _panel_tensors(
        kind,
        candidate_scale=1.0,
    )
    candidate = torch.zeros_like(reference)  # (b, l, d)

    with pytest.raises(AssertionError, match="relative_l2"):
        diagnostics._assert_and_record_esmc_diagnostic(
            SPEC,
            _output(candidate),
            _output(reference),
            residue_mask,
            backend="flex_attention",
            effective_backend="flex_attention",
            context=f"{SPEC.id}:bf16:flex_attention:{kind}",
            calibration_batch=batch,
            model=_candidate_model(),
            reference_metadata=_reference_metadata(),
        )
    assert not tuple(tmp_path.glob("*.json"))


def test_esmc_immutable_panels_fail_closed_on_drift() -> None:
    seed_drift = _batch("generated_kernel_boundary")
    seed_drift["seed"] = ESMC_CALIBRATION_SEED + 1
    with pytest.raises(ValueError, match="seed differs"):
        validate_esmc_calibration_batch(seed_drift)

    sequence_drift = _batch("real_biological_holdout")
    cases = sequence_drift["cases"]
    assert isinstance(cases, list)
    cases[0]["sequence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from the release contract"):
        validate_esmc_calibration_batch(sequence_drift)


def test_reference_request_uses_the_shared_immutable_panels() -> None:
    assert _esmc_calibration_batches() == list(esmc_calibration_batches())
    for batch in _esmc_calibration_batches():
        assert validate_esmc_calibration_batch(batch)["definition_sha256"]


def test_esmc_schema_rejects_stale_identity_and_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_candidate_environment(monkeypatch)
    report = _build_report(
        backend="flex_attention",
        kind="generated_kernel_boundary",
    )

    stale = copy.deepcopy(report)
    stale["candidate"]["weights_revision"] = "f" * 40
    stale["report_sha256"] = diagnostics._report_sha256(stale)
    with pytest.raises(ValueError, match="weights_revision differs"):
        diagnostics._validate_esmc_diagnostic_report(stale, SPEC)

    fallback = copy.deepcopy(report)
    fallback["effective_backend"] = "sdpa"
    fallback["report_sha256"] = diagnostics._report_sha256(fallback)
    with pytest.raises(ValueError, match="fallback"):
        diagnostics._validate_esmc_diagnostic_report(fallback, SPEC)

    panel_drift = copy.deepcopy(report)
    panel_drift["panel"]["definition_sha256"] = "0" * 64
    panel_drift["report_sha256"] = diagnostics._report_sha256(panel_drift)
    with pytest.raises(ValueError, match="immutable definition"):
        diagnostics._validate_esmc_diagnostic_report(panel_drift, SPEC)

    source_drift = copy.deepcopy(report)
    source_drift["reference"]["reference_sources"]["biohub-esm"]["tree_sha256"] = "0" * 64
    source_drift["report_sha256"] = diagnostics._report_sha256(source_drift)
    with pytest.raises(ValueError, match="reference source evidence tree_sha256"):
        diagnostics._validate_esmc_diagnostic_report(source_drift, SPEC)

    nonfinite = copy.deepcopy(report)
    nonfinite["panel_tensor_metrics"][0]["relative_l2"] = float("nan")
    nonfinite["report_sha256"] = diagnostics._report_sha256(nonfinite)
    with pytest.raises(ValueError, match="finite number"):
        diagnostics._validate_esmc_diagnostic_report(nonfinite, SPEC)

    digest_mismatch = copy.deepcopy(report)
    digest_mismatch["panel_tensor_metrics"][0]["context"] += ":tampered"
    with pytest.raises(ValueError, match="digest does not match"):
        diagnostics._validate_esmc_diagnostic_report(digest_mismatch, SPEC)


def test_esmc_report_write_is_atomic_idempotent_and_no_clobber(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_candidate_environment(monkeypatch)
    monkeypatch.setenv("FASTPLMS_DIAGNOSTIC_REPORTS", str(tmp_path))
    report = _build_report(
        backend="sdpa",
        kind="generated_kernel_boundary",
    )

    path = diagnostics._write_esmc_diagnostic_report(SPEC, report)
    assert diagnostics._write_esmc_diagnostic_report(SPEC, report) == path

    different = copy.deepcopy(report)
    different["panel_tensor_metrics"][0]["context"] += ":second-run"
    different["report_sha256"] = diagnostics._report_sha256(different)
    with pytest.raises(RuntimeError, match="Refusing to replace different ESMC evidence"):
        diagnostics._write_esmc_diagnostic_report(SPEC, different)
    assert len(tuple(tmp_path.glob("*.json"))) == 1
    assert not tuple(tmp_path.glob("*.tmp"))


def test_esmc_calibration_contains_no_expected_failures() -> None:
    source = (Path(__file__).resolve().parents[1] / "parity" / "test_native_results.py").read_text(
        encoding="utf-8"
    )
    assert "pytest.mark.xfail" not in source
