"""Offline CPU contracts for local benchmark artifact identity binding."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import pytest
from pathlib import Path
from types import SimpleNamespace

import benchmarks.suite as benchmark_suite
from benchmarks.suite import benchmark_cases, bind_local_artifacts
from fastplms.registry import ModelSpec, get_model_registry


_RUNTIME_REVISION = "1" * 40
_SOURCE_SHA256 = "2" * 64
_RUNTIME_BUNDLE_SHA256 = "3" * 64


def _write_identity_artifact(root: Path, spec: ModelSpec) -> Path:
    path = root / spec.fast.repo_id.rsplit("/", maxsplit=1)[1]
    path.mkdir(parents=True)
    config = {
        "fastplms_model_id": spec.id,
        "fastplms_checkpoint_repo_id": spec.artifact_checkpoint.repo_id,
        "fastplms_checkpoint_revision": spec.artifact_checkpoint.revision,
        "fastplms_weights_revision": spec.artifact_checkpoint.revision,
        "fastplms_runtime_revision": _RUNTIME_REVISION,
        "fastplms_source_tree_sha256": _SOURCE_SHA256,
        "fastplms_runtime_bundle_sha256": _RUNTIME_BUNDLE_SHA256,
    }
    provenance = {
        "model_id": spec.id,
        "artifact_checkpoint": {
            "repo_id": spec.artifact_checkpoint.repo_id,
            "revision": spec.artifact_checkpoint.revision,
        },
        "weights_revision": spec.artifact_checkpoint.revision,
        "runtime_revision": _RUNTIME_REVISION,
        "source_tree_sha256": _SOURCE_SHA256,
        "runtime_bundle_sha256": _RUNTIME_BUNDLE_SHA256,
        "canonical_weights": {
            "state_digest": {
                "schema_version": 1,
                "algorithm": "sha256",
                "sha256": "4" * 64,
            }
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "source-record.json").write_text(json.dumps(provenance), encoding="utf-8")
    (path / "artifact-manifest.json").write_text(
        json.dumps({"config.json": "sha256:" + "5" * 64}),
        encoding="utf-8",
    )
    return path


def _stub_complete_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark_suite, "_validate_built_artifact", lambda *_args: None)
    monkeypatch.setattr(
        benchmark_suite,
        "_frozen_runtime_identity",
        lambda *_args: (_RUNTIME_REVISION, _SOURCE_SHA256),
    )


def test_local_benchmark_artifact_identity_is_path_free_and_registry_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_complete_validation(monkeypatch)
    spec = get_model_registry()["esm2_8m"]
    artifact = _write_identity_artifact(tmp_path, spec)
    cases = list(benchmark_cases(family="esm2", quick=True, local_files_only=False))

    identities = bind_local_artifacts(cases, tmp_path, source_root=tmp_path)

    case = cases[0]
    assert case.model == spec.fast.repo_id
    assert case.revision == spec.fast.revision
    assert case.load_model == artifact.resolve()
    assert case.load_revision is None
    assert case.local_files_only is True
    assert identities[spec.id]["runtime_revision"] == _RUNTIME_REVISION
    assert identities[spec.id]["weights_revision"] == spec.artifact_checkpoint.revision
    assert str(tmp_path) not in json.dumps(identities, sort_keys=True)


@pytest.mark.parametrize(
    ("field", "stale_value"),
    (
        ("fastplms_model_id", "esm2_35m"),
        ("fastplms_runtime_revision", "6" * 40),
        ("fastplms_source_tree_sha256", "7" * 64),
    ),
)
def test_local_benchmark_artifact_rejects_swapped_or_stale_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    stale_value: str,
) -> None:
    _stub_complete_validation(monkeypatch)
    spec = get_model_registry()["esm2_8m"]
    artifact = _write_identity_artifact(tmp_path, spec)
    config_path = artifact / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = stale_value
    config_path.write_text(json.dumps(config), encoding="utf-8")
    cases = list(benchmark_cases(family="esm2", quick=True, local_files_only=True))

    with pytest.raises(ValueError, match="registry/frozen source"):
        bind_local_artifacts(cases, tmp_path, source_root=tmp_path)


def test_local_benchmark_artifact_rejects_linked_root_and_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_complete_validation(monkeypatch)
    spec = get_model_registry()["esm2_8m"]
    actual_root = tmp_path / "actual"
    actual_root.mkdir()
    _write_identity_artifact(actual_root, spec)
    cases = list(benchmark_cases(family="esm2", quick=True, local_files_only=True))

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(actual_root, target_is_directory=True)
    with pytest.raises(ValueError, match="link or junction"):
        bind_local_artifacts(cases, linked_root, source_root=tmp_path)

    linked_child_root = tmp_path / "linked-child-root"
    linked_child_root.mkdir()
    child = linked_child_root / spec.fast.repo_id.rsplit("/", maxsplit=1)[1]
    child.symlink_to(actual_root / child.name, target_is_directory=True)
    with pytest.raises(ValueError, match="Missing or invalid selected benchmark artifacts"):
        bind_local_artifacts(cases, linked_child_root, source_root=tmp_path)


def test_local_benchmark_artifact_propagates_complete_validator_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_complete_validation(monkeypatch)
    spec = get_model_registry()["esm2_8m"]
    _write_identity_artifact(tmp_path, spec)
    cases = list(benchmark_cases(family="esm2", quick=True, local_files_only=True))
    monkeypatch.setattr(
        benchmark_suite,
        "_validate_built_artifact",
        lambda *_args: (_ for _ in ()).throw(ValueError("digest mismatch")),
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        bind_local_artifacts(cases, tmp_path, source_root=tmp_path)


def test_prepublication_artifact_root_benchmark_command_is_documented() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative_name in ("docs/benchmarking.md", "benchmarks/README.md"):
        text = (root / relative_name).read_text(encoding="utf-8")
        assert "tools.artifacts.build_all" in text
        assert "--benchmark-suite" in text
        assert "benchmarks.suite" in text
        assert "--artifact-root dist/hub" in text
        assert "--local-files-only" in text


def test_suite_report_records_path_free_artifact_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report.json"
    junit = tmp_path / "junit.xml"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    spec = get_model_registry()["esm2_8m"]
    identity = {
        "model_id": spec.id,
        "registry_repo_id": spec.fast.repo_id,
        "registry_revision": spec.fast.revision,
        "checkpoint_repo_id": spec.artifact_checkpoint.repo_id,
        "weights_revision": spec.artifact_checkpoint.revision,
        "runtime_revision": "7" * 40,
        "source_tree_sha256": "8" * 64,
        "runtime_bundle_sha256": "9" * 64,
        "canonical_state_sha256": "a" * 64,
        "artifact_manifest_sha256": "b" * 64,
    }

    def fake_bind(cases, root):
        assert root == artifact_root
        for case in cases:
            case.load_model = root / "ESM2-8M"
            case.load_revision = None
            case.local_files_only = True
            case.artifact_identity = identity
            case.artifact_dependencies = {}
        return {"esm2_8m": identity}

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: None),
    )
    monkeypatch.setattr(benchmark_suite, "bind_local_artifacts", fake_bind)
    monkeypatch.setattr(benchmark_suite, "_require_torch", lambda: fake_torch)
    monkeypatch.setattr(
        benchmark_suite,
        "environment_fingerprint",
        lambda _torch: {"gpu": "synthetic", "gpu_capability": [0, 0]},
    )
    monkeypatch.setattr(
        benchmark_suite,
        "_load_model",
        lambda _case, _torch: (object(), 1.0),
    )
    monkeypatch.setattr(
        benchmark_suite,
        "run_case",
        lambda case, **_kwargs: {
            "model": case.model,
            "revision": case.revision,
            "blocks": [],
        },
    )

    assert (
        benchmark_suite.main(
            [
                "--quick",
                "--family",
                "esm2",
                "--artifact-root",
                str(artifact_root),
                "--output",
                str(output),
                "--junit-output",
                str(junit),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 3
    assert report["status"] == "complete"
    assert report["expected_case_count"] == report["completed_case_count"] == 1
    assert report["artifact_load_mode"] == "validated_local_build"
    assert report["artifacts"] == {"esm2_8m": identity}
    assert report["results"][0]["artifact"] == identity
    assert report["results"][0]["model"] == spec.fast.repo_id
    assert report["timing_contract"] == {
        "cold_compile_field": "results[].compile_ms",
        "first_forward_field": "results[].first_forward_ms",
        "warmup_field": "results[].warmup_samples_ms",
        "warm_throughput_field": "results[].blocks",
        "compile_amortized_into_throughput": False,
    }
    assert report["baseline_promotion_contract"]["requires_exact_environment_match"] is True
    assert report["baseline_promotion_contract"]["requires_exact_artifact_inventory_match"] is True
    junit_root = ET.parse(junit).getroot()
    assert junit_root.attrib["failures"] == "0"
    assert junit_root.find("testcase/failure") is None
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)
