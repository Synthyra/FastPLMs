"""Actual offline Transformers AutoClass dispatch from tiny remote-code artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import pytest

from pathlib import Path
from tests.unit.tiny_families import tiny_config

from fastplms.registry import ModelFamily, get_model_registry
from tools.artifacts.build import (
    _artifact_auto_map,
    _runtime_source_entries,
    _write_bootstrap,
    _write_runtime_bundle,
    _write_runtime_snapshot,
)
from tools.artifacts.offline_probe import _CPU_CONTRACT_MARKER, _runtime_site_packages


_ROOT = Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "tools" / "artifacts" / "offline_probe.py"
_STRUCTURE_FAMILIES = frozenset({"boltz2", "esmfold", "esmfold2"})
_REMOTE_RESAVE_FAMILIES = frozenset({"ankh", "dplm", "dplm2", "esm2", "esm3", "esm_plusplus"})


def _write_cpu_artifact(root: Path, family: ModelFamily) -> Path:
    registry = get_model_registry()
    spec = registry[family.representative]
    if dict(spec.auto_map) != dict(family.auto_map):
        raise AssertionError(f"Representative {spec.id} has a model-specific AutoMap override.")

    artifact = root / family.id / "artifact"
    artifact.mkdir(parents=True)
    config = tiny_config(family.id)
    config.auto_map = _artifact_auto_map(spec)
    config.fastplms_cpu_contract_only = True
    config.save_pretrained(artifact)
    config_path = artifact / "config.json"
    artifact_config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact_config["auto_map"] = _artifact_auto_map(spec)
    config_path.write_text(
        json.dumps(artifact_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact / _CPU_CONTRACT_MARKER).write_text(
        json.dumps(
            {"release_artifact": False, "schema_version": 1, "scope": "tests/cpu"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    runtime_root = root / family.id / "runtime" / "fastplms"
    payloads = {
        target.as_posix(): source.read_bytes()
        for source, target in _runtime_source_entries(_ROOT, spec)
    }
    _write_runtime_snapshot(runtime_root, payloads)
    runtime_hash = _write_runtime_bundle(artifact / "fastplms_bundle.py", runtime_root)
    _write_bootstrap(artifact / "modeling_fastplms.py", spec, runtime_hash)
    return artifact


def _case_payload(family: ModelFamily) -> list[dict[str, object]]:
    return [
        {
            "auto_class": auto_class,
            "class_path": class_path,
            "expected_missing_key_prefixes": [],
            "expected_unexpected_key_prefixes": [],
        }
        for auto_class, class_path in sorted(family.auto_map.items())
    ]


def _run_family_probe(
    root: Path,
    family: ModelFamily,
    artifact: Path,
) -> subprocess.CompletedProcess[str]:
    family_root = root / family.id
    cases_path = family_root / "cases.json"
    output_path = family_root / "results.json"
    cases_path.write_text(
        json.dumps(_case_payload(family), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-I",
        "-S",
        str(_PROBE),
        "--artifact",
        str(artifact),
        "--family",
        family.id,
        "--bf16-execution",
        family.bf16_execution,
        "--cases-file",
        str(cases_path),
        "--implementation",
        "artifact",
        "--output",
        str(output_path),
        "--tiny-cpu-contract",
    ]
    for site_packages in _runtime_site_packages():
        command.extend(("--runtime-site-package", str(site_packages)))
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HOME": str(family_root / "hf-home"),
            "HF_HUB_OFFLINE": "1",
            "HF_MODULES_CACHE": str(family_root / "modules"),
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return subprocess.run(
        command,
        cwd=family_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


_FAMILY_IDS = tuple(sorted(get_model_registry().families))


@pytest.mark.parametrize("family_id", _FAMILY_IDS)
def test_every_family_dispatches_all_advertised_remote_autoclasses_offline(
    family_id: str,
    tmp_path: Path,
) -> None:
    registry = get_model_registry()
    family = registry.families[family_id]
    artifact = _write_cpu_artifact(tmp_path, family)
    completed = _run_family_probe(tmp_path, family, artifact)
    output_path = tmp_path / family_id / "results.json"
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output_path.is_file()
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(output) == set(family.auto_map)

    for auto_class, class_report in output.items():
        expected = family.auto_map[auto_class]
        assert class_report["class"] == expected.rsplit(".", maxsplit=1)[1]
        if auto_class != "AutoConfig" and family_id not in _STRUCTURE_FAMILIES:
            assert class_report["resized_vocab"] >= 9
            assert class_report["tuple_fields"] >= 1
        if family_id in _STRUCTURE_FAMILIES and auto_class != "AutoConfig":
            assert class_report["structure_forward_delegated"] is True
    if family_id in _REMOTE_RESAVE_FAMILIES:
        assert output["AutoModel"]["resaved"] is True
    marker = json.loads((artifact / _CPU_CONTRACT_MARKER).read_text(encoding="utf-8"))
    assert marker == {
        "release_artifact": False,
        "schema_version": 1,
        "scope": "tests/cpu",
    }
    assert not (artifact / "artifact-manifest.json").exists()
    assert not (artifact / "source-record.json").exists()
    assert not (artifact / "runtime-attestation.json").exists()


def test_structure_auto_dispatch_is_complemented_by_public_forward_contracts() -> None:
    from tests.cpu import test_structure_contracts as structure_contracts

    for test_name in (
        "test_boltz_public_forward_honors_output_controls_backward_and_reload",
        "test_fast_esmfold_public_forward_honors_output_controls_and_backward",
        "test_fast_esmfold_tiny_model_saves_and_reloads_exact_state",
        "test_esmfold2_public_forward_honors_output_controls_and_sampler_overrides",
        "test_esmfold2_advertised_models_tiny_init_backward_and_save_reload",
    ):
        assert callable(getattr(structure_contracts, test_name))


def test_grouped_remote_dispatch_covers_exactly_45_family_entries() -> None:
    registry = get_model_registry()
    assert sum(len(family.auto_map) for family in registry.families.values()) == 45


def test_isolated_probe_blocks_reference_reads_before_remote_code_exec(
    tmp_path: Path,
) -> None:
    registry = get_model_registry()
    family = registry.families["esm2"]
    artifact = _write_cpu_artifact(tmp_path, family)
    bootstrap = artifact / "modeling_fastplms.py"
    forbidden = _ROOT / "vendor" / "upstream" / "forbidden-cpu-probe-read"
    bootstrap.write_text(
        "from pathlib import Path\n"
        f"Path({str(forbidden)!r}).read_bytes()\n" + bootstrap.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    completed = _run_family_probe(tmp_path, family, artifact)

    assert completed.returncode != 0
    assert "may not access submodule/reference path" in completed.stdout + completed.stderr
