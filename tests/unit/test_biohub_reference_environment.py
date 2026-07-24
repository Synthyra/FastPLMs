"""Fail-closed contracts for the pinned Biohub reference environment."""

from __future__ import annotations

import hashlib
import json
import sys
import pytest
from pathlib import Path
from types import ModuleType

from tools.remote.biohub_reference_requirements import (
    BiohubReferenceRequirementsError,
    extract_biohub_reference_requirements,
    write_biohub_reference_requirements,
)
from tools.remote.reference_source_attestation import (
    ReferenceSourceAttestationError,
    create_reference_source_attestation,
    validate_reference_source_evidence,
    validate_reference_sources_evidence,
    verify_reference_source,
)
from tools.source_provenance import actual_tree_paths, tracked_tree_digest


_TRANSFORMERS_MAIN = "transformers @ git+https://github.com/Biohub/transformers.git@main"


def _write_biohub_pyproject(path: Path, dependencies: list[str]) -> None:
    rendered = "\n".join(f'  "{requirement}",' for requirement in dependencies)
    path.write_text(
        f"[project]\nname = \"esm\"\ndependencies = [\n{rendered}\n]\n",
        encoding="utf-8",
    )


def test_biohub_requirements_remove_only_the_known_transformers_main_url(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    output = tmp_path / "requirements.txt"
    _write_biohub_pyproject(
        pyproject,
        ["torch>=2.2", _TRANSFORMERS_MAIN, "accelerate", "biotite>=1"],
    )

    assert extract_biohub_reference_requirements(pyproject) == (
        "torch>=2.2",
        "accelerate",
        "biotite>=1",
    )
    assert write_biohub_reference_requirements(pyproject, output) == (
        "torch>=2.2",
        "accelerate",
        "biotite>=1",
    )
    assert output.read_text(encoding="utf-8") == "torch>=2.2\naccelerate\nbiotite>=1\n"


@pytest.mark.parametrize(
    "dependencies",
    (
        ["torch", "transformers>=4.57"],
        ["torch", _TRANSFORMERS_MAIN, "other @ https://example.invalid/archive.whl"],
        ["torch", _TRANSFORMERS_MAIN, _TRANSFORMERS_MAIN],
        ["torch", _TRANSFORMERS_MAIN, "-r injected.txt"],
        ["torch", _TRANSFORMERS_MAIN, "--extra-index-url=https://example.invalid"],
        ["torch", _TRANSFORMERS_MAIN, "../local-wheel.whl"],
        ["torch", _TRANSFORMERS_MAIN, "package; python_version >= '3.12'"],
    ),
)
def test_biohub_requirements_reject_dependency_contract_drift(
    tmp_path: Path,
    dependencies: list[str],
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_biohub_pyproject(pyproject, dependencies)

    with pytest.raises(BiohubReferenceRequirementsError):
        extract_biohub_reference_requirements(pyproject)


def _reference_source_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    source_root = tmp_path / "source"
    package = source_root / "src" / "pinned_reference_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "1.2.3"\n', encoding="utf-8")
    (package / "model.py").write_text("VALUE = 7\n", encoding="utf-8")
    digest = tracked_tree_digest(source_root, actual_tree_paths(source_root))
    revision = "a" * 40
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_revision": revision,
                "tree_sha256": digest,
                "import_name": "pinned_reference_probe",
                "import_root": "src/pinned_reference_probe",
                "package_version": "1.2.3",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return source_root, contract, revision


def test_reference_source_attestation_rehashes_and_proves_import_origin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root, contract, revision = _reference_source_fixture(tmp_path)
    attestation = tmp_path / "attestation.json"
    created = create_reference_source_attestation(source_root, contract, attestation)
    assert created["source_revision"] == revision
    assert created["file_count"] == 2

    monkeypatch.syspath_prepend(str(source_root / "src"))
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    sys.modules.pop("pinned_reference_probe", None)
    evidence = verify_reference_source(
        source_root,
        attestation,
        contract,
        expected_revision=revision,
    )
    assert evidence == {
        "attestation_sha256": hashlib.sha256(attestation.read_bytes()).hexdigest(),
        "file_count": 2,
        "import_file": "src/pinned_reference_probe/__init__.py",
        "import_name": "pinned_reference_probe",
        "import_root": "src/pinned_reference_probe",
        "package_version": "1.2.3",
        "schema_version": 1,
        "source_revision": revision,
        "tree_sha256": created["tree_sha256"],
    }
    assert validate_reference_source_evidence(evidence) == evidence
    sources = {
        "biohub-esm": dict(evidence),
        "biohub-transformers": dict(evidence),
    }
    assert validate_reference_sources_evidence(
        sources,
        required_sources=("biohub-esm", "biohub-transformers"),
    ) == sources
    with pytest.raises(ReferenceSourceAttestationError, match="names differ"):
        validate_reference_sources_evidence(
            {"biohub-transformers": evidence},
            required_sources=("biohub-esm", "biohub-transformers"),
        )
    assert sys.dont_write_bytecode is False
    assert not any(path.name == "__pycache__" for path in source_root.rglob("*"))

    (source_root / "src/pinned_reference_probe/model.py").write_text(
        "VALUE = 8\n",
        encoding="utf-8",
    )
    with pytest.raises(ReferenceSourceAttestationError, match="changed after image construction"):
        verify_reference_source(source_root, attestation, contract, expected_revision=revision)


def test_reference_source_attestation_rejects_untracked_import_code(tmp_path: Path) -> None:
    source_root, contract, revision = _reference_source_fixture(tmp_path)
    attestation = tmp_path / "attestation.json"
    create_reference_source_attestation(source_root, contract, attestation)
    (source_root / "src/pinned_reference_probe/injected.py").write_text(
        "VALUE = 'untracked'\n",
        encoding="utf-8",
    )

    with pytest.raises(ReferenceSourceAttestationError, match="inventory differs"):
        verify_reference_source(source_root, attestation, contract, expected_revision=revision)


def test_reference_source_attestation_rejects_untracked_source_root_code(
    tmp_path: Path,
) -> None:
    source_root, contract, revision = _reference_source_fixture(tmp_path)
    attestation = tmp_path / "attestation.json"
    create_reference_source_attestation(source_root, contract, attestation)
    (source_root / "src/sitecustomize.py").write_text(
        "raise RuntimeError('must never execute')\n",
        encoding="utf-8",
    )

    with pytest.raises(ReferenceSourceAttestationError, match="inventory differs"):
        verify_reference_source(source_root, attestation, contract, expected_revision=revision)


def test_reference_source_attestation_rejects_self_authored_tree_identity(
    tmp_path: Path,
) -> None:
    source_root, contract, revision = _reference_source_fixture(tmp_path)
    attestation = tmp_path / "attestation.json"
    create_reference_source_attestation(source_root, contract, attestation)
    (source_root / "src/pinned_reference_probe/model.py").write_text(
        "VALUE = 9\n",
        encoding="utf-8",
    )
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    payload["tree_sha256"] = tracked_tree_digest(source_root, payload["tracked_files"])
    attestation.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ReferenceSourceAttestationError, match="checked-in contract"):
        verify_reference_source(source_root, attestation, contract, expected_revision=revision)


def test_reference_source_attestation_rejects_cached_external_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root, contract, revision = _reference_source_fixture(tmp_path)
    attestation = tmp_path / "attestation.json"
    create_reference_source_attestation(source_root, contract, attestation)
    rogue = ModuleType("pinned_reference_probe")
    rogue.__file__ = str(tmp_path / "rogue/pinned_reference_probe/__init__.py")
    monkeypatch.setitem(sys.modules, "pinned_reference_probe", rogue)

    with pytest.raises(ReferenceSourceAttestationError, match="outside the pinned source"):
        verify_reference_source(source_root, attestation, contract, expected_revision=revision)


def test_reference_source_attestation_rejects_cached_external_submodule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root, contract, revision = _reference_source_fixture(tmp_path)
    attestation = tmp_path / "attestation.json"
    create_reference_source_attestation(source_root, contract, attestation)
    monkeypatch.delitem(sys.modules, "pinned_reference_probe", raising=False)
    rogue = ModuleType("pinned_reference_probe.poison")
    rogue.__file__ = str(tmp_path / "rogue/pinned_reference_probe/poison.py")
    monkeypatch.setitem(sys.modules, "pinned_reference_probe.poison", rogue)

    with pytest.raises(ReferenceSourceAttestationError, match="outside the pinned source"):
        verify_reference_source(source_root, attestation, contract, expected_revision=revision)
