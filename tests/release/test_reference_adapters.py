"""Manifest-to-native-reference adapter release contracts."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from fastplms.registry import get_model_registry

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ROOT = ROOT / "tests" / "parity" / "support" / "reference_adapters"
_RECONSTRUCTED_FORWARD_ATTRIBUTES = frozenset(
    {
        "base_z_combine",
        "base_z_linear",
        "fc1",
        "fc2",
        "k_proj",
        "out_proj",
        "q_proj",
        "v_proj",
    }
)
_FAST_LOADER_NAMES = frozenset(
    {
        "_load_fast",
        "load_fast",
        "load_fast_model",
        "load_fastplms_model",
    }
)


def _adapter_path(module: str) -> Path:
    prefix = "tests.parity.support.reference_adapters."
    assert module.startswith(prefix), f"Reference adapter escapes compliance package: {module}"
    return ADAPTER_ROOT / f"{module.removeprefix(prefix)}.py"


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _is_fastplms_module(module: str | None) -> bool:
    return module == "fastplms" or bool(module and module.startswith("fastplms."))


def _adapter_violations(source: str, *, filename: str = "<adapter>") -> list[str]:
    """Return statically detectable violations of oracle independence."""

    tree = ast.parse(source, filename=filename)
    violations: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_fastplms_module(alias.name):
                    violations.append(f"line {line}: imports FastPLMs ({alias.name})")
                if alias.name == "unittest.mock" or alias.name.startswith("unittest.mock."):
                    violations.append(f"line {line}: imports monkeypatch support ({alias.name})")
        elif isinstance(node, ast.ImportFrom):
            if _is_fastplms_module(node.module):
                violations.append(f"line {line}: imports FastPLMs ({node.module})")
            if node.module == "unittest.mock" or bool(
                node.module and node.module.startswith("unittest.mock.")
            ):
                violations.append(f"line {line}: imports monkeypatch support ({node.module})")
            if node.module == "tests.parity.test_model_parity":
                violations.append(f"line {line}: reuses the FastPLMs parity loader")
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func) or ""
            leaf = name.rsplit(".", 1)[-1]
            if leaf in {"patch", "setattr", "delattr"} or name.endswith("patch.object"):
                violations.append(f"line {line}: monkeypatches runtime state ({name})")
            if leaf in _FAST_LOADER_NAMES or leaf.startswith(("_load_fastplms", "load_fastplms")):
                violations.append(f"line {line}: reuses a FastPLMs loader ({name})")
            if name in {"__import__", "importlib.import_module"} and node.args:
                module = node.args[0]
                if (
                    isinstance(module, ast.Constant)
                    and isinstance(module.value, str)
                    and _is_fastplms_module(module.value)
                ):
                    violations.append(f"line {line}: dynamically imports FastPLMs ({module.value})")
        elif isinstance(node, ast.Attribute):
            if node.attr in _RECONSTRUCTED_FORWARD_ATTRIBUTES:
                violations.append(
                    f"line {line}: accesses forward implementation detail ({node.attr})"
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            raw_targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
            for target in raw_targets:
                for nested in ast.walk(target):
                    if isinstance(nested, ast.Attribute) and nested.attr in {
                        "__class__",
                        "forward",
                        "from_pretrained",
                    }:
                        violations.append(
                            f"line {line}: replaces upstream behavior ({nested.attr})"
                        )
    return sorted(set(violations))


def test_every_manifest_reference_adapter_has_a_pinned_loader() -> None:
    for family in get_model_registry().families.values():
        path = _adapter_path(family.reference_adapter)
        assert path.is_file(), f"Missing reference adapter source: {path}"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        loaders = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "load_official_model"
        ]
        assert len(loaders) == 1, f"{path} must define exactly one load_official_model"
        arguments = {argument.arg for argument in loaders[0].args.args}
        assert {"reference_repo_id", "reference_revision", "device", "dtype"}.issubset(arguments), (
            f"{path} does not require the immutable reference loading contract"
        )


def test_reference_adapters_are_independent_oracles() -> None:
    paths = {
        _adapter_path(family.reference_adapter) for family in get_model_registry().families.values()
    }
    failures: list[str] = []
    for path in sorted(paths):
        for violation in _adapter_violations(path.read_text(encoding="utf-8"), filename=str(path)):
            failures.append(f"{path.relative_to(ROOT)}: {violation}")
    assert not failures, "Reference adapter independence violations:\n  - " + "\n  - ".join(
        failures
    )


@pytest.mark.parametrize(
    ("source", "expected_fragment"),
    [
        ("import fastplms", "imports FastPLMs"),
        ("from fastplms.runtime import load_model", "imports FastPLMs"),
        ("import importlib\nimportlib.import_module('fastplms.models.e1')", "dynamically imports"),
        ("from unittest.mock import patch", "monkeypatch support"),
        ("setattr(model, 'forward', replacement)", "monkeypatches runtime state"),
        ("from tests.parity.test_model_parity import _load_fast", "parity loader"),
        ("model = _load_fast(spec)", "FastPLMs loader"),
        ("weights = shim.base_z_combine.softmax(0)", "forward implementation detail"),
        ("model.forward = replacement", "replaces upstream behavior"),
    ],
)
def test_reference_adapter_static_rules_fail_closed(
    source: str,
    expected_fragment: str,
) -> None:
    violations = _adapter_violations(source)
    assert any(expected_fragment in violation for violation in violations), violations


def test_every_reference_adapter_maps_to_a_native_container() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")
    for family in get_model_registry().families.values():
        target = family.reference_container
        assert f"AS {target}" in dockerfile
        assert f"  {target}:" in compose


def test_reference_sources_use_target_specific_build_contexts() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    bake = (ROOT / "docker" / "docker-bake.hcl").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    contexts = {
        "upstream_ankh": "vendor/upstream/ankh",
        "upstream_biohub_esm": "vendor/upstream/biohub-esm",
        "upstream_biohub_transformers": "vendor/upstream/biohub-transformers",
        "upstream_boltz": "vendor/upstream/boltz",
        "upstream_dplm": "vendor/upstream/dplm",
        "upstream_e1": "vendor/upstream/e1",
        "upstream_fair_esm": "vendor/upstream/fair-esm",
        "upstream_openfold": "vendor/upstream/openfold",
        "upstream_protein_ttt": "vendor/upstream/protein-ttt",
    }
    assert "COPY vendor/upstream" not in dockerfile
    assert "!vendor/upstream" not in dockerignore
    for name, source in contexts.items():
        assert f"--from={name} --exclude=.git --exclude=.git/**" in dockerfile
        assignment = rf"{re.escape(name)}\s*=\s*\"{re.escape(source)}\""
        assert re.search(assignment, bake)


def test_biohub_reference_is_locked_attested_and_arm64_native() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    bake = (ROOT / "docker" / "docker-bake.hcl").read_text(encoding="utf-8")
    stage = dockerfile.split("FROM python312 AS reference-biohub-esm", maxsplit=1)[1].split(
        "FROM python312 AS reference-esm2", maxsplit=1
    )[0]

    assert "git+https://github.com/Biohub/transformers" not in stage
    assert "@main" not in stage
    assert 'test "${TARGETARCH}" = "arm64"' in stage
    assert 'test "$(uname -m)" = "aarch64"' in stage
    assert "--from=biohub_biotraj_wheel" in stage
    assert "biohub-reference.lock.txt" in stage
    assert "biohub-reference-lock.json" in stage
    assert "verify-contract" in stage
    assert "materialize-wheel-lock" in stage
    assert "--require-hashes --only-binary=:all: --no-deps" in stage
    assert stage.count("verify-inventory") == 2
    assert stage.count("--profile final") == 2
    assert "--no-deps" in stage
    assert "--no-build-isolation" in stage
    assert "-e /opt/oracle/vendor" not in stage
    assert "--force-reinstall" not in stage
    assert "cp -a /opt/oracle/vendor/upstream/biohub-transformers" in stage
    assert "cp -a /opt/oracle/vendor/upstream/biohub-esm" in stage
    assert stage.count("--exclude=**/__pycache__ ") == 2
    assert stage.count("--exclude=**/__pycache__/**") == 2
    assert stage.count("--exclude=**/*.pyc") == 2
    assert stage.count("--exclude=**/*.pyo") == 2
    assert "verify-pip-check" in stage
    assert "dist-info/WHEEL" not in stage
    assert "manylinux2014_sbsa" not in stage
    assert "test ! -e /opt/oracle/tools/remote/__init__.py" in stage
    assert "import tools.remote.biohub_reference_environment" in stage
    assert "import tools.remote.biohub_reference_lock" in stage
    assert "import tools.remote.reference_source_attestation" in stage
    assert "find_spec('tools.remote.run') is None" in stage
    assert "reference_source_attestation create" in stage
    assert "reference_source_attestation verify" in stage
    assert stage.count("--contract /opt/oracle/biohub-esm-source-contract.json") == 2
    assert stage.count("--contract /opt/oracle/biohub-transformers-source-contract.json") == 2
    assert stage.index("pip install --require-hashes") < stage.index(
        "COPY --from=reference-protocol"
    )
    assert stage.index("pip install --require-hashes") < stage.index("COPY THIRD_PARTY_NOTICES.md")
    environment = stage.split("ENV FASTPLMS_BIOHUB_ESM_REVISION", maxsplit=1)[1]
    assert "FASTPLMS_BIOHUB_ESM_CONTRACT=" in environment
    assert "FASTPLMS_BIOHUB_TRANSFORMERS_CONTRACT=" in environment
    assert "FASTPLMS_BIOHUB_LOCK_CONTRACT=" in environment
    assert "FASTPLMS_REFERENCE_CONTAINER_IDENTITIES=" in environment
    assert "FASTPLMS_REFERENCE_CONTAINER_TARGET=reference-biohub-esm" in environment
    assert "PYTHONPATH=/opt/oracle" in environment
    assert "biohub-transformers/src" not in environment
    assert "vendor/upstream/biohub-esm:/opt/oracle" not in environment

    assert 'target "biohub-biotraj-wheel"' in bake
    assert re.search(r'target\s*=\s*"biotraj-wheel-artifact"', bake)
    assert 'biohub_biotraj_wheel         = "target:biohub-biotraj-wheel"' in bake

    registry = get_model_registry()
    expected_sources = {
        "biohub-esm": {
            "import_name": "esm",
            "import_root": "esm",
            "package_version": "3.3.0",
            "tree_sha256": "c5489f1fc58de200978803de2c38e1a78f769cb183a2ee90be833f0f4a0212e8",
        },
        "biohub-transformers": {
            "import_name": "transformers",
            "import_root": "src/transformers",
            "package_version": "4.57.6",
            "tree_sha256": "28b910cc18b821870db2fb6d1c50376c2d14287ae18485080699e03fa4ba4f43",
        },
    }
    for source_id, expected in expected_sources.items():
        source = registry.upstreams[source_id]
        contract = json.loads(
            (ROOT / f"docker/constraints/{source_id}-source.json").read_text(encoding="utf-8")
        )
        assert contract == {
            **expected,
            "schema_version": 1,
            "source_revision": source.revision,
        }
        environment_name = source_id.removeprefix("biohub-").upper().replace("-", "_")
        assert f"FASTPLMS_BIOHUB_{environment_name}_REVISION={source.revision}" in stage

    import_boundaries = {
        "esm_plusplus.py": "from transformers import AutoTokenizer",
        "esmfold2.py": "from transformers.models.esmfold2.configuration_esmfold2 import",
        "esm3.py": "from esm.pretrained import load_local_model",
    }
    for filename, boundary in import_boundaries.items():
        adapter = (ADAPTER_ROOT / filename).read_text(encoding="utf-8")
        load_function = adapter.split("def load_official_model", maxsplit=1)[1]
        assert load_function.index("reference_sources()") < load_function.index(boundary)

    shared_gate = (ADAPTER_ROOT / "biohub_source.py").read_text(encoding="utf-8")
    for source_id, expected in expected_sources.items():
        assert registry.upstreams[source_id].revision in shared_gate
        assert expected["tree_sha256"] in shared_gate
    assert "capture_biohub_reference_environment" in shared_gate
    assert 'environment_prefix="FASTPLMS_BIOHUB_ESM"' in shared_gate
    assert 'environment_prefix="FASTPLMS_BIOHUB_TRANSFORMERS"' in shared_gate
    assert 'f"{environment_prefix}_CONTRACT"' in shared_gate

    structure_bundle = (ROOT / "tests/structure/support/esmfold2_bundle.py").read_text(
        encoding="utf-8"
    )
    reference_producer = structure_bundle.split("def produce_reference", maxsplit=1)[1].split(
        "def produce_candidate", maxsplit=1
    )[0]
    assert reference_producer.index("reference_sources()") < reference_producer.index(
        "_load_reference_model(request, device)"
    )
    assert reference_producer.index("reference_environment()") < reference_producer.index(
        "_load_reference_model(request, device)"
    )

    native_reference = (ROOT / "tests/parity/support/native_reference.py").read_text(
        encoding="utf-8"
    )
    assert 'metadata["reference_sources"] = reference_sources' in native_reference
    assert 'metadata["reference_environment"] = reference_environment' in native_reference

    esmfold2_stage = dockerfile.split(
        "FROM reference-biohub-esm AS reference-esmfold2", maxsplit=1
    )[1].split("FROM python310-reference AS reference-protein-ttt", maxsplit=1)[0]
    assert "FASTPLMS_REFERENCE_CONTAINER_TARGET=reference-esmfold2" in esmfold2_stage
    assert "reference_sources; reference_sources(); from transformers" in esmfold2_stage
