"""Manifest-to-native-reference adapter release contracts."""

from __future__ import annotations

import ast
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
