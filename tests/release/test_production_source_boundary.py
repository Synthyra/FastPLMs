"""Repository-wide boundary between production code and parity oracles."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/fastplms"
FORBIDDEN_IMPORT_ROOTS = {
    "E1",
    "boltz",
    "byprot",
    "esm",
    "openfold",
    "vendor",
}


def _absolute_import_roots(tree: ast.AST) -> set[str]:
    roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    )
    return roots


def _mutates_python_path(node: ast.Call) -> bool:
    function = node.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr in {"append", "extend", "insert"}
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "path"
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "sys"
    )


def _dynamic_upstream_import(node: ast.Call) -> bool:
    function = node.func
    is_import = (isinstance(function, ast.Name) and function.id == "__import__") or (
        isinstance(function, ast.Attribute) and function.attr == "import_module"
    )
    if not is_import or not node.args:
        return False
    module = node.args[0]
    if not isinstance(module, ast.Constant) or not isinstance(module.value, str):
        return False
    return module.value.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS


def test_production_source_never_imports_parity_oracles() -> None:
    failures: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden = sorted(_absolute_import_roots(tree) & FORBIDDEN_IMPORT_ROOTS)
        if forbidden:
            failures.append(f"{path.relative_to(ROOT)} imports upstream roots {forbidden}")
        if any(_mutates_python_path(node) for node in ast.walk(tree) if isinstance(node, ast.Call)):
            failures.append(f"{path.relative_to(ROOT)} mutates sys.path")
        if any(
            _dynamic_upstream_import(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
        ):
            failures.append(f"{path.relative_to(ROOT)} dynamically imports an upstream root")
    assert not failures, "\n" + "\n".join(failures)
