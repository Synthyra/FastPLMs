"""Fail closed when DPLM runtime units overlap the pinned parity oracle."""

from __future__ import annotations

import ast
import copy
import pytest
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL = ROOT / "src/fastplms/models/dplm/modeling_dplm.py"
UPSTREAM_MODEL = (
    ROOT
    / "vendor/upstream/dplm/src/byprot/models/dplm/modules/dplm_modeling_esm.py"
)
MAX_FUNCTION_SIMILARITY = 0.75
MAX_EXACT_BLOCK_LINES = 10
SOURCE_PAIRS = (
    ("ModifiedEsmSelfAttention.forward", "ModifiedEsmSelfAttention.forward"),
    ("ModifiedEsmAttention.__init__", "ModifiedEsmAttention.__init__"),
    ("ModifiedEsmLayer.__init__", "ModifiedEsmLayer.__init__"),
    ("ModifiedEsmEncoder.__init__", "ModifiedEsmEncoder.__init__"),
    ("FAST_DPLM_ENCODER.forward", "ModifiedEsmModel.forward"),
)


def _function(path: Path, qualified_name: str) -> ast.FunctionDef:
    body: list[ast.stmt] = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    ).body
    selected: ast.AST | None = None
    for part in qualified_name.split("."):
        selected = next(
            (
                node
                for node in body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name == part
            ),
            None,
        )
        assert selected is not None, f"{qualified_name!r} is absent from {path}"
        body = selected.body
    assert isinstance(selected, ast.FunctionDef)
    return selected


def _normalized_ast_lines(node: ast.FunctionDef) -> list[str]:
    normalized = copy.deepcopy(node)
    normalized.name = "function"
    normalized.decorator_list = []
    normalized.returns = None
    for argument in (
        *normalized.args.posonlyargs,
        *normalized.args.args,
        *normalized.args.kwonlyargs,
    ):
        argument.annotation = None
    if normalized.args.vararg is not None:
        normalized.args.vararg.annotation = None
    if normalized.args.kwarg is not None:
        normalized.args.kwarg.annotation = None
    if (
        normalized.body
        and isinstance(normalized.body[0], ast.Expr)
        and isinstance(normalized.body[0].value, ast.Constant)
        and isinstance(normalized.body[0].value.value, str)
    ):
        normalized.body.pop(0)
    ast.fix_missing_locations(normalized)
    return [
        " ".join(line.strip().split())
        for line in ast.unparse(normalized).splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize(
    ("local_name", "upstream_name"),
    SOURCE_PAIRS,
    ids=[local_name for local_name, _ in SOURCE_PAIRS],
)
def test_dplm_functions_are_independently_implemented(
    local_name: str,
    upstream_name: str,
) -> None:
    assert UPSTREAM_MODEL.is_file(), f"pinned DPLM source is missing: {UPSTREAM_MODEL}"
    local_lines = _normalized_ast_lines(_function(LOCAL_MODEL, local_name))
    upstream_lines = _normalized_ast_lines(_function(UPSTREAM_MODEL, upstream_name))
    matcher = SequenceMatcher(None, local_lines, upstream_lines, autojunk=False)
    similarity = matcher.ratio()
    assert similarity < MAX_FUNCTION_SIMILARITY, (
        f"{local_name} has normalized AST similarity {similarity:.3f} to "
        f"{UPSTREAM_MODEL.relative_to(ROOT)}::{upstream_name}"
    )
    exact_lines = max(block.size for block in matcher.get_matching_blocks())
    assert exact_lines <= MAX_EXACT_BLOCK_LINES, (
        f"{local_name} retains an exact {exact_lines}-line source block from "
        f"{UPSTREAM_MODEL.relative_to(ROOT)}::{upstream_name}"
    )


def test_dplm_source_does_not_import_the_parity_oracle() -> None:
    tree = ast.parse(LOCAL_MODEL.read_text(encoding="utf-8"), filename=str(LOCAL_MODEL))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    )
    assert imported_roots.isdisjoint({"byprot", "vendor"})
