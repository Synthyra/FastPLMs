"""Fail closed when ESM3 runtime functions overlap the Biohub parity oracle."""

from __future__ import annotations

import ast
import copy
import pytest
from difflib import SequenceMatcher
from pathlib import Path

from fastplms.models.esm3.modeling_esm3 import FastESM3PreTrainedModel


ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL = ROOT / "src/fastplms/models/esm3/modeling_esm3.py"
BIOHUB = ROOT / "vendor/upstream/biohub-esm/esm"
MAX_FUNCTION_SIMILARITY = 0.75
SOURCE_PAIRS = (
    (
        "EncodeInputs.__init__",
        BIOHUB / "models/esm3.py",
        "EncodeInputs.__init__",
    ),
    (
        "GeometricReasoningOriginalImpl.__init__",
        BIOHUB / "layers/geom_attention.py",
        "GeometricReasoningOriginalImpl.__init__",
    ),
    (
        "UnifiedTransformerBlock.forward",
        BIOHUB / "layers/blocks.py",
        "UnifiedTransformerBlock.forward",
    ),
)


def _function(path: Path, qualified_name: str) -> ast.FunctionDef:
    body: list[ast.stmt] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body
    selected: ast.AST | None = None
    for part in qualified_name.split("."):
        selected = next(
            (
                node
                for node in body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == part
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
    ("local_name", "upstream_path", "upstream_name"),
    SOURCE_PAIRS,
    ids=[local_name for local_name, _, _ in SOURCE_PAIRS],
)
def test_esm3_functions_are_independently_implemented(
    local_name: str,
    upstream_path: Path,
    upstream_name: str,
) -> None:
    assert upstream_path.is_file(), f"pinned Biohub source is missing: {upstream_path}"
    local_lines = _normalized_ast_lines(_function(LOCAL_MODEL, local_name))
    upstream_lines = _normalized_ast_lines(_function(upstream_path, upstream_name))
    similarity = SequenceMatcher(
        None,
        local_lines,
        upstream_lines,
        autojunk=False,
    ).ratio()
    assert similarity < MAX_FUNCTION_SIMILARITY, (
        f"{local_name} has normalized AST similarity {similarity:.3f} to "
        f"{upstream_path.relative_to(ROOT)}::{upstream_name}"
    )


def test_esm3_source_does_not_import_upstream_packages() -> None:
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
    assert imported_roots.isdisjoint({"esm", "vendor"})


def test_esm3_rejects_unavailable_flash_kernels() -> None:
    assert FastESM3PreTrainedModel._supports_flash_attn_2 is False
    assert FastESM3PreTrainedModel._supports_flash_attn_3 is False
    assert FastESM3PreTrainedModel._fastplms_attention_implementations == (
        "eager",
        "sdpa",
        "flex_attention",
    )
