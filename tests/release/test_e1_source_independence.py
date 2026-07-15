"""Fail closed when E1 runtime functions overlap the pinned parity oracle."""

from __future__ import annotations

import ast
import copy
from difflib import SequenceMatcher
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL = ROOT / "src/fastplms/models/e1/modeling_e1.py"
LOCAL_ATTENTION = ROOT / "src/fastplms/models/e1/attention.py"
LOCAL_PREPARATION = ROOT / "src/fastplms/models/e1/preparation.py"
UPSTREAM = ROOT / "vendor/upstream/e1/src/E1"
MAX_FUNCTION_SIMILARITY = 0.75

# Compare only functions that implement the same public or mathematical
# contract. Function-level ASTs cannot be diluted by unrelated model code.
SOURCE_PAIRS = (
    (
        "E1BatchPreparer.prepare_multiseq",
        LOCAL_PREPARATION,
        UPSTREAM / "batch_preparer.py",
        "E1BatchPreparer.prepare_multiseq",
    ),
    (
        "E1BatchPreparer.prepare_singleseq",
        LOCAL_PREPARATION,
        UPSTREAM / "batch_preparer.py",
        "E1BatchPreparer.prepare_singleseq",
    ),
    (
        "get_overlapping_blocks",
        LOCAL_ATTENTION,
        UPSTREAM / "model/varlen_flex_attention.py",
        "get_overlapping_blocks",
    ),
    (
        "direct_block_mask",
        LOCAL_ATTENTION,
        UPSTREAM / "model/varlen_flex_attention.py",
        "direct_block_mask",
    ),
    (
        "_get_unpad_data",
        LOCAL_ATTENTION,
        UPSTREAM / "model/flash_attention_utils.py",
        "_get_unpad_data",
    ),
    (
        "E1PreTrainedModel._init_weights",
        LOCAL_MODEL,
        UPSTREAM / "modeling.py",
        "E1PreTrainedModel._init_weights",
    ),
    (
        "FAST_E1_ENCODER.forward",
        LOCAL_MODEL,
        UPSTREAM / "modeling.py",
        "E1Model.forward",
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
    ("local_name", "local_path", "upstream_path", "upstream_name"),
    SOURCE_PAIRS,
    ids=[local_name for local_name, _, _, _ in SOURCE_PAIRS],
)
def test_e1_functions_are_independently_implemented(
    local_name: str,
    local_path: Path,
    upstream_path: Path,
    upstream_name: str,
) -> None:
    assert upstream_path.is_file(), f"pinned E1 source is missing: {upstream_path}"
    assert local_path.is_file(), f"local E1 source is missing: {local_path}"
    local_lines = _normalized_ast_lines(_function(local_path, local_name))
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
