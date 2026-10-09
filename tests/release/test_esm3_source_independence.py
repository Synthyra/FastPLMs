"""Fail closed when ESM3 runtime functions overlap the Biohub parity oracle."""

from __future__ import annotations

import ast
import pytest

from difflib import SequenceMatcher
from pathlib import Path
from tests.release.source_independence import function_node, normalized_ast_lines

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
    local_lines = normalized_ast_lines(function_node(LOCAL_MODEL, local_name))
    upstream_lines = normalized_ast_lines(function_node(upstream_path, upstream_name))
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
