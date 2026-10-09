"""Fail closed when E1 runtime functions overlap the pinned parity oracle."""

from __future__ import annotations

import pytest

from difflib import SequenceMatcher
from pathlib import Path
from tests.release.source_independence import function_node, normalized_ast_lines


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
    local_lines = normalized_ast_lines(function_node(local_path, local_name))
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
