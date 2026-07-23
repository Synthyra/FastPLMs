"""Fail closed when ESMFold2 runtime source overlaps its parity oracles."""

from __future__ import annotations

import ast
from difflib import SequenceMatcher
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "src/fastplms/models/esmfold2"
MAX_LINE_SIMILARITY = 0.75
BIOHUB_ESM = "vendor/upstream/biohub-esm/esm"
BIOHUB_TRANSFORMERS = "vendor/upstream/biohub-transformers/src/transformers/models/esmfold2"

# Every runtime module must be classified here or in ORIGINAL_RUNTIME_MODULES. This
# inventory makes a new derivative module a release failure instead of silently
# excluding it from the source-boundary check.
SOURCE_COUNTERPARTS = {
    "__init__.py": f"{BIOHUB_TRANSFORMERS}/__init__.py",
    "configuration_esmfold2.py": f"{BIOHUB_TRANSFORMERS}/configuration_esmfold2.py",
    "esmfold2_affine3d.py": f"{BIOHUB_ESM}/utils/structure/affine3d.py",
    "esmfold2_aligner.py": f"{BIOHUB_ESM}/utils/structure/aligner.py",
    "esmfold2_atom_indexer.py": f"{BIOHUB_ESM}/utils/structure/atom_indexer.py",
    "esmfold2_conformers.py": f"{BIOHUB_ESM}/models/esmfold2/conformers.py",
    "esmfold2_constants.py": f"{BIOHUB_ESM}/models/esmfold2/constants.py",
    "esmfold2_constants_esm3.py": f"{BIOHUB_ESM}/utils/constants/esm3.py",
    "esmfold2_input_builder.py": f"{BIOHUB_ESM}/utils/structure/input_builder.py",
    "esmfold2_metrics.py": f"{BIOHUB_ESM}/utils/structure/metrics.py",
    "esmfold2_misc.py": f"{BIOHUB_ESM}/utils/misc.py",
    "esmfold2_mmcif_parsing.py": f"{BIOHUB_ESM}/utils/structure/mmcif_parsing.py",
    "esmfold2_molecular_complex.py": f"{BIOHUB_ESM}/utils/structure/molecular_complex.py",
    "esmfold2_msa.py": f"{BIOHUB_ESM}/utils/msa/msa.py",
    "esmfold2_msa_filter_sequences.py": f"{BIOHUB_ESM}/utils/msa/filter_sequences.py",
    "esmfold2_normalize_coordinates.py": (f"{BIOHUB_ESM}/utils/structure/normalize_coordinates.py"),
    "esmfold2_output.py": f"{BIOHUB_ESM}/models/esmfold2/output.py",
    "esmfold2_paired_msa.py": f"{BIOHUB_ESM}/models/esmfold2/paired_msa.py",
    "esmfold2_parsing.py": f"{BIOHUB_ESM}/utils/parsing.py",
    "esmfold2_predicted_aligned_error.py": (
        f"{BIOHUB_ESM}/utils/structure/predicted_aligned_error.py"
    ),
    "esmfold2_prepare_input.py": f"{BIOHUB_ESM}/models/esmfold2/prepare_input.py",
    "esmfold2_processor.py": f"{BIOHUB_ESM}/models/esmfold2/processor.py",
    "esmfold2_protein_chain.py": f"{BIOHUB_ESM}/utils/structure/protein_chain.py",
    "esmfold2_protein_complex.py": f"{BIOHUB_ESM}/utils/structure/protein_complex.py",
    "esmfold2_protein_structure.py": (f"{BIOHUB_ESM}/utils/structure/protein_structure.py"),
    "esmfold2_residue_constants.py": f"{BIOHUB_ESM}/utils/residue_constants.py",
    "esmfold2_sequential_dataclass.py": f"{BIOHUB_ESM}/utils/sequential_dataclass.py",
    "esmfold2_system.py": f"{BIOHUB_ESM}/utils/system.py",
    "esmfold2_types.py": f"{BIOHUB_ESM}/models/esmfold2/types.py",
    "esmfold2_utils_types.py": f"{BIOHUB_ESM}/utils/types.py",
    "modeling_esmfold2.py": f"{BIOHUB_TRANSFORMERS}/modeling_esmfold2.py",
    "modeling_esmfold2_common.py": (f"{BIOHUB_TRANSFORMERS}/modeling_esmfold2_common.py"),
    "modeling_esmfold2_experimental.py": (
        f"{BIOHUB_TRANSFORMERS}/modeling_esmfold2_experimental.py"
    ),
    "protein_utils.py": f"{BIOHUB_TRANSFORMERS}/protein_utils.py",
    "reproducibility.py": f"{BIOHUB_ESM}/models/esmfold2/processor.py",
}
ORIGINAL_RUNTIME_MODULES = frozenset({"attention.py", "embedding.py"})


def _meaningful_lines(text: str) -> list[str]:
    return [
        " ".join(line.strip().split())
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_esmfold2_runtime_source_inventory_is_complete() -> None:
    runtime_modules = {path.name for path in RUNTIME.glob("*.py")}
    classified_modules = {*SOURCE_COUNTERPARTS, *ORIGINAL_RUNTIME_MODULES}
    assert runtime_modules == classified_modules, (
        "classify every ESMFold2 runtime module as original or bind it to its "
        "pinned upstream source counterpart; "
        f"missing={sorted(runtime_modules - classified_modules)}, "
        f"stale={sorted(classified_modules - runtime_modules)}"
    )


@pytest.mark.parametrize(
    ("runtime_name", "upstream_relative"),
    SOURCE_COUNTERPARTS.items(),
    ids=SOURCE_COUNTERPARTS,
)
def test_esmfold2_source_is_independently_organized(
    runtime_name: str, upstream_relative: str
) -> None:
    runtime_path = RUNTIME / runtime_name
    upstream_path = ROOT / upstream_relative
    assert upstream_path.is_file(), f"pinned source counterpart is missing: {upstream_relative}"
    runtime_text = runtime_path.read_text(encoding="utf-8")
    upstream_text = upstream_path.read_text(encoding="utf-8")
    assert runtime_text.encode() != upstream_text.encode()
    similarity = SequenceMatcher(
        None,
        _meaningful_lines(runtime_text),
        _meaningful_lines(upstream_text),
        autojunk=False,
    ).ratio()
    assert similarity < MAX_LINE_SIMILARITY, (
        f"{runtime_name} has line similarity {similarity:.3f} to {upstream_relative}; "
        "reimplement the public behavior instead of copying the parity oracle"
    )


def test_esmfold2_runtime_modules_do_not_import_upstream_packages() -> None:
    forbidden_roots = {"esm", "vendor"}
    for runtime_path in sorted(RUNTIME.glob("*.py")):
        source = runtime_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=runtime_path.name)
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
        assert imported_roots.isdisjoint(forbidden_roots), (
            f"{runtime_path.name} imports an upstream package: "
            f"{sorted(imported_roots & forbidden_roots)}"
        )
