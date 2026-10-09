"""Every model family keeps a `checkpoint`-marked test that loads its pinned weights and compares them.

Those tests read a real checkpoint and do not run in the default selection, so nothing else notices when one loses its
marker, is renamed, or disappears. Each entry below is a test that loads the family's weights and compares them with the
official checkpoint, a checked-in golden, or the exact-load invariant of ``tests/unit/checkpoint_cache.py``.
"""

from __future__ import annotations

import ast
import pytest

from pathlib import Path

from fastplms.registry import get_model_registry


ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_GOLDEN = "tests/integration/test_official_goldens.py::test_declared_sequence_golden_matches_candidate"
CACHED_BUFFERS = "tests/unit/test_pretrained_buffers.py::test_cached_checkpoint_loads_with_constructed_unsaved_tensors"
WEIGHT_LOAD_TESTS = {
    "ankh": (SEQUENCE_GOLDEN,),
    "boltz2": ("tests/unit/test_boltz_checkpoint_io.py::test_pinned_checkpoint_loads_every_saved_tensor_and_nothing_else",),
    "dplm": (SEQUENCE_GOLDEN, CACHED_BUFFERS),
    "dplm2": (SEQUENCE_GOLDEN, CACHED_BUFFERS),
    "e1": (
        SEQUENCE_GOLDEN,
        "tests/unit/test_e1_official_equivalence.py::test_the_official_checkpoint_loads_completely_and_matches_e1",
    ),
    "esm2": (
        SEQUENCE_GOLDEN,
        "tests/unit/test_esm2_official_equivalence.py::test_the_official_checkpoint_loads_key_for_key",
    ),
    "esm3": (SEQUENCE_GOLDEN, CACHED_BUFFERS),
    "esm_plusplus": (
        SEQUENCE_GOLDEN,
        "tests/unit/test_esmc_official_equivalence.py::test_the_official_checkpoint_loads_key_for_key",
    ),
    "esmfold": ("tests/structure/test_structure_official_goldens.py::test_esmfold_candidate_matches_checked_structure_golden",),
    "esmfold2": ("tests/structure/test_structure_official_goldens.py::test_esmfold2_candidate_matches_checked_structure_golden",),
}
LOCATIONS = [pytest.param(family, location, id=f"{family}-{location.rsplit('::', 1)[1]}") for family, items in WEIGHT_LOAD_TESTS.items() for location in items]


def _marker_names(nodes: list[ast.expr]) -> set[str]:
    names: set[str] = set()
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Attribute) and child.value.attr == "mark":
                names.add(child.attr)
    return names


def _marks_of_test(path: Path, test_name: str) -> set[str] | None:
    """The marker names that apply to ``test_name`` in ``path``, or None when the file holds no such function."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_marks: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in statement.targets):
            module_marks |= _marker_names([statement.value])

    def search(body: list[ast.stmt], inherited: set[str]) -> set[str] | None:
        for statement in body:
            if isinstance(statement, ast.ClassDef):
                found = search(statement.body, inherited | _marker_names(statement.decorator_list))
                if found is not None:
                    return found
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == test_name:
                return inherited | _marker_names(statement.decorator_list)
        return None

    return search(tree.body, module_marks)


def test_every_family_in_the_manifest_has_weight_load_tests() -> None:
    families = {spec.family.id for spec in get_model_registry().values()}

    assert families == set(WEIGHT_LOAD_TESTS), (
        f"families without weight-load tests: {sorted(families - set(WEIGHT_LOAD_TESTS))}; "
        f"entries for families the manifest dropped: {sorted(set(WEIGHT_LOAD_TESTS) - families)}"
    )
    assert all(WEIGHT_LOAD_TESTS.values())


@pytest.mark.parametrize(("family", "location"), LOCATIONS)
def test_each_named_weight_load_test_exists_and_carries_the_checkpoint_marker(family: str, location: str) -> None:
    relative, test_name = location.split("::")

    marks = _marks_of_test(ROOT / relative, test_name)

    assert marks is not None, f"{family}: {relative} defines no {test_name}"
    assert "checkpoint" in marks, f"{family}: {location} is not marked checkpoint"


def test_the_marker_search_reads_function_class_and_module_marks(tmp_path: Path) -> None:
    source = tmp_path / "test_marked.py"
    source.write_text(
        "import pytest\n"
        "pytestmark = pytest.mark.gpu\n"
        "@pytest.mark.checkpoint\n"
        "def test_direct(): pass\n"
        "@pytest.mark.slow\n"
        "class TestGroup:\n"
        "    @pytest.mark.parametrize('x', [1])\n"
        "    def test_member(self, x): pass\n",
        encoding="utf-8",
    )

    assert _marks_of_test(source, "test_direct") == {"gpu", "checkpoint"}
    assert _marks_of_test(source, "test_member") == {"gpu", "slow", "parametrize"}
    assert _marks_of_test(source, "test_absent") is None
