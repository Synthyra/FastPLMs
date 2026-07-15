"""Pinned Biohub parity for ESMFold2 protein-chain and complex data APIs."""

from __future__ import annotations

import importlib.util
import inspect
import io
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from fastplms.models.esmfold2 import esmfold2_affine3d as local_affine
from fastplms.models.esmfold2 import esmfold2_aligner as local_aligner
from fastplms.models.esmfold2 import esmfold2_atom_indexer as local_atom_indexer
from fastplms.models.esmfold2 import esmfold2_metrics as local_metrics
from fastplms.models.esmfold2 import esmfold2_misc as local_misc
from fastplms.models.esmfold2 import esmfold2_mmcif_parsing as local_mmcif
from fastplms.models.esmfold2 import (
    esmfold2_normalize_coordinates as local_normalize,
)
from fastplms.models.esmfold2 import esmfold2_protein_chain as local_chain
from fastplms.models.esmfold2 import esmfold2_protein_complex as local_complex
from fastplms.models.esmfold2 import esmfold2_protein_structure as local_structure
from fastplms.models.esmfold2 import esmfold2_residue_constants as local_residues
from fastplms.models.esmfold2 import esmfold2_utils_types as local_types

pytestmark = [pytest.mark.compliance, pytest.mark.gpu, pytest.mark.structure]

ROOT = Path(__file__).resolve().parents[2]
BIOHUB_ESM = ROOT / "vendor/upstream/biohub-esm/esm"
_MISSING = object()
SOURCE_PAIRS = {
    "esmfold2_protein_chain.py": BIOHUB_ESM / "utils/structure/protein_chain.py",
    "esmfold2_protein_complex.py": BIOHUB_ESM / "utils/structure/protein_complex.py",
}


def _package(name: str) -> types.ModuleType:
    package = types.ModuleType(name)
    package.__path__ = []  # type: ignore[attr-defined]
    return package


@contextmanager
def _temporary_modules(modules: dict[str, types.ModuleType]) -> Iterator[None]:
    previous = {name: sys.modules.get(name, _MISSING) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module  # type: ignore[assignment]


def _load_source(
    module_name: str,
    path: Path,
    aliases: dict[str, types.ModuleType],
) -> types.ModuleType:
    assert path.is_file(), f"pinned source is missing: {path}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    with _temporary_modules({**aliases, module_name: module}):
        specification.loader.exec_module(module)
    return module


def _base_aliases() -> dict[str, types.ModuleType]:
    return {
        "esm": _package("esm"),
        "esm.utils": _package("esm.utils"),
        "esm.utils.structure": _package("esm.utils.structure"),
        "esm.utils.residue_constants": local_residues,
        "esm.utils.misc": local_misc,
        "esm.utils.structure.affine3d": local_affine,
        "esm.utils.structure.aligner": local_aligner,
        "esm.utils.structure.atom_indexer": local_atom_indexer,
        "esm.utils.structure.metrics": local_metrics,
        "esm.utils.structure.mmcif_parsing": local_mmcif,
        "esm.utils.structure.normalize_coordinates": local_normalize,
        "esm.utils.structure.protein_structure": local_structure,
        "esm.utils.types": local_types,
    }


@cache
def _official_chain() -> types.ModuleType:
    return _load_source(
        "_fastplms_pinned_biohub_protein_chain",
        SOURCE_PAIRS["esmfold2_protein_chain.py"],
        _base_aliases(),
    )


@cache
def _official_complex() -> types.ModuleType:
    official_chain = _official_chain()
    return _load_source(
        "_fastplms_pinned_biohub_protein_complex",
        SOURCE_PAIRS["esmfold2_protein_complex.py"],
        {
            **_base_aliases(),
            "esm.utils.structure.protein_chain": official_chain,
        },
    )


def _assert_equal(actual: Any, expected: Any) -> None:
    if isinstance(actual, np.ndarray):
        assert isinstance(expected, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_chain_equal(actual: Any, expected: Any) -> None:
    for field_name in (
        "id",
        "sequence",
        "chain_id",
        "entity_id",
        "residue_index",
        "insertion_code",
        "atom37_positions",
        "atom37_mask",
        "confidence",
        "atom37_confidence",
    ):
        _assert_equal(getattr(actual, field_name), getattr(expected, field_name))


def _assert_complex_equal(actual: Any, expected: Any) -> None:
    for field_name in (
        "id",
        "sequence",
        "entity_id",
        "chain_id",
        "sym_id",
        "residue_index",
        "insertion_code",
        "atom37_positions",
        "atom37_mask",
        "confidence",
        "atom37_confidence",
    ):
        _assert_equal(getattr(actual, field_name), getattr(expected, field_name))
    _assert_equal(actual.metadata.entity_lookup, expected.metadata.entity_lookup)
    _assert_equal(actual.metadata.chain_lookup, expected.metadata.chain_lookup)
    _assert_equal(
        actual.metadata.assembly_composition,
        expected.metadata.assembly_composition,
    )


def _atom37_coordinates(offset: float = 0.0) -> np.ndarray:
    X = np.full((4, 37, 3), np.nan, dtype=np.float32)
    for residue_index in range(4):
        x = offset + 3.8 * residue_index
        atoms = {
            "N": (x, 0.1, 0.0),
            "CA": (x + 1.4, 0.2, 0.1),
            "C": (x + 2.4, 1.1, 0.2),
            "O": (x + 2.1, 2.2, 0.3),
            "CB": (x + 1.5, -0.8, 1.0),
        }
        for atom_name, coordinate in atoms.items():
            X[residue_index, local_residues.atom_order[atom_name]] = coordinate
    return X


def _make_chain(module: types.ModuleType, *, chain_id: str, entity_id: int, offset: float):
    return module.ProteinChain.from_atom37(
        _atom37_coordinates(offset),
        id="fixture",
        sequence="AGST",
        chain_id=chain_id,
        entity_id=entity_id,
        residue_index=np.asarray([4, 5, 5, 9], dtype=np.int64),
        insertion_code=np.asarray(["", "", "A", ""]),
        confidence=np.asarray([0.91, 0.83, 0.72, 0.65], dtype=np.float32),
    )


def test_protein_data_source_inventory_is_pinned() -> None:
    runtime = ROOT / "src/fastplms/models/esmfold2"
    for runtime_name, upstream_path in SOURCE_PAIRS.items():
        assert (runtime / runtime_name).is_file()
        assert upstream_path.is_file()
    gitmodules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert "vendor/upstream/biohub-esm" in gitmodules


def test_public_class_surfaces_match_pinned_biohub() -> None:
    official_chain = _official_chain()
    official_complex = _official_complex()
    for actual, expected in (
        (local_chain.ProteinChain, official_chain.ProteinChain),
        (local_complex.ProteinComplex, official_complex.ProteinComplex),
    ):
        actual_names = {
            name
            for name, value in inspect.getmembers(actual)
            if not name.startswith("__") and (callable(value) or inspect.isdatadescriptor(value))
        }
        expected_names = {
            name
            for name, value in inspect.getmembers(expected)
            if not name.startswith("__") and (callable(value) or inspect.isdatadescriptor(value))
        }
        assert actual_names == expected_names


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1", [("1",)]),
        ("1-3", [("1",), ("2",), ("3",)]),
        ("(1-2)(4,6)", [("4", "1"), ("4", "2"), ("6", "1"), ("6", "2")]),
    ],
)
def test_assembly_operation_expansion_matches_pinned_biohub(
    expression: str, expected: list[tuple[str, ...]]
) -> None:
    official = _official_complex()
    assert local_complex._parse_operation_expression(expression) == expected
    assert official._parse_operation_expression(expression) == expected


def test_assembly_transform_application_matches_pinned_biohub() -> None:
    official = _official_complex()
    local_input = _make_chain(local_chain, chain_id="A", entity_id=1, offset=0.0)
    official_input = _make_chain(_official_chain(), chain_id="A", entity_id=1, offset=0.0)
    transforms = {
        "1": SimpleNamespace(rotation=np.eye(3), target_translation=np.asarray([1.0, -2.0, 0.5])),
        "2": SimpleNamespace(
            rotation=np.diag([-1.0, 1.0, -1.0]),
            target_translation=np.asarray([0.25, 0.5, 1.0]),
        ),
    }
    operations = [("2", "1"), ("1",)]
    actual = local_complex._apply_transformations_fast([local_input], transforms, operations)
    expected = official._apply_transformations_fast([official_input], transforms, operations)
    assert len(actual) == len(expected) == 2
    for actual_chain, expected_chain in zip(actual, expected, strict=True):
        _assert_chain_equal(actual_chain, expected_chain)


def test_chain_construction_slicing_and_atom_views_match_pinned_biohub() -> None:
    official = _official_chain()
    actual = _make_chain(local_chain, chain_id="Q", entity_id=7, offset=0.0)
    expected = _make_chain(official, chain_id="Q", entity_id=7, offset=0.0)
    _assert_chain_equal(actual, expected)
    selection = np.asarray([True, False, True, True])
    _assert_chain_equal(actual[selection], expected[selection])
    _assert_equal(actual.atoms[["N", "CA", "C"]], expected.atoms[["N", "CA", "C"]])
    _assert_equal(actual.atom_mask["CB"], expected.atom_mask["CB"])
    _assert_equal(actual.residue_index_no_insertions, expected.residue_index_no_insertions)
    _assert_equal(actual.cbeta_contacts(), expected.cbeta_contacts())


def test_chain_geometry_and_encoder_inputs_match_pinned_biohub() -> None:
    official = _official_chain()
    actual = _make_chain(local_chain, chain_id="Q", entity_id=7, offset=0.0)
    expected = _make_chain(official, chain_id="Q", entity_id=7, offset=0.0)
    _assert_chain_equal(actual.infer_cbeta(), expected.infer_cbeta())
    _assert_chain_equal(actual.infer_oxygen(), expected.infer_oxygen())
    _assert_chain_equal(actual.normalize_coordinates(), expected.normalize_coordinates())
    for actual_tensor, expected_tensor in zip(
        actual.to_structure_encoder_inputs(),
        expected.to_structure_encoder_inputs(),
        strict=True,
    ):
        _assert_equal(actual_tensor, expected_tensor)
    assert actual.rmsd(actual, only_compute_backbone_rmsd=True) == pytest.approx(
        expected.rmsd(expected, only_compute_backbone_rmsd=True)
    )


def test_chain_compact_storage_matches_pinned_biohub() -> None:
    official = _official_chain()
    actual = _make_chain(local_chain, chain_id="Q", entity_id=7, offset=0.0)
    expected = _make_chain(official, chain_id="Q", entity_id=7, offset=0.0)
    _assert_equal(actual.state_dict(), expected.state_dict())
    assert actual.to_blob() == expected.to_blob()
    _assert_chain_equal(
        local_chain.ProteinChain.from_blob(actual.to_blob()),
        official.ProteinChain.from_blob(expected.to_blob()),
    )


def test_chain_structure_interchange_matches_pinned_biohub() -> None:
    official = _official_chain()
    actual = _make_chain(local_chain, chain_id="Q", entity_id=7, offset=0.0)
    expected = _make_chain(official, chain_id="Q", entity_id=7, offset=0.0)
    assert actual.to_pdb_string() == expected.to_pdb_string()
    assert actual.to_mmcif_string() == expected.to_mmcif_string()
    _assert_equal(actual.atom_array.b_factor, expected.atom_array.b_factor)
    _assert_equal(actual.atom_array.occupancy, expected.atom_array.occupancy)
    _assert_chain_equal(
        local_chain.ProteinChain.from_pdb(io.StringIO(actual.to_pdb_string()), is_predicted=True),
        official.ProteinChain.from_pdb(io.StringIO(expected.to_pdb_string()), is_predicted=True),
    )


def _make_complexes() -> tuple[Any, Any]:
    official_chain = _official_chain()
    official_complex = _official_complex()
    local_chains = [
        _make_chain(local_chain, chain_id="A", entity_id=1, offset=0.0),
        _make_chain(local_chain, chain_id="B", entity_id=2, offset=2.0),
    ]
    official_chains = [
        _make_chain(official_chain, chain_id="A", entity_id=1, offset=0.0),
        _make_chain(official_chain, chain_id="B", entity_id=2, offset=2.0),
    ]
    return (
        local_complex.ProteinComplex.from_chains(local_chains),
        official_complex.ProteinComplex.from_chains(official_chains),
    )


def test_complex_construction_views_and_topology_match_pinned_biohub() -> None:
    actual, expected = _make_complexes()
    _assert_complex_equal(actual, expected)
    assert actual.chain_boundaries == expected.chain_boundaries
    _assert_equal(actual.chain_lengths, expected.chain_lengths)
    _assert_equal(actual.chain_adjacency(), expected.chain_adjacency())
    _assert_equal(actual.chain_adjacency_by_index(0), expected.chain_adjacency_by_index(0))
    _assert_chain_equal(actual.get_chain_by_index(1), expected.get_chain_by_index(1))
    mask = np.asarray([True, True, False, False, False, True, True, True, True])
    _assert_complex_equal(actual[mask], expected[mask])


def test_complex_geometry_and_compact_storage_match_pinned_biohub() -> None:
    actual, expected = _make_complexes()
    _assert_complex_equal(actual.infer_cbeta(), expected.infer_cbeta())
    _assert_complex_equal(actual.infer_oxygen(), expected.infer_oxygen())
    _assert_equal(actual.state_dict(), expected.state_dict())
    assert actual.to_blob() == expected.to_blob()
    _assert_complex_equal(
        local_complex.ProteinComplex.from_blob(actual.to_blob()),
        _official_complex().ProteinComplex.from_blob(expected.to_blob()),
    )
    assert actual.to_mmcif_string() == expected.to_mmcif_string()


def test_chain_to_complex_adapter_matches_pinned_biohub() -> None:
    official_chain = _official_chain()
    official_complex = _official_complex()
    actual_chain = _make_chain(local_chain, chain_id="A", entity_id=1, offset=0.0)
    expected_chain = _make_chain(official_chain, chain_id="A", entity_id=1, offset=0.0)
    actual = local_complex.protein_chain_to_protein_complex(actual_chain)
    expected = official_complex.protein_chain_to_protein_complex(expected_chain)
    _assert_complex_equal(actual, expected)
