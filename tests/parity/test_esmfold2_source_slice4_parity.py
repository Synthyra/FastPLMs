"""Pinned Biohub parity for ESMFold2 input preparation and flat complexes."""

from __future__ import annotations

import importlib.util
import sys
import types
import numpy as np
import pytest
import torch
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastplms.models.esmfold2 import esmfold2_conformers as local_conformers
from fastplms.models.esmfold2 import esmfold2_constants as local_constants
from fastplms.models.esmfold2 import esmfold2_metrics as local_metrics
from fastplms.models.esmfold2 import esmfold2_mmcif_parsing as local_mmcif
from fastplms.models.esmfold2 import esmfold2_molecular_complex as local_complex
from fastplms.models.esmfold2 import esmfold2_paired_msa as local_paired_msa
from fastplms.models.esmfold2 import esmfold2_prepare_input as local_prepare
from fastplms.models.esmfold2 import esmfold2_protein_complex as local_protein_complex
from fastplms.models.esmfold2 import esmfold2_residue_constants as local_residues
from fastplms.models.esmfold2 import esmfold2_types as local_types


pytestmark = [pytest.mark.compliance, pytest.mark.gpu, pytest.mark.structure]

ROOT = Path(__file__).resolve().parents[2]
BIOHUB_ESM = ROOT / "vendor/upstream/biohub-esm/esm"
_MISSING = object()


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


def _biohub_packages() -> dict[str, types.ModuleType]:
    return {
        "esm": _package("esm"),
        "esm.models": _package("esm.models"),
        "esm.models.esmfold2": _package("esm.models.esmfold2"),
        "esm.utils": _package("esm.utils"),
        "esm.utils.structure": _package("esm.utils.structure"),
    }


def _prepare_aliases() -> dict[str, types.ModuleType]:
    return {
        **_biohub_packages(),
        "esm.models.esmfold2.conformers": local_conformers,
        "esm.models.esmfold2.constants": local_constants,
        "esm.models.esmfold2.paired_msa": local_paired_msa,
        "esm.models.esmfold2.types": local_types,
    }


def _official_prepare() -> types.ModuleType:
    return _load_source(
        "_fastplms_pinned_biohub_prepare_input",
        BIOHUB_ESM / "models/esmfold2/prepare_input.py",
        _prepare_aliases(),
    )


def _official_complex() -> types.ModuleType:
    aliases = _complex_aliases()
    return _load_source(
        "_fastplms_pinned_biohub_molecular_complex",
        BIOHUB_ESM / "utils/structure/molecular_complex.py",
        aliases,
    )


def _complex_aliases() -> dict[str, types.ModuleType]:
    return {
        **_biohub_packages(),
        "esm.utils.residue_constants": local_residues,
        "esm.utils.structure.metrics": local_metrics,
        "esm.utils.structure.mmcif_parsing": local_mmcif,
        "esm.utils.structure.protein_complex": local_protein_complex,
    }


def _assert_value_equal(actual: Any, expected: Any) -> None:
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


def _assert_records_equal(actual: Any, expected: Any) -> None:
    assert vars(actual).keys() == vars(expected).keys()
    for field_name, actual_value in vars(actual).items():
        _assert_value_equal(actual_value, vars(expected)[field_name])


def _assert_prepared_equal(
    actual: tuple[list[Any], list[Any], list[Any]],
    expected: tuple[list[Any], list[Any], list[Any]],
) -> None:
    actual_chains, actual_tokens, actual_atoms = actual
    expected_chains, expected_tokens, expected_atoms = expected
    assert len(actual_chains) == len(expected_chains)
    assert len(actual_tokens) == len(expected_tokens)
    assert len(actual_atoms) == len(expected_atoms)
    for actual_token, expected_token in zip(actual_tokens, expected_tokens, strict=True):
        _assert_records_equal(actual_token, expected_token)
    for actual_atom, expected_atom in zip(actual_atoms, expected_atoms, strict=True):
        _assert_records_equal(actual_atom, expected_atom)
    for actual_chain, expected_chain in zip(actual_chains, expected_chains, strict=True):
        for field_name in (
            "chain_id",
            "asym_id",
            "entity_id",
            "sym_id",
            "mol_type",
            "ligand_bonds",
        ):
            _assert_value_equal(
                getattr(actual_chain, field_name), getattr(expected_chain, field_name)
            )
        assert [vars(token) for token in actual_chain.tokens] == [
            vars(token) for token in expected_chain.tokens
        ]


def _install_fake_ccd(monkeypatch: pytest.MonkeyPatch, official: types.ModuleType) -> None:
    atom_records = {
        "MSE": [
            ("N", "N", 0),
            ("CA", "C", 0),
            ("C", "C", 0),
            ("O", "O", 0),
            ("SE", "Se", 0),
        ],
        "PSU": [("P", "P", 0), ("C1'", "C", 0), ("N1", "N", 0)],
        "LIG": [("C1", "C", 0), ("N1", "N", 1), ("O1", "O", -1)],
    }
    bonds = {
        "MSE": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "SE")],
        "PSU": [("P", "C1'"), ("C1'", "N1")],
        "LIG": [("C1", "N1"), ("N1", "O1")],
    }

    def idealized(residue_type: int, atom_name: str) -> np.ndarray:
        base = residue_type + sum(map(ord, atom_name)) / 1000
        return np.asarray([base, base + 1, base + 2], dtype=np.float32)

    def ligand_position(residue_name: str, atom_name: str) -> np.ndarray:
        base = (sum(map(ord, residue_name + atom_name)) % 97) / 10
        return np.asarray([base, base + 0.5, base + 1], dtype=np.float32)

    replacements = {
        "get_idealized_atom_pos": idealized,
        "get_ligand_idealized_atom_pos": ligand_position,
        "get_ligand_ccd_atoms_with_charges": atom_records.get,
        "get_ligand_ccd_bonds": bonds.get,
        "get_ccd_leaving_atoms": lambda name: {"O1"} if name == "LIG" else set(),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(local_prepare, name, replacement)
        monkeypatch.setattr(official, name, replacement)


def _mixed_input() -> local_types.StructurePredictionInput:
    msa = local_types.MSA.from_sequences(["AMG", "A-G"])
    return local_types.StructurePredictionInput(
        sequences=[
            local_types.ProteinInput(
                id=["A", "B"],
                sequence="AMG",
                modifications=[local_types.Modification(position=1, ccd="MSE")],
                msa=msa,
            ),
            local_types.DNAInput(id="C", sequence="ATN"),
            local_types.RNAInput(
                id="D",
                sequence="AUN",
                modifications=[local_types.Modification(position=1, ccd="PSU")],
            ),
            local_types.LigandInput(id="E", ccd=["LIG"]),
        ],
        covalent_bonds=[
            local_types.CovalentBond(
                chain_id1="A",
                res_idx1=1,
                atom_idx1=0,
                chain_id2="E",
                res_idx2=0,
                atom_idx2=0,
            )
        ],
    )


def test_mixed_input_pipeline_matches_pinned_biohub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert torch.cuda.is_available(), "the ESMFold2 parity suite requires CUDA"
    official = _official_prepare()
    _install_fake_ccd(monkeypatch, official)
    input_value = _mixed_input()
    actual_parts = local_prepare.build_chains_from_input(input_value, seed=71)
    expected_parts = official.build_chains_from_input(input_value, seed=71)
    _assert_prepared_equal(actual_parts, expected_parts)
    with _temporary_modules(_prepare_aliases()):
        actual = local_prepare.build_feature_tensors(*actual_parts, input_value)
        expected = official.build_feature_tensors(*expected_parts, input_value)
    assert actual.keys() == expected.keys()
    for feature_name in actual:
        _assert_value_equal(actual[feature_name], expected[feature_name])


def test_distogram_conditioning_matches_pinned_biohub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    official = _official_prepare()
    _install_fake_ccd(monkeypatch, official)
    msa = local_types.MSA.from_sequences(["ACD"])
    input_value = local_types.StructurePredictionInput(
        sequences=[local_types.ProteinInput(id="A", sequence="ACD", msa=msa)],
        distogram_conditioning=[
            local_types.DistogramConditioning(
                chain_id="A",
                distogram=np.asarray(
                    [[0.0, 4.0, 12.0], [4.0, 0.0, 22.0], [12.0, 22.0, 0.0]],
                    dtype=np.float32,
                ),
            )
        ],
    )
    actual_parts = local_prepare.build_chains_from_input(input_value)
    expected_parts = official.build_chains_from_input(input_value)
    actual = local_prepare.compute_distogram_conditioning(
        input_value, actual_parts[0], actual_parts[1], torch.zeros(3, 3)
    )
    expected = official.compute_distogram_conditioning(
        input_value, expected_parts[0], expected_parts[1], torch.zeros(3, 3)
    )
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        _assert_value_equal(actual_tensor, expected_tensor)


def test_smiles_topology_matches_pinned_biohub() -> None:
    official = _official_prepare()
    arguments = {
        "smiles": "CC(=O)N",
        "entity_id": 2,
        "asym_id": 3,
        "sym_id": 0,
        "token_offset": 7,
        "atom_offset": 19,
        "space_uid_offset": 5,
        "seed": 919,
    }
    actual_tokens, actual_atoms, actual_bonds = local_prepare.tokenize_ligand_smiles(**arguments)
    expected_tokens, expected_atoms, expected_bonds = official.tokenize_ligand_smiles(**arguments)
    for actual, expected in zip(actual_tokens, expected_tokens, strict=True):
        _assert_records_equal(actual, expected)
    for actual, expected in zip(actual_atoms, expected_atoms, strict=True):
        _assert_records_equal(actual, expected)
    assert actual_bonds == expected_bonds


def _protein_fixture() -> local_protein_complex.ProteinComplex:
    sequence = "AC|GG"
    n_positions = len(sequence)
    # positions: (n_positions, 37, 3)
    positions = np.full((n_positions, 37, 3), np.nan, dtype=np.float32)
    # mask: (n_positions, 37)
    mask = np.zeros((n_positions, 37), dtype=bool)
    atom_names = ("N", "CA", "C", "O", "CB", "SG")
    for sequence_index in (0, 1, 3, 4):
        for atom_offset, atom_name in enumerate(atom_names):
            atom_index = local_residues.atom_order[atom_name]
            positions[sequence_index, atom_index] = np.asarray(
                [sequence_index, atom_offset, sequence_index + atom_offset / 10],
                dtype=np.float32,
            )
            mask[sequence_index, atom_index] = True
    return local_protein_complex.ProteinComplex(
        id="fixture",
        sequence=sequence,
        entity_id=np.asarray([0, 0, -1, 1, 1], dtype=np.int64),
        chain_id=np.asarray([0, 0, -1, 1, 1], dtype=np.int64),
        sym_id=np.zeros(n_positions, dtype=np.int64),
        residue_index=np.asarray([1, 2, 0, 1, 2], dtype=np.int64),
        insertion_code=np.asarray([""] * n_positions, dtype=object),
        atom37_positions=positions,
        atom37_mask=mask,
        confidence=np.asarray([0.8, 0.7, 0.0, 0.9, 0.6], dtype=np.float32),
        metadata=local_protein_complex.ProteinComplexMetadata(
            entity_lookup={0: 0, 1: 1},
            chain_lookup={0: "A", 1: "ligand_1"},
            assembly_composition={"1": ["A", "ligand_1"]},
        ),
    )


def _assert_complex_equal(actual: Any, expected: Any) -> None:
    assert actual.id == expected.id
    assert actual.sequence == expected.sequence
    for field_name in (
        "atom_positions",
        "atom_elements",
        "token_to_atoms",
        "chain_id",
        "plddt",
        "atom_names",
        "atom_hetero",
    ):
        _assert_value_equal(getattr(actual, field_name), getattr(expected, field_name))
    assert vars(actual.metadata) == vars(expected.metadata)


def _assert_protein_equal(actual: Any, expected: Any) -> None:
    assert actual.id == expected.id
    assert actual.sequence == expected.sequence
    for field_name in (
        "entity_id",
        "chain_id",
        "sym_id",
        "residue_index",
        "insertion_code",
        "atom37_positions",
        "atom37_mask",
        "confidence",
    ):
        _assert_value_equal(getattr(actual, field_name), getattr(expected, field_name))
    assert actual.metadata.entity_lookup == expected.metadata.entity_lookup
    assert actual.metadata.chain_lookup == expected.metadata.chain_lookup
    assert actual.metadata.assembly_composition == expected.metadata.assembly_composition


def test_molecular_complex_conversion_extends_pinned_biohub_with_identity_storage() -> None:
    official = _official_complex()
    protein = _protein_fixture()
    actual = local_complex.MolecularComplex.from_protein_complex(protein)
    with _temporary_modules(_complex_aliases()):
        expected = official.MolecularComplex.from_protein_complex(protein)
    _assert_complex_equal(actual, expected)
    _assert_records_equal(actual[1], expected[1])
    with _temporary_modules(_complex_aliases()):
        expected_protein = expected.to_protein_complex()
    _assert_protein_equal(actual.to_protein_complex(), expected_protein)

    actual_mmcif = actual.to_mmcif()
    expected_mmcif = expected.to_mmcif()
    assert actual_mmcif == expected_mmcif
    _assert_complex_equal(
        local_complex.MolecularComplex.from_mmcif(expected_mmcif, id="roundtrip"),
        official.MolecularComplex.from_mmcif(expected_mmcif, id="roundtrip"),
    )
    actual_state = actual.state_dict()
    identity_state = {key: actual_state.pop(key) for key in ("entity_id", "sym_id")}
    assert actual_state == expected.state_dict()
    assert identity_state == {
        "entity_id": [0, 0, 1, 1],
        "sym_id": [0, 0, 0, 0],
    }
    restored = local_complex.MolecularComplex.from_blob(actual.to_blob())
    _assert_complex_equal(restored, actual)
    _assert_value_equal(restored.entity_id, actual.entity_id)
    _assert_value_equal(restored.sym_id, actual.sym_id)
    _assert_complex_equal(
        local_complex.MolecularComplex.from_blob(expected.to_blob()),
        official.MolecularComplex.from_blob(expected.to_blob()),
    )


def test_molecular_complex_metrics_match_pinned_biohub() -> None:
    official = _official_complex()
    protein = _protein_fixture()
    actual = local_complex.MolecularComplex.from_protein_complex(protein)
    with _temporary_modules(_complex_aliases()):
        expected = official.MolecularComplex.from_protein_complex(protein)
    shifted_positions = actual.atom_positions.copy()
    shifted_positions[:, 0] += np.linspace(0.0, 0.2, len(shifted_positions))
    actual_target = local_complex.MolecularComplex(
        **{**vars(actual), "atom_positions": shifted_positions}
    )
    expected_target = official.MolecularComplex(
        **{**vars(expected), "atom_positions": shifted_positions}
    )
    assert actual.rmsd(actual_target) == expected.rmsd(expected_target)
    assert actual.lddt_ca(actual_target) == expected.lddt_ca(expected_target)
