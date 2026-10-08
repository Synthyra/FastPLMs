"""The ESMFold2 conformer store reads atom names, charges, bonds, leaving atoms, and reference positions from components.

A miniature dictionary of RDKit molecules stands in for the hash-pinned chemical component dictionary, which the offline
tests do not have. The store logic is the same for any mapping of identifiers to named molecules.

Shapes: `n` heavy atoms in a component.
"""

import numpy as np
import pytest

from rdkit import Chem
from tests.unit.synthetic_ccd import component, install_mini_ccd, mini_ccd

from fastplms.models.esmfold2 import esmfold2_conformers as conformers
from fastplms.models.esmfold2.esmfold2_constants import PROTEIN_RESIDUE_TO_RES_TYPE


@pytest.fixture
def store(monkeypatch) -> conformers._ChemicalComponentStore:
    return install_mini_ccd(monkeypatch)


def heavy_conformer_positions(molecule: Chem.Mol, conformer_index: int) -> dict[str, np.ndarray]:
    """Heavy-atom positions by name in one conformer, read straight from RDKit."""
    heavy = Chem.RemoveHs(molecule, sanitize=False)
    conformer = heavy.GetConformer(conformer_index)
    return {  # atom name -> (3,) position
        atom.GetProp("name"): np.asarray(list(conformer.GetAtomPosition(atom.GetIdx())), dtype=np.float32)
        for atom in heavy.GetAtoms()
    }


def test_the_loaded_dictionary_is_returned_as_it_is_without_resolving_the_asset(store):
    assert conformers.load_ccd() is store.molecules
    assert conformers.load_ccd("an/unused/directory") is store.molecules
    assert sorted(store.molecules) == ["ALA", "EOH", "GLY", "SEP", "ZN"]


def test_the_conformer_of_a_component_is_its_heavy_atoms_by_name(store):
    positions = store.conformer("EOH")  # name -> (3,)
    expected = heavy_conformer_positions(mini_ccd()["EOH"], 0)

    assert sorted(positions) == ["C1", "C2", "O"] and all(point.shape == (3,) and point.dtype == np.float32 for point in positions.values())
    for name, point in expected.items():
        np.testing.assert_allclose(positions[name], point, atol=1e-5)
    assert 1.2 < float(np.linalg.norm(positions["C1"] - positions["C2"])) < 1.8
    assert store.conformer("EOH") is store.conformers["EOH"]


def test_the_computed_conformer_is_preferred_over_the_ideal_one(store):
    positions = store.conformer("SEP")
    computed = heavy_conformer_positions(mini_ccd()["SEP"], 1)
    ideal = heavy_conformer_positions(mini_ccd()["SEP"], 0)

    assert sorted(positions) == sorted(computed)
    for name, point in computed.items():
        np.testing.assert_allclose(positions[name], point, atol=1e-5)
    assert any(not np.allclose(positions[name], ideal[name], atol=1e-3) for name in ideal)


def test_a_component_that_is_missing_without_a_conformer_or_without_named_atoms_has_no_conformer(store):
    store.molecules["BARE"] = Chem.MolFromSmiles("CC")
    store.molecules["NAMELESS"] = component("CCO", ["C1", "", "O"])

    assert store.conformer("NOPE") is None and store.conformer("BARE") is None
    assert sorted(store.conformer("NAMELESS")) == ["C1", "O"]
    assert store.atom_records("NOPE") is None and store.bond_records("NOPE") is None and store.atom_records("BARE") is None


def test_atom_records_list_each_named_heavy_atom_with_its_element_and_charge(store):
    assert store.atom_records("EOH") == [("C1", "C", 0), ("C2", "C", 0), ("O", "O", 0)]
    assert store.atom_records("ZN") == [("ZN", "Zn", 2)]
    assert [name for name, _element, _charge in store.atom_records("SEP")] == [
        "N", "CA", "CB", "OG", "P", "O1P", "O2P", "O3P", "C", "O", "OXT",
    ]
    assert store.atom_records("EOH") is store.atoms["EOH"]


def test_bond_records_pair_the_names_of_bonded_heavy_atoms(store):
    assert store.bond_records("EOH") == [("C1", "C2"), ("C2", "O")]
    assert store.bond_records("ZN") is None
    assert ("P", "O3P") in store.bond_records("SEP") and ("C", "OXT") in store.bond_records("SEP")


def test_leaving_atoms_are_the_names_flagged_in_the_component(store):
    assert store.component_leaving_atoms("SEP") == {"OXT"}
    assert store.component_leaving_atoms("EOH") == set() and store.component_leaving_atoms("NOPE") == set()
    assert store.component_leaving_atoms("SEP") is store.leaving_atoms["SEP"]


def test_the_module_level_accessors_read_the_active_store(store):
    alanine = PROTEIN_RESIDUE_TO_RES_TYPE["ALA"]

    first = conformers.get_idealized_atom_pos(alanine, "CA")
    ligand_position = conformers.get_ligand_idealized_atom_pos("EOH", "O")

    assert conformers.get_ccd_conformer("EOH") is store.conformers["EOH"]
    assert first.shape == (3,) and conformers.get_idealized_atom_pos(alanine, "CA") is first
    assert conformers.get_idealized_atom_pos(alanine, "XYZ") is None and conformers.get_idealized_atom_pos(9999, "CA") is None
    np.testing.assert_allclose(ligand_position, store.conformer("EOH")["O"])
    assert conformers.get_ligand_idealized_atom_pos("EOH", "Q") is None and conformers.get_ligand_idealized_atom_pos("NOPE", "O") is None
    assert conformers.get_ligand_ccd_atoms_with_charges("ZN") == [("ZN", "Zn", 2)]
    assert conformers.get_ligand_ccd_bonds("EOH") == [("C1", "C2"), ("C2", "O")]
    assert conformers.get_ccd_leaving_atoms("SEP") == {"OXT"}
