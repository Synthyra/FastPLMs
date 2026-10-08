"""ESMFold2 mmCIF parsing: non-polymer entities and coordinates read from a synthetic structure file.

The file holds two protein chains and one zinc ion written by the molecular-complex writer plus a residue scheme, so the
parser sees a polymer table, a non-polymer entity, and a heteroatom group.
"""

import io
import numpy as np
import pytest

from tests.unit.synthetic_structures import cif_column, synthetic_chain, synthetic_mmcif

from fastplms.models.esmfold2 import esmfold2_mmcif_parsing as parsing
from fastplms.models.esmfold2.esmfold2_mmcif_parsing import MmcifWrapper
from fastplms.models.esmfold2.esmfold2_protein_chain import ProteinChain


ZINC_POSITION = np.array([100.0, 100.0, 100.0], dtype=np.float32)


@pytest.fixture
def wrapper() -> MmcifWrapper:
    chains = [synthetic_chain("ACDEFGHIKL"), synthetic_chain("MNPQRSTVWY", chain_id="B", entity_id=2, shift=12.0, seed=1)]
    return MmcifWrapper.read(io.StringIO(synthetic_mmcif(chains, zinc_at=ZINC_POSITION)), "syn")


def test_entities_of_the_non_polymer_kinds_are_found_by_their_type_in_any_case():
    block = io.StringIO(
        "data_x\n#\nloop_\n_entity.id\n_entity.type\n1 polymer\n2 non-polymer\n3 WATER\n4 branched\n#\n"
    )
    cif = parsing.pdbx.CIFFile.read(block)

    assert parsing._nonpolymer_entity_ids(cif.block) == {"2", "3", "4"}
    assert parsing._nonpolymer_entity_ids(parsing.pdbx.CIFFile.read(io.StringIO("data_y\n#\n")).block) == set()


def test_non_polymer_components_are_mapped_from_the_table_only_for_non_polymer_entities():
    table = (
        "data_x\n#\nloop_\n_pdbx_entity_nonpoly.entity_id\n_pdbx_entity_nonpoly.comp_id\n2 ZN\n3 HOH\n4 NAG\n#\n"
    )
    cif = parsing.pdbx.CIFFile.read(io.StringIO(table))
    bare = parsing.pdbx.CIFFile.read(io.StringIO("data_y\n#\n"))

    assert parsing._nonpolymer_component_map(cif.block, {"2", "4"}) == {"2": "ZN", "4": "NAG"}
    assert parsing._nonpolymer_component_map(cif.block, set()) == {} and parsing._nonpolymer_component_map(bare.block, {"2"}) == {}


def test_the_atoms_of_each_non_polymer_component_are_gathered_by_component_and_chain(wrapper):
    coordinates = wrapper.non_polymer_coords
    parsed = wrapper._parse_nonpoly_from_mmcif()

    assert list(coordinates) == [("ZN", "Z")] and list(parsed) == [("ZN", "Z")]
    assert len(coordinates[("ZN", "Z")]) == 1
    np.testing.assert_allclose(coordinates[("ZN", "Z")].coord[0], ZINC_POSITION, atol=1e-3)
    assert wrapper.non_polymer_coords is coordinates


def test_the_fallback_gathers_every_residue_that_is_not_a_standard_one(wrapper):
    fallback = wrapper._parse_nonpoly_fallback()

    assert list(fallback) == [("ZN", "Z")] and fallback[("ZN", "Z")].res_name.tolist() == ["ZN"]


def test_the_fallback_is_used_when_the_table_based_search_fails(wrapper, monkeypatch):
    def fail():
        raise ValueError("no entity table")

    monkeypatch.setattr(wrapper, "_parse_nonpoly_from_mmcif", fail)

    assert list(wrapper.non_polymer_coords) == [("ZN", "Z")]


def test_a_non_polymer_far_from_the_chain_makes_no_contacts_for_a_chain_read_with_its_source():
    text = io.StringIO(
        synthetic_mmcif(
            [synthetic_chain("ACDEFGHIKL")],
            zinc_at=ZINC_POSITION,
        )
    )

    chain = ProteinChain.from_mmcif(text, keep_source=True)

    assert chain.find_nonpolymer_contacts() == [] and ("ZN", "Z") in chain.mmcif.non_polymer_coords


def test_the_scheme_of_a_chain_maps_each_sequence_position_to_its_author_numbering(wrapper):
    mapping = wrapper.seqres_to_structure["A"]

    assert [mapping[index].residue_number for index in range(10)] == list(range(1, 11))
    assert all(mapping[index].insertion_code == "" and not mapping[index].hetflag for index in range(10))
    assert wrapper.chain_to_seqres["B"] == "MNPQRSTVWY" and wrapper.entities[3] == ["Z"]


def test_a_duplicate_author_number_is_renumbered_in_sequence_order_and_a_missing_one_stays_empty():
    scheme = (
        "data_x\n#\nloop_\n_pdbx_poly_seq_scheme.asym_id\n_pdbx_poly_seq_scheme.seq_id\n_pdbx_poly_seq_scheme.auth_seq_num\n"
        "A 1 5\nA 2 5\nA 3 ?\n#\n"
    )
    category = parsing.pdbx.CIFFile.read(io.StringIO(scheme)).block["pdbx_poly_seq_scheme"]

    per_chain, asym_to_author = parsing._scheme_residue_map(category)
    parsing._renumber_duplicate_residues(per_chain)

    assert asym_to_author == {"A": "A"}
    assert [per_chain["A"][position].residue_number for position in range(3)] == [5, 6, None]


def test_a_category_can_be_replaced_column_by_column_and_rounded_for_export(wrapper):
    atom_site = wrapper.raw.block["atom_site"]
    atom_site["Cartn_x"] = cif_column(["1.23456", "nan"] + ["0.5"] * (len(atom_site["Cartn_x"]) - 2))

    parsing.round_mmcif_columns(wrapper.raw)

    rounded = atom_site["Cartn_x"].as_array(str)
    assert rounded[0] == "1.235" and rounded[1] == "?" and rounded[2] == "0.500"
