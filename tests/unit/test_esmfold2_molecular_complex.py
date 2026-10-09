"""ESMFold2 molecular complexes: token and atom views, mmCIF reading, entity tables, scoring, and DockQ.

A molecular complex is the flat token and atom table the folding model returns. These tests build one from synthetic
protein chains (and a zinc ion for non-protein tokens), write it as mmCIF, and read it back. DockQ is an external
program; its report is stubbed with the text format that the scoring methods parse.

Shapes: `l` tokens, `a` atoms.
"""

import io
import numpy as np
import pytest

from dataclasses import replace
from biotite.structure.io.pdbx import CIFFile
from tests.unit.synthetic_structures import (
    dockq_report,
    stub_dockq,
    synthetic_chain,
    synthetic_mmcif,
    with_zinc,
)

from fastplms.models.esmfold2 import esmfold2_molecular_complex as molecular
from fastplms.models.esmfold2 import esmfold2_protein_complex as protein_complex_module
from fastplms.models.esmfold2.esmfold2_mmcif_parsing import MmcifWrapper
from fastplms.models.esmfold2.esmfold2_molecular_complex import MolecularComplex
from fastplms.models.esmfold2.esmfold2_protein_complex import ProteinComplex


FIRST = "ACDEFGHIKL"
SECOND = "MNPQRSTVWY"
THREE_LETTER = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
ZINC_POSITION = np.array([100.0, 100.0, 100.0], dtype=np.float32)


@pytest.fixture
def chains():
    return [synthetic_chain(FIRST), synthetic_chain(SECOND, chain_id="B", entity_id=2, shift=12.0, seed=1)]


@pytest.fixture
def base(chains) -> MolecularComplex:
    return MolecularComplex.from_protein_complex(ProteinComplex.from_chains(chains))


@pytest.fixture
def mmcif_text(chains) -> str:
    return synthetic_mmcif(chains, zinc_at=ZINC_POSITION)


def moved(structure: MolecularComplex) -> MolecularComplex:
    """The structure turned a quarter turn about z and shifted, atom by atom."""
    turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)  # (3, 3)
    return replace(structure, atom_positions=(structure.atom_positions @ turn.T + 3.0).astype(np.float32))


def test_a_molecular_complex_counts_its_tokens_and_exposes_the_atoms_of_each(base):
    first = base[0]

    assert len(base) == 20 and base.atom_coordinates is base.atom_positions
    assert first.token == "ALA" and first.token_idx == 0 and first.atom_positions.shape == (5, 3)
    assert sorted(first.atom_names) == ["C", "CA", "CB", "N", "O"] and not first.atom_hetero.any()
    assert first.confidence == base.plddt[0] and base[19].token == "TYR"
    for index in (-1, 20):
        with pytest.raises(IndexError, match="out of range for 20 tokens"):
            base[index]


def test_a_complex_reads_from_mmcif_text_or_a_file_with_tokens_ordered_by_chain_and_residue(mmcif_text, tmp_path):
    path = tmp_path / "model.cif"
    path.write_text(mmcif_text, encoding="utf-8")

    from_text = MolecularComplex.from_mmcif(mmcif_text, id="read")
    from_file = MolecularComplex.from_mmcif(str(path))
    anonymous = MolecularComplex.from_mmcif(mmcif_text)

    assert [str(token) for token in from_text.sequence] == [*THREE_LETTER, "ZN"] and len(from_text) == 21
    assert (from_text.id, from_file.id, anonymous.id) == ("read", "model", "complex_from_string")
    assert from_text.metadata.chain_lookup == {0: "A", 1: "B", 2: "Z"} and from_text.chain_id.tolist() == [0] * 10 + [1] * 10 + [2]
    assert from_text.atom_hetero[-1] and not from_text.atom_hetero[:-1].any() and str(from_text.atom_names[-1]) == "ZN"
    np.testing.assert_allclose(from_text.plddt[:10], synthetic_chain(FIRST).confidence, atol=1e-2)
    np.testing.assert_allclose(from_text.atom_positions[-1], ZINC_POSITION, atol=1e-3)
    assert from_text.token_to_atoms[-1].tolist() == [len(from_text.atom_positions) - 1, len(from_text.atom_positions)]


def test_water_is_left_out_when_a_complex_is_read_from_mmcif(base):
    wet = replace(with_zinc(base, (5.0, 5.0, 5.0)), sequence=[*base.sequence, "HOH"])

    read = MolecularComplex.from_mmcif(wet.to_mmcif())

    assert len(read) == 20 and "HOH" not in [str(token) for token in read.sequence]


def test_a_cif_source_is_read_from_a_path_when_one_exists_and_from_text_otherwise(mmcif_text, tmp_path):
    path = tmp_path / "model.cif"
    path.write_text(mmcif_text, encoding="utf-8")

    from_path = molecular._read_cif(str(path))
    from_text = molecular._read_cif(mmcif_text)

    assert "atom_site" in from_path.block and "atom_site" in from_text.block
    assert len(from_path.block["atom_site"]["id"]) == len(from_text.block["atom_site"]["id"])


def test_label_asym_ids_are_kept_only_when_they_cover_exactly_the_structure_atoms(mmcif_text):
    cif = CIFFile.read(io.StringIO(mmcif_text))
    structure = molecular._read_structure(cif)
    empty = CIFFile.read(io.StringIO("data_empty\n#\n"))

    labels = molecular._label_asym_ids(cif, len(structure))

    assert len(labels) == len(structure) and set(labels) == {"A", "B", "Z"} and labels[0] == "A"
    assert molecular._label_asym_ids(cif, len(structure) - 1) is None and molecular._label_asym_ids(empty, 1) is None


def test_entity_metadata_is_empty_for_a_file_without_an_entity_table():
    empty = CIFFile.read(io.StringIO("data_empty\n#\n"))

    assert molecular._entity_metadata(empty) == {}


def test_structure_atoms_are_grouped_by_chain_and_residue_then_flattened_in_sorted_order(mmcif_text):
    cif = CIFFile.read(io.StringIO(mmcif_text))
    structure = molecular._read_structure(cif)
    labels = molecular._label_asym_ids(cif, len(structure))

    grouped = molecular._group_structure_atoms(structure, labels)
    by_author_chain = molecular._group_structure_atoms(structure, None)
    tokens, positions, elements, names, hetero, spans, confidences, token_chains, chain_numbers = molecular._flatten_structure_groups(grouped)

    assert sorted(grouped) == ["A", "B", "Z"] and len(grouped["A"]) == 10 and sorted(by_author_chain) == ["A", "B", "Z"]
    assert chain_numbers == {"A": 0, "B": 1, "Z": 2} and len(tokens) == 21 and token_chains[:2] == [0, 0] and token_chains[-1] == 2
    assert spans[0] == (0, 5) and spans[-1][1] == len(positions) == len(elements) == len(names) == len(hetero)
    assert confidences[0] == pytest.approx(0.5, abs=1e-2) and max(confidences) <= 1.0
    water = {"W": {(1, "HOH"): {"atoms": [], "res_name": "HOH", "is_hetero": True}}}
    assert molecular._flatten_structure_groups(water)[0] == []


def test_a_structure_without_confidence_values_reads_each_residue_at_fifty_percent():
    atom = type("Atom", (), {"coord": np.zeros(3), "element": "ZN", "atom_name": "ZN", "hetero": True})()
    grouped = {"Z": {(1, "ZN"): {"atoms": [atom], "res_name": "ZN", "is_hetero": True}}}

    *_, confidences, _chains, _numbers = molecular._flatten_structure_groups(grouped)

    assert confidences == [0.5]


def test_atom_names_are_filled_in_from_the_residue_when_a_complex_has_none():
    assert molecular._fallback_atom_names("ALA", 5) == ["C", "CA", "CB", "N", "O"]
    assert molecular._fallback_atom_names("ALA", 7)[5:] == ["X6", "X7"] and molecular._fallback_atom_names("ALA", 2) == ["C", "CA"]
    assert molecular._fallback_atom_names("LIG", 3) == ["C1", "C2", "C3"]


def test_a_complex_without_atom_names_writes_mmcif_with_names_taken_from_its_residues(base):
    nameless = replace(base, atom_names=None, atom_hetero=None)

    text = nameless.to_mmcif()

    read = MmcifWrapper.read(io.StringIO(text))
    assert set(read.structure.atom_name[read.structure.res_name == "ALA"]) == {"C", "CA", "CB", "N", "O"}
    assert not read.structure.hetero.any()


def test_chains_with_the_same_token_sequence_share_an_entity_and_entity_tables_can_be_added(chains):
    homomer = MolecularComplex.from_protein_complex(ProteinComplex.from_chains([chains[0], replace(chains[0], chain_id="B")]))
    heteromer = MolecularComplex.from_protein_complex(ProteinComplex.from_chains(chains))
    cif = CIFFile.read(io.StringIO("data_added\n#\n"))

    by_chain, chain_entities, entity_sequences = homomer._get_entity_mapping()
    _, hetero_entities, hetero_sequences = heteromer._get_entity_mapping()
    heteromer._add_entity_information(cif, hetero_sequences)

    assert sorted(by_chain) == ["A", "B"] and chain_entities == {"A": 1, "B": 1} and list(entity_sequences) == [1]
    assert hetero_entities == {"A": 1, "B": 2}
    assert {"entity", "struct_asym", "entity_poly", "entity_poly_seq"} <= set(cif.block.keys())
    assert cif.block["entity_poly"]["pdbx_seq_one_letter_code_can"].as_array(str).tolist() == [FIRST, SECOND]


def test_a_moved_complex_has_zero_rmsd_and_perfect_lddt_over_its_token_centers(base):
    turned = moved(base)

    assert base.rmsd(turned) == pytest.approx(0.0, abs=1e-3)
    assert base.lddt_ca(turned) == pytest.approx(1.0)
    assert base.lddt_ca(turned, cutoff=5.0) == pytest.approx(1.0)


def test_scores_need_complexes_with_the_same_tokens_and_at_least_one_token_with_atoms(chains, base):
    shorter = MolecularComplex.from_protein_complex(ProteinComplex.from_chains([chains[0]]))
    atomless = replace(base, token_to_atoms=np.zeros_like(base.token_to_atoms))
    half = base.token_to_atoms.copy()
    half[10:] = 0
    half_atomless = replace(base, token_to_atoms=half)

    with pytest.raises(ValueError, match="same number of tokens: 20 vs 10"):
        base.rmsd(shorter)
    with pytest.raises(ValueError, match="No valid atoms found for RMSD computation"):
        base.rmsd(atomless)
    with pytest.raises(ValueError, match="No valid atoms found for LDDT computation"):
        base.lddt_ca(atomless)
    assert np.isfinite(base.lddt_ca(half_atomless)) and np.isfinite(base.rmsd(half_atomless))


def test_dockq_scores_through_the_protein_complex_comparison_when_the_program_works(base, monkeypatch):
    commands = stub_dockq(monkeypatch, protein_complex_module, dockq_report())

    scored = base.dockq(moved(base))

    assert commands[0][0] == "DockQ" and scored.total_dockq == pytest.approx(0.9) and scored.chain_mapping == {"A": "A", "B": "B"}


def test_dockq_falls_back_to_reading_the_total_score_when_the_comparison_cannot_use_the_program(base, monkeypatch):
    def refuse(command):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(protein_complex_module, "check_output", refuse)
    commands = stub_dockq(monkeypatch, molecular, dockq_report(total="0.83"))

    scored = base.dockq(base)

    assert commands[0][0] == "DockQ" and scored["total_dockq"] == pytest.approx(0.83) and scored["aligned"] is base
    assert "Total DockQ" in scored["raw_output"]


def test_the_manual_dockq_score_is_read_from_a_total_line_or_from_a_dockq_line(base, monkeypatch):
    stub_dockq(monkeypatch, molecular, b"Total DockQ over 2 interfaces: 0.41\n")
    from_total = base._compute_dockq_manual(base)
    stub_dockq(monkeypatch, molecular, b"Model chains: A\nDockQ: 0.7\n")
    from_line = base._compute_dockq_manual(base)
    stub_dockq(monkeypatch, molecular, b"DockQ: not-a-number\nDockQ: 0.6\n")
    skipped = base._compute_dockq_manual(base)

    assert from_total["total_dockq"] == pytest.approx(0.41) and from_line["total_dockq"] == pytest.approx(0.7)
    assert skipped["total_dockq"] == pytest.approx(0.6)


def test_the_manual_dockq_score_reports_a_missing_program_and_unreadable_output(base, monkeypatch):
    stub_dockq(monkeypatch, molecular, b"nothing useful\n")
    with pytest.raises(RuntimeError, match="DockQ computation failed: Could not parse DockQ score"):
        base._compute_dockq_manual(base)

    def refuse(command):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(molecular, "check_output", refuse)
    monkeypatch.setattr(protein_complex_module, "check_output", refuse)
    with pytest.raises(RuntimeError, match="DockQ is not installed"):
        base.dockq(base)


def test_dockq_names_the_reason_when_a_complex_has_no_protein_tokens(base):
    ligand_only = replace(base, sequence=["ZN"] * len(base))

    with pytest.raises(ValueError, match="Cannot convert MolecularComplex to ProteinComplex for DockQ: No protein tokens"):
        ligand_only.dockq(base)
    with pytest.raises(ValueError, match="Cannot convert MolecularComplex to ProteinComplex for DockQ"):
        base._compute_dockq_manual(ligand_only)
