"""ESMFold2 typed-input preparation: tokenization, entity assignment, frames, bonds, MSA rows, and atom tables.

A miniature chemical component dictionary stands in for the hash-pinned one (see `synthetic_ccd`), so ligands and
modified residues tokenize offline. Atom tables hold no coordinates at prediction time, so tests that need resolved
frames copy each atom's reference position into its position.

Shapes: `l` tokens, `a` atoms (padded to a multiple of 32), `m` MSA rows.
"""

import re
import numpy as np
import pytest
import torch

from tests.unit.synthetic_ccd import install_mini_ccd

from fastplms.models.esmfold2 import esmfold2_prepare_input as prepare
from fastplms.models.esmfold2.esmfold2_constants import (
    DNA_BACKBONE_ATOMS,
    DNA_RESIDUE_TO_RES_TYPE,
    DNA_RNA_LIGAND_INPUT_ID,
    DNA_UNK_RES_TYPE,
    ESM_PROTEIN_VOCAB,
    MOL_TYPE_DNA,
    MOL_TYPE_NONPOLYMER,
    MOL_TYPE_PROTEIN,
    MOL_TYPE_RNA,
    MSA_GAP_TOKEN_ID,
    PROTEIN_HEAVY_ATOMS,
    PROTEIN_RESIDUE_TO_RES_TYPE,
    PROTEIN_UNK_RES_TYPE,
    RNA_RESIDUE_TO_RES_TYPE,
    RNA_UNK_RES_TYPE,
)
from fastplms.models.esmfold2.esmfold2_msa import MSA
from fastplms.models.esmfold2.esmfold2_types import (
    CovalentBond,
    DistogramConditioning,
    DNAInput,
    LigandInput,
    Modification,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)


@pytest.fixture(autouse=True)
def mini_ccd(monkeypatch):
    return install_mini_ccd(monkeypatch)


def unit_offsets() -> dict[str, int]:
    return {"entity_id": 0, "asym_id": 0, "sym_id": 0, "token_offset": 0, "atom_offset": 0, "space_uid_offset": 0}


def with_reference_positions(atoms: list[prepare.AtomInfo]) -> list[prepare.AtomInfo]:
    """The atoms with each position set to its reference position, as a resolved structure would have."""
    for atom in atoms:
        atom.pos = atom.ref_pos.copy()
    return atoms


def mixed_input() -> StructurePredictionInput:
    return StructurePredictionInput(
        sequences=[
            ProteinInput(id=["A", "B"], sequence="AG", msa=MSA.from_sequences(["AG", "A-"])),
            ProteinInput(id="C", sequence="ASG", modifications=[Modification(position=1, ccd="SEP")], msa=MSA.from_sequences(["ASG"])),
            DNAInput(id="D", sequence="AC"),
            RNAInput(id="R", sequence="GN"),
            LigandInput(id="L", ccd=["EOH"]),
            LigandInput(id="M", smiles="CCO"),
            LigandInput(id="Z", ccd=["ZN"]),
        ],
        covalent_bonds=[CovalentBond("C", 1, 3, "L", 0, 0)],
    )


def test_atom_names_are_four_ascii_codes_offset_by_32_with_blanks_as_zero():
    assert prepare.encode_atom_name("CA") == [35, 33, 0, 0]
    assert prepare.encode_atom_name("ABCDE") == [33, 34, 35, 36]
    assert prepare.encode_atom_name("CA") is prepare.encode_atom_name("CA")


def test_element_symbols_map_to_atomic_numbers_in_any_case_and_unknown_symbols_to_zero():
    assert [prepare.get_element_atomic_num(symbol) for symbol in ("C", "N", "Zn", "zn", "Xx")] == [6, 7, 30, 30, 0]
    assert prepare.get_element_atomic_num("Zn") == prepare.get_element_atomic_num("Zn")


def test_an_element_is_inferred_from_an_atom_name():
    names = ["", "  ", "CA", "ZN", "FE", "1HB", "3", "OP1", "NZ"]

    assert [prepare._infer_element(name) for name in names] == ["C", "C", "C", "ZN", "FE", "H", "H", "O", "N"]


def test_residue_types_and_language_model_inputs_follow_the_molecule_type():
    assert prepare._compute_res_type("ALA", MOL_TYPE_PROTEIN) == PROTEIN_RESIDUE_TO_RES_TYPE["ALA"]
    assert prepare._compute_res_type("ZZZ", MOL_TYPE_PROTEIN) == PROTEIN_UNK_RES_TYPE
    assert prepare._compute_res_type("DA", MOL_TYPE_DNA) == DNA_RESIDUE_TO_RES_TYPE["DA"]
    assert prepare._compute_res_type("A", MOL_TYPE_DNA) == RNA_RESIDUE_TO_RES_TYPE["A"]
    assert prepare._compute_res_type("DA", MOL_TYPE_RNA) == DNA_RESIDUE_TO_RES_TYPE["DA"]
    assert prepare._compute_res_type("ZZ", MOL_TYPE_DNA) == DNA_UNK_RES_TYPE
    assert prepare._compute_res_type("ZZ", MOL_TYPE_RNA) == RNA_UNK_RES_TYPE
    assert prepare._compute_res_type("EOH", MOL_TYPE_NONPOLYMER) == PROTEIN_UNK_RES_TYPE
    assert prepare._compute_esm_input_id("ALA", MOL_TYPE_PROTEIN) == ESM_PROTEIN_VOCAB["A"]
    assert prepare._compute_esm_input_id("ZZZ", MOL_TYPE_PROTEIN) == DNA_RNA_LIGAND_INPUT_ID
    assert prepare._compute_esm_input_id("DA", MOL_TYPE_DNA) == DNA_RNA_LIGAND_INPUT_ID


def test_modifications_replace_residues_in_place_and_report_the_positions_that_changed():
    residues = ["ALA", "GLY", "SER"]

    changed = prepare._apply_modifications(residues, [Modification(position=2, ccd="SEP")])

    assert residues == ["ALA", "GLY", "SEP"] and changed == {2}
    assert prepare._apply_modifications(residues, None) == set()


def test_the_tokenization_state_numbers_atoms_tokens_and_spaces_as_it_goes():
    state = prepare._TokenizationState(token_index=10, atom_index=100, space_uid=3)
    specs = [("N", "N", 0, np.ones(3, dtype=np.float32)), ("CA", "C", 1, None)]
    fields = {"mol_type": 0, "res_type": 2, "input_id": 5, "asym_id": 0, "sym_id": 0, "entity_id": 0}

    state.add_residue_token(specs, residue_index=0, residue_name="ALA", **fields)
    state.add_atom_tokens(specs, residue_index=1, residue_name="SEP", **fields)

    assert [(token.token_index, token.atom_start, token.atom_count) for token in state.tokens] == [(10, 100, 2), (11, 102, 1), (12, 103, 1)]
    assert [(atom.token_index, atom.atom_index, atom.space_uid) for atom in state.atoms] == [(10, 100, 3), (10, 101, 3), (11, 102, 4), (12, 103, 4)]
    assert state.space_uid == 5 and state.token_index == 13 and state.atom_index == 104
    assert (state.atoms[0].ref_pos == 1).all() and not state.atoms[1].ref_pos.any() and state.atoms[0].ref_pos is not specs[0][3]
    assert state.atoms[1].charge == 1 and not state.atoms[0].pos.any()


def test_atom_specs_carry_inferred_elements_residue_charges_and_reference_positions():
    alanine = PROTEIN_RESIDUE_TO_RES_TYPE["ALA"]

    charged = prepare._ideal_atom_specs("LYS", PROTEIN_RESIDUE_TO_RES_TYPE["LYS"], ["N", "NZ"])
    neutral = prepare._ideal_atom_specs("LYS", PROTEIN_RESIDUE_TO_RES_TYPE["LYS"], ["N", "NZ"], charges=False)
    placed = prepare._ideal_atom_specs("ALA", alanine, ["N", "CA", "XX"])
    ligand = prepare._ccd_atom_specs("EOH", [("C1", "C", 0), ("C2", "C", 0), ("O", "O", 0)], {"C2"})
    pinned = prepare._ccd_atom_specs("EOH", [("C1", "C", 0), ("O", "O", 0)], set(), force_zero=True)

    assert [(name, element, charge) for name, element, charge, _position in charged] == [("N", "N", 0), ("NZ", "N", 1)]
    assert [charge for _name, _element, charge, _position in neutral] == [0, 0]
    assert placed[0][3].shape == (3,) and placed[1][3].shape == (3,) and placed[2][3] is None
    assert [name for name, *_rest in ligand] == ["C1", "O"] and all(position.shape == (3,) for *_rest, position in ligand)
    assert [position for *_rest, position in pinned] == [None, None]


def test_protein_residues_become_one_token_each_with_their_heavy_atoms():
    tokens, atoms = prepare.tokenize_protein("AG", None, entity_id=3, asym_id=2, sym_id=1, token_offset=10, atom_offset=100, space_uid_offset=7)

    assert [token.residue_name for token in tokens] == ["ALA", "GLY"]
    assert [(token.token_index, token.atom_start, token.atom_count) for token in tokens] == [(10, 100, 5), (11, 105, 4)]
    assert [atom.name for atom in atoms[:5]] == PROTEIN_HEAVY_ATOMS["ALA"] and [atom.space_uid for atom in atoms] == [7] * 5 + [8] * 4
    assert all((token.mol_type, token.entity_id, token.asym_id, token.sym_id) == (MOL_TYPE_PROTEIN, 3, 2, 1) for token in tokens)
    assert [token.res_type for token in tokens] == [PROTEIN_RESIDUE_TO_RES_TYPE["ALA"], PROTEIN_RESIDUE_TO_RES_TYPE["GLY"]]
    assert [token.input_id for token in tokens] == [ESM_PROTEIN_VOCAB["A"], ESM_PROTEIN_VOCAB["G"]]
    assert atoms[1].ref_pos.any() and atoms[1].element == "C" and atoms[0].element == "N"


def test_a_modified_residue_becomes_atom_tokens_that_drop_the_leaving_atom_unless_it_ends_the_chain():
    inner_tokens, inner_atoms = prepare.tokenize_protein("ASC", [Modification(position=1, ccd="SEP")], **unit_offsets())
    final_tokens, final_atoms = prepare.tokenize_protein("AS", [Modification(position=1, ccd="SEP")], **unit_offsets())

    inner = [token for token in inner_tokens if token.residue_name == "SEP"]
    final = [token for token in final_tokens if token.residue_name == "SEP"]
    assert len(inner) == 10 and len(final) == 11
    assert all(token.atom_count == 1 and token.res_type == PROTEIN_UNK_RES_TYPE and token.input_id == DNA_RNA_LIGAND_INPUT_ID for token in inner)
    assert "OXT" not in {inner_atoms[token.atom_start].name for token in inner}
    assert "OXT" in {final_atoms[token.atom_start].name for token in final}
    assert [token.residue_name for token in inner_tokens][0] == "ALA" and inner_tokens[-1].residue_name == "CYS"


def test_a_residue_missing_from_the_dictionary_keeps_four_backbone_atoms_and_a_lone_atom_has_no_reference_position():
    unknown_tokens, unknown_atoms = prepare.tokenize_protein("AS", [Modification(position=1, ccd="NOPE")], **unit_offsets())
    lone_tokens, lone_atoms = prepare.tokenize_protein("AS", [Modification(position=1, ccd="ZN")], **unit_offsets())
    placeholder_tokens, _ = prepare.tokenize_protein("X", None, **unit_offsets())

    assert [atom.name for atom in unknown_atoms[5:]] == ["N", "C", "C", "O"] and len(unknown_tokens) == 5
    assert [token.residue_name for token in lone_tokens] == ["ALA", "ZN"] and not lone_atoms[-1].ref_pos.any() and lone_atoms[-1].charge == 2
    assert placeholder_tokens[0].residue_name == "UNK" and placeholder_tokens[0].res_type == PROTEIN_UNK_RES_TYPE


def test_dna_and_rna_residues_become_tokens_and_unknown_bases_keep_only_the_backbone():
    dna_tokens, dna_atoms = prepare.tokenize_nucleotide("AN", None, MOL_TYPE_DNA, **unit_offsets())
    rna_tokens, rna_atoms = prepare.tokenize_nucleotide("UN", None, MOL_TYPE_RNA, **unit_offsets())

    assert [token.residue_name for token in dna_tokens] == ["DA", "UNK"] and [token.residue_name for token in rna_tokens] == ["U", "UNK"]
    assert [token.res_type for token in dna_tokens] == [DNA_RESIDUE_TO_RES_TYPE["DA"], DNA_UNK_RES_TYPE]
    assert [token.res_type for token in rna_tokens] == [RNA_RESIDUE_TO_RES_TYPE["U"], RNA_UNK_RES_TYPE]
    assert [atom.name for atom in dna_atoms[-len(DNA_BACKBONE_ATOMS):]] == list(DNA_BACKBONE_ATOMS)
    assert all(token.mol_type == MOL_TYPE_DNA and token.input_id == DNA_RNA_LIGAND_INPUT_ID for token in dna_tokens)
    assert all(token.mol_type == MOL_TYPE_RNA for token in rna_tokens)


def test_a_modified_nucleotide_becomes_atom_tokens_that_use_the_protein_unknown_type():
    tokens, atoms = prepare.tokenize_nucleotide("AC", [Modification(position=0, ccd="PSU")], MOL_TYPE_DNA, **unit_offsets())

    modified = [token for token in tokens if token.residue_name == "PSU"]
    assert len(modified) == len(DNA_BACKBONE_ATOMS) and all(token.res_type == PROTEIN_UNK_RES_TYPE for token in modified)
    assert tokens[-1].residue_name == "DC" and len(atoms) == len(DNA_BACKBONE_ATOMS) + len(prepare.DNA_HEAVY_ATOMS["DC"])


def test_a_dictionary_ligand_has_one_token_per_atom_and_covalent_attachment_drops_leaving_atoms():
    tokens, atoms = prepare.tokenize_ligand_ccd(["EOH", "ZN"], 4, 5, 0, 20, 200, 3, has_covalent_bond=False)
    free, _ = prepare.tokenize_ligand_ccd(["SEP"], 0, 0, 0, 0, 0, 0, has_covalent_bond=False)
    attached, _ = prepare.tokenize_ligand_ccd(["SEP"], 0, 0, 0, 0, 0, 0, has_covalent_bond=True)

    assert [(token.residue_index, token.residue_name) for token in tokens] == [(0, "EOH")] * 3 + [(1, "ZN")]
    assert [token.token_index for token in tokens] == [20, 21, 22, 23] and [atom.atom_index for atom in atoms] == [200, 201, 202, 203]
    assert all(token.mol_type == MOL_TYPE_NONPOLYMER and token.entity_id == 4 and token.asym_id == 5 for token in tokens)
    assert [atom.space_uid for atom in atoms] == [3, 3, 3, 4] and atoms[-1].charge == 2
    assert len(free) == 11 and len(attached) == 10
    with pytest.raises(ValueError, match="CCD component NOPE not found"):
        prepare.tokenize_ligand_ccd(["NOPE"], 0, 0, 0, 0, 0, 0, has_covalent_bond=False)


def test_a_smiles_ligand_is_embedded_and_tokenized_by_heavy_atom_with_canonical_names():
    tokens, atoms, bonds = prepare.tokenize_ligand_smiles("CCO", 1, 2, 0, 5, 50, 4, seed=3)
    _, repeated, _ = prepare.tokenize_ligand_smiles("CCO", 1, 2, 0, 5, 50, 4, seed=3)

    assert [token.residue_name for token in tokens] == ["LIG"] * 3 and [token.token_index for token in tokens] == [5, 6, 7]
    assert [atom.element for atom in atoms] == ["C", "C", "O"] and all(re.fullmatch(r"[A-Z]+\d+", atom.name) for atom in atoms)
    assert bonds == [(atoms[0].name, atoms[1].name), (atoms[1].name, atoms[2].name)]
    assert 1.2 < float(np.linalg.norm(atoms[0].ref_pos - atoms[1].ref_pos)) < 1.8
    for left, right in zip(atoms, repeated, strict=True):
        np.testing.assert_allclose(left.ref_pos, right.ref_pos)
    assert all(token.mol_type == MOL_TYPE_NONPOLYMER and (token.entity_id, token.asym_id) == (1, 2) for token in tokens)


def test_a_smiles_that_cannot_be_read_or_has_an_atom_name_too_long_for_four_characters_is_refused():
    with pytest.raises(ValueError, match="Failed to parse SMILES"):
        prepare.tokenize_ligand_smiles("not a smiles (", 0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="longer than 4 chars"):
        prepare.tokenize_ligand_smiles("C(Br)" * 70, 0, 0, 0, 0, 0, 0)


def test_sequence_keys_tell_entities_apart_by_kind_and_content():
    keys = [
        prepare._get_sequence_key(ProteinInput(id="A", sequence="AG")),
        prepare._get_sequence_key(DNAInput(id="A", sequence="AG")),
        prepare._get_sequence_key(RNAInput(id="A", sequence="AG")),
        prepare._get_sequence_key(LigandInput(id="A", ccd=["ATP", "ZN"])),
        prepare._get_sequence_key(LigandInput(id="A", smiles="CCO")),
    ]

    assert keys == ["PROTEIN:AG", "DNA:AG", "RNA:AG", "LIGAND_CCD:ATP,ZN", "LIGAND_SMILES:CCO"]
    with pytest.raises(ValueError, match="Unknown input type"):
        prepare._get_sequence_key(object())


def test_each_kind_of_input_is_tokenized_with_a_warning_when_it_is_underspecified():
    covalent = set()
    options = {"entity_id": 0, "asym_id": 0, "sym_id": 0, "token_offset": 0, "atom_offset": 0, "space_uid_offset": 0, "covalent_chains": covalent, "seed": 1}

    with pytest.warns(UserWarning, match="No MSA provided for A"):
        protein_tokens, _, protein_bonds = prepare._tokenize_chain(ProteinInput(id="A", sequence="AG"), "A", **options)
    nucleotide_tokens, _, _ = prepare._tokenize_chain(RNAInput(id="R", sequence="AC"), "R", **options)
    with pytest.warns(UserWarning, match="Both ccd and smiles provided"):
        ligand_tokens, _, _ = prepare._tokenize_chain(LigandInput(id="L", ccd=["EOH"], smiles="CCO"), "L", **options)
    smiles_tokens, _, smiles_bonds = prepare._tokenize_chain(LigandInput(id="M", smiles="CCO"), "M", **options)

    assert len(protein_tokens) == 2 and protein_bonds == [] and nucleotide_tokens[0].mol_type == MOL_TYPE_RNA
    assert [token.residue_name for token in ligand_tokens] == ["EOH"] * 3 and len(smiles_tokens) == 3 and len(smiles_bonds) == 2
    with pytest.raises(ValueError, match="either ccd or smiles"):
        prepare._tokenize_chain(LigandInput(id="X"), "X", **options)
    with pytest.raises(ValueError, match="Unknown input type"):
        prepare._tokenize_chain(object(), "X", **options)


def test_chains_share_an_entity_when_their_sequences_match_and_count_their_copies():
    chains, tokens, atoms = prepare.build_chains_from_input(mixed_input(), seed=5)

    assert [(chain.chain_id, chain.asym_id, chain.entity_id, chain.sym_id) for chain in chains] == [
        ("A", 0, 0, 0), ("B", 1, 0, 1), ("C", 2, 1, 0), ("D", 3, 2, 0), ("R", 4, 3, 0), ("L", 5, 4, 0), ("M", 6, 5, 0), ("Z", 7, 6, 0),
    ]
    assert [chain.mol_type for chain in chains] == [0, 0, 0, MOL_TYPE_DNA, MOL_TYPE_RNA, 3, 3, 3]
    assert [len(chain.tokens) for chain in chains] == [2, 2, 12, 2, 2, 3, 3, 1] and len(tokens) == 27
    assert [token.token_index for token in tokens] == list(range(27)) and [atom.atom_index for atom in atoms] == list(range(len(atoms)))
    assert chains[6].ligand_bonds and not chains[5].ligand_bonds
    assert len({atom.space_uid for atom in atoms}) == max(atom.space_uid for atom in atoms) + 1


def test_a_covalently_bound_ligand_loses_its_leaving_atoms():
    plain = StructurePredictionInput(sequences=[LigandInput(id="L", ccd=["SEP"]), ProteinInput(id="P", sequence="A", msa=MSA.from_sequences(["A"]))])
    bound = StructurePredictionInput(sequences=plain.sequences, covalent_bonds=[CovalentBond("P", 0, 1, "L", 0, 0)])

    free_chains, _, _ = prepare.build_chains_from_input(plain)
    bound_chains, _, _ = prepare.build_chains_from_input(bound)

    assert len(free_chains[0].tokens) == 11 and len(bound_chains[0].tokens) == 10


def test_named_atoms_are_indexed_by_token_and_atoms_marked_invalid_are_left_out():
    _, atoms = prepare.tokenize_protein("AG", None, **unit_offsets())
    atoms[1].is_valid = False

    indexed = prepare._atom_indices_by_name(atoms)

    assert indexed[0] == {"N": 0, "C": 2, "O": 3, "CB": 4} and indexed[1]["CA"] == 6


def test_protein_frames_use_n_ca_c_and_are_resolved_only_when_atoms_have_positions_at_a_wide_angle():
    tokens, atoms = prepare.tokenize_protein("AG", None, **unit_offsets())

    frames, unresolved = prepare.compute_frame_indices(tokens, atoms)
    _, resolved = prepare.compute_frame_indices(tokens, with_reference_positions(atoms))

    assert frames.tolist() == [[0, 1, 2], [5, 6, 7]] and unresolved.tolist() == [False, False] and resolved.tolist() == [True, True]


def test_ligand_frames_pair_each_atom_with_its_two_nearest_neighbours_and_lone_atoms_with_themselves():
    tokens, atoms = prepare.tokenize_ligand_ccd(["EOH", "ZN"], 0, 0, 0, 0, 0, 0, has_covalent_bond=False)

    frames, resolved = prepare.compute_frame_indices(tokens, with_reference_positions(atoms))

    assert frames.tolist()[3] == [3, 3, 3] and not resolved[3]
    assert [frame[1] for frame in frames.tolist()[:3]] == [0, 1, 2] and all(len(set(frame)) == 3 for frame in frames.tolist()[:3])
    assert resolved[:3].all()


def test_a_frame_follows_the_molecule_type_and_falls_back_to_the_first_atom():
    token = lambda mol_type, res_type: prepare.TokenInfo(0, 0, "X", mol_type, res_type, 0, 0, 0, 0, 0, 1)  # noqa: E731
    named = {"N": 4, "CA": 5, "C": 6, "C1'": 7, "C3'": 8, "C4'": 9}
    ligand_frames = {0: (1, 2, 3)}

    assert prepare._frame_for_token(token(MOL_TYPE_PROTEIN, 2), named, ligand_frames) == (4, 5, 6)
    assert prepare._frame_for_token(token(MOL_TYPE_PROTEIN, PROTEIN_UNK_RES_TYPE), named, ligand_frames) == (4, 4, 4)
    assert prepare._frame_for_token(token(MOL_TYPE_DNA, 28), named, ligand_frames) == (7, 8, 9)
    assert prepare._frame_for_token(token(MOL_TYPE_RNA, PROTEIN_UNK_RES_TYPE), named, ligand_frames) == (4, 4, 4)
    assert prepare._frame_for_token(token(MOL_TYPE_NONPOLYMER, 22), named, ligand_frames) == (1, 2, 3)
    assert prepare._frame_for_token(token(MOL_TYPE_NONPOLYMER, 22), named, {}) == (4, 4, 4)
    assert prepare._frame_for_token(token(9, 22), {}, {}) == (0, 0, 0)


def test_a_frame_is_resolved_only_when_its_atoms_have_positions_and_are_not_nearly_collinear():
    def atom(position):
        return prepare.AtomInfo("X", "C", 0, np.zeros(3, dtype=np.float32), np.asarray(position, dtype=np.float32))

    atoms = [atom((2, 1, 0)), atom((1, 1, 0)), atom((1, 2, 0)), atom((0, 1, 0)), atom((0, 0, 0))]
    token = prepare.TokenInfo(0, 0, "X", 0, 2, 0, 0, 0, 0, 0, 1)
    frames = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 1], [0, 1, 4]])

    resolved = prepare._resolved_frames(frames, [token] * 4, atoms)

    assert resolved.tolist() == [True, False, False, False]
    assert prepare._resolved_frames(frames[:0], [], atoms).shape == (0,)


def test_representative_atoms_are_cb_ca_or_the_nucleobase_anchor_by_molecule_type():
    tokens, atoms = prepare.tokenize_protein("AG", None, **unit_offsets())
    dna_tokens, dna_atoms = prepare.tokenize_nucleotide("AC", None, MOL_TYPE_DNA, 0, 0, 0, 2, len(atoms), 2)
    rna_tokens, rna_atoms = prepare.tokenize_nucleotide("GU", None, MOL_TYPE_RNA, 0, 0, 0, 4, len(atoms) + len(dna_atoms), 4)
    ligand_tokens, ligand_atoms = prepare.tokenize_ligand_ccd(["EOH"], 0, 0, 0, 6, len(atoms) + len(dna_atoms) + len(rna_atoms), 6, False)
    every_token = [*tokens, *dna_tokens, *rna_tokens, *ligand_tokens]
    every_atom = [*atoms, *dna_atoms, *rna_atoms, *ligand_atoms]

    representatives = prepare.compute_representative_atoms(every_token, every_atom)  # (l,)

    names = [every_atom[int(index)].name for index in representatives]
    assert representatives.dtype == torch.int64 and names[:2] == ["CB", "CA"]
    assert names[2] in ("C4", "C1'") and names[3] in ("C2", "C1'") and names[4] in ("C2", "C1'", "C4")
    assert names[6:] == ["C1", "C2", "O"]


def test_token_bonds_follow_ligand_bond_tables_modified_residue_links_and_covalent_bonds():
    chains, tokens, atoms = prepare.build_chains_from_input(mixed_input(), seed=5)

    bonds = prepare.compute_token_bonds(tokens, atoms, mixed_input(), chains)  # (l, l, 1)

    edges = {(int(left), int(right)) for left, right in torch.nonzero(bonds[..., 0]).tolist() if left < right}
    assert bonds.shape == (27, 27, 1) and torch.equal(bonds, bonds.transpose(0, 1))
    assert {(4, 5), (5, 6), (6, 13), (13, 15), (8, 20)} <= edges
    assert {(20, 21), (21, 22), (23, 24), (24, 25)} <= edges
    assert (0, 1) not in edges and (2, 3) not in edges and (4, 15) not in edges
    ligand_without_table = StructurePredictionInput(sequences=[LigandInput(id="L", ccd=["ZN", "ZN"])])
    chains, tokens, atoms = prepare.build_chains_from_input(ligand_without_table)
    assert prepare.compute_token_bonds(tokens, atoms, ligand_without_table, chains).sum() == 0


def test_atom_tokens_of_a_residue_without_a_bond_table_are_all_joined():
    tokens, atoms = prepare.tokenize_protein("AS", [Modification(position=1, ccd="NOPE")], **unit_offsets())
    chain = prepare.ChainInfo("A", 0, 0, 0, MOL_TYPE_PROTEIN, tokens)

    bonds = prepare.compute_token_bonds(tokens, atoms, StructurePredictionInput(sequences=[]), [chain])  # (l, l, 1)

    modified = [token.token_index for token in tokens if token.residue_name == "NOPE"]
    assert len(modified) == 4 and all(bonds[left, right, 0] == 1 for left in modified for right in modified if left != right)


def test_covalent_bonds_that_name_missing_chains_or_atoms_are_ignored():
    unknown_chain = CovalentBond("Q", 0, 0, "L", 0, 0)
    out_of_range = CovalentBond("L", 0, 99, "L", 0, 0)
    item = StructurePredictionInput(sequences=[LigandInput(id="L", ccd=["EOH"])], covalent_bonds=[unknown_chain, out_of_range])
    chains, tokens, atoms = prepare.build_chains_from_input(item)

    bonds = prepare.compute_token_bonds(tokens, atoms, item, chains)

    assert int(bonds.sum()) == 4  # only the two bonds of the dictionary table, each stored in both directions


def test_pocket_and_distogram_conditioning_inputs_bin_user_distances_for_their_own_chain_only():
    item = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="AG", msa=MSA.from_sequences(["AG"])), ProteinInput(id="B", sequence="G", msa=MSA.from_sequences(["G"]))],
        distogram_conditioning=[
            DistogramConditioning(chain_id="A", distogram=np.array([[0.0, 5.0], [100.0, 0.0]])),
            DistogramConditioning(chain_id="Q", distogram=np.zeros((9, 9))),
        ],
    )
    chains, tokens, _ = prepare.build_chains_from_input(item)

    bins, mask = prepare.compute_distogram_conditioning(item, chains, tokens, torch.zeros(len(tokens), 3))
    empty_bins, empty_mask = prepare.compute_distogram_conditioning(StructurePredictionInput(sequences=item.sequences), chains, tokens, torch.zeros(len(tokens), 3))

    assert bins.shape == (len(tokens), len(tokens)) and mask.tolist() == [[True, True, False], [True, True, False], [False, False, False]]
    assert bins[0, 0] == 0 and 0 < int(bins[0, 1]) < 63 and bins[1, 0] == 63 and not bins[:, 2].any()
    assert not empty_bins.any() and not empty_mask.any()
    mismatched = StructurePredictionInput(sequences=item.sequences, distogram_conditioning=[DistogramConditioning(chain_id="A", distogram=np.zeros((3, 3)))])
    with pytest.raises(ValueError, match="doesn't match chain length 2"):
        prepare.compute_distogram_conditioning(mismatched, chains, tokens, torch.zeros(len(tokens), 3))


def test_msa_rows_are_paired_per_token_with_gaps_for_chains_that_have_no_alignment():
    item = mixed_input()
    chains, tokens, _ = prepare.build_chains_from_input(item, seed=5)

    features = prepare.compute_msa_features(item, chains, tokens)  # each (m, l)

    assert features["msa"].shape == (2, 27) and features["msa"].dtype == torch.int64
    non_protein = [token.token_index for token in tokens if token.asym_id >= 3]
    assert features["msa"][0, non_protein].tolist() == [tokens[index].res_type for index in non_protein]
    assert (features["msa"][1, non_protein] == MSA_GAP_TOKEN_ID).all() and features["msa"][1, 0] == features["msa"][0, 0]
    assert features["msa"][1, 1] == MSA_GAP_TOKEN_ID and features["msa_attention_mask"].all()
    assert features["deletion_value"].shape == (2, 27) and not features["has_deletion"].any() and features["deletion_mean"].shape == (27,)
    assert prepare._msa_assignments(item, chains)[3] is None and prepare._msa_assignments(item, chains)[0] is item.sequences[0].msa


def test_a_protein_without_an_alignment_gets_its_own_sequence_as_a_one_row_alignment():
    item = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="AG")])
    with pytest.warns(UserWarning, match="No MSA provided"):
        chains, tokens, _ = prepare.build_chains_from_input(item)

    features = prepare.compute_msa_features(item, chains, tokens)

    assert features["msa"].shape[0] == 1 and features["msa"][0].tolist() == [token.res_type for token in tokens]
    assert prepare._msa_assignments(item, chains)[0].sequences == ["AG"]


def test_atoms_are_padded_to_a_multiple_of_32_with_invalid_rows():
    _, atoms = prepare.tokenize_protein("AG", None, **unit_offsets())

    padded = prepare._padded_atoms(atoms)
    empty = prepare._padded_atoms([])

    assert len(padded) == 32 and padded[:9] == atoms and not any(atom.is_valid for atom in padded[9:])
    assert [atom.atom_index for atom in padded[9:12]] == [9, 10, 11] and len(empty) == 32
    assert len(prepare._padded_atoms(padded)) == 32 and len(prepare._padded_atoms([*padded, atoms[0]])) == 64


def test_token_tensors_hold_one_integer_per_token_for_each_annotation():
    tokens, _ = prepare.tokenize_protein("AG", None, entity_id=3, asym_id=2, sym_id=1, token_offset=4, atom_offset=0, space_uid_offset=0)

    tensors = prepare._token_tensors(tokens)  # each (l,)

    assert set(tensors) == {"token_index", "residue_index", "asym_id", "sym_id", "entity_id", "mol_type", "res_type", "input_ids"}
    assert tensors["token_index"].tolist() == [4, 5] and tensors["residue_index"].tolist() == [0, 1] and tensors["entity_id"].tolist() == [3, 3]
    assert all(value.dtype == torch.int64 for value in tensors.values())


def test_atom_tensors_center_resolved_coordinates_and_zero_the_padding():
    def atom(position, valid=True):
        return prepare.AtomInfo("CA" if valid else "", "C" if valid else "", 1, np.ones(3, dtype=np.float32), np.asarray(position, dtype=np.float32), 0, 0, 5, valid)

    tensors = prepare._atom_tensors([atom((1, 2, 3)), atom((3, 4, 5)), atom((0, 0, 0)), atom((9, 9, 9), valid=False)])

    assert tensors["gt_coords"].shape == (1, 4, 3)
    assert tensors["gt_coords"][0].tolist() == [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [-2.0, -3.0, -4.0], [0.0, 0.0, 0.0]]
    assert tensors["is_resolved"].tolist() == [True, True, False, False] and tensors["atom_attention_mask"].tolist() == [True, True, True, False]
    assert tensors["ref_element"].tolist() == [6, 6, 6, 0] and tensors["ref_atom_name_chars"][0].tolist() == [35, 33, 0, 0]
    assert tensors["ref_charge"].dtype == torch.int8 and tensors["ref_space_uid"].tolist() == [5, 5, 5, 5]
    assert tensors["ref_pos"].shape == (4, 3) and tensors["atom_to_token"].tolist() == [0, 0, 0, 0]


def test_a_mixed_input_becomes_the_complete_unbatched_feature_dictionary():
    features, chains = prepare.prepare_esmfold2_input(mixed_input(), seed=5)

    tokens, atoms = 27, 128
    shapes = {name: tuple(value.shape) for name, value in features.items()}
    assert len(chains) == 8 and shapes["token_bonds"] == (tokens, tokens, 1) and shapes["ref_pos"] == (atoms, 3)
    assert shapes["gt_coords"] == (1, atoms, 3) and shapes["frames_idx"] == (tokens, 3) and shapes["msa"] == (2, tokens)
    assert shapes["disto_cond"] == (tokens, tokens) and shapes["distogram_atom_idx"] == (tokens,) and shapes["pocket_feature"] == (tokens,)
    assert features["token_attention_mask"].all() and not features["pocket_feature"].any() and not features["is_resolved"].any()
    assert int(features["atom_attention_mask"].sum()) == 119 and features["atom_to_token"][118] == tokens - 1
    assert features["ref_atom_name_chars"][0].tolist() == prepare.encode_atom_name("N") and features["ref_element"][0] == 7
    assert features["res_type"][:5].tolist() == [2, 9, 2, 9, 2] and features["input_ids"][0] == ESM_PROTEIN_VOCAB["A"]
    assert not features["disto_cond_mask"].any() and features["ref_space_uid"][-1] == 0


def test_features_are_repeatable_for_one_seed_and_follow_the_input_for_a_homomer():
    first, _ = prepare.prepare_esmfold2_input(StructurePredictionInput(sequences=[LigandInput(id="L", smiles="CCO")]), seed=11)
    second, _ = prepare.prepare_esmfold2_input(StructurePredictionInput(sequences=[LigandInput(id="L", smiles="CCO")]), seed=11)
    homomer, chains = prepare.prepare_esmfold2_input(
        StructurePredictionInput(sequences=[ProteinInput(id=["A", "B"], sequence="AG", msa=MSA.from_sequences(["AG"]))])
    )

    assert torch.equal(first["ref_pos"], second["ref_pos"])
    assert [chain.sym_id for chain in chains] == [0, 1] and homomer["entity_id"].tolist() == [0, 0, 0, 0]
    assert homomer["sym_id"].tolist() == [0, 0, 1, 1] and homomer["asym_id"].tolist() == [0, 0, 1, 1]
