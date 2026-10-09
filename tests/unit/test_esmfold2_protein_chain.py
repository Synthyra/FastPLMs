"""ESMFold2 protein chains: construction, validation, geometry, serialization, file formats, and structure metrics.

The chain is ten residues on a helix backbone with seeded side-chain atoms (see `synthetic_structures`), so every file it
writes can be read back and every metric has a known answer for a rigidly moved copy.

Shapes: `l` residues, `a` present atoms.
"""

import io
import numpy as np
import pytest
import torch

from collections import namedtuple
from dataclasses import replace
from types import SimpleNamespace
from tests.unit.synthetic_structures import moved_chain, synthetic_chain, synthetic_mmcif

from fastplms.models.esmfold2 import esmfold2_protein_chain as chain_module
from fastplms.models.esmfold2 import esmfold2_residue_constants as residue_constants
from fastplms.models.esmfold2.esmfold2_affine3d import Affine3D
from fastplms.models.esmfold2.esmfold2_mmcif_parsing import MmcifWrapper
from fastplms.models.esmfold2.esmfold2_protein_chain import ProteinChain


SEQUENCE = "ACDEFGHIKL"
CA = residue_constants.atom_order["CA"]
OXYGEN = residue_constants.atom_order["O"]
CB = residue_constants.atom_order["CB"]


@pytest.fixture
def chain() -> ProteinChain:
    return synthetic_chain(SEQUENCE)


@pytest.fixture
def second_chain() -> ProteinChain:
    return synthetic_chain("MNPQRSTVWY", chain_id="B", entity_id=2, shift=12.0, seed=1)


@pytest.fixture
def mmcif_text(chain, second_chain) -> str:
    return synthetic_mmcif([chain, second_chain])


def test_a_chain_refuses_fields_that_do_not_describe_one_protein(chain):
    wrong_length = chain.atom37_positions[:5]
    cases = [
        (TypeError, "id must be a string", {"id": 5}),
        (TypeError, "sequence must be a string", {"sequence": 5}),
        (ValueError, "chain_id must be a non-empty string", {"chain_id": ""}),
        (TypeError, "entity_id must be an integer", {"entity_id": True}),
        (ValueError, "does not align", {"atom37_positions": wrong_length}),
        (TypeError, "NumPy array", {"confidence": list(chain.confidence)}),
        (TypeError, "Boolean dtype", {"atom37_mask": chain.atom37_mask.astype(int)}),
        (TypeError, "residue_index must use an integer dtype", {"residue_index": chain.residue_index.astype(float)}),
        (TypeError, "insertion_code must use a string-compatible dtype", {"insertion_code": np.zeros(10, dtype=int)}),
        (TypeError, "confidence must use a numeric dtype", {"confidence": np.array(["a"] * 10)}),
        (ValueError, "atom37_positions must have shape", {"atom37_positions": np.zeros((10, 36, 3))}),
        (ValueError, "atom37_mask must have shape", {"atom37_mask": np.zeros((10, 36), dtype=bool)}),
        (TypeError, "atom37_confidence must be a NumPy array", {"atom37_confidence": [0.5]}),
        (ValueError, "atom37_confidence shape must match", {"atom37_confidence": np.zeros((10, 36))}),
        (TypeError, "atom37_confidence must use a numeric dtype", {"atom37_confidence": np.full((10, 37), "x")}),
    ]

    for error, message, change in cases:
        with pytest.raises(error, match=message):
            replace(chain, **change)


def test_a_chain_builds_from_atom37_coordinates_as_arrays_or_tensors(chain):
    positions = torch.from_numpy(chain.atom37_positions)  # (l, 37, 3)

    from_array = ProteinChain.from_atom37(chain.atom37_positions)
    from_batch = ProteinChain.from_atom37(
        positions[None],
        id="named",
        sequence=SEQUENCE,
        chain_id="C",
        entity_id=3,
        residue_index=torch.arange(10)[None],
        confidence=torch.full((1, 10), 0.5),
    )

    assert from_array.sequence == "A" * 10 and from_array.chain_id == "A" and from_array.id == ""
    assert from_array.residue_index.tolist() == list(range(1, 11)) and from_array.confidence.tolist() == [1.0] * 10
    assert np.array_equal(from_array.atom37_mask, chain.atom37_mask)
    assert (from_batch.id, from_batch.sequence, from_batch.chain_id, from_batch.entity_id) == ("named", SEQUENCE, "C", 3)
    assert from_batch.residue_index.tolist() == list(range(10)) and from_batch.confidence.tolist() == [0.5] * 10


def test_atom37_input_that_is_batched_mistyped_or_misshapen_is_refused(chain):
    positions = torch.from_numpy(chain.atom37_positions)  # (l, 37, 3)

    with pytest.raises(ValueError, match="Cannot handle batched inputs, atom37_positions"):
        ProteinChain.from_atom37(positions[None].repeat(2, 1, 1, 1))
    with pytest.raises(TypeError, match="atom37_positions must be a NumPy array"):
        ProteinChain.from_atom37(chain.atom37_positions.tolist())
    with pytest.raises(ValueError, match="shape \\(length, 37, 3\\)"):
        ProteinChain.from_atom37(np.zeros((10, 36, 3)))
    with pytest.raises(ValueError, match="residue_index has shape"):
        ProteinChain.from_atom37(positions, residue_index=torch.zeros(2, 10, dtype=torch.long))
    with pytest.raises(TypeError, match="residue_index must be a NumPy array"):
        ProteinChain.from_atom37(positions, residue_index=list(range(10)))
    with pytest.raises(ValueError, match="confidence has shape"):
        ProteinChain.from_atom37(positions, confidence=torch.zeros(2, 10))
    with pytest.raises(TypeError, match="confidence must be a NumPy array"):
        ProteinChain.from_atom37(positions, confidence=[1.0] * 10)


def test_a_chain_builds_from_backbone_coordinates_alone():
    backbone = torch.arange(27, dtype=torch.float32).reshape(1, 3, 3, 3)  # (1, l, 3, 3)

    from_tensor = ProteinChain.from_backbone_atom_coordinates(backbone, sequence="GGG", chain_id="Q")
    from_array = ProteinChain.from_backbone_atom_coordinates(backbone[0].numpy())

    assert from_tensor.atom37_mask[:, :3].all() and not from_tensor.atom37_mask[:, 3:].any()
    assert from_tensor.sequence == "GGG" and from_tensor.chain_id == "Q"
    assert np.array_equal(from_array.atoms["CA"], backbone[0, :, 1].numpy())
    with pytest.raises(ValueError, match="Cannot handle batched inputs"):
        ProteinChain.from_backbone_atom_coordinates(backbone.repeat(2, 1, 1, 1))
    with pytest.raises(TypeError, match="NumPy array or Torch tensor"):
        ProteinChain.from_backbone_atom_coordinates([[0.0]])
    with pytest.raises(ValueError, match="shape \\(length, 3, 3\\)"):
        ProteinChain.from_backbone_atom_coordinates(np.zeros((3, 4, 3)))


def test_a_chain_reads_from_a_stored_record_or_another_chain(chain):
    record = {
        "id": "stored",
        "chain_id": "C",
        "entity_id": 4,
        "sequence": SEQUENCE,
        "residue_index": chain.residue_index,
        "insertion_code": chain.insertion_code.tolist(),
        "atom37_positions": chain.atom37_positions,
        "atom37_mask": chain.atom37_mask.astype(int),
        "confidence": chain.confidence,
    }

    stored = ProteinChain.from_mds(record)
    copied = ProteinChain.from_open_source(chain)

    assert stored.id == "stored" and stored.atom37_mask.dtype == bool and stored.mmcif is None
    assert np.array_equal(stored.atom37_mask, chain.atom37_mask)
    assert copied is not chain and (copied.id, copied.sequence, copied.chain_id) == (chain.id, chain.sequence, chain.chain_id)
    assert np.array_equal(copied.atom37_positions, chain.atom37_positions, equal_nan=True)


def test_the_storage_dictionary_restores_integer_keys_except_where_told_not_to():
    stored = {"a": {"1": "x", "name": {"2": "y"}}, "assembly_composition": {"1": ["A"]}, "7": 1}

    restored = chain_module._str_key_to_int_key(stored, ignore_keys=["assembly_composition"])

    assert restored == {"a": {1: "x", "name": {2: "y"}}, "assembly_composition": {"1": ["A"]}, 7: 1}


def test_a_chain_survives_its_compact_state_and_blob_forms(chain, tmp_path):
    with_atom_confidence = replace(chain, atom37_confidence=np.where(chain.atom37_mask, 0.75, np.nan).astype(np.float32))
    state = chain.state_dict()
    backbone_state = chain.state_dict(backbone_only=True)
    text_state = chain.state_dict(json_serializable=True)
    confident_state = with_atom_confidence.state_dict()
    blob_path = tmp_path / "chain.blob"
    blob_path.write_bytes(chain.to_blob())

    restored = ProteinChain.from_state_dict(state)
    from_text = ProteinChain.from_state_dict(text_state)
    from_bytes = ProteinChain.from_blob(chain.to_blob())
    from_file = ProteinChain.from_blob(blob_path)
    from_string_path = ProteinChain.from_blob(str(blob_path))
    from_stream = ProteinChain.from_blob(io.BytesIO(chain.to_blob()))
    backbone_only = ProteinChain.from_blob(chain.to_blob(backbone_only=True))
    confident = ProteinChain.from_state_dict(confident_state)

    assert state["atom37_positions"].shape == (int(chain.atom37_mask.sum()), 3) and state["atom37_positions"].dtype == np.float16
    assert state["residue_index"].dtype == np.int32 and "atom37_confidence" not in state
    assert backbone_state["atom37_positions"].shape == (30, 3) and chain.atom37_mask[:, 3:].any()
    assert isinstance(text_state["atom37_mask"], list)
    for other in (restored, from_text, from_bytes, from_file, from_string_path, from_stream):
        assert other.sequence == SEQUENCE and np.array_equal(other.atom37_mask, chain.atom37_mask)
        np.testing.assert_allclose(other.atom37_positions[chain.atom37_mask], chain.atom37_positions[chain.atom37_mask], atol=0.05)
    assert backbone_only.atom37_mask[:, 3:].sum() == 0 and backbone_only.atom37_mask[:, :3].all()
    assert confident.atom37_confidence is not None and np.nanmax(confident.atom37_confidence) == pytest.approx(0.75)


def test_a_chain_reports_its_atoms_by_name_and_as_a_biotite_array(chain):
    atom_array = chain.atom_array
    without_insertions = chain.atom_array_no_insertions

    assert chain.atoms["CA"].shape == (10, 3) and chain.atom_mask["CA"].tolist() == [True] * 10
    assert np.array_equal(chain.atoms[["N", "CA"]][:, 1], chain.atoms["CA"])
    assert len(atom_array) == int(chain.atom37_mask.sum()) == len(without_insertions)
    assert atom_array.res_name[0] == "ALA" and atom_array.chain_id[0] == "A"
    assert atom_array.b_factor[0] == pytest.approx(chain.confidence[0] * 100, abs=1e-3)
    assert without_insertions.res_id.min() == 1 and without_insertions.res_id.max() == 10
    assert len(chain) == 10


def test_atom_confidence_replaces_residue_confidence_in_the_atom_array(chain):
    detailed = replace(chain, atom37_confidence=np.full((10, 37), 0.25, dtype=np.float32))

    assert set(detailed.atom_array.b_factor.round(3)) == {25.0}
    assert set(detailed.atom_array_no_insertions.b_factor.round(3)) == {25.0}


def test_insertion_codes_survive_the_atom_array_and_offset_the_residue_numbers(chain):
    codes = np.array([""] * 10, dtype="<U4")
    codes[3] = "A"
    inserted = replace(chain, insertion_code=codes)

    assert inserted.atom_array.ins_code[inserted.atom_array.res_id == 4].tolist() == ["A"] * int((inserted.atom_array.res_id == 4).sum())
    assert inserted.residue_index_no_insertions.tolist() == [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]


def test_the_normalization_frame_removes_the_pose_of_the_chain(chain):
    moved = moved_chain(chain)

    frame = chain.get_normalization_frame()
    normalized = chain.normalize_coordinates()
    normalized_after_moving = moved.normalize_coordinates()
    applied = chain.apply_frame(frame)

    assert isinstance(frame, Affine3D)
    np.testing.assert_allclose(normalized.atoms["CA"].mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(
        normalized_after_moving.atoms["CA"], normalized.atoms["CA"], atol=1e-3
    )
    np.testing.assert_allclose(applied.atoms["CA"], normalized.atoms["CA"], atol=1e-5)


def test_a_missing_oxygen_is_placed_from_the_backbone_but_the_last_residue_cannot_be_completed(chain):
    positions = chain.atom37_positions.copy()
    mask = chain.atom37_mask.copy()
    for residue in (2, 9):
        positions[residue, OXYGEN] = np.nan
        mask[residue, OXYGEN] = False
    lacking = replace(chain, atom37_positions=positions, atom37_mask=mask)

    completed = lacking.infer_oxygen()

    assert completed.atom37_mask[2, OXYGEN] and np.isfinite(completed.atom37_positions[2, OXYGEN]).all()
    assert not completed.atom37_mask[9, OXYGEN]
    assert np.linalg.norm(completed.atom37_positions[2, OXYGEN] - completed.atom37_positions[2, CA]) < 3.0


def test_c_beta_positions_are_inferred_for_every_residue_but_glycine_unless_asked(chain):
    inferred = chain.inferred_cbeta.copy()  # (l, 3)
    with_glycine = chain.infer_cbeta(infer_cbeta_for_glycine=True)
    without_glycine = chain.infer_cbeta()

    glycine = SEQUENCE.index("G")
    assert inferred.shape == (10, 3) and np.isfinite(inferred).all()
    assert with_glycine.atom37_mask[:, CB].all()
    np.testing.assert_allclose(with_glycine.atom37_positions[:, CB], inferred, atol=1e-5)
    assert not without_glycine.atom37_mask[glycine, CB] and np.isnan(without_glycine.atom37_positions[glycine, CB]).all()
    assert np.delete(without_glycine.atom37_mask[:, CB], glycine).all()


def test_pairwise_distances_and_contacts_come_from_the_alpha_and_inferred_beta_carbons(chain):
    ca_distances = chain.pdist_CA  # (l, l)
    cb_distances = chain.pdist_CB  # (l, l)
    contacts = chain.cbeta_contacts(distance_threshold=6.0)  # (l, l)

    assert ca_distances.shape == (10, 10) and np.allclose(ca_distances, ca_distances.T) and np.allclose(np.diag(ca_distances), 0)
    assert cb_distances.shape == (10, 10)
    assert contacts.shape == (10, 10) and (np.diag(contacts) == -1).all()
    assert set(np.unique(contacts)) <= {-1, 0, 1}
    assert contacts[0, 1] == int(cb_distances[0, 1] < 6.0)


def test_a_chain_converts_to_structure_encoder_inputs(chain):
    coordinates, confidence, residue_index = chain.to_structure_encoder_inputs()

    assert coordinates.shape == (1, 10, 37, 3) and coordinates.dtype == torch.float32
    assert confidence.shape == (1, 10) and residue_index.shape == (1, 10) and residue_index.dtype == torch.long
    assert residue_index[0].tolist() == list(range(1, 11))


def test_a_chain_is_indexed_by_position_slice_list_mask_or_tensor(chain):
    detailed = replace(chain, atom37_confidence=np.full((10, 37), 0.5, dtype=np.float32))

    assert chain[2].sequence == "D" and chain[2:5].sequence == "DEF"
    assert chain[[0, 9]].sequence == "AL" and chain[np.arange(10) < 3].sequence == "ACD"
    assert chain[torch.tensor([1, 2])].sequence == "CD"
    assert chain[2:5].atom37_positions.shape == (3, 37, 3) and detailed[2:5].atom37_confidence.shape == (3, 37)


def test_residues_are_selected_by_number_and_by_number_with_an_expected_letter(chain):
    selected = chain.select_residue_indices(["A1", "D3"])

    assert selected.sequence == "AD" and chain.select_residue_indices([2, 4]).sequence == "CE"
    with pytest.raises(RuntimeError, match="Position 3, Expected: A, Received: D"):
        chain.select_residue_indices(["A3"])
    unknown = replace(chain, sequence="XCDEFGHIKL")
    assert unknown.select_residue_indices(["A1"], ignore_x_mismatch=True).sequence == "X"


def test_chains_concatenate_with_or_without_a_chain_break_token(chain, second_chain):
    joined = ProteinChain.concat([chain, second_chain])
    fused = ProteinChain.concat([chain, second_chain], use_chainbreak=False)

    assert joined.sequence == f"{SEQUENCE}|MNPQRSTVWY" and len(joined) == 21
    assert joined.residue_index[10] == -1 and not joined.atom37_mask[10].any()
    assert np.isinf(joined.atom37_positions[10]).all()
    assert fused.sequence == SEQUENCE + "MNPQRSTVWY" and len(fused) == 20
    with pytest.raises(ValueError, match="at least one ProteinChain"):
        ProteinChain.concat([])
    with pytest.raises(TypeError, match="only ProteinChain instances"):
        ProteinChain.concat([chain, "B"])
    with pytest.raises(RuntimeError, match="deprecated"):
        ProteinChain.as_complex([chain])


def test_a_chain_writes_pdb_text_and_reads_it_back(chain, tmp_path):
    text = chain.to_pdb_string()
    path = tmp_path / "model.pdb"
    chain.to_pdb(path, include_insertions=False)

    from_stream = ProteinChain.from_pdb(io.StringIO(text), id="stream")
    from_path = ProteinChain.from_pdb(path, chain_id="A")
    from_string = ProteinChain.from_pdb(str(path))

    assert text.startswith("ATOM") and chain.to_pdb_string(include_insertions=False).count("\n") == text.count("\n")
    assert from_stream.id == "stream" and from_stream.sequence == SEQUENCE and from_stream.entity_id == 1
    assert from_path.id == "model" and from_string.id == "model"
    assert np.array_equal(from_stream.atom37_mask, chain.atom37_mask)
    np.testing.assert_allclose(from_stream.atom37_positions[chain.atom37_mask], chain.atom37_positions[chain.atom37_mask], atol=2e-3)
    with pytest.raises(ValueError, match="no amino-acid atoms for chain"):
        ProteinChain.from_pdb(io.StringIO(text), chain_id="Z")


def test_pdb_b_factors_are_read_as_confidence_only_for_predicted_structures(chain):
    text = chain.to_pdb_string()

    predicted = ProteinChain.from_pdb(io.StringIO(text), is_predicted=True)
    experimental = ProteinChain.from_pdb(io.StringIO(text), is_predicted=False)

    np.testing.assert_allclose(predicted.confidence, chain.confidence, atol=1e-2)
    assert experimental.confidence.tolist() == [1.0] * 10


def test_a_biotite_atom_array_converts_to_a_chain(chain):
    converted = ProteinChain.from_atomarray(chain.atom_array, id="array", is_predicted=True)

    assert converted.sequence == SEQUENCE and converted.id == "array"
    np.testing.assert_allclose(converted.confidence, chain.confidence, atol=1e-2)


def test_a_chain_writes_mmcif_with_per_residue_confidence_tables(chain, tmp_path):
    text = chain.to_mmcif_string()
    path = tmp_path / "model.cif"
    chain.to_mmcif(path)

    assert text.startswith("data_syn1") and "_ma_qa_metric_local.metric_value" in text
    assert path.read_text(encoding="utf-8") == text
    assert "50.0" in text and "90.0" in text


def test_a_chain_reads_from_mmcif_by_chain_entity_or_by_default(mmcif_text, second_chain):
    by_chain = ProteinChain.from_mmcif(io.StringIO(mmcif_text), chain_id="B")
    by_entity = ProteinChain.from_mmcif(io.StringIO(mmcif_text), entity_id=2, keep_source=True)
    default = ProteinChain.from_mmcif(io.StringIO(mmcif_text))

    assert by_chain.sequence == second_chain.sequence and by_chain.entity_id == 2 and by_chain.mmcif is None
    assert by_entity.chain_id == "B" and isinstance(by_entity.mmcif, MmcifWrapper)
    assert default.chain_id == "A" and default.sequence == SEQUENCE
    np.testing.assert_allclose(by_chain.atoms["CA"], second_chain.atoms["CA"], atol=2e-3)
    assert default.id == ""


def test_choosing_a_chain_from_mmcif_requires_a_consistent_and_existing_selection(mmcif_text):
    with pytest.raises(ValueError, match="at most one of chain_id or entity_id"):
        ProteinChain.from_mmcif(io.StringIO(mmcif_text), chain_id="A", entity_id=1)
    with pytest.raises(ValueError, match="does not contain entity `9`"):
        ProteinChain.from_mmcif(io.StringIO(mmcif_text), entity_id=9)
    empty = MmcifWrapper.read(io.StringIO(mmcif_text))
    empty.entities = {}
    with pytest.raises(ValueError, match="no entities"):
        ProteinChain.from_mmcif(empty)
    with pytest.warns(UserWarning, match="Failed to detect entity_id"), pytest.raises(ValueError, match="sequence mappings for chain"):
        ProteinChain.from_mmcif(empty, chain_id="Q")


def test_every_protein_chain_of_an_mmcif_file_can_be_read_in_turn(mmcif_text):
    chains = list(ProteinChain.chain_iterable_from_mmcif(io.StringIO(mmcif_text), keep_source=True))
    plain = list(ProteinChain.chain_iterable_from_mmcif(MmcifWrapper.read(io.StringIO(mmcif_text), "wrapped")))

    assert [str(item.chain_id) for item in chains] == ["A", "B"]
    assert [item.entity_id for item in chains] == [1, 2] and all(item.mmcif is not None for item in chains)
    assert [item.id for item in plain] == ["wrapped", "wrapped"] and plain[0].mmcif is None


def test_chain_arrays_come_from_the_residue_scheme_and_atom_table_of_an_mmcif_file(mmcif_text):
    wrapper = MmcifWrapper.read(io.StringIO(mmcif_text))
    with_b_factors = wrapper.structure.copy()
    with_b_factors.set_annotation("b_factor", np.full(with_b_factors.array_length(), 80.0, dtype=np.float32))

    sequence, positions, mask, residue_index, insertion_code, confidence, entity_id = chain_module.chain_to_ndarray(
        wrapper.structure, wrapper, "A"
    )  # (l,), (l, 37, 3), (l, 37), (l,), (l,), (l,)
    *_, predicted_confidence, _ = chain_module.chain_to_ndarray(with_b_factors, wrapper, "A", is_predicted=True)  # (l,)

    assert sequence == SEQUENCE and positions.shape == (10, 37, 3) and mask.shape == (10, 37) and entity_id == 1
    assert residue_index.tolist() == list(range(1, 11)) and insertion_code.tolist() == [""] * 10
    assert confidence.tolist() == [1.0] * 10
    np.testing.assert_allclose(predicted_confidence, 80.0 / chain_module.PLDDT_B_FACTOR_SCALE)
    assert chain_module._num_non_null_residues(wrapper.seqres_to_structure["A"]) == 10
    with pytest.raises(TypeError, match="AtomArray"):
        chain_module.chain_to_ndarray([], wrapper, "A")
    with pytest.raises(TypeError, match="MmcifWrapper"):
        chain_module.chain_to_ndarray(wrapper.structure, object(), "A")
    with pytest.raises(ValueError, match="non-empty string"):
        chain_module.chain_to_ndarray(wrapper.structure, wrapper, "")
    with pytest.raises(ValueError, match="sequence mappings for chain 'Z'"):
        chain_module.chain_to_ndarray(wrapper.structure, wrapper, "Z")


def test_a_chain_is_fetched_by_identifier_through_the_structure_database(mmcif_text, monkeypatch):
    requested = []

    def fetch(pdb_id, file_format):
        requested.append((pdb_id, file_format))
        return io.StringIO(mmcif_text)

    monkeypatch.setattr(chain_module.rcsb, "fetch", fetch)

    fetched = ProteinChain.from_rcsb("1ABC", chain_id="B")

    assert requested == [("1ABC", "cif")] and fetched.id == "1ABC" and fetched.chain_id == "B"


def test_beta_carbons_are_inferred_from_the_three_backbone_atoms_in_numpy():
    n = np.array([[-0.525, 1.363, 0.0]])  # (1, 3)
    ca = np.zeros((1, 3))  # (1, 3)
    c = np.array([[1.526, 0.0, 0.0]])  # (1, 3)

    beta = chain_module.infer_cb(c, n, ca)  # (1, 3)

    np.testing.assert_allclose(beta[0], [-0.529, -0.774, -1.205], atol=0.01)


def test_ligand_contacts_need_the_source_file_and_report_the_residues_within_five_angstroms(chain):
    ligand_type = namedtuple("Ligand", ["name", "comp_id"])
    near = chain.atoms["CA"][3] + np.array([1.0, 0.0, 0.0], dtype=np.float32)  # (3,)
    zinc = ligand_type(name="Zinc ion", comp_id="ZN")
    atoms_near = SimpleNamespace(coord=near[None])
    atoms_far = SimpleNamespace(coord=np.array([[500.0, 500.0, 500.0]]))
    atoms_none = SimpleNamespace(coord=None)
    with_source = lambda coords: replace(chain, mmcif=SimpleNamespace(non_polymer_coords={(zinc, "Z"): coords}))  # noqa: E731

    with pytest.raises(ValueError, match="keep_source=True"):
        chain.find_nonpolymer_contacts()
    assert with_source(atoms_far).find_nonpolymer_contacts() == []
    contacts = with_source(atoms_near).find_nonpolymer_contacts()
    assert contacts[0]["ligand"] == "Zinc ion" and contacts[0]["ligand_id"] == "ZN" and 3 in contacts[0]["contacting_residues"]
    with pytest.raises(ValueError, match="no coordinate table"):
        with_source(atoms_none).find_nonpolymer_contacts()


def test_solvent_accessible_area_is_reported_per_residue_or_per_atom_with_nan_for_missing_residues(chain):
    positions = chain.atom37_positions.copy()
    mask = chain.atom37_mask.copy()
    positions[4] = np.nan
    mask[4] = False
    gapped = replace(chain, atom37_positions=positions, atom37_mask=mask)

    by_residue = chain.sasa()  # (l,)
    by_atom = chain.sasa(by_residue=False)  # (a,)
    with_gap = gapped.sasa()  # (l,)

    assert by_residue.shape == (10,) and (by_residue >= 0).all()
    assert by_atom.shape == (int(chain.atom37_mask.sum()),)
    assert by_residue.sum() == pytest.approx(by_atom.sum(), rel=1e-4)
    assert np.isnan(with_gap[4]) and np.isfinite(np.delete(with_gap, 4)).all()


def test_spatial_aggregation_propensity_aggregates_by_atom_residue_or_protein(chain):
    by_atom = chain.sap_score()  # (a,)
    by_residue = chain.sap_score("residue")  # (l,)
    whole = chain.sap_score("protein")

    assert by_atom.shape == (int(chain.atom37_mask.sum()),) and by_residue.shape == (10,)
    assert whole == pytest.approx(by_atom[by_atom > 0].sum())
    with pytest.raises(ValueError, match="Invalid aggregation method"):
        chain.sap_score("domain")


def test_globularity_radius_of_gyration_and_the_ellipsoid_fit_describe_the_shape(chain):
    points = chain.atom37_positions[chain.atom37_mask]  # (a, 3)

    globularity = chain.globularity()
    radius = chain.radius_of_gyration()
    matrix, center = ProteinChain._mvee(points, tol=1e-3)  # (3, 3), (3, 1)

    assert 0.1 < globularity < 2.0 and radius == pytest.approx(5.4, abs=1.0)
    assert matrix.shape == (3, 3) and center.shape == (3, 1)
    assert np.allclose(center[:, 0], points.mean(axis=0), atol=2.0)
    with pytest.raises(ValueError, match="did not converge"):
        ProteinChain._mvee(points, tol=1e-12, max_iter=1)


def test_a_rigidly_moved_copy_aligns_onto_its_original_with_zero_rmsd_and_perfect_scores(chain, second_chain):
    moved = moved_chain(chain)

    aligned = moved.align(chain)
    distance = moved.rmsd(chain)
    best_of_both = moved.rmsd(chain, also_check_reflection=True)
    subset = moved.rmsd(chain, mobile_inds=[0, 1, 2, 3], target_inds=[0, 1, 2, 3], only_compute_backbone_rmsd=True)
    per_residue = moved.lddt_ca(chain, mobile_inds=slice(None), target_inds=slice(None))
    overall = moved.lddt_ca(chain, mobile_inds=slice(None), target_inds=slice(None), per_residue=False)
    gdt = moved.gdt_ts(chain, mobile_inds=slice(None), target_inds=slice(None))

    np.testing.assert_allclose(aligned.atom37_positions[chain.atom37_mask], chain.atom37_positions[chain.atom37_mask], atol=1e-3)
    assert distance == pytest.approx(0.0, abs=1e-3) and best_of_both <= distance + 1e-9 and subset == pytest.approx(0.0, abs=1e-3)
    np.testing.assert_allclose(per_residue, 1.0, atol=1e-4)
    assert overall == pytest.approx(1.0, abs=1e-4) and gdt == pytest.approx(1.0)
    assert moved.rmsd(second_chain) > 0.5
    with pytest.raises(ValueError, match="Support for bs.AtomArray removed"):
        chain.rmsd(chain.atom_array)
