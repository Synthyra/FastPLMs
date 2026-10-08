"""ESMFold2 protein complexes: assembly expansion, chain views, mmCIF and PDB export, compact storage, and scoring.

The complex joins two ten-residue synthetic chains with one separator row, so it has 21 rows. DockQ is an external
program; its report is stubbed with the text format that `ProteinComplex.dockq` parses.

Shapes: `l` rows (residues and separators), `c` chains.
"""

import io
import json
import numpy as np
import pytest

from dataclasses import replace
from types import SimpleNamespace
from tests.unit.synthetic_structures import (
    cif_column,
    dockq_report,
    moved_chain,
    stub_dockq,
    synthetic_chain,
    synthetic_mmcif,
    z_rotation,
)

from fastplms.models.esmfold2 import esmfold2_protein_complex as complex_module
from fastplms.models.esmfold2.esmfold2_mmcif_parsing import MmcifWrapper, NoProteinError
from fastplms.models.esmfold2.esmfold2_protein_chain import ProteinChain
from fastplms.models.esmfold2.esmfold2_protein_complex import (
    ProteinComplex,
    get_assembly_fast,
    protein_chain_to_protein_complex,
)


FIRST = "ACDEFGHIKL"
SECOND = "MNPQRSTVWY"
OXYGEN = 4  # atom37 slot of O


def two_chains() -> list[ProteinChain]:
    return [synthetic_chain(FIRST), synthetic_chain(SECOND, chain_id="B", entity_id=2, shift=12.0, seed=1)]


@pytest.fixture
def chains() -> list[ProteinChain]:
    return two_chains()


@pytest.fixture
def pair(chains) -> ProteinComplex:
    return ProteinComplex.from_chains(chains)


@pytest.fixture
def mmcif_text(chains) -> str:
    return synthetic_mmcif(chains)


def test_operation_expressions_expand_into_steps_in_application_order():
    expand = complex_module._parse_operation_expression

    assert expand("1") == [("1",)]
    assert expand("(1,2)") == [("1",), ("2",)]
    assert expand("(1-3)") == [("1",), ("2",), ("3",)]
    assert expand("(1-2)(4,5)") == [("4", "1"), ("4", "2"), ("5", "1"), ("5", "2")]


def test_operations_are_applied_step_by_step_to_a_copy_of_every_chain(chains):
    rotation = z_rotation(np.pi / 2).numpy().astype(np.float64)  # (3, 3)
    shift = SimpleNamespace(rotation=np.eye(3), target_translation=np.array([10.0, 0.0, 0.0]))
    turn = SimpleNamespace(rotation=rotation, target_translation=np.zeros(3))

    transformed = complex_module._apply_transformations_fast(chains, {"1": shift, "2": turn}, [("1",), ("1", "2")])

    assert len(transformed) == 4
    shifted_alpha_carbons = chains[0].atoms["CA"] + np.array([10.0, 0.0, 0.0])  # (l, 3)
    np.testing.assert_allclose(transformed[0].atoms["CA"], shifted_alpha_carbons, atol=1e-5)
    np.testing.assert_allclose(transformed[1].atoms["CA"], shifted_alpha_carbons @ rotation.T, atol=1e-4)
    assert transformed[2].sequence == SECOND and transformed[0].atom37_positions is not chains[0].atom37_positions
    np.testing.assert_allclose(chains[0].atoms["CA"], synthetic_chain(FIRST).atoms["CA"])


def test_a_complex_numbers_its_chains_and_entities_and_separates_chains_with_one_row(pair):
    assert pair.sequence == f"{FIRST}|{SECOND}" and len(pair) == 21 and pair.id == "syn1"
    assert pair.chain_id.tolist() == [0] * 10 + [-1] + [1] * 10
    assert pair.entity_id.tolist() == [0] * 10 + [-1] + [1] * 10
    assert pair.sym_id.tolist() == [0] * 10 + [-1] + [1] * 10
    assert pair.metadata.chain_lookup == {0: "A", 1: "B"} and pair.metadata.entity_lookup == {0: 1, 1: 2}
    assert pair.metadata.mmcif is None and pair.metadata.assembly_composition is None
    assert pair.residue_index[10] == -1 and not pair.atom37_mask[10].any() and np.isnan(pair.atom37_positions[10]).all()
    assert pair.num_chains == 2 and pair.chain_boundaries == [(0, 10), (11, 21)] and pair.chain_lengths.tolist() == [10, 10]
    with pytest.raises(ValueError, match="empty list of chains"):
        ProteinComplex.from_chains([])


def test_chains_that_share_an_identifier_share_a_number_and_a_chain_without_an_entity_gets_its_own(chains):
    repeated = ProteinComplex.from_chains([chains[0], replace(chains[0], entity_id=None), chains[1]])

    assert repeated.metadata.chain_lookup == {0: "A", 1: "B"}
    assert repeated.chain_id.tolist() == [0] * 10 + [-1] + [0] * 10 + [-1] + [1] * 10
    assert repeated.sym_id[[0, 11, 22]].tolist() == [0, 1, 2]
    assert repeated.entity_id[[0, 11, 22]].tolist() == [0, 1, 2] and repeated.metadata.entity_lookup == {0: 1, 2: 2}


def test_atom_confidence_of_any_chain_gives_the_whole_complex_an_atom_confidence_table(chains):
    detailed = replace(chains[0], atom37_confidence=np.full((10, 37), 0.5, dtype=np.float32))

    mixed = ProteinComplex.from_chains([detailed, chains[1]])

    assert mixed.atom37_confidence.shape == (21, 37)
    assert (mixed.atom37_confidence[:10] == 0.5).all() and np.isnan(mixed.atom37_confidence[10:]).all()
    assert mixed[0:10].atom37_confidence.shape == (10, 37) and mixed.as_chain(force_conversion=True).atom37_confidence.shape == (21, 37)


def test_a_complex_refuses_fields_that_do_not_describe_one_set_of_chains(pair):
    cases = [
        (TypeError, "sequence must be a string", {"sequence": 5}),
        (TypeError, "NumPy array", {"confidence": list(pair.confidence)}),
        (ValueError, "does not align", {"chain_id": pair.chain_id[:5]}),
        (ValueError, "does not align", {"atom37_positions": pair.atom37_positions[:5]}),
        (ValueError, "atom37_positions must have shape", {"atom37_positions": np.zeros((21, 36, 3))}),
        (ValueError, "atom37_mask must have shape", {"atom37_mask": np.zeros((21, 36), dtype=bool)}),
        (TypeError, "Boolean dtype", {"atom37_mask": pair.atom37_mask.astype(int)}),
        (TypeError, "numeric dtype", {"atom37_positions": np.full((21, 37, 3), "x")}),
        (ValueError, "residue_index must have shape", {"residue_index": np.zeros((21, 2))}),
        (TypeError, "confidence must use a numeric dtype", {"confidence": np.array(["a"] * 21)}),
        (TypeError, "atom37_confidence must be a NumPy array", {"atom37_confidence": [0.5]}),
        (ValueError, "atom37_confidence shape must match", {"atom37_confidence": np.zeros((21, 36))}),
    ]

    for error, message, change in cases:
        with pytest.raises(error, match=message):
            replace(pair, **change)


def test_a_complex_is_sliced_by_range_or_boolean_mask_while_keeping_one_separator_between_chains(pair):
    first_only = np.arange(21) < 10
    second_only = np.arange(21) > 10
    mixed = np.zeros(21, dtype=bool)
    mixed[[0, 1, 12, 13]] = True

    assert pair[0:10].sequence == FIRST and pair[11:21].sequence == SECOND and pair[5:15].sequence == "GHIKL|MNPQ"
    assert pair[first_only].sequence == FIRST and pair[second_only].sequence == SECOND and pair[mixed].sequence == "AC|NP"
    assert len(pair[0:0]) == 0 and len(pair[np.zeros(21, dtype=bool)]) == 0
    assert pair[first_only].atom37_positions.shape == (10, 37, 3)
    with pytest.raises(ValueError, match="doesn't supports indexing with lists"):
        pair[3]
    with pytest.raises(ValueError, match="doesn't supports indexing with lists"):
        pair[[1, 2]]


def test_chains_are_taken_out_of_a_complex_by_position_or_identifier(pair):
    by_index = pair.get_chain_by_index(1)
    by_id = pair.get_chain_by_id("A", sample_chain_if_duplicate=False)
    duplicated = replace(pair, metadata=replace(pair.metadata, chain_lookup={0: "A", 1: "A"}))

    assert by_index.sequence == SECOND and by_index.chain_id == "B" and by_index.entity_id == 2
    assert by_id.sequence == FIRST and by_id.chain_id == "A"
    assert [item.chain_id for item in pair.chain_iter()] == ["A", "B"]
    assert duplicated.get_chain_by_id("A").sequence in {FIRST, SECOND}
    with pytest.raises(IndexError, match="Chain index 5 out of bounds"):
        pair.get_chain_by_index(5)
    with pytest.raises(KeyError, match="Chain ID Z not found"):
        pair.get_chain_by_id("Z")
    with pytest.raises(ValueError, match="Multiple chains with chain ID A"):
        duplicated.get_chain_by_id("A", sample_chain_if_duplicate=False)


def test_a_complex_converts_to_one_chain_only_when_it_is_one_chain_of_one_entity_or_when_forced(chains, pair):
    single = ProteinComplex.from_chains([chains[0]])
    two_entities = replace(single, entity_id=np.where(np.arange(10) < 5, 0, 1))
    unnamed = replace(single, metadata=replace(single.metadata, chain_lookup={}, entity_lookup={}))

    flattened = pair.as_chain(force_conversion=True)
    converted = single.as_chain()
    with pytest.warns(UserWarning) as warned:
        defaulted = unnamed.as_chain()

    assert flattened.sequence == pair.sequence and flattened.chain_id == "A" and flattened.entity_id is None
    assert converted.chain_id == "A" and converted.entity_id == 1 and converted.sequence == FIRST
    assert defaulted.chain_id == "A" and defaulted.entity_id is None
    assert {"Chain ID not found in metadata, using 'A' as default", "Entity ID not found in metadata, using None as default"} <= {
        str(item.message) for item in warned
    }
    with pytest.raises(ValueError, match="multiple chains"):
        pair.as_chain()
    with pytest.raises(ValueError, match="multiple entities"):
        two_entities.as_chain()


def test_chain_adjacency_reports_which_chains_have_alpha_carbons_within_the_cutoff(pair):
    assert pair.chain_adjacency(cutoff=40.0).tolist() == [[False, True], [True, False]]
    assert not pair.chain_adjacency(cutoff=1.0).any() and not pair.chain_adjacency().any()
    assert pair.chain_adjacency_by_index(0, cutoff=40.0).tolist() == [False, True]
    assert pair.chain_adjacency_by_index(1, cutoff=1.0).tolist() == [False, False]
    assert len(pair.per_chain_kd_trees) == 2 and pair.atoms["CA"].shape == (21, 3) and pair.atom_mask["CA"].sum() == 20


def test_chains_are_renamed_with_a_prefix_or_with_single_letters_for_pdb_files(pair):
    prefixed = pair.add_prefix_to_chain_ids("x")
    lettered = prefixed.normalize_chain_ids_for_pdb()

    assert prefixed.metadata.chain_lookup == {0: "x_A", 1: "x_B"} and prefixed.sequence == pair.sequence
    assert lettered.metadata.chain_lookup == {0: "A", 1: "B"}


def test_the_missing_oxygen_and_c_beta_atoms_of_a_complex_are_inferred_from_the_backbone(pair):
    positions = pair.atom37_positions.copy()
    mask = pair.atom37_mask.copy()
    positions[2, OXYGEN] = np.nan
    mask[2, OXYGEN] = False
    lacking = replace(pair, atom37_positions=positions, atom37_mask=mask)
    glycine = FIRST.index("G")

    completed = lacking.infer_oxygen()
    without_glycine = pair.infer_cbeta()
    with_glycine = pair.infer_cbeta(infer_cbeta_for_glycine=True)

    assert completed.atom37_mask[2, OXYGEN] and np.isfinite(completed.atom37_positions[2, OXYGEN]).all()
    assert not completed.atom37_mask[10, OXYGEN]
    assert not without_glycine.atom37_mask[glycine, 3] and with_glycine.atom37_mask[glycine, 3] and with_glycine.atom37_mask[:10, 3].all()
    assert not with_glycine.atom37_mask[10, 3]


def test_complexes_with_one_identifier_concatenate_their_chains_and_complexes_with_different_ones_do_not(pair):
    other = ProteinComplex.from_chains([synthetic_chain("GG", chain_id="C", entity_id=3, shift=30.0, seed=2)])

    joined = ProteinComplex.concat([pair, other])

    assert joined.sequence == f"{FIRST}|{SECOND}|GG" and joined.num_chains == 3
    assert ProteinComplex.from_open_source(pair) is pair
    with pytest.raises(RuntimeError, match="different PDB ids"):
        ProteinComplex.concat([pair, replace(other, id="other")])


def test_a_complex_writes_mmcif_with_entity_tables_and_pdb_text_that_reads_back(pair, chains, tmp_path):
    repeated = ProteinComplex.from_chains([chains[0], replace(chains[0], chain_id="B")])
    path = tmp_path / "pair.pdb"

    text = pair.to_mmcif_string()
    shared_entity = repeated.to_mmcif_string()
    pair.to_pdb(path)
    from_text = ProteinComplex.from_pdb(io.StringIO(pair.to_pdb_string()), id="read")
    from_file = ProteinComplex.from_pdb(path)
    without_insertions = pair.to_pdb_string(include_insertions=False)

    assert text.startswith("data_syn1") and "Protein chain (entity 2)" in text and "pdbx_seq_one_letter_code" in text
    assert "Protein chain (entity 2)" not in shared_entity and "Protein chain (entity 1)" in shared_entity
    assert from_text.sequence == pair.sequence and from_text.num_chains == 2 and from_text.id == "read"
    assert from_file.sequence == pair.sequence
    np.testing.assert_allclose(
        from_text.atoms["CA"][np.r_[0:10, 11:21]], pair.atoms["CA"][np.r_[0:10, 11:21]], atol=2e-3
    )
    assert without_insertions.count("\n") > 0


def test_a_complex_survives_its_compact_state_and_blob_forms(tmp_path):
    state = ProteinComplex.from_chains(two_chains()).state_dict()
    backbone_state = ProteinComplex.from_chains(two_chains()).state_dict(backbone_only=True)
    text_state = ProteinComplex.from_chains(two_chains()).state_dict(json_serializable=True)
    blob = ProteinComplex.from_chains(two_chains()).to_blob()
    backbone_blob = ProteinComplex.from_chains(two_chains()).to_blob(backbone_only=True)
    path = tmp_path / "pair.blob"
    path.write_bytes(blob)
    original = ProteinComplex.from_chains(two_chains())

    restored = ProteinComplex.from_state_dict(state)
    from_text = ProteinComplex.from_state_dict(json.loads(json.dumps(text_state)))
    from_bytes = ProteinComplex.from_blob(blob)
    from_file = ProteinComplex.from_blob(path)
    from_string_path = ProteinComplex.from_blob(str(path))
    from_stream = ProteinComplex.from_blob(io.BytesIO(blob))
    backbone_only = ProteinComplex.from_blob(backbone_blob)

    assert state["metadata"]["mmcif"] is None and state["residue_index"].dtype == np.int32
    assert state["atom37_positions"].shape == (int(original.atom37_mask.sum()), 3) and state["atom37_positions"].dtype == np.float16
    assert backbone_state["atom37_positions"].shape == (60, 3) and isinstance(text_state["residue_index"], list)
    for other in (restored, from_text, from_bytes, from_file, from_string_path, from_stream):
        assert other.sequence == original.sequence and np.array_equal(other.atom37_mask, original.atom37_mask)
        assert other.metadata.chain_lookup == {0: "A", 1: "B"}
        np.testing.assert_allclose(
            other.atom37_positions[original.atom37_mask], original.atom37_positions[original.atom37_mask], atol=0.05
        )
    assert not backbone_only.atom37_mask[:, 3:].any() and backbone_only.atom37_mask[[0, 15], :3].all()


def test_the_compact_state_of_a_complex_with_atom_confidence_restores_it(chains):
    detailed = ProteinComplex.from_chains([replace(chains[0], atom37_confidence=np.full((10, 37), 0.75, dtype=np.float32)), chains[1]])

    restored = ProteinComplex.from_blob(detailed.to_blob())

    assert restored.atom37_confidence is not None and np.nanmax(restored.atom37_confidence) == pytest.approx(0.75)
    assert np.isnan(restored.atom37_confidence[11:]).all()


def test_a_complex_reads_the_first_assembly_of_an_mmcif_file_by_default_or_any_assembly_by_name(mmcif_text):
    default = ProteinComplex.from_mmcif(io.StringIO(mmcif_text), id="file")
    doubled = ProteinComplex.from_mmcif(io.StringIO(mmcif_text), assembly_id="2")

    first_copy, shifted_copy = doubled.chain_iter()
    assert default.sequence == f"{FIRST}|{SECOND}" and default.id == "file"
    assert default.metadata.assembly_composition == {"1": ["A", "B"]} and isinstance(default.metadata.mmcif, MmcifWrapper)
    assert doubled.sequence == f"{FIRST}|{FIRST}" and doubled.metadata.assembly_composition == {"2": ["A"]}
    np.testing.assert_allclose(shifted_copy.atoms["CA"] - first_copy.atoms["CA"], np.tile([20.0, 0.0, 0.0], (10, 1)), atol=1e-3)
    with pytest.raises(KeyError, match="Assembly ID '9'"):
        ProteinComplex.from_mmcif(io.StringIO(mmcif_text), assembly_id="9")


def test_assemblies_are_listed_by_chain_and_switched_through_the_retained_source(mmcif_text, pair):
    complex_of_assembly_one = ProteinComplex.from_mmcif(io.StringIO(mmcif_text))

    switched = complex_of_assembly_one.switch_assembly("2")

    assert complex_of_assembly_one.find_assembly_ids_with_chain("A") == ["1"]
    assert complex_of_assembly_one.find_assembly_ids_with_chain("Z") == []
    assert switched.sequence == f"{FIRST}|{FIRST}" and switched.find_assembly_ids_with_chain("A") == ["2"]
    with pytest.raises(ValueError, match="construct it from mmCIF"):
        pair.find_assembly_ids_with_chain("A")
    with pytest.raises(ValueError, match="without retained mmCIF source"):
        pair.switch_assembly("1")


def test_assembly_expansion_needs_loaded_data_and_at_least_one_protein_chain(mmcif_text):
    wrapper = MmcifWrapper.read(io.StringIO(mmcif_text))
    wrapper.raw.block["pdbx_struct_assembly_gen"]["asym_id_list"] = cif_column(["A,B", "Z"])

    with pytest.raises(complex_module.InvalidFileError, match="No mmCIF data loaded"):
        get_assembly_fast(MmcifWrapper(id="empty"))
    with pytest.raises(NoProteinError):
        get_assembly_fast(wrapper, assembly_id="2")
    assert get_assembly_fast(wrapper, assembly_id="1").num_chains == 2


def test_a_complex_is_fetched_by_identifier_through_the_structure_database(mmcif_text, monkeypatch):
    requested = []

    def fetch(pdb_id, file_format):
        requested.append((pdb_id, file_format))
        return io.StringIO(mmcif_text)

    monkeypatch.setattr(complex_module.rcsb, "fetch", fetch)

    fetched = ProteinComplex.from_rcsb("1ABC", keep_source=True)

    assert requested == [("1ABC", "cif")] and fetched.id == "1ABC" and fetched.num_chains == 2


def test_the_solvent_accessible_area_of_a_complex_has_one_entry_per_row_with_nan_for_separators(pair):
    by_residue = pair.sasa()  # (l,)
    by_atom = pair.sasa(by_residue=False)  # (a,)

    assert by_residue.shape == (21,) and np.isnan(by_residue[10]) and np.isfinite(np.delete(by_residue, 10)).all()
    assert by_atom.shape == (int(pair.atom37_mask.sum()),)


def test_a_chain_with_chain_breaks_splits_into_a_complex_of_its_pieces(chains):
    joined = ProteinChain.concat(chains)

    split = protein_chain_to_protein_complex(joined)
    unsplit = protein_chain_to_protein_complex(chains[0])

    assert split.sequence == f"{FIRST}|{SECOND}" and split.num_chains == 2
    assert split.metadata.chain_lookup == {0: "A", 1: "B"} and split.metadata.entity_lookup == {0: 0, 1: 1}
    assert unsplit.sequence == FIRST and unsplit.num_chains == 1


def test_a_moved_complex_scores_perfectly_against_the_original_when_the_chain_pairing_is_given(pair, chains):
    moved = ProteinComplex.from_chains([moved_chain(item) for item in chains])
    everything = slice(None)
    first_eight = np.arange(21) < 8

    distance = moved.rmsd(pair, compute_chain_assignment=False)
    best_of_both = moved.rmsd(pair, also_check_reflection=True, compute_chain_assignment=False)
    subset = moved.rmsd(pair, mobile_inds=first_eight, target_inds=first_eight, only_compute_backbone_rmsd=True, compute_chain_assignment=False)
    per_residue = moved.lddt_ca(pair, mobile_inds=everything, target_inds=everything, compute_chain_assignment=False)
    overall = moved.lddt_ca(pair, mobile_inds=everything, target_inds=everything, compute_chain_assignment=False, per_residue=False)
    gdt = moved.gdt_ts(pair, mobile_inds=everything, target_inds=everything, compute_chain_assignment=False)

    assert distance == pytest.approx(0.0, abs=1e-3) and best_of_both <= distance + 1e-9 and subset == pytest.approx(0.0, abs=1e-3)
    assert per_residue.shape == (21,) and np.allclose(per_residue[np.isfinite(per_residue)], 1.0, atol=1e-4)
    assert overall == pytest.approx(1.0, abs=1e-4) and gdt == pytest.approx(1.0)


def test_dockq_pairs_the_chains_of_the_model_with_those_of_the_native_complex_and_realigns_the_model(chains, monkeypatch):
    native = ProteinComplex.from_chains(chains)
    swapped = [replace(moved_chain(chains[1]), chain_id="A"), replace(moved_chain(chains[0]), chain_id="B")]
    model = ProteinComplex.from_chains(swapped)
    commands = stub_dockq(monkeypatch, complex_module, dockq_report(mapping="BA:AB"))

    scores = model.dockq(native)

    assert [command[0] for command in commands] == ["DockQ"] and all(part.endswith(".pdb") for part in commands[0][1:])
    assert scores.total_dockq == pytest.approx(0.9) and scores.native_interfaces == 1
    assert scores.chain_mapping == {"B": "A", "A": "B"}
    assert scores.aligned.sequence == native.sequence and scores.aligned_rmsd == pytest.approx(0.0, abs=1e-3)
    score = scores.interfaces[("A", "B")]
    assert score.native_chains == ("A", "B") and score.DockQ == pytest.approx(0.9) and score.interface_rms == pytest.approx(1.2)
    assert (score.ligand_rms, score.fnat, score.fnonnat, score.clashes, score.F1, score.DockQ_F1) == (2.3, 0.8, 0.1, 0.0, 0.85, 0.87)
    np.testing.assert_allclose(scores.aligned.atoms["CA"][:10], native.atoms["CA"][:10], atol=1e-3)


def test_scores_use_the_chain_pairing_that_dockq_finds_unless_told_not_to(chains, monkeypatch):
    native = ProteinComplex.from_chains(chains)
    model = ProteinComplex.from_chains([replace(moved_chain(chains[1]), chain_id="A"), replace(moved_chain(chains[0]), chain_id="B")])
    stub_dockq(monkeypatch, complex_module, dockq_report(mapping="BA:AB"))
    everything = slice(None)

    distance = model.rmsd(native)
    per_residue = model.lddt_ca(native, mobile_inds=everything, target_inds=everything)
    gdt = model.gdt_ts(native, mobile_inds=everything, target_inds=everything)

    assert distance == pytest.approx(0.0, abs=1e-3) and gdt == pytest.approx(1.0)
    assert np.allclose(per_residue[np.isfinite(per_residue)], 1.0, atol=1e-4)


def test_dockq_refuses_complexes_that_cannot_be_compared_or_written_with_single_letter_chain_names(chains, pair, monkeypatch):
    stub_dockq(monkeypatch, complex_module, dockq_report())
    one_long_chain = ProteinComplex.from_chains([synthetic_chain(FIRST + SECOND + "A")])
    prefixed = pair.add_prefix_to_chain_ids("x")
    same_ids = ProteinComplex.from_chains([chains[0], replace(chains[1], chain_id="A")])

    with pytest.raises(ValueError, match="same length"):
        pair.dockq(one_long_chain[0:10])
    with pytest.raises(ValueError, match="same number of chains"):
        pair.dockq(one_long_chain)
    with pytest.raises(ValueError, match="single letter chain IDs"):
        prefixed.dockq(pair)
    with pytest.raises(ValueError, match="Duplicate chain IDs"):
        same_ids.dockq(pair)


def test_a_dockq_report_without_a_total_line_that_can_be_read_is_an_error(pair, monkeypatch):
    stub_dockq(monkeypatch, complex_module, dockq_report().replace(b"native interfaces", b"interfaces"))

    with pytest.raises(RuntimeError, match="Failed to parse DockQ output"):
        pair.dockq(pair)
