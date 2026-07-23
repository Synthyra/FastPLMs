"""Fast, source-local contracts for ESMFold2 complex identity and storage."""

from __future__ import annotations

import numpy as np
import pytest

from fastplms.models.esmfold2.esmfold2_molecular_complex import MolecularComplex
from fastplms.models.esmfold2.esmfold2_protein_complex import (
    ProteinComplex,
    ProteinComplexMetadata,
)

pytestmark = pytest.mark.structure


def _homomer_with_repeated_chain_label() -> ProteinComplex:
    sequence = "AC|AC|GG"
    length = len(sequence)
    atom37_positions = np.full((length, 37, 3), np.nan, dtype=np.float32)
    atom37_mask = np.zeros((length, 37), dtype=bool)
    for index, residue in enumerate(sequence):
        if residue == "|":
            continue
        atom37_positions[index, 0] = (index, 0.0, 0.0)
        atom37_positions[index, 4] = (index, 1.0, 0.0)
        atom37_mask[index, (0, 4)] = True

    return ProteinComplex(
        id="repeated-chain-homomer",
        sequence=sequence,
        entity_id=np.asarray([7, 7, -1, 7, 7, -1, 9, 9], dtype=np.int64),
        chain_id=np.asarray([0, 0, -1, 0, 0, -1, 1, 1], dtype=np.int64),
        sym_id=np.asarray([0, 0, 0, 1, 1, 0, 0, 0], dtype=np.int64),
        residue_index=np.asarray([1, 2, -1, 1, 2, -1, 1, 2], dtype=np.int64),
        insertion_code=np.asarray([""] * length, dtype=object),
        atom37_positions=atom37_positions,
        atom37_mask=atom37_mask,
        confidence=np.asarray([0.9, 0.8, 0.0, 0.7, 0.6, 0.0, 0.5, 0.4]),
        metadata=ProteinComplexMetadata(
            entity_lookup={7: 101, 9: 202},
            chain_lookup={0: "A", 1: "B"},
            assembly_composition={"1": ["A", "A", "B"]},
        ),
    )


def test_molecular_round_trip_preserves_identity_and_repeated_chain_boundaries() -> None:
    original = _homomer_with_repeated_chain_label()

    molecular = MolecularComplex.from_protein_complex(original)
    restored_molecular = MolecularComplex.from_blob(molecular.to_blob())
    restored = restored_molecular.to_protein_complex()

    residue_rows = np.asarray(list(original.sequence)) != "|"
    assert molecular.entity_id is not None
    assert molecular.sym_id is not None
    np.testing.assert_array_equal(restored_molecular.entity_id, molecular.entity_id)
    np.testing.assert_array_equal(restored_molecular.sym_id, molecular.sym_id)
    assert restored.sequence == original.sequence
    np.testing.assert_array_equal(restored.chain_id, original.chain_id)
    np.testing.assert_array_equal(restored.entity_id, original.entity_id)
    np.testing.assert_array_equal(
        restored.sym_id[residue_rows], original.sym_id[residue_rows]
    )
    assert len(list(restored.chain_iter())) == 3
    assert restored.metadata.chain_lookup == original.metadata.chain_lookup
    assert restored.metadata.entity_lookup == original.metadata.entity_lookup
    assert restored.metadata.assembly_composition == original.metadata.assembly_composition


def test_backbone_state_dict_does_not_mutate_source_atom_mask() -> None:
    complex_value = _homomer_with_repeated_chain_label()
    original_mask = complex_value.atom37_mask.copy()

    backbone_state = complex_value.state_dict(backbone_only=True)

    np.testing.assert_array_equal(complex_value.atom37_mask, original_mask)
    assert original_mask[:, 4].any()
    assert not backbone_state["atom37_mask"][:, 3:].any()
    full_state = complex_value.state_dict()
    assert len(full_state["atom37_positions"]) == int(original_mask.sum())
