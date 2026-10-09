"""The typed ESMFold2 prediction input serializes to JSON and comes back with the same chains and conditioning."""

import json
import numpy as np
import pytest

from fastplms.models.esmfold2 import esmfold2_input_builder as builder
from fastplms.models.esmfold2.esmfold2_msa import MSA


def full_input() -> builder.StructurePredictionInput:
    return builder.StructurePredictionInput(
        sequences=[
            builder.ProteinInput(
                id=["A", "B"],
                sequence="ACDE",
                modifications=[builder.Modification(position=1, ccd="SEP", smiles="ignored")],
                msa=MSA.from_sequences(["ACDE", "AC-E"]),
            ),
            builder.ProteinInput(id="C", sequence="FGHI"),
            builder.RNAInput(id="R", sequence="ACGU", modifications=[builder.Modification(position=0, ccd="PSU")]),
            builder.DNAInput(id="D", sequence="ACGT"),
            builder.LigandInput(id="L", smiles="CCO"),
            builder.LigandInput(id="M", ccd=["ATP"]),
        ],
        pocket=builder.PocketConditioning(binder_chain_id="L", contacts=[("A", 2), ("C", 0)]),
        distogram_conditioning=[builder.DistogramConditioning(chain_id="A", distogram=np.arange(6, dtype=float).reshape(2, 3))],
        covalent_bonds=[builder.CovalentBond("A", 0, 1, "L", 0, 2)],
    )


def test_a_full_input_serializes_to_json_and_reads_back_unchanged():
    original = full_input()

    serialized = builder.serialize_structure_prediction_input(original)
    restored = builder.deserialize_structure_prediction_input(json.loads(json.dumps(serialized)))

    assert [chain["type"] for chain in serialized["sequences"]] == ["protein", "protein", "rna", "dna", "ligand", "ligand"]
    assert serialized["sequences"][0]["modifications"] == [{"position": 1, "ccd": "SEP"}]
    assert serialized["sequences"][0]["msa"] == {"sequences": ["ACDE", "AC-E"]}
    assert "modifications" not in serialized["sequences"][1] and serialized["sequences"][1]["msa"] is None
    assert [type(chain) for chain in restored.sequences] == [type(chain) for chain in original.sequences]
    assert restored.sequences[0].id == ["A", "B"] and restored.sequences[0].msa.sequences == ["ACDE", "AC-E"]
    assert restored.sequences[0].modifications == [builder.Modification(position=1, ccd="SEP")]
    assert restored.sequences[2].modifications == [builder.Modification(position=0, ccd="PSU")]
    assert restored.sequences[4].smiles == "CCO" and restored.sequences[5].ccd == ["ATP"]
    assert restored.pocket == builder.PocketConditioning(binder_chain_id="L", contacts=[("A", 2), ("C", 0)])
    assert np.array_equal(restored.distogram_conditioning[0].distogram, original.distogram_conditioning[0].distogram)
    assert restored.covalent_bonds == original.covalent_bonds


def test_an_input_without_conditioning_has_none_of_its_keys():
    plain = builder.StructurePredictionInput(sequences=[builder.ProteinInput(id="A", sequence="ACDE")])

    serialized = builder.serialize_structure_prediction_input(plain)
    restored = builder.deserialize_structure_prediction_input(serialized)

    assert set(serialized) == {"sequences"}
    assert restored.pocket is None and restored.distogram_conditioning is None and restored.covalent_bonds is None


def test_a_chain_that_is_not_one_of_the_input_kinds_is_refused():
    class Unknown:
        sequence = "ACDE"

    with pytest.raises(ValueError, match="Unsupported sequence input type"):
        builder._serialize_chain(Unknown())
    with pytest.raises(AttributeError, match="MSA must be None or MSA"):
        builder._serialize_chain(builder.ProteinInput(id="A", sequence="ACDE", msa=["ACDE"]))


def test_serialized_input_of_an_unknown_kind_or_a_malformed_alignment_is_refused():
    with pytest.raises(ValueError, match="Unsupported sequence type"):
        builder._deserialize_chain({"id": "A", "type": "lipid"})
    with pytest.raises(ValueError, match="Unexpected MSA value"):
        builder._deserialize_msa({"msa": {"sequences": "ACDE"}})
    assert builder._deserialize_msa({}) is None
    assert builder._deserialize_modifications({"modifications": []}) is None
