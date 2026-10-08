"""CPU checks for confidence-disabled ESMFold2 structure decoding."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from types import SimpleNamespace

from fastplms.models.esmfold2.esmfold2_constants import MOL_TYPE_NONPOLYMER, MOL_TYPE_PROTEIN
from fastplms.models.esmfold2.esmfold2_output import build_molecular_complex, get_element_symbol
from fastplms.models.esmfold2.esmfold2_processor import ESMFold2InputBuilder


def _token(index: int, residue_index: int, residue_name: str, atom_start: int) -> SimpleNamespace:
    return SimpleNamespace(
        token_index=index,
        residue_index=residue_index,
        residue_name=residue_name,
        atom_start=atom_start,
        atom_count=1,
    )


def _encoded_name(name: str) -> list[int]:
    return [ord(character) - 32 if character != " " else 0 for character in name.ljust(4)]


@pytest.mark.cpu_contract
def test_confidence_disabled_decode_preserves_two_chain_cif_without_scores() -> None:
    chain_infos = [
        SimpleNamespace(
            asym_id=0,
            entity_id=0,
            chain_id="A",
            mol_type=MOL_TYPE_PROTEIN,
            tokens=[_token(0, 0, "ALA", 0), _token(1, 1, "GLY", 1)],
        ),
        SimpleNamespace(
            asym_id=1,
            entity_id=1,
            chain_id="B",
            mol_type=MOL_TYPE_PROTEIN,
            tokens=[_token(2, 0, "SER", 2), _token(3, 1, "THR", 3)],
        ),
    ]
    features = {
        "atom_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "ref_element": torch.tensor([[6, 6, 6, 6]]),
        "ref_atom_name_chars": torch.tensor(
            [[_encoded_name(name) for name in ("CA", "CA", "CA", "CA")]]
        ),
    }
    output = {
        "sample_atom_coords": torch.arange(12, dtype=torch.float32).reshape(1, 4, 3),
        "distogram_logits": torch.zeros(1, 4, 4, 2),
    }

    builder = object.__new__(ESMFold2InputBuilder)
    decoded = builder.decode(output, features, chain_infos, num_diffusion_samples=1)

    assert decoded.plddt is None
    assert decoded.ptm is None
    assert decoded.iptm is None
    assert torch.isnan(torch.from_numpy(decoded.complex.plddt)).all()
    assert decoded.complex.chain_id.tolist() == [0, 0, 1, 1]

    cif = decoded.complex.to_mmcif()
    assert "?" in cif
    assert "nan" not in cif.lower()


def test_a_prepared_structure_decodes_into_one_token_per_residue_and_skips_absent_atoms() -> None:
    chain_type = np.dtype(
        [("asym_id", "i4"), ("mol_type", "i4"), ("entity_id", "i4"), ("name", "U4"), ("res_idx", "i4"), ("res_num", "i4")]
    )
    residue_type = np.dtype([("name", "U4"), ("atom_idx", "i4"), ("atom_num", "i4")])
    atom_type = np.dtype([("is_present", "?"), ("element", "i4"), ("name", "i4", (4,))])
    structure = SimpleNamespace(
        chains=np.array([(0, MOL_TYPE_PROTEIN, 0, "A", 0, 2), (1, MOL_TYPE_NONPOLYMER, 1, "L", 2, 1)], dtype=chain_type),
        residues=np.array([("ALA", 0, 2), ("GLY", 2, 1), ("ZN", 3, 1)], dtype=residue_type),
        atoms=np.array(
            [
                (True, 7, _encoded_name("N")),
                (True, 6, _encoded_name("CA")),
                (False, 6, _encoded_name("C")),
                (True, 30, _encoded_name("ZN")),
            ],
            dtype=atom_type,
        ),
    )  # three residues over four atoms, one of them absent
    coordinates = torch.arange(9, dtype=torch.float32).reshape(3, 3)  # (present atoms, 3)

    decoded = build_molecular_complex(structure, coordinates, torch.tensor([0.9, 0.8, 0.7]), "decoded")

    assert decoded.id == "decoded" and decoded.sequence == ["ALA", "GLY", "ZN"]
    assert decoded.chain_id.tolist() == [0, 0, 1] and decoded.token_to_atoms.tolist() == [[0, 2], [2, 2], [2, 3]]
    assert decoded.atom_names.tolist() == ["N", "CA", "ZN"] and decoded.atom_hetero.tolist() == [False, False, True]
    assert decoded.atom_elements.tolist() == [get_element_symbol(7), get_element_symbol(6), get_element_symbol(30)]
    assert decoded.plddt.tolist() == pytest.approx([0.9, 0.8, 0.7]) and decoded.atom_positions.tolist() == coordinates.tolist()
    assert decoded.metadata.chain_lookup == {0: "A", 1: "L"} and decoded.metadata.entity_lookup == {0: "polymer", 1: "non-polymer"}
