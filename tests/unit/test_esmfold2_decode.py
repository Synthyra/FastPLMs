"""CPU checks for confidence-disabled ESMFold2 structure decoding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastplms.models.esmfold2.esmfold2_constants import MOL_TYPE_PROTEIN
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
    result = builder.decode(output, features, chain_infos, num_diffusion_samples=1)

    assert result.plddt is None
    assert result.ptm is None
    assert result.iptm is None
    assert torch.isnan(torch.from_numpy(result.complex.plddt)).all()
    assert result.complex.chain_id.tolist() == [0, 0, 1, 1]

    cif = result.complex.to_mmcif()
    assert "?" in cif
    assert "nan" not in cif.lower()
