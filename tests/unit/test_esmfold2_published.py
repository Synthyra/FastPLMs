"""CPU checks for the published ESMFold2-300 validation helpers."""

from __future__ import annotations

import pytest
import torch

from tools.validation import esmfold2_published


@pytest.mark.cpu_contract
def test_published_output_validation_accepts_single_sample_coordinates() -> None:
    output = {"sample_atom_coords": torch.zeros(1, 32, 3)}

    assert esmfold2_published._validate_outputs(output) == {"sample_atom_coords": [1, 32, 3]}


@pytest.mark.cpu_contract
def test_published_output_validation_rejects_confidence_and_nonfinite_values() -> None:
    with pytest.raises(RuntimeError, match="Confidence output"):
        esmfold2_published._validate_outputs(
            {
                "sample_atom_coords": torch.zeros(1, 1, 32, 3),
                "plddt": torch.zeros(1, 32),
            }
        )

    with pytest.raises(RuntimeError, match="NaN or infinity"):
        esmfold2_published._validate_outputs(
            {"sample_atom_coords": torch.full((1, 1, 32, 3), float("nan"))}
        )
