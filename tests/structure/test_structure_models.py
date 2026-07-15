"""Tests for structure prediction models (Boltz2, ESMFold).

These models have a fundamentally different API from the MaskedLM sequence
models, so they live in a separate test file with the `structure` marker.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModel

from fastplms.registry import get_model_registry

REGISTRY = get_model_registry()
BOLTZ2 = REGISTRY["boltz2"]
ESMFOLD = REGISTRY["esmfold"]


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
def test_boltz2_loads() -> None:
    """Boltz2 loads via AutoModel with trust_remote_code=True."""
    model = (
        AutoModel.from_pretrained(
            BOLTZ2.fast.repo_id,
            revision=BOLTZ2.fast.revision,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )

    assert model is not None
    assert hasattr(model, "predict_structure")
    assert hasattr(model, "save_as_cif")

    del model
    torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
def test_boltz2_forward() -> None:
    """Boltz2 predict_structure returns valid coordinates and confidence scores."""
    model = (
        AutoModel.from_pretrained(
            BOLTZ2.fast.repo_id,
            revision=BOLTZ2.fast.revision,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )

    sequence = "MSTNPKPQRKTKRNTNRRPQDVKFPGG"
    output = model.predict_structure(
        amino_acid_sequence=sequence,
        recycling_steps=3,
        num_sampling_steps=50,
        diffusion_samples=1,
    )

    assert output.sample_atom_coords is not None
    assert output.sample_atom_coords.ndim in (2, 3), (
        f"Expected 2D or 3D coords, got {output.sample_atom_coords.ndim}D"
    )
    assert output.sample_atom_coords.shape[-1] == 3
    assert not torch.isnan(output.sample_atom_coords).any(), "NaN in predicted coordinates"
    assert output.plddt is not None
    assert output.sequence == sequence

    del model
    torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
def test_esmfold_loads() -> None:
    """ESMFold loads via AutoModel with trust_remote_code=True."""
    model = (
        AutoModel.from_pretrained(
            ESMFOLD.fast.repo_id,
            revision=ESMFOLD.fast.revision,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )

    assert model is not None
    assert hasattr(model, "infer")
    assert hasattr(model, "fold_protein")
    assert not any(name.startswith("mlm_head.") for name in model.state_dict())
    assert not any(name.startswith("esm.contact_head.") for name in model.state_dict())

    del model
    torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
def test_esmfold_forward() -> None:
    """ESMFold infer produces valid pLDDT and structure output."""
    config = AutoConfig.from_pretrained(
        ESMFOLD.fast.repo_id,
        revision=ESMFOLD.fast.revision,
        trust_remote_code=True,
    )
    model = (
        AutoModel.from_pretrained(
            ESMFOLD.fast.repo_id,
            revision=ESMFOLD.fast.revision,
            config=config,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )

    sequence = "MKTLLILAVVAAALA"

    with torch.no_grad():
        output = model.infer(sequence)

    assert "plddt" in output
    plddt = output["plddt"]
    assert not torch.isnan(plddt).any(), "NaN in pLDDT"

    del model
    torch.cuda.empty_cache()
