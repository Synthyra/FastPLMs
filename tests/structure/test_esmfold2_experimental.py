"""ESMFold2 experimental model tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from pathlib import Path

from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.modeling_esmfold2_common import NUM_RES_TYPES
from fastplms.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)
from fastplms.models.esmfold2.protein_utils import prepare_protein_features
from fastplms.registry import ModelSpec, get_model_registry


TEST_SEQUENCE = "MSTNPKPQRKTKRNT"
REGISTRY = get_model_registry()
EXPERIMENTAL_SPECS = tuple(
    spec for spec in REGISTRY.by_family("esmfold2") if "experimental" in spec.id
)
DEFAULT_EXPERIMENTAL_SPEC = REGISTRY["esmfold2_experimental_fast_cutoff2025"]
EXPERIMENTAL_AUTO_MAP = {
    auto_class: class_path.removeprefix("fastplms.models.esmfold2.")
    for auto_class, class_path in REGISTRY["esmfold2_experimental_cutoff2025"].auto_map.items()
}


def _load_fast_model(spec: ModelSpec) -> ESMFold2ExperimentalModel:
    return (
        ESMFold2ExperimentalModel.from_pretrained(
            spec.fast.repo_id,
            revision=spec.fast.revision,
            load_esmc=False,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )
@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
def test_esmfold2_experimental_res_type_soft_gradients() -> None:
    model = _load_fast_model(DEFAULT_EXPERIMENTAL_SPEC)
    features = {
        name: tensor.cuda() for name, tensor in prepare_protein_features(TEST_SEQUENCE).items()
    }
    # res_type_soft: (...)
    res_type_soft = F.one_hot(features["res_type"].long(), num_classes=NUM_RES_TYPES).float()
    res_type_soft.requires_grad_(True)

    output = model(
        **features,
        res_type_soft=res_type_soft,
        num_loops=0,
        num_sampling_steps=1,
        num_diffusion_samples=1,
        calculate_confidence=False,
        seed=0,
    )
    # loss: ()
    loss = output["distogram_logits"].float().mean()
    loss.backward()

    assert "representative_atom_coords" in output
    assert output["representative_atom_coords"].shape[-1] == 3
    assert output["representative_atom_coords"].shape[-2] == features["res_type"].shape[1]
    assert res_type_soft.grad is not None
    assert torch.isfinite(res_type_soft.grad).all()
    assert res_type_soft.grad.abs().sum().item() > 0

    del model, output, features
    torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("spec", EXPERIMENTAL_SPECS, ids=lambda spec: spec.id)
def test_esmfold2_experimental_model_loads(spec: ModelSpec) -> None:
    model = _load_fast_model(spec)

    assert callable(model.infer_protein_as_pdb)
    assert callable(model.fold)
    assert callable(model.prepare_structure_input)

    del model
    torch.cuda.empty_cache()


def test_esmfold2_experimental_export_config(tmp_path: Path) -> None:
    config = ESMFold2Config(type="experimental")
    config.auto_map = EXPERIMENTAL_AUTO_MAP
    config.architectures = ["ESMFold2ExperimentalModel"]
    config.save_pretrained(tmp_path)

    loaded = ESMFold2Config.from_pretrained(tmp_path)
    assert loaded.auto_map == EXPERIMENTAL_AUTO_MAP
    assert loaded.architectures == ["ESMFold2ExperimentalModel"]
