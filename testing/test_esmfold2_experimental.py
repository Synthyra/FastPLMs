"""ESMFold2 experimental model tests."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModel

from fastplms.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.esmfold2.get_weights import EXPERIMENTAL_AUTO_MAP
from fastplms.esmfold2.modeling_esmfold2_common import NUM_RES_TYPES
from fastplms.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)
from fastplms.esmfold2.protein_utils import prepare_protein_features


TEST_SEQUENCE = "MSTNPKPQRKTKRNT"
OFFICIAL_REPO = "biohub/ESMFold2-Experimental-Fast"
FAST_REPO = "Synthyra/ESMFold2-Experimental-Fast"
OUTPUT_TOLERANCES = {
    "distogram_logits": 0.0,
    "plddt": 1e-6,
    "pae": 0.0,
    "ptm": 0.0,
    "iptm": 0.0,
}


def _enable_deterministic_forward() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _load_official_model() -> torch.nn.Module:
    module = pytest.importorskip(
        "transformers.models.esmfold2.modeling_esmfold2_experimental"
    )
    official_cls = module.ESMFold2ExperimentalModel
    return (
        official_cls.from_pretrained(
            OFFICIAL_REPO,
            load_esmc=False,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )


def _load_fast_model() -> ESMFold2ExperimentalModel:
    return (
        ESMFold2ExperimentalModel.from_pretrained(
            OFFICIAL_REPO,
            load_esmc=False,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )


def _run_short_fold(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    common_module_name = (
        model.__class__.__module__.rsplit(".", 1)[0] + ".modeling_esmfold2_common"
    )
    common_module = importlib.import_module(common_module_name)
    with common_module._seed_context(0), torch.no_grad():
        return model.infer_protein(
            TEST_SEQUENCE,
            num_loops=1,
            num_sampling_steps=2,
            num_diffusion_samples=1,
            calculate_confidence=True,
            seed=0,
        )


def _assert_forward_parity() -> None:
    _enable_deterministic_forward()
    official_model = _load_official_model()
    fast_model = _load_fast_model()

    official_output = _run_short_fold(official_model)
    fast_output = _run_short_fold(fast_model)

    for key, atol in OUTPUT_TOLERANCES.items():
        torch.testing.assert_close(
            fast_output[key],
            official_output[key],
            rtol=0.0,
            atol=atol,
            msg=f"ESMFold2 experimental output mismatch: {key}",
        )

    del official_model, fast_model, official_output, fast_output
    torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
def test_esmfold2_experimental_weight_parity() -> None:
    official_model = _load_official_model()
    fast_model = _load_fast_model()

    official_state = official_model.state_dict()
    fast_state = fast_model.state_dict()
    assert official_state.keys() == fast_state.keys()

    for name, official_tensor in official_state.items():
        torch.testing.assert_close(
            fast_state[name],
            official_tensor,
            rtol=0.0,
            atol=0.0,
            msg=f"ESMFold2 experimental parameter mismatch: {name}",
        )

    del official_model, fast_model
    torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
def test_esmfold2_experimental_forward_parity() -> None:
    env = os.environ.copy()
    with tempfile.TemporaryDirectory() as module_cache:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["HF_MODULES_CACHE"] = module_cache
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--esmfold2-experimental-forward-parity",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    if result.returncode != 0 and "Skipped:" in result.stderr:
        pytest.skip(result.stderr.split("Skipped:", 1)[1].strip())
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
def test_esmfold2_experimental_res_type_soft_gradients() -> None:
    model = _load_fast_model()
    features = {
        name: tensor.cuda() for name, tensor in prepare_protein_features(TEST_SEQUENCE).items()
    }
    res_type_soft = F.one_hot(
        features["res_type"].long(), num_classes=NUM_RES_TYPES
    ).float()
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
    loss = output["distogram_logits"].float().mean()
    loss.backward()

    assert res_type_soft.grad is not None
    assert torch.isfinite(res_type_soft.grad).all()
    assert res_type_soft.grad.abs().sum().item() > 0

    del model, output, features
    torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
def test_esmfold2_experimental_automodel_loads() -> None:
    try:
        model = AutoModel.from_pretrained(
            FAST_REPO,
            trust_remote_code=True,
            load_esmc=False,
            dtype=torch.float32,
        )
    except OSError as exc:
        pytest.skip(f"{FAST_REPO} is not available yet: {exc}")
    model = model.eval().cuda()

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


if __name__ == "__main__":
    assert len(sys.argv) == 2
    assert sys.argv[1] == "--esmfold2-experimental-forward-parity"
    _assert_forward_parity()
