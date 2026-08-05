from __future__ import annotations

import inspect
import os
import subprocess
import sys

import pytest
import torch

from pathlib import Path
from types import MethodType, SimpleNamespace

from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.esmfold2_msa import MSA
from fastplms.models.esmfold2.esmfold2_processor import ESMFold2InputBuilder
from fastplms.models.esmfold2.esmfold2_types import ProteinInput, StructurePredictionInput
from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from fastplms.models.esmfold2.modeling_esmfold2_common import (
    MSA_CONDITIONING_INPUT_NAMES,
    PREPARED_AUXILIARY_INPUT_NAMES,
    validate_msa_conditioning_inputs,
    validate_prepared_auxiliary_inputs,
)
from fastplms.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)
from fastplms.registry import RegistryError, load_model_registry
from tools.artifacts.build import (
    ArtifactError,
    _apply_artifact_config_contract,
    _expected_registry_provenance,
)


ROOT = Path(__file__).resolve().parents[2]


def test_esmfold2_msa_conditioning_is_manifest_typed() -> None:
    registry = load_model_registry()
    assert {
        spec.id: spec.msa_conditioning for spec in registry.by_family("esmfold2")
    } == {
        "esmfold2": True,
        "esmfold2_fast": False,
        "esmfold2_experimental_cutoff2025": True,
        "esmfold2_experimental_fast_cutoff2025": False,
    }
    assert all(
        spec.msa_conditioning is None
        for spec in registry.values()
        if spec.family.id != "esmfold2"
    )


def test_esmfold2_msa_conditioning_is_required_and_family_scoped(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    candidate = tmp_path / "models.toml"
    candidate.write_text(manifest.replace("msa_conditioning = true\n", "", 1), encoding="utf-8")
    with pytest.raises(RegistryError, match="must be an explicit boolean"):
        load_model_registry(candidate)

    candidate.write_text(
        manifest.replace('id = "esm2_8m"\n', 'id = "esm2_8m"\nmsa_conditioning = false\n', 1),
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="only valid for ESMFold2"):
        load_model_registry(candidate)


@pytest.mark.parametrize("value", [None, "false", 0, 1])
def test_esmfold2_config_rejects_invalid_msa_contract_values(value: object) -> None:
    with pytest.raises(TypeError, match="msa_conditioning must be a boolean"):
        ESMFold2Config(msa_encoder={"enabled": False}, msa_conditioning=value)


def test_esmfold2_config_requires_encoder_contract_agreement() -> None:
    with pytest.raises(ValueError, match=r"must match msa_encoder\.enabled"):
        ESMFold2Config(msa_encoder={"enabled": True}, msa_conditioning=False)
    config = ESMFold2Config(msa_encoder={"enabled": True})
    assert config.msa_conditioning is True
    assert config.to_dict()["msa_conditioning"] is True


@pytest.mark.parametrize("msa_conditioning", [False, True])
def test_esmfold2_config_msa_contract_survives_save_reload(
    tmp_path: Path,
    msa_conditioning: bool,
) -> None:
    config = ESMFold2Config(
        msa_encoder={"enabled": msa_conditioning},
        msa_conditioning=msa_conditioning,
    )
    config.save_pretrained(tmp_path)
    reloaded = ESMFold2Config.from_pretrained(tmp_path)
    assert reloaded.msa_conditioning is msa_conditioning
    assert reloaded.msa_encoder.enabled is msa_conditioning
    assert reloaded.to_dict()["msa_conditioning"] is msa_conditioning


def test_artifact_config_materializes_manifest_msa_contract() -> None:
    registry = load_model_registry()
    for model_id, expected in (
        ("esmfold2", True),
        ("esmfold2_fast", False),
        ("esmfold2_experimental_cutoff2025", True),
        ("esmfold2_experimental_fast_cutoff2025", False),
    ):
        config = {"msa_encoder": {"enabled": expected}}
        _apply_artifact_config_contract(registry[model_id], config)
        assert config["msa_conditioning"] is expected
        assert config["msa_encoder"]["enabled"] is expected

    with pytest.raises(ArtifactError, match="msa_encoder must be an object"):
        _apply_artifact_config_contract(registry["esmfold2_fast"], {})


def test_artifact_provenance_materializes_manifest_msa_contract() -> None:
    registry = load_model_registry()
    for spec in registry.by_family("esmfold2"):
        provenance = _expected_registry_provenance(registry, spec)
        assert provenance["msa_conditioning"] is spec.msa_conditioning
    assert "msa_conditioning" not in _expected_registry_provenance(
        registry,
        registry["esm2_8m"],
    )


@pytest.mark.parametrize(
    ("model_id", "config_enabled"),
    (("esmfold2", False), ("esmfold2_fast", True)),
)
def test_artifact_config_rejects_manifest_msa_disagreement(
    model_id: str,
    config_enabled: bool,
) -> None:
    registry = load_model_registry()
    with pytest.raises(ArtifactError, match=r"differs from models\.toml"):
        _apply_artifact_config_contract(
            registry[model_id],
            {"msa_encoder": {"enabled": config_enabled}},
        )

    expected = registry[model_id].msa_conditioning
    with pytest.raises(ArtifactError, match=r"config\.msa_conditioning differs"):
        _apply_artifact_config_contract(
            registry[model_id],
            {
                "msa_encoder": {"enabled": expected},
                "msa_conditioning": not expected,
            },
        )


@pytest.mark.parametrize("provided_name", MSA_CONDITIONING_INPUT_NAMES)
def test_fast_checkpoint_rejects_every_low_level_msa_input(provided_name: str) -> None:
    config = ESMFold2Config(msa_encoder={"enabled": False}, msa_conditioning=False)
    values = {name: None for name in MSA_CONDITIONING_INPUT_NAMES}
    values[provided_name] = torch.zeros(1)  # (n=1,)
    with pytest.raises(ValueError, match=provided_name):
        validate_msa_conditioning_inputs(config, **values)


def test_full_checkpoint_accepts_low_level_msa_inputs() -> None:
    config = ESMFold2Config(msa_encoder={"enabled": True}, msa_conditioning=True)
    tensor = torch.zeros(1)  # (n=1,)
    validate_msa_conditioning_inputs(
        config,
        msa=tensor,
        msa_attention_mask=tensor,
        has_deletion=tensor,
        deletion_value=tensor,
        deletion_mean=tensor,
    )


@pytest.mark.parametrize("model_type", [ESMFold2Model, ESMFold2ExperimentalModel])
def test_forward_explicitly_accepts_every_prepared_auxiliary_input(
    model_type: type[ESMFold2Model] | type[ESMFold2ExperimentalModel],
) -> None:
    signature = inspect.signature(model_type.forward)
    assert set(PREPARED_AUXILIARY_INPUT_NAMES) <= set(signature.parameters)
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def test_inert_prepared_auxiliary_conditioning_is_accepted() -> None:
    validate_prepared_auxiliary_inputs(
        pocket_feature=torch.zeros(2, dtype=torch.long),  # (n=2,)
        disto_cond=torch.zeros(2, 2),  # (l=2, l=2)
        disto_cond_mask=torch.zeros(2, 2, dtype=torch.bool),  # (l=2, l=2)
    )


@pytest.mark.parametrize(
    ("provided_name", "values"),
    (
        (
            "pocket_feature",
            {
                "pocket_feature": torch.ones(1, dtype=torch.long),
                "disto_cond": None,
                "disto_cond_mask": None,
            },
        ),
        (
            "disto_cond",
            {
                "pocket_feature": None,
                "disto_cond": torch.ones(1, 1),
                "disto_cond_mask": None,
            },
        ),
        (
            "disto_cond_mask",
            {
                "pocket_feature": None,
                "disto_cond": None,
                "disto_cond_mask": torch.ones(1, 1, dtype=torch.bool),
            },
        ),
    ),
)
def test_active_prepared_auxiliary_conditioning_is_rejected(
    provided_name: str,
    values: dict[str, torch.Tensor | None],
) -> None:
    with pytest.raises(NotImplementedError, match=provided_name):
        validate_prepared_auxiliary_inputs(**values)


@pytest.mark.parametrize("model_type", [ESMFold2Model, ESMFold2ExperimentalModel])
@pytest.mark.parametrize("provided_name", MSA_CONDITIONING_INPUT_NAMES)
def test_fast_model_forward_rejects_every_msa_input_before_computation(
    model_type: type[ESMFold2Model] | type[ESMFold2ExperimentalModel],
    provided_name: str,
) -> None:
    config = ESMFold2Config(msa_encoder={"enabled": False}, msa_conditioning=False)
    model = SimpleNamespace(config=config)
    tensor = torch.zeros(1)  # (n=1,)
    required = {
        name: tensor
        for name in (
            "token_index",
            "residue_index",
            "asym_id",
            "sym_id",
            "entity_id",
            "mol_type",
            "res_type",
            "token_bonds",
            "token_attention_mask",
            "ref_pos",
            "ref_element",
            "ref_charge",
            "ref_atom_name_chars",
            "ref_space_uid",
            "atom_attention_mask",
            "atom_to_token",
            "distogram_atom_idx",
        )
    }
    required[provided_name] = tensor
    with pytest.raises(ValueError, match=provided_name):
        model_type.forward(model, **required)


def _builder_with_features(features: dict[str, torch.Tensor]) -> ESMFold2InputBuilder:
    builder = object.__new__(ESMFold2InputBuilder)

    def prepare_input(
        self: ESMFold2InputBuilder,
        input: object,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[object]]:
        del self, input, seed, device
        return dict(features), []

    builder.prepare_input = MethodType(prepare_input, builder)
    return builder


def test_fast_high_level_input_rejects_explicit_msa_and_strips_synthetic_features() -> None:
    features = {
        name: torch.zeros(1)  # (n=1,)
        for name in MSA_CONDITIONING_INPUT_NAMES
    }
    features["token_index"] = torch.zeros(1)  # (n=1,)
    builder = _builder_with_features(features)
    model = SimpleNamespace(config=SimpleNamespace(msa_conditioning=False))
    sequence_only = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="AC")]
    )
    prepared, _ = builder.prepare_model_input(model, sequence_only)
    assert set(prepared) == {"token_index"}

    explicit = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence="AC", msa=MSA.from_sequences(["AC", "AC"]))
        ]
    )
    with pytest.raises(ValueError, match="rejects explicit MSAs"):
        builder.prepare_model_input(model, explicit)


def test_full_high_level_input_preserves_msa_features() -> None:
    features = {
        name: torch.zeros(1)  # (n=1,)
        for name in MSA_CONDITIONING_INPUT_NAMES
    }
    builder = _builder_with_features(features)
    model = SimpleNamespace(config=SimpleNamespace(msa_conditioning=True))
    explicit = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence="AC", msa=MSA.from_sequences(["AC", "AC"]))
        ]
    )
    prepared, _ = builder.prepare_model_input(model, explicit)
    assert set(prepared) == set(MSA_CONDITIONING_INPUT_NAMES)


def test_public_validation_survives_python_optimized_mode() -> None:
    script = r'''
import numpy as np
import torch

from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.esmfold2_affine3d import (
    Affine3D,
    RotationMatrix,
    RotationQuat,
    build_affine3d_from_coordinates,
)
from fastplms.models.esmfold2.esmfold2_misc import concat_objects, merge_ranges
from fastplms.models.esmfold2.esmfold2_protein_chain import ProteinChain
from fastplms.models.esmfold2.modeling_esmfold2_common import DropoutResidual

def expect(error_type, fn):
    try:
        fn()
    except error_type:
        return
    raise RuntimeError(f"expected {error_type.__name__}")

expect(TypeError, lambda: RotationQuat([1, 0, 0, 0]))
expect(ValueError, lambda: RotationQuat(torch.zeros(3)))
expect(ValueError, lambda: RotationMatrix(torch.zeros(2, 2)))
expect(ValueError, lambda: Affine3D(torch.zeros(2, 2), RotationMatrix.identity((2,))))
expect(ValueError, lambda: build_affine3d_from_coordinates(torch.zeros(2, 3, 3)))
expect(ValueError, lambda: DropoutResidual(0.1, batch_dim=0))
expect(ValueError, lambda: DropoutResidual(0.1, batch_dim=True))
expect(TypeError, lambda: concat_objects(["A", "B"], separator=1))
expect(ValueError, lambda: merge_ranges([range(0, 1)], merge_gap_max=-1))
expect(ValueError, lambda: ProteinChain.from_atom37(np.zeros((2, 36, 3))))
expect(
    ValueError,
    lambda: ESMFold2Config(msa_encoder={"enabled": True}, msa_conditioning=False),
)
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
