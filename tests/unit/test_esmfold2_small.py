"""CPU contracts for the ESMFold2 300M and 600M checkpoints."""

from __future__ import annotations

import importlib
import json
import pytest
import torch

from pathlib import Path
from types import SimpleNamespace
from safetensors.torch import save_file
from torch import Tensor, nn

from fastplms.models.esmfold2.configuration_esmfold2 import (
    ESMFold2Config,
    normalize_esmc_id,
)
from fastplms.models.esmfold2.modeling_esmfold2 import (
    ESMFold2Output,
    _install_esmc_backbone,
    _load_fastplms_esmplusplus_for_esmfold2,
    _manifest_esmc_checkpoint_contract,
)
from fastplms.models.esmfold2.modeling_esmfold2_common import NUM_RES_TYPES
from fastplms.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)
from fastplms.models.esmfold2.protein_utils import prepare_protein_features
from fastplms.registry import RegistryError, get_model_registry, load_model_registry
from tools.validation import esmfold2_small as esmfold2_small_validation


ROOT = Path(__file__).resolve().parents[2]
SMALL_MODELS = (
    ("esmfold2_300", "fold300", 960, 30, "esmc_small"),
    ("esmfold2_600", "fold600", 1152, 36, "esmc_large"),
)
SMALL_ARTIFACTS = tuple((model_id, directory_name) for model_id, directory_name, *_ in SMALL_MODELS)
EXPECTED_AUTO_MAP = {
    "AutoConfig": ("fastplms.models.esmfold2.configuration_esmfold2.ESMFold2Config"),
    "AutoModel": (
        "fastplms.models.esmfold2.modeling_esmfold2_experimental.ESMFold2ExperimentalModel"
    ),
    "AutoModelForSequenceClassification": (
        "fastplms.models.esmfold2.modeling_esmfold2_classification."
        "ESMFold2ExperimentalForSequenceClassification"
    ),
    "AutoModelForTokenClassification": (
        "fastplms.models.esmfold2.modeling_esmfold2_classification."
        "ESMFold2ExperimentalForTokenClassification"
    ),
}
CONFIDENCE_OUTPUT_NAMES = (
    "plddt_logits",
    "plddt",
    "plddt_per_atom",
    "plddt_ca",
    "complex_plddt",
    "complex_iplddt",
    "pae_logits",
    "pae",
    "ptm",
    "iptm",
    "pair_chains_iptm",
)


@pytest.mark.cpu_contract
def test_small_validation_normalizes_candidate_backbone_prefix() -> None:
    assert (
        esmfold2_small_validation._canonical_backbone_name("model.transformer.norm.weight")
        == "transformer.norm.weight"
    )
    assert esmfold2_small_validation._is_encoder_backbone_name("model.transformer.norm.weight")
    assert not esmfold2_small_validation._is_encoder_backbone_name("model.sequence_head.0.weight")


@pytest.mark.cpu_contract
def test_small_validation_finds_features_through_artifact_wrapper_mro() -> None:
    class NativeRuntime:
        pass

    class ArtifactWrapper(NativeRuntime):
        pass

    NativeRuntime.__module__ = "fastplms.models.esmfold2.modeling_esmfold2_experimental"
    ArtifactWrapper.__module__ = "transformers_modules.ESMFold2_hyphen_300.modeling_fastplms"
    model = object.__new__(ArtifactWrapper)

    feature_module = esmfold2_small_validation._feature_module_for_model(model)

    assert feature_module.__name__ == "fastplms.models.esmfold2.protein_utils"


@pytest.mark.cpu_contract
def test_small_validation_accepts_documented_candidate_output_extensions() -> None:
    length = len(esmfold2_small_validation.SEQUENCE)
    tensors = {
        "output__last_hidden_state": torch.zeros(1, length, length, 2),
        "output__representative_atom_coords": torch.zeros(1, length, 3),
        "output__sample_atom_coords": torch.zeros(1, 1, 32, 3),
        "feature__distogram_atom_idx": torch.zeros(1, length, dtype=torch.long),
    }
    assert esmfold2_small_validation._validate_candidate_outputs(tensors) == []


def _fixture_config_path(directory_name: str) -> Path:
    return ROOT / "tests" / "fixtures" / "esmfold2_small" / f"{directory_name}.json"


@pytest.mark.cpu_contract
@pytest.mark.parametrize(
    ("model_id", "directory_name", "lm_d_model", "lm_num_layers", "backbone_id"),
    SMALL_MODELS,
    ids=("esmfold2_300", "esmfold2_600"),
)
def test_small_training_base_config_matches_manifest_contract(
    model_id: str,
    directory_name: str,
    lm_d_model: int,
    lm_num_layers: int,
    backbone_id: str,
) -> None:
    registry = get_model_registry()
    spec = registry[model_id]
    raw = json.loads(_fixture_config_path(directory_name).read_text(encoding="utf-8"))
    config = ESMFold2Config.from_pretrained(_fixture_config_path(directory_name))

    assert spec.msa_conditioning is False
    assert spec.backbone_model == backbone_id
    assert spec.backbone is not None
    assert raw["esmc_id"] == spec.backbone.repo_id
    assert raw["architectures"] == ["ESMFold2ExperimentalModel"]
    assert raw["type"] == config.type == "experimental"
    assert raw["lm_d_model"] == config.lm_d_model == lm_d_model
    assert raw["lm_num_layers"] == config.lm_num_layers == lm_num_layers
    assert raw["folding_trunk"]["n_layers"] == config.folding_trunk.n_layers == 24
    assert raw["msa_encoder"]["enabled"] is config.msa_encoder.enabled is False
    assert raw["disable_msa_features"] is config.disable_msa_features is True
    assert raw["confidence_head"]["enabled"] is config.confidence_head.enabled is False
    assert (
        raw["structure_head"]["inference_num_steps"]
        == (config.structure_head.inference_num_steps)
        == 15
    )
    assert config.esmc_id == registry[backbone_id].fast.repo_id


@pytest.mark.cpu_contract
@pytest.mark.parametrize(
    ("model_id", "directory_name"),
    SMALL_ARTIFACTS,
    ids=("esmfold2_300", "esmfold2_600"),
)
def test_small_training_base_config_roundtrips(
    tmp_path: Path,
    model_id: str,
    directory_name: str,
) -> None:
    del model_id
    config = ESMFold2Config.from_pretrained(_fixture_config_path(directory_name))
    config.save_pretrained(tmp_path)
    reloaded = ESMFold2Config.from_pretrained(tmp_path)

    expected = config.to_dict()
    observed = reloaded.to_dict()
    observed["_name_or_path"] = expected["_name_or_path"]
    assert observed == expected


@pytest.mark.cpu_contract
@pytest.mark.parametrize("model_id", ("esmfold2_300", "esmfold2_600"))
def test_small_models_advertise_the_experimental_autoclass_routes(
    model_id: str,
) -> None:
    spec = get_model_registry()[model_id]
    assert dict(spec.auto_map) == EXPECTED_AUTO_MAP

    for class_path in spec.auto_map.values():
        module_name, class_name = class_path.rsplit(".", maxsplit=1)
        assert getattr(importlib.import_module(module_name), class_name)


@pytest.mark.cpu_contract
@pytest.mark.parametrize(
    ("model_id", "backbone_id"),
    tuple(
        (model_id, backbone_id)
        for model_id, _directory_name, _lm_d_model, _lm_num_layers, backbone_id in SMALL_MODELS
    ),
    ids=("esmfold2_300", "esmfold2_600"),
)
def test_small_native_backbone_aliases_use_selected_manifest_revision(
    model_id: str,
    backbone_id: str,
) -> None:
    registry = get_model_registry()
    spec = registry[model_id]
    backbone = registry[backbone_id]
    assert spec.backbone is not None

    for alias in (spec.backbone.repo_id, backbone.official.repo_id):
        assert normalize_esmc_id(alias) == backbone.fast.repo_id

    revision, files = _manifest_esmc_checkpoint_contract(spec.backbone.repo_id)
    assert revision == backbone.fast.revision
    assert files == {item.path: item.encoded for item in backbone.fast.files}


@pytest.mark.cpu_contract
def test_small_backbone_source_requires_a_model_reference(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = manifest.replace('backbone_model = "esmc_small"\n', "", 1)
    assert invalid != manifest
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RegistryError, match="backbone_model is required"):
        load_model_registry(path)


@pytest.mark.cpu_contract
def test_small_backbone_model_requires_a_pinned_source(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    backbone_line = 'backbone = { repo = "biohub/ESMC-300M-1500000", '
    invalid = "\n".join(
        line for line in manifest.splitlines() if not line.startswith(backbone_line)
    )
    assert invalid != manifest
    path = tmp_path / "models.toml"
    path.write_text(invalid + "\n", encoding="utf-8")

    with pytest.raises(RegistryError, match=r"backbone must pin an ESM\+\+ dependency"):
        load_model_registry(path)


@pytest.mark.cpu_contract
@pytest.mark.parametrize(
    ("hidden_size", "num_hidden_layers", "expected_checkpoint_dtype"),
    ((960, 30, torch.bfloat16), (1152, 36, torch.bfloat16), (960, 36, torch.float32)),
    ids=("esmc_small", "esmc_large", "unrecognized_depth_shape"),
)
def test_small_loader_preserves_bf16_rounding_only_for_declared_backbones(
    monkeypatch: pytest.MonkeyPatch,
    hidden_size: int,
    num_hidden_layers: int,
    expected_checkpoint_dtype: torch.dtype,
) -> None:
    from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
        ESMplusplusConfig,
        ESMplusplusModel,
    )

    config = ESMplusplusConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=1,
    )
    observed: list[torch.dtype] = []

    class TinyESMplusplus(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.weight = nn.Parameter(torch.ones(1))

    def load_config(cls: type[ESMplusplusConfig], source: str, **kwargs: object):
        del cls, source, kwargs
        return config

    def load_model(cls: type[ESMplusplusModel], source: str, **kwargs: object):
        del cls, source
        observed.append(kwargs["torch_dtype"])
        return TinyESMplusplus()

    monkeypatch.setattr(ESMplusplusConfig, "from_pretrained", classmethod(load_config))
    monkeypatch.setattr(ESMplusplusModel, "from_pretrained", classmethod(load_model))

    adapter = _load_fastplms_esmplusplus_for_esmfold2(
        esmc_model_path="Synthyra/ESMplusplus_small",
        attn_backend="sdpa",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert observed == [expected_checkpoint_dtype]
    assert next(adapter.parameters()).dtype == torch.float32


@pytest.mark.cpu_contract
@pytest.mark.parametrize("lm_d_model,lm_num_layers", ((960, 30), (1152, 36)))
def test_small_fp8_rejection_precedes_gpu_and_transformer_engine_access(
    monkeypatch: pytest.MonkeyPatch,
    lm_d_model: int,
    lm_num_layers: int,
) -> None:
    from fastplms.models.esmfold2 import modeling_esmfold2

    model = SimpleNamespace(
        config=SimpleNamespace(lm_d_model=lm_d_model, lm_num_layers=lm_num_layers),
        device=torch.device("cpu"),
    )

    def fail_if_reached(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("small-backbone FP8 validation reached runtime access")

    monkeypatch.setattr(modeling_esmfold2, "_resolve_esmc_precision", fail_if_reached)
    monkeypatch.setattr(modeling_esmfold2.torch.cuda, "is_available", fail_if_reached)
    monkeypatch.setattr(modeling_esmfold2, "_load_transformer_engine", fail_if_reached)

    with pytest.raises(ValueError, match="only for the ESMC-6B backbone"):
        _install_esmc_backbone(model, "unused", precision="fp8")


class _TinyStructureHead(nn.Module):
    def sample(self, **kwargs: object) -> dict[str, Tensor]:
        ref_pos = kwargs["ref_pos"]
        if not isinstance(ref_pos, Tensor):
            raise TypeError("test structure head expects ref_pos")
        return {"sample_atom_coords": ref_pos.float()}


def _tiny_experimental_config() -> ESMFold2Config:
    atom_token_width = 8
    input_feature_width = atom_token_width // 2 + 2 * NUM_RES_TYPES + 1
    return ESMFold2Config(
        type="experimental",
        d_single=8,
        d_pair=8,
        num_loops=0,
        num_diffusion_samples=1,
        lm_d_model=8,
        lm_num_layers=1,
        disable_msa_features=True,
        inputs={
            "d_inputs": input_feature_width,
            "atom_encoder": {
                "d_atom": 8,
                "d_token": atom_token_width,
                "n_blocks": 0,
                "n_heads": 2,
                "swa_window_size": 32,
                "expansion_ratio": 2,
                "n_spatial_rope_pairs_per_axis": 1,
                "n_uid_rope_pairs": 1,
            },
        },
        folding_trunk={"n_layers": 0, "n_heads": 2, "dropout": 0.0},
        structure_head={
            "diffusion_module": {
                "c_atom": 8,
                "c_token": 8,
                "c_z": 8,
                "c_s_inputs": input_feature_width,
                "fourier_dim": 8,
                "atom_num_blocks": 0,
                "atom_num_heads": 2,
                "token_num_blocks": 0,
                "token_num_heads": 2,
                "transition_multiplier": 2,
            },
            "distogram_bins": 8,
            "inference_num_steps": 1,
        },
        confidence_head={
            "enabled": False,
            "folding_trunk": {"n_layers": 0, "n_heads": 2, "dropout": 0.0},
            "num_plddt_bins": 4,
            "num_pde_bins": 4,
            "num_pae_bins": 4,
            "distogram_bins": 8,
        },
        msa_encoder={"enabled": False},
        msa_conditioning=False,
        lm_encoder={"enabled": False, "n_layers": 0},
        parcae={"enabled": True, "min_steps": 1, "max_steps": 1, "coda_n_layers": 0},
    )


@pytest.mark.cpu_contract
def test_small_experimental_forward_omits_confidence_outputs() -> None:
    model = ESMFold2ExperimentalModel(_tiny_experimental_config()).eval()
    model.structure_head = _TinyStructureHead()
    features = prepare_protein_features("AC")
    for name in (
        "msa",
        "msa_attention_mask",
        "has_deletion",
        "deletion_value",
        "deletion_mean",
    ):
        features.pop(name, None)

    output = model(
        **features,
        calculate_confidence=True,
        num_loops=0,
        num_sampling_steps=1,
        num_diffusion_samples=1,
        seed=7,
    )

    assert isinstance(output, ESMFold2Output)
    assert model.confidence_head is None
    assert set(output.keys()).isdisjoint(CONFIDENCE_OUTPUT_NAMES)
    assert all(getattr(output, name) is None for name in CONFIDENCE_OUTPUT_NAMES)


def test_validation_reads_real_safetensors_key_inventory(tmp_path: Path) -> None:
    model = nn.Module()
    model._esmc = nn.Linear(2, 2)
    snapshot = tmp_path / "backbone"
    snapshot.mkdir()
    save_file(model._esmc.state_dict(), str(snapshot / "model.safetensors"))
    esmfold2_small_validation._validate_backbone_keys(snapshot, model)

    save_file({"weight": model._esmc.weight}, str(snapshot / "model.safetensors"))
    with pytest.raises(RuntimeError, match="Backbone state-key validation failed"):
        esmfold2_small_validation._validate_backbone_keys(snapshot, model)
