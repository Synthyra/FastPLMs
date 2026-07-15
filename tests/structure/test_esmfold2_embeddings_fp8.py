from __future__ import annotations

import inspect
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fastplms.embeddings import EmbeddingBatch
from fastplms.models.esmfold2.attention import ESMFold2AttentionMixin
from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.embedding import ESMFold2EmbeddingMixin
from fastplms.models.esmfold2.modeling_esmfold2 import (
    ESMFold2Model,
    _convert_esmc_attention_outputs_to_te,
    _drop_transient_esmc_state,
    _install_esmc_backbone,
    _resolve_esmc_precision,
)
from fastplms.models.esmfold2.modeling_esmfold2_common import (
    LanguageModelShim,
    compute_lm_hidden_states,
)
from fastplms.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)
from fastplms.registry import get_model_registry


def test_language_model_projection_matches_original_operation() -> None:
    torch.manual_seed(0)
    shim = LanguageModelShim(d_z=7, d_model=11, num_layers=3)
    H = torch.randn(2, 5, 4, 11)
    M = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])

    projected_states = shim.base_z_linear(H)
    expected = shim.base_z_combine.softmax(dim=0) @ projected_states
    expected = expected * M.unsqueeze(-1)
    Z = shim.project_sequence(H, M)

    assert torch.equal(Z, expected)
    assert tuple(Z.shape) == (2, 5, 7)
    assert not any(key.startswith("project") for key in shim.state_dict())
    assert set(shim.state_dict()) == {
        "base_z_combine",
        "base_z_mlp.0.downproject.weight",
        "base_z_mlp.0.downproject.bias",
        "base_z_mlp.0.output_mlp.0.weight",
        "base_z_mlp.0.output_mlp.0.bias",
        "base_z_mlp.0.output_mlp.2.weight",
        "base_z_mlp.0.output_mlp.2.bias",
        "base_z_mlp.1.weight",
        "base_z_mlp.1.bias",
        "base_z_linear.0.weight",
        "base_z_linear.0.bias",
        "base_z_linear.1.weight",
    }


def test_language_model_projection_preserves_single_residue_axis() -> None:
    shim = LanguageModelShim(d_z=7, d_model=11, num_layers=3)
    H = torch.randn(2, 1, 4, 11)
    M = torch.ones((2, 1), dtype=torch.bool)

    Z = shim.project_sequence(H, M)

    assert Z.shape == (2, 1, 7)


def test_language_model_projection_matches_checkpoint_dtype() -> None:
    shim = LanguageModelShim(d_z=7, d_model=11, num_layers=3).to(dtype=torch.bfloat16)
    H = torch.randn(2, 5, 4, 11, dtype=torch.float32)

    Z = shim.project_sequence(H)

    assert Z.dtype == torch.bfloat16
    assert torch.isfinite(Z).all()


def test_projection_validates_official_state_count() -> None:
    shim = LanguageModelShim()
    H = torch.zeros(1, 2, 80, 2560)
    with pytest.raises(ValueError, match="expected 81"):
        shim.project_sequence(H)

    model = SyntheticESMFold2()
    with pytest.raises(ValueError, match="official ordered 81-state"):
        model.project_esmc_hidden_states(torch.zeros(1, 2, 80, 4))


class SyntheticESMFold2(ESMFold2EmbeddingMixin, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self._esmc = object()
        self.language_model = LanguageModelShim(d_z=3, d_model=4, num_layers=80)

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def _compute_lm_hidden_states(
        self,
        input_ids: torch.Tensor,
        asym_id: torch.Tensor,
        residue_index: torch.Tensor,
        mol_type: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        del asym_id, residue_index, mol_type
        H = torch.zeros(*input_ids.shape, 81, 4)
        H[..., 0] = input_ids.unsqueeze(-1)
        return H * residue_mask[..., None, None]


def test_esmfold2_embedding_is_residue_only_and_rejects_complexes() -> None:
    model = SyntheticESMFold2()
    batch = model._embedding_batch(["ACD", "GG"])
    assert isinstance(batch, EmbeddingBatch)
    assert tuple(batch.X.shape) == (2, 3, 3)
    assert batch.residue_mask.tolist() == [[True, True, True], [True, True, False]]
    assert torch.equal(batch.X[1, 2], torch.zeros(3))
    result = model.embed_dataset(["ACD"], full_embeddings=True)
    assert result.metadata["layer"] == "all_81_esmc_states"
    assert result.metadata["projection"] == "esmfold2_learned_sequence_summary"
    assert "BOS" in result.metadata["token_policy"]["exclude"]
    with pytest.raises(ValueError, match="one ungapped protein chain"):
        model._embedding_batch(["ACD|GG"])
    with pytest.raises(ValueError, match="at least one protein residue"):
        model._embedding_batch([""])


def test_auto_precision_uses_bf16_without_probing_fp8(monkeypatch) -> None:
    import fastplms.models.esmfold2.modeling_esmfold2 as module

    monkeypatch.setattr(
        module,
        "_te_fp8_capability",
        lambda _device: pytest.fail("auto must not probe FP8 capability"),
    )
    status = _resolve_esmc_precision("auto", torch.device("cuda"))
    assert status.requested == "auto"
    assert status.resolved == "bf16"
    assert "defaults to BF16" in status.reason


def test_auto_precision_remains_bf16_when_transformer_engine_is_installed(
    monkeypatch,
) -> None:
    import fastplms.models.esmfold2.modeling_esmfold2 as module

    monkeypatch.setattr(
        module,
        "_transformer_engine_version",
        lambda: "2.12.0",
    )
    status = _resolve_esmc_precision("auto", torch.device("cuda"))
    assert status.requested == "auto"
    assert status.resolved == "bf16"
    assert "defaults to BF16" in status.reason
    assert status.transformer_engine_version == "2.12.0"


def test_explicit_fp8_is_strict(monkeypatch) -> None:
    import fastplms.models.esmfold2.modeling_esmfold2 as module

    monkeypatch.setattr(
        module,
        "_te_fp8_capability",
        lambda device: (False, f"FP8 unavailable on {device}"),
    )
    with pytest.raises(RuntimeError, match="FP8 unavailable on cuda"):
        _resolve_esmc_precision("fp8", torch.device("cuda"))


def test_fp8_capability_rejects_non_cuda_devices() -> None:
    import fastplms.models.esmfold2.modeling_esmfold2 as module

    available, reason = module._te_fp8_capability(torch.device("cpu"))
    assert available is False
    assert reason == "FP8 requires direct ESMC loading onto a CUDA device."


def test_fp8_converter_replaces_only_80_attention_output_projections(
    monkeypatch,
) -> None:
    import fastplms.models.esmfold2.modeling_esmfold2 as module

    class FakeTELinear(nn.Linear):
        def __init__(
            self,
            in_features,
            out_features,
            *,
            bias,
            params_dtype,
            device,
        ) -> None:
            super().__init__(
                in_features,
                out_features,
                bias=bias,
                device=device,
                dtype=params_dtype,
            )

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Module()
            self.attn.out_proj = nn.Linear(4, 4, bias=False)
            self.ffn = nn.Linear(4, 4, bias=False)

    backbone = nn.Module()
    backbone.layers = nn.ModuleList([Block() for _ in range(80)])
    expected_weights = [block.attn.out_proj.weight.detach().clone() for block in backbone.layers]
    monkeypatch.setattr(
        module,
        "_load_transformer_engine",
        lambda: (SimpleNamespace(Linear=FakeTELinear), SimpleNamespace()),
    )

    paths = _convert_esmc_attention_outputs_to_te(backbone)

    assert len(paths) == 80
    assert all(path.endswith(".attn.out_proj") for path in paths)
    for block, expected in zip(backbone.layers, expected_weights, strict=True):
        assert isinstance(block.attn.out_proj, FakeTELinear)
        assert isinstance(block.ffn, nn.Linear)
        assert torch.equal(block.attn.out_proj.weight, expected)


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=2)


class PrecisionOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            esmc_attn_backend="flex_attention",
            _attn_implementation="flex_attention",
            lm_d_model=4,
            lm_num_layers=2,
            esmc_id="canonical-esmc",
            esmc_precision="auto",
        )
        self._esmc = None
        self._esmc_fp8 = False
        self._ttt_lm_head = None

    @property
    def device(self) -> torch.device:
        return self.anchor.device


class _AttentionOwnerBase(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self._esmc = None

    def _check_and_adjust_attn_implementation(
        self,
        attn_implementation: str | None,
        is_init_check: bool = False,
        allow_all_kernels: bool = False,
    ) -> str:
        """Model the Transformers base-class hook used by the production owner."""

        del is_init_check, allow_all_kernels
        return attn_implementation or "sdpa"


class AttentionOwner(ESMFold2AttentionMixin, _AttentionOwnerBase):
    pass


class RecordingAttentionBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.implementations: list[str] = []

    def set_attn_implementation(self, implementation: str) -> None:
        self.implementations.append(implementation)


def test_attention_config_defaults_to_unspecified_and_normalizes_legacy_flex() -> None:
    default = ESMFold2Config()
    assert default.esmc_attn_backend is None
    assert default._attn_implementation is None

    legacy = ESMFold2Config(esmc_attn_backend="flex")
    assert legacy.esmc_attn_backend == "flex_attention"
    assert legacy._attn_implementation == "flex_attention"


def test_public_attention_setting_overrides_legacy_config_value() -> None:
    config = ESMFold2Config(
        esmc_attn_backend="flex",
        attn_implementation="eager",
    )
    assert config.esmc_attn_backend == "eager"
    assert config._attn_implementation == "eager"


def test_transformers_config_reload_applies_public_attention_override(tmp_path) -> None:
    ESMFold2Config(esmc_attn_backend="flex").save_pretrained(tmp_path)

    config = ESMFold2Config.from_pretrained(
        tmp_path,
        attn_implementation="eager",
    )

    assert config._attn_implementation == "eager"
    assert config.esmc_attn_backend == "eager"


def test_runtime_attention_setting_updates_loaded_esmc() -> None:
    config = ESMFold2Config(attn_implementation="sdpa")
    owner = AttentionOwner(config)
    esmc = RecordingAttentionBackbone()
    owner._esmc = esmc

    owner.set_attn_implementation("eager")

    assert owner.config._attn_implementation == "eager"
    assert owner.config.esmc_attn_backend == "eager"
    assert esmc.implementations == ["eager"]


@pytest.mark.parametrize("implementation", ["flex", "flash_attention_2", "unknown"])
def test_runtime_attention_rejects_unadvertised_names(implementation: str) -> None:
    owner = AttentionOwner(ESMFold2Config(attn_implementation="sdpa"))
    with pytest.raises(ValueError, match="does not support"):
        owner.set_attn_implementation(implementation)


def test_compile_helpers_do_not_mutate_global_dynamo_configuration() -> None:
    for model_class in (ESMFold2Model, ESMFold2ExperimentalModel):
        source = inspect.getsource(model_class.apply_torch_compile)
        assert "torch._dynamo.config" not in source


def test_fresh_esmfold2_import_does_not_mutate_global_torch_settings() -> None:
    script = """
import torch
import torch._dynamo

def snapshot():
    return (
        torch.get_default_dtype(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch._dynamo.config.cache_size_limit,
        torch._dynamo.config.accumulated_cache_size_limit,
        torch._dynamo.config.capture_scalar_outputs,
    )

before = snapshot()
import fastplms.embeddings
import fastplms.models.esmfold2.modeling_esmfold2
after = snapshot()
assert after == before, (before, after)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_install_records_bf16_auto_default_and_policy(monkeypatch) -> None:
    import fastplms.models.esmfold2.modeling_esmfold2 as module

    owner = PrecisionOwner()
    load_options: list[dict[str, object]] = []

    def load_backbone(**kwargs):
        load_options.append(kwargs)
        return FakeBackbone()

    monkeypatch.setattr(
        module,
        "_load_fastplms_esmplusplus_for_esmfold2",
        load_backbone,
    )
    _install_esmc_backbone(owner, "canonical-esmc", precision="auto", device=torch.device("cpu"))
    assert owner._esmc_fp8 is False
    assert owner._esmc_precision_status.resolved == "bf16"
    assert "defaults to BF16" in owner._esmc_precision_status.reason
    assert owner.config.esmc_precision == "auto"
    assert load_options[0]["attn_backend"] == "flex_attention"


def test_install_records_validated_fp8_projection_set(monkeypatch) -> None:
    import fastplms.models.esmfold2.modeling_esmfold2 as module

    owner = PrecisionOwner().to("meta")
    owner.anchor = nn.Parameter(torch.zeros((), device="meta"))
    backbone = FakeBackbone().to("meta")
    monkeypatch.setattr(
        module,
        "_te_fp8_capability",
        lambda device: (True, f"FP8 available on {device}"),
    )
    monkeypatch.setattr(
        module,
        "_load_fastplms_esmplusplus_for_esmfold2",
        lambda **kwargs: backbone,
    )
    paths = tuple(f"model.layers.{index}.attn.out_proj" for index in range(80))
    monkeypatch.setattr(
        module,
        "_convert_esmc_attention_outputs_to_te",
        lambda esmc: paths,
    )

    _install_esmc_backbone(owner, "canonical-esmc", precision="fp8", device="meta")

    assert owner._esmc_fp8 is True
    assert owner._esmc_fp8_module_paths == paths
    assert owner._esmc_precision_status.resolved == "fp8"
    assert "Converted 80 projections" in owner._esmc_precision_status.reason


def test_ttt_switches_runtime_fp8_back_to_bf16() -> None:
    calls: list[tuple[str, torch.device]] = []
    owner = SimpleNamespace(
        _esmc_fp8=True,
        _esmc_precision_policy="auto",
        _esmc_precision_status=SimpleNamespace(device="cuda:0", transformer_engine_version="2.16"),
        config=SimpleNamespace(esmc_precision="auto"),
        device=torch.device("cuda"),
        reload_esmc=lambda precision, device: calls.append((precision, device)),
    )
    ESMFold2Model._ensure_ttt_bf16(owner)
    assert calls == [("bf16", torch.device("cuda"))]
    assert owner.config.esmc_precision == "auto"
    assert owner._esmc_precision_status.requested == "auto"
    assert owner._esmc_precision_status.resolved == "bf16"


def test_runtime_esmc_state_is_not_persisted() -> None:
    state = {
        "language_model.base_z_combine": torch.ones(1),
        "_esmc.layer.weight": torch.ones(1),
        "_ttt_lm_head.weight": torch.ones(1),
    }
    _drop_transient_esmc_state(nn.Identity(), state, "", {})
    assert list(state) == ["language_model.base_z_combine"]


class RecordingESMC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_l = 0

    def forward(self, input_ids, sequence_id, output_hidden_states):
        del sequence_id, output_hidden_states
        b, sequence_length = input_ids.shape
        self.seen_l = sequence_length
        H = torch.zeros(81, b, sequence_length, 4)
        H[:] = torch.arange(81).view(81, 1, 1, 1)
        return SimpleNamespace(hidden_states=H)


def test_fp8_language_model_input_is_padded_to_multiple_of_16() -> None:
    esmc = RecordingESMC()
    sequence_length = 15
    input_ids = torch.full((1, sequence_length), 4, dtype=torch.long)
    asym_id = torch.zeros_like(input_ids)
    residue_index = torch.arange(sequence_length).unsqueeze(0)
    mol_type = torch.zeros_like(input_ids)
    M = torch.ones_like(input_ids, dtype=torch.bool)
    H = compute_lm_hidden_states(
        esmc,
        input_ids,
        asym_id,
        residue_index,
        mol_type,
        M,
        pad_to_multiple=16,
    )
    assert esmc.seen_l == 32
    assert tuple(H.shape) == (1, sequence_length, 81, 4)
    assert torch.equal(H[0, 0, :, 0], torch.arange(81, dtype=H.dtype))


def test_supported_variants_are_exactly_the_four_approved_sources() -> None:
    official_repositories = {
        spec.official.repo_id for spec in get_model_registry().by_family("esmfold2")
    }
    assert official_repositories == {
        "biohub/ESMFold2",
        "biohub/ESMFold2-Fast",
        "biohub/ESMFold2-Experimental-Cutoff2025",
        "biohub/ESMFold2-Experimental-Fast-Cutoff2025",
    }
    assert hasattr(ESMFold2Model, "project_esmc_hidden_states")
    assert hasattr(ESMFold2ExperimentalModel, "project_esmc_hidden_states")
