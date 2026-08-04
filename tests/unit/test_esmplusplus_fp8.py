"""CPU-safe contracts for the experimental ESM++ Transformer Engine FP8 path."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fastplms.models.esm_plusplus.modeling_esm_plusplus as esmpp_module
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusFP8Status,
    ESMplusplusForMaskedLM,
    ESMplusplusModel,
    _convert_esmc_attention_outputs_to_te,
    _esmplusplus_fp8_context,
    _te_fp8_capability,
)
from fastplms.registry import load_model_registry


def _tiny_config(*, num_hidden_layers: int = 2) -> ESMplusplusConfig:
    return ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=num_hidden_layers,
        attn_backend="eager",
        pad_token_id=1,
        mask_token_id=15,
    )


def test_fp8_is_experimental_for_every_esmplusplus_checkpoint() -> None:
    models = load_model_registry().by_family("esm_plusplus")

    assert {model.id for model in models} == {"esmc_small", "esmc_large", "esmc_6b"}
    assert models[0].family.precisions == ("default", "fp8")
    assert models[0].family.experimental_precisions == ("fp8",)


class _FakeTELinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        params_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            dtype=params_dtype,
            device=device,
        )


class _SyntheticBlock(nn.Module):
    def __init__(self, *, device: str = "meta") -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.out_proj = nn.Linear(4, 4, bias=False, device=device)
        self.ffn = nn.Linear(4, 4, bias=False, device=device)


class _SyntheticBackbone(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList(
            [_SyntheticBlock() for _ in range(layer_count)]
        )


def _install_fake_transformer_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        esmpp_module,
        "_load_transformer_engine",
        lambda: (SimpleNamespace(Linear=_FakeTELinear), SimpleNamespace()),
    )


def test_fp8_status_is_explicit_and_serializable() -> None:
    model = ESMplusplusModel(_tiny_config()).eval()

    status = model.esmc_precision_status

    assert isinstance(status, ESMplusplusFP8Status)
    assert status.enabled is False
    assert status.converted_projections == 0
    assert status.as_dict() == {
        "enabled": False,
        "reason": "FP8 has not been enabled; canonical checkpoint precision is unchanged.",
        "device": "cpu",
        "transformer_engine_version": status.transformer_engine_version,
        "converted_projections": 0,
    }


def test_fp8_is_an_explicit_runtime_opt_in_not_a_serialized_config_precision() -> None:
    config = _tiny_config()

    assert "precision" not in config.to_dict()
    assert "esmc_precision" not in config.to_dict()
    assert hasattr(ESMplusplusModel, "enable_fp8")


def test_fp8_capability_fails_closed_on_cpu() -> None:
    available, reason = _te_fp8_capability(torch.device("cpu"))

    assert available is False
    assert reason == "FP8 requires ESM++ on a CUDA device."


def test_enable_fp8_requires_eval_mode_before_capability_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ESMplusplusModel(_tiny_config()).train()
    monkeypatch.setattr(
        esmpp_module,
        "_te_fp8_capability",
        lambda _device: pytest.fail("training validation must run before capability probing"),
    )

    with pytest.raises(RuntimeError, match=r"inference-only; call eval\(\)"):
        model.enable_fp8()


def test_enable_fp8_rejects_non_bf16_canonical_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ESMplusplusModel(_tiny_config()).eval()
    monkeypatch.setattr(
        esmpp_module,
        "_te_fp8_capability",
        lambda device: (True, f"FP8 available on {device}"),
    )

    with pytest.raises(RuntimeError, match="requires canonical BF16 parameters"):
        model.enable_fp8()

    assert model.esmc_precision_status.enabled is False
    assert model._esmc_fp8_module_paths == ()


@pytest.mark.parametrize("layer_count", (30, 36, 80), ids=("300m", "600m", "6b"))
def test_fp8_conversion_replaces_one_attention_output_per_checkpoint_layer(
    layer_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = _SyntheticBackbone(layer_count)
    _install_fake_transformer_engine(monkeypatch)

    paths = _convert_esmc_attention_outputs_to_te(
        backbone,
        expected_projections=layer_count,
    )

    assert paths == tuple(
        f"transformer.blocks.{index}.attn.out_proj" for index in range(layer_count)
    )
    for block in backbone.transformer.blocks:
        assert isinstance(block.attn.out_proj, _FakeTELinear)
        assert type(block.ffn) is nn.Linear


def test_fp8_conversion_validates_the_complete_projection_set_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = _SyntheticBackbone(2)
    monkeypatch.setattr(
        esmpp_module,
        "_load_transformer_engine",
        lambda: pytest.fail("Transformer Engine must not load for an invalid module graph"),
    )

    with pytest.raises(RuntimeError, match="expected exactly 3.*found 2"):
        _convert_esmc_attention_outputs_to_te(backbone, expected_projections=3)

    assert all(
        type(block.attn.out_proj) is nn.Linear for block in backbone.transformer.blocks
    )


def test_enable_fp8_records_conversion_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ESMplusplusModel(_tiny_config()).to(dtype=torch.bfloat16).eval()
    _install_fake_transformer_engine(monkeypatch)
    monkeypatch.setattr(
        esmpp_module,
        "_te_fp8_capability",
        lambda device: (True, f"FP8 available on {device}"),
    )

    status = model.enable_fp8()
    second_status = model.enable_fp8()

    assert status is second_status
    assert status.enabled is True
    assert status.device == "cpu"
    assert status.converted_projections == 2
    assert model._esmc_fp8 is True
    assert model._esmc_fp8_module_paths == (
        "transformer.blocks.0.attn.out_proj",
        "transformer.blocks.1.attn.out_proj",
    )


def test_fp8_context_requires_inference_mode() -> None:
    with pytest.raises(RuntimeError, match="inference-only"):
        with _esmplusplus_fp8_context(True, torch.device("cpu")):
            pass


def test_fp8_context_nests_bf16_and_transformer_engine_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    @contextmanager
    def fake_torch_autocast(*, device_type: str, dtype: torch.dtype):
        events.append(("torch_enter", device_type, dtype))
        yield
        events.append("torch_exit")

    @contextmanager
    def fake_te_autocast(*, enabled: bool, recipe: object):
        events.append(("te_enter", enabled, recipe))
        yield
        events.append("te_exit")

    recipe_instance = object()
    fake_recipe = SimpleNamespace(
        Format=SimpleNamespace(HYBRID="hybrid"),
        Float8CurrentScaling=lambda **kwargs: (
            events.append(("recipe", kwargs)) or recipe_instance
        ),
    )
    monkeypatch.setattr(esmpp_module.torch, "autocast", fake_torch_autocast)
    monkeypatch.setattr(
        esmpp_module,
        "_load_transformer_engine",
        lambda: (SimpleNamespace(autocast=fake_te_autocast), fake_recipe),
    )

    with torch.inference_mode():
        with _esmplusplus_fp8_context(True, torch.device("cuda")):
            events.append("body")

    assert events == [
        ("recipe", {"use_power_2_scales": False, "fp8_format": "hybrid"}),
        ("torch_enter", "cuda", torch.bfloat16),
        ("te_enter", True, recipe_instance),
        "body",
        "te_exit",
        "torch_exit",
    ]


@pytest.mark.parametrize("model_class", (ESMplusplusModel, ESMplusplusForMaskedLM))
def test_fp8_forward_pads_to_16_and_trims_public_outputs(
    model_class: type[ESMplusplusModel] | type[ESMplusplusForMaskedLM],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = model_class(_tiny_config(num_hidden_layers=1)).eval()
    model._esmc_fp8 = True
    seen_lengths: list[int] = []
    original_forward = model.transformer.forward

    def recording_forward(*args, **kwargs):
        x = kwargs["x"]
        seen_lengths.append(x.shape[1])
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.transformer, "forward", recording_forward)

    @contextmanager
    def recording_context(enabled: bool, device: torch.device):
        assert enabled is True
        assert device.type == "cpu"
        yield

    monkeypatch.setattr(esmpp_module, "_esmplusplus_fp8_context", recording_context)
    kwargs = {"compute_logits": True} if model_class is ESMplusplusForMaskedLM else {}

    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[0, 3, 4, 2, 1]], dtype=torch.long),
            output_hidden_states=True,
            **kwargs,
        )

    assert seen_lengths == [16]
    assert output.last_hidden_state.shape == (1, 5, 16)
    assert output.hidden_states is not None
    assert all(state.shape == (1, 5, 16) for state in output.hidden_states)
    if model_class is ESMplusplusForMaskedLM:
        assert output.logits is not None
        assert output.logits.shape == (1, 5, 16)


class _SyntheticSAE(nn.Module):
    layer = 0

    def __init__(self) -> None:
        super().__init__()
        self.params = SimpleNamespace(d_model=16)
        self.register_buffer("idf", torch.ones(4))
        self.register_buffer("max", torch.ones(4))
        self.seen_hidden_shape: tuple[int, ...] | None = None
        self.seen_mask: torch.Tensor | None = None

    def get_sae_output(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> SimpleNamespace:
        self.seen_hidden_shape = tuple(hidden_states.shape)
        self.seen_mask = token_mask.detach().clone()
        return SimpleNamespace(feature_magnitudes=hidden_states[token_mask, :4])


def test_fp8_padding_is_excluded_from_sae_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ESMplusplusModel(_tiny_config(num_hidden_layers=1)).eval()
    sae = _SyntheticSAE()
    model.add_sae_models([sae])
    model._esmc_fp8 = True

    @contextmanager
    def passthrough_context(_enabled: bool, _device: torch.device):
        yield

    monkeypatch.setattr(esmpp_module, "_esmplusplus_fp8_context", passthrough_context)

    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[0, 3, 4, 2, 1]], dtype=torch.long),
            compute_sae=True,
        )

    assert sae.seen_hidden_shape == (1, 16, 16)
    assert sae.seen_mask is not None
    assert sae.seen_mask.sum().item() == 4
    assert output.sae_outputs is not None
    features = output.sae_outputs["layer0"]
    assert features.is_sparse
    assert features.shape == (4, 4)
    assert output.last_hidden_state.shape == (1, 5, 16)
