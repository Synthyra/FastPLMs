"""Fast, injected-core contracts for advertised structure AutoModels."""

from __future__ import annotations

import warnings
import pytest
import torch
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from torch import Tensor, nn
from transformers.models.esm.modeling_esmfold import EsmForProteinFolding

from fastplms.models.boltz import modeling_boltz2
from fastplms.models.boltz.modeling_boltz2 import (
    Boltz2Config,
    Boltz2Model,
    Boltz2ModelOutput,
)
from fastplms.models.esmfold.modeling_fast_esmfold import (
    FastEsmFoldConfig,
    FastEsmForProteinFolding,
    FastEsmForProteinFoldingOutput,
)
from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.modeling_esmfold2 import (
    ESMFold2Model,
    ESMFold2Output,
)
from fastplms.models.esmfold2.modeling_esmfold2_common import NUM_RES_TYPES
from fastplms.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)
from fastplms.models.esmfold2.protein_utils import prepare_protein_features
from fastplms.models.esmfold2.reproducibility import seed_context


def _assert_nested_close(left: Any, right: Any) -> None:
    if torch.is_tensor(left) and torch.is_tensor(right):
        torch.testing.assert_close(left, right)
        return
    if isinstance(left, tuple) and isinstance(right, tuple):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_close(left_item, right_item)
        return
    assert left == right


def _assert_tuple_matches_output(
    tuple_output: tuple[Any, ...],
    structured_output: Iterable[Any],
) -> None:
    expected = tuple(structured_output)
    assert len(tuple_output) == len(expected)
    for actual, reference in zip(tuple_output, expected, strict=True):
        _assert_nested_close(actual, reference)


class _TinyBoltzCore(nn.Module):
    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.0, width))  # (d=width,)

    def forward(self, feats: dict[str, Tensor], **_kwargs: Any) -> dict[str, Tensor]:
        # feats["signal"]: (b, l, d)
        signal = feats["signal"] * self.weight  # (b, l, d)
        pair = signal[:, :, None, :] + signal[:, None, :, :]  # (b, l, l, d)
        return {
            "pdistogram": pair.unsqueeze(-1),  # (b, l, l, d, 1)
            "s": signal,  # (b, l, d)
            "z": pair,  # (b, l, l, d)
            "sample_atom_coords": signal[..., :1].expand(-1, -1, 3),  # (b, l, xyz=3)
        }


def test_boltz_public_forward_honors_output_controls_backward_and_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(modeling_boltz2, "Boltz2InferenceCore", _TinyBoltzCore)
    config = Boltz2Config(core_kwargs={"width": 3})
    model = Boltz2Model(config)
    features = {
        "signal": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)  # (b=1, l=2, d=3)
    }

    structured = model(
        feats=features,
        output_hidden_states=True,
        return_dict=True,
    )
    tuple_output = model(
        feats=features,
        output_hidden_states=True,
        return_dict=False,
    )

    assert isinstance(structured, Boltz2ModelOutput)
    assert structured.last_hidden_state is structured.s
    assert structured.hidden_states is not None
    assert structured.hidden_states[0] is structured.s
    assert structured.hidden_states[1] is structured.z
    _assert_tuple_matches_output(tuple_output, structured.to_tuple())
    assert structured.last_hidden_state is not None
    structured.last_hidden_state.square().mean().backward()
    assert model.core.weight.grad is not None
    assert torch.isfinite(model.core.weight.grad).all()
    with pytest.raises(NotImplementedError, match="output_attentions=True"):
        model(feats=features, output_attentions=True)
    with pytest.raises(TypeError):
        model(feats=features, silently_ignored=True)

    model.save_pretrained(tmp_path)
    reloaded = Boltz2Model.from_pretrained(tmp_path, local_files_only=True)
    reloaded_output = reloaded(feats=features, return_dict=True)
    torch.testing.assert_close(
        reloaded_output.last_hidden_state,
        structured.last_hidden_state,
    )


def test_fast_esmfold_public_forward_honors_output_controls_and_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def native_forward(
        self: FastEsmForProteinFolding,
        input_ids: Tensor,
        **_kwargs: Any,
    ) -> dict[str, Tensor]:
        # input_ids: (b, l)
        state = self.contract_weight.expand(input_ids.shape[0], input_ids.shape[1], 2)
        return {"s_s": state, "plddt": state[..., :1]}  # (b, l, d=2); (b, l, 1)

    monkeypatch.setattr(EsmForProteinFolding, "forward", native_forward)
    model = FastEsmForProteinFolding.__new__(FastEsmForProteinFolding)
    nn.Module.__init__(model)
    model.register_parameter(
        "contract_weight",
        nn.Parameter(torch.ones(1, 1, 2)),  # (1, 1, d=2)
    )
    input_ids = torch.zeros((1, 2), dtype=torch.int64)  # (b=1, l=2)

    structured = model(
        input_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    tuple_output = model(
        input_ids,
        output_hidden_states=True,
        return_dict=False,
    )

    assert isinstance(structured, FastEsmForProteinFoldingOutput)
    assert structured.last_hidden_state is structured.s_s
    assert structured.hidden_states is not None
    assert structured.hidden_states[0] is structured.s_s
    assert torch.equal(structured.plddt, torch.full((1, 2, 1), 100.0))
    _assert_tuple_matches_output(tuple_output, structured.to_tuple())
    assert structured.last_hidden_state is not None
    structured.last_hidden_state.square().mean().backward()
    assert model.contract_weight.grad is not None
    attention_output = model(input_ids, output_attentions=True)
    assert attention_output.attentions == ()
    with pytest.raises(TypeError):
        model(input_ids, silently_ignored=True)


def _tiny_fast_esmfold_config(
    *,
    bypass_lm: bool,
    attn_backend: str = "sdpa",
) -> FastEsmFoldConfig:
    return FastEsmFoldConfig(
        vocab_size=33,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
        pad_token_id=1,
        mask_token_id=32,
        position_embedding_type="rotary",
        is_folding_model=True,
        attn_backend=attn_backend,
        esmfold_config={
            "fp16_esm": False,
            "bypass_lm": bypass_lm,
            "lddt_head_hid_dim": 4,
            "trunk": {
                "num_blocks": 1,
                "sequence_state_dim": 8,
                "pairwise_state_dim": 4,
                "sequence_head_width": 4,
                "pairwise_head_width": 2,
                "position_bins": 4,
                "max_recycles": 1,
                "chunk_size": None,
                "structure_module": {
                    "sequence_dim": 8,
                    "pairwise_dim": 4,
                    "ipa_dim": 2,
                    "resnet_dim": 4,
                    "num_heads_ipa": 2,
                    "num_qk_points": 1,
                    "num_v_points": 1,
                    "dropout_rate": 0.0,
                    "num_blocks": 1,
                    "num_transition_layers": 1,
                    "num_resnet_blocks": 1,
                    "num_angles": 7,
                },
            },
        },
    )


def test_fast_esmfold_output_attentions_uses_masked_per_call_eager_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def folding_stub(
        self: FastEsmForProteinFolding,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **_kwargs: Any,
    ) -> dict[str, Tensor]:
        # input_ids: (b, l); attention_mask: (b, l) or None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # (b, l)
        esmaa = self.af2_idx_to_esm_idx(input_ids, attention_mask)  # (b, l)
        representations = self.compute_language_model_representations(
            esmaa
        )  # (b, l, n_layers, d)
        state = representations[:, :, -1, :]  # (b, l, d)
        return {
            "s_s": state,  # (b, l, d)
            "plddt": torch.ones((*state.shape[:2], 1), device=state.device),  # (b, l, 1)
        }

    monkeypatch.setattr(EsmForProteinFolding, "forward", folding_stub)
    model = FastEsmForProteinFolding(
        _tiny_fast_esmfold_config(bypass_lm=False, attn_backend="sdpa")
    ).eval()
    input_ids = torch.tensor(((0, 1, 2), (3, 4, 0)), dtype=torch.int64)  # (b=2, l=3)
    attention_mask = torch.tensor(  # (b=2, l=3)
        ((1, 1, 1), (1, 1, 0)), dtype=torch.int64
    )
    configured_encoder_backend = model.esm.encoder.attention_backend
    attention_module = model.esm.encoder.layer[0].attention.self
    configured_layer_backend = attention_module.attn_backend

    with pytest.warns(
        RuntimeWarning,
        match=r"output_attentions=True.*requested 'sdpa'.*using 'eager'.*call only",
    ) as captured:
        output = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    assert len(captured) == 1
    assert output.attentions is not None and len(output.attentions) == 1
    attention = output.attentions[0]  # (b=2, h=2, l=3, l=3)
    assert attention.shape == (2, 2, 3, 3)
    torch.testing.assert_close(
        attention[1, :, :, 2],
        torch.zeros_like(attention[1, :, :, 2]),
        rtol=0.0,
        atol=0.0,
    )
    assert model.config.attn_backend == "sdpa"
    assert model.esm.encoder.attention_backend == configured_encoder_backend
    assert attention_module.attn_backend == configured_layer_backend

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        subsequent = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
        )
    assert subsequent.attentions is None
    assert attention_module.attn_backend == configured_layer_backend


def test_fast_esmfold_tiny_model_saves_and_reloads_exact_state(tmp_path: Path) -> None:
    config = _tiny_fast_esmfold_config(bypass_lm=True)
    source = FastEsmForProteinFolding(config).eval()
    source.save_pretrained(tmp_path, safe_serialization=True)

    restored = FastEsmForProteinFolding.from_pretrained(
        tmp_path,
        local_files_only=True,
    ).eval()
    assert set(restored.state_dict()) == set(source.state_dict())
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name],
            tensor,
            rtol=0.0,
            atol=0.0,
        )


def _tiny_esmfold2_config(model_type: str) -> ESMFold2Config:
    atom_token_width = 8
    input_feature_width = atom_token_width // 2 + 2 * NUM_RES_TYPES + 1
    return ESMFold2Config(
        type=model_type,
        d_single=8,
        d_pair=8,
        num_loops=0,
        num_diffusion_samples=1,
        lm_d_model=8,
        lm_num_layers=1,
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
        msa_encoder={
            "enabled": True,
            "d_msa": 8,
            "d_hidden": 4,
            "n_layers": 0,
            "n_heads_msa": 2,
            "msa_head_width": 4,
        },
        msa_conditioning=True,
        lm_encoder={"enabled": False, "n_layers": 0},
        parcae={"enabled": True, "min_steps": 1, "max_steps": 1, "coda_n_layers": 0},
    )


class _TinyStructureHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observed: dict[str, Any] = {}

    def sample(self, **kwargs: Any) -> dict[str, Tensor]:
        self.observed = {
            name: kwargs[name]
            for name in (
                "noise_scale",
                "step_scale",
                "max_inference_sigma",
                "denoising_early_exit_rmsd",
            )
        }
        coords = kwargs["ref_pos"].float()  # (b, a, xyz=3)
        multiplicity = int(kwargs["num_diffusion_samples"])
        return {
            "sample_atom_coords": coords.repeat_interleave(
                multiplicity, dim=0
            )  # (b * multiplicity, a, xyz=3)
        }


class _TinyConfidenceHead(nn.Module):
    def forward(
        self,
        z: Tensor,
        x_pred: Tensor,
        num_diffusion_samples: int,
        **_kwargs: Any,
    ) -> dict[str, Tensor]:
        # z: (b, l, l, d_pair); x_pred: (b * num_diffusion_samples, a, xyz=3)
        _batch_size, sequence_length = z.shape[:2]
        score = z.float().mean(dim=(1, 2, 3)).repeat_interleave(
            num_diffusion_samples
        )  # (b * num_diffusion_samples,)
        return {
            "plddt": score[:, None].expand(-1, sequence_length),  # (b * samples, l)
            "complex_plddt": score,  # (b * samples,)
            "ptm": score,  # (b * samples,)
            "iptm": score,  # (b * samples,)
            "pae": z.new_zeros(
                (x_pred.shape[0], sequence_length, sequence_length)
            ),  # (b * samples, l, l)
        }


@pytest.mark.parametrize(
    ("model_class", "model_type"),
    (
        (ESMFold2Model, "release"),
        (ESMFold2ExperimentalModel, "experimental"),
    ),
)
def test_esmfold2_public_forward_honors_output_controls_and_sampler_overrides(
    model_class: type[ESMFold2Model] | type[ESMFold2ExperimentalModel],
    model_type: str,
) -> None:
    model = model_class(_tiny_esmfold2_config(model_type)).eval()
    structure_head = _TinyStructureHead()
    model.structure_head = structure_head
    if model_type == "release":
        model.confidence_head = _TinyConfidenceHead()
    features = prepare_protein_features("AC")  # batched tensors for b=1, l=2 residues
    batch_size, sequence_length = features["res_type"].shape
    features.update(
        {
            "pocket_feature": torch.zeros_like(features["res_type"]),
            "gt_coords": torch.zeros_like(features["ref_pos"]),
            "is_resolved": torch.zeros_like(features["atom_attention_mask"]),
            "frames_idx": torch.zeros(
                batch_size,
                sequence_length,
                3,
                dtype=torch.long,
            ),  # (b, l, frame=3)
            "disto_cond": torch.zeros(
                batch_size,
                sequence_length,
                sequence_length,
                dtype=torch.long,
            ),  # (b, l, l)
            "disto_cond_mask": torch.zeros(
                batch_size,
                sequence_length,
                sequence_length,
                dtype=torch.bool,
            ),  # (b, l, l)
        }
    )
    common_kwargs = {
        "num_loops": 0,
        "num_sampling_steps": 1,
        "num_diffusion_samples": 1,
        "noise_scale": 0.25,
        "step_scale": 1.5,
        "max_inference_sigma": 32.0,
        "early_exit": True,
    }
    if model_type == "experimental":
        common_kwargs.update({"calculate_confidence": False, "seed": 7})
    else:
        common_kwargs.update(
            {"msa_column_mask_rate": 0.0, "msa_subsample_at_inference": False}
        )

    with seed_context(31):
        structured = model(
            **features,
            **common_kwargs,
            output_hidden_states=True,
            return_dict=True,
        )
    with seed_context(31):
        tuple_output = model(
            **features,
            **common_kwargs,
            output_hidden_states=True,
            return_dict=False,
        )

    assert isinstance(structured, ESMFold2Output)
    assert structured.last_hidden_state is not None
    assert structured.hidden_states is not None
    assert structured.hidden_states[-1] is structured.last_hidden_state
    assert structured.sample_atom_coords is not None
    _assert_tuple_matches_output(tuple_output, structured.to_tuple())
    assert structure_head.observed == {
        "noise_scale": 0.25,
        "step_scale": 1.5,
        "max_inference_sigma": 32.0,
        "denoising_early_exit_rmsd": 0.10,
    }
    with pytest.raises(NotImplementedError, match="output_attentions=True"):
        model(**features, output_attentions=True)
    with pytest.raises(TypeError):
        model(**features, silently_ignored=True)

    model.zero_grad(set_to_none=True)
    if model_type == "experimental":
        res_type_soft = torch.nn.functional.one_hot(  # (b=1, l=2, c=NUM_RES_TYPES)
            features["res_type"].long(),
            num_classes=NUM_RES_TYPES,
        ).float()
        res_type_soft.requires_grad_(True)
        differentiable = model(
            **features,
            **common_kwargs,
            res_type_soft=res_type_soft,
            return_dict=True,
        )
    else:
        differentiable = model(
            **features,
            **common_kwargs,
            return_dict=True,
        )
    assert differentiable.distogram_logits is not None
    differentiable.distogram_logits.square().mean().backward()
    if model_type == "experimental":
        assert res_type_soft.grad is not None
        assert torch.isfinite(res_type_soft.grad).all()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
