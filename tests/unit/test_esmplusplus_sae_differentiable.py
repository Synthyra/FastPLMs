"""Contracts for the attached, dense sparse-autoencoder path.

The default SAE path detaches its output and returns a sparse tensor, which is correct for
interpretation and matches the pinned Biohub implementation. Gradient-based sequence design needs
the same numbers on the tape and in a dense layout, so ``differentiable_sae=True`` changes the
tape and the layout and nothing else. These tests hold it to that: identical values, gradient
reaching the input embeddings, and the default behaviour untouched.

They also cover the second half of that path. Sparse autoencoders can now be requested from an
``inputs_embeds`` forward, where the token mask cannot be recovered from token identities and so
must be supplied by the caller, along with the mask-token precondition the caller then owns.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    ESMplusplusModel,
)

from .test_esmplusplus_sae import _SyntheticSAELayer, _config, _input_ids


def _model(model_class: type[nn.Module] = ESMplusplusModel) -> nn.Module:
    torch.manual_seed(7)
    return model_class(_config()).eval()


def test_differentiable_sae_matches_the_detached_path_exactly() -> None:
    input_ids = _input_ids()

    detached_model = _model()
    detached_model.add_sae_models([_SyntheticSAELayer(layer=0)])
    detached = detached_model(input_ids=input_ids).sae_outputs["layer0"]

    attached_model = _model()
    attached_model.add_sae_models([_SyntheticSAELayer(layer=0)])
    attached = attached_model(input_ids=input_ids, differentiable_sae=True).sae_outputs["layer0"]

    assert attached.layout == torch.strided
    torch.testing.assert_close(attached, detached.to_dense(), rtol=0.0, atol=0.0)


def test_differentiable_sae_keeps_the_graph_back_to_the_inputs() -> None:
    model = _model()
    model.add_sae_models([_SyntheticSAELayer(layer=0)])
    input_ids = _input_ids()
    inputs_embeds = model.embed(input_ids).detach().clone().requires_grad_(True)

    features = model(
        inputs_embeds=inputs_embeds,
        attention_mask=input_ids.ne(model.config.pad_token_id),
        differentiable_sae=True,
    ).sae_outputs["layer0"]
    features.sum().backward()

    assert features.requires_grad
    assert inputs_embeds.grad is not None
    assert torch.isfinite(inputs_embeds.grad).all()
    assert float(inputs_embeds.grad.abs().sum()) > 0.0


def test_the_default_path_remains_detached_and_sparse() -> None:
    model = _model()
    model.add_sae_models([_SyntheticSAELayer(layer=0)])

    features = model(input_ids=_input_ids()).sae_outputs["layer0"]

    assert features.layout == torch.sparse_coo
    assert not features.requires_grad


def test_sae_from_inputs_embeds_requires_an_explicit_mask() -> None:
    model = _model()
    model.add_sae_models([_SyntheticSAELayer(layer=0)])
    inputs_embeds = model.embed(_input_ids())

    with pytest.raises(ValueError, match="explicit attention_mask or sequence_id"):
        model(inputs_embeds=inputs_embeds)


def test_a_supplied_sequence_id_also_satisfies_the_mask_requirement() -> None:
    model = _model()
    model.add_sae_models([_SyntheticSAELayer(layer=0)])
    input_ids = _input_ids()
    sequence_id = torch.where(input_ids.ne(model.config.pad_token_id), 0, -1)

    output = model(inputs_embeds=model.embed(input_ids), sequence_id=sequence_id)

    assert output.sae_outputs is not None
    assert tuple(output.sae_outputs["layer0"].shape)[0] == int((sequence_id >= 0).sum())


def test_mask_tokens_are_still_rejected_when_token_ids_are_available() -> None:
    model = _model()
    model.add_sae_models([_SyntheticSAELayer(layer=0)])
    masked = _input_ids().clone()
    masked[0, 1] = model.config.mask_token_id

    with pytest.raises(ValueError, match="must not contain mask tokens"):
        model(input_ids=masked)


def test_multiple_layers_stay_differentiable_in_one_forward() -> None:
    model = _model()
    model.add_sae_models([_SyntheticSAELayer(layer=0), _SyntheticSAELayer(layer=2)])
    input_ids = _input_ids()
    inputs_embeds = model.embed(input_ids).detach().clone().requires_grad_(True)

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=input_ids.ne(model.config.pad_token_id),
        differentiable_sae=True,
    ).sae_outputs

    assert set(outputs) == {"layer0", "layer2"}
    sum(features.sum() for features in outputs.values()).backward()
    assert inputs_embeds.grad is not None


def test_the_masked_language_model_head_threads_the_flag() -> None:
    model = _model(ESMplusplusForMaskedLM)
    model.add_sae_models([_SyntheticSAELayer(layer=0)])

    output = model(input_ids=_input_ids(), compute_logits=False, differentiable_sae=True)

    assert output.sae_outputs["layer0"].layout == torch.strided


def test_normalization_composes_with_the_differentiable_path() -> None:
    input_ids = _input_ids()

    detached_model = _model()
    detached_model.add_sae_models([_SyntheticSAELayer(layer=0)])
    detached = detached_model(input_ids=input_ids, normalize_sae=True).sae_outputs["layer0"]

    attached_model = _model()
    attached_model.add_sae_models([_SyntheticSAELayer(layer=0)])
    attached = attached_model(
        input_ids=input_ids, normalize_sae=True, differentiable_sae=True
    ).sae_outputs["layer0"]

    torch.testing.assert_close(attached, detached.to_dense(), rtol=0.0, atol=0.0)


def test_config_shape_assumptions_hold() -> None:
    # Guards the shared fixture: these tests read hidden states by layer index, so a change to the
    # layer count in the shared config would silently narrow their coverage.
    config: ESMplusplusConfig = _config()

    assert config.num_hidden_layers == 2
    assert config.hidden_size == 4
