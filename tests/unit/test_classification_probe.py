"""Contracts for the shared Protify-style classification probe."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

import fastplms.models.classification_probe as probe_module
from fastplms.models.classification_probe import (
    ProteinTransformerProbe,
    SequenceClassificationProbe,
    SwiGLU,
    TokenClassificationProbe,
    token_classification_loss,
)


def _config(**overrides) -> SimpleNamespace:
    values = {
        "attn_backend": "eager",
        "classifier_hidden_size": 32,
        "classifier_pooling_types": ["mean"],
        "num_labels": 3,
        "output_attentions": False,
        "output_hidden_states": False,
        "problem_type": None,
        "use_return_dict": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_probe_has_exact_protify_aligned_architecture() -> None:
    config = _config()
    del config.classifier_hidden_size
    probe = SequenceClassificationProbe(config, input_size=24)

    assert probe.transformer.hidden_size == 512
    assert probe.transformer.attention.num_heads == 4
    assert probe.transformer.attention.head_size == 128
    assert probe.transformer.attention.dropout == 0.1
    swiglu_layers = [
        module for module in probe.transformer.modules() if isinstance(module, SwiGLU)
    ]
    assert len(swiglu_layers) == 1
    assert probe.transformer.feed_forward[0].out_features == 2 * 1536
    assert probe.classifier[1].out_features == 4096
    assert probe.classifier[3].p == 0.2
    assert probe.classifier[4].out_features == 256
    assert probe.classifier[6].p == 0.2
    assert probe.classifier[7].out_features == 3
    assert all(
        layer.bias is None
        for layer in probe.modules()
        if isinstance(layer, torch.nn.Linear)
    )


def test_token_classifier_has_protify_extra_projection() -> None:
    probe = TokenClassificationProbe(_config(), input_size=8)
    linear_layers = [
        layer for layer in probe.classifier if isinstance(layer, torch.nn.Linear)
    ]

    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (512, 32),
        (32, 256),
        (256, 256),
        (256, 3),
    ]


def test_classifier_bias_can_be_enabled_explicitly() -> None:
    probe = SequenceClassificationProbe(
        _config(classifier_use_bias=True),
        input_size=8,
    )

    assert all(
        layer.bias is not None
        for layer in probe.modules()
        if isinstance(layer, torch.nn.Linear)
    )


def test_sequence_probe_returns_hf_output_and_standard_tuple() -> None:
    probe = SequenceClassificationProbe(_config(), input_size=8).eval()
    embeddings = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    labels = torch.tensor([1, 2])

    output = probe(
        embeddings,
        mask,
        labels,
        output_attentions=True,
        output_hidden_states=True,
    )
    tuple_output = probe(embeddings, mask, labels, return_dict=False)

    assert isinstance(output, SequenceClassifierOutput)
    assert output.loss.ndim == 0
    assert output.logits.shape == (2, 3)
    assert output.hidden_states is not None and len(output.hidden_states) == 1
    assert output.attentions is not None and output.attentions[0].shape == (2, 4, 4, 4)
    assert tuple_output[0].ndim == 0
    assert tuple_output[1].shape == (2, 3)


def test_token_probe_returns_one_prediction_per_input_residue() -> None:
    probe = TokenClassificationProbe(_config(), input_size=8).eval()
    embeddings = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    labels = torch.tensor([[0, 1, 2, 0, 1], [2, 1, 0, -100, -100]])

    output = probe(embeddings, mask, labels)

    assert isinstance(output, TokenClassifierOutput)
    assert output.loss.ndim == 0
    assert output.logits.shape == (2, 5, 3)


@pytest.mark.parametrize("pooling", [["cls"], ["parti"], ["mean", "cls"]])
def test_sequence_probe_rejects_non_residue_pooling(pooling: list[str]) -> None:
    with pytest.raises(ValueError, match="residue-only"):
        SequenceClassificationProbe(
            _config(classifier_pooling_types=pooling),
            input_size=8,
        )


def test_probe_rejects_non_contract_dimensions() -> None:
    with pytest.raises(ValueError, match="512-wide"):
        ProteinTransformerProbe(
            _config(classifier_probe_hidden_size=256),
            input_size=8,
        )


def test_sdpa_executes_named_backend_and_fails_closed_for_attentions(monkeypatch) -> None:
    observed = []

    def tracked_sdpa(query, key, value, **kwargs):
        observed.append(kwargs)
        return torch.zeros_like(query)

    monkeypatch.setattr(probe_module.F, "scaled_dot_product_attention", tracked_sdpa)
    probe = ProteinTransformerProbe(_config(attn_backend="sdpa"), input_size=8).eval()
    embeddings = torch.randn(2, 3, 8)

    output = probe(embeddings)

    assert output.last_hidden_state.shape == (2, 3, 512)
    assert len(observed) == 1
    with pytest.raises(ValueError, match="select 'eager' explicitly"):
        probe(embeddings, output_attentions=True)


def test_unsupported_attention_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="support only"):
        ProteinTransformerProbe(_config(attn_backend="flash_attention_2"), input_size=8)


def test_token_regression_excludes_ignored_elements() -> None:
    logits = torch.tensor([[[1.0], [2.0], [9.0]]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, -100.0]])

    loss = token_classification_loss(
        logits,
        labels,
        problem_type="regression",
        num_labels=1,
    )
    loss.backward()

    assert torch.equal(loss.detach(), torch.tensor(2.0))
    assert logits.grad is not None
    assert logits.grad[0, 2, 0] == 0


def test_token_multilabel_excludes_ignored_elements() -> None:
    logits = torch.zeros(1, 2, 3, requires_grad=True)
    labels = torch.tensor([[[1.0, 0.0, -100.0], [-100.0, -100.0, -100.0]]])

    loss = token_classification_loss(
        logits,
        labels,
        problem_type="multi_label_classification",
        num_labels=3,
    )
    loss.backward()

    assert torch.allclose(loss.detach(), torch.log(torch.tensor(2.0)))
    assert logits.grad is not None
    assert torch.equal(logits.grad[0, 1], torch.zeros(3))
    assert logits.grad[0, 0, 2] == 0


def test_all_ignored_token_regression_remains_differentiable() -> None:
    logits = torch.randn(1, 2, 1, requires_grad=True)
    labels = torch.full((1, 2), -100.0)

    loss = token_classification_loss(
        logits,
        labels,
        problem_type="regression",
        num_labels=1,
    )
    loss.backward()

    assert loss == 0
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))
