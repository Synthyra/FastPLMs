"""CPU contracts for ESMFold sequence and residue task wrappers."""

from __future__ import annotations

import pytest
import torch

from fastplms.models.esmfold.modeling_fast_esmfold import (
    FastEsmFoldConfig,
    FastEsmForSequenceClassification,
    FastEsmForTokenClassification,
)


def _tiny_config(
    *,
    num_labels: int,
    classifier_train_scope: str = "probe",
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
        attn_backend="eager",
        num_labels=num_labels,
        classifier_train_scope=classifier_train_scope,
        classifier_probe_hidden_size=512,
        classifier_probe_num_heads=4,
        classifier_probe_dropout=0.0,
        classifier_hidden_size=16,
        classifier_dropout=0.0,
        esmfold_config={
            "fp16_esm": False,
            "bypass_lm": True,
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


def test_prepare_classifier_inputs_is_residue_only_and_rejects_complexes() -> None:
    prepared = FastEsmForSequenceClassification.prepare_classifier_inputs(
        ["ACDX", "WY"]
    )

    assert prepared["input_ids"].shape == (2, 4)
    assert prepared["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert prepared["input_ids"][0].tolist() == [0, 4, 3, 20]

    for invalid in ("", "AC:DE", "AC-DE", "AC U"):
        with pytest.raises(ValueError):
            FastEsmForSequenceClassification.prepare_classifier_inputs(invalid)


@pytest.mark.parametrize(
    "wrapper",
    [FastEsmForSequenceClassification, FastEsmForTokenClassification],
)
@pytest.mark.parametrize("scope", ["probe", "projection"])
def test_classifier_train_scope_is_exact(wrapper: type, scope: str) -> None:
    model = wrapper(_tiny_config(num_labels=2, classifier_train_scope=scope))
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    parameter_names = {name for name, _parameter in model.named_parameters()}

    assert {name for name in trainable if name.startswith("classifier.")} == {
        name for name in parameter_names if name.startswith("classifier.")
    }
    projection = {
        name
        for name in trainable
        if name == "esm_s_combine" or name.startswith("esm_s_mlp.")
    }
    if scope == "projection":
        assert projection == {
            name
            for name in parameter_names
            if name == "esm_s_combine" or name.startswith("esm_s_mlp.")
        }
    else:
        assert not projection
    assert not any(name.startswith("esm.") for name in trainable)
    assert not any(name.startswith("trunk.") for name in trainable)


def test_sequence_classifier_bypasses_folding_trunk() -> None:
    model = FastEsmForSequenceClassification(_tiny_config(num_labels=1)).eval()
    model.trunk.forward = lambda *_args, **_kwargs: pytest.fail("folding trunk ran")
    prepared = model.prepare_classifier_inputs(["ACD", "WY"])

    output = model(**prepared, labels=torch.tensor([0.5, -1.0]))

    assert output.logits.shape == (2, 1)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert model.config.problem_type == "regression"


@pytest.mark.parametrize("num_labels", [1, 3])
def test_token_classifier_masks_ignored_regression_and_multilabel_targets(
    num_labels: int,
) -> None:
    model = FastEsmForTokenClassification(_tiny_config(num_labels=num_labels)).eval()
    model.trunk.forward = lambda *_args, **_kwargs: pytest.fail("folding trunk ran")
    prepared = model.prepare_classifier_inputs(["ACD", "WY"])
    batch_size, residue_count = prepared["input_ids"].shape
    if num_labels == 1:
        labels = torch.zeros(batch_size, residue_count)
        labels[0, 1] = -100
    else:
        labels = torch.zeros(batch_size, residue_count, num_labels)
        labels[0, 1, :] = -100

    output = model(**prepared, labels=labels)

    assert output.logits.shape == (batch_size, residue_count, num_labels)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert model.config.problem_type in {"regression", "multi_label_classification"}


def test_classifier_rejects_all_padding_rows() -> None:
    model = FastEsmForTokenClassification(_tiny_config(num_labels=2)).eval()
    with pytest.raises(ValueError, match="at least one residue"):
        model(
            torch.zeros((1, 3), dtype=torch.int64),
            attention_mask=torch.zeros((1, 3), dtype=torch.int64),
        )
