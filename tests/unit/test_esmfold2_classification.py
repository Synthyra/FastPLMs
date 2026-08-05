"""Focused CPU contracts for the ESMFold2 task wrappers."""

from __future__ import annotations

from types import MethodType

import pytest
import torch
import torch.nn as nn

from fastplms.models.classification_probe import (
    SequenceClassificationProbe,
    TokenClassificationProbe,
)
from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from fastplms.models.esmfold2.modeling_esmfold2_classification import (
    ESMFold2ExperimentalForSequenceClassification,
    ESMFold2ExperimentalForTokenClassification,
    ESMFold2ForSequenceClassification,
    ESMFold2ForTokenClassification,
)
from fastplms.models.esmfold2.modeling_esmfold2_common import LanguageModelShim
from fastplms.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)


class _ForbiddenTrunk(nn.Module):
    def forward(self, *_args, **_kwargs):
        raise AssertionError("Classification must bypass the ESMFold2 folding trunk.")


def _tiny_config(*, experimental: bool = False, scope: str = "probe") -> ESMFold2Config:
    return ESMFold2Config(
        type="experimental" if experimental else "release",
        d_pair=4,
        lm_d_model=4,
        lm_num_layers=80,
        num_labels=3,
        classifier_train_scope=scope,
        classifier_probe_hidden_size=512,
        classifier_probe_num_heads=4,
        classifier_probe_dropout=0.0,
        classifier_hidden_size=12,
        classifier_dropout=0.0,
        classifier_pooling_types=["mean"],
        attn_implementation="eager",
    )


def _replace_base_initializers(monkeypatch: pytest.MonkeyPatch) -> None:
    def initialize(model, config: ESMFold2Config) -> None:
        nn.Module.__init__(model)
        model.config = config
        model.anchor = nn.Parameter(torch.zeros(()))
        model.language_model = LanguageModelShim(
            d_z=config.d_pair,
            d_model=config.lm_d_model,
            num_layers=config.lm_num_layers,
        )
        model._esmc = nn.Linear(1, 1)
        model.folding_trunk = _ForbiddenTrunk()

    monkeypatch.setattr(ESMFold2Model, "__init__", initialize)
    monkeypatch.setattr(ESMFold2ExperimentalModel, "__init__", initialize)


def _attach_tiny_esmc_output(model: nn.Module) -> None:
    def compute(
        self,
        input_ids: torch.Tensor,
        asym_id: torch.Tensor,
        residue_index: torch.Tensor,
        mol_type: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        del self, asym_id, residue_index, mol_type
        hidden = torch.zeros(*input_ids.shape, 81, 4, device=input_ids.device)
        hidden[..., 0] = input_ids.unsqueeze(-1)
        return hidden * residue_mask[..., None, None]

    model._compute_lm_hidden_states = MethodType(compute, model)


@pytest.mark.parametrize(
    ("model_class", "experimental", "probe_class"),
    [
        (ESMFold2ForSequenceClassification, False, SequenceClassificationProbe),
        (ESMFold2ForTokenClassification, False, TokenClassificationProbe),
        (
            ESMFold2ExperimentalForSequenceClassification,
            True,
            SequenceClassificationProbe,
        ),
        (ESMFold2ExperimentalForTokenClassification, True, TokenClassificationProbe),
    ],
)
def test_esmfold2_classifier_wrappers_bypass_structure_trunk(
    monkeypatch: pytest.MonkeyPatch,
    model_class: type[nn.Module],
    experimental: bool,
    probe_class: type[nn.Module],
) -> None:
    _replace_base_initializers(monkeypatch)
    model = model_class(_tiny_config(experimental=experimental))
    _attach_tiny_esmc_output(model)
    inputs = model.prepare_classifier_inputs(["ACD", "GG"])

    output = model(**inputs)

    assert isinstance(model.classifier, probe_class)
    assert output.logits.shape == (
        (2, 3) if probe_class is SequenceClassificationProbe else (2, 3, 3)
    )
    assert inputs["attention_mask"].tolist() == [
        [True, True, True],
        [True, True, False],
    ]


def test_esmfold2_classifier_training_scopes_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _replace_base_initializers(monkeypatch)
    model = ESMFold2ForSequenceClassification(_tiny_config(scope="probe"))

    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert not model.language_model.base_z_combine.requires_grad
    assert all(
        not parameter.requires_grad
        for parameter in model.language_model.base_z_linear.parameters()
    )
    assert all(not parameter.requires_grad for parameter in model._esmc.parameters())

    model.set_classifier_train_scope("projection")

    assert model.language_model.base_z_combine.requires_grad
    assert all(
        parameter.requires_grad
        for parameter in model.language_model.base_z_linear.parameters()
    )
    assert all(not parameter.requires_grad for parameter in model._esmc.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.folding_trunk.parameters()
    )


def test_esmfold2_classifier_inputs_reject_non_protein_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _replace_base_initializers(monkeypatch)
    model = ESMFold2ForTokenClassification(_tiny_config())
    _attach_tiny_esmc_output(model)

    with pytest.raises(ValueError, match="ungapped protein chain"):
        model.prepare_classifier_inputs("ACD|GG")
    with pytest.raises(ValueError, match="ungapped"):
        model.prepare_classifier_inputs("AC-D")
    with pytest.raises(ValueError, match="at least one sequence"):
        model.prepare_classifier_inputs([])

    inputs = model.prepare_classifier_inputs("ACD")
    inputs["input_ids"][0, 0] = 0
    with pytest.raises(ValueError, match="residue-only single-chain"):
        model(**inputs)


def test_esmfold2_classifier_config_round_trips(tmp_path) -> None:
    config = _tiny_config(scope="projection")
    config.save_pretrained(tmp_path)

    restored = ESMFold2Config.from_pretrained(tmp_path)

    assert restored.classifier_train_scope == "projection"
    assert restored.classifier_probe_hidden_size == 512
    assert restored.classifier_probe_num_heads == 4
    assert restored.classifier_pooling_types == ["mean"]


def test_esmfold2_classifier_config_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="classifier_train_scope"):
        ESMFold2Config(classifier_train_scope="backbone")
