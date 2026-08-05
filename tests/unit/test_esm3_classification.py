from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from fastplms.models.esm3.modeling_esm3 import (
    FastESM3Config,
    FastESM3ForSequenceClassification,
    FastESM3ForTokenClassification,
    FastESM3Model,
)


def _config(*, num_labels: int, problem_type: str | None = None) -> FastESM3Config:
    return FastESM3Config(
        hidden_size=16,
        num_attention_heads=4,
        num_vector_heads=4,
        num_hidden_layers=1,
        num_labels=num_labels,
        problem_type=problem_type,
        attn_backend="eager",
    )


def _input_ids() -> torch.Tensor:
    # BOS, two biological residues, EOS, then right padding.
    return torch.tensor(((0, 4, 5, 2, 1), (0, 6, 7, 2, 1)), dtype=torch.long)


def test_esm3_sequence_classifier_uses_final_residue_embeddings() -> None:
    model = FastESM3ForSequenceClassification(_config(num_labels=3)).eval()
    input_ids = _input_ids()

    with torch.inference_mode():
        backbone = FastESM3Model.forward(
            model,
            input_ids=input_ids,
            attention_mask=input_ids.ne(1),
            output_hidden_states=True,
            return_dict=True,
        )
        output = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    expected_features = backbone.last_hidden_state[:, 1:3].mean(dim=1)
    expected_logits = model.classifier(model.dropout(expected_features))
    assert isinstance(output, SequenceClassifierOutput)
    assert output.logits.shape == (2, 3)
    assert output.hidden_states is not None
    torch.testing.assert_close(output.logits, expected_logits, rtol=0.0, atol=0.0)


def test_esm3_sequence_classifier_is_right_padding_invariant() -> None:
    model = FastESM3ForSequenceClassification(_config(num_labels=3)).eval()
    unpadded = torch.tensor(((0, 4, 5, 2),), dtype=torch.long)
    padded = torch.tensor(((0, 4, 5, 2, 1, 1),), dtype=torch.long)

    with torch.inference_mode():
        unpadded_logits = model(input_ids=unpadded).logits
        padded_logits = model(input_ids=padded).logits

    torch.testing.assert_close(padded_logits, unpadded_logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("num_labels", "labels", "expected_problem_type"),
    (
        (1, torch.tensor((0.25, -0.5)), "regression"),
        (3, torch.tensor((1, 2)), "single_label_classification"),
        (
            3,
            torch.tensor(((1.0, 0.0, 1.0), (0.0, 1.0, 0.0))),
            "multi_label_classification",
        ),
    ),
)
def test_esm3_sequence_classifier_problem_type_losses(
    num_labels: int,
    labels: torch.Tensor,
    expected_problem_type: str,
) -> None:
    model = FastESM3ForSequenceClassification(_config(num_labels=num_labels)).train()

    output = model(input_ids=_input_ids(), labels=labels)

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert model.config.problem_type == expected_problem_type
    output.loss.backward()
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()


def test_esm3_token_classifier_masks_special_padding_and_ignored_labels() -> None:
    model = FastESM3ForTokenClassification(_config(num_labels=3)).eval()
    input_ids = _input_ids()
    labels = torch.tensor(
        ((2, 0, -100, 1, 0), (1, 2, 1, 0, 2)),
        dtype=torch.long,
    )

    with torch.inference_mode():
        output = model(input_ids=input_ids, labels=labels)

    expected_logits = torch.cat((output.logits[0, 1:2], output.logits[1, 1:3]), dim=0)
    expected_labels = torch.tensor((0, 2, 1), dtype=torch.long)
    expected_loss = F.cross_entropy(expected_logits, expected_labels)
    assert isinstance(output, TokenClassifierOutput)
    assert output.logits.shape == (2, 5, 3)
    assert output.loss is not None
    torch.testing.assert_close(output.loss, expected_loss, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("num_labels", "problem_type", "labels"),
    (
        (
            1,
            None,
            torch.tensor(((0.0, 0.25, -100.0, 0.0, 0.0),) * 2),
        ),
        (
            2,
            "regression",
            torch.tensor(
                (
                    (
                        (-100.0, -100.0),
                        (0.1, 0.2),
                        (0.3, 0.4),
                        (-100.0, -100.0),
                        (-100.0, -100.0),
                    ),
                    (
                        (-100.0, -100.0),
                        (0.5, 0.6),
                        (0.7, 0.8),
                        (-100.0, -100.0),
                        (-100.0, -100.0),
                    ),
                )
            ),
        ),
        (
            3,
            None,
            torch.tensor(
                (
                    (
                        (-100.0,) * 3,
                        (1.0, 0.0, 1.0),
                        (0.0, 1.0, 0.0),
                        (-100.0,) * 3,
                        (-100.0,) * 3,
                    ),
                    (
                        (-100.0,) * 3,
                        (0.0, 1.0, 1.0),
                        (1.0, 0.0, 0.0),
                        (-100.0,) * 3,
                        (-100.0,) * 3,
                    ),
                )
            ),
        ),
    ),
)
def test_esm3_token_classifier_regression_and_multilabel_losses(
    num_labels: int,
    problem_type: str | None,
    labels: torch.Tensor,
) -> None:
    model = FastESM3ForTokenClassification(
        _config(num_labels=num_labels, problem_type=problem_type)
    ).train()

    output = model(input_ids=_input_ids(), labels=labels)

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()


@pytest.mark.parametrize(
    "model_class",
    (FastESM3ForSequenceClassification, FastESM3ForTokenClassification),
)
def test_esm3_classifiers_support_tuple_and_dictionary_outputs(
    model_class: type[FastESM3ForSequenceClassification]
    | type[FastESM3ForTokenClassification],
) -> None:
    model = model_class(_config(num_labels=3)).eval()
    input_ids = _input_ids()

    with torch.inference_mode():
        dictionary_output = model(input_ids=input_ids, return_dict=True)
        tuple_output = model(input_ids=input_ids, return_dict=False)

    torch.testing.assert_close(tuple_output[0], dictionary_output.logits, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "model_class",
    (FastESM3ForSequenceClassification, FastESM3ForTokenClassification),
)
def test_esm3_classifier_save_reload_round_trip(
    model_class: type[FastESM3ForSequenceClassification]
    | type[FastESM3ForTokenClassification],
    tmp_path: Path,
) -> None:
    model = model_class(_config(num_labels=3)).eval()
    input_ids = _input_ids()
    save_path = tmp_path / model_class.__name__

    with torch.inference_mode():
        expected = model(input_ids=input_ids).logits
    model.save_pretrained(save_path, safe_serialization=True)
    reloaded = model_class.from_pretrained(
        save_path,
        local_files_only=True,
    ).eval()
    with torch.inference_mode():
        observed = reloaded(input_ids=input_ids).logits

    assert reloaded.config.auto_map[model_class._auto_class].endswith(model_class.__name__)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_esm3_sequence_classifier_preserves_multimodal_inputs() -> None:
    model = FastESM3ForSequenceClassification(_config(num_labels=2)).eval()
    structure_tokens = torch.tensor(((4098, 10, 11, 4097),), dtype=torch.long)
    attention_mask = torch.ones_like(structure_tokens)

    with torch.inference_mode():
        output = model(
            structure_tokens=structure_tokens,
            attention_mask=attention_mask,
        )

    assert output.logits.shape == (1, 2)
    assert torch.isfinite(output.logits).all()
