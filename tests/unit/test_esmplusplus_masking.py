"""Focused ESM++ input-mask contracts."""

from __future__ import annotations

import ast
import inspect
import textwrap
import pytest
import torch
from pathlib import Path

from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    ESMplusplusForSequenceClassification,
    ESMplusplusForTokenClassification,
    ESMplusplusModel,
    TransformerStack,
)


@pytest.mark.parametrize("model_class", (ESMplusplusModel, ESMplusplusForMaskedLM))
def test_esmplusplus_infers_padding_mask_from_input_ids(
    model_class: type[ESMplusplusModel] | type[ESMplusplusForMaskedLM],
) -> None:
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        attn_backend="eager",
        pad_token_id=1,
    )
    model = model_class(config).eval()
    input_ids = torch.tensor([[0, 3, 4, 1, 1]], dtype=torch.long)  # (b=1, l=5)
    attention_mask = input_ids.ne(config.pad_token_id)  # (b=1, l=5)

    kwargs = {"compute_logits": False} if model_class is ESMplusplusForMaskedLM else {}
    with torch.inference_mode():
        inferred = model(input_ids=input_ids, **kwargs).last_hidden_state  # (b=1, l=5, d=16)
        explicit = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        ).last_hidden_state  # (b=1, l=5, d=16)

    torch.testing.assert_close(inferred, explicit, rtol=0.0, atol=0.0)


def test_esmplusplus_boolean_sequence_id_matches_biohub_equality_mask() -> None:
    stack = TransformerStack(
        d_model=16,
        n_heads=4,
        n_layers=1,
        attn_backend="eager",
    )
    sequence_id = torch.tensor([[True, True, True, False, False]])  # (b=1, l=5)

    mask_2d, mask_4d, block_mask = stack._sequence_id_attention_masks(
        sequence_id=sequence_id,
        batch_size=1,
        seq_len=5,
        device=torch.device("cpu"),
    )  # mask_2d: (b=1, l=5); mask_4d: (b=1, 1, l=5, l=5); block_mask: None

    expected = sequence_id[:, None, :, None] == sequence_id[:, None, None, :]  # (1, 1, 5, 5)
    assert torch.equal(mask_2d, sequence_id)
    assert torch.equal(mask_4d, expected)
    assert block_mask is None


@pytest.mark.parametrize("model_class", (ESMplusplusModel, ESMplusplusForMaskedLM))
def test_esmplusplus_embedding_helper_infers_padding_mask(
    model_class: type[ESMplusplusModel] | type[ESMplusplusForMaskedLM],
) -> None:
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        attn_backend="eager",
        pad_token_id=1,
    )
    model = model_class(config).eval()
    input_ids = torch.tensor([[0, 3, 4, 1, 1]], dtype=torch.long)  # (b=1, l=5)
    attention_mask = input_ids.ne(config.pad_token_id)  # (b=1, l=5)

    with torch.inference_mode():
        inferred = model._embed(input_ids)  # (b=1, l=5, d=16)
        explicit = model._embed(input_ids, attention_mask=attention_mask)  # (1, 5, 16)

    torch.testing.assert_close(inferred, explicit, rtol=0.0, atol=0.0)


def _sequence_classifier_config() -> ESMplusplusConfig:
    return ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_labels=3,
        attn_backend="eager",
        pad_token_id=1,
    )


def test_esmplusplus_sequence_classifier_accepts_explicit_pooling_types() -> None:
    model = ESMplusplusForSequenceClassification(
        _sequence_classifier_config(),
        pooling_types=["mean"],
    ).eval()

    assert model.pooler.names == ("mean",)
    assert model.classifier[0].in_features == 16

    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[0, 3, 4, 2]], dtype=torch.long)  # (b=1, l=4)
        )

    assert output.logits is not None
    # b=1 sequences, c=3 classes.
    assert output.logits.shape == (1, 3)


def test_esmplusplus_sequence_classifier_pooling_round_trips(tmp_path: Path) -> None:
    model = ESMplusplusForSequenceClassification(
        _sequence_classifier_config(),
        pooling_types=["mean"],
    ).eval()
    model.save_pretrained(tmp_path)

    reloaded = ESMplusplusForSequenceClassification.from_pretrained(tmp_path).eval()

    assert reloaded.config.classifier_pooling_types == ["mean"]
    assert reloaded.pooler.names == ("mean",)
    assert reloaded.classifier[0].in_features == 16
    for name, value in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], value, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("model_class", "pooling_types", "labels", "expected_shape"),
    (
        (
            ESMplusplusForSequenceClassification,
            ["mean", "var"],
            torch.tensor([1, 2]),  # (b=2,)
            (2, 3),
        ),
        (
            ESMplusplusForTokenClassification,
            None,
            torch.tensor(((0, 1, 2, 1), (2, 1, 0, 1))),  # (b=2, l=4)
            (2, 4, 3),
        ),
    ),
)
def test_esmplusplus_wide_classifier_forward_backward_and_reload(
    model_class: (
        type[ESMplusplusForSequenceClassification]
        | type[ESMplusplusForTokenClassification]
    ),
    pooling_types: list[str] | None,
    labels: torch.Tensor,
    expected_shape: tuple[int, ...],
    tmp_path: Path,
) -> None:
    config = _sequence_classifier_config()
    kwargs = {} if pooling_types is None else {"pooling_types": pooling_types}
    model = model_class(config, **kwargs).train()
    assert model.classifier[0].out_features == config.hidden_size * 4
    assert model.classifier[3].in_features == config.hidden_size * 4
    input_ids = torch.tensor(((0, 3, 4, 2), (0, 5, 6, 2)))  # (b=2, l=4)

    output = model(input_ids=input_ids, labels=labels)
    assert output.logits.shape == expected_shape
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    classifier_gradients = [
        parameter.grad for parameter in model.classifier.parameters() if parameter.requires_grad
    ]
    assert classifier_gradients
    assert all(gradient is not None for gradient in classifier_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in classifier_gradients)

    model.eval()
    with torch.inference_mode():
        expected_logits = model(input_ids=input_ids).logits  # (b, c) or (b, l, c)
    save_path = tmp_path / model_class.__name__
    model.save_pretrained(save_path, safe_serialization=True)
    reloaded = model_class.from_pretrained(save_path, local_files_only=True).eval()
    with torch.inference_mode():
        actual_logits = reloaded(input_ids=input_ids).logits  # same shape as expected_logits
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("pooling_types", "exception"),
    [
        ([], ValueError),
        (("mean",), TypeError),
        (["mean", 1], TypeError),
        (["parti"], ValueError),
    ],
)
def test_esmplusplus_sequence_classifier_validates_pooling_types(
    pooling_types: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception, match="pooling_types"):
        ESMplusplusForSequenceClassification(
            _sequence_classifier_config(),
            pooling_types=pooling_types,
        )


def test_esmplusplus_sequence_classifier_is_right_padding_invariant_without_mask() -> None:
    model = ESMplusplusForSequenceClassification(_sequence_classifier_config()).eval()
    unpadded = torch.tensor([[0, 3, 4, 2]], dtype=torch.long)  # (b=1, l=4)
    right_padded = torch.tensor([[0, 3, 4, 2, 1, 1]], dtype=torch.long)  # (b=1, l=6)

    with torch.inference_mode():
        unpadded_logits = model(input_ids=unpadded).logits  # (b=1, c=3)
        padded_logits = model(input_ids=right_padded).logits  # (b=1, c=3)

    assert unpadded_logits is not None
    assert padded_logits is not None
    torch.testing.assert_close(padded_logits, unpadded_logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "model_class",
    (ESMplusplusModel, ESMplusplusForMaskedLM, ESMplusplusForSequenceClassification),
)
def test_esmplusplus_public_models_require_exactly_one_input_form(
    model_class: (
        type[ESMplusplusModel]
        | type[ESMplusplusForMaskedLM]
        | type[ESMplusplusForSequenceClassification]
    ),
) -> None:
    model = model_class(
        ESMplusplusConfig(
            vocab_size=16,
            hidden_size=16,
            num_attention_heads=4,
            num_hidden_layers=1,
            attn_backend="eager",
        )
    )
    input_ids = torch.ones(1, 2, dtype=torch.long)  # (b=1, l=2)
    inputs_embeds = torch.zeros(1, 2, 16)  # (b=1, l=2, d=16)

    with pytest.raises(ValueError, match="either input_ids or inputs_embeds"):
        model()
    with pytest.raises(ValueError, match="both input_ids and inputs_embeds"):
        model(input_ids=input_ids, inputs_embeds=inputs_embeds)


def test_esmplusplus_masked_lm_labels_require_logits() -> None:
    model = ESMplusplusForMaskedLM(
        ESMplusplusConfig(
            vocab_size=16,
            hidden_size=16,
            num_attention_heads=4,
            num_hidden_layers=1,
            attn_backend="eager",
        )
    )
    input_ids = torch.ones(1, 2, dtype=torch.long)  # (b=1, l=2)

    with pytest.raises(ValueError, match="labels require compute_logits=True"):
        model(input_ids=input_ids, labels=input_ids, compute_logits=False)


@pytest.mark.parametrize(
    "model_class",
    (ESMplusplusModel, ESMplusplusForMaskedLM, ESMplusplusForSequenceClassification),
)
def test_esmplusplus_public_input_validation_survives_python_optimization(
    model_class: (
        type[ESMplusplusModel]
        | type[ESMplusplusForMaskedLM]
        | type[ESMplusplusForSequenceClassification]
    ),
) -> None:
    forward_source = textwrap.dedent(inspect.getsource(model_class.forward))
    forward_tree = ast.parse(forward_source)

    assert not any(isinstance(node, ast.Assert) for node in ast.walk(forward_tree))
