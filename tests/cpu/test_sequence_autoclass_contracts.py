"""Tiny end-to-end contracts for sequence-family public and advertised models."""

from __future__ import annotations

import pytest
import torch
from pathlib import Path
from transformers.modeling_outputs import ModelOutput

from fastplms.models.esm2.modeling_fastesm import (
    FastEsmConfig,
    FastEsmForMaskedLM,
    FastEsmForSequenceClassification,
    FastEsmForTokenClassification,
    FastEsmModel,
)
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    ESMplusplusForSequenceClassification,
    ESMplusplusForTokenClassification,
    ESMplusplusModel,
)
from tests.integration import test_dplm_generation as dplm_contracts
from tests.unit import test_e1_cache_contract as e1_contracts


def _esm2_config() -> FastEsmConfig:
    return FastEsmConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=16,
        pad_token_id=1,
        mask_token_id=5,
        num_labels=3,
        position_embedding_type="absolute",
        attn_backend="eager",
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([[0, 3, 4, 2, 1], [0, 6, 2, 1, 1]])  # (b=2, l=5)
    return input_ids, input_ids.ne(1)  # (b, l), (b, l)


def _dplm_config_values(vocab_size: int) -> dict[str, object]:
    values = dplm_contracts._common_config(vocab_size)
    values.update(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "max_position_embeddings": 16,
            "attn_backend": "eager",
        }
    )
    return values


def _dplm2_config_values() -> dict[str, object]:
    """Return a tiny config using DPLM2's sole manifest backend."""

    values = _dplm_config_values(64)
    values["attn_backend"] = "sdpa"
    return values


@pytest.mark.parametrize("serialized_backend", (None, "sdpa"))
def test_dplm2_legacy_or_explicit_backend_resolves_to_manifest_sdpa(
    serialized_backend: str | None,
) -> None:
    values = _dplm2_config_values()
    values["attn_backend"] = serialized_backend
    model = dplm_contracts.DPLM2Model(
        dplm_contracts.DPLM2Config(**values)
    )

    assert model.config.attn_backend == "sdpa"
    assert model.attn_backend == "sdpa"


def test_dplm2_rejects_eager_instead_of_expanding_its_backend_claim() -> None:
    values = _dplm2_config_values()
    values["attn_backend"] = "eager"

    with pytest.raises(ValueError, match="does not support 'eager'"):
        dplm_contracts.DPLM2Model(dplm_contracts.DPLM2Config(**values))


@pytest.mark.parametrize(
    "model_class",
    (
        dplm_contracts.DPLM2Model,
        dplm_contracts.DPLM2ForMaskedLM,
        dplm_contracts.DPLM2ForSequenceClassification,
        dplm_contracts.DPLM2ForTokenClassification,
    ),
)
def test_dplm2_multimodal_wrappers_require_types_with_precomputed_embeddings(
    model_class: type,
) -> None:
    model = model_class(
        dplm_contracts.DPLM2Config(
            **_dplm2_config_values(),
            num_labels=3,
        )
    ).eval()
    input_ids = torch.tensor([[0, 6, 7, 2]])
    inputs_embeds = model.get_input_embeddings()(input_ids)

    with pytest.raises(ValueError, match="type_ids is required"):
        model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones_like(input_ids),
        )


def _assert_nested_output_close(
    actual: object,
    expected: object,
    *,
    exact: bool = False,
) -> None:
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        if exact:
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(actual, expected)
        return
    if isinstance(expected, e1_contracts.DynamicCache):
        assert isinstance(actual, e1_contracts.DynamicCache)
        assert len(actual.key_cache) == len(expected.key_cache)
        assert len(actual.value_cache) == len(expected.value_cache)
        for actual_tensor, expected_tensor in zip(
            actual.key_cache + actual.value_cache,
            expected.key_cache + expected.value_cache,
            strict=True,
        ):
            if exact:
                torch.testing.assert_close(
                    actual_tensor,
                    expected_tensor,
                    rtol=0.0,
                    atol=0.0,
                )
            else:
                torch.testing.assert_close(actual_tensor, expected_tensor)
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_output_close(actual_value, expected_value, exact=exact)
        return
    assert actual == expected


def _assert_exact_state_round_trip(
    model: torch.nn.Module,
    reloaded: torch.nn.Module,
) -> None:
    source_state = model.state_dict()  # parameter name -> checkpoint-shaped tensor
    restored_state = reloaded.state_dict()  # parameter name -> checkpoint-shaped tensor
    assert set(restored_state) == set(source_state)
    for name, tensor in source_state.items():
        torch.testing.assert_close(
            restored_state[name],
            tensor,
            rtol=0.0,
            atol=0.0,
        )


def _assert_exact_output_round_trip(
    model: torch.nn.Module,
    reloaded: torch.nn.Module,
    **model_inputs: torch.Tensor,
) -> None:
    forward_controls = {
        "output_attentions": True,
        "output_hidden_states": True,
        "output_s_max": True,
    }
    with torch.inference_mode():
        source_output = model(
            **model_inputs,
            **forward_controls,
            return_dict=True,
        )
        restored_output = reloaded(
            **model_inputs,
            **forward_controls,
            return_dict=True,
        )
        restored_tuple = reloaded(
            **model_inputs,
            **forward_controls,
            return_dict=False,
        )

    assert type(restored_output) is type(source_output)
    assert tuple(restored_output.keys()) == tuple(source_output.keys())
    _assert_nested_output_close(
        restored_output.to_tuple(),
        source_output.to_tuple(),
        exact=True,
    )
    _assert_nested_output_close(
        restored_tuple,
        source_output.to_tuple(),
        exact=True,
    )


@pytest.mark.parametrize(
    "model_class",
    (
        FastEsmModel,
        FastEsmForMaskedLM,
        FastEsmForSequenceClassification,
        FastEsmForTokenClassification,
    ),
)
def test_esm2_advertised_models_forward_loss_backward_resize_and_reload(
    model_class: type,
    tmp_path: Path,
) -> None:
    model = model_class(_esm2_config()).eval()
    input_ids, attention_mask = _inputs()
    structured = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        output_s_max=True,
        return_dict=True,
    )
    tuple_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        output_s_max=True,
        return_dict=False,
    )

    assert isinstance(structured, ModelOutput)
    assert isinstance(tuple_output, tuple)
    _assert_nested_output_close(tuple_output, structured.to_tuple())
    primary = (
        structured.last_hidden_state
        if model_class is FastEsmModel
        else structured.logits
    )
    torch.testing.assert_close(tuple_output[0], primary)
    assert structured.hidden_states is not None
    assert structured.attentions is not None
    assert structured.s_max is not None
    with pytest.raises(TypeError, match="unexpected_cpu_contract"):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            unexpected_cpu_contract=True,
        )

    if model_class is FastEsmForMaskedLM:
        labels = input_ids.masked_fill(~attention_mask, -100)
    elif model_class is FastEsmForSequenceClassification:
        labels = torch.tensor([1, 2])
    elif model_class is FastEsmForTokenClassification:
        labels = input_ids.remainder(3).masked_fill(~attention_mask, -100)
    else:
        labels = None
    if labels is None:
        loss = structured.last_hidden_state.square().mean()
    else:
        loss_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            output_s_max=True,
            return_dict=True,
        )
        loss_tuple = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            output_s_max=True,
            return_dict=False,
        )
        _assert_nested_output_close(loss_tuple, loss_output.to_tuple())
        torch.testing.assert_close(loss_tuple[0], loss_output.loss)
        torch.testing.assert_close(loss_tuple[1], loss_output.logits)
        loss = loss_output.loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    model.resize_token_embeddings(19)
    assert model.get_input_embeddings().num_embeddings == 19
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        assert output_embeddings.out_features == 19

    save_dir = tmp_path / model_class.__name__
    model.save_pretrained(save_dir, safe_serialization=True)
    reloaded = model_class.from_pretrained(save_dir, local_files_only=True).eval()
    assert reloaded.get_input_embeddings().num_embeddings == 19
    _assert_exact_state_round_trip(model, reloaded)
    _assert_exact_output_round_trip(
        model,
        reloaded,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )


@pytest.mark.parametrize(
    ("model_class", "kind"),
    (
        (ESMplusplusModel, "base"),
        (ESMplusplusForMaskedLM, "mlm"),
        (ESMplusplusForSequenceClassification, "sequence"),
        (ESMplusplusForTokenClassification, "token"),
    ),
)
def test_esmc_public_models_forward_loss_backward_resize_and_reload(
    model_class: type,
    kind: str,
    tmp_path: Path,
) -> None:
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        dropout=0.0,
        pad_token_id=1,
        mask_token_id=5,
        num_labels=3,
        attn_backend="eager",
    )
    model = model_class(config).eval()
    input_ids, attention_mask = _inputs()
    structured = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        output_s_max=True,
        return_dict=True,
    )
    tuple_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        output_s_max=True,
        return_dict=False,
    )

    assert isinstance(structured, ModelOutput)
    assert isinstance(tuple_output, tuple)
    _assert_nested_output_close(tuple_output, structured.to_tuple())
    primary = structured.last_hidden_state if kind == "base" else structured.logits
    torch.testing.assert_close(tuple_output[0], primary)
    assert structured.hidden_states is not None
    assert structured.attentions is not None
    assert structured.s_max is not None
    with pytest.raises(TypeError, match="unexpected_cpu_contract"):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            unexpected_cpu_contract=True,
        )

    if kind == "mlm":
        labels = input_ids.masked_fill(~attention_mask, -100)
    elif kind == "sequence":
        labels = torch.tensor([1, 2])
    elif kind == "token":
        labels = input_ids.remainder(3).masked_fill(~attention_mask, -100)
    else:
        labels = None
    if labels is not None:
        loss_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            output_s_max=True,
            return_dict=True,
        )
        loss_tuple = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            output_s_max=True,
            return_dict=False,
        )
        _assert_nested_output_close(loss_tuple, loss_output.to_tuple())
        torch.testing.assert_close(loss_tuple[0], loss_output.loss)
        torch.testing.assert_close(loss_tuple[1], loss_output.logits)
        loss = loss_output.loss
    else:
        loss = structured.last_hidden_state.square().mean()
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    model.resize_token_embeddings(19)
    assert model.get_input_embeddings().num_embeddings == 19
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        assert output_embeddings.out_features == 19

    save_dir = tmp_path / model_class.__name__
    model.save_pretrained(save_dir, safe_serialization=True)
    reloaded = model_class.from_pretrained(save_dir, local_files_only=True).eval()
    assert reloaded.get_input_embeddings().num_embeddings == 19
    _assert_exact_state_round_trip(model, reloaded)
    _assert_exact_output_round_trip(
        model,
        reloaded,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )


@pytest.mark.parametrize(
    ("model_class", "config_class", "vocab_size", "kind"),
    (
        (dplm_contracts.DPLMModel, dplm_contracts.DPLMConfig, 33, "base"),
        (dplm_contracts.DPLMForMaskedLM, dplm_contracts.DPLMConfig, 33, "mlm"),
        (
            dplm_contracts.DPLMForSequenceClassification,
            dplm_contracts.DPLMConfig,
            33,
            "sequence",
        ),
        (
            dplm_contracts.DPLMForTokenClassification,
            dplm_contracts.DPLMConfig,
            33,
            "token",
        ),
        (dplm_contracts.DPLM2Model, dplm_contracts.DPLM2Config, 64, "base"),
        (dplm_contracts.DPLM2ForMaskedLM, dplm_contracts.DPLM2Config, 64, "mlm"),
        (
            dplm_contracts.DPLM2ForSequenceClassification,
            dplm_contracts.DPLM2Config,
            64,
            "sequence",
        ),
        (
            dplm_contracts.DPLM2ForTokenClassification,
            dplm_contracts.DPLM2Config,
            64,
            "token",
        ),
    ),
)
def test_dplm_advertised_models_forward_loss_backward_resize_and_reload(
    model_class: type,
    config_class: type,
    vocab_size: int,
    kind: str,
    tmp_path: Path,
) -> None:
    config_values = (
        _dplm2_config_values()
        if config_class is dplm_contracts.DPLM2Config
        else _dplm_config_values(vocab_size)
    )
    config = config_class(
        **config_values,
        num_labels=3,
        return_dict=True,
    )
    model = model_class(config).eval()
    input_ids = torch.tensor([[0, 6, 7, 2, 1], [0, 8, 2, 1, 1]])  # (b=2, l=5)
    attention_mask = input_ids.ne(1)  # (b, l)
    structured = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        output_s_max=True,
        return_dict=True,
    )
    tuple_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        output_s_max=True,
        return_dict=False,
    )

    assert isinstance(structured, ModelOutput)
    assert isinstance(tuple_output, tuple)
    _assert_nested_output_close(tuple_output, structured.to_tuple())
    primary = structured.last_hidden_state if kind == "base" else structured.logits
    torch.testing.assert_close(tuple_output[0], primary)
    assert structured.hidden_states is not None
    assert structured.attentions is not None
    assert structured.s_max is not None
    with pytest.raises(TypeError, match="unexpected_cpu_contract"):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            unexpected_cpu_contract=True,
        )
    if kind == "mlm":
        labels = input_ids.masked_fill(~attention_mask, -100)
    elif kind == "sequence":
        labels = torch.tensor([1, 2])
    elif kind == "token":
        labels = input_ids.remainder(3).masked_fill(~attention_mask, -100)
    else:
        labels = None
    if labels is None:
        loss = structured.last_hidden_state.square().mean()
    else:
        loss_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            output_s_max=True,
            return_dict=True,
        )
        loss_tuple = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            output_s_max=True,
            return_dict=False,
        )
        _assert_nested_output_close(loss_tuple, loss_output.to_tuple())
        torch.testing.assert_close(loss_tuple[0], loss_output.loss)
        torch.testing.assert_close(loss_tuple[1], loss_output.logits)
        loss = loss_output.loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    model.resize_token_embeddings(vocab_size + 3)
    assert model.get_input_embeddings().num_embeddings == vocab_size + 3
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        assert output_embeddings.out_features == vocab_size + 3
    save_dir = tmp_path / f"{model_class.__name__}-{vocab_size}"
    model.save_pretrained(save_dir, safe_serialization=True)
    reloaded = model_class.from_pretrained(save_dir, local_files_only=True).eval()
    assert reloaded.get_input_embeddings().num_embeddings == vocab_size + 3
    _assert_exact_state_round_trip(model, reloaded)
    _assert_exact_output_round_trip(
        model,
        reloaded,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )


@pytest.mark.parametrize(
    ("model_class", "kind"),
    (
        (e1_contracts.E1Model, "base"),
        (e1_contracts.E1ForMaskedLM, "mlm"),
        (e1_contracts.E1ForSequenceClassification, "sequence"),
        (e1_contracts.E1ForTokenClassification, "token"),
    ),
)
def test_e1_advertised_models_forward_loss_backward_resize_and_reload(
    model_class: type,
    kind: str,
    tmp_path: Path,
) -> None:
    config = e1_contracts._tiny_e1_config()
    config.num_labels = 3
    model = model_class(config).eval()
    batch = e1_contracts._tiny_e1_batch()
    structured = model(
        **batch,
        output_attentions=True,
        output_hidden_states=True,
        output_s_max=True,
        return_dict=True,
    )
    tuple_output = model(
        **batch,
        output_attentions=True,
        output_hidden_states=True,
        output_s_max=True,
        return_dict=False,
    )

    assert isinstance(structured, ModelOutput)
    assert isinstance(tuple_output, tuple)
    _assert_nested_output_close(tuple_output, structured.to_tuple())
    primary = structured.last_hidden_state if kind == "base" else structured.logits
    torch.testing.assert_close(tuple_output[0], primary)
    assert structured.hidden_states is not None
    assert structured.attentions is not None
    assert structured.s_max is not None
    with pytest.raises(TypeError, match="unexpected_cpu_contract"):
        model(**batch, unexpected_cpu_contract=True)
    input_ids = batch["input_ids"]
    if kind == "mlm":
        labels = input_ids.clone()
    elif kind == "sequence":
        labels = torch.tensor([1])
    elif kind == "token":
        labels = input_ids.remainder(3)
    else:
        labels = None
    if labels is None:
        loss = structured.last_hidden_state.square().mean()
    else:
        loss_output = model(
            **batch,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            output_s_max=True,
            return_dict=True,
        )
        loss_tuple = model(
            **batch,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            output_s_max=True,
            return_dict=False,
        )
        _assert_nested_output_close(loss_tuple, loss_output.to_tuple())
        torch.testing.assert_close(loss_tuple[0], loss_output.loss)
        torch.testing.assert_close(loss_tuple[1], loss_output.logits)
        loss = loss_output.loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    model.resize_token_embeddings(39)
    assert model.get_input_embeddings().num_embeddings == 39
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        assert output_embeddings.out_features == 39
    save_dir = tmp_path / model_class.__name__
    model.save_pretrained(save_dir, safe_serialization=True)
    reloaded = model_class.from_pretrained(save_dir, local_files_only=True).eval()
    assert reloaded.get_input_embeddings().num_embeddings == 39
    _assert_exact_state_round_trip(model, reloaded)
    _assert_exact_output_round_trip(model, reloaded, **batch)
