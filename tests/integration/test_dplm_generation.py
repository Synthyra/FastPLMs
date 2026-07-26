"""DPLM and DPLM2 diffusion-generation feature contracts."""

from __future__ import annotations

import pytest
import torch
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from fastplms.models.dplm.modeling_dplm import (
    DPLMConfig,
    DPLMForMaskedLM,
    DPLMForSequenceClassification,
    DPLMForTokenClassification,
    DPLMModel,
    DPLMSequenceClassifierOutput,
    DPLMTokenClassifierOutput,
)
from fastplms.models.dplm2.modeling_dplm2 import (
    FAST_DPLM2_ENCODER,
    DPLM2Config,
    DPLM2EncoderOutput,
    DPLM2ForMaskedLM,
    DPLM2ForSequenceClassification,
    DPLM2ForTokenClassification,
    DPLM2MaskedLMOutput,
    DPLM2Model,
    DPLM2ModelOutput,
    DPLM2SequenceClassifierOutput,
    DPLM2TokenClassifierOutput,
    ModifiedRotaryEmbedding,
)


pytestmark = pytest.mark.feature


def _common_config(vocab_size: int) -> dict[str, object]:
    return {
        "vocab_size": vocab_size,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 64,
        "pad_token_id": 1,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "mask_token_id": 32,
        "position_embedding_type": "rotary",
        "attn_backend": "sdpa",
    }


def _assert_nested_close(actual, expected) -> None:
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        torch.testing.assert_close(actual, expected)
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_close(actual_value, expected_value)
        return
    assert actual == expected


def test_dplm_argmax_generation_preserves_fixed_positions() -> None:
    torch.manual_seed(13)
    model = DPLMForMaskedLM(DPLMConfig(**_common_config(33)), dropout=0.0).eval()
    # input_tokens: (1, 6)
    input_tokens = torch.tensor([[0, 6, 7, 8, 2, 1]])
    # fixed: (1, 6)
    fixed = torch.tensor([[False, True, False, False, False, False]])

    output_tokens = model.generate(
        input_tokens,
        max_iter=3,
        partial_masks=fixed,
        sampling_strategy="argmax",
        disable_resample=True,
    )

    assert output_tokens.shape == input_tokens.shape
    assert output_tokens[0, 1].item() == 6
    assert torch.equal(output_tokens[0, [0, 4, 5]], input_tokens[0, [0, 4, 5]])
    generated = output_tokens[0, 2:4]
    assert not bool(torch.isin(generated, torch.tensor([0, 1, 2, 24, 32])).any())


def test_dplm_vanilla_default_is_zero_temperature() -> None:
    model = DPLMForMaskedLM(DPLMConfig(**_common_config(33)), dropout=0.0).eval()
    # input_tokens: (1, 5)
    input_tokens = torch.tensor([[0, 6, 7, 8, 2]])

    torch.manual_seed(29)
    default_output = model.generate(
        input_tokens,
        max_iter=2,
        sampling_strategy="vanilla",
        disable_resample=True,
    )
    torch.manual_seed(31)
    zero_temperature_output = model.generate(
        input_tokens,
        max_iter=2,
        temperature=0.0,
        sampling_strategy="vanilla",
        disable_resample=True,
    )

    assert torch.equal(default_output, zero_temperature_output)


@pytest.mark.parametrize(
    ("model_class", "config_class", "vocab_size"),
    (
        (DPLMForMaskedLM, DPLMConfig, 33),
        (DPLM2ForMaskedLM, DPLM2Config, 64),
    ),
)
def test_dplm_families_reject_static_bf16_inference(
    model_class: type[torch.nn.Module],
    config_class: type,
    vocab_size: int,
) -> None:
    model = (
        model_class(
            config_class(**_common_config(vocab_size)),
            dropout=0.0,
        )
        .to(dtype=torch.bfloat16)
        .eval()
    )
    # X: (1, 4)
    X = torch.tensor([[0, 6, 7, 2]])

    with pytest.raises(RuntimeError, match="FP32-resident parameters"):
        model(input_ids=X, attention_mask=torch.ones_like(X))


@pytest.mark.parametrize(
    ("model_class", "config_class", "vocab_size"),
    (
        (DPLMForMaskedLM, DPLMConfig, 33),
        (DPLM2ForMaskedLM, DPLM2Config, 64),
    ),
)
def test_masked_lm_dropout_round_trips_through_config_and_weights(
    model_class: type[torch.nn.Module],
    config_class: type,
    vocab_size: int,
    tmp_path,
) -> None:
    config = config_class(**_common_config(vocab_size))
    config.hidden_dropout_prob = 0.37
    model = model_class(config).eval()
    model.save_pretrained(tmp_path)

    reloaded = model_class.from_pretrained(tmp_path, local_files_only=True).eval()

    assert model.config.hidden_dropout_prob == pytest.approx(0.37)
    assert reloaded.config.hidden_dropout_prob == pytest.approx(0.37)


@pytest.mark.parametrize(
    ("model_class", "config_class", "vocab_size"),
    (
        (DPLMForMaskedLM, DPLMConfig, 33),
        (DPLM2ForMaskedLM, DPLM2Config, 64),
    ),
)
def test_masked_lm_resize_updates_input_and_output_projections(
    model_class: type[torch.nn.Module],
    config_class: type,
    vocab_size: int,
    tmp_path,
) -> None:
    model = model_class(config_class(**_common_config(vocab_size))).eval()
    resized_vocab_size = vocab_size + 5
    # input_ids: (1, 4)
    input_ids = torch.tensor([[0, 6, 7, 2]])
    with torch.inference_mode():
        original_logits = model(input_ids=input_ids).logits

    model.resize_token_embeddings(resized_vocab_size)

    assert model.get_input_embeddings().num_embeddings == resized_vocab_size
    assert model.get_output_embeddings().out_features == resized_vocab_size
    assert model.lm_head.bias.shape == (resized_vocab_size,)
    assert model.config.vocab_size == resized_vocab_size
    assert model.get_output_embeddings().bias is None

    with torch.inference_mode():
        output = model(input_ids=input_ids)
    assert output.logits.shape == (*input_ids.shape, resized_vocab_size)
    torch.testing.assert_close(output.logits[..., :vocab_size], original_logits)

    save_path = tmp_path / model_class.__name__
    model.save_pretrained(save_path, safe_serialization=True)
    reloaded = model_class.from_pretrained(save_path, local_files_only=True).eval()
    assert reloaded.get_input_embeddings().num_embeddings == resized_vocab_size
    assert reloaded.get_output_embeddings().out_features == resized_vocab_size
    assert reloaded.get_output_embeddings().bias is None
    assert reloaded.lm_head.bias.shape == (resized_vocab_size,)


@pytest.mark.parametrize(
    ("model_class", "config_class", "vocab_size", "output_class", "labels"),
    (
        (
            DPLMForSequenceClassification,
            DPLMConfig,
            33,
            DPLMSequenceClassifierOutput,
            [1],
        ),
        (
            DPLMForTokenClassification,
            DPLMConfig,
            33,
            DPLMTokenClassifierOutput,
            [[1, 1, 1, 1]],
        ),
        (
            DPLM2ForSequenceClassification,
            DPLM2Config,
            64,
            DPLM2SequenceClassifierOutput,
            [1],
        ),
        (
            DPLM2ForTokenClassification,
            DPLM2Config,
            64,
            DPLM2TokenClassifierOutput,
            [[1, 1, 1, 1]],
        ),
    ),
)
def test_dplm_task_heads_honor_config_and_explicit_return_dict(
    model_class: type[torch.nn.Module],
    config_class: type,
    vocab_size: int,
    output_class: type,
    labels: list[int] | list[list[int]],
) -> None:
    config = config_class(**_common_config(vocab_size), num_labels=3, return_dict=False)
    model = model_class(config).eval()
    # input_ids: (1, 4)
    input_ids = torch.tensor([[0, 6, 7, 2]])
    # label_tensor: (...)
    label_tensor = torch.tensor(labels)

    with torch.inference_mode():
        unlabeled_tuple = model(input_ids=input_ids)
        unlabeled_output = model(input_ids=input_ids, return_dict=True)
        labeled_tuple = model(
            input_ids=input_ids,
            labels=label_tensor,
            output_attentions=True,
            output_hidden_states=True,
            output_s_max=True,
        )
        labeled_output = model(
            input_ids=input_ids,
            labels=label_tensor,
            output_attentions=True,
            output_hidden_states=True,
            output_s_max=True,
            return_dict=True,
        )

    assert type(unlabeled_output) is output_class
    assert tuple(unlabeled_output.keys()) == ("logits",)
    assert isinstance(unlabeled_tuple, tuple)
    assert len(unlabeled_tuple) == 1
    torch.testing.assert_close(unlabeled_tuple[0], unlabeled_output.logits)

    assert type(labeled_output) is output_class
    assert isinstance(labeled_output, (SequenceClassifierOutput, TokenClassifierOutput))
    assert tuple(labeled_output.keys()) == (
        "loss",
        "logits",
        "hidden_states",
        "attentions",
        "s_max",
    )
    assert isinstance(labeled_tuple, tuple)
    _assert_nested_close(labeled_tuple, labeled_output.to_tuple())


@pytest.mark.parametrize(
    ("model_class", "output_class", "expected_keys"),
    (
        (
            DPLM2Model,
            DPLM2ModelOutput,
            ("last_hidden_state", "hidden_states", "attentions", "s_max"),
        ),
        (
            DPLM2ForMaskedLM,
            DPLM2MaskedLMOutput,
            (
                "loss",
                "logits",
                "hidden_states",
                "attentions",
                "s_max",
                "last_hidden_state",
            ),
        ),
    ),
)
def test_dplm2_base_and_mlm_preserve_full_structured_tuple_contract(
    model_class: type[torch.nn.Module],
    output_class: type,
    expected_keys: tuple[str, ...],
) -> None:
    model = model_class(DPLM2Config(**_common_config(64))).eval()
    # input_ids: (1, 4)
    input_ids = torch.tensor([[0, 6, 7, 2]])
    call_kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "output_attentions": True,
        "output_hidden_states": True,
        "output_s_max": True,
    }
    if model_class is DPLM2ForMaskedLM:
        call_kwargs["labels"] = input_ids

    with torch.inference_mode():
        structured = model(**call_kwargs, return_dict=True)
        tuple_output = model(**call_kwargs, return_dict=False)

    assert type(structured) is output_class
    assert tuple(structured.keys()) == expected_keys
    assert structured.s_max is not None
    assert all(value is not None for value in tuple_output)
    _assert_nested_close(tuple_output, structured.to_tuple())


@pytest.mark.parametrize(
    "model_class",
    (
        DPLMForSequenceClassification,
        DPLMForTokenClassification,
        DPLM2ForSequenceClassification,
        DPLM2ForTokenClassification,
    ),
)
@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("use_cache", True),
        ("past_key_values", ((torch.zeros(1), torch.zeros(1)),)),
        ("encoder_hidden_states", torch.zeros(1, 2, 32)),
        ("unexpected_option", "typo"),
    ),
)
def test_dplm_task_heads_reject_every_unexpected_argument(
    model_class: type[torch.nn.Module],
    argument: str,
    value: object,
) -> None:
    config_class = DPLM2Config if model_class.__name__.startswith("DPLM2") else DPLMConfig
    vocab_size = 64 if config_class is DPLM2Config else 33
    model = model_class(config_class(**_common_config(vocab_size), num_labels=3)).eval()

    with pytest.raises(TypeError, match=argument):
        model(input_ids=torch.tensor([[0, 6, 7, 2]]), **{argument: value})


_DPLM2_INPUT_CONTRACT_MODELS = (
    FAST_DPLM2_ENCODER,
    DPLM2Model,
    DPLM2ForMaskedLM,
    DPLM2ForSequenceClassification,
    DPLM2ForTokenClassification,
)


@pytest.mark.parametrize("model_class", _DPLM2_INPUT_CONTRACT_MODELS)
def test_dplm2_public_models_require_exactly_one_input_representation(
    model_class: type[torch.nn.Module],
) -> None:
    model = model_class(DPLM2Config(**_common_config(64))).eval()
    # input_ids: (1, 4)
    input_ids = torch.tensor([[0, 6, 7, 2]])
    inputs_embeds = model.get_input_embeddings()(input_ids)

    with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
        model()
    with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
        model(input_ids=input_ids, inputs_embeds=inputs_embeds)


@pytest.mark.parametrize("model_class", _DPLM2_INPUT_CONTRACT_MODELS)
@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("attention_mask", torch.ones(1, 3, dtype=torch.long)),
        ("type_ids", torch.ones(2, 4, dtype=torch.long)),
    ),
)
def test_dplm2_public_models_validate_mask_and_type_shapes_before_forward(
    model_class: type[torch.nn.Module],
    argument: str,
    value: torch.Tensor,
) -> None:
    # value: (...)
    model = model_class(DPLM2Config(**_common_config(64))).eval()

    with pytest.raises(ValueError, match=rf"{argument} must have shape \(1, 4\)"):
        model(input_ids=torch.tensor([[0, 6, 7, 2]]), **{argument: value})


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("decoder_input_ids", torch.ones(1, 2, dtype=torch.long)),
        ("decoder_attention_mask", torch.ones(1, 2, dtype=torch.long)),
        ("decoder_inputs_embeds", torch.ones(1, 2, 32)),
        ("encoder_hidden_states", torch.ones(1, 2, 32)),
        ("encoder_attention_mask", torch.ones(1, 2, dtype=torch.long)),
    ),
)
def test_dplm_masked_lm_rejects_decoder_and_cross_attention_arguments(
    argument: str,
    value: torch.Tensor,
) -> None:
    # value: (...)
    model = DPLMForMaskedLM(DPLMConfig(**_common_config(33)), dropout=0.0).eval()
    # input_ids: (1, 3)
    input_ids = torch.tensor([[0, 6, 2]])

    with pytest.raises(ValueError, match=argument):
        model(input_ids=input_ids, **{argument: value})


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("past_key_values", ((torch.zeros(1), torch.zeros(1)),)),
        ("use_cache", True),
        ("encoder_hidden_states", torch.ones(1, 2, 32)),
    ),
)
def test_dplm_automodel_rejects_decoder_cache_contracts(
    argument: str,
    value: object,
) -> None:
    model = DPLMModel(DPLMConfig(**_common_config(33))).eval()
    # input_ids: (1, 3)
    input_ids = torch.tensor([[0, 6, 2]])

    with pytest.raises(ValueError, match=argument):
        model(input_ids=input_ids, **{argument: value})


def test_dplm2_rotary_cache_follows_frequency_buffer_dtype() -> None:
    rotary = ModifiedRotaryEmbedding(dim=8, aa_type=1, struct_type=0, pad_type=2)
    # Q and K are query and key tensors with shape (b, h, l, d).
    query = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    # type_ids: (1, 4)
    type_ids = torch.ones(1, 4, dtype=torch.long)

    rotary(query, key, type_ids)
    assert rotary._cos_cached is not None
    assert rotary._cos_cached.dtype == rotary.inv_freq.dtype == torch.float32

    rotary.align_frequency_buffer(device=query.device, dtype=torch.bfloat16)
    assert rotary._cos_cached is None
    assert rotary._sin_cached is None
    rotary(query, key, type_ids)
    assert rotary._cos_cached is not None
    assert rotary._cos_cached.dtype == rotary.inv_freq.dtype == torch.bfloat16


def test_dplm2_direct_esm_checkpoint_applies_embeddings_once() -> None:
    config = DPLM2Config(**_common_config(64), dplm_type="dplm_esm")
    model = DPLM2ForMaskedLM(config, dropout=0.0).eval()
    calls = 0

    def count_embedding_calls(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        _output: object,
    ) -> None:
        nonlocal calls
        calls += 1

    handle = model.esm.embeddings.register_forward_hook(count_embedding_calls)
    try:
        model(
            input_ids=torch.tensor([[0, 6, 7, 2]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )
    finally:
        handle.remove()

    assert calls == 1


def test_dplm2_automodel_infers_official_multimodal_types_and_returns_pooling() -> None:
    model = object.__new__(DPLM2Model)
    torch.nn.Module.__init__(model)
    model.config = DPLM2Config(**_common_config(64))
    # input_ids: (1, 8)
    input_ids = torch.tensor([[33, 50, 34, 1, 0, 6, 2, 1]])
    # expected_mask: (...)
    expected_mask = input_ids.ne(model.config.pad_token_id)
    # expected_types: (1, 8)
    expected_types = torch.tensor([[0, 0, 0, 2, 1, 1, 1, 2]])
    observed: dict[str, torch.Tensor] = {}

    class CapturingEncoder(torch.nn.Module):
        def forward(self, **kwargs: object) -> DPLM2EncoderOutput:
            # observed['attention_mask']: (b, l)
            observed["attention_mask"] = kwargs["attention_mask"].detach().clone()
            # observed['type_ids']: (...)
            observed["type_ids"] = kwargs["type_ids"].detach().clone()
            return DPLM2EncoderOutput(
                last_hidden_state=torch.zeros(
                    1, input_ids.shape[1], model.config.hidden_size
                )
            )

    model.esm = CapturingEncoder()
    model.pooler = torch.nn.Identity()
    output = model(input_ids=input_ids)
    tuple_output = model(input_ids=input_ids, return_dict=False)

    assert torch.equal(observed["attention_mask"], expected_mask)
    assert torch.equal(observed["type_ids"], expected_types)
    assert output.pooler_output is not None
    assert tuple_output[1] is not None


def test_dplm2_predict_contacts_derives_padding_mask_when_omitted() -> None:
    model = DPLM2ForMaskedLM(DPLM2Config(**_common_config(64)), dropout=0.0).eval()
    # input_ids: (1, 5)
    input_ids = torch.tensor([[0, 6, 7, 2, 1]])
    # expected_mask: (...)
    expected_mask = input_ids.ne(model.config.pad_token_id)
    observed: dict[str, torch.Tensor] = {}

    def capture_modality_type(input_ids, attention_mask):
        # observed['attention_mask']: (b, l)
        observed["attention_mask"] = attention_mask.detach().clone()
        return torch.ones_like(input_ids)

    model.esm._get_modality_type = capture_modality_type
    contacts = model.predict_contacts(input_ids)

    assert torch.equal(observed["attention_mask"], expected_mask)
    assert contacts.shape == (1, input_ids.shape[1] - 2, input_ids.shape[1] - 2)
    assert torch.isfinite(contacts).all()


def test_dplm2_argmax_generation_preserves_modalities_and_fixed_positions() -> None:
    torch.manual_seed(17)
    model = DPLM2ForMaskedLM(DPLM2Config(**_common_config(64)), dropout=0.0).eval()
    # X packs the structure track first and the amino-acid track second, as in
    # the official DPLM2 co-generation utility.
    # input_tokens: (1, 8)
    input_tokens = torch.tensor([[33, 50, 50, 34, 0, 6, 6, 2]])
    # fixed: (1, 8)
    fixed = torch.tensor([[False, True, False, False, False, True, False, False]])
    model_inputs: list[torch.Tensor] = []

    def capture_input(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        # input_tensor: (...)
        input_tensor = kwargs["input_ids"]
        assert torch.is_tensor(input_tensor)
        model_inputs.append(input_tensor.detach().clone())

    handle = model.register_forward_pre_hook(capture_input, with_kwargs=True)

    try:
        output = model.generate(
            input_tokens,
            max_iter=3,
            partial_masks=fixed,
            unmasking_strategy="deterministic",
            sampling_strategy="argmax",
        )
    finally:
        handle.remove()
    output_tokens = output["output_tokens"]

    assert model_inputs[0][0, 2].item() == model.config.vocab_size - 1
    assert model_inputs[0][0, 6].item() == 32
    assert output_tokens.shape == input_tokens.shape
    assert output_tokens[0, 1].item() == 50
    assert output_tokens[0, 5].item() == 6
    assert torch.equal(output_tokens[0, [0, 3, 4, 7]], input_tokens[0, [0, 3, 4, 7]])
    assert int(output_tokens[0, 2]) >= 37
    amino_acid_token = output_tokens[0, 6]
    assert int(amino_acid_token) < 33
    assert int(amino_acid_token) not in {0, 1, 2, 3, 24, 25, 26, 27, 28, 32}


@pytest.mark.parametrize("family", ("dplm", "dplm2"))
def test_seeded_stochastic_generation_is_repeatable(family: str) -> None:
    if family == "dplm":
        model = DPLMForMaskedLM(DPLMConfig(**_common_config(33)), dropout=0.0).eval()
        # X: (1, 5)
        X = torch.tensor([[0, 6, 7, 8, 2]])
        kwargs: dict[str, object] = {"max_iter": 2}
    else:
        model = DPLM2ForMaskedLM(DPLM2Config(**_common_config(64)), dropout=0.0).eval()
        # X: (1, 8)
        X = torch.tensor([[33, 50, 50, 34, 0, 6, 6, 2]])
        kwargs = {"max_iter": 2}

    outputs = []
    for _ in range(2):
        torch.manual_seed(23)
        generated = model.generate(X, **kwargs)
        outputs.append(generated["output_tokens"] if isinstance(generated, dict) else generated)
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize(
    ("family", "arguments", "message"),
    (
        ("dplm", {"max_iter": 0}, "max_iter"),
        ("dplm", {"max_iter": 1, "sampling_strategy": "unknown"}, "sampling strategy"),
        ("dplm2", {"max_iter": 1, "unmasking_strategy": "unknown"}, "unmasking strategy"),
        ("dplm2", {"max_iter": 1, "sampling_strategy": "unknown"}, "sampling strategy"),
        ("dplm2", {"max_iter": 1, "sampling_strategy": "annealing@bad"}, "Annealing"),
    ),
)
def test_generation_rejects_invalid_controls(
    family: str,
    arguments: dict[str, object],
    message: str,
) -> None:
    if family == "dplm":
        model = DPLMForMaskedLM(DPLMConfig(**_common_config(33)), dropout=0.0).eval()
        # X: (1, 3)
        X = torch.tensor([[0, 32, 2]])
    else:
        model = DPLM2ForMaskedLM(DPLM2Config(**_common_config(64)), dropout=0.0).eval()
        # X: (1, 6)
        X = torch.tensor([[33, 36, 34, 0, 32, 2]])
    with pytest.raises(ValueError, match=message):
        model.generate(X, **arguments)
