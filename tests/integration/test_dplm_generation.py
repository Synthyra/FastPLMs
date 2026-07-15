"""DPLM and DPLM2 diffusion-generation feature contracts."""

from __future__ import annotations

import pytest
import torch

from fastplms.models.dplm.modeling_dplm import DPLMConfig, DPLMForMaskedLM
from fastplms.models.dplm2.modeling_dplm2 import (
    DPLM2Config,
    DPLM2ForMaskedLM,
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


def test_dplm_argmax_generation_preserves_fixed_positions() -> None:
    torch.manual_seed(13)
    model = DPLMForMaskedLM(DPLMConfig(**_common_config(33)), dropout=0.0).eval()
    input_tokens = torch.tensor([[0, 6, 7, 8, 2, 1]])
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
    X = torch.tensor([[0, 6, 7, 2]])

    with pytest.raises(RuntimeError, match="FP32-resident parameters"):
        model(input_ids=X, attention_mask=torch.ones_like(X))


def test_dplm2_rotary_cache_follows_frequency_buffer_dtype() -> None:
    rotary = ModifiedRotaryEmbedding(dim=8, aa_type=1, struct_type=0, pad_type=2)
    # Q and K are query and key tensors with shape (b, h, l, d).
    query = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    key = torch.randn_like(query)
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


def test_dplm2_argmax_generation_preserves_modalities_and_fixed_positions() -> None:
    torch.manual_seed(17)
    model = DPLM2ForMaskedLM(DPLM2Config(**_common_config(64)), dropout=0.0).eval()
    # X packs the structure track first and the amino-acid track second, as in
    # the official DPLM2 co-generation utility.
    input_tokens = torch.tensor([[33, 50, 50, 34, 0, 6, 6, 2]])
    fixed = torch.tensor([[False, True, False, False, False, True, False, False]])
    model_inputs: list[torch.Tensor] = []

    def capture_input(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
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
        X = torch.tensor([[0, 6, 7, 8, 2]])
        kwargs: dict[str, object] = {"max_iter": 2}
    else:
        model = DPLM2ForMaskedLM(DPLM2Config(**_common_config(64)), dropout=0.0).eval()
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
        X = torch.tensor([[0, 32, 2]])
    else:
        model = DPLM2ForMaskedLM(DPLM2Config(**_common_config(64)), dropout=0.0).eval()
        X = torch.tensor([[33, 36, 34, 0, 32, 2]])
    with pytest.raises(ValueError, match=message):
        model.generate(X, **arguments)
