"""Exact parity tests for independently maintained E1 runtime contracts."""

from __future__ import annotations

import importlib
import sys
import pytest
import torch
import torch.nn as nn
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from fastplms.models.e1.modeling_e1 import (
    FAST_E1_ENCODER,
    AttentionArgs,
    DataPrepConfig,
    DecoderLayer,
    E1BatchPreparer,
    E1Config,
    E1PreTrainedModel,
    RMSNorm,
    _get_unpad_data,
    build_block_causal_mask_4d,
    build_within_seq_mask_4d,
    create_within_seq_block_mask,
    direct_block_mask,
    get_overlapping_blocks,
    get_tokenizer,
)


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SRC = ROOT / "vendor/upstream/e1/src"


@contextmanager
def _official_e1_modules() -> Iterator[tuple[ModuleType, ModuleType, ModuleType]]:
    assert (UPSTREAM_SRC / "E1").is_dir(), "pinned E1 submodule is missing"
    previous_path = list(sys.path)
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "E1" or name.startswith("E1.")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(UPSTREAM_SRC))
    try:
        yield (
            importlib.import_module("E1.batch_preparer"),
            importlib.import_module("E1.model.varlen_flex_attention"),
            importlib.import_module("E1.model.flash_attention_utils"),
        )
    finally:
        sys.path[:] = previous_path
        for name in tuple(sys.modules):
            if name == "E1" or name.startswith("E1."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


@pytest.fixture(scope="module")
def official_e1() -> Iterator[tuple[ModuleType, ModuleType, ModuleType]]:
    with _official_e1_modules() as modules:
        yield modules


def _assert_encoding_equal(candidate: dict[str, Any], official: dict[str, Any]) -> None:
    assert candidate.keys() == official.keys()
    for key in candidate:
        if isinstance(candidate[key], torch.Tensor):
            assert torch.equal(candidate[key], official[key]), key
        else:
            assert candidate[key] == official[key], key


@pytest.mark.parametrize("remove_x_tokens", (False, True))
@pytest.mark.parametrize("preserve_context_labels", (False, True))
def test_e1_sequence_preparation_is_exact(
    official_e1: tuple[ModuleType, ModuleType, ModuleType],
    remove_x_tokens: bool,
    preserve_context_labels: bool,
) -> None:
    official_batch, _, _ = official_e1
    tokenizer = get_tokenizer()
    local = E1BatchPreparer(
        data_prep_config=DataPrepConfig(remove_X_tokens=remove_x_tokens),
        tokenizer=tokenizer,
        preserve_context_labels=preserve_context_labels,
    )
    official = official_batch.E1BatchPreparer(
        data_prep_config=official_batch.DataPrepConfig(remove_X_tokens=remove_x_tokens),
        tokenizer=tokenizer,
        preserve_context_labels=preserve_context_labels,
        device=torch.device("cpu"),
    )

    sequences = ("ACD?X", "ACDX,EF?G", "XACD,XXE")
    for sequence in sequences:
        _assert_encoding_equal(
            local.prepare_multiseq(sequence),
            official.prepare_multiseq(sequence),
        )
    _assert_encoding_equal(
        local.get_batch_kwargs(list(sequences)),
        official.get_batch_kwargs(list(sequences)),
    )

    local_single = local.prepare_singleseq("AX?D")
    official_single = official.prepare_singleseq("AX?D")
    _assert_encoding_equal(local_single, official_single)
    assert local_single["input_ids"].data_ptr() == local_single["labels"].data_ptr()
    assert official_single["input_ids"].data_ptr() == official_single["labels"].data_ptr()


@pytest.mark.parametrize(
    ("sequence", "max_sequences", "max_positions"),
    (
        ("AcD", 4, 16),
        ("ABCDE", 4, 4),
        ("A,C,D", 2, 16),
    ),
)
def test_e1_sequence_preparation_errors_are_exact(
    official_e1: tuple[ModuleType, ModuleType, ModuleType],
    sequence: str,
    max_sequences: int,
    max_positions: int,
) -> None:
    official_batch, _, _ = official_e1
    tokenizer = get_tokenizer()
    local = E1BatchPreparer(
        data_prep_config=DataPrepConfig(
            max_num_sequences=max_sequences,
            max_num_positions_within_seq=max_positions,
        ),
        tokenizer=tokenizer,
    )
    official = official_batch.E1BatchPreparer(
        data_prep_config=official_batch.DataPrepConfig(
            max_num_sequences=max_sequences,
            max_num_positions_within_seq=max_positions,
        ),
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError) as local_error:
        local.prepare_multiseq(sequence)
    with pytest.raises(ValueError) as official_error:
        official.prepare_multiseq(sequence)
    assert str(local_error.value) == str(official_error.value)


@pytest.mark.parametrize(
    "sequence_ids",
    (
        torch.tensor([[0, 0, 1, 1, -1], [0, 0, 0, -1, -1]]),
        torch.tensor([[3, -1, 3, 4, -1], [7, 8, 8, 8, 9]]),
        torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]]),
    ),
)
def test_e1_unpadding_metadata_is_exact(
    official_e1: tuple[ModuleType, ModuleType, ModuleType],
    sequence_ids: torch.Tensor,
) -> None:
    # sequence_ids: (b, l)
    _, _, official_flash = official_e1
    candidate = _get_unpad_data(sequence_ids)
    official = official_flash._get_unpad_data(sequence_ids)
    assert torch.equal(candidate[0], official[0])
    assert torch.equal(candidate[1], official[1])
    assert candidate[2] == official[2]


@pytest.mark.parametrize(
    ("q_lengths", "k_lengths"),
    (
        (torch.tensor([64, 128, 65]), torch.tensor([127, 2, 128])),
        (torch.tensor([1, 255, 1]), torch.tensor([129, 128])),
        (torch.tensor([128, 128]), torch.tensor([256])),
    ),
)
def test_e1_block_classification_and_mask_are_exact(
    official_e1: tuple[ModuleType, ModuleType, ModuleType],
    q_lengths: torch.Tensor,
    k_lengths: torch.Tensor,
) -> None:
    # q_lengths: (...), k_lengths: (...)
    _, official_flex, _ = official_e1
    candidate_full, candidate_partial = get_overlapping_blocks(q_lengths, k_lengths)
    official_full, official_partial = official_flex.get_overlapping_blocks(q_lengths, k_lengths)
    assert torch.equal(candidate_full, official_full)
    assert torch.equal(candidate_partial, official_partial)

    candidate_mask = direct_block_mask(q_lengths, k_lengths)
    official_mask = official_flex.direct_block_mask(q_lengths, k_lengths)
    assert candidate_mask.shape == official_mask.shape
    assert torch.equal(candidate_mask.to_dense(), official_mask.to_dense())
    for attribute in (
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
        "q_num_blocks",
        "q_indices",
        "full_q_num_blocks",
        "full_q_indices",
    ):
        candidate_value = getattr(candidate_mask, attribute)
        official_value = getattr(official_mask, attribute)
        if candidate_value is None or official_value is None:
            assert candidate_value is official_value
        else:
            assert torch.equal(candidate_value, official_value), attribute

    q_index = torch.arange(int(q_lengths.sum().item()))[:, None]
    k_index = torch.arange(int(k_lengths.sum().item()))[None, :]
    # zero: (...)
    zero = torch.tensor(0)
    assert torch.equal(
        candidate_mask.mask_mod(zero, zero, q_index, k_index),
        official_mask.mask_mod(zero, zero, q_index, k_index),
    )


def _tiny_config(attn_backend: str = "sdpa") -> E1Config:
    return E1Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_num_sequences=8,
        max_num_positions_within_seq=64,
        max_num_positions_global=256,
        dtype="float32",
        attn_backend=attn_backend,
    )


def test_e1_parameter_initialization_is_exact() -> None:
    model = FAST_E1_ENCODER(_tiny_config()).eval()  # type: ignore[no-untyped-call]

    actual_linear = nn.Linear(9, 7)
    expected_linear = nn.Linear(9, 7)
    torch.manual_seed(41)
    expected_linear.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    expected_linear.bias.data.zero_()
    torch.manual_seed(41)
    model._init_weights(actual_linear)
    assert torch.equal(actual_linear.weight, expected_linear.weight)
    assert torch.equal(actual_linear.bias, expected_linear.bias)

    actual_embedding = nn.Embedding(11, 5, padding_idx=3)
    expected_embedding = nn.Embedding(11, 5, padding_idx=3)
    torch.manual_seed(43)
    expected_embedding.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    expected_embedding.weight.data[3].zero_()
    torch.manual_seed(43)
    model._init_weights(actual_embedding)
    assert torch.equal(actual_embedding.weight, expected_embedding.weight)

    norm = RMSNorm(13)
    norm.weight.data.fill_(7.0)
    model._init_weights(norm)
    assert torch.equal(norm.weight, torch.ones_like(norm.weight))


def test_e1_keeps_flash_flags_disabled() -> None:
    assert E1PreTrainedModel._supports_flash_attn_2 is False
    assert E1PreTrainedModel._supports_flash_attn_3 is False
    assert E1PreTrainedModel._fastplms_attention_implementations == (
        "eager",
        "sdpa",
        "flex_attention",
    )


def _manual_encoder_forward(
    model: FAST_E1_ENCODER,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    sequence_ids = batch["sequence_ids"]
    hidden_states = model.embed_tokens(batch["input_ids"])
    # hidden_states: (..., d)
    hidden_states = hidden_states + model.embed_seq_id(sequence_ids.clamp_min(0))
    first_layer = cast(DecoderLayer, model.layers[0])
    # hidden_states: (..., d)
    hidden_states = hidden_states.to(first_layer.norm_attn_norm.self_attn.q_proj.weight.dtype)

    use_flex = model._attn_backend.value == "flex_attention"
    use_dense = model._attn_backend.value in {"eager", "sdpa"}
    attention_args = AttentionArgs(
        block_causal_block_mask=None,
        within_seq_block_mask=(create_within_seq_block_mask(sequence_ids) if use_flex else None),
        within_seq_mask_4d=(build_within_seq_mask_4d(sequence_ids) if use_dense else None),
        block_causal_mask_4d=(build_block_causal_mask_4d(sequence_ids) if use_dense else None),
    )
    hidden_history = []
    for raw_layer in model.layers:
        layer = cast(DecoderLayer, raw_layer)
        hidden_history.append(hidden_states)
        hidden_states, _, _, _ = layer(
            hidden_states,
            within_seq_position_ids=batch["within_seq_position_ids"],
            global_position_ids=batch["global_position_ids"],
            sequence_ids=sequence_ids,
            attention_args=attention_args,
        )
    hidden_states = model.norm(hidden_states)
    hidden_history.append(hidden_states)
    return hidden_states, tuple(hidden_history)


@pytest.mark.gpu
@pytest.mark.parametrize("attn_backend", ("eager", "sdpa", "flex_attention"))
def test_e1_refactored_forward_is_exact_on_h100(attn_backend: str) -> None:
    assert torch.cuda.is_available(), "E1 BF16 forward parity requires CUDA"
    torch.manual_seed(47)
    model = (
        FAST_E1_ENCODER(_tiny_config(attn_backend))
        .eval()
        .to(  # type: ignore[no-untyped-call]
            "cuda",
            dtype=torch.bfloat16,
        )
    )
    prepared = E1BatchPreparer(tokenizer=get_tokenizer()).get_batch_kwargs(
        ["ACDEFG", "ACD,EFGH"],
        device=torch.device("cuda"),
    )
    batch: dict[str, torch.Tensor] = {}
    for key in (
        "input_ids",
        "within_seq_position_ids",
        "global_position_ids",
        "sequence_ids",
    ):
        value = prepared[key]
        assert isinstance(value, torch.Tensor)
        batch[key] = value
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    with torch.no_grad():
        expected_last, expected_history = _manual_encoder_forward(model, batch)
        output = model(**batch, output_hidden_states=True)
        combined_embeddings = model.embed_tokens(batch["input_ids"])
        # combined_embeddings: (...)
        combined_embeddings = combined_embeddings + model.embed_seq_id(
            batch["sequence_ids"].clamp_min(0)
        )
        soft_output = model(
            inputs_embeds=combined_embeddings,
            within_seq_position_ids=batch["within_seq_position_ids"],
            global_position_ids=batch["global_position_ids"],
            sequence_ids=batch["sequence_ids"],
            output_hidden_states=True,
        )

    assert torch.equal(output.last_hidden_state, expected_last)
    assert output.hidden_states is not None
    assert len(output.hidden_states) == len(expected_history)
    for actual, expected in zip(output.hidden_states, expected_history, strict=True):
        assert torch.equal(actual, expected)
    assert torch.equal(soft_output.last_hidden_state, output.last_hidden_state)
    assert model.state_dict().keys() == before.keys()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name
