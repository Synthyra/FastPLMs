"""FlashAttention varlen metadata is built once per forward and changes no value.

A reference varlen kernel stands in for the CUDA kernel, so these checks run on
CPU. They compare the per-layer metadata path with the once-per-forward path.
"""

from __future__ import annotations

import itertools
import pytest
import torch

from collections.abc import Callable
from types import SimpleNamespace
from torch.nn import functional as F

from fastplms.attention import AttentionBackend, FlashPaddingLayout, _core, get_flash_padding_layout
from fastplms.models.dplm import modeling_dplm as dplm_module
from fastplms.models.esm2 import modeling_fastesm as esm2_module
from fastplms.models.esm_plusplus import modeling_esm_plusplus as esmpp_module


NUM_LAYERS = 3
HIDDEN_SIZE = 8
# Row 1 has an interior gap, which right-padding alone would not exercise.
PADDING_MASK = torch.tensor(
    [
        [True, True, True, True, True, True],
        [True, True, False, True, False, False],
        [True, False, False, False, False, False],
    ]
)  # (b=3, l=6)


def _reference_varlen_attention(**kwargs: object) -> torch.Tensor:
    """Attend within each packed row, as the FlashAttention varlen kernels do."""
    queries, keys, values = kwargs["q"], kwargs["k"], kwargs["v"]  # each (t, h, d_h)
    row_bounds = kwargs["cu_seqlens_q"].tolist()  # b + 1 offsets into the t packed tokens
    assert torch.equal(kwargs["cu_seqlens_q"], kwargs["cu_seqlens_k"])
    longest_row = max(stop - start for start, stop in itertools.pairwise(row_bounds))
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == longest_row
    rows = []
    for start, stop in itertools.pairwise(row_bounds):
        attended = F.scaled_dot_product_attention(  # (h, l_row, d_h)
            queries[start:stop].transpose(0, 1),
            keys[start:stop].transpose(0, 1),
            values[start:stop].transpose(0, 1),
            scale=kwargs["softmax_scale"],
            is_causal=kwargs["causal"],
        )
        rows.append(attended.transpose(0, 1))  # (l_row, h, d_h)
    return torch.cat(rows)  # (t, h, d_h)


@pytest.fixture
def reference_flash_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    def dense_is_not_expected(**_kwargs: object) -> torch.Tensor:
        raise AssertionError("a padded call must use the varlen kernel")

    kernel = SimpleNamespace(
        flash_attn_func=dense_is_not_expected,
        flash_attn_varlen_func=_reference_varlen_attention,
    )
    monkeypatch.setattr(
        _core, "_ensure_flash_kernels_loaded", lambda _implementation: (kernel, "flash_attn2")
    )
    monkeypatch.setattr(
        _core, "_validate_kernels_flash_device", lambda query, _key, _value, _name: query.device
    )
    monkeypatch.setattr(
        _core, "_validate_kernels_flash_dtype", lambda query, _key, _value, _name: query.dtype
    )


def _count_layout_builds(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    built_from: list[torch.Tensor] = []
    build_layout = _core._flash_padding_layout

    def counting_build(attention_mask_2d: torch.Tensor) -> FlashPaddingLayout:
        built_from.append(attention_mask_2d)
        return build_layout(attention_mask_2d)

    monkeypatch.setattr(_core, "_flash_padding_layout", counting_build)
    return built_from


def _esm2_encoder() -> tuple[torch.nn.Module, Callable[..., torch.Tensor]]:
    encoder = esm2_module.EsmEncoder(
        esm2_module.FastEsmConfig(
            vocab_size=16,
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=NUM_LAYERS,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            pad_token_id=1,
            mask_token_id=5,
            position_embedding_type="rotary",
            attn_backend="flash_attention_2",
        )
    ).eval()
    return encoder, lambda hidden_states, mask: encoder(hidden_states, mask).last_hidden_state


def _dplm_encoder() -> tuple[torch.nn.Module, Callable[..., torch.Tensor]]:
    encoder = dplm_module.ModifiedEsmEncoder(
        dplm_module.DPLMConfig(
            vocab_size=16,
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=NUM_LAYERS,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            position_embedding_type="rotary",
            attn_backend="flash_attention_3",
        )
    ).eval()
    return encoder, lambda hidden_states, mask: encoder(hidden_states, mask).last_hidden_state


def _esmpp_stack() -> tuple[torch.nn.Module, Callable[..., torch.Tensor]]:
    stack = esmpp_module.TransformerStack(
        d_model=HIDDEN_SIZE, n_heads=2, n_layers=NUM_LAYERS, attn_backend="flash_attention_2"
    ).eval()
    return stack, lambda hidden_states, mask: stack(hidden_states, mask).last_hidden_state


_FAMILIES = {
    "esm2": (_esm2_encoder, esm2_module),
    "dplm": (_dplm_encoder, dplm_module),
    "esm_plusplus": (_esmpp_stack, esmpp_module),
}


def test_layout_records_the_metadata_the_varlen_kernel_expects() -> None:
    layout = get_flash_padding_layout(AttentionBackend.FLASH_ATTENTION_2, PADDING_MASK)

    assert layout is not None
    assert layout.attention_mask_2d is PADDING_MASK
    assert layout.indices.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 9, 12]
    assert layout.cu_seqlens.tolist() == [0, 6, 9, 10]
    assert layout.cu_seqlens.dtype == torch.int32
    assert layout.max_seqlen == 6


@pytest.mark.parametrize("backend", ("eager", "sdpa", "flex_attention"))
def test_only_padded_flash_calls_get_a_layout(backend: str) -> None:
    assert get_flash_padding_layout(AttentionBackend(backend), PADDING_MASK) is None
    assert get_flash_padding_layout(AttentionBackend.FLASH_ATTENTION_3, None) is None


@pytest.mark.usefixtures("reference_flash_kernel")
def test_shared_layout_matches_per_call_metadata_bitwise_with_gradients() -> None:
    generator = torch.Generator().manual_seed(11)
    layout = get_flash_padding_layout(AttentionBackend.FLASH_ATTENTION_2, PADDING_MASK)
    outputs, input_gradients = [], []
    for padding_layout in (None, layout):
        generator.manual_seed(11)
        states = [  # Q, K, V: each (b=3, l=6, h=2, d_h=4)
            torch.randn(3, 6, 2, 4, generator=generator).requires_grad_() for _ in range(3)
        ]
        output = _core.kernels_flash_attention_func(  # (b, l, h, d_h)
            *states,
            attention_mask_2d=PADDING_MASK,
            softmax_scale=0.5,
            implementation="flash_attention_2",
            padding_layout=padding_layout,
        )
        # Weight every position, so a gradient that reached padding would show.
        (output * torch.arange(output.numel()).reshape(output.shape)).sum().backward()
        outputs.append(output.detach())
        input_gradients.append([state.grad for state in states])

    assert torch.equal(outputs[0], outputs[1])
    for per_call_gradient, shared_gradient in zip(*input_gradients, strict=True):
        assert torch.equal(per_call_gradient, shared_gradient)
        assert torch.count_nonzero(shared_gradient[~PADDING_MASK]) == 0
    # The scatter into zeros already clears padding; no separate fill is needed.
    assert torch.count_nonzero(outputs[1][~PADDING_MASK]) == 0
    assert torch.count_nonzero(outputs[1][PADDING_MASK]) > 0


@pytest.mark.usefixtures("reference_flash_kernel")
def test_layout_is_rejected_when_it_cannot_belong_to_the_call() -> None:
    states = torch.zeros(3, 6, 2, 4)  # (b, l, h, d_h)
    layout = get_flash_padding_layout(AttentionBackend.FLASH_ATTENTION_2, PADDING_MASK)
    other_layout = get_flash_padding_layout(
        AttentionBackend.FLASH_ATTENTION_2, PADDING_MASK[:, :5].clone()
    )

    with pytest.raises(ValueError, match="requires the mask it was built from"):
        _core.kernels_flash_attention_func(
            states, states, states, implementation="flash_attention_2", padding_layout=layout
        )
    with pytest.raises(ValueError, match=r"built for mask shape \(3, 5\).*uses \(3, 6\)"):
        _core.kernels_flash_attention_func(
            states,
            states,
            states,
            attention_mask_2d=PADDING_MASK,
            implementation="flash_attention_2",
            padding_layout=other_layout,
        )


@pytest.mark.usefixtures("reference_flash_kernel")
@pytest.mark.parametrize("family", tuple(_FAMILIES))
def test_encoder_builds_the_layout_once_and_matches_per_layer_metadata(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_encoder, family_module = _FAMILIES[family]
    torch.manual_seed(5)
    _, run_encoder = build_encoder()
    hidden_states = torch.randn(3, 6, HIDDEN_SIZE)  # (b, l, d)
    built_from = _count_layout_builds(monkeypatch)

    shared_output = run_encoder(hidden_states, PADDING_MASK)  # (b, l, d)
    assert len(built_from) == 1

    # Without a shared layout every layer derives the metadata again.
    built_from.clear()
    monkeypatch.setattr(family_module, "get_flash_padding_layout", lambda _backend, _mask: None)
    per_layer_output = run_encoder(hidden_states, PADDING_MASK)  # (b, l, d)
    assert len(built_from) == NUM_LAYERS

    assert torch.equal(shared_output, per_layer_output)
    assert torch.isfinite(shared_output).all()


@pytest.mark.usefixtures("reference_flash_kernel")
@pytest.mark.parametrize("family", tuple(_FAMILIES))
def test_unpadded_and_attention_weight_calls_build_no_layout(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_encoder, _ = _FAMILIES[family]
    encoder, _ = build_encoder()
    hidden_states = torch.randn(3, 6, HIDDEN_SIZE)  # (b, l, d)
    built_from = _count_layout_builds(monkeypatch)
    monkeypatch.setattr(
        _core,
        "_kernels_flash_forward",
        lambda **kwargs: kwargs["query_states"],  # (b, l, h, d_h)
    )

    encoder(hidden_states, None)
    with pytest.warns(RuntimeWarning, match="output_attentions=True requires"):
        encoder(hidden_states, PADDING_MASK, output_attentions=True)

    assert built_from == []


def test_gather_and_scatter_without_gradients_match_the_autograd_wrappers() -> None:
    generator = torch.Generator().manual_seed(4)
    layout = get_flash_padding_layout(AttentionBackend.FLASH_ATTENTION_2, PADDING_MASK)
    assert layout is not None
    for dtype in (torch.float32, torch.bfloat16):
        states = torch.randn(18, 2, 4, generator=generator).to(dtype)  # (b * l, h, d_h)
        tracked = states.clone().requires_grad_()  # (18, 2, 4)

        selected = _core._select_first_axis(states, layout.indices)  # (t, h, d_h)
        tracked_selected = _core._select_first_axis(tracked, layout.indices)  # (t, h, d_h)
        restored = _core.pad_input(selected, layout.indices, 3, 6)  # (b, l, h, d_h)
        tracked_restored = _core.pad_input(tracked_selected, layout.indices, 3, 6)

        assert selected.grad_fn is None and tracked_selected.grad_fn is not None
        assert torch.equal(selected, tracked_selected)
        assert torch.equal(restored, tracked_restored)
        assert restored.shape == (3, 6, 2, 4) and restored.is_contiguous()
        assert restored.stride() == tracked_restored.stride()
        assert torch.count_nonzero(restored[~PADDING_MASK]) == 0
