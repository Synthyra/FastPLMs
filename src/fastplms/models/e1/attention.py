"""E1 attention mask, unpadding, FlexAttention, and kernel adapters."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
from torch.nn.attention.flex_attention import _create_sparse_block_from_block_mask

from fastplms.attention import (
    BlockMask,
    _ensure_flash_kernels_loaded,
    _get_flex_attention_fn,
    _kernels_flash_forward,
    _kernels_flash_varlen_forward,
    create_block_mask,
    flex_attention,
    index_first_axis,
    pad_input,
)


@torch.compiler.disable
def create_block_causal_mask_optimized(sequence_ids: torch.Tensor) -> BlockMask:
    # Assumes sequence_ids is sorted in increasing order for each batch item, except for
    # the -1 values, which are used to indicate the padding tokens.
    def document_mask(b, h, q_idx, kv_idx):  # type: ignore[no-untyped-def]
        return (
            (sequence_ids[b, q_idx] >= sequence_ids[b, kv_idx])
            & (sequence_ids[b, q_idx] != -1)
            & (sequence_ids[b, kv_idx] != -1)
        )

    batch_size, seqlen = sequence_ids.shape
    return create_block_mask(
        document_mask, batch_size, 1, seqlen, seqlen, device=sequence_ids.device
    )


@torch.compiler.disable
def create_within_seq_block_mask(sequence_ids: torch.Tensor) -> BlockMask:
    def document_mask(b, h, q_idx, kv_idx):  # type: ignore[no-untyped-def]
        return (
            (sequence_ids[b, q_idx] == sequence_ids[b, kv_idx])
            & (sequence_ids[b, q_idx] != -1)
            & (sequence_ids[b, kv_idx] != -1)
        )

    batch_size, seqlen = sequence_ids.shape
    return create_block_mask(
        document_mask, batch_size, 1, seqlen, seqlen, device=sequence_ids.device
    )


def build_within_seq_mask_4d(sequence_ids: torch.Tensor) -> torch.Tensor:
    not_pad = sequence_ids != -1
    same_seq = sequence_ids.unsqueeze(-1) == sequence_ids.unsqueeze(-2)
    valid = not_pad.unsqueeze(-1) & not_pad.unsqueeze(-2)
    return (same_seq & valid).unsqueeze(1)


def build_block_causal_mask_4d(sequence_ids: torch.Tensor) -> torch.Tensor:
    not_pad = sequence_ids != -1
    causal = sequence_ids.unsqueeze(-1) >= sequence_ids.unsqueeze(-2)
    valid = not_pad.unsqueeze(-1) & not_pad.unsqueeze(-2)
    return (causal & valid).unsqueeze(1)


def flex_attention_func(
    query_states: torch.Tensor,  # Q has shape (b, l, h, d).
    key_states: torch.Tensor,  # K has shape (b, l, h_kv, d).
    value_states: torch.Tensor,  # V has shape (b, l, h_kv, d).
    score_mod: Callable | None = None,
    block_mask: BlockMask | None = None,
    sequence_lengths: tuple[int, ...] | None = None,
    mask_semantics: str = "within_sequence",
) -> torch.Tensor:
    assert flex_attention is not None, "Flex Attention is not available in this environment"
    assert score_mod is None, "Score mod is not supported yet"
    query_states = query_states.transpose(1, 2).contiguous()  # (bs, nh, seqlen, hs)
    key_states = key_states.transpose(1, 2).contiguous()  # (bs, nkv, seqlen, hs)
    value_states = value_states.transpose(1, 2).contiguous()  # (bs, nkv, seqlen, hs)

    fn = _get_flex_attention_fn(
        device=query_states.device,
        dtype=query_states.dtype,
        shape=tuple(query_states.shape),
        sequence_lengths=sequence_lengths,
        mask_semantics=mask_semantics,
    )
    outputs = fn(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        score_mod=score_mod,
        enable_gqa=query_states.shape[1] != key_states.shape[1],  # if nkv != nh
    )

    outputs = outputs.transpose(1, 2)  # (bs, seqlen, nh, hs)
    return outputs


def kernels_flash_attention_func(
    query_states: torch.Tensor,  # (bs, seqlen, nh, hs)
    key_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    value_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
    causal: bool = False,
    implementation: str = "flash_attention_3",
) -> torch.Tensor:  # (bs, seqlen, nh, hs)
    _ensure_flash_kernels_loaded(implementation)

    if not causal:
        batch_size, q_len = query_states.shape[0], query_states.shape[1]
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        ) = _unpad_input(query_states, key_states, value_states, q_sequence_ids, k_sequence_ids)

        attn_output_unpad = _kernels_flash_varlen_forward(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_in_batch_q=max_seqlen_in_batch_q,
            max_seqlen_in_batch_k=max_seqlen_in_batch_k,
            causal=False,
            implementation=implementation,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, q_len)

    else:
        attn_output = _kernels_flash_forward(
            query_states, key_states, value_states, causal=True, implementation=implementation
        )

    return attn_output


def block_min_max_seq_ids(
    sequence_lengths: torch.Tensor,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each physical attention block to its first and last sequence."""

    total_tokens = sequence_lengths.sum()
    block_count = int(
        torch.div(
            total_tokens + block_size - 1,
            block_size,
            rounding_mode="floor",
        ).item()
    )
    padded_tokens = block_count * block_size - total_tokens
    lengths_with_tail = torch.cat(
        (sequence_lengths, padded_tokens.to(sequence_lengths).reshape(1)),
    )
    sequence_ends = lengths_with_tail.to(torch.long).cumsum(dim=0)
    block_starts = torch.arange(
        start=0,
        end=block_count * block_size,
        step=block_size,
        dtype=torch.long,
        device=sequence_lengths.device,
    )
    block_last_tokens = block_starts + block_size - 1
    first_sequence = torch.searchsorted(sequence_ends, block_starts, right=True)
    last_sequence = torch.searchsorted(sequence_ends, block_last_tokens, right=True)
    return first_sequence, last_sequence


def get_overlapping_blocks(
    q_lengths: torch.Tensor,
    k_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify query/key block pairs as full, partial, or disjoint."""

    q_first, q_last = block_min_max_seq_ids(q_lengths)
    k_first, k_last = block_min_max_seq_ids(k_lengths)
    intersection_start = torch.maximum(q_first[:, None], k_first[None, :])
    intersection_end = torch.minimum(q_last[:, None], k_last[None, :])
    intersects = intersection_start <= intersection_end
    both_blocks_are_single_sequence = (q_first == q_last)[:, None] & (k_first == k_last)[None, :]
    full_blocks = intersects & both_blocks_are_single_sequence
    return full_blocks, intersects & ~both_blocks_are_single_sequence


def _document_ids(sequence_lengths: torch.Tensor) -> torch.Tensor:
    sequence_numbers = torch.arange(
        sequence_lengths.numel(),
        device=sequence_lengths.device,
        dtype=torch.long,
    )
    return sequence_numbers.repeat_interleave(sequence_lengths.to(torch.long))


@torch.compiler.disable
def direct_block_mask(q_lengths: torch.Tensor, k_lengths: torch.Tensor) -> BlockMask:
    """Build a packed-sequence mask from preclassified sparse blocks."""

    full, partial = get_overlapping_blocks(q_lengths, k_lengths)
    q_document = _document_ids(q_lengths)
    k_document = _document_ids(k_lengths)

    def same_document(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        q_index: torch.Tensor,
        k_index: torch.Tensor,
    ) -> torch.Tensor:
        return q_document[q_index].eq(k_document[k_index])

    return _create_sparse_block_from_block_mask(
        (partial[None, None], full[None, None]),
        same_document,
        seq_lengths=(q_document.numel(), k_document.numel()),
        Q_BLOCK_SIZE=128,
        KV_BLOCK_SIZE=128,
    )


@torch.compiler.disable
def doc_id_mask(q_lengths: torch.Tensor, k_lengths: torch.Tensor) -> BlockMask:
    q_document = _document_ids(q_lengths)
    k_document = _document_ids(k_lengths)

    def same_document(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        q_index: torch.Tensor,
        k_index: torch.Tensor,
    ) -> torch.Tensor:
        return q_document[q_index].eq(k_document[k_index])

    return create_block_mask(
        same_document,
        1,
        1,
        q_document.numel(),
        k_document.numel(),
        BLOCK_SIZE=128,
        device=q_lengths.device,
    )


def varlen_flex_attention_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size, q_len = query_states.shape[0], query_states.shape[1]
    (
        query_states,
        key_states,
        value_states,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (_max_seqlen_in_batch_q, _max_seqlen_in_batch_k),
    ) = _unpad_input(query_states, key_states, value_states, q_sequence_ids, k_sequence_ids)

    query_states = query_states.unsqueeze(0).transpose(1, 2).contiguous()
    key_states = key_states.unsqueeze(0).transpose(1, 2).contiguous()
    value_states = value_states.unsqueeze(0).transpose(1, 2).contiguous()

    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    block_mask = block_mask_creator(seqlens_q, seqlens_k)

    packed_lengths = (
        *(int(length) for length in seqlens_q.tolist()),
        -1,
        *(int(length) for length in seqlens_k.tolist()),
    )
    fn = _get_flex_attention_fn(
        device=query_states.device,
        dtype=query_states.dtype,
        shape=tuple(query_states.shape) + tuple(key_states.shape),
        sequence_lengths=packed_lengths,
        mask_semantics="packed_document_equality",
    )
    attn_output_unpad = fn(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=query_states.shape[1] != key_states.shape[1],
    )

    attn_output = pad_input(
        attn_output_unpad.transpose(1, 2).squeeze(0), indices_q, batch_size, q_len
    )

    return attn_output


def _get_unpad_data(sequence_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return packed indices and run lengths for the non-padding sequence IDs."""

    flat_ids = sequence_ids.reshape(-1)
    non_pad_indices = torch.where(flat_ids.ne(-1))[0]
    if non_pad_indices.numel() == 0:
        raise ValueError("Packed attention requires at least one non-padding token.")

    valid_ids = flat_ids.index_select(0, non_pad_indices)
    row_ids = torch.div(
        non_pad_indices,
        sequence_ids.shape[1],
        rounding_mode="floor",
    )
    run_starts = torch.ones_like(valid_ids, dtype=torch.bool)
    run_starts[1:] = (valid_ids[1:] != valid_ids[:-1]) | (row_ids[1:] != row_ids[:-1])
    start_indices = torch.where(run_starts)[0]
    end_indices = torch.cat((start_indices[1:], start_indices.new_tensor([valid_ids.numel()])))
    sequence_lengths = end_indices - start_indices
    cumulative_lengths = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=sequence_ids.device),
            sequence_lengths.cumsum(dim=0, dtype=torch.int32),
        ),
    )
    return non_pad_indices, cumulative_lengths, int(sequence_lengths.max().item())


def _unpad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    tuple[int, int],
]:
    batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
    query_length, num_q_heads = query_layer.shape[1], query_layer.shape[2]
    assert query_layer.shape[:2] == q_sequence_ids.shape, (
        "Shape mismatch between query layer and query sequence ids: "
        f"{query_layer.shape[:2]} != {q_sequence_ids.shape}"
    )
    assert key_layer.shape[:2] == k_sequence_ids.shape, (
        "Shape mismatch between key layer and key sequence ids: "
        f"{key_layer.shape[:2]} != {k_sequence_ids.shape}"
    )
    assert query_length <= kv_seq_len, (
        "Query length should be less than or equal to KV sequence length: "
        f"{query_length} <= {kv_seq_len}"
    )

    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(k_sequence_ids)

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
    )

    if torch.equal(q_sequence_ids, k_sequence_ids):
        indices_q = indices_k
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
    else:
        indices_q, cu_seqlens_q, max_seqlen_in_batch_q = _get_unpad_data(q_sequence_ids)

    query_layer = index_first_axis(
        query_layer.reshape(batch_size * query_length, num_q_heads, head_dim), indices_q
    )

    assert cu_seqlens_q.shape == cu_seqlens_k.shape, (
        "Query and KV should have the same number of sequences: "
        f"{cu_seqlens_q.shape} != {cu_seqlens_k.shape}"
    )

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


block_mask_creator = direct_block_mask if os.getenv("FAST_BLOCK_MASK", "1") == "1" else doc_id_mask
