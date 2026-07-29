"""E1 attention mask, unpadding, FlexAttention, and kernel adapters."""

from __future__ import annotations

import os
import torch
from collections.abc import Callable
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
    # sequence_ids: (b, l)
    if create_block_mask is None:
        raise RuntimeError("Flex Attention block-mask creation is unavailable in this environment.")
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
    # sequence_ids: (b, l)
    if create_block_mask is None:
        raise RuntimeError("Flex Attention block-mask creation is unavailable in this environment.")
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
    # sequence_ids: (b, l)
    not_pad = sequence_ids != -1  # (b, l)
    same_seq = sequence_ids.unsqueeze(-1) == sequence_ids.unsqueeze(-2)  # (b, l, l)
    valid = not_pad.unsqueeze(-1) & not_pad.unsqueeze(-2)  # (b, l, l)
    return (same_seq & valid).unsqueeze(1)  # (b, 1, l, l)


def build_block_causal_mask_4d(sequence_ids: torch.Tensor) -> torch.Tensor:
    # sequence_ids: (b, l)
    not_pad = sequence_ids != -1  # (b, l)
    causal = sequence_ids.unsqueeze(-1) >= sequence_ids.unsqueeze(-2)  # (b, l, l)
    valid = not_pad.unsqueeze(-1) & not_pad.unsqueeze(-2)  # (b, l, l)
    return (causal & valid).unsqueeze(1)  # (b, 1, l, l)


def flex_attention_func(
    query_states: torch.Tensor,  # Q has shape (b, l, h, d).
    key_states: torch.Tensor,  # K has shape (b, l, h_kv, d).
    value_states: torch.Tensor,  # V has shape (b, l, h_kv, d).
    score_mod: Callable | None = None,
    block_mask: BlockMask | None = None,
    sequence_lengths: tuple[int, ...] | None = None,
    mask_semantics: str = "within_sequence",
) -> torch.Tensor:
    if flex_attention is None:
        raise RuntimeError("Flex Attention is not available in this environment.")
    if score_mod is not None:
        raise NotImplementedError("E1 Flex Attention does not support score_mod.")
    query_states = query_states.transpose(1, 2).contiguous()  # (b, h, l, d)
    key_states = key_states.transpose(1, 2).contiguous()  # (b, h_kv, l, d)
    value_states = value_states.transpose(1, 2).contiguous()  # (b, h_kv, l, d)

    fn = _get_flex_attention_fn(
        device=query_states.device,
        dtype=query_states.dtype,
        shape=tuple(query_states.shape),
        sequence_lengths=sequence_lengths,
        mask_semantics=mask_semantics,
    )
    if fn is None:
        raise RuntimeError("Flex Attention is not available in this environment.")
    outputs = fn(  # (b, h, l, d)
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        score_mod=score_mod,
        enable_gqa=query_states.shape[1] != key_states.shape[1],  # if nkv != nh
    )

    outputs = outputs.transpose(1, 2)  # (b, l, h, d)
    return outputs  # (b, l, h, d)


def kernels_flash_attention_func(
    query_states: torch.Tensor,  # (b, l_q, h, d)
    key_states: torch.Tensor,  # (b, l_kv, h_kv, d)
    value_states: torch.Tensor,  # (b, l_kv, h_kv, d)
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
    causal: bool = False,
    implementation: str = "flash_attention_3",
) -> torch.Tensor:  # (b, l_q, h, d)
    # q_sequence_ids: (b, l_q); k_sequence_ids: (b, l_kv)
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
        ) = _unpad_input(  # Q: (t_q, h, d); K/V: (t_kv, h_kv, d)
            query_states,
            key_states,
            value_states,
            q_sequence_ids,
            k_sequence_ids,
        )

        attn_output_unpad = _kernels_flash_varlen_forward(  # (t_q, h, d)
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
        attn_output = pad_input(  # (b, l_q, h, d)
            attn_output_unpad,
            indices_q,
            batch_size,
            q_len,
        )

    else:
        attn_output = _kernels_flash_forward(  # (b, l_q, h, d)
            query_states, key_states, value_states, causal=True, implementation=implementation
        )

    return attn_output  # (b, l_q, h, d)


def block_min_max_seq_ids(
    sequence_lengths: torch.Tensor,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each physical attention block to its first and last sequence."""

    # sequence_lengths: (n,)
    total_tokens = sequence_lengths.sum()  # ()
    block_count = int(
        torch.div(
            total_tokens + block_size - 1,
            block_size,
            rounding_mode="floor",
        ).item()
    )
    padded_tokens = block_count * block_size - total_tokens  # ()
    lengths_with_tail = torch.cat(  # (n + 1,)
        (sequence_lengths, padded_tokens.to(sequence_lengths).reshape(1)),
    )
    sequence_ends = lengths_with_tail.to(torch.long).cumsum(dim=0)  # (n + 1,)
    block_starts = torch.arange(  # (n_blocks,)
        start=0,
        end=block_count * block_size,
        step=block_size,
        dtype=torch.long,
        device=sequence_lengths.device,
    )
    block_last_tokens = block_starts + block_size - 1  # (n_blocks,)
    first_sequence = torch.searchsorted(sequence_ends, block_starts, right=True)  # (n_blocks,)
    last_sequence = torch.searchsorted(sequence_ends, block_last_tokens, right=True)  # (n_blocks,)
    return first_sequence, last_sequence  # (n_blocks,), (n_blocks,)


def get_overlapping_blocks(
    q_lengths: torch.Tensor,
    k_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify query/key block pairs as full, partial, or disjoint."""

    # q_lengths: (n_q,); k_lengths: (n_k,)
    q_first, q_last = block_min_max_seq_ids(q_lengths)  # (q_blocks,), (q_blocks,)
    k_first, k_last = block_min_max_seq_ids(k_lengths)  # (k_blocks,), (k_blocks,)
    intersection_start = torch.maximum(  # (q_blocks, k_blocks)
        q_first[:, None],
        k_first[None, :],
    )
    intersection_end = torch.minimum(  # (q_blocks, k_blocks)
        q_last[:, None],
        k_last[None, :],
    )
    intersects = intersection_start <= intersection_end  # (q_blocks, k_blocks)
    both_blocks_are_single_sequence = (  # (q_blocks, k_blocks)
        (q_first == q_last)[:, None] & (k_first == k_last)[None, :]
    )
    full_blocks = intersects & both_blocks_are_single_sequence  # (q_blocks, k_blocks)
    return full_blocks, intersects & ~both_blocks_are_single_sequence  # both (q_blocks, k_blocks)


def _document_ids(sequence_lengths: torch.Tensor) -> torch.Tensor:
    # sequence_lengths: (n,)
    sequence_numbers = torch.arange(  # (n,)
        sequence_lengths.numel(),
        device=sequence_lengths.device,
        dtype=torch.long,
    )
    return sequence_numbers.repeat_interleave(sequence_lengths.to(torch.long))  # (t,)


@torch.compiler.disable
def direct_block_mask(q_lengths: torch.Tensor, k_lengths: torch.Tensor) -> BlockMask:
    """Build a packed-sequence mask from preclassified sparse blocks."""

    full, partial = get_overlapping_blocks(q_lengths, k_lengths)  # both (q_blocks, k_blocks)
    q_document = _document_ids(q_lengths)  # (t_q,)
    k_document = _document_ids(k_lengths)  # (t_k,)

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
    if create_block_mask is None:
        raise RuntimeError("Flex Attention block-mask creation is unavailable in this environment.")
    q_document = _document_ids(q_lengths)  # (t_q,)
    k_document = _document_ids(k_lengths)  # (t_k,)

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
    # query_states: (b, l_q, h, d); key_states, value_states: (b, l_kv, h_kv, d)
    # q_sequence_ids: (b, l_q); k_sequence_ids: (b, l_kv)
    if flex_attention is None:
        raise RuntimeError("Flex Attention is not available in this environment.")
    batch_size, q_len = query_states.shape[0], query_states.shape[1]
    (
        query_states,
        key_states,
        value_states,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (_max_seqlen_in_batch_q, _max_seqlen_in_batch_k),
    ) = _unpad_input(  # Q: (t_q, h, d); K/V: (t_kv, h_kv, d)
        query_states,
        key_states,
        value_states,
        q_sequence_ids,
        k_sequence_ids,
    )

    query_states = query_states.unsqueeze(0).transpose(1, 2).contiguous()  # (1, h, t_q, d)
    key_states = key_states.unsqueeze(0).transpose(1, 2).contiguous()  # (1, h_kv, t_kv, d)
    value_states = value_states.unsqueeze(0).transpose(1, 2).contiguous()  # (1, h_kv, t_kv, d)

    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]  # (n,)
    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]  # (n,)
    block_mask = block_mask_creator(seqlens_q, seqlens_k)

    packed_lengths = _packed_lengths_cache_key(seqlens_q, seqlens_k)
    fn = _get_flex_attention_fn(
        device=query_states.device,
        dtype=query_states.dtype,
        shape=tuple(query_states.shape) + tuple(key_states.shape),
        sequence_lengths=packed_lengths,
        mask_semantics="packed_document_equality",
    )
    if fn is None:
        raise RuntimeError("Flex Attention is not available in this environment.")
    attn_output_unpad = fn(  # (1, h, t_q, d)
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=query_states.shape[1] != key_states.shape[1],
    )

    attn_output = pad_input(  # (b, l_q, h, d)
        attn_output_unpad.transpose(1, 2).squeeze(0), indices_q, batch_size, q_len
    )

    return attn_output  # (b, l_q, h, d)


@torch.compiler.disable
def _packed_lengths_cache_key(
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
) -> tuple[int, ...]:
    """Convert packed lengths to an immutable Flex cache key eagerly."""

    return (
        *(int(length) for length in query_lengths.tolist()),
        -1,
        *(int(length) for length in key_lengths.tolist()),
    )


@torch.compiler.disable
def _get_unpad_data(sequence_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return packed indices and run lengths for the non-padding sequence IDs."""

    # sequence_ids: (b, l)
    flat_ids = sequence_ids.reshape(-1)  # (b * l,)
    non_pad_indices = torch.where(flat_ids.ne(-1))[0]  # (t,)
    if non_pad_indices.numel() == 0:
        raise ValueError("Packed attention requires at least one non-padding token.")

    valid_ids = flat_ids.index_select(0, non_pad_indices)  # (t,)
    row_ids = torch.div(  # (t,)
        non_pad_indices,
        sequence_ids.shape[1],
        rounding_mode="floor",
    )
    run_starts = torch.ones_like(valid_ids, dtype=torch.bool)  # (t,)
    run_starts[1:] = (valid_ids[1:] != valid_ids[:-1]) | (row_ids[1:] != row_ids[:-1])
    start_indices = torch.where(run_starts)[0]  # (n,)
    end_indices = torch.cat(  # (n,)
        (start_indices[1:], start_indices.new_tensor([valid_ids.numel()]))
    )
    sequence_lengths = end_indices - start_indices  # (n,)
    cumulative_lengths = torch.cat(  # (n + 1,)
        (
            torch.zeros(1, dtype=torch.int32, device=sequence_ids.device),
            sequence_lengths.cumsum(dim=0, dtype=torch.int32),
        ),
    )
    return non_pad_indices, cumulative_lengths, int(  # (t,), (n + 1,), scalar
        sequence_lengths.max().item()
    )


@torch.compiler.disable
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
    # query_layer: (b, l_q, h, d); key_layer, value_layer: (b, l_kv, h_kv, d)
    # q_sequence_ids: (b, l_q); k_sequence_ids: (b, l_kv)
    for name, layer in (
        ("query_layer", query_layer),
        ("key_layer", key_layer),
        ("value_layer", value_layer),
    ):
        if layer.ndim != 4:
            raise ValueError(
                f"{name} must have shape (batch, sequence, heads, head_dim); "
                f"got {tuple(layer.shape)}."
            )
    if value_layer.shape != key_layer.shape:
        raise ValueError(
            "key_layer and value_layer must have identical shapes; "
            f"got {tuple(key_layer.shape)} and {tuple(value_layer.shape)}."
        )
    if query_layer.shape[0] != key_layer.shape[0]:
        raise ValueError(
            "Query and KV batch sizes must match; "
            f"got {query_layer.shape[0]} and {key_layer.shape[0]}."
        )
    if query_layer.shape[-1] != key_layer.shape[-1]:
        raise ValueError(
            "Query and KV head dimensions must match; "
            f"got {query_layer.shape[-1]} and {key_layer.shape[-1]}."
        )
    batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
    query_length, num_q_heads = query_layer.shape[1], query_layer.shape[2]
    if query_layer.shape[:2] != q_sequence_ids.shape:
        raise ValueError(
            "Shape mismatch between query layer and query sequence ids: "
            f"{query_layer.shape[:2]} != {q_sequence_ids.shape}"
        )
    if key_layer.shape[:2] != k_sequence_ids.shape:
        raise ValueError(
            "Shape mismatch between key layer and key sequence ids: "
            f"{key_layer.shape[:2]} != {k_sequence_ids.shape}"
        )
    if query_length > kv_seq_len:
        raise ValueError(
            "Query length must be less than or equal to KV sequence length: "
            f"{query_length} > {kv_seq_len}"
        )

    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(  # (t_kv,), (n + 1,), scalar
        k_sequence_ids
    )

    key_layer = index_first_axis(  # (t_kv, h_kv, d)
        key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
    )
    value_layer = index_first_axis(  # (t_kv, h_kv, d)
        value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
    )

    if torch.equal(q_sequence_ids, k_sequence_ids):
        indices_q = indices_k
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
    else:
        # (t_q,), (n + 1,), scalar
        indices_q, cu_seqlens_q, max_seqlen_in_batch_q = _get_unpad_data(q_sequence_ids)

    query_layer = index_first_axis(  # (t_q, h, d)
        query_layer.reshape(batch_size * query_length, num_q_heads, head_dim), indices_q
    )

    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError(
            "Query and KV must have the same number of sequences: "
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
