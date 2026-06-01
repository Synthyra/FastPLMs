from __future__ import annotations

import hashlib
import itertools
import os
import pickle
import random
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections import defaultdict, namedtuple
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tqdm.auto import tqdm
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

try:
    from fastplms.attention import (
        AttentionBackend, VALID_ATTENTION_BACKENDS,
        resolve_attention_backend,
        _get_flex_attention_fn,
        _ensure_flash_kernels_loaded, FLASH_KERNEL, FLASH_KERNEL_VARIANT,
        _kernels_flash_forward, _kernels_flash_varlen_forward,
        index_first_axis, index_put_first_axis, pad_input,
        create_block_mask, flex_attention, BlockMask,
    )
    from fastplms.embedding_mixin import (
        Pooler, EmbeddingMixin, ProteinDataset, parse_fasta, build_collator,
        select_hidden_state_embeddings,
    )
except ImportError:
    pass  # Running as HF Hub composite; shared definitions are above


logger = logging.get_logger(__name__)

from torch.nn.attention.flex_attention import _create_sparse_block_from_block_mask

try:
    from kernels import get_kernel
    layer_norm = get_kernel("kernels-community/triton-layer-norm")
except Exception as e:
    logger.warning(f"Failed to load triton layer norm kernel: {e}; Will be using PyTorch RMSNorm instead")
    layer_norm = None


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
    return create_block_mask(document_mask, batch_size, 1, seqlen, seqlen, device=sequence_ids.device)


@torch.compiler.disable
def create_within_seq_block_mask(sequence_ids: torch.Tensor) -> BlockMask:
    def document_mask(b, h, q_idx, kv_idx):  # type: ignore[no-untyped-def]
        return (
            (sequence_ids[b, q_idx] == sequence_ids[b, kv_idx])
            & (sequence_ids[b, q_idx] != -1)
            & (sequence_ids[b, kv_idx] != -1)
        )

    batch_size, seqlen = sequence_ids.shape
    return create_block_mask(document_mask, batch_size, 1, seqlen, seqlen, device=sequence_ids.device)


def build_within_seq_mask_4d(sequence_ids: torch.Tensor) -> torch.Tensor:
    not_pad = (sequence_ids != -1)
    same_seq = sequence_ids.unsqueeze(-1) == sequence_ids.unsqueeze(-2)
    valid = not_pad.unsqueeze(-1) & not_pad.unsqueeze(-2)
    return (same_seq & valid).unsqueeze(1)


def build_block_causal_mask_4d(sequence_ids: torch.Tensor) -> torch.Tensor:
    not_pad = (sequence_ids != -1)
    causal = sequence_ids.unsqueeze(-1) >= sequence_ids.unsqueeze(-2)
    valid = not_pad.unsqueeze(-1) & not_pad.unsqueeze(-2)
    return (causal & valid).unsqueeze(1)


def flex_attention_func(
    query_states: torch.Tensor,  # (bs, seqlen, nh, hs)
    key_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    value_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    score_mod: Optional[Callable] = None,
    block_mask: Optional[BlockMask] = None,
) -> torch.Tensor:
    assert flex_attention is not None, "Flex Attention is not available in this environment"
    assert score_mod is None, "Score mod is not supported yet"
    query_states = query_states.transpose(1, 2).contiguous()  # (bs, nh, seqlen, hs)
    key_states = key_states.transpose(1, 2).contiguous()  # (bs, nkv, seqlen, hs)
    value_states = value_states.transpose(1, 2).contiguous()  # (bs, nkv, seqlen, hs)

    fn = _get_flex_attention_fn()
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
) -> torch.Tensor:  # (bs, seqlen, nh, hs)
    assert FLASH_KERNEL is not None, "Kernel Flash Attention is not available in this environment."

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
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, q_len)

    else:
        attn_output = _kernels_flash_forward(query_states, key_states, value_states, causal=True)

    return attn_output


def block_min_max_seq_ids(SLEN: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    device = SLEN.device
    total_tokens = torch.sum(SLEN)
    B = (total_tokens + block_size - 1) // block_size
    padding_tokens = B * block_size - total_tokens
    SLEN = torch.cat([SLEN, padding_tokens.reshape(1).to(device=device, dtype=SLEN.dtype)], dim=0)

    assert torch.sum(SLEN) == B * block_size

    # Cumulative ends (exclusive) for each sequence; cum[i] == end offset of seq i
    cum = torch.cumsum(SLEN.to(torch.long), dim=0)  # (N,)
    total_tokens = cum[-1].item()

    # Block start/end offsets [start, end) in token index space
    block_starts = torch.arange(0, B * block_size, block_size, device=device, dtype=torch.long)  # (B,)
    block_ends = torch.minimum(block_starts + block_size, torch.tensor(total_tokens, device=device))  # (B,)

    # MIN_SEQ_ID[i] = first sequence whose end > block_start
    # searchsorted with right=True returns first index where cum > value
    MIN_SEQ_ID = torch.searchsorted(cum, block_starts, right=True)

    # MAX_SEQ_ID[i] = sequence containing the last token in the block (block_end - 1)
    # For empty tail beyond total_tokens we already clipped block_ends.
    last_token_in_block = torch.clamp(block_ends - 1, min=0)  # valid only if block has at least 1 token
    MAX_SEQ_ID = torch.searchsorted(cum, last_token_in_block, right=True)

    return MIN_SEQ_ID, MAX_SEQ_ID


def get_overlapping_blocks(SLEN_Q: torch.Tensor, SLEN_K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    MIN_Q, MAX_Q = block_min_max_seq_ids(SLEN_Q)
    MIN_K, MAX_K = block_min_max_seq_ids(SLEN_K)

    cond1 = MIN_Q.unsqueeze(1) <= MAX_K.unsqueeze(0)
    cond2 = MIN_K.unsqueeze(0) <= MAX_Q.unsqueeze(1)
    overlap = cond1 & cond2

    cond1 = (MIN_Q == MAX_Q).unsqueeze(1)
    cond2 = (MIN_K == MAX_K).unsqueeze(0)
    same_seq_in_qk = cond1 & cond2

    full_blocks = overlap & same_seq_in_qk
    partial_blocks = overlap & ~same_seq_in_qk

    return full_blocks, partial_blocks


@torch.compiler.disable
def direct_block_mask(SLEN_Q: torch.Tensor, SLEN_K: torch.Tensor) -> BlockMask:
    full_blocks, partial_blocks = get_overlapping_blocks(SLEN_Q, SLEN_K)
    partial_blocks = partial_blocks[None, None]
    full_blocks = full_blocks[None, None]

    q_doc_id = torch.repeat_interleave(SLEN_Q)
    k_doc_id = torch.repeat_interleave(SLEN_K)

    def doc_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return q_doc_id[q_idx] == k_doc_id[kv_idx]

    total_q_len = q_doc_id.shape[0]
    total_k_len = k_doc_id.shape[0]

    return _create_sparse_block_from_block_mask(
        (partial_blocks, full_blocks),
        doc_mask,
        seq_lengths=(total_q_len, total_k_len),
        Q_BLOCK_SIZE=128,
        KV_BLOCK_SIZE=128,
    )


@torch.compiler.disable
def doc_id_mask(SLEN_Q: torch.Tensor, SLEN_K: torch.Tensor) -> BlockMask:
    q_doc_id = torch.repeat_interleave(SLEN_Q)
    k_doc_id = torch.repeat_interleave(SLEN_K)

    def doc_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return q_doc_id[q_idx] == k_doc_id[kv_idx]

    total_q_len = q_doc_id.shape[0]
    total_k_len = k_doc_id.shape[0]

    return create_block_mask(doc_mask, 1, 1, total_q_len, total_k_len, BLOCK_SIZE=128, device=SLEN_Q.device)


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
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    ) = _unpad_input(query_states, key_states, value_states, q_sequence_ids, k_sequence_ids)

    query_states = query_states.unsqueeze(0).transpose(1, 2).contiguous()
    key_states = key_states.unsqueeze(0).transpose(1, 2).contiguous()
    value_states = value_states.unsqueeze(0).transpose(1, 2).contiguous()

    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    block_mask = block_mask_creator(seqlens_q, seqlens_k)

    fn = _get_flex_attention_fn()
    attn_output_unpad = fn(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=query_states.shape[1] != key_states.shape[1],
    )

    attn_output = pad_input(attn_output_unpad.transpose(1, 2).squeeze(0), indices_q, batch_size, q_len)

    return attn_output


def _get_unpad_data(sequence_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    non_pad_indices = sequence_ids != -1
    non_pad_indices = torch.nonzero(non_pad_indices.flatten(), as_tuple=False).flatten()
    sequence_ids = sequence_ids + torch.arange(len(sequence_ids), device=sequence_ids.device)[:, None] * 1e5
    sequence_ids = sequence_ids.flatten()[non_pad_indices]
    _, seqlens_in_batch = torch.unique_consecutive(sequence_ids, return_counts=True)
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return non_pad_indices, cu_seqlens, max_seqlen_in_batch


def _unpad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
    batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
    query_length, num_q_heads = query_layer.shape[1], query_layer.shape[2]
    assert query_layer.shape[:2] == q_sequence_ids.shape, (
        f"Shape mismatch between query layer and query sequence ids: {query_layer.shape[:2]} != {q_sequence_ids.shape}"
    )
    assert key_layer.shape[:2] == k_sequence_ids.shape, (
        f"Shape mismatch between key layer and key sequence ids: {key_layer.shape[:2]} != {k_sequence_ids.shape}"
    )
    assert query_length <= kv_seq_len, (
        f"Query length should be less than or equal to KV sequence length: {query_length} <= {kv_seq_len}"
    )

    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(k_sequence_ids)

    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
    value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

    if torch.equal(q_sequence_ids, k_sequence_ids):
        indices_q = indices_k
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
    else:
        indices_q, cu_seqlens_q, max_seqlen_in_batch_q = _get_unpad_data(q_sequence_ids)

    query_layer = index_first_axis(query_layer.reshape(batch_size * query_length, num_q_heads, head_dim), indices_q)

    assert cu_seqlens_q.shape == cu_seqlens_k.shape, (
        f"Query and KV should have the same number of sequences: {cu_seqlens_q.shape} != {cu_seqlens_k.shape}"
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
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
E1_VOCAB_SIZE = 34
E1_TOKENIZER_REPO_ID = "Synthyra/Profluent-E1-150M"


def _load_tokenizer_file(fname: str) -> Tokenizer:
    tokenizer: Tokenizer = Tokenizer.from_file(fname)
    assert tokenizer.padding["pad_id"] == PAD_TOKEN_ID, (
        f"Padding token id must be {PAD_TOKEN_ID}, but got {tokenizer.padding['pad_id']}"
    )
    return tokenizer


def get_tokenizer(
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
    *,
    local_files_only: bool = False,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
) -> Tokenizer:
    source_path = None
    checked_local_source = False
    if pretrained_model_name_or_path is not None:
        source_path = os.fspath(pretrained_model_name_or_path)
        if os.path.isdir(source_path):
            checked_local_source = True
            fname = os.path.join(source_path, "tokenizer.json")
            if os.path.isfile(fname):
                return _load_tokenizer_file(fname)

    fname = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    if os.path.isfile(fname):
        return _load_tokenizer_file(fname)

    if local_files_only and checked_local_source:
        raise FileNotFoundError(
            f"E1 tokenizer.json was not found in {source_path} or next to {__file__}."
        )

    from huggingface_hub import hf_hub_download

    repo_id = E1_TOKENIZER_REPO_ID
    if source_path is not None and not checked_local_source:
        repo_id = source_path
    try:
        fname = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            cache_dir=os.fspath(cache_dir) if cache_dir is not None else None,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
    except Exception as error:
        raise FileNotFoundError(
            f"E1 tokenizer.json was not found locally and could not be loaded from {repo_id}."
        ) from error
    return _load_tokenizer_file(fname)


@dataclass
class DataPrepConfig:
    max_num_sequences: int = 512
    max_num_positions_within_seq: int = 8192
    remove_X_tokens: bool = False


def get_context(sequence: str) -> Optional[str]:
    if "," in sequence:
        return sequence.rsplit(",", 1)[0]
    return None


class E1BatchPreparer:
    def __init__(
        self,
        data_prep_config: Optional[DataPrepConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
        tokenizer_source: Optional[Union[str, os.PathLike]] = None,
        local_files_only: bool = False,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        revision: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        preserve_context_labels: bool = False,
    ):
        self.tokenizer = tokenizer or get_tokenizer(
            tokenizer_source,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
        )
        self.data_prep_config = data_prep_config or DataPrepConfig()
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.preserve_context_labels = preserve_context_labels
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        self.boundary_token_ids = torch.tensor(
            [self.tokenizer.token_to_id(token) for token in ["<bos>", "<eos>", "1", "2", "<pad>"]], device=device
        ).long()
        self.mask_token = "?"  # nosec
        self.mask_token_id = self.tokenizer.token_to_id(self.mask_token)
        self.X_token_id = self.tokenizer.token_to_id("X")
        self.vocab = self.tokenizer.get_vocab()

    def get_batch_kwargs(  # type: ignore[override]
        self, sequences: List[str], device: torch.device = torch.device("cpu"), non_blocking: bool = False
    ) -> Dict[str, Union[torch.Tensor, List[str], List[int]]]:
        sequence_encodings = [self.prepare_multiseq(sequence) for sequence in sequences]
        return self.pad_encodings(sequence_encodings, device, non_blocking)

    def pad_encodings(
        self,
        sequence_encodings: List[Dict[str, torch.Tensor]],
        device: torch.device = torch.device("cpu"),
        non_blocking: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[str], List[int]]]:
        non_blocking = non_blocking and device.type == "cuda"
        padded_encodings = {}
        # Note: We use -1 as the padding value for sequence and position ids because the 0 value
        # is a valid value for sequence and position ids. -1 is then used to distinguish valid
        # tokens from padding tokens, for example, when doing padding/unpadding for flash attention.
        for key, padding_value in {
            "input_ids": self.pad_token_id,
            "sequence_ids": -1,
            "within_seq_position_ids": -1,
            "global_position_ids": -1,
            "labels": self.pad_token_id,
        }.items():
            padded_encodings[key] = pad_sequence(
                [enc[key] for enc in sequence_encodings], batch_first=True, padding_value=padding_value
            ).to(device=device, dtype=torch.long, non_blocking=non_blocking)

        padded_encodings["context"] = [enc["context"] for enc in sequence_encodings]
        padded_encodings["context_len"] = [enc["context_len"] for enc in sequence_encodings]

        return padded_encodings

    def prepare_multiseq(self, sequence: str) -> Dict[str, Union[torch.Tensor, str, int]]:
        single_sequences = sequence.split(",")
        if len(single_sequences) > self.data_prep_config.max_num_sequences:
            raise ValueError(
                f"Number of sequences {len(single_sequences)} exceeds max number of sequences {self.data_prep_config.max_num_sequences}"
                " in the provided multi-sequence instance. Please remove some homologous sequences before trying again."
            )

        single_sequence_encodings = [self.prepare_singleseq(sequence) for sequence in single_sequences]

        num_tokens = [len(x["input_ids"]) for x in single_sequence_encodings]
        input_ids = torch.cat([x["input_ids"] for x in single_sequence_encodings])
        labels = torch.cat([x["labels"] for x in single_sequence_encodings])

        within_seq_position_ids = torch.cat([encoding["position_ids"] for encoding in single_sequence_encodings])
        global_position_ids, ctx_len = [], 0
        for encoding in single_sequence_encodings:
            global_position_ids.append(encoding["position_ids"] + ctx_len)
            ctx_len = max(ctx_len, encoding["position_ids"].max().item() + ctx_len + 1)
        global_position_ids = torch.cat(global_position_ids)

        sequence_ids = torch.repeat_interleave(torch.tensor(num_tokens))

        # Get multi-seq context & mask out all but last sequence in multi-seq instance if desired
        context_len = sum(num_tokens[:-1])
        context = self.tokenizer.decode(input_ids[:context_len].tolist(), skip_special_tokens=False)
        if not self.preserve_context_labels:
            labels[:context_len] = self.pad_token_id

        assert (
            input_ids.shape
            == sequence_ids.shape
            == within_seq_position_ids.shape
            == global_position_ids.shape
            == labels.shape
        ), "Input ids, sequence ids, within seq position ids, global position ids, and labels must have the same shape"

        assert input_ids.shape[0] >= context_len, "Input ids must have at least as many tokens as the context length"

        return {
            "input_ids": input_ids,
            "sequence_ids": sequence_ids,
            "within_seq_position_ids": within_seq_position_ids,
            "global_position_ids": global_position_ids,
            "labels": labels,
            "context": context,
            "context_len": context_len,
        }

    def prepare_singleseq(self, sequence: str) -> Dict[str, torch.Tensor]:
        if not self.validate_sequence(sequence):
            raise ValueError(f"Invalid sequence: {sequence}; Input sequence should contain [A-Z] or ? characters only")

        if len(sequence) > self.data_prep_config.max_num_positions_within_seq:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds max length {self.data_prep_config.max_num_positions_within_seq}"
            )

        # Can also use `tokens = torch.tensor(self.tokenizer.encode(f"<bos>1{sequence}2<eos>").ids)`
        # but following is faster since our vocabulary is simple.
        tokens = torch.tensor([self.vocab[token] for token in ["<bos>", "1", *sequence, "2", "<eos>"]])
        position_ids = torch.arange(len(tokens))

        if self.data_prep_config.remove_X_tokens:
            X_positions = torch.where(tokens != self.X_token_id)[0]
            tokens = tokens[X_positions]
            position_ids = position_ids[X_positions]

        return {"input_ids": tokens, "labels": tokens, "position_ids": position_ids}

    def get_boundary_token_mask(self, tokens: torch.Tensor) -> torch.BoolTensor:
        return torch.isin(tokens, self.boundary_token_ids.to(tokens.device))

    def get_mask_positions_mask(self, tokens: torch.Tensor) -> torch.BoolTensor:
        return tokens == self.mask_token_id

    def validate_sequence(self, sequence: str) -> bool:
        assert isinstance(sequence, str), "Sequence must be a string"
        sequence = sequence.replace(self.mask_token, "")
        return sequence.isalpha() and sequence.isupper()


class E1Config(PretrainedConfig):
    model_type = "E1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(  # type: ignore
        self,
        # Model architecture/initialization
        vocab_size=None,
        hidden_size=4096,
        intermediate_size=16384,
        gated_mlp=False,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        dtype="bfloat16",
        gradient_checkpointing=False,
        no_ffn_gradient_checkpointing=False,
        # Tokenization
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        # Attention implementation & rotary positional embeddings
        global_attention_every_n_layers=0,
        max_num_sequences=512,
        max_num_positions_within_seq=8192,
        max_num_positions_global=1024 * 128,
        rope_theta_within_seq=10000.0,
        rope_theta_global=100000.0,
        clip_qkv=None,
        attn_backend="sdpa",
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=PAD_TOKEN_ID,
            bos_token_id=BOS_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
            tie_word_embeddings=tie_word_embeddings,
            dtype=dtype,
            **kwargs,
        )

        self.hidden_size = hidden_size
        if intermediate_size is None:
            intermediate_size = 3 * hidden_size if gated_mlp else 4 * hidden_size
        self.intermediate_size = intermediate_size
        self.gated_mlp = gated_mlp
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_num_positions_within_seq = max_num_positions_within_seq
        self.max_num_positions_global = max_num_positions_global

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta_within_seq = rope_theta_within_seq
        self.rope_theta_global = rope_theta_global
        self.max_num_sequences = max_num_sequences
        assert clip_qkv is None or clip_qkv > 0
        self.clip_qkv = clip_qkv
        self.global_attention_every_n_layers = global_attention_every_n_layers

        self.vocab_size = E1_VOCAB_SIZE
        self.gradient_checkpointing = gradient_checkpointing
        self.no_ffn_gradient_checkpointing = no_ffn_gradient_checkpointing
        self.attn_backend = attn_backend

        if vocab_size is not None:
            if vocab_size < self.vocab_size:
                logger.warning(
                    f"Using vocab_size {vocab_size} smaller than {self.vocab_size} from tokenizer. MAKE SURE THIS IS INTENTIONAL."
                )
                self.vocab_size = vocab_size
            elif vocab_size > self.vocab_size:
                logger.warning(
                    f"Using vocab_size {vocab_size} instead of smaller {self.vocab_size} "
                    "from E1 tokenizer contract."
                )
                self.vocab_size = vocab_size
        if pad_token_id is not None and pad_token_id != self.pad_token_id:
            logger.warning(f"Ignoring pad_token_id. Using {self.pad_token_id} from E1 tokenizer contract")
        if bos_token_id is not None and bos_token_id != self.bos_token_id:
            logger.warning(f"Ignoring bos_token_id. Using {self.bos_token_id} from E1 tokenizer contract")
        if eos_token_id is not None and eos_token_id != self.eos_token_id:
            logger.warning(f"Ignoring eos_token_id. Using {self.eos_token_id} from E1 tokenizer contract")


class DynamicCache:
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as tensors of shape `[batch_size, seq_len, num_heads, head_dim]`.

    Args:
        key_cache (`list[torch.Tensor]`): The list of key states.
        value_cache (`list[torch.Tensor]`): The list of value states.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache of shape [batch_size, seq_len, num_heads, head_dim]
            value_states (`torch.Tensor`): The new value states to cache of shape [batch_size, seq_len, num_heads, head_dim]
            layer_idx (`int`): The index of the layer to update.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states of shape [batch_size, seq_len, num_heads, head_dim].
        """
        # Lazy initialization
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append(torch.tensor([]))
                self.value_cache.append(torch.tensor([]))
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif (
            not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
        ):  # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=1)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[1] if not is_empty_layer else 0
        return layer_seq_length

    def crop(self, max_length: int) -> None:
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        assert max_length > 0, "max_length must be positive"

        if self.get_seq_length() <= max_length:
            return

        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :max_length, ...]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :max_length, ...]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class KVCache:
    def __init__(self, cache_size: int = 4) -> None:
        self.cache_size = cache_size
        self.tensor_input_field_names = [
            "input_ids",
            "within_seq_position_ids",
            "global_position_ids",
            "sequence_ids",
            "labels",
        ]
        self.tensor_output_field_names = ["logits", "embeddings"]
        self.cache_dict: Dict[str, DynamicCache] = {}
        self.cache_queue: List[str] = []

    def reset(self) -> None:
        for k in list(self.cache_dict.keys()):
            del self.cache_dict[k]
        del self.cache_dict
        self.cache_dict = {}
        self.cache_queue = []

        torch.cuda.empty_cache()

    def before_forward(self, batch: Dict[str, torch.Tensor]) -> None:
        contexts: Optional[List[str]] = batch.get("context", None)
        if contexts is None or "context_len" not in batch:
            logger.warning_once(
                "KVCache requires the batch dict to have both `context` and `context_len` keys to trigger. Skipping."
            )
            return

        context_lens: List[int] = list(set(batch["context_len"]))
        contexts: List[str] = list(set(contexts))  # type: ignore[no-redef]
        if len(contexts) != 1 or len(context_lens) != 1:
            logger.warning(
                "SingleContextKVCache requires a single context and context length. "
                "Multiple contexts or context lengths found in a single batch. Skipping."
            )
            return

        batch_size = batch["input_ids"].shape[0]

        unique_context = contexts[0]
        unique_context_len = context_lens[0]
        batch["use_cache"] = True

        if unique_context not in self.cache_dict:
            return

        self.cache_dict[unique_context].batch_repeat_interleave(batch_size)
        past_key_values = self.cache_dict[unique_context]
        batch["past_key_values"] = past_key_values

        # Remove context from the input fields
        for field_name in self.tensor_input_field_names:
            if batch.get(field_name, None) is not None:
                batch[field_name] = batch[field_name][:, unique_context_len:]

    def after_forward(self, batch: Dict[str, Any], outputs: ModelOutput) -> None:
        contexts = batch.get("context", None)
        context_lens = batch.get("context_len", [])
        if contexts is None or len(set(contexts)) != 1 or len(set(context_lens)) != 1 or context_lens[0] == 0:
            return

        assert batch["use_cache"]
        unique_context = contexts[0]
        unique_context_len = context_lens[0]

        past_key_values = getattr(outputs, "past_key_values", None)
        if not isinstance(past_key_values, DynamicCache):
            logger.warning_once("KVCache is incompatible with models that don't return a DynamicCache. Skipping.")
            return

        if "past_key_values" not in batch:
            if len(self.cache_queue) == self.cache_size:
                last_context = self.cache_queue.pop(0)
                if last_context not in self.cache_queue:
                    del self.cache_dict[last_context]
                    torch.cuda.empty_cache()

            self.cache_dict[unique_context] = past_key_values
            self.cache_queue.append(unique_context)

            # Remove context from the input fields
            for field_name in self.tensor_input_field_names:
                if field_name in batch and batch[field_name] is not None:
                    batch[field_name] = batch[field_name][:, unique_context_len:]

            # Remove context from the output fields
            for field_name in self.tensor_output_field_names:
                if field_name in outputs and outputs[field_name] is not None:
                    outputs[field_name] = outputs[field_name][:, unique_context_len:]
            if "hidden_states" in outputs and outputs["hidden_states"] is not None:
                outputs["hidden_states"] = [h[:, unique_context_len:] for h in outputs["hidden_states"]]

        self.cache_dict[unique_context].crop(unique_context_len)
        self.cache_dict[unique_context].batch_select_indices([0])


DOCKER_IMAGE = "ghcr.io/soedinglab/mmseqs2"
COLABFOLD_HOST = "https://api.colabfold.com"
LOWERCASE_CHARS = b"abcdefghijklmnopqrstuvwxyz"
DEFAULT_MAX_CONTEXT_TOKENS = [6144, 12288, 24576]
DEFAULT_SIMILARITY_THRESHOLDS = [1.0, 0.95, 0.9, 0.7, 0.5]
DEFAULT_EMBED_MAX_TOKENS = 8192
DEFAULT_EMBED_SIMILARITY = 0.95

IdSequence = namedtuple("IdSequence", ["id", "sequence"])
IndexedSequence = Tuple[int, str]


@dataclass
class ContextSpecification:
    max_num_samples: int = 511
    max_token_length: int = 32768
    max_query_similarity: float = 1.0
    min_query_similarity: float = 0.0
    neighbor_similarity_lower_bound: float = 0.8


class E1Prediction(TypedDict, total=False):
    id: str | int
    context_id: str | int | None
    logits: torch.Tensor
    token_embeddings: torch.Tensor
    mean_token_embeddings: torch.Tensor


def read_fasta_sequences(path: str) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    header: Optional[str] = None
    parts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequences[header] = "".join(parts)
                header = line[1:].strip()
                parts = []
            else:
                assert header is not None, f"FASTA sequence found before header in {path}"
                parts.append(line)
    if header is not None:
        sequences[header] = "".join(parts)
    return sequences


def write_fasta_sequences(path: str, sequences: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for header, sequence in sequences.items():
            handle.write(f">{header}\n{sequence}\n")


def parse_msa(path: str) -> List[IdSequence]:
    records = read_fasta_sequences(path)
    sequences = []
    for record_id, record_seq in records.items():
        sequence = str(record_seq).replace("\x00", "").replace(".", "-")
        sequences.append(IdSequence(record_id, sequence))
    assert len(sequences) > 0, f"No sequences found in MSA file: {path}"
    return sequences


def convert_to_tensor(sequences: List[IdSequence], device: Optional[torch.device] = None) -> torch.ByteTensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    byte_sequences = [
        sequence.sequence.encode("ascii").translate(None, LOWERCASE_CHARS)
        for sequence in sequences
    ]
    lengths = {len(byte_sequence) for byte_sequence in byte_sequences}
    assert len(lengths) == 1, f"MSA rows must have equal aligned lengths after removing insertions: {sorted(lengths)}"
    array = np.vstack([np.frombuffer(byte_sequence, dtype=np.uint8) for byte_sequence in byte_sequences])
    return torch.from_numpy(array).to(device)


def get_num_neighbors(byte_seqs: torch.ByteTensor, sim_threshold: float = 0.8) -> List[int]:
    gap_token_id = np.frombuffer(b"-", np.uint8)[0].item()
    seq_lens = (byte_seqs != gap_token_id).sum(dim=1)
    num_neighbors: List[int] = []
    for i in range(byte_seqs.shape[0]):
        query_non_gaps = byte_seqs[i] != gap_token_id
        seqs_sim = (byte_seqs[:, query_non_gaps] == byte_seqs[i, query_non_gaps]).sum(dim=1) / seq_lens
        num_neighbors.append(int((seqs_sim >= sim_threshold).sum().item()))
    return num_neighbors


def get_similarity_to_query(byte_seqs: torch.ByteTensor) -> torch.FloatTensor:
    return (byte_seqs == byte_seqs[0, :]).sum(dim=1) / byte_seqs.shape[1]


def sample_context(
    msa_path: str,
    max_num_samples: int,
    max_token_length: int,
    max_query_similarity: float = 1.0,
    min_query_similarity: float = 0.0,
    neighbor_similarity_lower_bound: float = 0.8,
    use_full_sequences_in_context: bool = False,
    full_sequences_path: Optional[str] = None,
    seed: int = 0,
    device: Optional[torch.device] = None,
    cache_num_neighbors_path: Optional[str] = None,
) -> Tuple[str, List[str]]:
    msa_sequences = parse_msa(msa_path)
    msa_as_byte_tensor = convert_to_tensor(msa_sequences, device)
    if cache_num_neighbors_path is not None and os.path.exists(cache_num_neighbors_path):
        num_neighbors = np.load(cache_num_neighbors_path)
    else:
        num_neighbors = np.array(get_num_neighbors(msa_as_byte_tensor, neighbor_similarity_lower_bound))
        if cache_num_neighbors_path is not None:
            np.save(cache_num_neighbors_path, num_neighbors)

    sampling_weights = 1.0 / num_neighbors
    query_similarity = get_similarity_to_query(msa_as_byte_tensor)
    filtered_mask = (query_similarity <= max_query_similarity) & (query_similarity >= min_query_similarity)
    assert filtered_mask.sum() >= 1, (
        f"No sequences found with similarity to query within range "
        f"{min_query_similarity} <= query_similarity <= {max_query_similarity}."
    )

    filtered_weights = np.where(filtered_mask.cpu().numpy(), sampling_weights, 0.0)
    sampled_indices = np.random.default_rng(seed).choice(
        len(filtered_weights),
        size=min(max_num_samples, int(filtered_mask.sum())),
        p=filtered_weights / filtered_weights.sum(),
        replace=False,
        shuffle=True,
    )

    if use_full_sequences_in_context:
        assert full_sequences_path is not None, "full_sequences_path is required when use_full_sequences_in_context=True"
        full_sequences = parse_msa(full_sequences_path)
        assert len(full_sequences) == len(msa_sequences), "Number of full sequences must match number of MSA sequences"
        for i, (full_seq, msa_seq) in enumerate(zip(full_sequences, msa_sequences)):
            assert full_seq.id == msa_seq.id, (
                "Full sequences and MSA sequences must be in the same order and have the same ids. "
                f"Found differing id for sample {i}: {full_seq.id} != {msa_seq.id}"
            )
        sampled_sequences = [full_sequences[int(i)] for i in sampled_indices]
    else:
        sampled_sequences = [msa_sequences[int(i)] for i in sampled_indices]

    context_sequences: List[str] = []
    context_ids: List[str] = []
    context_length = 0
    for seq in sampled_sequences:
        seq_str = seq.sequence.upper().encode("ascii").translate(None, b"-").decode("ascii")
        if context_length + len(seq_str) > max_token_length:
            break
        context_sequences.append(seq_str)
        context_ids.append(seq.id)
        context_length += len(seq_str)
    return ",".join(context_sequences), context_ids


def sample_multiple_contexts(
    msa_path: str,
    context_specifications: List[ContextSpecification],
    use_full_sequences_in_context: bool = False,
    full_sequences_path: Optional[str] = None,
    seed: int = 0,
    device: Optional[torch.device] = None,
    cache_num_neighbors_path: Optional[str] = None,
) -> Tuple[List[str], List[List[str]]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        if cache_num_neighbors_path is None:
            cache_num_neighbors_path = os.path.join(temp_dir, "num_neighbors.npy")

        contexts: List[str] = []
        context_ids: List[List[str]] = []
        for i, context_specification in enumerate(context_specifications):
            context, ids = sample_context(
                msa_path=msa_path,
                max_num_samples=context_specification.max_num_samples,
                max_token_length=context_specification.max_token_length,
                max_query_similarity=context_specification.max_query_similarity,
                min_query_similarity=context_specification.min_query_similarity,
                neighbor_similarity_lower_bound=context_specification.neighbor_similarity_lower_bound,
                use_full_sequences_in_context=use_full_sequences_in_context,
                full_sequences_path=full_sequences_path,
                seed=seed + i,
                device=device,
                cache_num_neighbors_path=cache_num_neighbors_path,
            )
            contexts.append(context)
            context_ids.append(ids)
    return contexts, context_ids


def get_context_id(max_tokens: int, sim_threshold: float) -> str:
    return f"identity_{sim_threshold}_tokens_{max_tokens}"


def build_context_specifications(
    max_context_tokens: Optional[List[int]] = None,
    similarity_thresholds: Optional[List[float]] = None,
    min_query_similarity: float = 0.3,
) -> List[Tuple[ContextSpecification, str]]:
    if max_context_tokens is None:
        max_context_tokens = DEFAULT_MAX_CONTEXT_TOKENS
    if similarity_thresholds is None:
        similarity_thresholds = DEFAULT_SIMILARITY_THRESHOLDS

    specs = []
    for max_tokens in max_context_tokens:
        for sim_threshold in similarity_thresholds:
            spec = ContextSpecification(
                max_num_samples=511,
                max_token_length=max_tokens,
                max_query_similarity=sim_threshold,
                min_query_similarity=min_query_similarity,
                neighbor_similarity_lower_bound=0.8,
            )
            specs.append((spec, get_context_id(max_tokens, sim_threshold)))
    return specs


def sample_contexts_for_msa(
    a3m_path: str,
    context_specs: List[Tuple[ContextSpecification, str]],
    seed: int = 42,
) -> Dict[str, str]:
    specs_only = [spec for spec, _ in context_specs]
    context_ids = [context_id for _, context_id in context_specs]
    contexts, _ = sample_multiple_contexts(
        msa_path=a3m_path,
        context_specifications=specs_only,
        seed=seed,
    )
    return dict(zip(context_ids, contexts))


def _strip_a3m_insertions(sequence: str) -> str:
    uppercase_or_gap = [char for char in sequence if char.isupper() or char in "-."]
    return "".join(uppercase_or_gap).replace("-", "").replace(".", "")


def get_query_from_a3m(path: str) -> str:
    header_found = False
    seq_parts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header_found:
                    break
                header_found = True
                continue
            if header_found:
                seq_parts.append(line)
    assert header_found, f"No FASTA header found in A3M file: {path}"
    return _strip_a3m_insertions("".join(seq_parts))


def load_msa_dir(msa_dir: str) -> Dict[str, str]:
    msa_lookup: Dict[str, str] = {}
    a3m_files = list(Path(msa_dir).rglob("*.a3m"))
    if not a3m_files:
        raise FileNotFoundError(f"No .a3m files found in {msa_dir}")
    for a3m_path in tqdm(a3m_files, desc="Loading MSAs"):
        query_seq = get_query_from_a3m(str(a3m_path))
        msa_lookup[query_seq] = str(a3m_path)
    logger.info("Loaded %d MSAs from %s", len(msa_lookup), msa_dir)
    return msa_lookup


def _safe_extract_tar(tar: tarfile.TarFile, output_dir: str) -> None:
    output_root = Path(output_dir).resolve()
    for member in tar.getmembers():
        target = (output_root / member.name).resolve()
        if output_root != target and output_root not in target.parents:
            raise ValueError(f"Unsafe tar member path: {member.name}")
    tar.extractall(output_root)


def load_msa_from_hf(
    hf_path: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, str]:
    from huggingface_hub import snapshot_download

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fastplms_msa")
    os.makedirs(cache_dir, exist_ok=True)
    local_dir = os.path.join(cache_dir, hf_path.replace("/", "_"))
    if not os.path.exists(local_dir) or not any(Path(local_dir).rglob("*.a3m")):
        local_dir = snapshot_download(
            repo_id=hf_path,
            repo_type="dataset",
            local_dir=local_dir,
            token=token,
        )
        for tar_path in Path(local_dir).rglob("*.tar.gz"):
            with tarfile.open(tar_path) as tar:
                _safe_extract_tar(tar, str(tar_path.parent))
    return load_msa_dir(local_dir)


def get_msa_for_sequence(sequence: str, msa_lookup: Dict[str, str], min_identity: float = 0.95) -> Optional[str]:
    if sequence in msa_lookup:
        return msa_lookup[sequence]

    best_match_path: Optional[str] = None
    best_identity = 0.0
    for query_seq, a3m_path in msa_lookup.items():
        if abs(len(query_seq) - len(sequence)) > 10:
            continue
        min_len = min(len(query_seq), len(sequence))
        if min_len == 0:
            continue
        matches = sum(a == b for a, b in zip(query_seq[:min_len], sequence[:min_len]))
        identity = matches / min_len
        if identity > best_identity:
            best_identity = identity
            best_match_path = a3m_path

    if best_identity >= min_identity:
        return best_match_path
    return None


class ContextCache:
    def __init__(self, cache_dir: str, specs_hash: str, seed: int) -> None:
        self.cache_dir = cache_dir
        self.specs_hash = specs_hash
        self.seed = seed
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> str:
        safe_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{safe_key}_seed{self.seed}_{self.specs_hash}.pkl")

    def load(self, key: str) -> Optional[Dict[str, str]]:
        path = self._cache_path(key)
        if os.path.exists(path):
            with open(path, "rb") as handle:
                return pickle.load(handle)
        return None

    def store(self, key: str, contexts: Dict[str, str]) -> None:
        path = self._cache_path(key)
        with open(path, "wb") as handle:
            pickle.dump(contexts, handle)


def compute_ppll(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    assert token_ids.numel() > 0, "Cannot score an empty token sequence"
    if token_ids.device != logits.device:
        token_ids = token_ids.to(logits.device)
    if logits.shape[0] != token_ids.shape[0]:
        raise ValueError(f"Logits length {logits.shape[0]} != token_ids length {token_ids.shape[0]}")
    probs = logits.softmax(dim=-1)
    token_probs = probs.gather(dim=1, index=token_ids.unsqueeze(1)).squeeze(1)
    return float(token_probs.mean().item())


class _E1ContextPredictor:
    def __init__(
        self,
        model: PreTrainedModel,
        data_prep_config: Optional[DataPrepConfig] = None,
        max_batch_tokens: int = 65536,
        use_cache: bool = True,
        cache_size: int = 4,
        save_masked_positions_only: bool = False,
        fields_to_save: Optional[List[str]] = None,
        keep_predictions_in_gpu: bool = False,
        progress: bool = True,
    ) -> None:
        self.model = model
        self.max_batch_tokens = max_batch_tokens
        self.batch_preparer = E1BatchPreparer(data_prep_config=data_prep_config)
        self.model.eval()
        self.kv_cache = KVCache(cache_size=cache_size) if use_cache else None
        self.fields_to_save = fields_to_save or ["logits", "token_embeddings", "mean_token_embeddings"]
        self.save_masked_positions_only = save_masked_positions_only
        self.keep_predictions_in_gpu = keep_predictions_in_gpu
        self.progress = progress

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def group_by_length(self, indexed_sequences: List[IndexedSequence]) -> List[List[IndexedSequence]]:
        batches: List[List[IndexedSequence]] = [[]]
        for idx, seq in sorted(indexed_sequences, key=lambda idx_seq: (len(idx_seq[1]), idx_seq[0])):
            if len(batches[-1]) > 0 and len(seq) * (len(batches[-1]) + 1) > self.max_batch_tokens:
                batches.append([])
            batches[-1].append((idx, seq))
        return batches

    def group_by_context(self, indexed_sequences: List[IndexedSequence]) -> List[List[IndexedSequence]]:
        batches: Dict[Optional[str], List[IndexedSequence]] = defaultdict(list)
        for idx, seq in indexed_sequences:
            batches[get_context(seq)].append((idx, seq))
        return list(batches.values())

    def batch_sequences(self, sequences: List[str]) -> List[List[int]]:
        indexed_sequences: List[IndexedSequence] = list(enumerate(sequences))
        indexed_batches = self.group_by_context(indexed_sequences)
        indexed_batches = list(
            itertools.chain.from_iterable([self.group_by_length(batch) for batch in indexed_batches])
        )
        batches = [[item[0] for item in batch] for batch in indexed_batches]
        assert sorted(sum(batches, [])) == list(range(len(sequences))), (
            "Batches must contain all indices with no repetition"
        )
        return batches

    @torch.no_grad()
    def predict_batch(self, sequences: List[str], sequence_metadata: List[Dict[str, str | int]]) -> List[E1Prediction]:
        outputs = self.predict_batch_padded(sequences)
        outputs["logits"] = outputs["logits"].float()
        outputs["embeddings"] = outputs["embeddings"].float()

        token_mask = outputs["non_boundary_token_mask"] & outputs["last_sequence_mask"]
        if self.save_masked_positions_only:
            token_mask = token_mask & outputs["mask_positions_mask"]

        predictions: List[E1Prediction] = []
        for i in range(len(sequences)):
            pred: E1Prediction = {"id": sequence_metadata[i]["id"]}
            if "context_id" in sequence_metadata[i]:
                pred["context_id"] = sequence_metadata[i]["context_id"]
            if "logits" in self.fields_to_save:
                pred["logits"] = outputs["logits"][i, token_mask[i]]
                if not self.keep_predictions_in_gpu:
                    pred["logits"] = pred["logits"].to("cpu")
            if "token_embeddings" in self.fields_to_save:
                pred["token_embeddings"] = outputs["embeddings"][i, token_mask[i]]
                if not self.keep_predictions_in_gpu:
                    pred["token_embeddings"] = pred["token_embeddings"].to("cpu")
            if "mean_token_embeddings" in self.fields_to_save:
                pred["mean_token_embeddings"] = outputs["embeddings"][i, token_mask[i]].mean(dim=0)
                if not self.keep_predictions_in_gpu:
                    pred["mean_token_embeddings"] = pred["mean_token_embeddings"].to("cpu")
            predictions.append(pred)
        return predictions

    @torch.no_grad()
    def predict_batch_padded(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        device = self.device
        autocast_enabled = device.type == "cuda"
        with torch.autocast(device.type, torch.bfloat16, enabled=autocast_enabled):
            batch = self.batch_preparer.get_batch_kwargs(sequences, device=device)
            if self.kv_cache is not None:
                self.kv_cache.before_forward(batch)

            past_key_values = batch["past_key_values"] if "past_key_values" in batch else None
            use_cache = bool(batch["use_cache"]) if "use_cache" in batch else False
            output: E1MaskedLMOutputWithPast = self.model(
                input_ids=batch["input_ids"],
                within_seq_position_ids=batch["within_seq_position_ids"],
                global_position_ids=batch["global_position_ids"],
                sequence_ids=batch["sequence_ids"],
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=False,
                output_hidden_states=False,
            )
            if self.kv_cache is not None:
                self.kv_cache.after_forward(batch, output)

            padding_mask = batch["input_ids"] == self.batch_preparer.pad_token_id
            last_sequence_mask = batch["sequence_ids"] == batch["sequence_ids"].max(dim=1).values[:, None]
            boundary_token_mask = self.batch_preparer.get_boundary_token_mask(batch["input_ids"])
            mask_positions_mask = self.batch_preparer.get_mask_positions_mask(batch["input_ids"])
            return {
                "logits": output.logits,
                "embeddings": output.last_hidden_state,
                "last_sequence_mask": last_sequence_mask,
                "non_boundary_token_mask": ~boundary_token_mask,
                "mask_positions_mask": mask_positions_mask,
                "valid_token_mask": ~padding_mask,
            }

    @torch.no_grad()
    def predict(
        self,
        sequences: Sequence[str],
        sequence_ids: Optional[Sequence[int | str]] = None,
        context_seqs: Optional[Dict[str, str]] = None,
    ) -> Iterator[E1Prediction]:
        if sequence_ids is None:
            sequence_ids = list(range(len(sequences)))
        if context_seqs:
            sequences_with_context = [
                (ctx + "," + seq, {"context_id": ctx_id, "id": sequence_id})
                for ctx_id, ctx in context_seqs.items()
                for seq, sequence_id in zip(sequences, sequence_ids)
            ]
        else:
            sequences_with_context = [(seq, {"id": sequence_id}) for seq, sequence_id in zip(sequences, sequence_ids)]

        batched_sequences, sequence_metadata = tuple(zip(*sequences_with_context))
        batches = self.batch_sequences(list(batched_sequences))
        iterator = tqdm(batches, desc="Predicting batches", disable=not self.progress)
        for indices in iterator:
            sequence_batch = [batched_sequences[i] for i in indices]
            sequence_batch_metadata = [sequence_metadata[i] for i in indices]
            yield from self.predict_batch(sequence_batch, sequence_batch_metadata)


def _pool_hidden_states(
    hidden_list: List[torch.Tensor],
    pooling_types: List[str],
    device: torch.device,
) -> torch.Tensor:
    pooler = Pooler(pooling_types)
    max_len = max(hidden.shape[0] for hidden in hidden_list)
    hidden_dim = hidden_list[0].shape[1]
    batch_size = len(hidden_list)
    padded = torch.zeros(batch_size, max_len, hidden_dim, device=device)
    attention_mask = torch.zeros(batch_size, max_len, device=device)
    for i, hidden in enumerate(hidden_list):
        seq_len = hidden.shape[0]
        padded[i, :seq_len] = hidden
        attention_mask[i, :seq_len] = 1.0
    return pooler(padded, attention_mask)


def _forward_for_embedding(
    model: PreTrainedModel,
    sequences: List[str],
    context: Optional[str],
    max_batch_tokens: int,
    progress: bool,
) -> List[torch.Tensor]:
    predictor = _E1ContextPredictor(
        model=model,
        data_prep_config=DataPrepConfig(remove_X_tokens=True),
        max_batch_tokens=max_batch_tokens,
        fields_to_save=["token_embeddings"],
        keep_predictions_in_gpu=True,
        use_cache=False,
        cache_size=1,
        progress=progress,
    )
    context_seqs = {"embed_ctx": context} if context else None
    predictions = list(
        predictor.predict(
            sequences=sequences,
            sequence_ids=list(range(len(sequences))),
            context_seqs=context_seqs,
        )
    )
    predictions.sort(key=lambda prediction: prediction["id"])
    return [prediction["token_embeddings"] for prediction in predictions]


class HomologueSearcher:
    def __init__(
        self,
        target_db: str,
        docker_image: str = DOCKER_IMAGE,
        sensitivity: float = 7.5,
        max_seqs: int = 1000,
        min_seq_id: float = 0.0,
        coverage: float = 0.8,
        split_memory_limit: Optional[str] = None,
        use_gpu: bool = True,
    ) -> None:
        self.target_db = target_db
        self.docker_image = docker_image
        self.sensitivity = sensitivity
        self.max_seqs = max_seqs
        self.min_seq_id = min_seq_id
        self.coverage = coverage
        self.split_memory_limit = split_memory_limit
        self.use_gpu = use_gpu

    @staticmethod
    def _seq_hash(sequence: str) -> str:
        return hashlib.md5(sequence.encode()).hexdigest()[:12]

    def _run_docker_command(self, cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, **kwargs)

    def _validate_paths_under_cwd(self, *paths: str) -> None:
        cwd = os.path.abspath(os.getcwd())
        for path in paths:
            absolute_path = os.path.abspath(path)
            if not (absolute_path == cwd or absolute_path.startswith(cwd + os.sep)):
                raise ValueError(
                    "Path must be under the current working directory for docker volume mount. "
                    f"cwd={cwd!r}, path={absolute_path!r}"
                )

    def _path_in_container(self, local_path: str) -> str:
        self._validate_paths_under_cwd(local_path)
        rel = os.path.relpath(os.path.abspath(local_path), start=os.path.abspath(os.getcwd()))
        return rel.replace(os.sep, "/")

    def _docker_base_cmd(self) -> List[str]:
        cmd = ["docker", "run", "--rm", "-v", f"{os.getcwd()}:/app", "-w", "/app"]
        if self.use_gpu and torch.cuda.is_available():
            cmd.extend(["--gpus", "all"])
        cmd.append(self.docker_image)
        return cmd

    def _ensure_docker_image(self) -> None:
        subprocess.run(["docker", "version"], capture_output=True, text=True, check=True)
        inspect = subprocess.run(["docker", "image", "inspect", self.docker_image], capture_output=True, text=True)
        if inspect.returncode == 0:
            return
        self._run_docker_command(["docker", "pull", self.docker_image], check=True, text=True)

    def create_db(self, fasta_path: str, db_path: str) -> str:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        if os.path.exists(f"{db_path}.dbtype"):
            return db_path
        self._ensure_docker_image()
        self._validate_paths_under_cwd(fasta_path, db_path)
        self._run_docker_command(
            self._docker_base_cmd() + [
                "createdb",
                self._path_in_container(fasta_path),
                self._path_in_container(db_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return db_path

    def create_index(self, db_path: str, tmp_dir: Optional[str] = None) -> None:
        if tmp_dir is None:
            tmp_dir = os.path.join(os.path.dirname(db_path), "tmp_index")
        os.makedirs(tmp_dir, exist_ok=True)
        self._ensure_docker_image()
        self._validate_paths_under_cwd(db_path, tmp_dir)
        self._run_docker_command(
            self._docker_base_cmd() + [
                "createindex",
                self._path_in_container(db_path),
                self._path_in_container(tmp_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def search(self, sequence: str, output_dir: str, seq_id: Optional[str] = None) -> str:
        if seq_id is None:
            seq_id = self._seq_hash(sequence)
        seq_output_dir = os.path.join(output_dir, seq_id)
        a3m_output = os.path.join(seq_output_dir, f"{seq_id}.a3m")
        if os.path.exists(a3m_output):
            return a3m_output

        self._ensure_docker_image()
        os.makedirs(seq_output_dir, exist_ok=True)
        query_fasta = os.path.join(seq_output_dir, "query.fasta")
        write_fasta_sequences(query_fasta, {seq_id: sequence})
        query_db = os.path.join(seq_output_dir, "queryDB")
        result_db = os.path.join(seq_output_dir, "resultDB")
        tmp_dir = os.path.join(seq_output_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        self._validate_paths_under_cwd(query_fasta, query_db, self.target_db, seq_output_dir, result_db, tmp_dir)

        docker_base = self._docker_base_cmd()
        self._run_docker_command(
            docker_base + [
                "createdb",
                self._path_in_container(query_fasta),
                self._path_in_container(query_db),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        search_cmd = docker_base + [
            "search",
            self._path_in_container(query_db),
            self._path_in_container(self.target_db),
            self._path_in_container(result_db),
            self._path_in_container(tmp_dir),
            "-s",
            str(self.sensitivity),
            "--max-seqs",
            str(self.max_seqs),
            "--min-seq-id",
            str(self.min_seq_id),
            "-c",
            str(self.coverage),
        ]
        if self.split_memory_limit is not None:
            search_cmd.extend(["--split-memory-limit", self.split_memory_limit])
        if self.use_gpu and torch.cuda.is_available():
            search_cmd.extend(["--gpu", "1"])
        self._run_docker_command(search_cmd, check=True, capture_output=True, text=True)
        self._run_docker_command(
            docker_base + [
                "result2msa",
                self._path_in_container(query_db),
                self._path_in_container(self.target_db),
                self._path_in_container(result_db),
                self._path_in_container(a3m_output),
                "--msa-format-mode",
                "6",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for pattern in ["queryDB*", "resultDB*"]:
            for path in Path(seq_output_dir).glob(pattern):
                path.unlink(missing_ok=True)
        tmp_path = Path(tmp_dir)
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        return a3m_output

    def batch_search(
        self,
        sequences: List[str],
        output_dir: str,
        seq_ids: Optional[List[str]] = None,
        continue_on_error: bool = True,
    ) -> Dict[str, str]:
        if seq_ids is None:
            seq_ids = [self._seq_hash(seq) for seq in sequences]
        os.makedirs(output_dir, exist_ok=True)
        results: Dict[str, str] = {}
        for seq, sid in tqdm(list(zip(sequences, seq_ids)), desc="Searching homologues"):
            try:
                results[seq] = self.search(seq, output_dir, sid)
            except Exception:
                if not continue_on_error:
                    raise
        return results


class ColabFoldSearcher:
    def __init__(
        self,
        host_url: str = COLABFOLD_HOST,
        user_agent: str = "",
        mode: str = "env",
        timeout: float = 30.0,
        max_retries: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        inter_request_delay: Tuple[float, float] = (1.0, 3.0),
        max_wait_time: int = 600,
    ) -> None:
        import requests

        self.requests = requests
        self.host_url = host_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.inter_request_delay = inter_request_delay
        self.max_wait_time = max_wait_time
        self.session = requests.Session()
        if user_agent:
            self.session.headers["User-Agent"] = user_agent

    @staticmethod
    def _seq_hash(sequence: str) -> str:
        return hashlib.md5(sequence.encode()).hexdigest()[:12]

    def _backoff_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        return delay + random.uniform(0, delay * 0.5)

    def _request_with_retries(self, method: str, url: str, **kwargs):
        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, timeout=self.timeout, **kwargs)
                else:
                    response = self.session.post(url, timeout=self.timeout, **kwargs)
                if response.status_code == 429:
                    retry_after = (
                        float(response.headers["Retry-After"])
                        if "Retry-After" in response.headers
                        else self._backoff_delay(attempt)
                    )
                    time.sleep(retry_after)
                    continue
                if response.status_code >= 500:
                    time.sleep(self._backoff_delay(attempt))
                    continue
                return response
            except (self.requests.exceptions.Timeout, self.requests.exceptions.ConnectionError):
                time.sleep(self._backoff_delay(attempt))
        raise RuntimeError(f"Request to {url} failed after {self.max_retries} attempts")

    def _submit(self, sequence: str, mode: Optional[str] = None) -> Dict[str, Any]:
        mode = mode or self.mode
        query = f">101\n{sequence}\n"
        for attempt in range(self.max_retries):
            response = self._request_with_retries(
                "POST",
                f"{self.host_url}/ticket/msa",
                data={"q": query, "mode": mode},
            )
            data = response.json()
            status = data["status"] if "status" in data else "UNKNOWN"
            if status in ("RATELIMIT", "UNKNOWN"):
                time.sleep(self._backoff_delay(attempt))
                continue
            return data
        raise RuntimeError(f"Failed to submit sequence after {self.max_retries} attempts")

    def _poll(self, ticket_id: str) -> Dict[str, Any]:
        total_wait = 0.0
        poll_interval = 1.0
        while True:
            response = self._request_with_retries("GET", f"{self.host_url}/ticket/{ticket_id}")
            data = response.json()
            status = data["status"] if "status" in data else "ERROR"
            if status in ("COMPLETE", "ERROR"):
                return data
            if status not in ("RUNNING", "PENDING", "UNKNOWN"):
                return data
            wait = min(poll_interval + random.uniform(0, 0.5), 5.0)
            time.sleep(wait)
            total_wait += wait
            poll_interval = min(poll_interval + 1.0, 5.0)
            if total_wait > self.max_wait_time:
                raise TimeoutError(f"Job {ticket_id} did not complete within {self.max_wait_time}s")

    def _download(self, ticket_id: str, output_path: str) -> None:
        response = self._request_with_retries("GET", f"{self.host_url}/result/download/{ticket_id}")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as handle:
            handle.write(response.content)

    def _extract_a3m(self, tar_path: str, output_dir: str, seq_id: str) -> str:
        with tarfile.open(tar_path) as tar:
            _safe_extract_tar(tar, output_dir)

        uniref_a3m = os.path.join(output_dir, "uniref.a3m")
        env_a3m = os.path.join(output_dir, "bfd.mgnify30.metaeuk30.smag30.a3m")
        a3m_files: List[str] = []
        if os.path.exists(uniref_a3m):
            a3m_files.append(uniref_a3m)
        if "env" in self.mode and os.path.exists(env_a3m):
            a3m_files.append(env_a3m)
        combined_path = os.path.join(output_dir, f"{seq_id}.a3m")
        if len(a3m_files) == 1:
            os.replace(a3m_files[0], combined_path)
        elif len(a3m_files) > 1:
            with open(combined_path, "w", encoding="utf-8") as out_handle:
                for a3m_file in a3m_files:
                    with open(a3m_file, "r", encoding="utf-8") as in_handle:
                        out_handle.write(in_handle.read())
        else:
            raise RuntimeError("No .a3m files found in downloaded archive")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        for a3m_file in a3m_files:
            if os.path.exists(a3m_file) and a3m_file != combined_path:
                os.remove(a3m_file)
        return combined_path

    def search(self, sequence: str, output_dir: str, seq_id: Optional[str] = None) -> str:
        if seq_id is None:
            seq_id = self._seq_hash(sequence)
        seq_output_dir = os.path.join(output_dir, seq_id)
        a3m_output = os.path.join(seq_output_dir, f"{seq_id}.a3m")
        if os.path.exists(a3m_output):
            return a3m_output
        os.makedirs(seq_output_dir, exist_ok=True)
        result = self._submit(sequence)
        status = result["status"] if "status" in result else "UNKNOWN"
        if status == "ERROR":
            raise RuntimeError(f"ColabFold API error for {seq_id}")
        if status == "MAINTENANCE":
            raise RuntimeError("ColabFold API is under maintenance")
        ticket_id = result["id"]
        result = self._poll(ticket_id)
        status = result["status"] if "status" in result else "UNKNOWN"
        if status != "COMPLETE":
            raise RuntimeError(f"Job failed for {seq_id}: {status}")
        tar_path = os.path.join(seq_output_dir, f"{seq_id}.tar.gz")
        self._download(ticket_id, tar_path)
        return self._extract_a3m(tar_path, seq_output_dir, seq_id)

    def batch_search(
        self,
        sequences: List[str],
        output_dir: str,
        seq_ids: Optional[List[str]] = None,
        continue_on_error: bool = True,
    ) -> Dict[str, str]:
        if seq_ids is None:
            seq_ids = [self._seq_hash(seq) for seq in sequences]
        os.makedirs(output_dir, exist_ok=True)
        results: Dict[str, str] = {}
        pairs = list(zip(sequences, seq_ids))
        for i, (seq, sid) in enumerate(tqdm(pairs, desc="ColabFold search")):
            try:
                results[seq] = self.search(seq, output_dir, sid)
            except Exception:
                if not continue_on_error:
                    raise
            if i < len(pairs) - 1:
                time.sleep(random.uniform(*self.inter_request_delay))
        return results


def _make_homologue_searcher(provider: str, target_db: Optional[str], **kwargs) -> HomologueSearcher | ColabFoldSearcher:
    if provider == "mmseqs2":
        assert target_db is not None, "target_db is required for MMseqs2 homologue search"
        return HomologueSearcher(target_db=target_db, **kwargs)
    if provider == "colabfold":
        return ColabFoldSearcher(**kwargs)
    raise ValueError(f"Unknown homologue search provider: {provider}")


class AttentionLayerType(Enum):
    WITHIN_SEQ = "within_seq"
    GLOBAL = "global"


class AttentionArgs(TypedDict, total=False):
    within_seq_block_mask: Optional[BlockMask]
    block_causal_block_mask: Optional[BlockMask]
    within_seq_mask_4d: Optional[torch.Tensor]
    block_causal_mask_4d: Optional[torch.Tensor]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch,
    num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device: Optional[torch.device] = None
    ):
        super().__init__()

        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = base ** -(torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_sin_cos_cache(seq_len=max_position_embeddings, device=self.inv_freq.device)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _set_sin_cos_cache(self, seq_len: int, device: torch.device) -> None:
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        angles = torch.outer(t, self.inv_freq.to(device))
        angles = torch.cat((angles, angles), dim=1)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.LongTensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bsz, seq_len, num_attention_heads, head_size]
        device, dtype = q.device, q.dtype
        seq_len = position_ids.max().item() + 1 if seq_len is None else seq_len

        if seq_len > self.max_seq_len_cached:
            self._set_sin_cos_cache(seq_len=seq_len, device=device)

        # angles_cached[position_ids] gets us something of shape (batch_size, seq_len, head_dim),
        # so unsqueeze dimension -2 to broadcast to (batch_size, seq_len, n_heads, head_dim).
        idxs = position_ids.to(device)
        cos = self.cos_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]
        sin = self.sin_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]

        # Apply rotary positional embeddings to q and k (treating them as complex numbers). The first half is
        # Re[x exp(it)] = Re[x] cos(t) - Im[x] sin(t), while the second half is
        # Im[x exp(it)] = Im[x] cos(t) + Re[x] sin(t). This works b/c both halves of cos/sin are the same.
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: E1Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.max_num_seqs = config.max_num_sequences
        self.clip_qkv = config.clip_qkv

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if self.config.global_attention_every_n_layers > 0:
            self.layer_type = (
                AttentionLayerType.GLOBAL
                if (self.layer_idx + 1) % self.config.global_attention_every_n_layers == 0
                else AttentionLayerType.WITHIN_SEQ
            )
        else:
            self.layer_type = AttentionLayerType.WITHIN_SEQ

        self.rope_theta = (
            config.rope_theta_within_seq
            if self.layer_type == AttentionLayerType.WITHIN_SEQ
            else config.rope_theta_global
        )
        self.max_position_embeddings = (
            config.max_num_positions_within_seq
            if self.layer_type == AttentionLayerType.WITHIN_SEQ
            else config.max_num_positions_global
        )

        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta
        )

        self.attn_backend = resolve_attention_backend(config.attn_backend)

    def prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Optional[DynamicCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        query_states: torch.Tensor = self.q_proj(hidden_states)
        key_states: torch.Tensor = self.k_proj(hidden_states)
        val_states: torch.Tensor = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        val_states = val_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        if self.clip_qkv is not None:
            query_states = query_states.clamp(-self.clip_qkv, self.clip_qkv)
            key_states = key_states.clamp(-self.clip_qkv, self.clip_qkv)
            val_states = val_states.clamp(-self.clip_qkv, self.clip_qkv)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        if use_cache and past_key_value is not None:
            key_states, val_states = past_key_value.update(key_states, val_states, self.layer_idx)

        input_dtype = query_states.dtype
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = self.q_proj.weight.dtype
        if input_dtype != target_dtype:
            logger.warning_once(
                f"The input hidden states seems to be silently casted in {input_dtype}. "
                f"This might be because you have upcasted embedding or layer norm layers "
                f"in {input_dtype}. We will cast back the input in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            val_states = val_states.to(target_dtype)

        return query_states, key_states, val_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: Optional[AttentionArgs] = None,
        past_key_value: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[DynamicCache], Optional[List[torch.Tensor]]]:
        is_cache_prefilled = (
            use_cache and past_key_value is not None and past_key_value.get_seq_length(self.layer_idx) > 0
        )

        query_states, key_states, val_states = self.prepare_qkv(
            hidden_states=hidden_states,
            position_ids=within_seq_position_ids
            if self.layer_type == AttentionLayerType.WITHIN_SEQ
            else global_position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        attn_output, attn_weights, s_max = self._attn(
            query_states=query_states,
            key_states=key_states,
            val_states=val_states,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            is_cache_prefilled=is_cache_prefilled,
        )

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value, s_max

    def _attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        sequence_ids: torch.Tensor,
        attention_args: Optional[AttentionArgs] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        is_cache_prefilled: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        effective_layer_type = self.layer_type
        if is_cache_prefilled and self.layer_type == AttentionLayerType.GLOBAL:
            effective_layer_type = AttentionLayerType.WITHIN_SEQ

        if output_attentions:
            return self._manual_attn(
                query_states, key_states, val_states,
                sequence_ids=sequence_ids,
                attention_args=attention_args,
                effective_layer_type=effective_layer_type,
                output_s_max=output_s_max,
                is_cache_prefilled=is_cache_prefilled,
            )

        if self.attn_backend == AttentionBackend.KERNELS_FLASH:
            if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
                attn_output, attn_weights = self._kernels_flash_attn(
                    query_states, key_states, val_states,
                    sequence_ids=sequence_ids,
                    is_cache_prefilled=is_cache_prefilled,
                )
            else:
                attn_output, attn_weights = self._flex_attn(
                    query_states, key_states, val_states,
                    attention_args=attention_args,
                    effective_layer_type=effective_layer_type,
                )
        elif self.attn_backend == AttentionBackend.FLEX:
            attn_output, attn_weights = self._flex_attn(
                query_states, key_states, val_states,
                attention_args=attention_args,
                effective_layer_type=effective_layer_type,
            )
        elif self.attn_backend == AttentionBackend.SDPA:
            attn_output, attn_weights = self._sdpa_attn(
                query_states, key_states, val_states,
                sequence_ids=sequence_ids,
                attention_args=attention_args,
                effective_layer_type=effective_layer_type,
                is_cache_prefilled=is_cache_prefilled,
            )
        else:
            raise AssertionError(f"Unsupported resolved backend: {self.attn_backend}")

        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None
        return attn_output, attn_weights, s_max

    @torch.no_grad()
    def _compute_s_max(
        self,
        query_states: torch.Tensor,  # (B, L, H, D)
        key_states: torch.Tensor,    # (B, L, Hkv, D)
    ) -> List[torch.Tensor]:
        query_BHLD = query_states.transpose(1, 2).contiguous()
        key_BHLD = key_states.transpose(1, 2).contiguous()
        key_BHLD = repeat_kv(key_BHLD, self.num_key_value_groups)
        scale = 1.0 / (self.head_dim ** 0.5)
        q_norm = torch.linalg.vector_norm(query_BHLD, dim=-1)
        k_norm = torch.linalg.vector_norm(key_BHLD, dim=-1)
        s_max_bound = (q_norm.max(dim=-1).values * k_norm.max(dim=-1).values).max(dim=0).values * scale
        return [s_max_bound[h] for h in range(self.num_heads)]

    def _kernels_flash_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        sequence_ids: torch.Tensor,
        is_cache_prefilled: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        bsz, q_len = query_states.shape[0], query_states.shape[1]
        _, kv_len = key_states.shape[0], key_states.shape[1]

        if self.layer_type == AttentionLayerType.GLOBAL and not is_cache_prefilled:
            q_sequence_ids = sequence_ids
            if q_len < kv_len:
                first_token_id = sequence_ids[:, 0].unsqueeze(1)
                k_sequence_ids = torch.cat([first_token_id.expand(bsz, kv_len - q_len), sequence_ids], dim=-1)
            else:
                k_sequence_ids = sequence_ids
        else:
            if q_len < kv_len:
                key_states = key_states[:, -q_len:]
                val_states = val_states[:, -q_len:]
            q_sequence_ids = k_sequence_ids = sequence_ids

        attn_output = kernels_flash_attention_func(
            query_states, key_states, val_states,
            q_sequence_ids=q_sequence_ids,
            k_sequence_ids=k_sequence_ids,
            causal=False,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        return attn_output, None

    def _flex_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        attention_args: Optional[AttentionArgs] = None,
        effective_layer_type: AttentionLayerType = AttentionLayerType.WITHIN_SEQ,
    ) -> Tuple[torch.Tensor, None]:
        bsz, q_len = query_states.shape[0], query_states.shape[1]
        if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
            block_mask = attention_args["within_seq_block_mask"] if attention_args is not None else None
        else:
            block_mask = attention_args["block_causal_block_mask"] if attention_args is not None else None
        outputs = flex_attention_func(query_states, key_states, val_states, block_mask=block_mask)
        outputs = outputs.reshape(bsz, q_len, self.hidden_size).contiguous()
        return outputs, None

    def _sdpa_attn(
        self,
        query_states: torch.Tensor,   # (B, L, H, D)
        key_states: torch.Tensor,      # (B, L, Hkv, D)
        val_states: torch.Tensor,      # (B, L, Hkv, D)
        sequence_ids: torch.Tensor,
        attention_args: Optional[AttentionArgs] = None,
        effective_layer_type: AttentionLayerType = AttentionLayerType.WITHIN_SEQ,
        is_cache_prefilled: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        bsz, q_len = query_states.shape[:2]
        kv_len = key_states.shape[1]

        if is_cache_prefilled and q_len < kv_len:
            if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
                key_states = key_states[:, -q_len:]
                val_states = val_states[:, -q_len:]
            attention_mask_4d = build_within_seq_mask_4d(sequence_ids) if effective_layer_type == AttentionLayerType.WITHIN_SEQ else None
        elif attention_args is not None:
            if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
                attention_mask_4d = attention_args["within_seq_mask_4d"]
            else:
                attention_mask_4d = attention_args["block_causal_mask_4d"]
        else:
            attention_mask_4d = None

        query_BHLD = query_states.transpose(1, 2).contiguous()
        key_BHLD = key_states.transpose(1, 2).contiguous()
        val_BHLD = val_states.transpose(1, 2).contiguous()
        key_BHLD = repeat_kv(key_BHLD, self.num_key_value_groups)
        val_BHLD = repeat_kv(val_BHLD, self.num_key_value_groups)
        context_BHLD = F.scaled_dot_product_attention(query_BHLD, key_BHLD, val_BHLD, attn_mask=attention_mask_4d)
        attn_output = context_BHLD.transpose(1, 2).reshape(bsz, q_len, self.hidden_size).contiguous()
        return attn_output, None

    def _manual_attn(
        self,
        query_states: torch.Tensor,   # (B, L, H, D)
        key_states: torch.Tensor,      # (B, L, Hkv, D)
        val_states: torch.Tensor,      # (B, L, Hkv, D)
        sequence_ids: torch.Tensor,
        attention_args: Optional[AttentionArgs] = None,
        effective_layer_type: AttentionLayerType = AttentionLayerType.WITHIN_SEQ,
        output_s_max: bool = False,
        is_cache_prefilled: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        bsz, q_len = query_states.shape[:2]
        kv_len = key_states.shape[1]

        if is_cache_prefilled and q_len < kv_len:
            if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
                key_states = key_states[:, -q_len:]
                val_states = val_states[:, -q_len:]
            attention_mask_4d = build_within_seq_mask_4d(sequence_ids) if effective_layer_type == AttentionLayerType.WITHIN_SEQ else None
        elif attention_args is not None:
            if effective_layer_type == AttentionLayerType.WITHIN_SEQ:
                attention_mask_4d = attention_args["within_seq_mask_4d"]
            else:
                attention_mask_4d = attention_args["block_causal_mask_4d"]
        else:
            attention_mask_4d = None

        query_BHLD = query_states.transpose(1, 2).contiguous()
        key_BHLD = key_states.transpose(1, 2).contiguous()
        val_BHLD = val_states.transpose(1, 2).contiguous()
        key_BHLD = repeat_kv(key_BHLD, self.num_key_value_groups)
        val_BHLD = repeat_kv(val_BHLD, self.num_key_value_groups)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = torch.matmul(query_BHLD, key_BHLD.transpose(-2, -1)) * scale
        if attention_mask_4d is not None:
            attn_weights = attn_weights.masked_fill(attention_mask_4d.logical_not(), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        context_BHLD = torch.matmul(attn_weights, val_BHLD)
        attn_output = context_BHLD.transpose(1, 2).reshape(bsz, q_len, self.hidden_size).contiguous()
        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None
        return attn_output, attn_weights, s_max


class MLP(nn.Module):
    def __init__(self, config: E1Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(hidden_states)))


class GLUMLP(nn.Module):
    def __init__(self, config: E1Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


class FFN(nn.Module):
    def __init__(self, config: E1Config):
        super().__init__()
        mlp_cls = GLUMLP if config.gated_mlp else MLP
        self.mlp = mlp_cls(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


@dataclass
class E1ModelOutputWithPast(ModelOutput):
    """Base class for model's outputs, with potential hidden states and attentions.

    Attributes:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[DynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    s_max: Optional[Tuple[List[torch.Tensor], ...]] = None


@dataclass
class E1MaskedLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[DynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    s_max: Optional[Tuple[List[torch.Tensor], ...]] = None


@dataclass
class E1ClassificationOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[DynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    s_max: Optional[Tuple[List[torch.Tensor], ...]] = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        if layer_norm is None:
            return torch.nn.functional.rms_norm(
                hidden_states, (self.hidden_size,), self.weight, self.variance_epsilon
            ).to(input_dtype)
        else:
            return layer_norm.rms_norm_fn(
                x=hidden_states,
                weight=self.weight,
                bias=None,  # no bias
                residual=None,
                eps=self.variance_epsilon,
                dropout_p=0.0,  # no dropout by default
                prenorm=False,
                residual_in_fp32=False,
            ).to(input_dtype)


class NormAttentionNorm(nn.Module):
    def __init__(self, config: E1Config, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: Optional[AttentionArgs] = None,
        past_key_value: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[DynamicCache], Optional[List[torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value, s_max = self.self_attn(
            hidden_states=hidden_states,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, residual, self_attn_weights, present_key_value, s_max


class DecoderLayer(nn.Module):
    def __init__(self, config: E1Config, layer_idx: int):
        super().__init__()
        self.initializer_range = config.initializer_range
        self.hidden_size = config.hidden_size
        self.norm_attn_norm = NormAttentionNorm(config, layer_idx)
        self.ffn = FFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: Optional[AttentionArgs] = None,
        past_key_value: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[DynamicCache], Optional[List[torch.Tensor]]]:
        hidden_states, residual, self_attn_weights, present_key_value, s_max = self.norm_attn_norm(
            hidden_states=hidden_states,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            use_cache=use_cache,
        )

        # Fully Connected
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights, present_key_value, s_max


class E1PreTrainedModel(PreTrainedModel):
    config_class = E1Config
    config: E1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _transformer_layer_cls = [DecoderLayer]
    _skip_keys_device_placement = "past_key_values"
    all_tied_weights_keys = {}
    _tokenizer_source: Optional[Union[str, os.PathLike]] = None
    _tokenizer_local_files_only = False
    _tokenizer_cache_dir: Optional[Union[str, os.PathLike]] = None
    _tokenizer_revision: Optional[str] = None
    _tokenizer_token: Optional[Union[str, bool]] = None

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *model_args: Any,
        **kwargs: Any,
    ) -> "E1PreTrainedModel":
        previous_source = E1PreTrainedModel._tokenizer_source
        previous_local_files_only = E1PreTrainedModel._tokenizer_local_files_only
        previous_cache_dir = E1PreTrainedModel._tokenizer_cache_dir
        previous_revision = E1PreTrainedModel._tokenizer_revision
        previous_token = E1PreTrainedModel._tokenizer_token
        E1PreTrainedModel._tokenizer_source = pretrained_model_name_or_path
        E1PreTrainedModel._tokenizer_local_files_only = (
            bool(kwargs["local_files_only"]) if "local_files_only" in kwargs else False
        )
        E1PreTrainedModel._tokenizer_cache_dir = kwargs["cache_dir"] if "cache_dir" in kwargs else None
        E1PreTrainedModel._tokenizer_revision = kwargs["revision"] if "revision" in kwargs else None
        if "token" in kwargs:
            E1PreTrainedModel._tokenizer_token = kwargs["token"]
        elif "use_auth_token" in kwargs:
            E1PreTrainedModel._tokenizer_token = kwargs["use_auth_token"]
        else:
            E1PreTrainedModel._tokenizer_token = None
        try:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        finally:
            E1PreTrainedModel._tokenizer_source = previous_source
            E1PreTrainedModel._tokenizer_local_files_only = previous_local_files_only
            E1PreTrainedModel._tokenizer_cache_dir = previous_cache_dir
            E1PreTrainedModel._tokenizer_revision = previous_revision
            E1PreTrainedModel._tokenizer_token = previous_token

    @staticmethod
    def _tokenizer_kwargs_from_config(config: E1Config) -> Dict[str, Any]:
        tokenizer_source = E1PreTrainedModel._tokenizer_source
        if tokenizer_source is None and isinstance(config._name_or_path, str) and len(config._name_or_path) > 0:
            tokenizer_source = config._name_or_path
        return {
            "tokenizer_source": tokenizer_source,
            "local_files_only": E1PreTrainedModel._tokenizer_local_files_only,
            "cache_dir": E1PreTrainedModel._tokenizer_cache_dir,
            "revision": E1PreTrainedModel._tokenizer_revision,
            "token": E1PreTrainedModel._tokenizer_token,
        }

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def _backward_compatibility_gradient_checkpointing(self) -> None:
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable(dict(use_reentrant=False))

    def post_init(self) -> None:
        super().post_init()

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def attn_backend(self) -> str:
        return self.config.attn_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        assert backend in VALID_ATTENTION_BACKENDS, (
            f"Unsupported attn_backend: {backend}. Expected one of {VALID_ATTENTION_BACKENDS}."
        )
        self.config.attn_backend = backend
        resolved = resolve_attention_backend(backend)
        for module in self.modules():
            if isinstance(module, FAST_E1_ENCODER):
                module._attn_backend = resolved
            elif isinstance(module, Attention):
                module.attn_backend = resolved


class FAST_E1_ENCODER(E1PreTrainedModel, EmbeddingMixin):
    config: E1Config
    config_class = E1Config
    def __init__(self, config: E1Config, **kwargs):
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_seq_id = nn.Embedding(config.max_num_sequences, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.prep_tokens = E1BatchPreparer(
            **E1PreTrainedModel._tokenizer_kwargs_from_config(config)
        )
        self._attn_backend = resolve_attention_backend(config.attn_backend)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def _embed(
        self,
        sequences: List[str],
        return_attention_mask: bool = False,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        batch = self.prep_tokens.get_batch_kwargs(sequences, device=self._device)
        output_hidden_states = store_all_hidden_states or hidden_state_index != -1
        output = self.forward(
            **batch,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
        )
        embeddings = select_hidden_state_embeddings(
            output.last_hidden_state,
            output.hidden_states,
            hidden_state_index=hidden_state_index,
            store_all_hidden_states=store_all_hidden_states,
        )
        if return_attention_mask:
            attention_mask = (batch['sequence_ids'] != -1).long()
            return embeddings, attention_mask
        else:
            return embeddings

    # Ignore copy
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        within_seq_position_ids: Optional[torch.LongTensor] = None,
        global_position_ids: Optional[torch.LongTensor] = None,
        sequence_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
        **kwargs
    ) -> E1ModelOutputWithPast:
        """
        Args:
            input_ids: (batch_size, seq_length)
            within_seq_position_ids: (batch_size, seq_length)
                This tensor contains the position of each residue within the sequence itself.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos><pad>"],
                the tensor would be [[0,1,2,3,4,5,6,0,1,2,3,4,5,6], [0,1,2,3,4,5,0,1,2,3,4,5,6,-1]]
            global_position_ids: (batch_size, seq_length)
                This tensor contains the position of each residue within the global sequence.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos>"],
                the tensor would be [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1]]
            sequence_ids: (batch_size, seq_length)
                This tensor contains the sequence id of each residue.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos>"],
                the tensor would be [[0,0,0,0,0,0,0,1,1,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1,1,1,-1]]
            inputs_embeds: (batch_size, seq_length, hidden_size) - pre-computed embeddings,
                bypasses embed_tokens and embed_seq_id when provided. Used by PDE for
                differentiable soft sequence optimization.
            past_key_values: DynamicCache
            use_cache: bool
            output_attentions: bool
            output_hidden_states: bool
            output_s_max: bool

        Returns:
            E1ModelOutputWithPast: Model Outputs
        """
        assert not (input_ids is not None and inputs_embeds is not None), (
            "Cannot specify both input_ids and inputs_embeds"
        )
        assert input_ids is not None or inputs_embeds is not None, (
            "Must specify either input_ids or inputs_embeds"
        )

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]

        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        elif not use_cache:
            past_key_values = None

        # Synthesize positional IDs for soft embedding path (single-sequence)
        if inputs_embeds is not None:
            device = inputs_embeds.device
            if within_seq_position_ids is None:
                within_seq_position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
            if global_position_ids is None:
                global_position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
            if sequence_ids is None:
                sequence_ids = torch.zeros(batch_size, seq_length, device=device, dtype=torch.long)

        global_position_ids = global_position_ids.view(-1, seq_length).long()
        within_seq_position_ids = within_seq_position_ids.view(-1, seq_length).long()
        sequence_ids = sequence_ids.view(-1, seq_length).long()

        max_position_id = torch.max(within_seq_position_ids).item()
        min_position_id = torch.min(within_seq_position_ids).item()
        assert max_position_id < self.config.max_num_positions_within_seq and min_position_id >= -1, (
            f"Position ids must be in the range [-1, {self.config.max_num_positions_within_seq}); got max {max_position_id} and min {min_position_id}"
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds + self.embed_seq_id(sequence_ids.clamp(min=0))

        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = self.layers[0].norm_attn_norm.self_attn.q_proj.weight.dtype
        hidden_states = inputs_embeds.to(target_dtype)

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        attn_backend = self._attn_backend
        has_global_layers = self.config.global_attention_every_n_layers > 0
        needs_4d_masks = (attn_backend == AttentionBackend.SDPA) or output_attentions
        needs_block_causal_flex = (
            (attn_backend == AttentionBackend.FLEX and has_global_layers)
            or (attn_backend == AttentionBackend.KERNELS_FLASH and has_global_layers)
        )
        needs_within_seq_flex = (attn_backend == AttentionBackend.FLEX)

        attention_args: Optional[AttentionArgs] = None
        if past_key_values_length == 0:
            attention_args = AttentionArgs(
                block_causal_block_mask=create_block_causal_mask_optimized(sequence_ids) if needs_block_causal_flex else None,
                within_seq_block_mask=create_within_seq_block_mask(sequence_ids) if needs_within_seq_flex else None,
                within_seq_mask_4d=build_within_seq_mask_4d(sequence_ids) if needs_4d_masks else None,
                block_causal_mask_4d=build_block_causal_mask_4d(sequence_ids) if needs_4d_masks else None,
            )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        full_s_max = () if output_s_max else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore[operator]

            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    within_seq_position_ids,
                    global_position_ids,
                    sequence_ids,
                    attention_args,
                    past_key_values,
                    output_attentions,
                    output_s_max,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    within_seq_position_ids=within_seq_position_ids,
                    global_position_ids=global_position_ids,
                    sequence_ids=sequence_ids,
                    attention_args=attention_args,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_s_max=output_s_max,
                    use_cache=use_cache,
                )

            hidden_states, self_attn_weights, present_key_value, s_max = layer_outputs

            if use_cache:
                next_decoder_cache = past_key_values = present_key_value

            if output_attentions:
                all_self_attns += (self_attn_weights,)  # type: ignore[operator]

            if full_s_max is not None:
                full_s_max += (s_max,)  # type: ignore[operator]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # type: ignore[operator]

        next_cache = next_decoder_cache if use_cache else None

        return E1ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            s_max=full_s_max,
        )


class E1Model(E1PreTrainedModel, EmbeddingMixin):
    config: E1Config
    config_class = E1Config

    def __init__(self, config: E1Config, **kwargs):
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.model: FAST_E1_ENCODER = FAST_E1_ENCODER(config, **kwargs)
        self.prep_tokens = self.model.prep_tokens
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.set_input_embeddings(value)

    def _embed(self, sequences: List[str], return_attention_mask: bool = False, **kwargs) -> torch.Tensor:
        return self.model._embed(sequences, return_attention_mask=return_attention_mask, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        within_seq_position_ids: Optional[torch.LongTensor] = None,
        global_position_ids: Optional[torch.LongTensor] = None,
        sequence_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
        **kwargs,
    ) -> E1ModelOutputWithPast:
        return self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            **kwargs,
        )


class E1ForMaskedLM(E1PreTrainedModel, EmbeddingMixin):
    config: E1Config
    config_class = E1Config
    def __init__(self, config: E1Config, **kwargs):
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.model: FAST_E1_ENCODER = FAST_E1_ENCODER(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.mlm_head = torch.nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size, bias=True),
        )
        self.gradient_checkpointing = config.gradient_checkpointing
        self.prep_tokens = self.model.prep_tokens
        self.post_init()

    @property
    def device_mesh(self) -> torch.distributed.device_mesh.DeviceMesh:
        return self.model.device_mesh

    def _embed(self, sequences: List[str], return_attention_mask: bool = False, **kwargs) -> torch.Tensor:
        return self.model._embed(sequences, return_attention_mask=return_attention_mask, **kwargs)

    def search_homologues(
        self,
        sequence: str,
        output_dir: str,
        provider: str = "colabfold",
        target_db: Optional[str] = None,
        seq_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        searcher = _make_homologue_searcher(provider=provider, target_db=target_db, **kwargs)
        return searcher.search(sequence=sequence, output_dir=output_dir, seq_id=seq_id)

    def batch_search_homologues(
        self,
        sequences: List[str],
        output_dir: str,
        provider: str = "colabfold",
        target_db: Optional[str] = None,
        seq_ids: Optional[List[str]] = None,
        continue_on_error: bool = True,
        **kwargs,
    ) -> Dict[str, str]:
        searcher = _make_homologue_searcher(provider=provider, target_db=target_db, **kwargs)
        return searcher.batch_search(
            sequences=sequences,
            output_dir=output_dir,
            seq_ids=seq_ids,
            continue_on_error=continue_on_error,
        )

    def sample_msa_contexts(
        self,
        a3m_path: str,
        seed: int = 42,
        max_context_tokens: Optional[List[int]] = None,
        similarity_thresholds: Optional[List[float]] = None,
        min_query_similarity: float = 0.3,
        context_cache_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        context_specs = build_context_specifications(
            max_context_tokens=max_context_tokens,
            similarity_thresholds=similarity_thresholds,
            min_query_similarity=min_query_similarity,
        )
        cache = None
        if context_cache_dir is not None:
            key = repr((max_context_tokens, similarity_thresholds, min_query_similarity))
            specs_hash = hashlib.md5(key.encode()).hexdigest()[:8]
            cache = ContextCache(context_cache_dir, specs_hash, seed)
            cached = cache.load(a3m_path)
            if cached is not None:
                return cached
        contexts = sample_contexts_for_msa(a3m_path, context_specs, seed=seed)
        if cache is not None:
            cache.store(a3m_path, contexts)
        return contexts

    @torch.inference_mode()
    def score_ppll(
        self,
        sequences: List[str],
        a3m_path: str,
        ensemble: bool = True,
        seed: int = 42,
        max_context_tokens: Optional[List[int]] = None,
        similarity_thresholds: Optional[List[float]] = None,
        min_query_similarity: float = 0.3,
        max_batch_tokens: int = 131072,
        cache_size: int = 1,
        context_cache_dir: Optional[str] = None,
        progress: bool = True,
    ) -> List[float] | List[List[float]]:
        """Score sequences with FastPLMs PPLL reduction over sampled E1 MSA contexts.

        This intentionally differs from Profluent's official E1Scorer, which scores
        mutants against a parent sequence with wildtype or masked marginal log-prob
        deltas. Here each sequence is scored by mean correct-token probability and
        optionally averaged across sampled contexts.
        """
        contexts = self.sample_msa_contexts(
            a3m_path=a3m_path,
            seed=seed,
            max_context_tokens=max_context_tokens,
            similarity_thresholds=similarity_thresholds,
            min_query_similarity=min_query_similarity,
            context_cache_dir=context_cache_dir,
        )
        assert len(contexts) > 0, "At least one sampled MSA context is required for PPLL scoring"

        predictor = _E1ContextPredictor(
            model=self,
            data_prep_config=DataPrepConfig(remove_X_tokens=True),
            max_batch_tokens=max_batch_tokens,
            fields_to_save=["logits"],
            save_masked_positions_only=False,
            keep_predictions_in_gpu=False,
            use_cache=True,
            cache_size=cache_size,
            progress=progress,
        )
        vocab = predictor.batch_preparer.vocab
        seq_token_ids = [
            torch.tensor([vocab[aa] for aa in seq if aa != "X"], device=self.device)
            for seq in sequences
        ]
        context_ids = list(contexts.keys())
        all_scores = torch.zeros(len(sequences), len(context_ids), device=self.device)

        iterator = tqdm(context_ids, desc="Scoring with contexts", disable=not progress)
        for ctx_idx, ctx_id in enumerate(iterator):
            predictions = list(
                predictor.predict(
                    sequences=sequences,
                    sequence_ids=list(range(len(sequences))),
                    context_seqs={ctx_id: contexts[ctx_id]},
                )
            )
            for prediction in predictions:
                seq_idx = prediction["id"]
                assert isinstance(seq_idx, int), "Expected integer sequence ids for score aggregation"
                all_scores[seq_idx, ctx_idx] = compute_ppll(prediction["logits"], seq_token_ids[seq_idx])
            if predictor.kv_cache is not None:
                predictor.kv_cache.reset()

        if ensemble:
            return all_scores.mean(dim=1).tolist()
        return all_scores.tolist()

    @torch.inference_mode()
    def embed_with_msa(
        self,
        sequences: List[str],
        a3m_path: Optional[str] = None,
        context: Optional[str] = None,
        pooling_types: Optional[List[str]] = None,
        pooling: str = "mean",
        matrix_embed: bool = False,
        seed: int = 42,
        max_batch_tokens: int = 131072,
        embed_max_tokens: int = DEFAULT_EMBED_MAX_TOKENS,
        embed_similarity: float = DEFAULT_EMBED_SIMILARITY,
        min_query_similarity: float = 0.3,
        progress: bool = True,
    ) -> torch.Tensor | List[torch.Tensor]:
        if a3m_path is not None and context is None:
            spec = ContextSpecification(
                max_num_samples=511,
                max_token_length=embed_max_tokens,
                max_query_similarity=embed_similarity,
                min_query_similarity=min_query_similarity,
            )
            contexts, _ = sample_multiple_contexts(
                msa_path=a3m_path,
                context_specifications=[spec],
                seed=seed,
            )
            context = contexts[0] if contexts else None

        hidden_list = _forward_for_embedding(
            model=self,
            sequences=sequences,
            context=context,
            max_batch_tokens=max_batch_tokens,
            progress=progress,
        )
        if matrix_embed:
            return hidden_list
        if pooling_types is not None:
            return _pool_hidden_states(hidden_list, pooling_types, self.device)
        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls' when pooling_types is not provided")
        embeddings = [hidden.mean(dim=0) if pooling == "mean" else hidden[0] for hidden in hidden_list]
        return torch.stack(embeddings)

    @torch.inference_mode()
    def embed_dataset_with_msa(
        self,
        sequences: List[str],
        msa_lookup: Optional[Dict[str, str]] = None,
        msa_dir: Optional[str] = None,
        msa_hf_path: Optional[str] = None,
        batch_size: int = 2,
        max_len: int = 2048,
        pooling_types: Optional[List[str]] = None,
        pooling: str = "mean",
        matrix_embed: bool = False,
        embed_dtype: torch.dtype = torch.bfloat16,
        embed_max_tokens: int = DEFAULT_EMBED_MAX_TOKENS,
        embed_similarity: float = DEFAULT_EMBED_SIMILARITY,
        min_query_similarity: float = 0.3,
        seed: int = 42,
        progress: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if msa_lookup is None:
            if msa_dir is not None:
                msa_lookup = load_msa_dir(msa_dir)
            elif msa_hf_path is not None:
                msa_lookup = load_msa_from_hf(msa_hf_path)
            else:
                msa_lookup = {}

        truncated_map = {seq: seq[:max_len] for seq in sequences}
        unique_seqs = sorted(set(truncated_map.values()), key=len, reverse=True)
        context_map: Dict[str, Optional[str]] = {}
        spec = ContextSpecification(
            max_num_samples=511,
            max_token_length=embed_max_tokens,
            max_query_similarity=embed_similarity,
            min_query_similarity=min_query_similarity,
        )
        for seq in unique_seqs:
            a3m_path = get_msa_for_sequence(seq, msa_lookup)
            if a3m_path is None:
                context_map[seq] = None
                continue
            contexts, _ = sample_multiple_contexts(
                msa_path=a3m_path,
                context_specifications=[spec],
                seed=seed,
            )
            context_map[seq] = contexts[0] if contexts else None

        context_groups: Dict[Optional[str], List[str]] = defaultdict(list)
        for seq in unique_seqs:
            context_groups[context_map[seq]].append(seq)

        embeddings_dict: Dict[str, torch.Tensor] = {}
        total_batches = sum((len(seqs) + batch_size - 1) // batch_size for seqs in context_groups.values())
        pbar = tqdm(total=total_batches, desc="Embedding with MSA", disable=not progress)
        for ctx, ctx_seqs in context_groups.items():
            for i in range(0, len(ctx_seqs), batch_size):
                batch_seqs = ctx_seqs[i:i + batch_size]
                batch_embeddings = self.embed_with_msa(
                    sequences=batch_seqs,
                    context=ctx,
                    pooling_types=pooling_types,
                    pooling=pooling,
                    matrix_embed=matrix_embed,
                    seed=seed,
                    embed_max_tokens=embed_max_tokens,
                    embed_similarity=embed_similarity,
                    min_query_similarity=min_query_similarity,
                    progress=False,
                )
                if matrix_embed:
                    assert isinstance(batch_embeddings, list)
                    for seq, hidden in zip(batch_seqs, batch_embeddings):
                        embeddings_dict[seq] = hidden.to(embed_dtype).cpu()
                else:
                    assert isinstance(batch_embeddings, torch.Tensor)
                    for j, seq in enumerate(batch_seqs):
                        embeddings_dict[seq] = batch_embeddings[j].to(embed_dtype).cpu()
                pbar.update(1)
        pbar.close()

        result: Dict[str, torch.Tensor] = {}
        for seq in sequences:
            trunc = truncated_map[seq]
            if trunc in embeddings_dict:
                result[seq] = embeddings_dict[trunc]
        return result


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        within_seq_position_ids: Optional[torch.LongTensor] = None,
        global_position_ids: Optional[torch.LongTensor] = None,
        sequence_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
        **kwargs,
    ) -> E1MaskedLMOutputWithPast:
        """
        Args:
            input_ids: (batch_size, seq_length)
            within_seq_position_ids: (batch_size, seq_length)
                This tensor contains the position of each residue within the sequence itself.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos><pad>"],
                the tensor would be [[0,1,2,3,4,5,6,0,1,2,3,4,5,6], [0,1,2,3,4,5,0,1,2,3,4,5,6,-1]]
            global_position_ids: (batch_size, seq_length)
                This tensor contains the position of each residue within the global sequence.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos>"],
                the tensor would be [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1]]
            sequence_ids: (batch_size, seq_length)
                This tensor contains the sequence id of each residue.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos>"],
                the tensor would be [[0,0,0,0,0,0,0,1,1,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1,1,1,-1]]
            inputs_embeds: (batch_size, seq_length, hidden_size) - pre-computed embeddings
            labels: (batch_size, seq_length)
            past_key_values: DynamicCache
            use_cache: bool
            output_attentions: bool
            output_hidden_states: bool
            output_s_max: bool

        Returns:
            E1MaskedLMOutputWithPast: Model Outputs
        """
        outputs: E1ModelOutputWithPast = self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )

        last_hidden_state = outputs.last_hidden_state
        loss = None

        mlm_logits = self.mlm_head(last_hidden_state).float()
        mlm_loss = 0.0
        if labels is not None:
            mlm_logits_flat = mlm_logits.contiguous().view(-1, self.config.vocab_size)
            mlm_labels_flat = labels.to(mlm_logits_flat.device).contiguous().view(-1)
            mlm_loss = F.cross_entropy(mlm_logits_flat, mlm_labels_flat, reduction="none")
            mask = mlm_labels_flat != self.model.padding_idx
            n_mlm = mask.sum()
            mlm_loss = (mlm_loss * mask.to(mlm_loss)).sum() / (1 if n_mlm == 0 else n_mlm)
            loss = 0.0
            loss += mlm_loss

        return E1MaskedLMOutputWithPast(
            loss=loss,
            mlm_loss=mlm_loss,
            logits=mlm_logits,
            last_hidden_state=last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class E1ForSequenceClassification(E1PreTrainedModel, EmbeddingMixin):
    config: E1Config
    config_class = E1Config
    def __init__(self, config: E1Config, **kwargs):
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.model: FAST_E1_ENCODER = FAST_E1_ENCODER(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.num_labels = config.num_labels
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 4),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size * 4),
            nn.Linear(config.hidden_size * 4, config.num_labels),
        )
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.prep_tokens = self.model.prep_tokens

        if 'pooling_types' in kwargs and isinstance(kwargs['pooling_types'], List[str]) and len(kwargs['pooling_types']) > 0:
            pooling_types = kwargs['pooling_types']
        else:
            pooling_types = ['mean', 'var']
        self.pooler = Pooler(pooling_types)
        self.post_init()

    @property
    def device_mesh(self) -> torch.distributed.device_mesh.DeviceMesh:
        return self.model.device_mesh

    def _embed(self, sequences: List[str], return_attention_mask: bool = False, **kwargs) -> torch.Tensor:
        return self.model._embed(sequences, return_attention_mask=return_attention_mask, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        within_seq_position_ids: Optional[torch.LongTensor] = None,
        global_position_ids: Optional[torch.LongTensor] = None,
        sequence_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
        **kwargs,
    ) -> E1ClassificationOutputWithPast:
        outputs: E1ModelOutputWithPast = self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )

        attention_mask = (sequence_ids != -1).long() if sequence_ids is not None else torch.ones(outputs.last_hidden_state.shape[:2], device=outputs.last_hidden_state.device, dtype=torch.long)
        x = outputs.last_hidden_state
        features = self.pooler(x, attention_mask)
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = self.mse(logits.flatten(), labels.flatten())
                else:
                    loss = self.mse(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = self.ce(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = self.bce(logits, labels)

        return E1ClassificationOutputWithPast(
            loss=loss,
            logits=logits,
            last_hidden_state=x,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class E1ForTokenClassification(E1PreTrainedModel, EmbeddingMixin):
    config: E1Config
    config_class = E1Config
    def __init__(self, config: E1Config, **kwargs):
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.model: FAST_E1_ENCODER = FAST_E1_ENCODER(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.num_labels = config.num_labels
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 4),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size * 4),
            nn.Linear(config.hidden_size * 4, config.num_labels),
        )
        self.loss_fct = nn.CrossEntropyLoss()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.prep_tokens = self.model.prep_tokens
        self.post_init()

    @property
    def device_mesh(self) -> torch.distributed.device_mesh.DeviceMesh:
        return self.model.device_mesh

    def _embed(self, sequences: List[str], return_attention_mask: bool = False, **kwargs) -> torch.Tensor:
        return self.model._embed(sequences, return_attention_mask=return_attention_mask, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        within_seq_position_ids: Optional[torch.LongTensor] = None,
        global_position_ids: Optional[torch.LongTensor] = None,
        sequence_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
        **kwargs,
    ) -> E1ClassificationOutputWithPast:
        outputs: E1ModelOutputWithPast = self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )

        x = outputs.last_hidden_state
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return E1ClassificationOutputWithPast(
            loss=loss,
            logits=logits,
            last_hidden_state=x,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


if __name__ == "__main__":
    import random

    import torch

    from torch import Tensor

    def print_tensor_shapes(prefix: str, obj):
        if isinstance(obj, Tensor):
            print(f"{prefix}{obj.shape}")
        elif isinstance(obj, dict):
            for name, value in obj.items():
                print_tensor_shapes(f"{prefix}{name}.", value)
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                print_tensor_shapes(f"{prefix}[{idx}].", value)
        elif isinstance(obj, tuple):
            for idx, value in enumerate(obj):
                print_tensor_shapes(f"{prefix}[{idx}].", value)
        elif hasattr(obj, "__dict__"):
            for name, value in vars(obj).items():
                if name.startswith("_"):
                    continue
                print_tensor_shapes(f"{prefix}{name}.", value)
        else:
            print(f"{prefix}{type(obj)}")

    def get_e1_batch(tokenizer, sequences: List[str], device: torch.device):
        preparer = E1BatchPreparer(data_prep_config=DataPrepConfig(max_num_positions_within_seq=64), tokenizer=tokenizer)
        return preparer.get_batch_kwargs(sequences=sequences, device=device)

    random.seed(0)
    torch.manual_seed(0)

    num_attention_heads = random.choice([2, 4])
    config = E1Config(
        hidden_size=16 * num_attention_heads,
        intermediate_size=64 * num_attention_heads,
        num_hidden_layers=random.choice([1, 2]),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        max_num_positions_within_seq=128,
        max_num_positions_global=256,
        max_num_sequences=8,
        dtype="float32",
    )
    model = E1ForMaskedLM(config=config).eval()
    tokenizer = get_tokenizer()
    batch = get_e1_batch(tokenizer=tokenizer, sequences=["ACDEFG", "MKTW"], device=torch.device("cpu"))
    batch["labels"] = batch["labels"].clone()

    with torch.no_grad():
        output = model(
            input_ids=batch["input_ids"],
            within_seq_position_ids=batch["within_seq_position_ids"],
            global_position_ids=batch["global_position_ids"],
            sequence_ids=batch["sequence_ids"],
            labels=batch["labels"],
        )

    print("Batch shape:")
    print_tensor_shapes("", batch)
    print("Output shape:")
    print_tensor_shapes("", output)
