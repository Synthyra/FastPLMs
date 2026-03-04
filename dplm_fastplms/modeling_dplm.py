### Embedding Mixin + Pooler
import os
import sqlite3
import networkx as nx
import numpy as np
import torch
from tqdm.auto import tqdm
from typing import Callable, List, Optional
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase


class Pooler:
    def __init__(self, pooling_types: List[str]):
        self.pooling_types = pooling_types
        self.pooling_options = {
            'mean': self.mean_pooling,
            'max': self.max_pooling,
            'norm': self.norm_pooling,
            'median': self.median_pooling,
            'std': self.std_pooling,
            'var': self.var_pooling,
            'cls': self.cls_pooling,
            'parti': self._pool_parti,
        }

    def _create_pooled_matrices_across_layers(self, attentions: torch.Tensor) -> torch.Tensor:
        maxed_attentions = torch.max(attentions, dim=1)[0]
        return maxed_attentions

    def _page_rank(self, attention_matrix, personalization=None, nstart=None, prune_type="top_k_outdegree"):
        # Run PageRank on the attention matrix converted to a graph.
        # Raises exceptions if the graph doesn't match the token sequence or has no edges.
        # Returns the PageRank scores for each token node.
        G = self._convert_to_graph(attention_matrix)
        if G.number_of_nodes() != attention_matrix.shape[0]:
            raise Exception(
                f"The number of nodes in the graph should be equal to the number of tokens in sequence! You have {G.number_of_nodes()} nodes for {attention_matrix.shape[0]} tokens.")
        if G.number_of_edges() == 0:
            raise Exception(f"You don't seem to have any attention edges left in the graph.")

        return nx.pagerank(G, alpha=0.85, tol=1e-06, weight='weight', personalization=personalization, nstart=nstart, max_iter=100)

    def _convert_to_graph(self, matrix):
        # Convert a matrix (e.g., attention scores) to a directed graph using networkx.
        # Each element in the matrix represents a directed edge with a weight.
        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        return G

    def _calculate_importance_weights(self, dict_importance, attention_mask: Optional[torch.Tensor] = None):
        # Remove keys where attention_mask is 0
        if attention_mask is not None:
            for k in list(dict_importance.keys()):
                if attention_mask[k] == 0:
                    del dict_importance[k]

        #dict_importance[0] # remove cls
        #dict_importance[-1] # remove eos
        total = sum(dict_importance.values())
        return np.array([v / total for _, v in dict_importance.items()])

    def _pool_parti(self, emb: torch.Tensor, attentions: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        maxed_attentions = self._create_pooled_matrices_across_layers(attentions).numpy()
        # emb is (b, L, d), maxed_attentions is (b, L, L)
        emb_pooled = []
        for e, a, mask in zip(emb, maxed_attentions, attention_mask):
            dict_importance = self._page_rank(a)
            importance_weights = self._calculate_importance_weights(dict_importance, mask)
            num_tokens = int(mask.sum().item())
            emb_pooled.append(np.average(e[:num_tokens], weights=importance_weights, axis=0))
        pooled = torch.tensor(np.array(emb_pooled))
        return pooled

    def mean_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.mean(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    def max_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.max(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).max(dim=1).values

    def norm_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.norm(dim=1, p=2)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).norm(dim=1, p=2)

    def median_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.median(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).median(dim=1).values
    
    def std_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.std(dim=1)
        else:
            # Compute variance correctly over non-masked positions, then take sqrt
            var = self.var_pooling(emb, attention_mask, **kwargs)
            return torch.sqrt(var)
    
    def var_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.var(dim=1)
        else:
            # Correctly compute variance over only non-masked positions
            attention_mask = attention_mask.unsqueeze(-1)  # (b, L, 1)
            # Compute mean over non-masked positions
            mean = (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)  # (b, d)
            mean = mean.unsqueeze(1)  # (b, 1, d)
            # Compute squared differences from mean, only over non-masked positions
            squared_diff = (emb - mean) ** 2  # (b, L, d)
            # Sum squared differences over non-masked positions and divide by count
            var = (squared_diff * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)  # (b, d)
            return var

    def cls_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs): # (b, L, d) -> (b, d)
        return emb[:, 0, :]

    def __call__(
            self,
            emb: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            attentions: Optional[torch.Tensor] = None
        ): # [mean, max]
        final_emb = []
        for pooling_type in self.pooling_types:
            final_emb.append(self.pooling_options[pooling_type](emb=emb, attention_mask=attention_mask, attentions=attentions)) # (b, d)
        return torch.cat(final_emb, dim=-1) # (b, n_pooling_types * d)


class ProteinDataset(TorchDataset):
    """Simple dataset for protein sequences."""
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def build_collator(tokenizer: PreTrainedTokenizerBase) -> Callable[[list[str]], dict[str, torch.Tensor]]:
    def _collate_fn(sequences: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(sequences, return_tensors="pt", padding='longest')
    return _collate_fn


class EmbeddingMixin:
    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def _read_sequences_from_db(self, db_path: str) -> set[str]:
        """Read sequences from SQLite database."""
        sequences = []
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT sequence FROM embeddings")
            while True:
                row = c.fetchone()
                if row is None:
                    break
                sequences.append(row[0])
        return set(sequences)

    def _ensure_embeddings_table(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "sequence TEXT PRIMARY KEY, "
            "embedding BLOB NOT NULL, "
            "shape TEXT, "
            "dtype TEXT"
            ")"
        )
        cursor.execute("PRAGMA table_info(embeddings)")
        rows = cursor.fetchall()
        column_names = [row[1] for row in rows]
        if "shape" not in column_names:
            cursor.execute("ALTER TABLE embeddings ADD COLUMN shape TEXT")
        if "dtype" not in column_names:
            cursor.execute("ALTER TABLE embeddings ADD COLUMN dtype TEXT")
        conn.commit()

    def load_embeddings_from_pth(self, save_path: str) -> dict[str, torch.Tensor]:
        assert os.path.exists(save_path), f"Embedding file does not exist: {save_path}"
        payload = torch.load(save_path, map_location="cpu", weights_only=True)
        assert isinstance(payload, dict), "Expected .pth embeddings file to contain a dictionary."
        for sequence, tensor in payload.items():
            assert isinstance(sequence, str), "Expected embedding dictionary keys to be sequences (str)."
            assert isinstance(tensor, torch.Tensor), "Expected embedding dictionary values to be tensors."
        return payload

    def load_embeddings_from_db(self, db_path: str, sequences: Optional[List[str]] = None) -> dict[str, torch.Tensor]:
        assert os.path.exists(db_path), f"Embedding database does not exist: {db_path}"
        loaded: dict[str, torch.Tensor] = {}
        with sqlite3.connect(db_path) as conn:
            self._ensure_embeddings_table(conn)
            cursor = conn.cursor()
            if sequences is None:
                cursor.execute("SELECT sequence, embedding, shape, dtype FROM embeddings")
            else:
                if len(sequences) == 0:
                    return loaded
                placeholders = ",".join(["?"] * len(sequences))
                cursor.execute(
                    f"SELECT sequence, embedding, shape, dtype FROM embeddings WHERE sequence IN ({placeholders})",
                    tuple(sequences),
                )

            rows = cursor.fetchall()
            for row in rows:
                sequence = row[0]
                embedding_bytes = row[1]
                shape_text = row[2]
                dtype_text = row[3]
                assert shape_text is not None, "Missing shape metadata in embeddings table."
                assert dtype_text is not None, "Missing dtype metadata in embeddings table."
                shape_values = [int(value) for value in shape_text.split(",") if len(value) > 0]
                assert len(shape_values) > 0, f"Invalid shape metadata for sequence: {sequence}"
                expected_size = int(np.prod(shape_values))
                np_dtype = np.dtype(dtype_text)
                array = np.frombuffer(embedding_bytes, dtype=np_dtype)
                assert array.size == expected_size, f"Shape mismatch while reading sequence: {sequence}"
                reshaped = array.copy().reshape(tuple(shape_values))
                loaded[sequence] = torch.from_numpy(reshaped)
        return loaded

    def embed_dataset(
        self,
        sequences: List[str],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 2,
        max_len: int = 512,
        truncate: bool = True,
        full_embeddings: bool = False,
        embed_dtype: torch.dtype = torch.float32,
        pooling_types: List[str] = ['mean'],
        num_workers: int = 0,
        sql: bool = False,
        save: bool = True,
        sql_db_path: str = 'embeddings.db',
        save_path: str = 'embeddings.pth',
        **kwargs,
    ) -> Optional[dict[str, torch.Tensor]]:
        """
        Embed a dataset of protein sequences.

        Supports two modes:
        - Tokenizer mode (ESM2/ESM++): provide `tokenizer`, `_embed(input_ids, attention_mask)` is used.
        - Sequence mode (E1): pass `tokenizer=None`, `_embed(sequences, return_attention_mask=True, **kwargs)` is used.
        """
        sequences = list(set([seq[:max_len] if truncate else seq for seq in sequences]))
        sequences = sorted(sequences, key=len, reverse=True)
        hidden_size = self.config.hidden_size
        pooler = Pooler(pooling_types) if not full_embeddings else None
        tokenizer_mode = tokenizer is not None
        if tokenizer_mode:
            collate_fn = build_collator(tokenizer)
            device = self.device
        else:
            collate_fn = None
            device = None

        def get_embeddings(residue_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            if full_embeddings or residue_embeddings.ndim == 2:
                return residue_embeddings
            return pooler(residue_embeddings, attention_mask)

        def iter_batches(to_embed: List[str]):
            if tokenizer_mode:
                assert collate_fn is not None
                assert device is not None
                dataset = ProteinDataset(to_embed)
                dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
                for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                    seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    residue_embeddings = self._embed(input_ids, attention_mask)
                    yield seqs, residue_embeddings, attention_mask
            else:
                for batch_start in tqdm(range(0, len(to_embed), batch_size), desc='Embedding batches'):
                    seqs = to_embed[batch_start:batch_start + batch_size]
                    batch_output = self._embed(seqs, return_attention_mask=True, **kwargs)
                    assert isinstance(batch_output, tuple), "Sequence mode _embed must return (last_hidden_state, attention_mask)."
                    assert len(batch_output) == 2, "Sequence mode _embed must return exactly two values."
                    residue_embeddings, attention_mask = batch_output
                    assert isinstance(attention_mask, torch.Tensor), "Sequence mode _embed must return attention_mask as a torch.Tensor."
                    yield seqs, residue_embeddings, attention_mask

        if sql:
            conn = sqlite3.connect(sql_db_path)
            self._ensure_embeddings_table(conn)
            c = conn.cursor()
            already_embedded = self._read_sequences_from_db(sql_db_path)
            to_embed = [seq for seq in sequences if seq not in already_embedded]
            print(f"Found {len(already_embedded)} already embedded sequences in {sql_db_path}")
            print(f"Embedding {len(to_embed)} new sequences")
            if len(to_embed) > 0:
                with torch.no_grad():
                    for i, (seqs, residue_embeddings, attention_mask) in enumerate(iter_batches(to_embed)):
                        embeddings = get_embeddings(residue_embeddings, attention_mask).to(embed_dtype)
                        for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                            if full_embeddings:
                                emb = emb[mask.bool()].reshape(-1, hidden_size)
                            emb_np = emb.cpu().numpy()
                            emb_shape = ",".join([str(dim) for dim in emb_np.shape])
                            emb_dtype = str(emb_np.dtype)
                            c.execute(
                                "INSERT OR REPLACE INTO embeddings (sequence, embedding, shape, dtype) VALUES (?, ?, ?, ?)",
                                (seq, emb_np.tobytes(), emb_shape, emb_dtype),
                            )
                        if tokenizer_mode and (i + 1) % 100 == 0:
                            conn.commit()
                conn.commit()
            conn.close()
            return None

        embeddings_dict = {}
        if os.path.exists(save_path):
            embeddings_dict = self.load_embeddings_from_pth(save_path)
            to_embed = [seq for seq in sequences if seq not in embeddings_dict]
            print(f"Found {len(embeddings_dict)} already embedded sequences in {save_path}")
            print(f"Embedding {len(to_embed)} new sequences")
        else:
            to_embed = sequences
            print(f"Embedding {len(to_embed)} new sequences")

        if len(to_embed) > 0:
            with torch.no_grad():
                for seqs, residue_embeddings, attention_mask in iter_batches(to_embed):
                    embeddings = get_embeddings(residue_embeddings, attention_mask).to(embed_dtype)
                    for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                        if full_embeddings:
                            emb = emb[mask.bool()].reshape(-1, hidden_size)
                        embeddings_dict[seq] = emb.cpu()

        if save:
            torch.save(embeddings_dict, save_path)

        return embeddings_dict


# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
FastPLMs-compatible DPLM implementation.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from einops import rearrange

from transformers import EsmTokenizer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import (
    EsmAttention,
    EsmClassificationHead,
    EsmContactPredictionHead,
    EsmEmbeddings,
    EsmEncoder,
    EsmIntermediate,
    EsmLayer,
    EsmLMHead,
    EsmOutput,
    EsmPooler,
    EsmPreTrainedModel,
    EsmSelfAttention,
    EsmSelfOutput,
)

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask
except (ImportError, AttributeError):
    create_block_mask = None
    flex_attention = None
    BlockMask = None

from enum import Enum


### Kernels Flash Attention Detection
def _infer_kernels_flash_variant(kernel) -> str | None:
    if hasattr(kernel, "fwd") and hasattr(kernel, "varlen_fwd"):
        return "flash_attn2"
    if hasattr(kernel, "flash_attn_func") and hasattr(kernel, "flash_attn_varlen_func"):
        return "flash_attn3"
    return None


def _try_get_kernels_flash():
    try:
        from kernels import get_kernel
    except ImportError:
        return None, None

    flash_kernel = None
    flash_kernel_variant = None
    try:
        flash_kernel = get_kernel("kernels-community/flash-attn3")
        flash_kernel_variant = _infer_kernels_flash_variant(flash_kernel)
        assert flash_kernel_variant is not None, "Loaded flash-attn3 kernel does not expose a supported API."
    except Exception:
        try:
            flash_kernel = get_kernel("kernels-community/flash-attn2")
            flash_kernel_variant = _infer_kernels_flash_variant(flash_kernel)
            assert flash_kernel_variant is not None, "Loaded flash-attn2 kernel does not expose a supported API."
        except Exception:
            flash_kernel = None
            flash_kernel_variant = None
    return flash_kernel, flash_kernel_variant


FLASH_KERNEL, FLASH_KERNEL_VARIANT = _try_get_kernels_flash()


def _kernels_flash_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    assert FLASH_KERNEL is not None, "Kernel Flash Attention is not available in this environment."
    if FLASH_KERNEL_VARIANT == "flash_attn2":
        return FLASH_KERNEL.fwd(q=query_states, k=key_states, v=value_states, is_causal=causal)[0]
    if FLASH_KERNEL_VARIANT == "flash_attn3":
        try:
            output = FLASH_KERNEL.flash_attn_func(q=query_states, k=key_states, v=value_states, causal=causal)
        except TypeError:
            output = FLASH_KERNEL.flash_attn_func(query_states, key_states, value_states, 0.0, None, causal)
        if isinstance(output, tuple):
            return output[0]
        return output
    raise AssertionError(f"Unsupported kernels flash attention variant: {FLASH_KERNEL_VARIANT}")


def _kernels_flash_varlen_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_in_batch_q: int,
    max_seqlen_in_batch_k: int,
    causal: bool = False,
) -> torch.Tensor:
    assert FLASH_KERNEL is not None, "Kernel Flash Attention is not available in this environment."
    if FLASH_KERNEL_VARIANT == "flash_attn2":
        return FLASH_KERNEL.varlen_fwd(
            q=query_states, k=key_states, v=value_states,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k,
            is_causal=causal,
        )[0]
    if FLASH_KERNEL_VARIANT == "flash_attn3":
        try:
            output = FLASH_KERNEL.flash_attn_varlen_func(
                q=query_states, k=key_states, v=value_states,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k,
                causal=causal,
            )
        except TypeError:
            output = FLASH_KERNEL.flash_attn_varlen_func(
                query_states, key_states, value_states,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_in_batch_q, max_seqlen_in_batch_k,
                0.0, None, causal,
            )
        if isinstance(output, tuple):
            return output[0]
        return output
    raise AssertionError(f"Unsupported kernels flash attention variant: {FLASH_KERNEL_VARIANT}")


### Unpad / Pad helpers for varlen flash attention
class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0, indices.unsqueeze(1).expand(-1, second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor, None]:
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]], device=grad_output.device, dtype=grad_output.dtype
        )
        grad_input.scatter_(0, indices.unsqueeze(1).expand(-1, grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        return grad_output[indices], None, None


index_first_axis = IndexFirstAxis.apply
index_put_first_axis = IndexPutFirstAxis.apply


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def _unpad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
    batch_size, seq_len, num_heads, head_dim = query_layer.shape
    seqlens = attention_mask_2d.sum(dim=1).int()
    cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item())
    indices = attention_mask_2d.flatten().nonzero(as_tuple=False).flatten()
    query_layer = index_first_axis(query_layer.reshape(batch_size * seq_len, num_heads, head_dim), indices)
    key_layer = index_first_axis(key_layer.reshape(batch_size * seq_len, num_heads, head_dim), indices)
    value_layer = index_first_axis(value_layer.reshape(batch_size * seq_len, num_heads, head_dim), indices)
    return query_layer, key_layer, value_layer, indices, (cu_seqlens, cu_seqlens), (max_seqlen, max_seqlen)


def kernels_flash_attention_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask_2d: torch.Tensor | None = None,
    causal: bool = False,
) -> torch.Tensor:
    assert FLASH_KERNEL is not None, "Kernel Flash Attention is not available in this environment."
    if not causal and attention_mask_2d is not None:
        batch_size, q_len = query_states.shape[:2]
        (
            query_states, key_states, value_states,
            indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k),
        ) = _unpad_input(query_states, key_states, value_states, attention_mask_2d)
        attn_output_unpad = _kernels_flash_varlen_forward(
            query_states=query_states, key_states=key_states, value_states=value_states,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_in_batch_q=max_seqlen_q, max_seqlen_in_batch_k=max_seqlen_k,
        )
        return pad_input(attn_output_unpad, indices_q, batch_size, q_len)
    else:
        return _kernels_flash_forward(
            query_states=query_states, key_states=key_states, value_states=value_states, causal=causal,
        )


### Attention Backend Enum & Resolution
class AttentionBackend(Enum):
    AUTO = "auto"
    KERNELS_FLASH = "kernels_flash"
    FLEX = "flex"
    SDPA = "sdpa"


VALID_ATTENTION_BACKENDS = tuple(b.value for b in AttentionBackend)


def resolve_attention_backend(requested_backend: str) -> AttentionBackend:
    assert requested_backend in VALID_ATTENTION_BACKENDS, (
        f"Unsupported attention backend: {requested_backend}. Expected one of {VALID_ATTENTION_BACKENDS}."
    )
    if requested_backend == AttentionBackend.AUTO.value:
        if FLASH_KERNEL is not None:
            return AttentionBackend.KERNELS_FLASH
        if flex_attention is not None:
            return AttentionBackend.FLEX
        return AttentionBackend.SDPA
    if requested_backend == AttentionBackend.KERNELS_FLASH.value:
        assert FLASH_KERNEL is not None, "Kernels Flash Attention is not available in this environment."
        return AttentionBackend.KERNELS_FLASH
    if requested_backend == AttentionBackend.FLEX.value:
        assert flex_attention is not None, "Flex Attention is not available in this environment."
        return AttentionBackend.FLEX
    if requested_backend == AttentionBackend.SDPA.value:
        return AttentionBackend.SDPA
    raise AssertionError(f"Unsupported attention backend: {requested_backend}")


from transformers import PreTrainedTokenizerBase


class BaseSequenceTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, sequences, **kwargs):
        raise NotImplementedError


def get_attention_mask(
    effective_backend: AttentionBackend,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, "BlockMask | None"]:
    if attention_mask is None:
        return None, None, None

    attention_mask_2d = attention_mask.bool()

    if effective_backend == AttentionBackend.KERNELS_FLASH:
        return attention_mask_2d, None, None

    if effective_backend == AttentionBackend.FLEX:
        assert create_block_mask is not None, "Flex attention backend requested but torch.create_block_mask is unavailable."
        valid_lens = attention_mask_2d.sum(dim=-1)

        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            return (q_idx < valid_lens[batch_idx]) & (kv_idx < valid_lens[batch_idx])

        flex_block_mask = create_block_mask(mask_mod, batch_size, 1, seq_len, seq_len, device=device)
        return attention_mask_2d, None, flex_block_mask

    attention_mask_4d = attention_mask_2d[:, None, :, None] & attention_mask_2d[:, None, None, :]
    return attention_mask_2d, attention_mask_4d, None


@dataclass
class DPLMMaskedLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    s_max: Optional[Tuple[list[torch.Tensor], ...]] = None


@dataclass
class DPLMEncoderOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    s_max: Optional[Tuple[list[torch.Tensor], ...]] = None


class DPLMConfig(EsmConfig):
    model_type = "dplm"

    def __init__(
        self,
        attn_backend: str = "auto",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn_backend = attn_backend
        self.tie_word_embeddings = False


class DPLMPreTrainedModel(EsmPreTrainedModel):
    config_class = DPLMConfig
    base_model_prefix = "dplm"
    supports_gradient_checkpointing = True
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    all_tied_weights_keys = {}

    @classmethod
    def is_remote_code(cls) -> bool:
        # Prevent post-load reinitialization of tensors already loaded from checkpoints.
        return True

    @property
    def attn_backend(self) -> str:
        return self.config.attn_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        assert backend in VALID_ATTENTION_BACKENDS, f"Unsupported attn_backend: {backend}. Expected one of {VALID_ATTENTION_BACKENDS}."
        self.config.attn_backend = backend
        resolved = resolve_attention_backend(backend)
        for module in self.modules():
            if isinstance(module, ModifiedEsmEncoder):
                module.attention_backend = resolved
            elif isinstance(module, ModifiedEsmSelfAttention):
                module.attn_backend = resolved


class ModifiedEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.config = config
        self.scale = self.attention_head_size**-0.5
        self.attn_backend = resolve_attention_backend(config.attn_backend)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_s_max: Optional[bool] = False,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[torch.Tensor]]]:
        if past_key_values is not None:
            past_key_value = past_key_values

        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            cross_attn_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            cross_attn_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            cross_attn_mask = None
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            cross_attn_mask = None

        query_layer = self.transpose_for_scores(mixed_query_layer) * self.scale

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if self.position_embedding_type in ["relative_key", "relative_key_query"]:
            raise NotImplementedError

        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()

        if is_cross_attention:
            if output_attentions:
                attn_output, attn_weights, s_max = self._manual_attn(
                    query_layer, key_layer, value_layer, cross_attn_mask, output_s_max,
                )
            else:
                attn_output, attn_weights = self._sdpa_attn(
                    query_layer, key_layer, value_layer, cross_attn_mask,
                )
                s_max = self._compute_s_max(query_layer, key_layer) if output_s_max else None
        else:
            attn_output, attn_weights, s_max = self._attn(
                query_layer, key_layer, value_layer,
                attention_mask_2d=attention_mask_2d,
                attention_mask_4d=attention_mask_4d,
                flex_block_mask=flex_block_mask,
                output_attentions=output_attentions,
                output_s_max=output_s_max,
            )

        if head_mask is not None and torch.is_tensor(head_mask):
            batch_size, seq_len, _ = attn_output.shape
            attn_output = attn_output.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
            attn_output = attn_output.permute(0, 2, 1, 3) * head_mask
            attn_output = rearrange(attn_output, "b h s d -> b s (h d)")

        return attn_output, attn_weights, s_max

    def _attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: "BlockMask | None" = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        if output_attentions:
            return self._manual_attn(query_BHLD, key_BHLD, value_BHLD, attention_mask_4d, output_s_max)

        if self.attn_backend == AttentionBackend.KERNELS_FLASH:
            attn_output, attn_weights = self._kernels_flash_attn(query_BHLD, key_BHLD, value_BHLD, attention_mask_2d)
        elif self.attn_backend == AttentionBackend.FLEX:
            attn_output, attn_weights = self._flex_attn(query_BHLD, key_BHLD, value_BHLD, flex_block_mask)
        elif self.attn_backend == AttentionBackend.SDPA:
            attn_output, attn_weights = self._sdpa_attn(query_BHLD, key_BHLD, value_BHLD, attention_mask_4d)
        else:
            raise AssertionError(f"Unsupported resolved backend: {self.attn_backend}")

        s_max = self._compute_s_max(query_BHLD, key_BHLD) if output_s_max else None
        return attn_output, attn_weights, s_max

    @torch.no_grad()
    def _compute_s_max(self, query_BHLD: torch.Tensor, key_BHLD: torch.Tensor) -> list[torch.Tensor]:
        q_norm = torch.linalg.vector_norm(query_BHLD, dim=-1)
        k_norm = torch.linalg.vector_norm(key_BHLD, dim=-1)
        s_max_bound = (q_norm.max(dim=-1).values * k_norm.max(dim=-1).values).max(dim=0).values
        return [s_max_bound[h] for h in range(self.num_attention_heads)]

    def _manual_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        attention_mask_4d: torch.Tensor | None = None,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        attn_weights = torch.matmul(query_BHLD, key_BHLD.transpose(-1, -2))
        if attention_mask_4d is not None:
            attn_weights = attn_weights.masked_fill(attention_mask_4d.logical_not(), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        context_BHLD = torch.matmul(attn_weights, value_BHLD)
        attn_output = rearrange(context_BHLD, "b h s d -> b s (h d)")
        s_max = self._compute_s_max(query_BHLD, key_BHLD) if output_s_max else None
        return attn_output, attn_weights, s_max

    def _kernels_flash_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        query_BLHD = query_BHLD.transpose(1, 2).contiguous()
        key_BLHD = key_BHLD.transpose(1, 2).contiguous()
        value_BLHD = value_BHLD.transpose(1, 2).contiguous()
        attn_output = kernels_flash_attention_func(
            query_states=query_BLHD, key_states=key_BLHD, value_states=value_BLHD,
            attention_mask_2d=attention_mask_2d, causal=False,
        )
        return rearrange(attn_output, "b s h d -> b s (h d)"), None

    def _flex_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        flex_block_mask: "BlockMask | None" = None,
    ) -> tuple[torch.Tensor, None]:
        assert flex_attention is not None, "Flex attention is not available in this environment."
        assert query_BHLD.dtype in (torch.float16, torch.bfloat16), (
            f"Flex attention requires float16 or bfloat16, got {query_BHLD.dtype}."
        )
        context_BHLD = flex_attention(query_BHLD, key_BHLD, value_BHLD, block_mask=flex_block_mask, scale=1.0)
        return rearrange(context_BHLD, "b h s d -> b s (h d)"), None

    def _sdpa_attn(
        self,
        query_BHLD: torch.Tensor,
        key_BHLD: torch.Tensor,
        value_BHLD: torch.Tensor,
        attention_mask_4d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        context_BHLD = F.scaled_dot_product_attention(
            query_BHLD, key_BHLD, value_BHLD,
            attn_mask=attention_mask_4d,
            scale=1.0,
        )
        return rearrange(context_BHLD, "b h s d -> b s (h d)"), None


class ModifiedEsmAttention(EsmAttention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.self = ModifiedEsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[torch.Tensor]]]:
        hidden_states_ln = self.LayerNorm(hidden_states)
        attn_output, attn_weights, s_max = self.self(
            hidden_states_ln,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        attention_output = self.output(attn_output, hidden_states)
        return attention_output, attn_weights, s_max


class ModifiedEsmLayer(EsmLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ModifiedEsmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if self.is_decoder is False:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ModifiedEsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[torch.Tensor]]]:
        attention_output, attn_weights, s_max = self.attention(
            hidden_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            past_key_value=past_key_value[:2] if past_key_value is not None else None,
        )

        if self.is_decoder and encoder_hidden_states is not None:
            if self.add_cross_attention is False:
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention "
                    "layers by setting `config.add_cross_attention=True`"
                )
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_output, _, _ = self.crossattention(
                attention_output,
                attention_mask_2d=attention_mask_2d,
                attention_mask_4d=attention_mask_4d,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                output_s_max=False,
            )
            attention_output = cross_attention_output

        layer_output = self.feed_forward_chunk(attention_output)
        return layer_output, attn_weights, s_max


class ModifiedEsmEncoder(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.attention_backend = resolve_attention_backend(config.attn_backend)
        self.layer = nn.ModuleList([ModifiedEsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[Tuple[torch.FloatTensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
    ) -> DPLMEncoderOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        full_s_max = () if output_s_max else None

        attention_mask_2d, attention_mask_4d, flex_block_mask = get_attention_mask(
            effective_backend=self.attention_backend,
            batch_size=hidden_states.shape[0],
            seq_len=hidden_states.shape[1],
            device=hidden_states.device,
            attention_mask=attention_mask,
        )

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights, s_max = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask_2d,
                    attention_mask_4d,
                    flex_block_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    output_s_max,
                )
            else:
                hidden_states, attn_weights, s_max = layer_module(
                    hidden_states,
                    attention_mask_2d=attention_mask_2d,
                    attention_mask_4d=attention_mask_4d,
                    flex_block_mask=flex_block_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    output_s_max=output_s_max,
                )

            if all_self_attentions is not None:
                all_self_attentions = all_self_attentions + (attn_weights,)
            if full_s_max is not None:
                full_s_max = full_s_max + (s_max,)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return DPLMEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            s_max=full_s_max,
        )


class FAST_DPLM_ENCODER(DPLMPreTrainedModel, EmbeddingMixin):
    """Inner encoder class that holds the actual ESM-style weights (embeddings, encoder,
    contact_head) so that the weight keys are prefixed with 'esm.' in the outer DPLMModel,
    matching pretrained DPLM checkpoints."""

    def __init__(self, config, **kwargs):
        DPLMPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = ModifiedEsmEncoder(config)
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads,
            bias=True,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        embedding_output = self.embeddings(input_ids, attention_mask=attention_mask)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
        )
        return encoder_outputs.last_hidden_state

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attns = self(input_ids, attention_mask=attention_mask, output_attentions=True).attentions
        attns = torch.stack(attns, dim=1)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        return self.contact_head(input_ids, attns)

    def _convert_head_mask_to_5d(self, head_mask: torch.Tensor, num_hidden_layers: int) -> torch.Tensor:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, got {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    def get_head_mask(
        self,
        head_mask: Optional[torch.Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> Union[torch.Tensor, List[None]]:
        if head_mask is None:
            return [None] * num_hidden_layers
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DPLMEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask_2d = torch.ones((batch_size, seq_length), device=device).bool()
        elif attention_mask.dim() == 2:
            attention_mask_2d = attention_mask.bool()
        elif attention_mask.dim() == 4:
            assert input_ids is not None, "4D attention_mask requires input_ids to infer token-level mask."
            attention_mask_2d = input_ids.ne(self.config.pad_token_id)
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

        encoder_extended_attention_mask = encoder_attention_mask
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask_2d,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask_2d,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )
        sequence_output = encoder_outputs.last_hidden_state

        if return_dict is False:
            return (sequence_output,) + encoder_outputs[1:]

        return DPLMEncoderOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            s_max=encoder_outputs.s_max,
        )


class DPLMModel(DPLMPreTrainedModel, EmbeddingMixin):
    config_class = DPLMConfig

    def __init__(self, config, add_pooling_layer=True):
        DPLMPreTrainedModel.__init__(self, config)
        self.config = config
        self.esm = FAST_DPLM_ENCODER(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.esm.embeddings.word_embeddings = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DPLMEncoderOutput]:
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if return_dict is False:
            return (sequence_output, pooled_output) + outputs[1:]

        return DPLMEncoderOutput(
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class DPLMForMaskedLM(DPLMPreTrainedModel, EmbeddingMixin):
    config_class = DPLMConfig

    def __init__(self, config, dropout: float = 0.1):
        config.hidden_dropout_prob = dropout
        DPLMPreTrainedModel.__init__(self, config)
        self.esm = FAST_DPLM_ENCODER(config)
        self.lm_head = EsmLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

        self.tokenizer = self.__class__.tokenizer
        if isinstance(config._name_or_path, str) and len(config._name_or_path) > 0:
            try:
                self.tokenizer = EsmTokenizer.from_pretrained(config._name_or_path)
            except Exception:
                self.tokenizer = self.__class__.tokenizer

        self.mask_id = self.tokenizer.mask_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.x_id = self.tokenizer.convert_tokens_to_ids("X")
        self.contact_head = None

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], DPLMMaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if attention_mask is None and input_ids is not None:
            attention_mask = input_ids.ne(self.pad_id)

        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if return_dict is False:
            output = (logits, sequence_output, outputs.hidden_states, outputs.attentions)
            if loss is not None:
                return (loss,) + output
            return output

        return DPLMMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class DPLMForSequenceClassification(DPLMPreTrainedModel, EmbeddingMixin):
    config_class = DPLMConfig

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.get_input_embeddings()

    def __init__(self, config):
        DPLMPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.esm = FAST_DPLM_ENCODER(config)
        self.classifier = EsmClassificationHead(config)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.post_init()

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], DPLMMaskedLMOutput]:
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

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
                    loss = self.mse(logits.squeeze(), labels.squeeze())
                else:
                    loss = self.mse(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = self.ce(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = self.bce(logits, labels)

        return DPLMMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class DPLMForTokenClassification(DPLMPreTrainedModel, EmbeddingMixin):
    config_class = DPLMConfig

    def get_input_embeddings(self) -> nn.Module:
        return self.esm.get_input_embeddings()

    def __init__(self, config):
        DPLMPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.esm = FAST_DPLM_ENCODER(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], DPLMMaskedLMOutput]:
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            return_dict=True,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return DPLMMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


if __name__ == "__main__":
    import random

    import torch

    from torch import Tensor
    from transformers import EsmTokenizer

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

    random.seed(0)
    torch.manual_seed(0)

    num_attention_heads = random.choice([2, 4])
    config = DPLMConfig(
        hidden_size=16 * num_attention_heads,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=random.choice([1, 2]),
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        attn_backend="sdpa",
    )
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    batch = tokenizer(["ACDEFG", "MKTW"], return_tensors="pt", padding="longest")
    batch["labels"] = batch["input_ids"].clone()
    model = DPLMForMaskedLM(config=config).eval()

    with torch.no_grad():
        output = model(**batch, return_dict=True)

    print("Batch shape:")
    print_tensor_shapes("", batch)
    print("Output shape:")
    print_tensor_shapes("", output)
