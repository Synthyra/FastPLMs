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


"""
ESM++ model implementation.

ESM++ is a faithful implementation of ESMC that allows for batching and standard Huggingface compatibility
The ESM Python package is not required

Modified from https://github.com/evolutionaryscale/esm
License: https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Optional, Tuple, Union, List
from einops import rearrange, repeat
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
try:
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    create_block_mask = None
    flex_attention = None



def _create_pad_block_mask(attention_mask_2d: torch.Tensor):
    assert create_block_mask is not None, "Flex attention block mask requires create_block_mask."
    token_valid = attention_mask_2d.bool()
    batch_size, seq_len = token_valid.shape

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        return token_valid[batch_idx, q_idx] & token_valid[batch_idx, kv_idx]

    return create_block_mask(
        mask_mod,
        batch_size,
        1,
        seq_len,
        seq_len,
        device=attention_mask_2d.device,
    )


class ESMplusplusConfig(PretrainedConfig):
    """Configuration class for ESM++ model.
    
    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Dimension of hidden layers
        num_attention_heads: Number of attention heads
        num_hidden_layers: Number of transformer layers
        num_labels: Number of output labels for classification
        problem_type: Type of problem - regression, single/multi label classification
    """
    model_type = "ESMplusplus"
    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 960,
        num_attention_heads: int = 15,
        num_hidden_layers: int = 30,
        num_labels: int = 2,
        problem_type: str | None = None,
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        attn_backend: str = "sdpa",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.tie_word_embeddings = False
        self.attn_backend = attn_backend


### Rotary Embeddings
def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    _inplace: bool = False,
) -> torch.Tensor:
    """Apply rotary embeddings to input based on cos and sin."""
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    cos = repeat(cos, "s d -> s 1 (2 d)")
    sin = repeat(sin, "s d -> s 1 (2 d)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class RotaryEmbedding(torch.nn.Module):
    """Rotary position embeddings.
    
    Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    Args:
        dim: Dimension of the embedding
        base: Base for computing angular frequencies
        interleaved: Whether to use interleaved rotations
        scale_base: Base for scaling
        scaling_factor: Factor for scaling positions
        pos_idx_in_fp32: Whether to compute position indices in fp32
        device: Computation device
    """
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        scaling_factor: float = 1.0,
        pos_idx_in_fp32: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.device = device

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self._inv_freq_compute_device: Optional[torch.device] = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the embedding."""
        if "inv_freq" in self._buffers and isinstance(self._buffers["inv_freq"], torch.Tensor):
            buffer_device = self._buffers["inv_freq"].device
        else:
            buffer_device = self.device
        inv_freq = self._compute_inv_freq(buffer_device)
        self._inv_freq_compute_device = inv_freq.device
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        arange = torch.arange(0, self.dim, 2, device=buffer_device, dtype=torch.float32)
        scale = (
            (arange + 0.4 * self.dim) / (1.4 * self.dim)
            if self.scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute inverse frequency bands."""
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Update the cached cosine and sine values."""
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)

            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.
        
        Args:
            q: Query tensor of shape (batch, seqlen, nheads, headdim)
            k: Key tensor of shape (batch, seqlen, nheads, headdim)
            
        Returns:
            Tuple of rotated query and key tensors
        """
        assert self._inv_freq_compute_device is not None, "Rotary inv_freq compute device should be set after initialization."
        if self._inv_freq_compute_device != q.device:
            self.reset_parameters()
        self._update_cos_sin_cache(q.shape[1], device=q.device, dtype=q.dtype)
        assert self._cos_cached is not None
        assert self._sin_cached is not None
        if self.scale is None:
            return (
                apply_rotary_emb_torch(
                    q,
                    self._cos_cached,
                    self._sin_cached,
                    self.interleaved,
                    True,  # inplace=True
                ),
                apply_rotary_emb_torch(
                    k,
                    self._cos_cached,
                    self._sin_cached,
                    self.interleaved,
                    True,  # inplace=True
                ),
            )  # type: ignore
        else:
            assert False


### Feedforward Network Components
def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    """Compute corrected dimension for SwiGLU."""
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float) -> nn.Sequential:
    """Create SwiGLU feedforward network with layer normalization."""
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=False
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=False),
    )


### Attention
class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary embeddings.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.attn_backend = attn_backend
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=False)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_ln = nn.LayerNorm(d_model, bias=False)
        self.k_ln = nn.LayerNorm(d_model, bias=False)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key."""
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor after self attention, and optionally attention weights
        """
        attn_weights = None
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)
        query_BHLD, key_BHLD, value_BHLD = map(self.reshaper, (query_BLD, key_BLD, value_BLD))
        scale = 1 / math.sqrt(self.d_head)

        if output_attentions: # Manual attention computation
            b, h, l, _ = query_BHLD.shape
            attn_bias = torch.zeros(b, h, l, l, dtype=query_BLD.dtype, device=query_BLD.device)
            if attention_mask is not None:
                attn_bias.masked_fill_(attention_mask.logical_not(), float('-inf'))
            attn_weights = torch.matmul(query_BHLD, key_BHLD.transpose(-2, -1)) * scale
            attn_weights += attn_bias
            attn_weights = F.softmax(attn_weights, dim=-1)
            context_BHLD = torch.matmul(attn_weights, value_BHLD)
        else:
            if self.attn_backend == "flex":
                assert flex_attention is not None, "Flex attention backend requested but torch.flex_attention is unavailable."
                assert query_BHLD.dtype in (torch.float16, torch.bfloat16), (
                    f"Flex attention backend requires float16 or bfloat16, got {query_BHLD.dtype}."
                )
                if attention_mask is not None:
                    assert flex_block_mask is not None, (
                        "Flex attention backend requires a block mask when attention_mask is provided."
                    )
                context_BHLD = flex_attention(
                    query_BHLD,
                    key_BHLD,
                    value_BHLD,
                    block_mask=flex_block_mask,
                    scale=scale,
                )
            else:
                sdpa_mask = None
                if attention_mask is not None:
                    sdpa_mask = torch.zeros_like(attention_mask, dtype=query_BHLD.dtype)
                    sdpa_mask.masked_fill_(attention_mask.logical_not(), float("-inf"))
                context_BHLD = F.scaled_dot_product_attention(
                    query_BHLD,
                    key_BHLD,
                    value_BHLD,
                    attn_mask=sdpa_mask,
                    scale=scale,
                )
            
        context_BLD = rearrange(context_BHLD, "b h s d -> b s (h d)")
        output = self.out_proj(context_BLD)
        return output, attn_weights


### Regression Head
def RegressionHead(d_model: int, output_dim: int, hidden_dim: Optional[int] = None) -> nn.Module:
    """Create a regression head with optional hidden dimension.
    
    Args:
        d_model: Input dimension
        output_dim: Output dimension
        hidden_dim: Optional hidden dimension (defaults to d_model)
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


### Transformer Block
class UnifiedTransformerBlock(nn.Module):
    """Transformer block with attention and feedforward layers.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        residue_scaling_factor: Factor for scaling residual connections
        expansion_ratio: Expansion ratio for feedforward network
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        residue_scaling_factor: float = 1,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.0,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_backend=attn_backend,
        )
        self.ffn = swiglu_ln_ffn(d_model, expansion_ratio)
        self.scaling_factor = residue_scaling_factor
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor after transformer block, and optionally attention weights
        """
        attn_output, attn_weights = self.attn(
            x,
            attention_mask,
            flex_block_mask,
            output_attentions,
        )
        x = x + self.dropout(attn_output) / self.scaling_factor
        x = x + self.dropout(self.ffn(x)) / self.scaling_factor
        return x, attn_weights


### Model Outputs
@dataclass
class TransformerOutput(ModelOutput):
    """Output type for transformer encoder."""
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


@dataclass
class ESMplusplusOutput(ModelOutput):
    """Output type for ESM++ models."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


### Transformer Stack
class TransformerStack(nn.Module):
    """Stack of transformer blocks.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout: Dropout rate
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        self.attn_backend = attn_backend
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    residue_scaling_factor=math.sqrt(n_layers / 36),
                    dropout=dropout,
                    attn_backend=attn_backend,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> TransformerOutput:
        """
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            TransformerOutput containing last hidden state and optionally all hidden states and attention weights
        """
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        
        if attention_mask is not None:
            assert attention_mask.ndim == 2, f"Expected 2D token attention mask, got shape {attention_mask.shape}."
            token_attention_mask = attention_mask.bool()
            if self.attn_backend == "flex" and not output_attentions:
                assert create_block_mask is not None, (
                    "Flex attention backend requested but torch.create_block_mask is unavailable."
                )
                flex_block_mask = _create_pad_block_mask(token_attention_mask)
                attention_mask = None
            else:
                pairwise_attention_mask = token_attention_mask.unsqueeze(-1) & token_attention_mask.unsqueeze(-2)
                attention_mask = pairwise_attention_mask.unsqueeze(1)
                flex_block_mask = None
        else:
            flex_block_mask = None
            
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, attn_weights = self._gradient_checkpointing_func(
                    block.__call__,
                    x,
                    attention_mask,
                    flex_block_mask,
                    output_attentions,
                )
            else:
                x, attn_weights = block(x, attention_mask, flex_block_mask, output_attentions)

            if attentions is not None:
                attentions += (attn_weights,)
                
            if output_hidden_states:
                assert hidden_states is not None
                hidden_states += (x,)
                
        return TransformerOutput(
            last_hidden_state=self.norm(x), 
            hidden_states=hidden_states,
            attentions=attentions
        )


class PreTrainedESMplusplusModel(PreTrainedModel):
    """
    init weights for ESM++ models
    """
    config_class = ESMplusplusConfig
    base_model_prefix = "esm++"
    supports_gradient_checkpointing = True
    all_tied_weights_keys = {}

    def _init_weights(self, module):
        """Initialize the weights"""
        # HF from_pretrained marks loaded parameters with `_is_hf_initialized`.
        # Skip this module if any local parameter is already marked as loaded.
        for parameter in module.parameters(recurse=False):
            if "_is_hf_initialized" in parameter.__dict__ and parameter.__dict__["_is_hf_initialized"]:
                return

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _reset_rotary_embeddings(self):
        """Refresh non-persistent rotary buffers after checkpoint loading."""
        for module in self.modules():
            if isinstance(module, RotaryEmbedding):
                module.reset_parameters()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        output_loading_info = bool(kwargs["output_loading_info"]) if "output_loading_info" in kwargs else False
        loaded = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if output_loading_info:
            model, loading_info = loaded
            model._reset_rotary_embeddings()
            return model, loading_info
        loaded._reset_rotary_embeddings()
        return loaded

    @classmethod
    def from_pretrained_esm(cls, model_name: str):
        """Load a pretrained ESM++ model."""
        if '300' in model_name:
            return ESMplusplus_300M()
        elif '600' in model_name:
            return ESMplusplus_600M()
        else:
            raise ValueError(f"Invalid model name: {model_name}")


### ESM++ Models
class ESMplusplusModel(PreTrainedESMplusplusModel, EmbeddingMixin):
    """
    ESM++ model. transformer model with no heads
    """
    config_class = ESMplusplusConfig
    def __init__(self, config: ESMplusplusConfig, **kwargs):
        PreTrainedESMplusplusModel.__init__(self, config, **kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(self.vocab_size, config.hidden_size)
        self.transformer = TransformerStack(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            n_layers=config.num_hidden_layers,
            dropout=config.dropout,
            attn_backend=config.attn_backend,
        )
        self.tokenizer = EsmSequenceTokenizer()
        self.init_weights()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.transformer(x, attention_mask, output_hidden_states=False, output_attentions=False).last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> TransformerOutput:
        """Forward pass for masked language modeling.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Optional precomputed embeddings
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            TransformerOutput containing last hidden state and optionally all hidden states and attention weights
        """
        if inputs_embeds is None:
            x = self.embed(input_ids)
        else:
            x = inputs_embeds
        return self.transformer(x, attention_mask, output_hidden_states, output_attentions)
        

class ESMplusplusForMaskedLM(PreTrainedESMplusplusModel, EmbeddingMixin):
    """
    ESM++ model for masked language modeling.
    Implements the base ESM++ architecture with a masked language modeling head.
    """
    config_class = ESMplusplusConfig
    def __init__(self, config: ESMplusplusConfig, **kwargs):
        PreTrainedESMplusplusModel.__init__(self, config, **kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(self.vocab_size, config.hidden_size)
        self.transformer = TransformerStack(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            n_layers=config.num_hidden_layers,
            dropout=config.dropout,
            attn_backend=config.attn_backend,
        )
        self.sequence_head = RegressionHead(config.hidden_size, self.vocab_size)
        self.ce_loss = nn.CrossEntropyLoss()
        self.tokenizer = EsmSequenceTokenizer()
        self.init_weights()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.sequence_head[-1]

    def set_output_embeddings(self, new_embeddings):
        self.sequence_head[-1] = new_embeddings

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.transformer(x, attention_mask, output_hidden_states=False, output_attentions=False).last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> ESMplusplusOutput:
        """Forward pass for masked language modeling.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Optional precomputed embeddings
            labels: Optional labels for masked tokens
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            ESMplusplusOutput containing loss, logits, hidden states and attention weights
        """
        if inputs_embeds is None:
            x = self.embed(input_ids)
        else:
            x = inputs_embeds
        output = self.transformer(x, attention_mask, output_hidden_states, output_attentions)
        x = output.last_hidden_state
        logits = self.sequence_head(x)
        loss = None
        if labels is not None:
            loss = self.ce_loss(logits.view(-1, self.vocab_size), labels.view(-1))
        return ESMplusplusOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=x,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class ESMplusplusForSequenceClassification(ESMplusplusForMaskedLM, EmbeddingMixin):
    """
    ESM++ model for sequence classification.
    Extends the base ESM++ model with a classification head.
    """
    def __init__(self, config: ESMplusplusConfig, **kwargs):
        ESMplusplusForMaskedLM.__init__(self, config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.classifier = RegressionHead(config.hidden_size * 2, config.num_labels, config.hidden_size * 4)
        # Large intermediate projections help with sequence classification tasks (*4)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        # if kwargs has pooling_types, use them, otherwise use ['cls', 'mean']
        if 'pooling_types' in kwargs and isinstance(kwargs['pooling_types'], List[str]) and len(kwargs['pooling_types']) > 0:
            pooling_types = kwargs['pooling_types']
        else:
            pooling_types = ['mean', 'var']
        self.pooler = Pooler(pooling_types)
        self.init_weights()

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.transformer(x, attention_mask, output_hidden_states=False, output_attentions=False).last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> ESMplusplusOutput:
        """Forward pass for sequence classification.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Optional precomputed embeddings
            labels: Optional labels for classification
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            ESMplusplusOutput containing loss, logits, and hidden states
        """
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        x = output.last_hidden_state
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

        return ESMplusplusOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=x,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class ESMplusplusForTokenClassification(ESMplusplusForMaskedLM, EmbeddingMixin):
    """
    ESM++ model for token classification.
    Extends the base ESM++ model with a token classification head.
    """
    def __init__(self, config: ESMplusplusConfig, **kwargs):
        ESMplusplusForMaskedLM.__init__(self, config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.classifier = RegressionHead(config.hidden_size, config.num_labels, config.hidden_size * 4)
        # Large intermediate projections help with sequence classification tasks (*4)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.transformer(x, attention_mask, output_hidden_states=False, output_attentions=False).last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> ESMplusplusOutput:
        """Forward pass for token classification.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Optional precomputed embeddings
            labels: Optional labels for token classification
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            ESMplusplusOutput containing loss, logits, and hidden states
        """
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        x = output.last_hidden_state
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return ESMplusplusOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=x,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


### Loading from EvolutionaryScale
_ESMC_CHECKPOINT_SPECS = {
    "esmc-300": {
        "repo_id": "EvolutionaryScale/esmc-300m-2024-12",
        "weights_relpath": "data/weights/esmc_300m_2024_12_v0.pth",
        "hidden_size": 960,
        "num_attention_heads": 15,
        "num_hidden_layers": 30,
    },
    "esmc-600": {
        "repo_id": "EvolutionaryScale/esmc-600m-2024-12",
        "weights_relpath": "data/weights/esmc_600m_2024_12_v0.pth",
        "hidden_size": 1152,
        "num_attention_heads": 18,
        "num_hidden_layers": 36,
    },
}


def _resolve_esmc_checkpoint_key(model: str) -> str:
    if "esmc-300" in model:
        return "esmc-300"
    if "esmc-600" in model:
        return "esmc-600"
    raise ValueError(f"{model=} is an invalid ESMC model name.")


@staticmethod
@cache
def data_root(model: str):
    if "INFRA_PROVIDER" in os.environ:
        return Path("")
    key = _resolve_esmc_checkpoint_key(model)
    return Path(snapshot_download(repo_id=_ESMC_CHECKPOINT_SPECS[key]["repo_id"]))


def get_esmc_checkpoint_path(model: str) -> Path:
    key = _resolve_esmc_checkpoint_key(model)
    return data_root(key) / _ESMC_CHECKPOINT_SPECS[key]["weights_relpath"]


def _load_esmc_checkpoint_model(
    config: ESMplusplusConfig,
    model: str,
    device: torch.device | str = "cpu",
) -> ESMplusplusForMaskedLM:
    key = _resolve_esmc_checkpoint_key(model)
    spec = _ESMC_CHECKPOINT_SPECS[key]
    assert config.hidden_size == spec["hidden_size"], (
        f"ESMC loader expected hidden_size={spec['hidden_size']} for {key}, "
        f"but got {config.hidden_size}."
    )
    assert config.num_attention_heads == spec["num_attention_heads"], (
        f"ESMC loader expected num_attention_heads={spec['num_attention_heads']} for {key}, "
        f"but got {config.num_attention_heads}."
    )
    assert config.num_hidden_layers == spec["num_hidden_layers"], (
        f"ESMC loader expected num_hidden_layers={spec['num_hidden_layers']} for {key}, "
        f"but got {config.num_hidden_layers}."
    )
    with torch.device(device):
        model_obj = ESMplusplusForMaskedLM(config)
    state_dict = torch.load(get_esmc_checkpoint_path(key), map_location=device)
    model_obj.load_state_dict(state_dict)
    return model_obj


def ESMplusplus_300M(device: torch.device | str = "cpu"):
    config = ESMplusplusConfig(
        hidden_size=960,
        num_attention_heads=15,
        num_hidden_layers=30,
    )
    return _load_esmc_checkpoint_model(config=config, model="esmc-300", device=device)


def ESMplusplus_600M(device: torch.device | str = "cpu"):
    config = ESMplusplusConfig(
        hidden_size=1152,
        num_attention_heads=18,
        num_hidden_layers=36,
    )
    return _load_esmc_checkpoint_model(config=config, model="esmc-600", device=device)


### Tokenization
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]

class EsmSequenceTokenizer(PreTrainedTokenizerFast):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        chain_break_token="|",
        **kwargs,
    ):
        all_tokens = SEQUENCE_VOCAB
        token_to_id = {tok: ind for ind, tok in enumerate(all_tokens)}

        # a character-level tokenizer is the same as BPE with no token merges
        bpe = BPE(token_to_id, merges=[], unk_token=unk_token)
        tokenizer = Tokenizer(bpe)
        special_tokens = [
            cls_token,
            pad_token,
            mask_token,
            eos_token,
            chain_break_token,
        ]
        self.cb_token = chain_break_token
        additional_special_tokens = [chain_break_token]

        tokenizer.add_special_tokens(special_tokens)

        # This is where we configure the automatic addition of special tokens when we call
        # tokenizer(text, add_special_tokens=True). Note that you can also configure how two
        # sequences are merged if you want.
        tokenizer.post_processor = TemplateProcessing(  # type: ignore
            single="<cls> $A <eos>",
            pair="<cls>:0 $A:0 <eos>:0 $B:1 <eos>:1",
            special_tokens=[
                ("<cls>", tokenizer.token_to_id("<cls>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )
        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    # These are a footgun, we never use the `bos` token anywhere so we're just overriding it here.
    @property
    def bos_token(self):
        return self.cls_token

    @property
    def bos_token_id(self):
        return self.cls_token_id

    @property
    def chain_break_token(self):
        return self.cb_token

    @property
    def chain_break_token_id(self):
        return self.convert_tokens_to_ids(self.chain_break_token)

    @property
    def all_token_ids(self):
        return list(range(self.vocab_size))

    @property
    def special_token_ids(self):
        return self.all_special_ids


