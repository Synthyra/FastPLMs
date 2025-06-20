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
from typing import Optional, Tuple, Union, List, Callable, Dict
from einops import rearrange, repeat
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizerBase, PretrainedConfig
from transformers.modeling_outputs import ModelOutput


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
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the embedding."""
        inv_freq = self._compute_inv_freq(self.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        arange = torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32)
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
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
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

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

        if output_attentions: # Manual attention computation
            b, h, l, d = query_BHLD.shape
            scale = 1 / math.sqrt(d)
            attn_bias = torch.zeros(b, h, l, l, dtype=query_BLD.dtype, device=query_BLD.device)
            if attention_mask is not None:
                attn_bias.masked_fill_(attention_mask.logical_not(), float('-inf'))
            attn_weights = torch.matmul(query_BHLD, key_BHLD.transpose(-2, -1)) * scale
            attn_weights += attn_bias
            attn_weights = F.softmax(attn_weights, dim=-1)
            context_BHLD = torch.matmul(attn_weights, value_BHLD)
        else:
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, attention_mask
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
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = swiglu_ln_ffn(d_model, expansion_ratio)
        self.scaling_factor = residue_scaling_factor
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
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
        attn_output, attn_weights = self.attn(x, attention_mask, output_attentions)
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
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    residue_scaling_factor=math.sqrt(n_layers / 36),
                    dropout=dropout,
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
        batch_size, seq_len, _ = x.shape
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len).bool()
            
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, attn_weights = self._gradient_checkpointing_func(
                    block.__call__,
                    x,
                    attention_mask,
                    output_attentions,
                )
            else:
                x, attn_weights = block(x, attention_mask, output_attentions)

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


### Support for embedding datasets with low code
class Pooler:
    def __init__(self, pooling_types: List[str]):
        self.pooling_types = pooling_types
        self.pooling_options = {
            'mean': self.mean_pooling,
            'max': self.max_pooling,
            'min': self.min_pooling,
            'norm': self.norm_pooling,
            'prod': self.prod_pooling,
            'median': self.median_pooling,
            'std': self.std_pooling,
            'var': self.var_pooling,
            'cls': self.cls_pooling,
        }

    def mean_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.mean(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    def max_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.max(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).max(dim=1).values
    
    def min_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.min(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).min(dim=1).values

    def norm_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.norm(dim=1, p=2)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).norm(dim=1, p=2)

    def prod_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        length = emb.shape[1]
        if attention_mask is None:
            return emb.prod(dim=1) / length
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return ((emb * attention_mask).prod(dim=1) / attention_mask.sum(dim=1)) / length

    def median_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.median(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).median(dim=1).values
    
    def std_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.std(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).std(dim=1)
    
    def var_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.var(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).var(dim=1)

    def cls_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        return emb[:, 0, :]

    def __call__(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # [mean, max]
        final_emb = []
        for pooling_type in self.pooling_types:
            final_emb.append(self.pooling_options[pooling_type](emb, attention_mask)) # (b, d)
        return torch.cat(final_emb, dim=-1) # (b, n_pooling_types * d)


class ProteinDataset(TorchDataset):
    """Simple dataset for protein sequences."""
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def build_collator(tokenizer) -> Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]:
    def _collate_fn(sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function for batching sequences."""
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
        import sqlite3
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

    def embed_dataset(
        self,
        sequences: List[str],
        tokenizer: PreTrainedTokenizerBase,
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
    ) -> Optional[dict[str, torch.Tensor]]:
        """Embed a dataset of protein sequences.
        
        Args:
            sequences: List of protein sequences
            batch_size: Batch size for processing
            max_len: Maximum sequence length
            full_embeddings: Whether to return full residue-wise (True) embeddings or pooled (False)
            pooling_type: Type of pooling ('mean' or 'cls')
            num_workers: Number of workers for data loading, 0 for the main process
            sql: Whether to store embeddings in SQLite database - will be stored in float32
            sql_db_path: Path to SQLite database
            
        Returns:
            Dictionary mapping sequences to embeddings, or None if sql=True

        Note:
            - If sql=True, embeddings can only be stored in float32
            - sql is ideal if you need to stream a very large dataset for training in real-time
            - save=True is ideal if you can store the entire embedding dictionary in RAM
            - sql will be used if it is True and save is True or False
            - If your sql database or .pth file is already present, they will be scanned first for already embedded sequences
            - Sequences will be truncated to max_len and sorted by length in descending order for faster processing

        Example:
            >>> embedder = EmbeddingMixin()
            >>> embedding_dict = embedder.embed_dataset(
                sequences=[
                    'MALWMRLLPLLALLALWGPDPAAA', ... # list of protein sequences
                ],
                batch_size=2, # adjust for your GPU memory
                max_len=512, # adjust for your needs
                full_embeddings=False, # if True, no pooling is performed
                embed_dtype=torch.float32, # cast to what dtype you want
                pooling_type=['mean', 'cls'], # more than one pooling type will be concatenated together
                num_workers=0, # if you have many cpu cores, we find that num_workers = 4 is fast for large datasets
                sql=False, # if True, embeddings will be stored in SQLite database
                sql_db_path='embeddings.db',
                save=True, # if True, embeddings will be saved as a .pth file
                save_path='embeddings.pth',
            )
            >>> # embedding_dict is a dictionary mapping sequences to their embeddings as tensors for .pth or numpy arrays for sql
        """
        sequences = list(set([seq[:max_len] if truncate else seq for seq in sequences]))
        sequences = sorted(sequences, key=len, reverse=True)
        hidden_size = self.config.hidden_size
        collate_fn = build_collator(tokenizer)
        device = self.device
        pooler = Pooler(pooling_types) if not full_embeddings else None

        def get_embeddings(residue_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            if full_embeddings or residue_embeddings.ndim == 2: # if already pooled or want residue-wise embeddings
                return residue_embeddings
            else:
                return pooler(residue_embeddings, attention_mask)

        if sql:
            import sqlite3
            conn = sqlite3.connect(sql_db_path)
            c = conn.cursor()
            c.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')
            already_embedded = self._read_sequences_from_db(sql_db_path)
            to_embed = [seq for seq in sequences if seq not in already_embedded]
            print(f"Found {len(already_embedded)} already embedded sequences in {sql_db_path}")
            print(f"Embedding {len(to_embed)} new sequences")
            if len(to_embed) > 0:
                dataset = ProteinDataset(to_embed)
                dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
                with torch.no_grad():
                    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                        seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                        residue_embeddings = self._embed(input_ids, attention_mask).float() # sql requires float32
                        embeddings = get_embeddings(residue_embeddings, attention_mask)
                        for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                            if full_embeddings:
                                emb = emb[mask.bool()].reshape(-1, hidden_size)
                            c.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", 
                                    (seq, emb.cpu().numpy().tobytes()))
                        
                        if (i + 1) % 100 == 0:
                            conn.commit()
            
                conn.commit()
            conn.close()
            return None

        embeddings_dict = {}
        if os.path.exists(save_path):
            embeddings_dict = torch.load(save_path, map_location='cpu', weights_only=True)
            to_embed = [seq for seq in sequences if seq not in embeddings_dict]
            print(f"Found {len(embeddings_dict)} already embedded sequences in {save_path}")
            print(f"Embedding {len(to_embed)} new sequences")
        else:
            to_embed = sequences
            print(f"Embedding {len(to_embed)} new sequences")

        if len(to_embed) > 0:
            dataset = ProteinDataset(to_embed)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                    seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    residue_embeddings = self._embed(input_ids, attention_mask)
                    embeddings = get_embeddings(residue_embeddings, attention_mask).to(embed_dtype)
                    for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                        if full_embeddings:
                            emb = emb[mask.bool()].reshape(-1, hidden_size)
                        embeddings_dict[seq] = emb.cpu()

        if save:
            torch.save(embeddings_dict, save_path)

        return embeddings_dict

class PreTrainedESMplusplusModel(PreTrainedModel):
    """
    init weights for ESM++ models
    """
    config_class = ESMplusplusConfig
    base_model_prefix = "esm++"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
        self.transformer = TransformerStack(config.hidden_size, config.num_attention_heads, config.num_hidden_layers, config.dropout)
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
        self.transformer = TransformerStack(config.hidden_size, config.num_attention_heads, config.num_hidden_layers, config.dropout)
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
        self.pooler = Pooler(['cls','mean'])
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
@staticmethod
@cache
def data_root(model: str):
    if "INFRA_PROVIDER" in os.environ:
        return Path("")
    # Try to download from hugginface if it doesn't exist
    if model.startswith("esmc-300"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12"))
    elif model.startswith("esmc-600"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-600m-2024-12"))
    else:
        raise ValueError(f"{model=} is an invalid model name.")
    return path


def ESMplusplus_300M(device: torch.device | str = "cpu"):
    with torch.device(device):
        config = ESMplusplusConfig(
            hidden_size=960,
            num_attention_heads=15,
            num_hidden_layers=30,
        )
        model = ESMplusplusForMaskedLM(config)
    state_dict = torch.load(
        data_root("esmc-300") / "data/weights/esmc_300m_2024_12_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


def ESMplusplus_600M(device: torch.device | str = "cpu"):
    with torch.device(device):
        config = ESMplusplusConfig(
            hidden_size=1152,
            num_attention_heads=18,
            num_hidden_layers=36,
        )
        model = ESMplusplusForMaskedLM(config)
    state_dict = torch.load(
        data_root("esmc-600") / "data/weights/esmc_600m_2024_12_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


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


if __name__ == "__main__":    
    # Set device to CPU for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test tokenizer
    tokenizer = EsmSequenceTokenizer()
    sample_sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    encoding = tokenizer(sample_sequence, return_tensors="pt")
    print(f"Input sequence length: {len(sample_sequence)}")
    print(f"Tokenized sequence: {encoding['input_ids'].shape}")
    
    # Prepare inputs
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Test base model with smaller config for quick testing
    print("\n=== Testing ESMplusplus Base Model ===")
    base_config = ESMplusplusConfig(
        hidden_size=384,
        num_attention_heads=6,
        num_hidden_layers=4
    )
    base_model = ESMplusplusModel(base_config).to(device)
    
    with torch.no_grad():
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
    
    # Test embedding functionality
    print("\nTesting embedding functionality:")
    with torch.no_grad():
        embeddings = base_model._embed(input_ids, attention_mask)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Test masked language modeling
    print("\n=== Testing ESMplusplus For Masked LM ===")
    mlm_model = ESMplusplusForMaskedLM(base_config).to(device)
    
    with torch.no_grad():
        outputs = mlm_model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
    print(f"Logits shape: {outputs.logits.shape}")
    
    # Test sequence classification model
    print("\n=== Testing Sequence Classification Model ===")
    classification_model = ESMplusplusForSequenceClassification(base_config).to(device)
    
    with torch.no_grad():
        outputs = classification_model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
    print(f"Logits shape: {outputs.logits.shape}")
    
    # Test token classification model
    print("\n=== Testing Token Classification Model ===")
    token_model = ESMplusplusForTokenClassification(base_config).to(device)
    
    with torch.no_grad():
        outputs = token_model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
    print(f"Logits shape: {outputs.logits.shape}")
    
    # Test embedding dataset functionality with a mini dataset
    print("\n=== Testing Embed Dataset Functionality ===")
    mini_dataset = [sample_sequence, sample_sequence[:50], sample_sequence[:30]]
    print(f"Creating embeddings for {len(mini_dataset)} sequences")
    
    # Only run this if save path doesn't exist to avoid overwriting
    if not os.path.exists("test_embeddings.pth"):
        embeddings = mlm_model.embed_dataset(
            sequences=mini_dataset,
            tokenizer=tokenizer,
            batch_size=2,
            max_len=100,
            full_embeddings=False,
            pooling_types=['mean'],
            save_path="test_embeddings.pth"
        )
        if embeddings:
            print(f"Embedding dictionary size: {len(embeddings)}")
            for seq, emb in embeddings.items():
                print(f"Sequence length: {len(seq)}, Embedding shape: {emb.shape}")
                break
    else:
        print("Skipping embedding test as test_embeddings.pth already exists")
    
    print("\nAll tests completed successfully!")
    
