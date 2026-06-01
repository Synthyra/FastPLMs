from __future__ import annotations

"""
Hugging Face compatible ESM3 wrapper.

This module keeps Biohub's ESM3 implementation as the execution core and adds
the FastPLMs conventions around it: AutoModel loading, sequence-only
`input_ids` forwarding, and direct multimodal track arguments.
"""

import sys
import math
import functools
import importlib
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PretrainedConfig
from transformers.modeling_outputs import ModelOutput

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
    )
except ImportError:
    BlockMask = None
    create_block_mask = None
    flex_attention = None


ESM3_OPEN_SMALL = "esm3_sm_open_v1"
ESM3_OPEN_SMALL_ALIASES = {
    "ESM3_small",
    "esm3_small",
    "esm3_sm_open_v1",
    "esm3-open-2024-03",
    "esm3-sm-open-v1",
    "esm3-open",
}

SEQUENCE_BOS_TOKEN = 0
SEQUENCE_PAD_TOKEN = 1
SEQUENCE_EOS_TOKEN = 2
SEQUENCE_CHAINBREAK_TOKEN = 31
SEQUENCE_MASK_TOKEN = 32

VQVAE_CODEBOOK_SIZE = 4096
STRUCTURE_MASK_TOKEN = VQVAE_CODEBOOK_SIZE
STRUCTURE_EOS_TOKEN = VQVAE_CODEBOOK_SIZE + 1
STRUCTURE_BOS_TOKEN = VQVAE_CODEBOOK_SIZE + 2
STRUCTURE_PAD_TOKEN = VQVAE_CODEBOOK_SIZE + 3
STRUCTURE_CHAINBREAK_TOKEN = VQVAE_CODEBOOK_SIZE + 4

SASA_PAD_TOKEN = 0
SS8_PAD_TOKEN = 0
INTERPRO_PAD_TOKEN = 0
RESIDUE_PAD_TOKEN = 0
MAX_RESIDUE_ANNOTATIONS = 16
FUNCTION_TOKENS_DEPTH = 8

SEQUENCE_VOCAB = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "|",
    "<mask>",
]

_SUPPORTED_ATTENTION_BACKENDS = ("auto", "flex", "sdpa")
_compiled_flex_attention = None


class AttentionBackend(Enum):
    AUTO = "auto"
    FLEX = "flex"
    SDPA = "sdpa"


def _get_flex_attention_fn():
    global _compiled_flex_attention
    if flex_attention is None:
        return None
    flex_mod = torch.nn.attention.flex_attention
    if getattr(flex_mod, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG", False):
        return flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(
            flex_attention,
            dynamic=False,
        )
    return _compiled_flex_attention


def resolve_attention_backend(requested_backend: str) -> AttentionBackend:
    assert requested_backend in _SUPPORTED_ATTENTION_BACKENDS, (
        f"Unsupported ESM3 attention backend: {requested_backend}. "
        f"Expected one of {_SUPPORTED_ATTENTION_BACKENDS}."
    )
    if requested_backend == AttentionBackend.AUTO.value:
        if flex_attention is not None:
            return AttentionBackend.FLEX
        return AttentionBackend.SDPA
    if requested_backend == AttentionBackend.FLEX.value:
        assert flex_attention is not None, "Flex Attention is not available in this environment."
        return AttentionBackend.FLEX
    if requested_backend == AttentionBackend.SDPA.value:
        return AttentionBackend.SDPA
    raise AssertionError(f"Unsupported ESM3 attention backend: {requested_backend}")

_ESM3_CHECKPOINT_SPECS = {
    ESM3_OPEN_SMALL: {
        "repo_id": "biohub/esm3-sm-open-v1",
        "hidden_size": 1536,
        "num_attention_heads": 24,
        "num_vector_heads": 256,
        "num_hidden_layers": 48,
    },
}


class FastESM3Config(PretrainedConfig):
    model_type = "fast_esm3"

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 1536,
        num_attention_heads: int = 24,
        num_vector_heads: int = 256,
        num_hidden_layers: int = 48,
        initializer_range: float = 0.02,
        attn_backend: str = "sdpa",
        model_name: str = ESM3_OPEN_SMALL,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert hidden_size % FUNCTION_TOKENS_DEPTH == 0
        assert hidden_size % num_attention_heads == 0
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_vector_heads = num_vector_heads
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.attn_backend = attn_backend
        self.model_name = _resolve_esm3_checkpoint_key(model_name)
        self.tie_word_embeddings = False


@dataclass
class FastESM3Output(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    sequence_logits: Optional[torch.Tensor] = None
    structure_logits: Optional[torch.Tensor] = None
    secondary_structure_logits: Optional[torch.Tensor] = None
    sasa_logits: Optional[torch.Tensor] = None
    function_logits: Optional[torch.Tensor] = None
    residue_logits: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None


class EsmSequenceTokenizer(PreTrainedTokenizerFast):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        unk_token: str = "<unk>",
        cls_token: str = "<cls>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        eos_token: str = "<eos>",
        chain_break_token: str = "|",
        **kwargs,
    ):
        token_to_id = {token: index for index, token in enumerate(SEQUENCE_VOCAB)}
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
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.post_processor = TemplateProcessing(
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
            additional_special_tokens=[chain_break_token],
            **kwargs,
        )

    @property
    def bos_token(self) -> str:
        return self.cls_token

    @property
    def bos_token_id(self) -> int:
        return self.cls_token_id

    @property
    def chain_break_token(self) -> str:
        return self.cb_token

    @property
    def chain_break_token_id(self) -> int:
        token_id = self.convert_tokens_to_ids(self.chain_break_token)
        assert isinstance(token_id, int)
        return token_id

    @property
    def all_token_ids(self) -> list[int]:
        return list(range(self.vocab_size))

    @property
    def special_token_ids(self) -> list[int]:
        return self.all_special_ids


@dataclass
class FastESM3TokenizerCollection:
    sequence: EsmSequenceTokenizer
    structure: Optional[object] = None
    secondary_structure: Optional[object] = None
    sasa: Optional[object] = None
    function: Optional[object] = None
    residue_annotations: Optional[object] = None


def rbf(values: torch.Tensor, v_min: float, v_max: float, n_bins: int = 16) -> torch.Tensor:
    centers = torch.linspace(
        v_min,
        v_max,
        n_bins,
        device=values.device,
        dtype=values.dtype,
    )
    centers = centers.view([1] * len(values.shape) + [-1])
    std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - centers) / std
    return torch.exp(-(z**2))


def RegressionHead(
    d_model: int,
    output_dim: int,
    hidden_dim: Optional[int] = None,
) -> nn.Module:
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return rearrange(
        torch.stack((-x2, x1), dim=-1),
        "... d two -> ... (d two)",
        two=2,
    )


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    cos = einops.repeat(cos, "s d -> s 1 (2 d)")
    sin = einops.repeat(sin, "s d -> s 1 (2 d)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class RotaryEmbedding(nn.Module):
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
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
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(
        self,
        seqlen: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
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
                inv_freq = self.inv_freq.to(torch.float32)
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)

            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                raise NotImplementedError("Scaled rotary embeddings are not used by ESM3.")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin_cache(
            q.shape[1] + seqlen_offset,
            device=q.device,
            dtype=q.dtype,
        )
        assert self._cos_cached is not None
        assert self._sin_cached is not None
        return (
            apply_rotary_emb_torch(
                q,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self.interleaved,
            ),
            apply_rotary_emb_torch(
                k,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self.interleaved,
            ),
        )


def fp32_autocast_context(device_type: str):
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type=device_type, enabled=False)


class RotationMatrix:
    def __init__(self, rots: torch.Tensor):
        if rots.shape[-1] == 9:
            rots = rots.unflatten(-1, (3, 3))
        assert rots.shape[-1] == 3
        assert rots.shape[-2] == 3
        self._rots = rots.to(torch.float32)

    @classmethod
    def identity(cls, shape: tuple[int, ...], **tensor_kwargs) -> "RotationMatrix":
        rots = torch.eye(3, **tensor_kwargs)
        rots = rots.view(*[1 for _ in range(len(shape))], 3, 3)
        rots = rots.expand(*shape, -1, -1)
        return cls(rots)

    def __getitem__(self, idx) -> "RotationMatrix":
        indices = (idx,) if isinstance(idx, int) or idx is None else tuple(idx)
        return RotationMatrix(self._rots[indices + (slice(None), slice(None))])

    @property
    def shape(self) -> torch.Size:
        return self._rots.shape[:-2]

    @property
    def tensor(self) -> torch.Tensor:
        return self._rots.flatten(-2)

    @property
    def device(self) -> torch.device:
        return self._rots.device

    def as_matrix(self) -> "RotationMatrix":
        return self

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        with fp32_autocast_context(self.device.type):
            p = p.to(self._rots.dtype)
            if self._rots.shape[-3] == 1:
                return p @ self._rots.transpose(-1, -2).squeeze(-3)
            return torch.einsum("...ij,...j", self._rots, p)

    def invert(self) -> "RotationMatrix":
        return RotationMatrix(self._rots.transpose(-1, -2))

    @staticmethod
    def from_graham_schmidt(
        x_axis: torch.Tensor,
        xy_plane: torch.Tensor,
        eps: float = 1e-12,
    ) -> "RotationMatrix":
        with fp32_autocast_context(x_axis.device.type):
            e1 = xy_plane
            denom = torch.sqrt((x_axis**2).sum(dim=-1, keepdim=True) + eps)
            x_axis = x_axis / denom
            dot = (x_axis * e1).sum(dim=-1, keepdim=True)
            e1 = e1 - x_axis * dot
            denom = torch.sqrt((e1**2).sum(dim=-1, keepdim=True) + eps)
            e1 = e1 / denom
            e2 = torch.cross(x_axis, e1, dim=-1)
            return RotationMatrix(torch.stack([x_axis, e1, e2], dim=-1))


@dataclass(frozen=True)
class Affine3D:
    trans: torch.Tensor
    rot: RotationMatrix

    def __post_init__(self) -> None:
        assert self.trans.shape[:-1] == self.rot.shape

    def __getitem__(self, idx) -> "Affine3D":
        indices = (idx,) if isinstance(idx, int) or idx is None else tuple(idx)
        return Affine3D(
            trans=self.trans[indices + (slice(None),)],
            rot=self.rot[idx],
        )

    @property
    def shape(self) -> torch.Size:
        return self.trans.shape[:-1]

    @property
    def dtype(self) -> torch.dtype:
        return self.trans.dtype

    @property
    def device(self) -> torch.device:
        return self.trans.device

    @property
    def tensor(self) -> torch.Tensor:
        return torch.cat([self.rot.tensor, self.trans], dim=-1)

    def as_matrix(self) -> "Affine3D":
        return Affine3D(trans=self.trans, rot=self.rot.as_matrix())

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        return self.rot.apply(p) + self.trans

    @staticmethod
    def from_tensor(t: torch.Tensor) -> "Affine3D":
        match t.shape[-1]:
            case 12:
                trans = t[..., -3:]
                rot = RotationMatrix(t[..., :-3].unflatten(-1, (3, 3)))
            case _:
                raise RuntimeError(
                    f"Cannot detect rotation format from {t.shape[-1] - 3}-d flat vector"
                )
        return Affine3D(trans, rot)

    @staticmethod
    def from_graham_schmidt(
        neg_x_axis: torch.Tensor,
        origin: torch.Tensor,
        xy_plane: torch.Tensor,
        eps: float = 1e-10,
    ) -> "Affine3D":
        x_axis = origin - neg_x_axis
        xy_plane = xy_plane - origin
        return Affine3D(
            trans=origin,
            rot=RotationMatrix.from_graham_schmidt(x_axis, xy_plane, eps),
        )


def build_affine3d_from_coordinates(coords: torch.Tensor) -> tuple[Affine3D, torch.Tensor]:
    max_supported_distance = 1e6
    coord_mask = torch.all(
        torch.all(torch.isfinite(coords) & (coords < max_supported_distance), dim=-1),
        dim=-1,
    )

    def atom3_to_backbone_affine(bb_positions: torch.Tensor) -> Affine3D:
        n_atom, ca_atom, c_atom = bb_positions.unbind(dim=-2)
        return Affine3D.from_graham_schmidt(c_atom, ca_atom, n_atom)

    coords = coords.clone().float()
    coords[~coord_mask] = 0
    average_per_n_ca_c = coords.masked_fill(~coord_mask[..., None, None], 0).sum(1) / (
        coord_mask.sum(-1)[..., None, None] + 1e-8
    )
    affine_from_average = atom3_to_backbone_affine(
        average_per_n_ca_c.float()
    ).as_matrix()

    batch_size, seq_len, _, _ = coords.shape
    affine_rot_mats = affine_from_average.rot.tensor[..., None, :].expand(
        batch_size,
        seq_len,
        9,
    )
    affine_trans = affine_from_average.trans[..., None, :].expand(batch_size, seq_len, 3)
    identity_rot = RotationMatrix.identity(
        (batch_size, seq_len),
        dtype=torch.float32,
        device=coords.device,
        requires_grad=False,
    )
    affine_rot_mats = affine_rot_mats.where(
        coord_mask.any(-1)[..., None, None],
        identity_rot.tensor,
    )
    black_hole_affine = Affine3D(affine_trans, RotationMatrix(affine_rot_mats))

    affine = atom3_to_backbone_affine(coords.float())
    affine = Affine3D.from_tensor(
        affine.tensor.where(coord_mask[..., None], black_hole_affine.tensor)
    )
    return affine, coord_mask


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.scale = self.d_head**-0.5
        self.attn_backend = resolve_attention_backend(attn_backend)
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3, bias=bias),
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()
        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        seq_id: Optional[torch.Tensor],
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        qkv = self.layernorm_qkv(x)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        query = self.q_ln(query).to(query.dtype)
        key = self.k_ln(key).to(query.dtype)
        query, key = self._apply_rotary(query, key)

        reshaper = functools.partial(
            einops.rearrange,
            pattern="b s (h d) -> b h s d",
            h=self.n_heads,
        )
        query, key, value = map(reshaper, (query, key, value))

        if seq_id is not None:
            mask = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask = mask.unsqueeze(1)
        else:
            mask = None

        if output_attentions:
            attn_scores = torch.einsum("bhld,bhsd->bhls", query, key) * self.scale
            if mask is not None:
                attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = torch.einsum("bhls,bhsd->bhld", attn_weights, value)
        else:
            attn_weights = None
            if self.attn_backend == AttentionBackend.FLEX:
                block_mask = self._create_flex_block_mask(seq_id, query)
                fn = _get_flex_attention_fn()
                assert fn is not None, "Flex Attention is not available in this environment."
                context = fn(
                    query,
                    key,
                    value,
                    block_mask=block_mask,
                    scale=self.scale,
                )
            elif self.attn_backend == AttentionBackend.SDPA:
                context = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=mask,
                    scale=self.scale,
                )
            else:
                raise AssertionError(f"Unsupported resolved ESM3 backend: {self.attn_backend}")

        context = einops.rearrange(context, "b h s d -> b s (h d)")
        return self.out_proj(context), attn_weights

    @staticmethod
    def _create_flex_block_mask(
        seq_id: Optional[torch.Tensor],
        query: torch.Tensor,
    ) -> Optional["BlockMask"]:
        if seq_id is None:
            return None
        assert create_block_mask is not None, (
            "Flex Attention requested but torch.create_block_mask is unavailable."
        )
        batch_size, _, seq_len, _ = query.shape

        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            return seq_id[batch_idx, q_idx] == seq_id[batch_idx, kv_idx]

        return create_block_mask(
            mask_mod,
            batch_size,
            1,
            seq_len,
            seq_len,
            device=query.device,
        )


class GeometricReasoningOriginalImpl(nn.Module):
    def __init__(
        self,
        c_s: int,
        v_heads: int,
        num_vector_messages: int = 1,
        mask_and_zero_frameless: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.c_s = c_s
        self.v_heads = v_heads
        self.num_vector_messages = num_vector_messages
        self.mask_and_zero_frameless = mask_and_zero_frameless
        self.s_norm = nn.LayerNorm(c_s, bias=bias)
        dim_proj = 4 * self.v_heads * 3 + self.v_heads * 3 * self.num_vector_messages
        self.proj = nn.Linear(c_s, dim_proj, bias=bias)
        channels_out = self.v_heads * 3 * self.num_vector_messages
        self.out_proj = nn.Linear(channels_out, c_s, bias=bias)
        self.distance_scale_per_head = nn.Parameter(torch.zeros((self.v_heads)))
        self.rotation_scale_per_head = nn.Parameter(torch.zeros((self.v_heads)))

    def forward(
        self,
        s: torch.Tensor,
        affine: Affine3D,
        affine_mask: torch.Tensor,
        sequence_id: Optional[torch.Tensor],
        chain_id: torch.Tensor,
    ) -> torch.Tensor:
        if sequence_id is None:
            sequence_id = torch.zeros_like(s[..., 0], dtype=torch.int64)
        attn_bias = sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2)
        attn_bias = attn_bias.unsqueeze(1).float()
        attn_bias = attn_bias.masked_fill(
            ~affine_mask[:, None, None, :],
            torch.finfo(attn_bias.dtype).min,
        )
        chain_id_mask = chain_id.unsqueeze(1) != chain_id.unsqueeze(2)
        attn_bias = attn_bias.masked_fill(
            chain_id_mask.unsqueeze(1),
            torch.finfo(s.dtype).min,
        )

        ns = self.s_norm(s)
        vec_rot, vec_dist = self.proj(ns).split(
            [
                self.v_heads * 2 * 3 + self.v_heads * 3 * self.num_vector_messages,
                self.v_heads * 2 * 3,
            ],
            dim=-1,
        )

        query_rot, key_rot, value = (
            affine.rot[..., None]
            .apply(rearrange(vec_rot, "... (h c) -> ... h c", c=3))
            .split(
                [self.v_heads, self.v_heads, self.v_heads * self.num_vector_messages],
                dim=-2,
            )
        )
        query_dist, key_dist = (
            affine[..., None]
            .apply(rearrange(vec_dist, "... (h c) -> ... h c", c=3))
            .chunk(2, dim=-2)
        )

        query_dist = rearrange(query_dist, "b s h d -> b h s 1 d")
        key_dist = rearrange(key_dist, "b s h d -> b h 1 s d")
        query_rot = rearrange(query_rot, "b s h d -> b h s d")
        key_rot = rearrange(key_rot, "b s h d -> b h d s")
        value = rearrange(
            value,
            "b s (h m) d -> b h s (m d)",
            m=self.num_vector_messages,
        )

        distance_term = (query_dist - key_dist).norm(dim=-1) / math.sqrt(3)
        rotation_term = query_rot.matmul(key_rot) / math.sqrt(3)
        distance_term_weight = rearrange(
            F.softplus(self.distance_scale_per_head),
            "h -> h 1 1",
        )
        rotation_term_weight = rearrange(
            F.softplus(self.rotation_scale_per_head),
            "h -> h 1 1",
        )
        attn_weight = (
            rotation_term * rotation_term_weight - distance_term * distance_term_weight
        )

        s_q = attn_weight.size(2)
        s_k = attn_weight.size(3)
        offset_q = max(0, attn_bias.size(2) - s_q)
        offset_k = max(0, attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:, :, offset_q:, offset_k:]
        attn_weight = torch.softmax(attn_weight + attn_bias, dim=-1)

        attn_out = attn_weight.matmul(value)
        attn_out = (
            affine.rot[..., None]
            .invert()
            .apply(
                rearrange(
                    attn_out,
                    "b h s (m d) -> b s (h m) d",
                    m=self.num_vector_messages,
                )
            )
        )
        attn_out = rearrange(
            attn_out,
            "b s (h m) d -> b s (h m d)",
            m=self.num_vector_messages,
        )
        if self.mask_and_zero_frameless:
            attn_out = attn_out.masked_fill(~affine_mask[..., None], 0.0)
        attn_out = attn_out.to(self.out_proj.weight.dtype)
        return self.out_proj(attn_out)


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model,
            swiglu_correction_fn(expansion_ratio, d_model) * 2,
            bias=bias,
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool) -> nn.Module:
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )


class UnifiedTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_geom_attn: bool = False,
        use_plain_attn: bool = True,
        v_heads: Optional[int] = None,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1.0,
        mask_and_zero_frameless: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        self.use_plain_attn = use_plain_attn
        if self.use_plain_attn:
            self.attn = MultiHeadAttention(
                d_model,
                n_heads,
                bias,
                qk_layernorm=qk_layernorm,
                attn_backend=attn_backend,
            )
        self.use_geom_attn = use_geom_attn
        if self.use_geom_attn:
            assert v_heads is not None
            self.geom_attn = GeometricReasoningOriginalImpl(
                c_s=d_model,
                v_heads=v_heads,
                bias=bias,
                mask_and_zero_frameless=mask_and_zero_frameless,
            )
        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        self.scaling_factor = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: Optional[torch.Tensor],
        frames: Affine3D,
        frames_mask: torch.Tensor,
        chain_id: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_weights = None
        if self.use_plain_attn:
            r1, attn_weights = self.attn(
                x,
                sequence_id,
                output_attentions=output_attentions,
            )
            x = x + r1 / self.scaling_factor

        if self.use_geom_attn:
            r2 = self.geom_attn(x, frames, frames_mask, sequence_id, chain_id)
            x = x + r2 / self.scaling_factor

        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3
        return x, attn_weights


class TransformerStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: Optional[int],
        n_layers: int,
        n_layers_geom: int = 1,
        scale_residue: bool = True,
        mask_and_zero_frameless: bool = False,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",
        expansion_ratio: float = 8 / 3,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    v_heads=v_heads,
                    use_geom_attn=index < n_layers_geom,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    expansion_ratio=expansion_ratio,
                    mask_and_zero_frameless=mask_and_zero_frameless,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                    ffn_type=ffn_type,
                    attn_backend=attn_backend,
                )
                for index in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: Optional[torch.Tensor] = None,
        affine: Optional[Affine3D] = None,
        affine_mask: Optional[torch.Tensor] = None,
        chain_id: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        Optional[tuple[torch.Tensor, ...]],
    ]:
        *batch_dims, _ = x.shape
        if chain_id is None:
            chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)
        assert affine is not None
        assert affine_mask is not None
        all_hidden_states = []
        all_attentions = []
        for block in self.blocks:
            x, attn_weights = block(
                x,
                sequence_id,
                affine,
                affine_mask,
                chain_id,
                output_attentions=output_attentions,
            )
            all_hidden_states.append(x)
            if output_attentions and attn_weights is not None:
                all_attentions.append(attn_weights)
        hidden_states = tuple(all_hidden_states)
        attentions = tuple(all_attentions) if output_attentions else None
        return self.norm(x), x, hidden_states, attentions


class EncodeInputs(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.sequence_embed = nn.Embedding(64, d_model)
        self.plddt_projection = nn.Linear(16, d_model)
        self.structure_per_res_plddt_projection = nn.Linear(16, d_model)
        self.structure_tokens_embed = nn.Embedding(4096 + 5, d_model)
        self.ss8_embed = nn.Embedding(8 + 3, d_model)
        self.sasa_embed = nn.Embedding(16 + 3, d_model)
        self.function_embed = nn.ModuleList(
            [nn.Embedding(260, d_model // 8, padding_idx=0) for _ in range(8)]
        )
        self.residue_embed = nn.EmbeddingBag(1478, d_model, mode="sum", padding_idx=0)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        structure_tokens: torch.Tensor,
        average_plddt: torch.Tensor,
        per_res_plddt: torch.Tensor,
        ss8_tokens: torch.Tensor,
        sasa_tokens: torch.Tensor,
        function_tokens: torch.Tensor,
        residue_annotation_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sequence_embed = self.sequence_embed(sequence_tokens)
        rbf_16_fn = functools.partial(rbf, v_min=0.0, v_max=1.0, n_bins=16)
        plddt_embed = self.plddt_projection(
            rbf_16_fn(average_plddt).to(self.plddt_projection.weight.dtype)
        )
        structure_per_res_plddt = self.structure_per_res_plddt_projection(
            rbf_16_fn(per_res_plddt).to(
                self.structure_per_res_plddt_projection.weight.dtype
            )
        )
        structure_embed = self.structure_tokens_embed(structure_tokens)
        ss8_embed = self.ss8_embed(ss8_tokens)
        sasa_embed = self.sasa_embed(sasa_tokens)
        function_embed = torch.cat(
            [
                embed_fn(funcs)
                for embed_fn, funcs in zip(
                    self.function_embed,
                    function_tokens.unbind(-1),
                )
            ],
            -1,
        )

        batch_size, seq_len, num_annotations = residue_annotation_tokens.shape
        residue_embed = self.residue_embed(
            rearrange(
                residue_annotation_tokens,
                "b l n -> (b l) n",
                b=batch_size,
                l=seq_len,
                n=num_annotations,
            )
        )
        residue_embed = rearrange(
            residue_embed,
            "(b l) d -> b l d",
            b=batch_size,
            l=seq_len,
        )

        return (
            sequence_embed
            + plddt_embed
            + structure_per_res_plddt
            + structure_embed
            + ss8_embed
            + sasa_embed
            + function_embed
            + residue_embed
        )


@dataclass
class ESM3CoreOutput:
    sequence_logits: torch.Tensor
    structure_logits: torch.Tensor
    secondary_structure_logits: torch.Tensor
    sasa_logits: torch.Tensor
    function_logits: torch.Tensor
    residue_logits: torch.Tensor
    embeddings: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    attentions: Optional[tuple[torch.Tensor, ...]] = None


class OutputHeads(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.sequence_head = RegressionHead(d_model, 64)
        self.structure_head = RegressionHead(d_model, 4096)
        self.ss8_head = RegressionHead(d_model, 8 + 3)
        self.sasa_head = RegressionHead(d_model, 16 + 3)
        self.function_head = RegressionHead(d_model, 260 * 8)
        self.residue_head = RegressionHead(d_model, 1478)

    def forward(
        self,
        x: torch.Tensor,
        embed: torch.Tensor,
        hidden_states: tuple[torch.Tensor, ...],
        attentions: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> ESM3CoreOutput:
        function_logits = self.function_head(x)
        function_logits = rearrange(function_logits, "... (k v) -> ... k v", k=8)
        return ESM3CoreOutput(
            sequence_logits=self.sequence_head(x),
            structure_logits=self.structure_head(x),
            secondary_structure_logits=self.ss8_head(x),
            sasa_logits=self.sasa_head(x),
            function_logits=function_logits,
            residue_logits=self.residue_head(x),
            embeddings=embed,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class ESM3Core(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int,
        n_layers: int,
        tokenizers: FastESM3TokenizerCollection,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        self.encoder = EncodeInputs(d_model)
        self.transformer = TransformerStack(
            d_model,
            n_heads,
            v_heads,
            n_layers,
            mask_and_zero_frameless=True,
            attn_backend=attn_backend,
        )
        self.output_heads = OutputHeads(d_model)
        self.tokenizers = tokenizers

    def forward(
        self,
        *,
        sequence_tokens: Optional[torch.Tensor] = None,
        structure_tokens: Optional[torch.Tensor] = None,
        ss8_tokens: Optional[torch.Tensor] = None,
        sasa_tokens: Optional[torch.Tensor] = None,
        function_tokens: Optional[torch.Tensor] = None,
        residue_annotation_tokens: Optional[torch.Tensor] = None,
        average_plddt: Optional[torch.Tensor] = None,
        per_res_plddt: Optional[torch.Tensor] = None,
        structure_coords: Optional[torch.Tensor] = None,
        chain_id: Optional[torch.Tensor] = None,
        sequence_id: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> ESM3CoreOutput:
        output_attentions = bool(output_attentions)
        present_inputs = [
            sequence_tokens,
            structure_tokens,
            ss8_tokens,
            sasa_tokens,
            structure_coords,
            function_tokens,
            residue_annotation_tokens,
        ]
        try:
            seq_len, device = next(
                (x.shape[1], x.device) for x in present_inputs if x is not None
            )
        except StopIteration:
            raise ValueError("At least one of the inputs must be non-None")

        def defaults(x: Optional[torch.Tensor], token: int) -> torch.Tensor:
            if x is None:
                return torch.full(
                    (1, seq_len),
                    token,
                    dtype=torch.long,
                    device=device,
                )
            return x

        sequence_tokens = defaults(sequence_tokens, self.tokenizers.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, SASA_PAD_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)

        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (1, seq_len, MAX_RESIDUE_ANNOTATIONS),
                RESIDUE_PAD_TOKEN,
                dtype=torch.long,
                device=device,
            )
        if function_tokens is None:
            function_tokens = torch.full(
                (1, seq_len, FUNCTION_TOKENS_DEPTH),
                INTERPRO_PAD_TOKEN,
                dtype=torch.long,
                device=device,
            )
        if structure_coords is None:
            structure_coords = torch.full(
                (1, seq_len, 3, 3),
                float("nan"),
                dtype=torch.float,
                device=device,
            )

        structure_coords = structure_coords[..., :3, :]
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = defaults(structure_tokens, STRUCTURE_MASK_TOKEN)
        structure_tokens = (
            structure_tokens.masked_fill(structure_tokens == -1, STRUCTURE_MASK_TOKEN)
            .masked_fill(sequence_tokens == SEQUENCE_BOS_TOKEN, STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == SEQUENCE_PAD_TOKEN, STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == SEQUENCE_EOS_TOKEN, STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == SEQUENCE_CHAINBREAK_TOKEN,
                STRUCTURE_CHAINBREAK_TOKEN,
            )
        )

        x = self.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )
        x, embedding, hidden_states, attentions = self.transformer(
            x,
            sequence_id,
            affine,
            affine_mask,
            chain_id,
            output_attentions=output_attentions,
        )
        return self.output_heads(
            x,
            embedding,
            hidden_states=hidden_states,
            attentions=attentions,
        )


def _resolve_esm3_checkpoint_key(model_name: str) -> str:
    if model_name in ESM3_OPEN_SMALL_ALIASES:
        return ESM3_OPEN_SMALL
    raise ValueError(
        f"Unsupported ESM3 checkpoint {model_name}. "
        f"Supported names: {sorted(ESM3_OPEN_SMALL_ALIASES)}"
    )


def parse_fasta(fasta_path: str) -> list[str]:
    assert os.path.exists(fasta_path), f"FASTA file does not exist: {fasta_path}"
    sequences = []
    current_seq = []
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if len(stripped) == 0:
                continue
            if stripped.startswith(">"):
                if len(current_seq) > 0:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(stripped)
    if len(current_seq) > 0:
        sequences.append("".join(current_seq))
    return sequences


def _ensure_official_esm_on_path() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "official" / "esm"
        if (candidate / "esm" / "models" / "esm3.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


def _make_structure_encoder(device: Union[torch.device, str]) -> nn.Module:
    _ensure_official_esm_on_path()
    pretrained = importlib.import_module("esm.pretrained")
    return pretrained.ESM3_structure_encoder_v0(device)


def _make_structure_decoder(device: Union[torch.device, str]) -> nn.Module:
    _ensure_official_esm_on_path()
    pretrained = importlib.import_module("esm.pretrained")
    return pretrained.ESM3_structure_decoder_v0(device)


def _make_function_decoder(device: Union[torch.device, str]) -> nn.Module:
    _ensure_official_esm_on_path()
    pretrained = importlib.import_module("esm.pretrained")
    return pretrained.ESM3_function_decoder_v0(device)


def _build_official_esm3(config: FastESM3Config) -> nn.Module:
    return ESM3Core(
        d_model=config.hidden_size,
        n_heads=config.num_attention_heads,
        v_heads=config.num_vector_heads,
        n_layers=config.num_hidden_layers,
        tokenizers=FastESM3TokenizerCollection(sequence=EsmSequenceTokenizer()),
        attn_backend=config.attn_backend,
    )


class FastESM3PreTrainedModel(PreTrainedModel):
    config_class = FastESM3Config
    base_model_prefix = "esm3"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    all_tied_weights_keys = {}

    @classmethod
    def is_remote_code(cls) -> bool:
        return True

    def _init_weights(self, module: nn.Module) -> None:
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

    @property
    def attn_backend(self) -> str:
        return self.config.attn_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        assert backend in _SUPPORTED_ATTENTION_BACKENDS, (
            f"ESM3 currently supports only {_SUPPORTED_ATTENTION_BACKENDS}; got {backend}."
        )
        self.config.attn_backend = backend
        resolved = resolve_attention_backend(backend)
        for module in self.modules():
            if isinstance(module, MultiHeadAttention):
                module.attn_backend = resolved

    @classmethod
    def from_pretrained_esm(
        cls,
        model_name: str = ESM3_OPEN_SMALL,
        device: Union[torch.device, str] = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> "FastESM3Model":
        key = _resolve_esm3_checkpoint_key(model_name)
        spec = _ESM3_CHECKPOINT_SPECS[key]
        config = FastESM3Config(
            hidden_size=spec["hidden_size"],
            num_attention_heads=spec["num_attention_heads"],
            num_vector_heads=spec["num_vector_heads"],
            num_hidden_layers=spec["num_hidden_layers"],
            model_name=key,
        )
        model = FastESM3Model(config)
        checkpoint_root = Path(
            snapshot_download(
                repo_id=spec["repo_id"],
                allow_patterns=["data/weights/esm3_sm_open_v1.pth"],
            )
        )
        state_dict = torch.load(
            checkpoint_root / "data" / "weights" / "esm3_sm_open_v1.pth",
            map_location=torch.device(device),
        )
        load_result = model.esm3.load_state_dict(state_dict, strict=True)
        assert len(load_result.missing_keys) == 0, load_result.missing_keys
        assert len(load_result.unexpected_keys) == 0, load_result.unexpected_keys
        model = model.to(device)
        if dtype is not None:
            model = model.to(dtype=dtype)
        model.eval()
        return model


class FastESM3Model(FastESM3PreTrainedModel):
    config_class = FastESM3Config

    def __init__(self, config: FastESM3Config, **kwargs):
        super().__init__(config, **kwargs)
        self.tokenizer = EsmSequenceTokenizer()
        self.esm3 = _build_official_esm3(config)
        self.__dict__["_official_sdk_model"] = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def raw_model(self) -> nn.Module:
        return self.esm3

    def _get_official_sdk_model(self) -> nn.Module:
        cached_model = self.__dict__["_official_sdk_model"]
        if cached_model is not None:
            return cached_model
        _ensure_official_esm_on_path()
        esm3_module = importlib.import_module("esm.models.esm3")
        tokenization = importlib.import_module("esm.tokenization")

        sdk_model = esm3_module.ESM3(
            d_model=self.config.hidden_size,
            n_heads=self.config.num_attention_heads,
            v_heads=self.config.num_vector_heads,
            n_layers=self.config.num_hidden_layers,
            structure_encoder_fn=_make_structure_encoder,
            structure_decoder_fn=_make_structure_decoder,
            function_decoder_fn=_make_function_decoder,
            tokenizers=tokenization.get_esm3_model_tokenizers(self.config.model_name),
        )
        load_result = sdk_model.load_state_dict(self.esm3.state_dict(), strict=True)
        assert len(load_result.missing_keys) == 0, load_result.missing_keys
        assert len(load_result.unexpected_keys) == 0, load_result.unexpected_keys
        dtype = next(self.esm3.parameters()).dtype
        sdk_model = sdk_model.to(self.device).to(dtype=dtype).eval()
        self.__dict__["_official_sdk_model"] = sdk_model
        return sdk_model

    def get_input_embeddings(self) -> nn.Module:
        return self.esm3.encoder.sequence_embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.esm3.encoder.sequence_embed = value

    def tokenize_sequences(
        self,
        sequences: Union[str, list[str]],
        padding: bool = True,
        return_tensors: str = "pt",
        device: Optional[Union[torch.device, str]] = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            sequences,
            padding=padding,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )
        if device is None:
            return tokenized
        return {name: tensor.to(device) for name, tensor in tokenized.items()}

    def forward_sequence(
        self,
        sequences: Union[str, list[str]],
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> FastESM3Output:
        if device is None:
            device = self.device
        tokenized = self.tokenize_sequences(sequences, device=device)
        return self(**tokenized, **kwargs)

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        if store_all_hidden_states:
            assert output.hidden_states is not None, "store_all_hidden_states requires hidden states."
            return torch.stack(tuple(output.hidden_states), dim=1)
        if hidden_state_index == -1:
            return output.last_hidden_state
        assert output.hidden_states is not None, "hidden_state_index selection requires hidden states."
        return output.hidden_states[hidden_state_index]

    def _pool_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_types: list[str],
    ) -> torch.Tensor:
        pooled = []
        mask = attention_mask.to(dtype=embeddings.dtype).unsqueeze(-1)
        for pooling_type in pooling_types:
            if pooling_type == "mean":
                pooled.append((embeddings * mask).sum(dim=1) / mask.sum(dim=1))
            elif pooling_type == "cls":
                pooled.append(embeddings[:, 0, :])
            elif pooling_type == "max":
                bool_mask = attention_mask.unsqueeze(-1).bool()
                pooled.append(
                    embeddings.masked_fill(~bool_mask, float("-inf")).max(dim=1).values
                )
            else:
                raise ValueError(
                    f"Unsupported ESM3 pooling type {pooling_type}. "
                    "Supported values are 'mean', 'cls', and 'max'."
                )
        return torch.cat(pooled, dim=-1)

    def embed_dataset(
        self,
        sequences: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        batch_size: int = 2,
        max_len: int = 512,
        truncate: bool = True,
        full_embeddings: bool = False,
        embed_dtype: torch.dtype = torch.float32,
        pooling_types: List[str] = ["mean"],
        num_workers: int = 0,
        sql: bool = False,
        save: bool = True,
        sql_db_path: str = "embeddings.db",
        save_path: str = "embeddings.pth",
        fasta_path: Optional[str] = None,
        padding: str = "longest",
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del num_workers, sql_db_path
        assert not sql, "ESM3 embed_dataset currently supports .pth saves, not SQLite."
        assert isinstance(hidden_state_index, int), "hidden_state_index must be an integer."
        assert full_embeddings or not store_all_hidden_states, (
            "store_all_hidden_states=True requires full_embeddings=True."
        )
        if tokenizer is None:
            tokenizer = self.tokenizer
        if fasta_path is not None:
            fasta_sequences = parse_fasta(fasta_path)
            sequences = list(sequences or []) + fasta_sequences
        assert sequences is not None and len(sequences) > 0, (
            "Must provide at least one sequence via `sequences` or `fasta_path`."
        )

        unique_sequences = []
        seen_sequences = set()
        for sequence in sequences:
            prepared_sequence = sequence[:max_len] if truncate else sequence
            if prepared_sequence not in seen_sequences:
                unique_sequences.append(prepared_sequence)
                seen_sequences.add(prepared_sequence)
        unique_sequences = sorted(unique_sequences, key=len, reverse=True)

        embeddings_by_sequence: Dict[str, torch.Tensor] = {}
        was_training = self.training
        self.eval()
        for batch_start in range(0, len(unique_sequences), batch_size):
            batch_sequences = unique_sequences[batch_start : batch_start + batch_size]
            tokenized = tokenizer(
                batch_sequences,
                padding=padding,
                truncation=truncate,
                max_length=max_len + 2,
                return_tensors="pt",
            )
            tokenized = {
                name: tensor.to(self.device) for name, tensor in tokenized.items()
            }
            with torch.inference_mode():
                residue_embeddings = self._embed(
                    **tokenized,
                    hidden_state_index=hidden_state_index,
                    store_all_hidden_states=store_all_hidden_states,
                    **kwargs,
                )
            attention_mask = tokenized["attention_mask"]
            if full_embeddings:
                batch_embeddings = residue_embeddings.to(embed_dtype).cpu()
                for sequence, embedding, mask in zip(
                    batch_sequences,
                    batch_embeddings,
                    attention_mask.cpu(),
                ):
                    if embedding.ndim == 3:
                        embeddings_by_sequence[sequence] = embedding[:, mask.bool(), :]
                    else:
                        embeddings_by_sequence[sequence] = embedding[mask.bool()]
            else:
                pooled_embeddings = self._pool_embeddings(
                    residue_embeddings,
                    attention_mask,
                    pooling_types,
                )
                pooled_embeddings = pooled_embeddings.to(embed_dtype).cpu()
                for sequence, embedding in zip(batch_sequences, pooled_embeddings):
                    embeddings_by_sequence[sequence] = embedding

        if was_training:
            self.train()
        if save:
            torch.save(embeddings_by_sequence, save_path)
        return embeddings_by_sequence

    def encode(self, input):
        return self._get_official_sdk_model().encode(input)

    def decode(self, input):
        return self._get_official_sdk_model().decode(input)

    def generate(self, input, config):
        return self._get_official_sdk_model().generate(input, config)

    def batch_generate(self, inputs, configs):
        return self._get_official_sdk_model().batch_generate(inputs, configs)

    def forward_and_sample(self, input, sampling_configuration):
        return self._get_official_sdk_model().forward_and_sample(
            input,
            sampling_configuration,
        )

    def logits(self, input=None, config=None, **kwargs):
        if input is None:
            return self.forward(**kwargs)
        if isinstance(input, torch.Tensor):
            return self.forward(sequence_tokens=input, **kwargs)
        if config is None:
            return self._get_official_sdk_model().logits(input)
        return self._get_official_sdk_model().logits(input, config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_tokens: Optional[torch.Tensor] = None,
        structure_tokens: Optional[torch.Tensor] = None,
        ss8_tokens: Optional[torch.Tensor] = None,
        sasa_tokens: Optional[torch.Tensor] = None,
        function_tokens: Optional[torch.Tensor] = None,
        residue_annotation_tokens: Optional[torch.Tensor] = None,
        average_plddt: Optional[torch.Tensor] = None,
        per_res_plddt: Optional[torch.Tensor] = None,
        structure_coords: Optional[torch.Tensor] = None,
        chain_id: Optional[torch.Tensor] = None,
        sequence_id: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> FastESM3Output:
        del output_hidden_states, return_dict, kwargs
        if sequence_tokens is None:
            sequence_tokens = input_ids
        if sequence_id is None and attention_mask is not None:
            sequence_id = attention_mask.to(dtype=torch.bool)

        output = self.esm3(
            sequence_tokens=sequence_tokens,
            structure_tokens=structure_tokens,
            ss8_tokens=ss8_tokens,
            sasa_tokens=sasa_tokens,
            function_tokens=function_tokens,
            residue_annotation_tokens=residue_annotation_tokens,
            average_plddt=average_plddt,
            per_res_plddt=per_res_plddt,
            structure_coords=structure_coords,
            chain_id=chain_id,
            sequence_id=sequence_id,
            output_attentions=output_attentions,
        )

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                output.sequence_logits.view(-1, output.sequence_logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
            )

        return FastESM3Output(
            loss=loss,
            logits=output.sequence_logits,
            last_hidden_state=output.embeddings,
            sequence_logits=output.sequence_logits,
            structure_logits=output.structure_logits,
            secondary_structure_logits=output.secondary_structure_logits,
            sasa_logits=output.sasa_logits,
            function_logits=output.function_logits,
            residue_logits=output.residue_logits,
            embeddings=output.embeddings,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
