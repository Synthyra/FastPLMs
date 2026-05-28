"""Load official ESM3 from the official/esm submodule for comparison."""
import math
from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn
from torch.nn import functional as F

from testing.official import use_esm_submodule

use_esm_submodule()


def _patch_official_geom_attention_dtype_cast() -> None:
    from esm.layers.geom_attention import GeometricReasoningOriginalImpl

    if getattr(GeometricReasoningOriginalImpl, "_fastplms_dtype_patch", False):
        return

    def forward(self, s, affine, affine_mask, sequence_id, chain_id):
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
            .apply(einops.rearrange(vec_rot, "... (h c) -> ... h c", c=3))
            .split(
                [self.v_heads, self.v_heads, self.v_heads * self.num_vector_messages],
                dim=-2,
            )
        )
        query_dist, key_dist = (
            affine[..., None]
            .apply(einops.rearrange(vec_dist, "... (h c) -> ... h c", c=3))
            .chunk(2, dim=-2)
        )

        query_dist = einops.rearrange(query_dist, "b s h d -> b h s 1 d")
        key_dist = einops.rearrange(key_dist, "b s h d -> b h 1 s d")
        query_rot = einops.rearrange(query_rot, "b s h d -> b h s d")
        key_rot = einops.rearrange(key_rot, "b s h d -> b h d s")
        value = einops.rearrange(
            value,
            "b s (h m) d -> b h s (m d)",
            m=self.num_vector_messages,
        )

        distance_term = (query_dist - key_dist).norm(dim=-1) / math.sqrt(3)
        rotation_term = query_rot.matmul(key_rot) / math.sqrt(3)
        distance_term_weight = einops.rearrange(
            F.softplus(self.distance_scale_per_head),
            "h -> h 1 1",
        )
        rotation_term_weight = einops.rearrange(
            F.softplus(self.rotation_scale_per_head),
            "h -> h 1 1",
        )
        attn_weight = (
            rotation_term * rotation_term_weight
            - distance_term * distance_term_weight
        )

        if attn_bias is not None:
            seq_q = attn_weight.size(2)
            seq_k = attn_weight.size(3)
            bias_q = max(0, attn_bias.size(2) - seq_q)
            bias_k = max(0, attn_bias.size(3) - seq_k)
            attn_bias = attn_bias[:, :, bias_q:, bias_k:]
            attn_weight = attn_weight + attn_bias

        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_out = attn_weight.matmul(value)
        attn_out = (
            affine.rot[..., None]
            .invert()
            .apply(
                einops.rearrange(
                    attn_out,
                    "b h s (m d) -> b s (h m) d",
                    m=self.num_vector_messages,
                )
            )
        )
        attn_out = einops.rearrange(
            attn_out,
            "b s (h m) d -> b s (h m d)",
            m=self.num_vector_messages,
        )
        if self.mask_and_zero_frameless:
            attn_out = attn_out.masked_fill(~affine_mask[..., None], 0.0)
        attn_out = attn_out.to(self.out_proj.weight.dtype)
        return self.out_proj(attn_out)

    GeometricReasoningOriginalImpl.forward = forward
    GeometricReasoningOriginalImpl._fastplms_dtype_patch = True


_patch_official_geom_attention_dtype_cast()


class _ESM3ComplianceOutput:
    def __init__(
        self,
        logits: torch.Tensor,
        last_hidden_state: torch.Tensor,
        hidden_states: Tuple[torch.Tensor, ...],
        sequence_logits: torch.Tensor,
        structure_logits: torch.Tensor,
        function_logits: torch.Tensor,
        residue_logits: torch.Tensor,
    ) -> None:
        self.logits = logits
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.sequence_logits = sequence_logits
        self.structure_logits = structure_logits
        self.function_logits = function_logits
        self.residue_logits = residue_logits


class _ESM3StateDictRoot(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.esm3 = model


class _OfficialESM3ForwardWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = _ESM3StateDictRoot(model)
        self.tokenizer = model.tokenizers.sequence

    @property
    def esm3(self) -> nn.Module:
        return self.model.esm3

    def _encode_inputs(
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
        from esm.utils.misc import rbf

        encoder = self.esm3.encoder
        sequence_embed = encoder.sequence_embed(sequence_tokens)
        rbf_16_fn = lambda x: rbf(x, v_min=0.0, v_max=1.0, n_bins=16)
        plddt_embed = encoder.plddt_projection(
            rbf_16_fn(average_plddt).to(encoder.plddt_projection.weight.dtype)
        )
        structure_per_res_plddt = encoder.structure_per_res_plddt_projection(
            rbf_16_fn(per_res_plddt).to(
                encoder.structure_per_res_plddt_projection.weight.dtype
            )
        )
        structure_embed = encoder.structure_tokens_embed(structure_tokens)
        ss8_embed = encoder.ss8_embed(ss8_tokens)
        sasa_embed = encoder.sasa_embed(sasa_tokens)
        function_embed = torch.cat(
            [
                embed_fn(funcs)
                for embed_fn, funcs in zip(
                    encoder.function_embed,
                    function_tokens.unbind(-1),
                )
            ],
            -1,
        )

        batch_size, seq_len, num_annotations = residue_annotation_tokens.shape
        residue_embed = encoder.residue_embed(
            einops.rearrange(
                residue_annotation_tokens,
                "b l n -> (b l) n",
                b=batch_size,
                l=seq_len,
                n=num_annotations,
            )
        )
        residue_embed = einops.rearrange(
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> _ESM3ComplianceOutput:
        from esm.utils.constants import esm3 as C
        from esm.utils.structure.affine3d import build_affine3d_from_coordinates

        del output_hidden_states, kwargs
        output_attentions = bool(output_attentions)
        if sequence_tokens is None:
            sequence_tokens = input_ids
        if sequence_id is None and attention_mask is not None:
            sequence_id = attention_mask.to(dtype=torch.bool)

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

        sequence_tokens = defaults(sequence_tokens, self.tokenizer.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)

        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (1, seq_len, C.MAX_RESIDUE_ANNOTATIONS),
                C.RESIDUE_PAD_TOKEN,
                dtype=torch.long,
                device=device,
            )
        if function_tokens is None:
            function_tokens = torch.full(
                (1, seq_len, C.FUNCTION_TOKENS_DEPTH),
                C.INTERPRO_PAD_TOKEN,
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
        structure_tokens = defaults(structure_tokens, C.STRUCTURE_MASK_TOKEN)
        structure_tokens = (
            structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
                C.STRUCTURE_CHAINBREAK_TOKEN,
            )
        )

        x = self._encode_inputs(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )
        x, embedding, hidden_states, attentions = self.esm3.transformer(
            x,
            sequence_id,
            affine,
            affine_mask,
            chain_id,
            output_attentions=output_attentions,
        )
        output = self.esm3.output_heads(x, embedding, attentions=attentions)

        return _ESM3ComplianceOutput(
            logits=output.sequence_logits,
            last_hidden_state=output.embeddings,
            hidden_states=hidden_states,
            sequence_logits=output.sequence_logits,
            structure_logits=output.structure_logits,
            function_logits=output.function_logits,
            residue_logits=output.residue_logits,
        )


def _normalize_reference_repo_id(reference_repo_id: str) -> str:
    if reference_repo_id == "biohub/esm3-sm-open-v1":
        return "esm3-sm-open-v1"
    return reference_repo_id


def load_official_model(
    reference_repo_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[nn.Module, object]:
    from esm.pretrained import load_local_model
    from esm.utils.constants.models import normalize_model_name

    model_name = normalize_model_name(_normalize_reference_repo_id(reference_repo_id))
    model = load_local_model(model_name, device=device).eval()
    model = model.to(device=device, dtype=dtype).eval()
    wrapped = _OfficialESM3ForwardWrapper(model).to(device=device, dtype=dtype).eval()
    return wrapped, wrapped.tokenizer
