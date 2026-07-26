"""Precompute pair and atom biases used by the Boltz2 diffusion stack."""

from __future__ import annotations

import torch
from torch import nn

from .vb_modules_encodersv2 import AtomEncoder, PairwiseConditioning


def _bias_projections(
    depth: int,
    input_dim: int,
    num_heads: int,
) -> nn.ModuleList:
    """Build one normalized, bias-free projection per transformer block."""

    return nn.ModuleList(
        [
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, num_heads, bias=False),
            )
            for _ in range(depth)
        ]
    )


def _concatenate_biases(
    projections: nn.ModuleList,
    pair_features: torch.Tensor,
) -> torch.Tensor:
    # pair_features: (..., d_pair); each projection: (..., h).
    return torch.cat(
        [projection(pair_features) for projection in projections], dim=-1
    )  # (..., depth * h)


class DiffusionConditioning(nn.Module):
    """Prepare conditioned atom features and per-layer attention biases."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
    ) -> None:
        super().__init__()
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )
        self.atom_encoder = AtomEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=True,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )
        self.atom_enc_proj_z = _bias_projections(
            atom_encoder_depth,
            atom_z,
            atom_encoder_heads,
        )
        self.atom_dec_proj_z = _bias_projections(
            atom_decoder_depth,
            atom_z,
            atom_decoder_heads,
        )
        self.token_trans_proj_z = _bias_projections(
            token_transformer_depth,
            token_z,
            token_transformer_heads,
        )

    def forward(
        self,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        relative_position_encoding: torch.Tensor,
        feats: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return conditioned atom tensors and concatenated layer biases.

        ``S`` has shape ``(b, n, d_s)`` and each ``Z`` tensor has shape
        ``(b, n, n, d_z)``.  Biases are concatenated in transformer-block
        order so downstream code can select them by layer.
        """

        # b is batch size, t token count, a atom count, and k the atom-window count.
        z_conditioned = self.pairwise_conditioner(
            z_trunk,
            relative_position_encoding,
        )  # (b, t, t, d_z)
        q, c, p, to_keys = self.atom_encoder(
            feats=feats,
            s_trunk=s_trunk,
            z=z_conditioned,
        )  # q/c: (b, a, d_a); p: (b, k, w, h_k, d_p); to_keys: callable
        atom_encoder_bias = _concatenate_biases(
            self.atom_enc_proj_z, p
        )  # (b, k, w, h_k, depth_enc * heads_enc)
        atom_decoder_bias = _concatenate_biases(
            self.atom_dec_proj_z, p
        )  # (b, k, w, h_k, depth_dec * heads_dec)
        token_transformer_bias = _concatenate_biases(
            self.token_trans_proj_z,
            z_conditioned,
        )  # (b, t, t, depth_token * heads_token)
        return (
            q,
            c,
            to_keys,
            atom_encoder_bias,
            atom_decoder_bias,
            token_transformer_bias,
        )  # tensor shapes are traced above; to_keys is the atom-key gatherer
