"""Input, template, MSA, and prediction-head modules for the Boltz2 trunk."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import vb_const as const
from .vb_layers_dropout import get_dropout_mask
from .vb_layers_outer_product_mean import OuterProductMean
from .vb_layers_pair_averaging import PairWeightedAveraging
from .vb_layers_pairformer import PairformerNoSeqLayer, PairformerNoSeqModule
from .vb_layers_transition import Transition
from .vb_modules_encodersv2 import (
    AtomAttentionEncoder,
    AtomEncoder,
    FourierEmbedding,
)


class ContactConditioning(nn.Module):
    """Encode selected, unselected, and distance-threshold contact constraints."""

    def __init__(self, token_z: int, cutoff_min: float, cutoff_max: float) -> None:
        super().__init__()
        self.fourier_embedding = FourierEmbedding(token_z)
        input_width = token_z + len(const.contact_conditioning_info) - 1
        self.encoder = nn.Linear(input_width, token_z)
        self.encoding_unspecified = nn.Parameter(torch.zeros(token_z))
        self.encoding_unselected = nn.Parameter(torch.zeros(token_z))
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        """Return contact tensor C with shape ``(b, l, l, d_z)``."""

        if const.contact_conditioning_info["UNSPECIFIED"] != 0:
            raise ValueError("UNSPECIFIED contact conditioning must use channel zero")
        if const.contact_conditioning_info["UNSELECTED"] != 1:
            raise ValueError("UNSELECTED contact conditioning must use channel one")

        categories = feats["contact_conditioning"]
        threshold = feats["contact_threshold"]
        normalized = (threshold - self.cutoff_min) / (self.cutoff_max - self.cutoff_min)
        fourier = self.fourier_embedding(normalized.flatten()).reshape((*normalized.shape, -1))
        selected_features = torch.cat(
            [categories[..., 2:], normalized.unsqueeze(-1), fourier],
            dim=-1,
        )
        selected = self.encoder(selected_features)
        special = categories[..., :2]
        return cast(
            Tensor,
            selected * (1 - special.sum(dim=-1, keepdim=True))
            + self.encoding_unspecified * special[..., 0:1]
            + self.encoding_unselected * special[..., 1:2],
        )


class InputEmbedder(nn.Module):
    """Combine atom, residue, profile, and optional experimental features."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        activation_checkpointing: bool = False,
        add_method_conditioning: bool = False,
        add_modified_flag: bool = False,
        add_cyclic_flag: bool = False,
        add_mol_type_feat: bool = False,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
    ) -> None:
        super().__init__()
        self.token_s = token_s
        self.add_method_conditioning = add_method_conditioning
        self.add_modified_flag = add_modified_flag
        self.add_cyclic_flag = add_cyclic_flag
        self.add_mol_type_feat = add_mol_type_feat
        self.atom_encoder = AtomEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=False,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )
        self.atom_enc_proj_z = nn.Sequential(
            nn.LayerNorm(atom_z),
            nn.Linear(atom_z, atom_encoder_depth * atom_encoder_heads, bias=False),
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=False,
            activation_checkpointing=activation_checkpointing,
        )
        self.res_type_encoding = nn.Linear(const.num_tokens, token_s, bias=False)
        self.msa_profile_encoding = nn.Linear(const.num_tokens + 1, token_s, bias=False)

        if add_method_conditioning:
            self.method_conditioning_init = nn.Embedding(const.num_method_types, token_s)
            self.method_conditioning_init.weight.data.fill_(0)
        if add_modified_flag:
            self.modified_conditioning_init = nn.Embedding(2, token_s)
            self.modified_conditioning_init.weight.data.fill_(0)
        if add_cyclic_flag:
            self.cyclic_conditioning_init = nn.Linear(1, token_s, bias=False)
            self.cyclic_conditioning_init.weight.data.fill_(0)
        if add_mol_type_feat:
            self.mol_type_conditioning_init = nn.Embedding(
                len(const.chain_type_ids),
                token_s,
            )
            self.mol_type_conditioning_init.weight.data.fill_(0)

    def forward(self, feats: dict[str, Tensor], affinity: bool = False) -> Tensor:
        """Return embedded sequence tensor S with shape ``(b, l, d_s)``."""

        residue_type = feats["res_type"].float()
        suffix = "_affinity" if affinity else ""
        profile = feats[f"profile{suffix}"]
        deletion_mean = feats[f"deletion_mean{suffix}"].unsqueeze(-1)
        atom_queries, atom_conditioning, atom_pairs, to_keys = self.atom_encoder(feats)
        atom_bias = self.atom_enc_proj_z(atom_pairs)
        atom_output, _, _, _ = self.atom_attention_encoder(
            feats=feats,
            q=atom_queries,
            c=atom_conditioning,
            atom_enc_bias=atom_bias,
            to_keys=to_keys,
        )
        output = (
            atom_output
            + self.res_type_encoding(residue_type)
            + self.msa_profile_encoding(torch.cat([profile, deletion_mean], dim=-1))
        )
        if self.add_method_conditioning:
            output = output + self.method_conditioning_init(feats["method_feature"])
        if self.add_modified_flag:
            output = output + self.modified_conditioning_init(feats["modified"])
        if self.add_cyclic_flag:
            cyclic = feats["cyclic_period"].clamp(max=1.0).unsqueeze(-1)
            output = output + self.cyclic_conditioning_init(cyclic)
        if self.add_mol_type_feat:
            output = output + self.mol_type_conditioning_init(feats["mol_type"])
        return cast(Tensor, output)


class _TemplateBase(nn.Module):
    def __init__(
        self,
        token_z: int,
        template_dim: int,
        template_blocks: int,
        dropout: float,
        pairwise_head_width: int,
        pairwise_num_heads: int,
        post_layer_norm: bool,
        activation_checkpointing: bool,
        min_dist: float,
        max_dist: float,
        num_bins: int,
    ) -> None:
        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.relu = nn.ReLU()
        self.z_norm = nn.LayerNorm(token_z)
        self.v_norm = nn.LayerNorm(template_dim)
        self.z_proj = nn.Linear(token_z, template_dim, bias=False)
        feature_width = const.num_tokens * 2 + num_bins + 5
        self.a_proj = nn.Linear(feature_width, template_dim, bias=False)
        self.u_proj = nn.Linear(template_dim, token_z, bias=False)
        self.pairformer = PairformerNoSeqModule(
            template_dim,
            num_blocks=template_blocks,
            dropout=dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            post_layer_norm=post_layer_norm,
            activation_checkpointing=activation_checkpointing,
        )

    def _template_pair_mask(self, feats: dict[str, Tensor], count: int) -> Tensor:
        raise NotImplementedError

    def _template_features(
        self,
        feats: dict[str, Tensor],
        template_pair_mask: Tensor,
    ) -> Tensor:
        residue_type = feats["template_restype"]
        cb_mask = feats["template_mask_cb"]
        frame_mask = feats["template_mask_frame"]
        cb_pair_mask = (cb_mask[..., :, None] * cb_mask[..., None, :]).unsqueeze(-1)
        frame_pair_mask = (frame_mask[..., :, None] * frame_mask[..., None, :]).unsqueeze(-1)
        with torch.autocast(device_type="cuda", enabled=False):
            cb_distances = torch.cdist(feats["template_cb"], feats["template_cb"])
            boundaries = torch.linspace(
                self.min_dist,
                self.max_dist,
                self.num_bins - 1,
            ).to(cb_distances.device)
            bins = (cb_distances[..., None] > boundaries).sum(dim=-1).long()
            distogram = F.one_hot(bins, num_classes=self.num_bins)

            rotations = feats["template_frame_rot"].unsqueeze(2).transpose(-1, -2)
            translations = feats["template_frame_t"].unsqueeze(2).unsqueeze(-1)
            ca_coordinates = feats["template_ca"].unsqueeze(3).unsqueeze(-1)
            vectors = torch.matmul(rotations, ca_coordinates - translations)
            norms = torch.norm(vectors, dim=-1, keepdim=True)
            unit_vectors = torch.where(
                norms > 0,
                vectors / norms,
                torch.zeros_like(vectors),
            ).squeeze(-1)
            pair_features = torch.cat(
                [distogram, cb_pair_mask, unit_vectors, frame_pair_mask],
                dim=-1,
            )
            pair_features = pair_features * template_pair_mask.unsqueeze(-1)
            residue_i = residue_type[:, :, :, None].expand(
                -1,
                -1,
                -1,
                residue_type.size(2),
                -1,
            )
            residue_j = residue_type[:, :, None, :].expand(
                -1,
                -1,
                residue_type.size(2),
                -1,
                -1,
            )
            return cast(
                Tensor,
                self.a_proj(torch.cat([pair_features, residue_i, residue_j], dim=-1)),
            )

    def forward(
        self,
        z: Tensor,
        feats: dict[str, Tensor],
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        """Aggregate template pair tensor V into trunk update U."""

        residue_type = feats["template_restype"]
        batch_size, template_count = residue_type.shape[:2]
        template_present = feats["template_mask"].any(dim=2).float()
        present_count = template_present.sum(dim=1).clamp(min=1)
        features = self._template_features(
            feats,
            self._template_pair_mask(feats, template_count),
        )
        expanded_mask = pair_mask[:, None].expand(-1, template_count, -1, -1)
        expanded_mask = expanded_mask.reshape(
            batch_size * template_count,
            *expanded_mask.shape[2:],
        )
        template_states = self.z_proj(self.z_norm(z[:, None])) + features
        template_states = template_states.view(
            batch_size * template_count,
            *template_states.shape[2:],
        )
        template_states = template_states + self.pairformer(
            template_states,
            expanded_mask,
            use_kernels=use_kernels,
        )
        template_states = self.v_norm(template_states).view(
            batch_size,
            template_count,
            *template_states.shape[1:],
        )
        weights = template_present[:, :, None, None, None]
        aggregate = (template_states * weights).sum(dim=1)
        aggregate = aggregate / present_count[:, None, None, None].to(template_states)
        return cast(Tensor, self.u_proj(self.relu(aggregate)))


class TemplateModule(_TemplateBase):
    """Aggregate templates while restricting features to the same chain."""

    def __init__(
        self,
        token_z: int,
        template_dim: int,
        template_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        num_bins: int = 38,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(
            token_z,
            template_dim,
            template_blocks,
            dropout,
            pairwise_head_width,
            pairwise_num_heads,
            post_layer_norm,
            activation_checkpointing,
            min_dist,
            max_dist,
            num_bins,
        )

    def _template_pair_mask(self, feats: dict[str, Tensor], count: int) -> Tensor:
        asym_id = feats["asym_id"]
        same_chain = (asym_id[:, :, None] == asym_id[:, None, :]).float()
        return same_chain[:, None].expand(-1, count, -1, -1)


class TemplateV2Module(_TemplateBase):
    """Aggregate templates under per-template visibility groups."""

    def __init__(
        self,
        token_z: int,
        template_dim: int,
        template_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        num_bins: int = 38,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(
            token_z,
            template_dim,
            template_blocks,
            dropout,
            pairwise_head_width,
            pairwise_num_heads,
            post_layer_norm,
            activation_checkpointing,
            min_dist,
            max_dist,
            num_bins,
        )

    def _template_pair_mask(self, feats: dict[str, Tensor], count: int) -> Tensor:
        del count
        visibility = feats["visibility_ids"]
        return (visibility[..., :, None] == visibility[..., None, :]).float()


class MSAModule(nn.Module):
    """Embed and update an MSA before returning its accumulated pair update."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        token_s: int,
        msa_blocks: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        activation_checkpointing: bool = False,
        use_paired_feature: bool = True,
        subsample_msa: bool = False,
        num_subsampled_msa: int = 1024,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.msa_blocks = msa_blocks
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.use_paired_feature = use_paired_feature
        self.activation_checkpointing = activation_checkpointing
        self.subsample_msa = subsample_msa
        self.num_subsampled_msa = num_subsampled_msa
        self.s_proj = nn.Linear(token_s, msa_s, bias=False)
        input_width = const.num_tokens + 2 + int(use_paired_feature)
        self.msa_proj = nn.Linear(input_width, msa_s, bias=False)
        self.layers = nn.ModuleList(
            [
                MSALayer(
                    msa_s,
                    token_z,
                    msa_dropout,
                    z_dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                )
                for _ in range(msa_blocks)
            ]
        )

    @staticmethod
    def _chunk_configuration(
        pair_states: Tensor,
        training: bool,
    ) -> tuple[bool, int | None, int | None, int | None, int | None]:
        if training:
            return False, None, None, None, None
        if pair_states.shape[1] > const.chunk_size_threshold:
            return True, 64, 32, 4, 128
        return False, None, None, None, 512

    def forward(
        self,
        z: Tensor,
        emb: Tensor,
        feats: dict[str, Tensor],
        use_kernels: bool = False,
    ) -> Tensor:
        """Return updated pair tensor Z after every MSA block."""

        chunking = self._chunk_configuration(z, self.training)
        msa = feats["msa"]
        if msa.dtype in (torch.long, torch.int32, torch.int64):
            msa = F.one_hot(msa, num_classes=const.num_tokens).float()
        msa_mask = feats["msa_mask"]
        components = [
            msa,
            feats["has_deletion"].unsqueeze(-1),
            feats["deletion_value"].unsqueeze(-1),
        ]
        if self.use_paired_feature:
            components.append(feats["msa_paired"].unsqueeze(-1))
        msa_input = torch.cat(components, dim=-1)
        if self.subsample_msa:
            indices = torch.randperm(msa.shape[1])[: self.num_subsampled_msa]
            msa_input = msa_input[:, indices]
            msa_mask = msa_mask[:, indices]

        msa_states = self.msa_proj(msa_input) + self.s_proj(emb).unsqueeze(1)
        token_mask = feats["token_pad_mask"].float()
        pair_mask = token_mask[:, :, None] * token_mask[:, None, :]
        pair_states = z
        for layer in self.layers:
            arguments = (
                pair_states,
                msa_states,
                pair_mask,
                msa_mask,
                *chunking,
                use_kernels,
            )
            if self.activation_checkpointing and self.training:
                pair_states, msa_states = checkpoint(
                    layer,
                    *arguments,
                )
            else:
                pair_states, msa_states = layer(*arguments)
        return pair_states


class MSALayer(nn.Module):
    """Exchange information between MSA tensor M and pair tensor Z."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.msa_dropout = msa_dropout
        self.msa_transition = Transition(msa_s, msa_s * 4)
        self.pair_weighted_averaging = PairWeightedAveraging(
            c_m=msa_s,
            c_z=token_z,
            c_h=32,
            num_heads=8,
        )
        self.pairformer_layer = PairformerNoSeqLayer(
            token_z=token_z,
            dropout=z_dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
        )
        self.outer_product_mean = OuterProductMean(msa_s, 32, token_z)

    def forward(
        self,
        z: Tensor,
        m: Tensor,
        token_mask: Tensor,
        msa_mask: Tensor,
        chunk_heads_pwa: bool = False,
        chunk_size_transition_z: int | None = None,
        chunk_size_transition_msa: int | None = None,
        chunk_size_outer_product: int | None = None,
        chunk_size_tri_attn: int | None = None,
        use_kernels: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return updated Z and M tensors."""

        del chunk_size_transition_z
        dropout = get_dropout_mask(self.msa_dropout, m, self.training)
        msa_states = m + dropout * self.pair_weighted_averaging(
            m,
            z,
            token_mask,
            chunk_heads_pwa,
        )
        msa_states = msa_states + self.msa_transition(
            msa_states,
            chunk_size_transition_msa,
        )
        pair_states = z + self.outer_product_mean(
            msa_states,
            msa_mask,
            chunk_size_outer_product,
        )
        pair_states = self.pairformer_layer(
            pair_states,
            token_mask,
            chunk_size_tri_attn,
            use_kernels=use_kernels,
        )
        return pair_states, msa_states


class BFactorModule(nn.Module):
    """Predict a per-token B-factor histogram."""

    def __init__(self, token_s: int, num_bins: int) -> None:
        super().__init__()
        self.bfactor = nn.Linear(token_s, num_bins)
        self.num_bins = num_bins

    def forward(self, s: Tensor) -> Tensor:
        return cast(Tensor, self.bfactor(s))


class DistogramModule(nn.Module):
    """Predict symmetric residue-pair distance histograms."""

    def __init__(self, token_z: int, num_bins: int, num_distograms: int = 1) -> None:
        super().__init__()
        self.distogram = nn.Linear(token_z, num_distograms * num_bins)
        self.num_distograms = num_distograms
        self.num_bins = num_bins

    def forward(self, z: Tensor) -> Tensor:
        symmetric = z + z.transpose(1, 2)
        logits = self.distogram(symmetric)
        return cast(
            Tensor,
            logits.reshape(
                symmetric.shape[0],
                symmetric.shape[1],
                symmetric.shape[2],
                self.num_distograms,
                self.num_bins,
            ),
        )
