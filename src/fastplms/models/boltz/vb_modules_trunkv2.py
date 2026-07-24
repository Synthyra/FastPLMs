"""Input, template, MSA, and prediction-head modules for the Boltz2 trunk."""

from __future__ import annotations

import torch
from typing import Any, cast
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
        self.encoding_unspecified = nn.Parameter(torch.zeros(token_z))  # (d_z,)
        self.encoding_unselected = nn.Parameter(torch.zeros(token_z))  # (d_z,)
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        """Return contact tensor C with shape ``(b, l, l, d_z)``."""

        if const.contact_conditioning_info["UNSPECIFIED"] != 0:
            raise ValueError("UNSPECIFIED contact conditioning must use channel zero")
        if const.contact_conditioning_info["UNSELECTED"] != 1:
            raise ValueError("UNSELECTED contact conditioning must use channel one")

        # c_contact is the number of contact-conditioning categories.
        categories = feats["contact_conditioning"]  # (b, l, l, c_contact)
        threshold = feats["contact_threshold"]  # (b, l, l)
        # (b, l, l)
        normalized = (threshold - self.cutoff_min) / (self.cutoff_max - self.cutoff_min)
        # (b, l, l, d_z)
        fourier = self.fourier_embedding(normalized.flatten()).reshape((*normalized.shape, -1))
        selected_features = torch.cat(  # (b, l, l, c_contact - 1 + d_z)
            [categories[..., 2:], normalized.unsqueeze(-1), fourier],
            dim=-1,
        )
        selected = self.encoder(selected_features)  # (b, l, l, d_z)
        special = categories[..., :2]  # (b, l, l, 2)
        return cast(
            Tensor,
            selected * (1 - special.sum(dim=-1, keepdim=True))
            + self.encoding_unspecified * special[..., 0:1]
            + self.encoding_unselected * special[..., 1:2],
        )  # (b, l, l, d_z)


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
            self.method_conditioning_init.weight.data.fill_(0)  # (n_method, d_s)
        if add_modified_flag:
            self.modified_conditioning_init = nn.Embedding(2, token_s)
            self.modified_conditioning_init.weight.data.fill_(0)  # (2, d_s)
        if add_cyclic_flag:
            self.cyclic_conditioning_init = nn.Linear(1, token_s, bias=False)
            self.cyclic_conditioning_init.weight.data.fill_(0)  # (d_s, 1)
        if add_mol_type_feat:
            self.mol_type_conditioning_init = nn.Embedding(
                len(const.chain_type_ids),
                token_s,
            )
            self.mol_type_conditioning_init.weight.data.fill_(0)  # (n_mol_type, d_s)

    def forward(self, feats: dict[str, Tensor], affinity: bool = False) -> Tensor:
        """Return embedded sequence tensor S with shape ``(b, l, d_s)``."""

        # n_atom is the padded atom count; k is the number of atom windows.
        residue_type = feats["res_type"].float()  # (b, l, n_token_type)
        suffix = "_affinity" if affinity else ""
        profile = feats[f"profile{suffix}"]  # (b, l, n_token_type)
        deletion_mean = feats[f"deletion_mean{suffix}"].unsqueeze(-1)  # (b, l, 1)
        # (b, n_atom, d_a), (b, n_atom, d_a), (b, k, w_q, w_k, d_az), callable
        atom_queries, atom_conditioning, atom_pairs, to_keys = self.atom_encoder(feats)
        atom_bias = self.atom_enc_proj_z(atom_pairs)  # (b, k, w_q, w_k, n_layer * h)
        atom_output, _, _, _ = self.atom_attention_encoder(
            feats=feats,
            q=atom_queries,
            c=atom_conditioning,
            atom_enc_bias=atom_bias,
            to_keys=to_keys,
        )  # (b, l, d_s), (b, n_atom, d_a), (b, n_atom, d_a), callable
        output = (  # (b, l, d_s)
            atom_output
            + self.res_type_encoding(residue_type)
            + self.msa_profile_encoding(torch.cat([profile, deletion_mean], dim=-1))
        )
        if self.add_method_conditioning:
            # method_feature: (b, l); output: (b, l, d_s).
            output = output + self.method_conditioning_init(feats["method_feature"])
        if self.add_modified_flag:
            # modified: (b, l); output: (b, l, d_s).
            output = output + self.modified_conditioning_init(feats["modified"])
        if self.add_cyclic_flag:
            cyclic = feats["cyclic_period"].clamp(max=1.0).unsqueeze(-1)  # (b, l, 1)
            output = output + self.cyclic_conditioning_init(cyclic)  # (b, l, d_s)
        if self.add_mol_type_feat:
            # mol_type: (b, l); output: (b, l, d_s).
            output = output + self.mol_type_conditioning_init(feats["mol_type"])
        return cast(Tensor, output)  # (b, l, d_s)


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
        # residue_type: (b, t, l, n_token_type); template_pair_mask: (b, t, l, l).
        residue_type = feats["template_restype"]  # (b, t, l, n_token_type)
        cb_mask = feats["template_mask_cb"]  # (b, t, l)
        frame_mask = feats["template_mask_frame"]  # (b, t, l)
        # (b, t, l, l, 1)
        cb_pair_mask = (cb_mask[..., :, None] * cb_mask[..., None, :]).unsqueeze(-1)
        # (b, t, l, l, 1)
        frame_pair_mask = (frame_mask[..., :, None] * frame_mask[..., None, :]).unsqueeze(-1)
        with torch.autocast(device_type="cuda", enabled=False):
            # template_cb: (b, t, l, 3).
            # (b, t, l, l)
            cb_distances = torch.cdist(feats["template_cb"], feats["template_cb"])
            boundaries = torch.linspace(  # (n_bin - 1,)
                self.min_dist,
                self.max_dist,
                self.num_bins - 1,
            ).to(cb_distances.device)
            bins = (cb_distances[..., None] > boundaries).sum(dim=-1).long()  # (b, t, l, l)
            distogram = F.one_hot(bins, num_classes=self.num_bins)  # (b, t, l, l, n_bin)

            # (b, t, 1, l, 3, 3)
            rotations = feats["template_frame_rot"].unsqueeze(2).transpose(-1, -2)
            # (b, t, 1, l, 3, 1)
            translations = feats["template_frame_t"].unsqueeze(2).unsqueeze(-1)
            # (b, t, l, 1, 3, 1)
            ca_coordinates = feats["template_ca"].unsqueeze(3).unsqueeze(-1)
            # (b, t, l, l, 3, 1)
            vectors = torch.matmul(rotations, ca_coordinates - translations)
            norms = torch.norm(vectors, dim=-1, keepdim=True)  # (b, t, l, l, 3, 1)
            unit_vectors = torch.where(  # (b, t, l, l, 3)
                norms > 0,
                vectors / norms,
                torch.zeros_like(vectors),
            ).squeeze(-1)
            pair_features = torch.cat(  # (b, t, l, l, n_bin + 5)
                [distogram, cb_pair_mask, unit_vectors, frame_pair_mask],
                dim=-1,
            )
            # (b, t, l, l, n_bin + 5)
            pair_features = pair_features * template_pair_mask.unsqueeze(-1)
            residue_i = residue_type[:, :, :, None].expand(  # (b, t, l, l, n_token_type)
                -1,
                -1,
                -1,
                residue_type.size(2),
                -1,
            )
            residue_j = residue_type[:, :, None, :].expand(  # (b, t, l, l, n_token_type)
                -1,
                -1,
                residue_type.size(2),
                -1,
                -1,
            )
            return cast(
                Tensor,
                self.a_proj(torch.cat([pair_features, residue_i, residue_j], dim=-1)),
            )  # (b, t, l, l, d_t)

    def forward(
        self,
        z: Tensor,
        feats: dict[str, Tensor],
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        """Aggregate template pair tensor V into trunk update U."""

        # z: (b, l, l, d_z); pair_mask: (b, l, l).
        residue_type = feats["template_restype"]  # (b, t, l, n_token_type)
        batch_size, template_count = residue_type.shape[:2]
        template_present = feats["template_mask"].any(dim=2).float()  # (b, t)
        present_count = template_present.sum(dim=1).clamp(min=1)  # (b,)
        features = self._template_features(  # (b, t, l, l, d_t)
            feats,
            self._template_pair_mask(feats, template_count),
        )
        expanded_mask = pair_mask[:, None].expand(-1, template_count, -1, -1)  # (b, t, l, l)
        expanded_mask = expanded_mask.reshape(  # (b * t, l, l)
            batch_size * template_count,
            *expanded_mask.shape[2:],
        )
        template_states = self.z_proj(self.z_norm(z[:, None])) + features  # (b, t, l, l, d_t)
        template_states = template_states.view(  # (b * t, l, l, d_t)
            batch_size * template_count,
            *template_states.shape[2:],
        )
        template_states = template_states + self.pairformer(  # (b * t, l, l, d_t)
            template_states,
            expanded_mask,
            use_kernels=use_kernels,
        )
        template_states = self.v_norm(template_states).view(  # (b, t, l, l, d_t)
            batch_size,
            template_count,
            *template_states.shape[1:],
        )
        weights = template_present[:, :, None, None, None]  # (b, t, 1, 1, 1)
        aggregate = (template_states * weights).sum(dim=1)  # (b, l, l, d_t)
        # (b, l, l, d_t)
        aggregate = aggregate / present_count[:, None, None, None].to(template_states)
        return cast(Tensor, self.u_proj(self.relu(aggregate)))  # (b, l, l, d_z)


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
        asym_id = feats["asym_id"]  # (b, l)
        same_chain = (asym_id[:, :, None] == asym_id[:, None, :]).float()  # (b, l, l)
        return same_chain[:, None].expand(-1, count, -1, -1)  # (b, t, l, l)


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
        visibility = feats["visibility_ids"]  # (b, t, l)
        return (visibility[..., :, None] == visibility[..., None, :]).float()  # (b, t, l, l)


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
        # pair_states: (b, l, l, d_z).
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
        # z: (b, l, l, d_z); emb: (b, l, d_s).
        # s is MSA depth; n_token_type is the residue vocabulary size.
        msa = feats["msa"]  # (b, s, l) or (b, s, l, n_token_type)
        if msa.dtype in (torch.long, torch.int32, torch.int64):
            msa = F.one_hot(msa, num_classes=const.num_tokens).float()  # (b, s, l, n_token_type)
        msa_mask = feats["msa_mask"]  # (b, s, l)
        components = [  # (b, s, l, n_token_type), then two (b, s, l, 1) tensors
            msa,
            feats["has_deletion"].unsqueeze(-1),  # (b, s, l, 1)
            feats["deletion_value"].unsqueeze(-1),  # (b, s, l, 1)
        ]
        if self.use_paired_feature:
            components.append(feats["msa_paired"].unsqueeze(-1))  # (b, s, l, 1)
        msa_input = torch.cat(components, dim=-1)  # (b, s, l, n_token_type + 2 or 3)
        if self.subsample_msa:
            indices = torch.randperm(msa.shape[1])[: self.num_subsampled_msa]  # (s_sub,)
            msa_input = msa_input[:, indices]  # (b, s_sub, l, n_token_type + 2 or 3)
            msa_mask = msa_mask[:, indices]  # (b, s_sub, l)

        msa_states = self.msa_proj(msa_input) + self.s_proj(emb).unsqueeze(1)  # (b, s, l, d_m)
        token_mask = feats["token_pad_mask"].float()  # (b, l)
        pair_mask = token_mask[:, :, None] * token_mask[:, None, :]  # (b, l, l)
        pair_states = z  # (b, l, l, d_z)
        for layer in self.layers:
            # Tensor arguments: pair_states (b, l, l, d_z), msa_states (b, s, l, d_m),
            # pair_mask (b, l, l), msa_mask (b, s, l).
            arguments = (
                pair_states,
                msa_states,
                pair_mask,
                msa_mask,
                *chunking,
                use_kernels,
            )
            if self.activation_checkpointing and self.training:
                pair_states, msa_states = checkpoint(  # (b, l, l, d_z), (b, s, l, d_m)
                    layer,
                    *arguments,
                )
            else:
                pair_states, msa_states = layer(*arguments)  # (b, l, l, d_z), (b, s, l, d_m)
        return pair_states  # (b, l, l, d_z)


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
        # z: (b, l, l, d_z); m: (b, s, l, d_m).
        # token_mask: (b, l, l); msa_mask: (b, s, l).
        dropout = get_dropout_mask(self.msa_dropout, m, self.training)  # (b, s, 1, 1)
        msa_states = m + dropout * self.pair_weighted_averaging(  # (b, s, l, d_m)
            m,
            z,
            token_mask,
            chunk_heads_pwa,
        )
        msa_states = msa_states + self.msa_transition(  # (b, s, l, d_m)
            msa_states,
            chunk_size_transition_msa,
        )
        pair_states = z + self.outer_product_mean(  # (b, l, l, d_z)
            msa_states,
            msa_mask,
            chunk_size_outer_product,
        )
        pair_states = self.pairformer_layer(  # (b, l, l, d_z)
            pair_states,
            token_mask,
            chunk_size_tri_attn,
            use_kernels=use_kernels,
        )
        return pair_states, msa_states  # (b, l, l, d_z), (b, s, l, d_m)


class BFactorModule(nn.Module):
    """Predict a per-token B-factor histogram."""

    def __init__(self, token_s: int, num_bins: int) -> None:
        super().__init__()
        self.bfactor = nn.Linear(token_s, num_bins)
        self.num_bins = num_bins

    def forward(self, s: Tensor) -> Tensor:
        # s: (..., d_s).
        return cast(Tensor, self.bfactor(s))  # (..., n_bin)


class DistogramModule(nn.Module):
    """Predict symmetric residue-pair distance histograms."""

    def __init__(self, token_z: int, num_bins: int, num_distograms: int = 1) -> None:
        super().__init__()
        self.distogram = nn.Linear(token_z, num_distograms * num_bins)
        self.num_distograms = num_distograms
        self.num_bins = num_bins

    def forward(self, z: Tensor) -> Tensor:
        # z: (b, l, l, d_z).
        symmetric = z + z.transpose(1, 2)  # (b, l, l, d_z)
        logits = self.distogram(symmetric)  # (b, l, l, n_distogram * n_bin)
        return cast(
            Tensor,
            logits.reshape(
                symmetric.shape[0],
                symmetric.shape[1],
                symmetric.shape[2],
                self.num_distograms,
                self.num_bins,
            ),
        )  # (b, l, l, n_distogram, n_bin)
