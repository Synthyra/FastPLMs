"""Token, pair, and atom encoders used by the local Boltz2 runtime.

The classes retain checkpoint-facing submodule names while separating feature
assembly, window indexing, conditioning, and atom-token aggregation into small
mechanism-specific units.  No production import depends on the upstream Boltz
package.

The Fourier and atom-attention mechanisms derive from AlphaFold 3 community
implementations under MIT terms.  See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from math import pi

import torch
from torch import nn
from torch.nn.functional import one_hot

from . import vb_layers_initialize as init
from .vb_layers_transition import Transition
from .vb_modules_transformersv2 import AtomTransformer
from .vb_modules_utils import LinearNoBias


def _transition_stack(
    count: int,
    dim: int,
    hidden_dim: int,
) -> nn.ModuleList:
    return nn.ModuleList([Transition(dim=dim, hidden=hidden_dim) for _ in range(count)])


class FourierEmbedding(nn.Module):
    """Embed diffusion time with fixed random Fourier frequencies."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(1, dim)
        nn.init.normal_(self.proj.weight, mean=0, std=1)
        nn.init.normal_(self.proj.bias, mean=0, std=1)
        self.proj.requires_grad_(False)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Map times with shape ``(b,)`` to an embedding ``H: (b, d)``."""

        random_phase = self.proj(times.reshape(-1, 1))
        return torch.cos(2 * pi * random_phase)


def _pairwise_difference(values: torch.Tensor) -> torch.Tensor:
    return values[:, :, None] - values[:, None, :]


class RelativePositionEncoder(nn.Module):
    """Encode residue, token, entity, and symmetry relationships."""

    def __init__(
        self,
        token_z: int,
        r_max: int = 32,
        s_max: int = 2,
        fix_sym_check: bool = False,
        cyclic_pos_enc: bool = False,
    ) -> None:
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        input_dim = 4 * (r_max + 1) + 2 * (s_max + 1) + 1
        self.linear_layer = LinearNoBias(input_dim, token_z)
        self.fix_sym_check = fix_sym_check
        self.cyclic_pos_enc = cyclic_pos_enc

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return relative pair features ``Z: (b, n, n, token_z)``."""

        same_chain = feats["asym_id"][:, :, None] == feats["asym_id"][:, None, :]
        same_residue = feats["residue_index"][:, :, None] == feats["residue_index"][:, None, :]
        same_entity = feats["entity_id"][:, :, None] == feats["entity_id"][:, None, :]

        residue_offset = _pairwise_difference(feats["residue_index"])
        if self.cyclic_pos_enc and torch.any(feats["cyclic_period"] > 0):
            period = torch.where(
                feats["cyclic_period"] > 0,
                feats["cyclic_period"],
                torch.zeros_like(feats["cyclic_period"]) + 10000,
            )
            residue_offset = (residue_offset - period * torch.round(residue_offset / period)).long()
        residue_offset = torch.clip(
            residue_offset + self.r_max,
            0,
            2 * self.r_max,
        )
        residue_offset = torch.where(
            same_chain,
            residue_offset,
            torch.zeros_like(residue_offset) + 2 * self.r_max + 1,
        )
        residue_features = one_hot(residue_offset, 2 * self.r_max + 2)

        token_offset = torch.clip(
            _pairwise_difference(feats["token_index"]) + self.r_max,
            0,
            2 * self.r_max,
        )
        token_offset = torch.where(
            same_chain & same_residue,
            token_offset,
            torch.zeros_like(token_offset) + 2 * self.r_max + 1,
        )
        token_features = one_hot(token_offset, 2 * self.r_max + 2)

        symmetry_offset = torch.clip(
            _pairwise_difference(feats["sym_id"]) + self.s_max,
            0,
            2 * self.s_max,
        )
        invalid_symmetry = ~same_entity if self.fix_sym_check else same_chain
        symmetry_offset = torch.where(
            invalid_symmetry,
            torch.zeros_like(symmetry_offset) + 2 * self.s_max + 1,
            symmetry_offset,
        )
        symmetry_features = one_hot(symmetry_offset, 2 * self.s_max + 2)

        pair_features = torch.cat(
            (
                residue_features.float(),
                token_features.float(),
                same_entity.unsqueeze(-1).float(),
                symmetry_features.float(),
            ),
            dim=-1,
        )
        return self.linear_layer(pair_features)


class SingleConditioning(nn.Module):
    """Condition token features on trunk inputs and diffusion time."""

    def __init__(
        self,
        sigma_data: float,
        token_s: int = 384,
        dim_fourier: int = 256,
        num_transitions: int = 2,
        transition_expansion_factor: int = 2,
        eps: float = 1e-20,
        disable_times: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.sigma_data = sigma_data
        self.disable_times = disable_times
        conditioning_dim = 2 * token_s
        self.norm_single = nn.LayerNorm(conditioning_dim)
        self.single_embed = nn.Linear(conditioning_dim, conditioning_dim)
        if not disable_times:
            self.fourier_embed = FourierEmbedding(dim_fourier)
            self.norm_fourier = nn.LayerNorm(dim_fourier)
            self.fourier_to_single = LinearNoBias(dim_fourier, conditioning_dim)
        self.transitions = _transition_stack(
            num_transitions,
            conditioning_dim,
            transition_expansion_factor * conditioning_dim,
        )

    def forward(
        self,
        times: torch.Tensor,
        s_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return conditioned token features ``S: (b, n, 2 * token_s)``."""

        single = torch.cat((s_trunk, s_inputs), dim=-1)
        single = self.single_embed(self.norm_single(single))
        normalized_fourier = None
        if not self.disable_times:
            fourier = self.fourier_embed(times)
            normalized_fourier = self.norm_fourier(fourier)
            time_condition = self.fourier_to_single(normalized_fourier)
            single = time_condition[:, None, :] + single
        for transition in self.transitions:
            single = transition(single) + single
        return single, normalized_fourier


class PairwiseConditioning(nn.Module):
    """Fuse trunk pair features with relative-position features."""

    def __init__(
        self,
        token_z: int,
        dim_token_rel_pos_feats: int,
        num_transitions: int = 2,
        transition_expansion_factor: int = 2,
    ) -> None:
        super().__init__()
        combined_dim = token_z + dim_token_rel_pos_feats
        self.dim_pairwise_init_proj = nn.Sequential(
            nn.LayerNorm(combined_dim),
            LinearNoBias(combined_dim, token_z),
        )
        self.transitions = _transition_stack(
            num_transitions,
            token_z,
            transition_expansion_factor * token_z,
        )

    def forward(
        self,
        z_trunk: torch.Tensor,
        token_rel_pos_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Return conditioned pair features ``Z: (b, n, n, token_z)``."""

        pair = self.dim_pairwise_init_proj(torch.cat((z_trunk, token_rel_pos_feats), dim=-1))
        for transition in self.transitions:
            pair = transition(pair) + pair
        return pair


def get_indexing_matrix(
    k: int,
    w: int,
    h: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the atom-window gather matrix used for local attention keys."""

    for name, value in (("k", k), ("w", w), ("h", h)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if w % 2 != 0:
        raise ValueError(f"w must be even, got {w}.")
    half_window = w // 2
    if h % half_window != 0:
        raise ValueError(f"h must be divisible by w // 2 ({half_window}), got {h}.")
    key_blocks = h // half_window
    if key_blocks % 2 != 0:
        raise ValueError(
            f"h must contain an even number of half-window key blocks; received {key_blocks}."
        )

    positions = torch.arange(2 * k, device=device)
    relative_blocks = ((positions.unsqueeze(0) - positions.unsqueeze(1)) + key_blocks // 2).clamp(
        min=0, max=key_blocks + 1
    )
    relative_blocks = relative_blocks.view(k, 2, 2 * k)[:, 0, :]
    selectors = one_hot(relative_blocks, num_classes=key_blocks + 2)[..., 1:-1].transpose(1, 0)
    return selectors.reshape(2 * k, key_blocks * k).float()


def single_to_keys(
    single: torch.Tensor,
    indexing_matrix: torch.Tensor,
    w: int,
    h: int,
) -> torch.Tensor:
    """Gather a sequence tensor into overlapping local key windows."""

    b, n, d = single.shape
    k = n // w
    half_windows = single.view(b, 2 * k, w // 2, d)
    gathered = torch.einsum("b j i d, j k -> b k i d", half_windows, indexing_matrix)
    return gathered.reshape(b, k, h, d)


class AtomEncoder(nn.Module):
    """Encode reference atoms and local atom-pair geometry."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        structure_prediction: bool = True,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
    ) -> None:
        super().__init__()
        self.embed_atom_features = nn.Linear(atom_feature_dim, atom_s)
        self.embed_atompair_ref_pos = LinearNoBias(3, atom_z)
        self.embed_atompair_ref_dist = LinearNoBias(1, atom_z)
        self.embed_atompair_mask = LinearNoBias(1, atom_z)
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.use_no_atom_char = use_no_atom_char
        self.use_atom_backbone_feat = use_atom_backbone_feat
        self.use_residue_feats_atoms = use_residue_feats_atoms
        self.structure_prediction = structure_prediction

        if structure_prediction:
            self.s_to_c_trans = nn.Sequential(
                nn.LayerNorm(token_s),
                LinearNoBias(token_s, atom_s),
            )
            init.final_init_(self.s_to_c_trans[1].weight)
            self.z_to_p_trans = nn.Sequential(
                nn.LayerNorm(token_z),
                LinearNoBias(token_z, atom_z),
            )
            init.final_init_(self.z_to_p_trans[1].weight)

        self.c_to_p_trans_k = nn.Sequential(nn.ReLU(), LinearNoBias(atom_s, atom_z))
        init.final_init_(self.c_to_p_trans_k[1].weight)
        self.c_to_p_trans_q = nn.Sequential(nn.ReLU(), LinearNoBias(atom_s, atom_z))
        init.final_init_(self.c_to_p_trans_q[1].weight)
        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
        )
        init.final_init_(self.p_mlp[5].weight)

    def _assemble_atom_features(
        self,
        feats: dict[str, torch.Tensor],
        b: int,
        n: int,
    ) -> torch.Tensor:
        feature_parts = [
            feats["ref_pos"],
            feats["ref_charge"].unsqueeze(-1),
            feats["ref_element"],
        ]
        if not self.use_no_atom_char:
            feature_parts.append(feats["ref_atom_name_chars"].reshape(b, n, 4 * 64))
        if self.use_atom_backbone_feat:
            feature_parts.append(feats["atom_backbone_feat"])
        if self.use_residue_feats_atoms:
            residue_features = torch.cat(
                (
                    feats["res_type"],
                    feats["modified"].unsqueeze(-1),
                    one_hot(feats["mol_type"], num_classes=4).float(),
                ),
                dim=-1,
            )
            feature_parts.append(torch.bmm(feats["atom_to_token"].float(), residue_features))
        return torch.cat(feature_parts, dim=-1)

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        s_trunk: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Callable[[torch.Tensor], torch.Tensor],
    ]:
        """Return atom queries, conditioned singles, pairs, and key gatherer."""

        with torch.autocast("cuda", enabled=False):
            b, n, _ = feats["ref_pos"].shape
            atom_mask = feats["atom_pad_mask"].bool()
            atom_positions = feats["ref_pos"]
            atom_space = feats["ref_space_uid"]
            atom_features = self._assemble_atom_features(feats, b, n)
            conditioned_single = self.embed_atom_features(atom_features)

            w = self.atoms_per_window_queries
            h = self.atoms_per_window_keys
            b, n = conditioned_single.shape[:2]
            k = n // w
            indexing = get_indexing_matrix(k, w, h, conditioned_single.device)
            to_keys = partial(single_to_keys, indexing_matrix=indexing, w=w, h=h)

            position_queries = atom_positions.view(b, k, w, 1, 3)
            position_keys = to_keys(atom_positions).view(b, k, 1, h, 3)
            displacement = position_keys - position_queries
            inverse_squared_distance = 1 / (
                1 + torch.sum(displacement * displacement, dim=-1, keepdim=True)
            )

            mask_queries = atom_mask.view(b, k, w, 1)
            mask_keys = to_keys(atom_mask.unsqueeze(-1).float()).view(b, k, 1, h).bool()
            space_queries = atom_space.view(b, k, w, 1)
            space_keys = to_keys(atom_space.unsqueeze(-1).float()).view(b, k, 1, h).long()
            valid_pair = (
                (mask_queries & mask_keys & (space_queries == space_keys)).float().unsqueeze(-1)
            )

            pair = self.embed_atompair_ref_pos(displacement) * valid_pair
            pair = pair + self.embed_atompair_ref_dist(inverse_squared_distance) * valid_pair
            pair = pair + self.embed_atompair_mask(valid_pair) * valid_pair
            query = conditioned_single

            if self.structure_prediction:
                if s_trunk is None or z is None:
                    raise ValueError("structure prediction requires S and Z trunk tensors")
                atom_to_token = feats["atom_to_token"].float()
                single_update = self.s_to_c_trans(s_trunk.float())
                single_update = torch.bmm(atom_to_token, single_update)
                conditioned_single = conditioned_single + single_update.to(conditioned_single)

                token_queries = atom_to_token.view(b, k, w, atom_to_token.shape[-1])
                token_keys = to_keys(atom_to_token)
                pair_update = self.z_to_p_trans(z.float())
                pair_update = torch.einsum(
                    "bijd,bwki,bwlj->bwkld",
                    pair_update,
                    token_queries,
                    token_keys,
                )
                pair = pair + pair_update.to(pair)

            pair = pair + self.c_to_p_trans_q(
                conditioned_single.view(b, k, w, 1, conditioned_single.shape[-1])
            )
            pair = pair + self.c_to_p_trans_k(
                to_keys(conditioned_single).view(
                    b,
                    k,
                    1,
                    h,
                    conditioned_single.shape[-1],
                )
            )
            pair = pair + self.p_mlp(pair)
        return query, conditioned_single, pair, to_keys


class AtomAttentionEncoder(nn.Module):
    """Run local atom attention and aggregate atoms to tokens."""

    def __init__(
        self,
        atom_s: int,
        token_s: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        structure_prediction: bool = True,
        activation_checkpointing: bool = False,
        transformer_post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.r_to_q_trans = LinearNoBias(3, atom_s)
            init.final_init_(self.r_to_q_trans.weight)
        self.atom_encoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            activation_checkpointing=activation_checkpointing,
            post_layer_norm=transformer_post_layer_norm,
        )
        output_dim = 2 * token_s if structure_prediction else token_s
        self.atom_to_token_trans = nn.Sequential(
            LinearNoBias(atom_s, output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        q: torch.Tensor,
        c: torch.Tensor,
        atom_enc_bias: torch.Tensor,
        to_keys: Callable[[torch.Tensor], torch.Tensor],
        r: torch.Tensor | None = None,
        multiplicity: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable]:
        b, n, _ = feats["ref_pos"].shape
        del b, n
        atom_mask = feats["atom_pad_mask"].bool()
        if self.structure_prediction:
            if r is None:
                raise ValueError("structure prediction requires atom coordinates R")
            q = q.repeat_interleave(multiplicity, 0)
            q = q + self.r_to_q_trans(r)
        c = c.repeat_interleave(multiplicity, 0)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        q = self.atom_encoder(
            q=q,
            mask=atom_mask,
            c=c,
            bias=atom_enc_bias,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        with torch.autocast("cuda", enabled=False):
            atom_update = self.atom_to_token_trans(q).float()
            atom_to_token = feats["atom_to_token"].float()
            atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
            atom_to_token_mean = atom_to_token / (atom_to_token.sum(dim=1, keepdim=True) + 1e-6)
            token_update = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_update)
        return token_update.to(q), q, c, to_keys


class AtomAttentionDecoder(nn.Module):
    """Map token updates back to atoms and predict coordinate displacements."""

    def __init__(
        self,
        atom_s: int,
        token_s: int,
        attn_window_queries: int,
        attn_window_keys: int,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        activation_checkpointing: bool = False,
        transformer_post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.a_to_q_trans = LinearNoBias(2 * token_s, atom_s)
        init.final_init_(self.a_to_q_trans.weight)
        self.atom_decoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            post_layer_norm=transformer_post_layer_norm,
        )
        if transformer_post_layer_norm:
            self.atom_feat_to_atom_pos_update = LinearNoBias(atom_s, 3)
            init.final_init_(self.atom_feat_to_atom_pos_update.weight)
        else:
            self.atom_feat_to_atom_pos_update = nn.Sequential(
                nn.LayerNorm(atom_s),
                LinearNoBias(atom_s, 3),
            )
            init.final_init_(self.atom_feat_to_atom_pos_update[1].weight)

    def forward(
        self,
        a: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        atom_dec_bias: torch.Tensor,
        feats: dict[str, torch.Tensor],
        to_keys: Callable[[torch.Tensor], torch.Tensor],
        multiplicity: int = 1,
    ) -> torch.Tensor:
        """Return atom-coordinate updates ``R_update: (b, a, 3)``."""

        with torch.autocast("cuda", enabled=False):
            atom_to_token = feats["atom_to_token"].float()
            atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
            token_update = self.a_to_q_trans(a.float())
            atom_update = torch.bmm(atom_to_token, token_update)
        q = q + atom_update.to(q)
        atom_mask = feats["atom_pad_mask"].repeat_interleave(multiplicity, 0)
        q = self.atom_decoder(
            q=q,
            mask=atom_mask,
            c=c,
            bias=atom_dec_bias,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )
        return self.atom_feat_to_atom_pos_update(q)
