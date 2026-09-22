"""Predict atom and token confidence from Boltz2 sequence and pair states."""

import torch

from typing import Any
from torch import nn

from . import vb_const as const
from . import vb_layers_initialize as init
from .vb_layers_confidence_utils import (
    compute_aggregated_metric,
    compute_ptms,
)
from .vb_layers_pairformer import PairformerModule
from .vb_modules_encodersv2 import RelativePositionEncoder
from .vb_modules_trunkv2 import (
    ContactConditioning,
)
from .vb_modules_utils import LinearNoBias


def _token_slot_logits_to_atom_logits(
    token_logits: torch.Tensor,
    atom_to_token: torch.Tensor,
    atom_pad_mask: torch.Tensor,
    *,
    multiplicity: int,
) -> torch.Tensor:
    """Gather per-token atom-slot logits onto each example's atom table.

    ``token_logits`` is ordered as ``(batch * multiplicity, token, slot,
    channel)``. ``atom_to_token`` determines both the owning token and the
    within-token slot of every atom, so heterogeneous atom counts do not leak
    across batch rows.
    """

    # b: batch; m: multiplicity; a: atoms; t: tokens; c: channels.
    if token_logits.ndim != 4:
        raise ValueError(
            "token_logits must have shape (batch * multiplicity, token, slot, channel), "
            f"got {tuple(token_logits.shape)}."
        )
    if atom_to_token.ndim != 3:
        raise ValueError(
            "atom_to_token must have shape (batch, atom, token), "
            f"got {tuple(atom_to_token.shape)}."
        )
    if atom_pad_mask.shape != atom_to_token.shape[:2]:
        raise ValueError(
            "atom_pad_mask must match the batch and atom axes of atom_to_token; "
            f"got {tuple(atom_pad_mask.shape)} and {tuple(atom_to_token.shape)}."
        )
    if multiplicity < 1:
        raise ValueError(f"multiplicity must be positive, got {multiplicity}.")

    batch_size, atom_count, token_count = atom_to_token.shape
    if token_logits.shape[0] != batch_size * multiplicity:
        raise ValueError(
            "token_logits batch axis must equal batch * multiplicity; "
            f"got {token_logits.shape[0]} and {batch_size} * {multiplicity}."
        )
    if token_logits.shape[1] != token_count:
        raise ValueError(
            "token_logits and atom_to_token disagree on token count; "
            f"got {token_logits.shape[1]} and {token_count}."
        )

    valid_atoms = atom_pad_mask.bool()  # (b, a)
    assignments = atom_to_token.bool() & valid_atoms.unsqueeze(-1)  # (b, a, t)
    token_index = assignments.to(dtype=torch.int64).argmax(dim=-1)  # (b, a)
    # Cumulative one-hot counts give each atom its ordinal within its owning
    # token without assuming that atoms from different tokens are contiguous.
    cumulative_slots = assignments.to(dtype=torch.int64).cumsum(dim=1) - 1  # (b, a, t)
    slot_index = (cumulative_slots * assignments).sum(dim=-1)  # (b, a)

    slots_per_token = token_logits.shape[2]
    flattened_index = token_index * slots_per_token + slot_index  # (b, a)
    flattened_index = flattened_index.masked_fill(~valid_atoms, 0)  # (b, a)
    flattened_index = flattened_index.repeat_interleave(multiplicity, dim=0)  # (b * m, a)
    expanded_atom_mask = valid_atoms.repeat_interleave(multiplicity, dim=0)  # (b * m, a)

    flattened_logits = token_logits.flatten(1, 2)  # (b * m, t * n_slot, c)
    gather_index = flattened_index.unsqueeze(-1).expand(
        -1,
        atom_count,
        flattened_logits.shape[-1],
    )  # (b * m, a, c)
    atom_logits = torch.gather(flattened_logits, dim=1, index=gather_index)  # (b * m, a, c)
    return atom_logits * expanded_atom_mask.unsqueeze(-1).to(dtype=atom_logits.dtype)  # (b * m, a, c)


class ConfidenceModule(nn.Module):
    """Update confidence features with predicted geometry and a pairformer stack."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        pairformer_args: dict[str, Any],
        num_dist_bins: int = 64,
        token_level_confidence: bool = True,
        max_dist: float = 22,
        add_s_to_z_prod: bool = False,
        add_s_input_to_s: bool = False,
        add_z_input_to_z: bool = False,
        maximum_bond_distance: int = 0,
        bond_type_feature: bool = False,
        confidence_args: dict[str, Any] | None = None,
        compile_pairformer: bool = False,
        fix_sym_check: bool = False,
        cyclic_pos_enc: bool = False,
        return_latent_feats: bool = False,
        conditioning_cutoff_min: float | None = None,
        conditioning_cutoff_max: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.max_num_atoms_per_token = 23
        self.no_update_s = pairformer_args.get("no_update_s", False)
        boundaries = torch.linspace(2, max_dist, num_dist_bins - 1)  # (n_dist_bin - 1,)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        init.gating_init_(self.dist_bin_pairwise_embed.weight)
        self.token_level_confidence = token_level_confidence

        self.s_to_z = LinearNoBias(token_s, token_z)
        self.s_to_z_transpose = LinearNoBias(token_s, token_z)
        init.gating_init_(self.s_to_z.weight)
        init.gating_init_(self.s_to_z_transpose.weight)

        self.add_s_to_z_prod = add_s_to_z_prod
        if add_s_to_z_prod:
            self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
            self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)
            self.s_to_z_prod_out = LinearNoBias(token_z, token_z)
            init.gating_init_(self.s_to_z_prod_out.weight)

        self.s_inputs_norm = nn.LayerNorm(token_s)
        if not self.no_update_s:
            self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        self.add_s_input_to_s = add_s_input_to_s
        if add_s_input_to_s:
            self.s_input_to_s = LinearNoBias(token_s, token_s)
            init.gating_init_(self.s_input_to_s.weight)

        self.add_z_input_to_z = add_z_input_to_z
        if add_z_input_to_z:
            self.rel_pos = RelativePositionEncoder(
                token_z, fix_sym_check=fix_sym_check, cyclic_pos_enc=cyclic_pos_enc
            )
            self.token_bonds = nn.Linear(
                1 if maximum_bond_distance == 0 else maximum_bond_distance + 2,
                token_z,
                bias=False,
            )
            self.bond_type_feature = bond_type_feature
            if bond_type_feature:
                self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

            self.contact_conditioning = ContactConditioning(
                token_z=token_z,
                cutoff_min=conditioning_cutoff_min,
                cutoff_max=conditioning_cutoff_max,
            )
        pairformer_args["v2"] = True
        self.pairformer_stack = PairformerModule(
            token_s,
            token_z,
            **pairformer_args,
        )
        self.return_latent_feats = return_latent_feats

        self.confidence_heads = ConfidenceHeads(
            token_s,
            token_z,
            token_level_confidence=token_level_confidence,
            **confidence_args,
        )

    def forward(
        self,
        s_inputs: torch.Tensor,  # (b, t, d_s)
        s: torch.Tensor,  # (b, t, d_s)
        z: torch.Tensor,  # (b, t, t, d_z)
        x_pred: torch.Tensor,  # (b * m, a, 3) or (b, m, a, 3)
        feats: dict[str, torch.Tensor],
        pred_distogram_logits: torch.Tensor,  # (b, t, t, n_bin)
        multiplicity: int = 1,
        run_sequentially: bool = False,
        use_kernels: bool = False,
    ) -> dict[str, Any]:
        # b: base batch; m: diffusion multiplicity; t: tokens; a: atoms.
        if run_sequentially and multiplicity > 1:
            batch_size = z.shape[0]
            expected_shape = (batch_size, multiplicity)
            if x_pred.ndim >= 4 and x_pred.shape[:2] == expected_shape:
                sample_coordinates = x_pred  # (b, m, a, 3)
            elif x_pred.shape[0] == batch_size * multiplicity:
                sample_coordinates = x_pred.reshape(
                    batch_size,
                    multiplicity,
                    *x_pred.shape[1:],
                )  # (b, m, a, 3)
            else:
                raise ValueError(
                    "Sequential confidence expected coordinates with leading shape "
                    f"{expected_shape} or {batch_size * multiplicity}, got "
                    f"{tuple(x_pred.shape)}."
                )
            out_dicts = []
            for sample_idx in range(multiplicity):
                out_dicts.append(
                    self.forward(
                        s_inputs,
                        s,
                        z,
                        sample_coordinates[:, sample_idx],
                        feats,
                        pred_distogram_logits,
                        multiplicity=1,
                        run_sequentially=False,
                        use_kernels=use_kernels,
                    )
                )

            out_dict = {}
            for key in out_dicts[0]:
                if key != "pair_chains_iptm":
                    values = [out[key] for out in out_dicts]  # list of sample tensors, each (b, *feature_shape)
                    out_dict[key] = torch.stack(values, dim=1).flatten(0, 1)  # (b * m, *feature_shape)
                else:
                    pair_chains_iptm = {}
                    for chain_idx1 in out_dicts[0][key]:
                        chains_iptm = {}
                        for chain_idx2 in out_dicts[0][key][chain_idx1]:
                            values = [
                                out[key][chain_idx1][chain_idx2]
                                for out in out_dicts
                            ]  # list of sample chain-pair scores, each (b,)
                            chains_iptm[chain_idx2] = torch.stack(
                                values,
                                dim=1,
                            ).flatten(0, 1)  # (b * m,)
                        pair_chains_iptm[chain_idx1] = chains_iptm
                    out_dict[key] = pair_chains_iptm  # mapping of chain-pair scores, each (b * m,)
            return out_dict

        s_inputs = self.s_inputs_norm(s_inputs)  # (b, t, d_s)
        if not self.no_update_s:
            s = self.s_norm(s)  # (b, t, d_s)

        if self.add_s_input_to_s:
            s = s + self.s_input_to_s(s_inputs)  # (b, t, d_s)

        z = self.z_norm(z)  # (b, t, t, d_z)

        if self.add_z_input_to_z:
            relative_position_encoding = self.rel_pos(feats)  # (b, t, t, d_z)
            z = z + relative_position_encoding  # (b, t, t, d_z)
            z = z + self.token_bonds(feats["token_bonds"].float())  # (b, t, t, d_z)
            if self.bond_type_feature:
                z = z + self.token_bonds_type(feats["type_bonds"].long())  # (b, t, t, d_z)
            z = z + self.contact_conditioning(feats)  # (b, t, t, d_z)

        s = s.repeat_interleave(multiplicity, 0)  # (b * m, t, d_s)

        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )  # (b, t, t, d_z)
        if self.add_s_to_z_prod:
            z = z + self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
                * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
            )  # (b, t, t, d_z)

        z = z.repeat_interleave(multiplicity, 0)  # (b * m, t, t, d_z)
        s_inputs = s_inputs.repeat_interleave(multiplicity, 0)  # (b * m, t, d_s)

        token_to_rep_atom = feats["token_to_rep_atom"]  # (b, t, a)
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)  # (b * m, t, a)
        if len(x_pred.shape) == 4:
            b, multiplicity, n, _ = x_pred.shape
            x_pred = x_pred.reshape(b * multiplicity, n, -1)  # (b * m, a, 3)
        else:
            _, n, _ = x_pred.shape
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)  # (b * m, t, 3)
        d = torch.cdist(x_pred_repr, x_pred_repr)  # (b * m, t, t)
        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()  # (b * m, t, t)
        distogram = self.dist_bin_pairwise_embed(distogram)  # (b * m, t, t, d_z)
        z = z + distogram  # (b * m, t, t, d_z)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)  # (b * m, t)
        pair_mask = mask[:, :, None] * mask[:, None, :]  # (b * m, t, t)

        s_t, z_t = self.pairformer_stack(
            s, z, mask=mask, pair_mask=pair_mask, use_kernels=use_kernels
        )  # (b * m, t, d_s), (b * m, t, t, d_z)

        # Confidence heads consume the pairformer states without an additional residual.
        s = s_t  # (b * m, t, d_s)
        z = z_t  # (b * m, t, t, d_z)

        out_dict = {}

        if self.return_latent_feats:
            out_dict["s_conf"] = s  # (b * m, t, d_s)
            out_dict["z_conf"] = z  # (b * m, t, t, d_z)

        out_dict.update(
            self.confidence_heads(
                s=s,
                z=z,
                x_pred=x_pred,
                d=d,
                feats=feats,
                multiplicity=multiplicity,
                pred_distogram_logits=pred_distogram_logits,
            )
        )
        return out_dict


class ConfidenceHeads(nn.Module):
    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_plddt_bins: int = 50,
        num_pde_bins: int = 64,
        num_pae_bins: int = 64,
        token_level_confidence: bool = True,
        use_separate_heads: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.max_num_atoms_per_token = 23
        self.token_level_confidence = token_level_confidence
        self.use_separate_heads = use_separate_heads

        if self.use_separate_heads:
            self.to_pae_intra_logits = LinearNoBias(token_z, num_pae_bins)
            self.to_pae_inter_logits = LinearNoBias(token_z, num_pae_bins)
        else:
            self.to_pae_logits = LinearNoBias(token_z, num_pae_bins)

        if self.use_separate_heads:
            self.to_pde_intra_logits = LinearNoBias(token_z, num_pde_bins)
            self.to_pde_inter_logits = LinearNoBias(token_z, num_pde_bins)
        else:
            self.to_pde_logits = LinearNoBias(token_z, num_pde_bins)

        if self.token_level_confidence:
            self.to_plddt_logits = LinearNoBias(token_s, num_plddt_bins)
            self.to_resolved_logits = LinearNoBias(token_s, 2)
        else:
            self.to_plddt_logits = LinearNoBias(
                token_s, num_plddt_bins * self.max_num_atoms_per_token
            )
            self.to_resolved_logits = LinearNoBias(token_s, 2 * self.max_num_atoms_per_token)

    def forward(
        self,
        s: torch.Tensor,  # (b * m, t, d_s)
        z: torch.Tensor,  # (b * m, t, t, d_z)
        x_pred: torch.Tensor,  # (b * m, a, 3)
        d: torch.Tensor,  # (b * m, t, t)
        feats: dict[str, torch.Tensor],
        pred_distogram_logits: torch.Tensor,  # (b, t, t, n_bin)
        multiplicity: int = 1,
    ) -> dict[str, Any]:
        # b: base batch; m: diffusion multiplicity; t: tokens; a: atoms.
        if self.use_separate_heads:
            asym_id_token = feats["asym_id"]  # (b, t)
            is_same_chain = asym_id_token.unsqueeze(-1) == asym_id_token.unsqueeze(-2)  # (b, t, t)
            is_different_chain = ~is_same_chain  # (b, t, t)

        if self.use_separate_heads:
            pae_intra_logits = self.to_pae_intra_logits(z)  # (b * m, t, t, n_pae)
            pae_intra_logits = pae_intra_logits * is_same_chain.float().unsqueeze(-1)  # (b * m, t, t, n_pae)

            pae_inter_logits = self.to_pae_inter_logits(z)  # (b * m, t, t, n_pae)
            pae_inter_logits = pae_inter_logits * is_different_chain.float().unsqueeze(-1)  # (b * m, t, t, n_pae)

            pae_logits = pae_inter_logits + pae_intra_logits  # (b * m, t, t, n_pae)
        else:
            pae_logits = self.to_pae_logits(z)  # (b * m, t, t, n_pae)

        if self.use_separate_heads:
            pde_intra_logits = self.to_pde_intra_logits(z + z.transpose(1, 2))  # (b * m, t, t, n_pde)
            pde_intra_logits = pde_intra_logits * is_same_chain.float().unsqueeze(-1)  # (b * m, t, t, n_pde)

            pde_inter_logits = self.to_pde_inter_logits(z + z.transpose(1, 2))  # (b * m, t, t, n_pde)
            pde_inter_logits = pde_inter_logits * is_different_chain.float().unsqueeze(-1)  # (b * m, t, t, n_pde)

            pde_logits = pde_inter_logits + pde_intra_logits  # (b * m, t, t, n_pde)
        else:
            pde_logits = self.to_pde_logits(z + z.transpose(1, 2))  # (b * m, t, t, n_pde)
        resolved_logits = self.to_resolved_logits(s)  # (b * m, t, c_resolved); c_resolved is 2 or 2 * n_slot
        plddt_logits = self.to_plddt_logits(s)  # (b * m, t, c_plddt); c_plddt is n_plddt or n_slot * n_plddt

        ligand_weight = 20
        non_interface_weight = 1
        interface_weight = 10

        token_type = feats["mol_type"]  # (b, t)
        token_type = token_type.repeat_interleave(multiplicity, 0)  # (b * m, t)
        is_ligand_token = (token_type == const.chain_type_ids["NONPOLYMER"]).float()  # (b * m, t)

        if self.token_level_confidence:
            plddt = compute_aggregated_metric(plddt_logits)  # (b * m, n_item); n_item is t for token confidence, a for atom confidence
            token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)  # (b * m, t)
            complex_plddt = (plddt * token_pad_mask).sum(dim=-1) / token_pad_mask.sum(dim=-1)  # (b * m,)

            is_contact = (d < 8).float()  # (b * m, t, t)
            is_different_chain = (
                feats["asym_id"].unsqueeze(-1) != feats["asym_id"].unsqueeze(-2)
            ).float()  # (b, t, t)
            is_different_chain = is_different_chain.repeat_interleave(multiplicity, 0)  # (b * m, t, t)
            token_interface_mask = torch.max(
                is_contact * is_different_chain * (1 - is_ligand_token).unsqueeze(-1),
                dim=-1,
            ).values  # (b * m, t)
            token_non_interface_mask = (1 - token_interface_mask) * (1 - is_ligand_token)  # (b * m, t)
            iplddt_weight = (
                is_ligand_token * ligand_weight
                + token_interface_mask * interface_weight
                + token_non_interface_mask * non_interface_weight
            )  # (b * m, t)
            complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(dim=-1) / torch.sum(
                token_pad_mask * iplddt_weight, dim=-1
            )  # (b * m,)

        else:
            # token to atom conversion for resolved logits
            b, n, _ = resolved_logits.shape
            resolved_logits = resolved_logits.reshape(b, n, self.max_num_atoms_per_token, 2)  # (b * m, t, n_slot, 2)
            resolved_logits = _token_slot_logits_to_atom_logits(
                resolved_logits,
                feats["atom_to_token"],
                feats["atom_pad_mask"],
                multiplicity=multiplicity,
            )  # (b * m, a, 2)
            plddt_logits = plddt_logits.reshape(b, n, self.max_num_atoms_per_token, -1)  # (b * m, t, n_slot, n_plddt)
            plddt_logits = _token_slot_logits_to_atom_logits(
                plddt_logits,
                feats["atom_to_token"],
                feats["atom_pad_mask"],
                multiplicity=multiplicity,
            )  # (b * m, a, n_plddt)
            atom_pad_mask = feats["atom_pad_mask"].repeat_interleave(multiplicity, 0)  # (b * m, a)
            plddt = compute_aggregated_metric(plddt_logits)  # (b * m, n_item); n_item is t for token confidence, a for atom confidence

            complex_plddt = (plddt * atom_pad_mask).sum(dim=-1) / atom_pad_mask.sum(dim=-1)  # (b * m,)
            atom_to_token = feats["atom_to_token"].float().repeat_interleave(
                multiplicity,
                0,
            )  # (b * m, a, t)
            chain_id_token = feats["asym_id"].float().repeat_interleave(
                multiplicity,
                0,
            )  # (b * m, t)
            atom_type = torch.bmm(
                atom_to_token,
                token_type.float().unsqueeze(-1),
            ).squeeze(-1)  # (b * m, a)
            is_ligand_atom = (atom_type == const.chain_type_ids["NONPOLYMER"]).float()  # (b * m, a)
            d_atom = torch.cdist(x_pred, x_pred)  # (b * m, a, a)
            is_contact = (d_atom < 8).float()  # (b * m, a, a)
            chain_id_atom = torch.bmm(atom_to_token, chain_id_token.unsqueeze(-1)).squeeze(-1)  # (b * m, a)
            is_different_chain = (
                chain_id_atom.unsqueeze(-1) != chain_id_atom.unsqueeze(-2)
            ).float()  # (b * m, a, a)

            atom_interface_mask = torch.max(
                is_contact * is_different_chain * (1 - is_ligand_atom).unsqueeze(-1),
                dim=-1,
            ).values  # (b * m, a)
            atom_non_interface_mask = (1 - atom_interface_mask) * (1 - is_ligand_atom)  # (b * m, a)
            iplddt_weight = (
                is_ligand_atom * ligand_weight
                + atom_interface_mask * interface_weight
                + atom_non_interface_mask * non_interface_weight
            )  # (b * m, a)

            complex_iplddt = (plddt * atom_pad_mask * iplddt_weight).sum(dim=-1) / torch.sum(
                atom_pad_mask * iplddt_weight,
                dim=-1,
            )  # (b * m,)

        # Compute the gPDE and giPDE
        pde = compute_aggregated_metric(pde_logits, end=32)  # (b * m, t, t)
        pred_distogram_prob = nn.functional.softmax(
            pred_distogram_logits, dim=-1
        ).repeat_interleave(multiplicity, 0)  # (b * m, t, t, n_bin)
        contacts = torch.zeros((1, 1, 1, 64), dtype=pred_distogram_prob.dtype).to(
            pred_distogram_prob.device
        )  # (1, 1, 1, 64)
        contacts[:, :, :, :20] = 1.0  # (1, 1, 1, 20) selected bins; contacts retains (1, 1, 1, 64)
        prob_contact = (pred_distogram_prob * contacts).sum(-1)  # (b * m, t, t)
        token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)  # (b * m, t)
        token_pad_pair_mask = (
            token_pad_mask.unsqueeze(-1)
            * token_pad_mask.unsqueeze(-2)
            * (1 - torch.eye(token_pad_mask.shape[1], device=token_pad_mask.device).unsqueeze(0))
        )  # (b * m, t, t)
        token_pair_mask = token_pad_pair_mask * prob_contact  # (b * m, t, t)
        complex_pde_numerator = (pde * token_pair_mask).sum(dim=(1, 2))  # (b * m,)
        complex_pde_denominator = token_pair_mask.sum(dim=(1, 2))  # (b * m,)
        complex_pde = complex_pde_numerator / torch.where(
            complex_pde_denominator > 0,
            complex_pde_denominator,
            torch.ones_like(complex_pde_denominator),
        )  # (b * m,)
        asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)  # (b * m, t)
        token_interface_pair_mask = token_pair_mask * (
            asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        )  # (b * m, t, t)
        complex_ipde = (pde * token_interface_pair_mask).sum(dim=(1, 2)) / (
            token_interface_pair_mask.sum(dim=(1, 2)) + 1e-5
        )  # (b * m,)
        out_dict = dict(
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            pde=pde,
            plddt=plddt,
            complex_plddt=complex_plddt,
            complex_iplddt=complex_iplddt,
            complex_pde=complex_pde,
            complex_ipde=complex_ipde,
        )
        out_dict["pae_logits"] = pae_logits  # (b * m, t, t, n_pae)
        out_dict["pae"] = compute_aggregated_metric(pae_logits, end=32)  # (b * m, t, t)

        ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = compute_ptms(
            pae_logits, x_pred, feats, multiplicity
        )  # four (b * m,) scores and a mapping of chain-pair scores (b * m,)
        out_dict["ptm"] = ptm  # (b * m,)
        out_dict["iptm"] = iptm  # (b * m,)
        out_dict["ligand_iptm"] = ligand_iptm  # (b * m,)
        out_dict["protein_iptm"] = protein_iptm  # (b * m,)
        out_dict["pair_chains_iptm"] = pair_chains_iptm  # each chain-pair tensor: (b * m,)

        return out_dict
