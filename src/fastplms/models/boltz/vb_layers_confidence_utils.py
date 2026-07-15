"""Confidence geometry and aggregate metrics for Boltz2 outputs."""

from __future__ import annotations

import torch
from torch import nn

from . import vb_const as const


def compute_collinear_mask(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Mark nondegenerate vector pairs whose angle defines a stable frame."""

    norm1 = torch.norm(v1, dim=1, keepdim=True)
    norm2 = torch.norm(v2, dim=1, keepdim=True)
    unit1 = v1 / (norm1 + 1e-6)
    unit2 = v2 / (norm2 + 1e-6)
    separated = torch.abs(torch.sum(unit1 * unit2, dim=1)) < 0.9063
    return separated & (norm1.reshape(-1) > 1e-2) & (norm2.reshape(-1) > 1e-2)


def _atom_chain_ids(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        return torch.bmm(
            feats["atom_to_token"].float(),
            feats["asym_id"].unsqueeze(-1).float(),
        ).squeeze(-1)


def _replace_nonpolymer_frames(
    coordinates: torch.Tensor,
    frame_indices: torch.Tensor,
    feats: dict[str, torch.Tensor],
    atom_chain_ids: torch.Tensor,
    resolved_mask: torch.Tensor | None,
    *,
    inference: bool,
) -> None:
    token_chain_ids = feats["asym_id"]
    for batch_index, sample_coordinates in enumerate(coordinates):
        token_offset = 0
        atom_offset = 0
        for chain_id in torch.unique(token_chain_ids[batch_index]):
            token_mask = (token_chain_ids[batch_index] == chain_id) * feats["token_pad_mask"][
                batch_index
            ]
            atom_mask = (atom_chain_ids[batch_index] == chain_id) * feats["atom_pad_mask"][
                batch_index
            ]
            token_count = int(token_mask.sum().item())
            atom_count = int(atom_mask.sum().item())
            is_nonpolymer = (
                feats["mol_type"][batch_index, token_offset] == const.chain_type_ids["NONPOLYMER"]
            )
            if is_nonpolymer and atom_count >= 3:
                chain_resolved = (
                    feats["atom_pad_mask"][batch_index]
                    if inference
                    else (
                        feats["atom_resolved_mask"][batch_index]
                        if resolved_mask is None
                        else resolved_mask[batch_index]
                    )
                )
                chain_atom_mask = atom_mask.bool()
                chain_coordinates = sample_coordinates[:, chain_atom_mask]
                differences = chain_coordinates[:, None, :, :] - chain_coordinates[:, :, None, :]
                distances = differences.square().sum(dim=-1) ** 0.5
                valid = chain_resolved[chain_atom_mask]
                invalid_pairs = 1 - (valid[None, :] * valid[:, None]).to(torch.float32)
                invalid_pairs[invalid_pairs == 1] = torch.inf
                nearest = torch.sort(distances + invalid_pairs, dim=2).indices
                frames = (
                    torch.cat(
                        (nearest[:, :, 1:2], nearest[:, :, 0:1], nearest[:, :, 2:3]),
                        dim=2,
                    )
                    + atom_offset
                )
                frame_indices[
                    batch_index,
                    :,
                    token_offset : token_offset + atom_count,
                    :,
                ] = frames
            token_offset += token_count
            atom_offset += atom_count


def compute_frame_pred(
    pred_atom_coords: torch.Tensor,
    frames_idx_true: torch.Tensor,
    feats: dict[str, torch.Tensor],
    multiplicity: int,
    resolved_mask: torch.Tensor | None = None,
    inference: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct predicted local frames and their validity mask.

    ``pred_atom_coords`` has shape ``(b * multiplicity, a, 3)``.  Polymer
    frames retain the supplied atom indices; nonpolymer frames use each
    atom's three nearest resolved neighbors.
    """

    atom_chain_ids = _atom_chain_ids(feats)
    expanded_batch, _, _ = pred_atom_coords.shape
    base_batch = expanded_batch // multiplicity
    coordinates = pred_atom_coords.reshape(base_batch, multiplicity, -1, 3)
    frame_indices = frames_idx_true.clone().repeat_interleave(multiplicity, 0)
    frame_indices = frame_indices.reshape(base_batch, multiplicity, -1, 3)
    _replace_nonpolymer_frames(
        coordinates,
        frame_indices,
        feats,
        atom_chain_ids,
        resolved_mask,
        inference=inference,
    )

    batch_indices = torch.arange(base_batch, device=frame_indices.device)[:, None, None, None]
    sample_indices = torch.arange(multiplicity, device=frame_indices.device)[None, :, None, None]
    frame_coordinates = coordinates[
        batch_indices,
        sample_indices,
        frame_indices,
    ].reshape(-1, 3, 3)
    valid_frames = compute_collinear_mask(
        frame_coordinates[:, 1] - frame_coordinates[:, 0],
        frame_coordinates[:, 1] - frame_coordinates[:, 2],
    ).reshape(base_batch, multiplicity, -1)
    return frame_indices, valid_frames * feats["token_pad_mask"][:, None, :]


def compute_aggregated_metric(logits: torch.Tensor, end: float = 1.0) -> torch.Tensor:
    """Convert categorical confidence logits to bin-center expectations."""

    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    centers = torch.arange(
        0.5 * bin_width,
        end,
        bin_width,
        device=logits.device,
    )
    probabilities = nn.functional.softmax(logits, dim=-1)
    broadcast_shape = (1,) * (probabilities.ndim - 1) + centers.shape
    return torch.sum(probabilities * centers.view(broadcast_shape), dim=-1)


def tm_function(d: torch.Tensor, n_res: torch.Tensor) -> torch.Tensor:
    """Evaluate the TM-score distance kernel."""

    distance_scale = 1.24 * (torch.clip(n_res, min=19) - 15) ** (1 / 3) - 1.8
    return 1 / (1 + (d / distance_scale) ** 2)


def _maximum_masked_tm(
    expected_tm: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    per_anchor = torch.sum(expected_tm * pair_mask, dim=-1) / (torch.sum(pair_mask, dim=-1) + 1e-5)
    return torch.max(per_anchor, dim=1).values


def compute_ptms(
    logits: torch.Tensor,
    x_preds: torch.Tensor,
    feats: dict[str, torch.Tensor],
    multiplicity: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[int, dict[int, torch.Tensor]],
]:
    """Compute pTM, ipTM, interface-type, and chain-pair confidence scores."""

    _, frame_mask = compute_frame_pred(
        x_preds,
        feats["frames_idx"],
        feats,
        multiplicity,
        inference=True,
    )
    token_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
    valid_anchor = frame_mask.reshape(-1, frame_mask.shape[-1])
    base_pair_mask = valid_anchor[:, :, None] * token_mask[:, None, :] * token_mask[:, :, None]
    asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    interface_mask = base_pair_mask * (asym_id[:, None, :] != asym_id[:, :, None])

    num_bins = logits.shape[-1]
    pae_centers = torch.arange(
        0.5 * (32.0 / num_bins),
        32.0,
        32.0 / num_bins,
        device=logits.device,
    ).unsqueeze(0)
    n_res = token_mask.sum(dim=-1, keepdim=True)
    tm_values = tm_function(pae_centers, n_res).unsqueeze(1).unsqueeze(2)
    expected_tm = torch.sum(nn.functional.softmax(logits, dim=-1) * tm_values, dim=-1)

    ptm = _maximum_masked_tm(expected_tm, base_pair_mask)
    iptm = _maximum_masked_tm(expected_tm, interface_mask)

    token_type = feats["mol_type"].repeat_interleave(multiplicity, 0)
    ligand = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    protein = (token_type == const.chain_type_ids["PROTEIN"]).float()
    ligand_protein = (
        ligand[:, :, None] * protein[:, None, :] + protein[:, :, None] * ligand[:, None, :]
    )
    protein_protein = protein[:, :, None] * protein[:, None, :]
    ligand_iptm = _maximum_masked_tm(expected_tm, interface_mask * ligand_protein)
    protein_iptm = _maximum_masked_tm(expected_tm, interface_mask * protein_protein)

    chain_pair_iptm: dict[int, dict[int, torch.Tensor]] = {}
    for first_chain in torch.unique(asym_id).tolist():
        scores: dict[int, torch.Tensor] = {}
        for second_chain in torch.unique(asym_id).tolist():
            chain_mask = base_pair_mask
            chain_mask = chain_mask * (asym_id[:, None, :] == first_chain)
            chain_mask = chain_mask * (asym_id[:, :, None] == second_chain)
            scores[second_chain] = _maximum_masked_tm(expected_tm, chain_mask)
        chain_pair_iptm[first_chain] = scores

    return ptm, iptm, ligand_iptm, protein_iptm, chain_pair_iptm
