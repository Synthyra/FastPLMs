import math
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.functional import one_hot

from .minimal_structures import ProteinStructureTemplate
from .vendored_boltz.boltz.data import const


_ELEMENT_TO_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "P": 15,
    "S": 16,
}


def _normalize_sequence(sequence: str) -> str:
    seq = sequence.strip().upper()
    assert len(seq) > 0, "Amino acid sequence must be non-empty."
    for aa in seq:
        assert aa in const.prot_letter_to_token, f"Unsupported residue code '{aa}'."
    return seq


def _atom_name_to_element(atom_name: str) -> str:
    name = atom_name.strip().upper()
    if len(name) == 0:
        return "C"
    if name[0].isdigit():
        name = name[1:]
    if len(name) >= 2 and name[0:2] in ("CL", "BR", "FE", "MG", "ZN", "NA", "CA"):
        return name[0]
    return name[0]


def _atom_name_to_codes(atom_name: str) -> torch.Tensor:
    clipped = atom_name.strip()[:4]
    vals = [ord(ch) - 32 for ch in clipped]
    while len(vals) < 4:
        vals.append(0)
    out = torch.tensor(vals, dtype=torch.long)
    assert torch.all(out >= 0) and torch.all(out < 64), (
        f"Invalid atom-name encoding for '{atom_name}'."
    )
    return out


def _default_atom_offset(atom_name: str, atom_idx: int) -> np.ndarray:
    if atom_name == "N":
        return np.array([-1.20, 0.80, 0.00], dtype=np.float32)
    if atom_name == "CA":
        return np.array([0.00, 0.00, 0.00], dtype=np.float32)
    if atom_name == "C":
        return np.array([1.35, 0.65, 0.00], dtype=np.float32)
    if atom_name == "O":
        return np.array([2.25, 0.10, 0.00], dtype=np.float32)
    if atom_name == "CB":
        return np.array([0.15, -1.20, 1.00], dtype=np.float32)
    angle = (atom_idx + 1) * 0.7
    radius = 1.4 + 0.03 * atom_idx
    return np.array(
        [
            radius * math.cos(angle),
            radius * math.sin(angle),
            0.1 * ((atom_idx % 5) - 2),
        ],
        dtype=np.float32,
    )


def _build_template(
    sequence: str,
) -> Tuple[
    ProteinStructureTemplate,
    List[str],
    List[int],
    List[int],
    List[int],
    List[np.ndarray],
    List[int],
]:
    residue_names: List[str] = []
    residue_token_ids: List[int] = []
    atom_names: List[str] = []
    atom_elements: List[str] = []
    atom_residue_index: List[int] = []
    atom_chain_id: List[str] = []
    atom_positions: List[np.ndarray] = []
    residue_center_atom_idx: List[int] = []
    residue_disto_atom_idx: List[int] = []
    residue_frame_atom_idx: List[int] = []

    global_atom_idx = 0
    for res_idx, aa in enumerate(sequence):
        token_name = const.prot_letter_to_token[aa]
        residue_names.append(token_name)
        residue_token_ids.append(const.token_ids[token_name])

        residue_atoms = const.ref_atoms[token_name]
        assert len(residue_atoms) > 0, f"No reference atoms for residue {token_name}."
        center_atom_name = const.res_to_center_atom[token_name]
        disto_atom_name = const.res_to_disto_atom[token_name]

        base = np.array([3.8 * res_idx, 0.0, 0.0], dtype=np.float32)
        center_idx = -1
        disto_idx = -1
        n_idx = -1
        ca_idx = -1
        c_idx = -1

        for local_idx, atom_name in enumerate(residue_atoms):
            atom_names.append(atom_name)
            element = _atom_name_to_element(atom_name)
            atom_elements.append(element)
            atom_residue_index.append(res_idx)
            atom_chain_id.append("A")

            atom_pos = base + _default_atom_offset(atom_name, local_idx)
            atom_positions.append(atom_pos)

            if atom_name == center_atom_name:
                center_idx = global_atom_idx
            if atom_name == disto_atom_name:
                disto_idx = global_atom_idx
            if atom_name == "N":
                n_idx = global_atom_idx
            if atom_name == "CA":
                ca_idx = global_atom_idx
            if atom_name == "C":
                c_idx = global_atom_idx
            global_atom_idx += 1

        if center_idx == -1:
            center_idx = global_atom_idx - len(residue_atoms)
        if disto_idx == -1:
            disto_idx = center_idx
        if n_idx == -1:
            n_idx = center_idx
        if ca_idx == -1:
            ca_idx = center_idx
        if c_idx == -1:
            c_idx = center_idx

        residue_center_atom_idx.append(center_idx)
        residue_disto_atom_idx.append(disto_idx)
        residue_frame_atom_idx.extend([n_idx, ca_idx, c_idx])

    template = ProteinStructureTemplate(
        sequence=sequence,
        residue_names=residue_names,
        atom_names=atom_names,
        atom_elements=atom_elements,
        atom_residue_index=atom_residue_index,
        atom_chain_id=atom_chain_id,
    )

    return (
        template,
        residue_names,
        residue_token_ids,
        residue_center_atom_idx,
        residue_disto_atom_idx,
        atom_positions,
        residue_frame_atom_idx,
    )


def build_boltz2_features(
    amino_acid_sequence: str,
    num_bins: int = 64,
    atoms_per_window_queries: int = 32,
) -> Tuple[Dict[str, torch.Tensor], ProteinStructureTemplate]:
    sequence = _normalize_sequence(amino_acid_sequence)
    (
        template,
        residue_names,
        residue_token_ids,
        residue_center_atom_idx,
        residue_disto_atom_idx,
        atom_positions_np,
        residue_frame_atom_idx_flat,
    ) = _build_template(sequence)

    num_tokens = len(residue_names)
    num_atoms = len(atom_positions_np)
    assert num_tokens > 0 and num_atoms > 0

    atom_positions = torch.tensor(np.asarray(atom_positions_np), dtype=torch.float32)
    atom_positions = atom_positions - atom_positions.mean(dim=0, keepdim=True)

    token_index = torch.arange(num_tokens, dtype=torch.long).unsqueeze(0)
    residue_index = torch.arange(num_tokens, dtype=torch.long).unsqueeze(0)
    asym_id = torch.zeros((1, num_tokens), dtype=torch.long)
    entity_id = torch.zeros((1, num_tokens), dtype=torch.long)
    sym_id = torch.zeros((1, num_tokens), dtype=torch.long)
    mol_type = torch.full(
        (1, num_tokens),
        fill_value=const.chain_type_ids["PROTEIN"],
        dtype=torch.long,
    )

    res_type_ids = torch.tensor(residue_token_ids, dtype=torch.long)
    res_type = one_hot(res_type_ids, num_classes=const.num_tokens).float().unsqueeze(0)

    token_bonds = torch.zeros((num_tokens, num_tokens), dtype=torch.float32)
    type_bonds = torch.zeros((num_tokens, num_tokens), dtype=torch.long)
    if "COVALENT" in const.bond_type_ids:
        covalent_bond_id = const.bond_type_ids["COVALENT"]
    else:
        covalent_bond_id = 1
    for idx in range(num_tokens - 1):
        token_bonds[idx, idx + 1] = 1.0
        token_bonds[idx + 1, idx] = 1.0
        type_bonds[idx, idx + 1] = covalent_bond_id
        type_bonds[idx + 1, idx] = covalent_bond_id
    token_bonds = token_bonds.unsqueeze(0).unsqueeze(-1)
    type_bonds = type_bonds.unsqueeze(0)

    token_pad_mask = torch.ones((1, num_tokens), dtype=torch.float32)
    token_resolved_mask = torch.ones((1, num_tokens), dtype=torch.float32)
    token_disto_mask = torch.ones((1, num_tokens), dtype=torch.float32)

    num_contact_classes = len(const.contact_conditioning_info)
    unspecified_id = const.contact_conditioning_info["UNSPECIFIED"]
    contact_ids = torch.full(
        (num_tokens, num_tokens),
        fill_value=unspecified_id,
        dtype=torch.long,
    )
    contact_conditioning = one_hot(
        contact_ids,
        num_classes=num_contact_classes,
    ).float().unsqueeze(0)
    contact_threshold = torch.zeros((1, num_tokens, num_tokens), dtype=torch.float32)

    assert "x-ray diffraction" in const.method_types_ids
    method_feature = torch.full(
        (1, num_tokens),
        fill_value=const.method_types_ids["x-ray diffraction"],
        dtype=torch.long,
    )
    modified = torch.zeros((1, num_tokens), dtype=torch.long)
    cyclic_period = torch.zeros((1, num_tokens), dtype=torch.float32)
    affinity_token_mask = torch.zeros((1, num_tokens), dtype=torch.float32)

    ref_pos = atom_positions.unsqueeze(0)
    atom_pad_mask = torch.ones((1, num_atoms), dtype=torch.float32)
    atom_resolved_mask = torch.ones((1, num_atoms), dtype=torch.float32)

    atom_name_codes = torch.stack(
        [_atom_name_to_codes(atom_name) for atom_name in template.atom_names],
        dim=0,
    )
    ref_atom_name_chars = one_hot(atom_name_codes, num_classes=64).float().unsqueeze(0)

    atomic_numbers = []
    for element in template.atom_elements:
        if element in _ELEMENT_TO_Z:
            z_value = _ELEMENT_TO_Z[element]
        else:
            z_value = _ELEMENT_TO_Z["C"]
        assert z_value < const.num_elements
        atomic_numbers.append(z_value)
    ref_element = one_hot(
        torch.tensor(atomic_numbers, dtype=torch.long),
        num_classes=const.num_elements,
    ).float().unsqueeze(0)

    ref_charge = torch.zeros((1, num_atoms), dtype=torch.float32)
    ref_chirality = torch.zeros((1, num_atoms), dtype=torch.long)
    ref_space_uid = torch.tensor(template.atom_residue_index, dtype=torch.long).unsqueeze(0)

    atom_to_token = one_hot(
        torch.tensor(template.atom_residue_index, dtype=torch.long),
        num_classes=num_tokens,
    ).float().unsqueeze(0)
    token_to_rep_atom = one_hot(
        torch.tensor(residue_disto_atom_idx, dtype=torch.long),
        num_classes=num_atoms,
    ).float().unsqueeze(0)
    token_to_center_atom = one_hot(
        torch.tensor(residue_center_atom_idx, dtype=torch.long),
        num_classes=num_atoms,
    ).float().unsqueeze(0)
    r_set_to_rep_atom = token_to_center_atom.clone()

    num_backbone_classes = (
        1
        + len(const.protein_backbone_atom_index)
        + len(const.nucleic_backbone_atom_index)
    )
    backbone_ids = []
    for atom_name in template.atom_names:
        if atom_name in const.protein_backbone_atom_index:
            backbone_ids.append(const.protein_backbone_atom_index[atom_name] + 1)
        else:
            backbone_ids.append(0)
    atom_backbone_feat = one_hot(
        torch.tensor(backbone_ids, dtype=torch.long),
        num_classes=num_backbone_classes,
    ).float().unsqueeze(0)

    coords = ref_pos.unsqueeze(1).contiguous()
    disto_coords = torch.stack(
        [atom_positions[idx] for idx in residue_disto_atom_idx],
        dim=0,
    )
    disto_coords_ensemble = disto_coords.unsqueeze(0).unsqueeze(0).contiguous()

    bfactor = torch.zeros((1, num_atoms), dtype=torch.float32)
    atom_plddt = torch.ones((1, num_atoms), dtype=torch.float32)

    assert atoms_per_window_queries > 0
    pad_atoms = (
        ((num_atoms - 1) // atoms_per_window_queries + 1) * atoms_per_window_queries
        - num_atoms
    )
    if pad_atoms > 0:
        ref_pos = torch.nn.functional.pad(ref_pos, (0, 0, 0, pad_atoms), value=0.0)
        atom_pad_mask = torch.nn.functional.pad(atom_pad_mask, (0, pad_atoms), value=0.0)
        atom_resolved_mask = torch.nn.functional.pad(
            atom_resolved_mask,
            (0, pad_atoms),
            value=0.0,
        )
        ref_atom_name_chars = torch.nn.functional.pad(
            ref_atom_name_chars,
            (0, 0, 0, 0, 0, pad_atoms),
            value=0.0,
        )
        ref_element = torch.nn.functional.pad(ref_element, (0, 0, 0, pad_atoms), value=0.0)
        ref_charge = torch.nn.functional.pad(ref_charge, (0, pad_atoms), value=0.0)
        ref_chirality = torch.nn.functional.pad(ref_chirality, (0, pad_atoms), value=0)
        atom_backbone_feat = torch.nn.functional.pad(
            atom_backbone_feat,
            (0, 0, 0, pad_atoms),
            value=0.0,
        )
        ref_space_uid = torch.nn.functional.pad(ref_space_uid, (0, pad_atoms), value=0)
        coords = torch.nn.functional.pad(coords, (0, 0, 0, pad_atoms), value=0.0)
        atom_to_token = torch.nn.functional.pad(atom_to_token, (0, 0, 0, pad_atoms), value=0.0)
        token_to_rep_atom = torch.nn.functional.pad(
            token_to_rep_atom,
            (0, pad_atoms),
            value=0.0,
        )
        token_to_center_atom = torch.nn.functional.pad(
            token_to_center_atom,
            (0, pad_atoms),
            value=0.0,
        )
        r_set_to_rep_atom = torch.nn.functional.pad(
            r_set_to_rep_atom,
            (0, pad_atoms),
            value=0.0,
        )
        bfactor = torch.nn.functional.pad(bfactor, (0, pad_atoms), value=0.0)
        atom_plddt = torch.nn.functional.pad(atom_plddt, (0, pad_atoms), value=0.0)

    frames_idx = torch.tensor(
        residue_frame_atom_idx_flat,
        dtype=torch.long,
    ).reshape(num_tokens, 3)
    frames_idx = frames_idx.unsqueeze(0).unsqueeze(1)
    frame_resolved_mask = torch.ones((1, 1, num_tokens), dtype=torch.float32)

    msa = torch.tensor(residue_token_ids, dtype=torch.long).unsqueeze(0).unsqueeze(0)
    msa_paired = torch.ones((1, 1, num_tokens), dtype=torch.float32)
    deletion_value = torch.zeros((1, 1, num_tokens), dtype=torch.float32)
    has_deletion = torch.zeros((1, 1, num_tokens), dtype=torch.float32)
    msa_mask = torch.ones((1, 1, num_tokens), dtype=torch.float32)
    deletion_mean = torch.zeros((1, num_tokens), dtype=torch.float32)
    profile = one_hot(
        torch.tensor(residue_token_ids, dtype=torch.long),
        num_classes=const.num_tokens,
    ).float().unsqueeze(0)

    template_restype = one_hot(
        torch.zeros((1, 1, num_tokens), dtype=torch.long),
        num_classes=const.num_tokens,
    ).float()
    template_frame_rot = torch.zeros((1, 1, num_tokens, 3, 3), dtype=torch.float32)
    template_frame_t = torch.zeros((1, 1, num_tokens, 3), dtype=torch.float32)
    template_cb = torch.zeros((1, 1, num_tokens, 3), dtype=torch.float32)
    template_ca = torch.zeros((1, 1, num_tokens, 3), dtype=torch.float32)
    template_mask_cb = torch.zeros((1, 1, num_tokens), dtype=torch.float32)
    template_mask_frame = torch.zeros((1, 1, num_tokens), dtype=torch.float32)
    template_mask = torch.zeros((1, 1, num_tokens), dtype=torch.float32)
    query_to_template = torch.zeros((1, 1, num_tokens), dtype=torch.long)
    visibility_ids = torch.zeros((1, 1, num_tokens), dtype=torch.float32)

    disto_target = torch.zeros(
        (1, num_tokens, num_tokens, 1, num_bins),
        dtype=torch.float32,
    )
    disto_center = torch.stack(
        [atom_positions[idx] for idx in residue_disto_atom_idx],
        dim=0,
    ).unsqueeze(0)

    features: Dict[str, torch.Tensor] = {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "disto_center": disto_center,
        "token_bonds": token_bonds,
        "type_bonds": type_bonds,
        "token_pad_mask": token_pad_mask,
        "token_resolved_mask": token_resolved_mask,
        "token_disto_mask": token_disto_mask,
        "contact_conditioning": contact_conditioning,
        "contact_threshold": contact_threshold,
        "method_feature": method_feature,
        "modified": modified,
        "cyclic_period": cyclic_period,
        "affinity_token_mask": affinity_token_mask,
        "ref_pos": ref_pos,
        "atom_resolved_mask": atom_resolved_mask,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_chirality": ref_chirality,
        "atom_backbone_feat": atom_backbone_feat,
        "ref_space_uid": ref_space_uid,
        "coords": coords,
        "atom_pad_mask": atom_pad_mask,
        "atom_to_token": atom_to_token,
        "token_to_rep_atom": token_to_rep_atom,
        "r_set_to_rep_atom": r_set_to_rep_atom,
        "token_to_center_atom": token_to_center_atom,
        "disto_target": disto_target,
        "disto_coords_ensemble": disto_coords_ensemble,
        "bfactor": bfactor,
        "plddt": atom_plddt,
        "frames_idx": frames_idx,
        "frame_resolved_mask": frame_resolved_mask,
        "msa": msa,
        "msa_paired": msa_paired,
        "deletion_value": deletion_value,
        "has_deletion": has_deletion,
        "deletion_mean": deletion_mean,
        "profile": profile,
        "msa_mask": msa_mask,
        "template_restype": template_restype,
        "template_frame_rot": template_frame_rot,
        "template_frame_t": template_frame_t,
        "template_cb": template_cb,
        "template_ca": template_ca,
        "template_mask_cb": template_mask_cb,
        "template_mask_frame": template_mask_frame,
        "template_mask": template_mask,
        "query_to_template": query_to_template,
        "visibility_ids": visibility_ids,
    }

    return features, template
