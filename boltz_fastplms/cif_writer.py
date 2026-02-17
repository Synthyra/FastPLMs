from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .minimal_structures import ProteinStructureTemplate


def _confidence_per_atom(
    plddt: Optional[torch.Tensor],
    atom_to_residue: list[int],
    num_atoms: int,
    sample_index: int,
) -> np.ndarray:
    if plddt is None:
        return np.ones((num_atoms,), dtype=np.float32) * 100.0

    values = plddt.detach().cpu()
    if values.ndim == 1:
        values = values.unsqueeze(0)
    assert values.ndim == 2, "Expected pLDDT with shape [samples, tokens/atoms]."
    assert sample_index < values.shape[0], "sample_index out of range for pLDDT."

    selected = values[sample_index]
    if selected.shape[0] == num_atoms:
        return (selected.numpy() * 100.0).astype(np.float32)

    num_residues = max(atom_to_residue) + 1
    if selected.shape[0] == num_residues:
        expanded = np.zeros((num_atoms,), dtype=np.float32)
        selected_np = selected.numpy()
        for atom_idx, residue_idx in enumerate(atom_to_residue):
            expanded[atom_idx] = selected_np[residue_idx] * 100.0
        return expanded

    return np.ones((num_atoms,), dtype=np.float32) * 100.0


def write_cif(
    structure_template: ProteinStructureTemplate,
    atom_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    output_path: str,
    plddt: Optional[torch.Tensor] = None,
    sample_index: int = 0,
) -> str:
    coords = atom_coords.detach().cpu()
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
    assert coords.ndim == 3, "Expected coordinates with shape [samples, atoms, 3]."
    assert sample_index < coords.shape[0], "sample_index out of range."
    selected_coords_tensor = coords[sample_index]
    all_non_finite = torch.logical_not(torch.isfinite(selected_coords_tensor))
    assert not torch.any(all_non_finite), (
        "CIF export received non-finite coordinates. "
        f"Non-finite count: {int(all_non_finite.sum().item())}"
    )
    selected_coords = selected_coords_tensor.numpy()

    mask = atom_mask.detach().cpu()
    if mask.ndim == 2:
        mask = mask[0]
    assert mask.ndim == 1, "Expected atom mask with shape [atoms]."
    assert mask.shape[0] == selected_coords.shape[0], "Atom mask/coord size mismatch."
    assert torch.any(mask > 0), "Atom mask has no valid atoms for CIF export."
    valid_non_finite = torch.logical_not(torch.isfinite(selected_coords_tensor[mask > 0]))
    assert not torch.any(valid_non_finite), (
        "CIF export has non-finite coordinates in unmasked atoms. "
        f"Non-finite count: {int(valid_non_finite.sum().item())}"
    )

    b_iso = _confidence_per_atom(
        plddt=plddt,
        atom_to_residue=structure_template.atom_residue_index,
        num_atoms=structure_template.num_atoms,
        sample_index=sample_index,
    )
    assert b_iso.shape[0] == structure_template.num_atoms

    lines = [
        "data_boltz2_prediction",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_PDB_model_num",
    ]

    atom_id = 1
    for idx in range(structure_template.num_atoms):
        if mask[idx] <= 0:
            continue

        residue_idx = structure_template.atom_residue_index[idx]
        residue_name = structure_template.residue_names[residue_idx]
        atom_name = structure_template.atom_names[idx]
        element = structure_template.atom_elements[idx]
        chain_id = structure_template.atom_chain_id[idx]
        x_val, y_val, z_val = selected_coords[idx].tolist()
        b_factor = float(b_iso[idx])

        line = (
            f"ATOM {atom_id} {element} {atom_name} {residue_name} {chain_id} "
            f"{residue_idx + 1} {x_val:.3f} {y_val:.3f} {z_val:.3f} 1.00 {b_factor:.2f} 1"
        )
        lines.append(line)
        atom_id += 1

    lines.append("#")
    text = "\n".join(lines) + "\n"

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return str(out_path)
