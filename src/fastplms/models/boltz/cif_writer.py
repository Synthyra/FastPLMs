import numpy as np
import torch
from pathlib import Path

from .minimal_structures import ProteinStructureTemplate


def _confidence_per_atom(
    plddt: torch.Tensor | None,
    atom_to_residue: list[int],
    num_atoms: int,
    sample_index: int,
) -> np.ndarray:
    # plddt: (n_i,) or (n_s, n_i); atom_to_residue: length n_a; num_atoms: n_a
    if plddt is None:
        return np.ones((num_atoms,), dtype=np.float32) * 100.0  # (n_a,)

    values = plddt.detach().cpu()  # (n_i,) or (n_s, n_i)
    if values.ndim == 1:
        values = values.unsqueeze(0)  # (1, n_i)
    if values.ndim != 2:
        raise ValueError("Expected pLDDT tensor S with shape (n_samples, n_items).")
    # values: (n_s, n_i) beyond this point
    if not 0 <= sample_index < values.shape[0]:
        raise IndexError("sample_index out of range for pLDDT.")

    selected = values[sample_index]  # (n_i,)
    if selected.shape[0] == num_atoms:
        return (selected.numpy() * 100.0).astype(np.float32)  # (n_a,)

    num_residues = max(atom_to_residue) + 1  # n_r
    if selected.shape[0] == num_residues:
        expanded = np.zeros((num_atoms,), dtype=np.float32)  # (n_a,)
        selected_np = selected.numpy()  # (n_r,)
        for atom_idx, residue_idx in enumerate(atom_to_residue):
            expanded[atom_idx] = selected_np[residue_idx] * 100.0  # () -> ()
        return expanded  # (n_a,)

    raise ValueError(
        "pLDDT item count must match either atoms or residues: "
        f"received {selected.shape[0]}, expected {num_atoms} atoms or "
        f"{num_residues} residues."
    )


def write_cif(
    structure_template: ProteinStructureTemplate,
    atom_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    output_path: str,
    plddt: torch.Tensor | None = None,
    sample_index: int = 0,
) -> str:
    # atom_coords: (n_a, 3) or (n_s, n_a, 3)
    # atom_mask: (n_a,) or (n_m, n_a); plddt: (n_i,) or (n_s, n_i)
    coords = atom_coords.detach().cpu()  # (n_a, 3) or (n_s, n_a, 3)
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)  # (1, n_a, 3)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(
            "Expected coordinate tensor X with shape (n_samples, n_atoms, 3)."
        )
    # coords: (n_s, n_a, 3) beyond this point
    if not 0 <= sample_index < coords.shape[0]:
        raise IndexError("sample_index out of range.")
    selected_coords_tensor = coords[sample_index]  # (n_a, 3)
    all_non_finite = torch.logical_not(torch.isfinite(selected_coords_tensor))  # (n_a, 3)
    if torch.any(all_non_finite):
        raise ValueError(
            "CIF export received non-finite coordinates. "
            f"Non-finite count: {int(all_non_finite.sum().item())}"
        )
    selected_coords = selected_coords_tensor.numpy()  # (n_a, 3)

    mask = atom_mask.detach().cpu()  # (n_a,) or (n_m, n_a)
    if mask.ndim == 2:
        mask = mask[0]  # (n_a,)
    if mask.ndim != 1:
        raise ValueError("Expected atom mask M with shape (n_atoms,).")
    # mask: (n_a,) beyond this point
    if mask.shape[0] != selected_coords.shape[0]:
        raise ValueError("Atom mask/coord size mismatch.")
    if not torch.any(mask > 0):
        raise ValueError("Atom mask has no valid atoms for CIF export.")
    valid_non_finite = torch.logical_not(torch.isfinite(selected_coords_tensor[mask > 0]))
    # valid_non_finite: (n_v, 3), where n_v is the valid-atom count
    if torch.any(valid_non_finite):
        raise ValueError(
            "CIF export has non-finite coordinates in unmasked atoms. "
            f"Non-finite count: {int(valid_non_finite.sum().item())}"
        )

    b_iso = _confidence_per_atom(
        plddt=plddt,
        atom_to_residue=structure_template.atom_residue_index,
        num_atoms=structure_template.num_atoms,
        sample_index=sample_index,
    )  # (n_a,)
    if b_iso.shape[0] != structure_template.num_atoms:
        raise RuntimeError("CIF confidence values do not match the structure atom count.")

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
        x_val, y_val, z_val = selected_coords[idx].tolist()  # (3,) -> three scalars
        b_factor = float(b_iso[idx])  # () -> Python float

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
