import math
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.functional import one_hot

from .minimal_structures import ProteinStructureTemplate
from . import vb_const as const


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


# Canonical RDKit conformer positions extracted from official Boltz2 mol files
# (boltz-community/boltz-2 mols.tar, first conformer, centered per residue).
# These match the geometry the model was trained on.
_RDKIT_CONFORMERS: dict[str, dict[str, list[float]]] = {
    "ALA": {"N": [-0.944785, 0.952743, 0.876326], "CA": [-0.287002, -0.317661, 0.564429], "C": [1.098243, -0.090637, 0.019913], "O": [1.267603, 0.576468, -1.036548], "CB": [-1.13406, -1.120912, -0.42412]},
    "ARG": {"N": [3.318334, -1.792721, -0.81714], "CA": [2.256519, -0.785164, -0.782524], "C": [2.865221, 0.579152, -0.9479], "O": [2.530447, 1.305604, -1.921779], "CB": [1.446111, -0.888115, 0.531287], "CG": [0.26845, 0.099716, 0.625405], "CD": [-0.837403, -0.192693, -0.39904], "NE": [-2.021791, 0.62151, -0.123045], "CZ": [-2.953151, 0.387273, 0.951242], "NH1": [-4.01513, 1.322306, 1.16201], "NH2": [-2.857608, -0.656868, 1.721485]},
    "ASN": {"N": [-1.76737, -1.671462, 0.274097], "CA": [-0.894725, -0.51088, 0.459045], "C": [-1.437261, 0.658619, -0.316984], "O": [-1.745491, 0.527762, -1.53251], "CB": [0.535815, -0.851978, 0.002785], "CG": [1.496584, 0.259422, 0.3155], "OD1": [2.006304, 0.340489, 1.464867], "ND2": [1.806145, 1.248028, -0.666799]},
    "ASP": {"N": [-0.531828, -1.5551, 0.564551], "CA": [-0.798781, -0.116987, 0.471405], "C": [-2.220416, 0.13533, 0.042978], "O": [-2.637844, -0.292258, -1.067311], "CB": [0.18409, 0.554469, -0.501854], "CG": [1.594766, 0.436104, -0.012856], "OD1": [2.333189, -0.495262, -0.431504], "OD2": [2.076824, 1.333703, 0.934591]},
    "CYS": {"N": [0.008485, 1.680076, -0.119503], "CA": [0.001723, 0.385667, -0.808584], "C": [-1.206733, -0.441379, -0.446529], "O": [-1.665274, -0.434109, 0.728016], "CB": [1.310427, -0.380146, -0.555459], "SG": [1.551372, -0.810109, 1.202058]},
    "GLN": {"N": [-1.932297, -1.099026, -1.887635], "CA": [-1.370158, -0.7752, -0.575112], "C": [-2.304423, 0.155162, 0.147051], "O": [-2.803483, -0.184897, 1.253191], "CB": [0.038652, -0.158687, -0.716379], "CG": [0.736106, 0.008065, 0.640487], "CD": [2.117149, 0.560394, 0.455758], "OE1": [2.309056, 1.803816, 0.522783], "NE2": [3.209399, -0.309626, 0.159856]},
    "GLU": {"N": [-1.750645, -1.298566, -1.148627], "CA": [-1.677927, -0.690629, 0.182405], "C": [-2.320926, 0.670409, 0.183131], "O": [-2.310494, 1.385902, -0.855225], "CB": [-0.218639, -0.616385, 0.677413], "CG": [0.704486, 0.181646, -0.256505], "CD": [2.107661, 0.167064, 0.261748], "OE1": [2.495054, 1.060279, 1.061799], "OE2": [2.97143, -0.85972, -0.106138]},
    "GLY": {"N": [-1.416855, 0.862616, -0.1801], "CA": [-0.6149, -0.033688, 0.644675], "C": [0.809702, 0.000186, 0.194889], "O": [1.222053, -0.829114, -0.659464]},
    "HIS": {"N": [1.26543, -1.579115, 0.588736], "CA": [1.426323, -0.419409, -0.29228], "C": [2.884431, -0.088226, -0.469953], "O": [3.407859, -0.14071, -1.615113], "CB": [0.668818, 0.794698, 0.269345], "CG": [-0.807376, 0.534987, 0.323389], "ND1": [-1.680612, 0.632102, -0.797403], "CD2": [-1.481273, 0.09683, 1.377023], "CE1": [-2.846859, 0.264305, -0.380313], "NE2": [-2.836741, -0.095463, 0.996569]},
    "ILE": {"N": [-0.963969, -1.670572, 0.035929], "CA": [-1.004097, -0.255306, 0.424676], "C": [-1.714092, 0.593857, -0.603418], "O": [-1.755267, 0.246984, -1.815103], "CB": [0.416162, 0.290836, 0.741822], "CG1": [1.380111, 0.189293, -0.469113], "CG2": [0.995443, -0.424464, 1.975358], "CD1": [2.645709, 1.029371, -0.290151]},
    "LEU": {"N": [1.265833, -0.791579, -1.184426], "CA": [0.933538, 0.306792, -0.27234], "C": [2.051763, 0.509151, 0.713934], "O": [2.543725, -0.475272, 1.329405], "CB": [-0.373342, 0.011933, 0.498041], "CG": [-1.632816, -0.15124, -0.388179], "CD1": [-2.835081, -0.534398, 0.48526], "CD2": [-1.953621, 1.124614, -1.181694]},
    "LYS": {"N": [-1.908918, -1.413217, -1.088748], "CA": [-1.724515, -0.558426, 0.087009], "C": [-2.96641, 0.258056, 0.315903], "O": [-3.48391, 0.312278, 1.463711], "CB": [-0.523791, 0.388858, -0.107149], "CG": [0.821393, -0.35366, -0.108072], "CD": [1.993822, 0.633537, -0.161446], "CE": [3.338346, -0.105923, -0.176016], "NZ": [4.453983, 0.838498, -0.225193]},
    "MET": {"N": [-1.522666, -0.831762, 1.959218], "CA": [-0.996081, 0.012809, 0.885437], "C": [-2.091048, 0.292064, -0.106017], "O": [-2.445252, 1.479197, -0.337323], "CB": [0.220238, -0.653951, 0.206767], "CG": [0.915991, 0.285443, -0.785597], "SD": [2.357592, -0.540499, -1.548879], "CE": [3.561227, -0.043301, -0.273607]},
    "PHE": {"N": [3.046798, -1.689795, -0.208668], "CA": [1.829154, -0.89776, -0.391539], "C": [2.202709, 0.507016, -0.775299], "O": [1.770264, 1.003795, -1.849745], "CB": [0.962876, -0.916233, 0.885596], "CG": [-0.373751, -0.255463, 0.660531], "CD1": [-0.576773, 1.029007, 1.007455], "CD2": [-1.46686, -1.011364, -0.000788], "CE1": [-1.882158, 1.672384, 0.749291], "CE2": [-2.646851, -0.423965, -0.236002], "CZ": [-2.865409, 0.982376, 0.159168]},
    "PRO": {"N": [-0.685006, -0.370164, -0.768919], "CA": [0.404323, 0.377887, -0.13805], "C": [1.731651, -0.324892, -0.273548], "O": [1.975993, -1.024105, -1.293633], "CB": [-0.004718, 0.595698, 1.311196], "CG": [-1.517474, 0.663762, 1.260234], "CD": [-1.904769, 0.081814, -0.097281]},
    "SER": {"N": [1.015962, -1.698341, -0.119567], "CA": [0.101187, -0.56256, -0.236622], "C": [0.88759, 0.717914, -0.271742], "O": [0.652684, 1.578473, -1.161893], "CB": [-0.87741, -0.552018, 0.948557], "OG": [-1.780012, 0.516532, 0.841267]},
    "THR": {"N": [-0.05857, 1.577455, 0.452633], "CA": [0.359545, 0.187419, 0.662184], "C": [1.654837, -0.077979, -0.057809], "O": [1.861664, 0.408605, -1.20258], "CB": [-0.727051, -0.803664, 0.176961], "OG1": [-1.139906, -0.486103, -1.128492], "CG2": [-1.950518, -0.805733, 1.097103]},
    "TRP": {"N": [-3.22168, 1.119624, 0.055731], "CA": [-2.641395, 0.081435, -0.801364], "C": [-2.364426, -1.166119, -0.006875], "O": [-2.514206, -2.296758, -0.543038], "CB": [-1.384007, 0.576331, -1.552364], "CG": [-0.24222, 0.940856, -0.643481], "CD1": [-0.016615, 2.14194, -0.107979], "CD2": [0.800451, 0.051232, -0.1373], "NE1": [1.132969, 2.107427, 0.738111], "CE2": [1.575163, 0.761216, 0.659678], "CE3": [1.058378, -1.378818, -0.38353], "CZ2": [2.729299, 0.158745, 1.347069], "CZ3": [2.107441, -1.950409, 0.235009], "CH2": [2.980848, -1.146704, 1.140333]},
    "TYR": {"N": [-2.152606, 0.331726, -1.283488], "CA": [-2.201241, -0.428961, -0.031756], "C": [-3.597879, -0.441616, 0.531374], "O": [-4.172602, 0.638246, 0.83694], "CB": [-1.21174, 0.150443, 0.994648], "CG": [0.209673, 0.053123, 0.503276], "CD1": [0.911845, -1.084226, 0.654794], "CD2": [0.82966, 1.201129, -0.203917], "CE1": [2.294408, -1.180811, 0.141538], "CE2": [2.080826, 1.10993, -0.671525], "CZ": [2.85374, -0.136912, -0.490516], "OH": [4.155917, -0.21207, -0.981367]},
    "VAL": {"N": [0.631715, -1.334948, 0.719228], "CA": [0.419681, -0.467832, -0.445221], "C": [1.457942, 0.623882, -0.483986], "O": [1.935858, 1.0943, 0.583992], "CB": [-1.010701, 0.140743, -0.455789], "CG1": [-2.075373, -0.919582, -0.774314], "CG2": [-1.359123, 0.863438, 0.85609]},
    "UNK": {"N": [0.8287, -1.182096, -0.645721], "CA": [-0.174671, -0.11586, -0.67851], "C": [0.301419, 1.045573, 0.149013], "O": [0.589973, 0.885289, 1.365851], "CB": [-1.545421, -0.632906, -0.190632]},
}


def _get_atom_position(res_name: str, atom_name: str, atom_idx: int) -> np.ndarray:
    """Get the canonical RDKit conformer position for an atom.

    Uses pre-extracted positions from official Boltz2 mol files. Falls back
    to a simple geometric placement for unknown residue/atom combinations.
    """
    if res_name in _RDKIT_CONFORMERS and atom_name in _RDKIT_CONFORMERS[res_name]:
        return np.array(_RDKIT_CONFORMERS[res_name][atom_name], dtype=np.float32)
    # Fallback for unknown atoms (should not happen for canonical AAs)
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

            atom_pos = _get_atom_position(token_name, atom_name, local_idx)
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


def _random_rotation_matrix() -> torch.Tensor:
    """Sample a uniform random 3x3 rotation matrix (Algorithm 19 from AF2/Boltz)."""
    q = torch.randn(4)
    q = q / q.norm()
    # Quaternion to rotation matrix
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=torch.float32)


def _center_and_augment_atoms_per_residue(
    atom_positions: torch.Tensor,
    atom_residue_index: List[int],
    num_residues: int,
) -> torch.Tensor:
    """Center atoms per residue and apply random rotation per residue.

    Matches the official Boltz2 featurizer which applies center_random_augmentation
    to each residue's ref_pos independently (featurizerv2.py lines 1495-1500).
    """
    result = atom_positions.clone()
    residue_index_tensor = torch.tensor(atom_residue_index, dtype=torch.long)
    for residue_idx in range(num_residues):
        residue_mask = residue_index_tensor == residue_idx
        assert torch.any(residue_mask), f"Residue index {residue_idx} has no atoms."
        residue_coords = result[residue_mask]
        # Center
        residue_center = residue_coords.mean(dim=0, keepdim=True)
        residue_coords = residue_coords - residue_center
        # Random rotation (matching official center_random_augmentation with centering=True)
        R = _random_rotation_matrix()
        residue_coords = residue_coords @ R.T
        result[residue_mask] = residue_coords
    return result


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
    atom_positions = _center_and_augment_atoms_per_residue(
        atom_positions=atom_positions,
        atom_residue_index=template.atom_residue_index,
        num_residues=num_tokens,
    )

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

    # token_bonds encodes explicit covalent cross-links from structure bonds,
    # NOT backbone peptide bonds (those are implicit via residue_index + asym_id).
    # For a standard single-chain protein without cross-links, this is all zeros.
    # This matches the official Boltz2 featurizer (featurizerv2.py lines 696-705).
    token_bonds = torch.zeros((num_tokens, num_tokens), dtype=torch.float32)
    type_bonds = torch.zeros((num_tokens, num_tokens), dtype=torch.long)
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
