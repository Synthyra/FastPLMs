"""Fold confidence-training targets online and label every diffusion sample.

ESMFold2 tokenizes each standard residue as one token whose heavy atoms follow AtlasFold's
atom14 slot order (N, CA, C, O, CB, then the side chain), so true coordinates map onto native
atoms through each token's atom span. Each sample receives its own chain permutation and
symmetric-atom assignment before labels are computed, because different samples of one target
can place equivalent chains and atoms differently.

Shape symbols: k samples, a padded atoms, t padded tokens, l residues, p symmetric atom pairs,
and g groups of symmetric pairs. A c suffix selects one chain; d_pair is pair-channel width.
"""

from __future__ import annotations

import numpy as np
import torch

from collections.abc import Sequence
from dataclasses import dataclass

from scipy.optimize import linear_sum_assignment
from torch import Tensor

from fastplms.models.esmfold2.esmfold2_constants import PROTEIN_HEAVY_ATOMS
from fastplms.models.esmfold2.esmfold2_input_builder import ProteinInput, StructurePredictionInput
from fastplms.models.esmfold2.reproducibility import seed_context
from .labels import PAE_MAX_ANGSTROM, compute_targets


FOLDING_KERNEL_BACKEND = "cuequivariance"
# The defaults of `model.fold` in both model families. Heads train and are evaluated on these
# samples: on 12 validation targets, 15-step samples scored 0.013 higher all-atom lDDT on every
# target, a bias that would miscalibrate a head trained on short rollouts.
INFERENCE_LOOPS = 3
INFERENCE_SAMPLING_STEPS = 50
UNCHUNKED_HEAD_TOKENS = 1024  # longer inputs contract head triangles in chunks to bound memory
HEAD_CHUNK_SIZE = 256

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}  # fmt: skip

# Atom14 slots of symmetric heavy atoms, swapped together per residue. The groups follow the
# equivalent-atom table of the pinned AtlasFold release used by the pilot.
AMBIGUOUS_SLOTS: dict[str, tuple[tuple[int, int], ...]] = {
    "D": ((6, 7),),  # OD1, OD2
    "E": ((7, 8),),  # OE1, OE2
    "F": ((6, 7), (8, 9)),  # CD1/CD2, CE1/CE2
    "Y": ((6, 7), (8, 9)),  # CD1/CD2, CE1/CE2
    "R": ((9, 10),),  # NH1, NH2
    "L": ((6, 7),),  # CD1, CD2
    "V": ((5, 6),),  # CG1, CG2
}

ATOM14_NAMES = {
    "A": ("N", "CA", "C", "O", "CB"),
    "R": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "N": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    "D": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    "C": ("N", "CA", "C", "O", "CB", "SG"),
    "Q": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    "E": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    "G": ("N", "CA", "C", "O"),
    "H": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "I": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "L": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    "K": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    "M": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    "F": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "P": ("N", "CA", "C", "O", "CB", "CG", "CD"),
    "S": ("N", "CA", "C", "O", "CB", "OG"),
    "T": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "W": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "Y": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "V": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
}
if any(list(ATOM14_NAMES[letter]) != PROTEIN_HEAVY_ATOMS[code] for letter, code in ONE_TO_THREE.items()):
    raise ImportError("ESMFold2 heavy-atom order no longer matches AtlasFold atom14 slots")


@dataclass(frozen=True)
class TargetStructure:
    target_id: str
    sequences: tuple[str, ...]
    positions: np.ndarray  # (l, 14, 3) float32 over all chains; NaN for absent or unresolved atoms


@dataclass(frozen=True)
class AtomLayout:
    """Static correspondence between native atoms and true atom14 coordinates of one target."""

    true_index: Tensor  # (a,) index into flattened (l * 14) positions; -1 for padding atoms
    chain_atoms: tuple[Tensor, ...]  # per chain (a_c,) native atom indices
    chain_ca: tuple[Tensor, ...]  # per chain (l_c,) native CA atom indices in residue order
    entity_chains: tuple[tuple[int, ...], ...]  # chain indices grouped by identical sequence
    ambiguous_left: Tensor  # (p,) native atom index of the first atom of each symmetric pair
    ambiguous_right: Tensor  # (p,)
    ambiguous_group: Tensor  # (p,) residue group; pairs of one residue swap together
    backbone_indices: Tensor  # (t, 3) native N, CA, C atom index per token; -1 for padding


@dataclass
class Rollout:
    """One frozen fold with `k` diffusion samples and per-sample labels."""

    head_inputs: dict[str, Tensor]  # native confidence-head inputs with a batch axis of 1
    x_pred: Tensor  # (k, a, 3)
    true_coords: Tensor  # (k, a, 3)
    targets: list[dict[str, Tensor]]  # per-sample `compute_targets` outputs
    quality: list[dict[str, float]]  # per-sample lddt, lddt_ca, true_ptm, true_iptm
    layout: AtomLayout
    num_chains: int
    native_confidence: dict[str, Tensor] | None  # the model's own head outputs, when requested


def use_fast_folding_kernels(model: torch.nn.Module) -> None:
    """Fold with cuEquivariance kernels while the model's confidence head keeps the default kernels.

    On the GH200, a 1,024-token complex folds about 8 times faster than on the default path with
    32-row chunks. Its pair representation then differs from a default fold by about 2%, as much as
    two default folds of the same input differ from each other, so heads see inputs from the
    inference distribution.
    """
    model.set_kernel_backend(FOLDING_KERNEL_BACKEND)  # type: ignore[operator]
    model.set_chunk_size(None)  # type: ignore[operator]
    if model.confidence_head is not None:
        model.confidence_head.set_kernel_backend(None)  # type: ignore[union-attr, operator]


def head_chunk_size(num_tokens: int) -> int | None:
    """Triangle chunk size for a confidence head; chunked and unchunked heads give identical outputs."""
    return None if num_tokens <= UNCHUNKED_HEAD_TOKENS else HEAD_CHUNK_SIZE


def chain_label(index: int) -> str:
    """Chain ids A..Z, then AA, AB, ... for large complexes."""
    return chr(65 + index) if index < 26 else chain_label(index // 26 - 1) + chr(65 + index % 26)


def atom_layout(chain_infos: Sequence[object], sequences: Sequence[str], num_atoms: int, num_tokens: int) -> AtomLayout:
    residue_offsets = np.cumsum([0, *[len(sequence) for sequence in sequences]])  # (chains + 1,)
    true_index = np.full(num_atoms, -1, dtype=np.int64)  # (a,)
    backbone = np.full((num_tokens, 3), -1, dtype=np.int64)  # (t, 3)
    chain_atoms, chain_ca, left, right, group = [], [], [], [], []
    for chain_number, (chain, sequence) in enumerate(zip(chain_infos, sequences, strict=True)):
        tokens = chain.tokens  # type: ignore[attr-defined]
        if len(tokens) != len(sequence):
            raise ValueError(f"chain {chain_number} has {len(tokens)} tokens for {len(sequence)} residues")
        atoms, cas = [], []
        for token in tokens:
            letter = sequence[token.residue_index]
            if token.atom_count != len(ATOM14_NAMES[letter]):
                raise ValueError(f"residue {letter} has {token.atom_count} native atoms")
            span = np.arange(token.atom_start, token.atom_start + token.atom_count)  # (n_atom_residue,)
            residue = residue_offsets[chain_number] + token.residue_index
            true_index[span] = residue * 14 + np.arange(token.atom_count)  # (residue atoms,)
            backbone[token.token_index] = span[:3]  # (3,) N/CA/C indices
            atoms.append(span)
            cas.append(token.atom_start + 1)
            for left_slot, right_slot in AMBIGUOUS_SLOTS.get(letter, ()):
                left.append(token.atom_start + left_slot)
                right.append(token.atom_start + right_slot)
                group.append(residue)
        chain_atoms.append(torch.from_numpy(np.concatenate(atoms)))  # (a_c,)
        chain_ca.append(torch.tensor(cas, dtype=torch.long))  # (l_c,)
    entity_of: dict[str, list[int]] = {}
    for chain_number, sequence in enumerate(sequences):
        entity_of.setdefault(sequence, []).append(chain_number)
    return AtomLayout(
        true_index=torch.from_numpy(true_index),
        chain_atoms=tuple(chain_atoms),
        chain_ca=tuple(chain_ca),
        entity_chains=tuple(tuple(chains) for chains in entity_of.values()),
        ambiguous_left=torch.tensor(left, dtype=torch.long),
        ambiguous_right=torch.tensor(right, dtype=torch.long),
        ambiguous_group=torch.tensor(group, dtype=torch.long),
        backbone_indices=torch.from_numpy(backbone),
    )


def kabsch_transform(mobile: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation that map row-vector `mobile` (n, 3) onto `fixed` (n, 3)."""
    mobile_center, fixed_center = mobile.mean(0), fixed.mean(0)  # (3,), (3,)
    covariance = (mobile - mobile_center).T @ (fixed - fixed_center)  # (3, 3)
    left, _, right = np.linalg.svd(covariance)  # (3, 3), (3,), (3, 3)
    correction = np.diag([1.0, 1.0, np.sign(np.linalg.det(left @ right))])  # (3, 3)
    rotation = left @ correction @ right  # (3, 3); x_aligned = x @ rotation + translation
    return rotation, fixed_center - mobile_center @ rotation  # (3, 3), (3,)


def chain_assignment(predicted_ca: Sequence[np.ndarray], true_ca: Sequence[np.ndarray], entity_chains: Sequence[Sequence[int]]) -> list[int]:
    """Assign each native chain the true chain of its entity that it models best.

    Returns `assignment[native_chain] = true_chain`. An anchor chain from the entity with the
    fewest copies fixes the frame for each candidate; chains of every entity are then matched by
    resolved CA centroid distance, and the candidate with the lowest CA RMSD wins.
    """
    # predicted_ca/true_ca contain per-chain (l_c, 3) coordinate arrays.
    num_chains = len(predicted_ca)
    identity = list(range(num_chains))
    if all(len(chains) == 1 for chains in entity_chains):
        return identity
    anchor_entity = min(entity_chains, key=lambda chains: (len(chains), -len(true_ca[chains[0]])))
    anchor = anchor_entity[0]
    best, best_rmsd = identity, np.inf
    for candidate in anchor_entity:
        resolved = np.isfinite(true_ca[candidate]).all(-1)  # (l_anchor,)
        if resolved.sum() < 3:
            continue
        # (3, 3), (3,)
        rotation, translation = kabsch_transform(true_ca[candidate][resolved], predicted_ca[anchor][resolved])
        assignment = identity.copy()
        for chains in entity_chains:
            cost = np.zeros((len(chains), len(chains)))  # (native copy, true copy)
            for row, native in enumerate(chains):
                for column, true_chain in enumerate(chains):
                    mask = np.isfinite(true_ca[true_chain]).all(-1)  # (l_c,)
                    if mask.sum() == 0:
                        continue
                    true_centroid = (true_ca[true_chain][mask] @ rotation + translation).mean(0)  # (3,)
                    cost[row, column] = np.linalg.norm(predicted_ca[native][mask].mean(0) - true_centroid)  # ()
            rows, columns = linear_sum_assignment(cost)  # each (entity copies,)
            for row, column in zip(rows, columns, strict=True):
                assignment[chains[row]] = chains[column]
        squared, count = 0.0, 0
        for native in range(num_chains):
            mask = np.isfinite(true_ca[assignment[native]]).all(-1)  # (chain residues,)
            aligned = true_ca[assignment[native]][mask] @ rotation + translation  # (resolved chain residues, 3)
            squared += float(np.square(aligned - predicted_ca[native][mask]).sum())
            count += int(mask.sum())
        rmsd = np.sqrt(squared / max(count, 1))
        if rmsd < best_rmsd:
            best, best_rmsd = assignment, rmsd
    return best


def _aligned(mobile: Tensor, fixed: Tensor, valid: Tensor) -> Tensor:
    """Rigidly align `mobile` (a, 3) onto `fixed` (a, 3) using `valid` (a,) atoms."""
    rotation, translation = kabsch_transform(
        mobile[valid].double().cpu().numpy(), fixed[valid].double().cpu().numpy()
    )  # (3, 3), (3,)
    rotation_t = torch.as_tensor(rotation, dtype=mobile.dtype, device=mobile.device)  # (3, 3)
    translation_t = torch.as_tensor(translation, dtype=mobile.dtype, device=mobile.device)  # (3,)
    return mobile @ rotation_t + translation_t  # (a, 3)


def resolve_ambiguous_atoms(predicted: Tensor, true: Tensor, layout: AtomLayout) -> Tensor:
    """Swap symmetric atom labels of a residue when the swap sits closer to the prediction."""
    # predicted/true: (a, 3); layout holds the per-atom and per-pair index vectors.
    if layout.ambiguous_left.numel() == 0:
        return true  # (a, 3)
    valid = torch.isfinite(true).all(-1) & torch.isfinite(predicted).all(-1)  # (a,)
    if int(valid.sum()) < 3:
        return true  # (a, 3)
    aligned = _aligned(torch.nan_to_num(true), predicted, valid)  # (a, 3)
    left, right = layout.ambiguous_left.to(true.device), layout.ambiguous_right.to(true.device)  # (p,)
    group = torch.unique(layout.ambiguous_group.to(true.device), return_inverse=True)[1]  # (p,)
    num_groups = int(group.max()) + 1
    direct = (predicted[left] - aligned[left]).square().sum(-1) + (predicted[right] - aligned[right]).square().sum(-1)  # (p,)
    swapped = (predicted[left] - aligned[right]).square().sum(-1) + (predicted[right] - aligned[left]).square().sum(-1)  # (p,)
    pair_valid = (valid[left] & valid[right]).float()  # (p,)
    group_direct = torch.zeros(num_groups, device=true.device).index_add_(0, group, direct * pair_valid)  # (g,)
    group_swapped = torch.zeros(num_groups, device=true.device).index_add_(0, group, swapped * pair_valid)  # (g,)
    group_valid = torch.ones(num_groups, device=true.device).index_reduce_(0, group, pair_valid, "amin")  # (g,)
    swap = (group_valid > 0) & (group_swapped < group_direct)  # (g,)
    pair_swap = swap[group]  # (p,)
    output = true.clone()  # (a, 3)
    output[left[pair_swap]] = true[right[pair_swap]]  # (swapped pairs, 3)
    output[right[pair_swap]] = true[left[pair_swap]]  # (swapped pairs, 3)
    return output  # (a, 3)


def true_tm_scores(pae_error: Tensor, pae_mask: Tensor, asym_id: Tensor, token_mask: Tensor) -> tuple[float, float]:
    """pTM and ipTM of the true aligned errors, defined as the head defines its predictions."""
    # pae_error/pae_mask: (t, t); asym_id/token_mask: (t,).
    num_tokens = token_mask.float().sum()  # ()
    d0 = 1.24 * (num_tokens.clamp(min=19) - 15) ** (1 / 3) - 1.8  # ()
    tm = 1.0 / (1.0 + (pae_error.clamp(max=PAE_MAX_ANGSTROM) / d0) ** 2)  # (t, t)
    mask = pae_mask.float()  # (t, t)
    inter_chain = mask * (asym_id[:, None] != asym_id[None, :]).float()  # (t, t)
    ptm = ((tm * mask).sum(-1) / mask.sum(-1).clamp(min=1)).max()  # ()
    rows = inter_chain.sum(-1) > 0  # (t,)
    # ()
    iptm = ((tm * inter_chain).sum(-1)[rows] / inter_chain.sum(-1)[rows]).max() if rows.any() else torch.tensor(float("nan"))
    return float(ptm), float(iptm)


@torch.no_grad()
def fold(
    model: torch.nn.Module,
    structure: TargetStructure,
    num_samples: int,
    seed: int,
    num_loops: int = INFERENCE_LOOPS,
    num_sampling_steps: int = INFERENCE_SAMPLING_STEPS,
    native_confidence: bool = False,
) -> Rollout:
    """Run the frozen model once and label each of its `num_samples` diffusion samples.

    The call is seeded the way `model.fold` seeds it, which both the experimental and the
    production model families support. A model whose own confidence head is enabled also returns
    that head's outputs when `native_confidence` is set.
    """
    device = next(model.parameters()).device
    inputs = StructurePredictionInput(
        sequences=[ProteinInput(id=chain_label(index), sequence=sequence) for index, sequence in enumerate(structure.sequences)]
    )
    features, chain_infos = model.prepare_structure_input(inputs, seed=seed)
    features = {name: value.to(device) for name, value in features.items()}  # per-field shapes unchanged
    if native_confidence:
        model.confidence_head.set_chunk_size(head_chunk_size(features["token_attention_mask"].shape[-1]))  # type: ignore[union-attr, operator]
    with torch.autocast(device.type, dtype=torch.bfloat16), seed_context(seed):
        output = model(
            **features,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_samples,
            output_hidden_states=True,
            return_dict=True,
        )
        relative_position = model.rel_pos(
            residue_index=features["residue_index"],
            asym_id=features["asym_id"],
            sym_id=features["sym_id"],
            entity_id=features["entity_id"],
            token_index=features["token_index"],
        )  # (1, t, t, d_pair)
        token_bonds = model.token_bonds(features["token_bonds"].float())  # (1, t, t, d_pair)

    atom_to_token = features["atom_to_token"].reshape(-1).long()  # (a,)
    atom_mask = features["atom_attention_mask"].reshape(-1).bool()  # (a,)
    token_mask = features["token_attention_mask"].reshape(-1).bool()  # (t,)
    asym_id = features["asym_id"].reshape(-1)  # (t,)
    layout = atom_layout(chain_infos, structure.sequences, atom_to_token.numel(), token_mask.numel())
    x_pred = output["sample_atom_coords"].reshape(num_samples, -1, 3).float()  # (k, a, 3)

    flat_positions = torch.from_numpy(structure.positions.reshape(-1, 3)).to(device)  # (l * 14, 3)
    true_base = torch.full_like(x_pred[0], float("nan"))  # (a, 3)
    mapped = layout.true_index >= 0  # (a,)
    true_base[mapped.to(device)] = flat_positions[layout.true_index[mapped].to(device)]  # (mapped atoms, 3)
    true_ca = [true_base[ca.to(device)].double().cpu().numpy() for ca in layout.chain_ca]  # per chain (l_c, 3)

    true_coords, targets, quality = [], [], []
    backbone = layout.backbone_indices.to(device)  # (t, 3)
    for sample in range(num_samples):
        predicted = x_pred[sample]  # (a, 3)
        # per chain (l_c, 3)
        predicted_ca = [predicted[ca.to(device)].double().cpu().numpy() for ca in layout.chain_ca]
        assignment = chain_assignment(predicted_ca, true_ca, layout.entity_chains)
        true = true_base.clone()  # (a, 3)
        for native, true_chain in enumerate(assignment):
            if native != true_chain:
                # (a_c, 3)
                true[layout.chain_atoms[native].to(device)] = true_base[layout.chain_atoms[true_chain].to(device)]
        true = resolve_ambiguous_atoms(predicted, true, layout)  # (a, 3)
        resolved = torch.isfinite(true).all(-1) & atom_mask  # (a,)
        sample_targets = compute_targets(predicted, true, resolved, atom_to_token, backbone, token_mask)
        ptm, iptm = true_tm_scores(sample_targets["pae_error"], sample_targets["pae_mask"], asym_id, token_mask)
        quality.append(
            {
                "lddt": float(sample_targets["plddt_score"][sample_targets["plddt_mask"]].mean()),
                "lddt_ca": float(sample_targets["lddt_ca"][sample_targets["lddt_ca_mask"]].mean()),
                "true_ptm": ptm,
                "true_iptm": iptm,
            }
        )
        true_coords.append(true)
        targets.append(sample_targets)

    head_inputs = {
        "s_inputs": output.hidden_states[0].detach().float(),  # (1, t, d_inputs)
        "z": output.hidden_states[1].detach().float(),  # (1, t, t, d_pair)
        "distogram_atom_idx": features["distogram_atom_idx"],
        "token_attention_mask": features["token_attention_mask"],
        "atom_to_token": features["atom_to_token"],
        "atom_attention_mask": features["atom_attention_mask"],
        "asym_id": features["asym_id"],
        "mol_type": features["mol_type"],
        "relative_position_encoding": relative_position.detach().float(),  # (1, t, t, d_pair)
        "token_bonds_encoding": token_bonds.detach().float(),  # (1, t, t, d_pair)
    }
    return Rollout(
        head_inputs=head_inputs,
        x_pred=x_pred,
        true_coords=torch.stack(true_coords),  # (k, a, 3)
        targets=targets,
        quality=quality,
        layout=layout,
        num_chains=len(structure.sequences),
        native_confidence=(
            {name: output[name].detach() for name in ("plddt_logits", "plddt_per_atom", "pae_logits", "ptm", "iptm")}
            if native_confidence
            else None
        ),
    )
