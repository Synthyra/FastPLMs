"""Backbone frames, C-beta placement, lDDT, RMSD, GDT-TS, and contact precision on small synthetic structures.

The structures are an idealized backbone placed along a helix, so a rigid motion, a missing residue, or a stretched
copy has an answer worked out by hand.

Shapes: `b` structures, `l` residues, `n` atoms or points, `a` atoms per residue.
"""

import numpy as np
import pytest
import torch

from tests.unit.synthetic_structures import IDEAL_BACKBONE, atom37_of, helix_backbone, moved

from fastplms.models.esmfold2 import esmfold2_residue_constants as residue_constants
from fastplms.models.esmfold2.esmfold2_affine3d import Affine3D
from fastplms.models.esmfold2.esmfold2_metrics import (
    compute_gdt_ts,
    compute_lddt,
    compute_lddt_ca,
    compute_lddt_from_dmat,
    compute_rmsd,
    contact_precision,
)
from fastplms.models.esmfold2.esmfold2_normalize_coordinates import (
    apply_frame_to_coords,
    atom3_to_backbone_frames,
    get_protein_normalization_frame,
    normalize_coordinates,
)
from fastplms.models.esmfold2.esmfold2_protein_structure import (
    compute_gdt_ts_no_alignment,
    infer_cbeta_from_atom37,
)


IDEAL_CBETA = [-0.529, -0.774, -1.205]  # the C-beta that goes with IDEAL_BACKBONE


@pytest.mark.filterwarnings("ignore:Using torch.cross without specifying the dim arg:UserWarning")
def test_c_beta_inference_gives_the_ideal_position_for_numpy_and_torch_coordinates():
    one_residue = atom37_of(torch.tensor([IDEAL_BACKBONE]))  # (1, 37, 3)

    from_torch = infer_cbeta_from_atom37(one_residue)  # (1, 3)
    from_numpy = infer_cbeta_from_atom37(one_residue.numpy())  # (1, 3)

    torch.testing.assert_close(from_torch, torch.tensor([IDEAL_CBETA]), atol=0.01, rtol=0)
    np.testing.assert_allclose(from_numpy, from_torch.numpy(), atol=1e-5)


def test_a_frame_from_backbone_atoms_sits_on_the_alpha_carbon_and_points_from_carbon_to_alpha_carbon():
    backbone = helix_backbone(3)  # (l, 3, 3)

    frames = atom3_to_backbone_frames(backbone)  # frames (l,)

    assert frames.shape == (3,)
    assert torch.equal(frames.trans, backbone[:, 1])
    axis = torch.nn.functional.normalize(backbone[:, 1] - backbone[:, 2], dim=-1)  # (l, 3)
    torch.testing.assert_close(frames.rot.to_3x3()[..., 0], axis, atol=1e-5, rtol=0)


def test_normalizing_coordinates_removes_the_pose_and_keeps_unresolved_atoms_unresolved():
    backbone = helix_backbone(6)  # (l, 3, 3)

    normalized = normalize_coordinates(atom37_of(backbone))  # (l, 37, 3)
    normalized_after_moving = normalize_coordinates(atom37_of(moved(backbone)))  # (l, 37, 3)

    torch.testing.assert_close(normalized, normalized_after_moving, atol=1e-4, rtol=0)
    ca = normalized[:, residue_constants.atom_order["CA"]]  # (l, 3)
    torch.testing.assert_close(ca.mean(dim=0), torch.zeros(3), atol=1e-4, rtol=0)
    assert torch.isinf(normalized[:, residue_constants.atom_order["CB"]]).all()


def test_the_normalization_frame_ignores_residues_without_a_backbone():
    backbone = helix_backbone(5)  # (l, 3, 3)
    with_gap = backbone.clone()
    with_gap[2] = torch.nan
    kept = torch.cat([backbone[:2], backbone[3:]])  # (l - 1, 3, 3)

    frame = get_protein_normalization_frame(atom37_of(with_gap))  # one frame
    expected = get_protein_normalization_frame(atom37_of(kept))  # one frame

    torch.testing.assert_close(frame.trans, expected.trans, atol=1e-5, rtol=0)
    torch.testing.assert_close(frame.rot.to_3x3(), expected.rot.to_3x3(), atol=1e-5, rtol=0)


def test_a_frame_that_sits_at_the_origin_leaves_coordinates_unchanged():
    coords = atom37_of(helix_backbone(3))  # (l, 37, 3)
    no_frame = Affine3D.identity(())  # a frame with zero translation, which counts as unset

    unchanged = apply_frame_to_coords(coords, no_frame)  # (l, 37, 3)

    assert torch.equal(unchanged, coords)


def test_lddt_is_one_for_identical_structures_and_follows_the_distance_error_bands_when_stretched():
    line = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])  # (l, 3)
    present = torch.ones(4)  # (l,)

    same = compute_lddt(line, line, present)  # (l,)
    per_residue = compute_lddt(2 * line, line, present)  # (l,)
    overall = compute_lddt(2 * line, line, present, per_residue=False)  # ()

    torch.testing.assert_close(same, torch.ones(4), atol=1e-4, rtol=0)
    # Doubled distances err by 1, 2, and 3 angstroms where the true distances are 1, 2, and 3: scores 0.5, 0.25, 0.25.
    torch.testing.assert_close(per_residue, torch.tensor([1 / 3, 5 / 12, 5 / 12, 1 / 3]), atol=1e-4, rtol=0)
    torch.testing.assert_close(overall, torch.tensor(0.375), atol=1e-4, rtol=0)


def test_lddt_from_distance_matrices_scores_only_pairs_within_the_cutoff_and_honors_pair_masks_and_chains():
    line = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [20.0, 0.0, 0.0]])  # (l, 3)
    stretched = line * torch.tensor([2.0, 1.0, 1.0])  # (l, 3), the first two residues are 2 angstroms apart
    true_distances = torch.cdist(line, line)  # (l, l)
    predicted_distances = torch.cdist(stretched, stretched)  # (l, l)
    everything = torch.ones(3, 3)  # (l, l)
    present = torch.ones(3)  # (l,)
    without_first_pair = everything.clone()
    without_first_pair[0, 1] = without_first_pair[1, 0] = 0.0

    scores = compute_lddt_from_dmat(predicted_distances, true_distances, everything, cutoff=15.0)  # (l,)
    first_pair_unscored = compute_lddt_from_dmat(predicted_distances, true_distances, without_first_pair)  # (l,)
    same_chain = compute_lddt(stretched, line, present, sequence_id=torch.tensor([0, 0, 1]))  # (l,)
    split_chains = compute_lddt(stretched, line, present, sequence_id=torch.tensor([0, 1, 1]))  # (l,)
    masked_pairs = compute_lddt(stretched, line, present, pairwise_all_atom_mask=without_first_pair)  # (l,)

    # The first two residues err by 1 angstrom in their one pair: the band score 0.5. The far residue, 20 angstroms
    # from the others, has no pair inside the cutoff, and a residue with nothing to score gets 1.
    assert scores.tolist() == pytest.approx([0.5, 0.5, 1.0], abs=1e-4)
    assert first_pair_unscored.tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)
    assert same_chain.tolist() == pytest.approx([0.5, 0.5, 1.0], abs=1e-4)
    assert split_chains.tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)
    assert masked_pairs.tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)


def test_c_alpha_lddt_reads_the_alpha_carbons_from_atom37_or_from_a_c_alpha_only_prediction():
    truth = atom37_of(helix_backbone(5))[None]  # (b, l, 37, 3)
    mask = torch.isfinite(truth).all(dim=-1).float()  # (b, l, 37)
    ca = truth[:, :, residue_constants.atom_order["CA"]]  # (b, l, 3)

    from_atom37 = compute_lddt_ca(truth, truth, mask)  # (b, l)
    from_ca_only = compute_lddt_ca(ca, truth, mask, sequence_id=torch.zeros(1, 5, dtype=torch.long))  # (b, l)
    stretched = compute_lddt_ca(ca * 1.5, truth, mask)  # (b, l)

    torch.testing.assert_close(from_atom37, torch.ones(1, 5), atol=1e-4, rtol=0)
    torch.testing.assert_close(from_ca_only, torch.ones(1, 5), atol=1e-4, rtol=0)
    assert stretched.max() < 1.0


def test_rmsd_is_zero_for_a_rigidly_moved_copy_and_positive_for_a_distorted_one():
    target = helix_backbone(4).reshape(1, 12, 3)  # (b, n, 3)
    copy = moved(target)  # (b, n, 3)
    distorted = copy.clone()
    distorted[0, 0] += torch.tensor([3.0, 0.0, 0.0])

    batch = compute_rmsd(copy, target)  # ()
    per_sample = compute_rmsd(copy, target, reduction="per_sample")  # (b,)
    per_residue = compute_rmsd(copy, target, reduction="per_residue")  # (b, n / 3)
    larger = compute_rmsd(distorted, target)  # ()

    assert batch.item() == pytest.approx(0.0, abs=1e-3)
    assert per_sample.shape == (1,) and per_residue.shape == (1, 4)
    assert per_residue.abs().max() < 1e-3
    assert 0.1 < larger.item() < 3.0


def test_rmsd_accepts_per_atom_masks_and_packed_sequences():
    target = helix_backbone(4)  # (l, 3, 3)
    copy = moved(target)  # (l, 3, 3)
    padded = copy.clone()
    padded[0, 0] = 100.0
    keep = torch.ones(1, 4, 3, dtype=torch.bool)  # (b, l, a)
    keep[0, 0, 0] = False
    packed_ids = torch.tensor([[0, 0, 1, 1]])  # (b, l)

    masked = compute_rmsd(padded[None], target[None], atom_exists_mask=keep)  # ()
    by_sequence = compute_rmsd(copy[None], target[None], sequence_id=packed_ids, reduction="per_sample")  # (b_eff,)
    by_residue = compute_rmsd(copy[None], target[None], sequence_id=packed_ids, reduction="per_residue")  # (b, l)

    assert masked.item() == pytest.approx(0.0, abs=1e-3)
    assert by_sequence.shape == (2,) and by_sequence.abs().max() < 1e-3
    assert by_residue.shape == (1, 4)


def test_gdt_ts_counts_the_atoms_within_each_distance_threshold():
    aligned = torch.zeros(1, 5, 3)  # (b, n, 3)
    aligned[0, :, 0] = torch.tensor([0.5, 1.5, 3.0, 6.0, 20.0])
    target = torch.zeros(1, 5, 3)  # (b, n, 3)
    everything = torch.ones(1, 5, dtype=torch.bool)  # (b, n)

    per_sample = compute_gdt_ts_no_alignment(aligned, target, everything, reduction="per_sample")  # (b,)
    batch = compute_gdt_ts_no_alignment(aligned, target, None, reduction="batch")  # ()

    # 1, 2, 3, and 4 of the 5 atoms lie within 1, 2, 4, and 8 angstroms: scores 0.2, 0.4, 0.6, 0.8.
    assert per_sample.tolist() == pytest.approx([0.5])
    assert batch.item() == pytest.approx(0.5)


def test_gdt_ts_after_alignment_is_one_for_a_rigidly_moved_copy_and_less_for_a_displaced_atom():
    target = helix_backbone(4).reshape(1, 12, 3)  # (b, n, 3)
    copy = moved(target)  # (b, n, 3)
    displaced = copy.clone()
    displaced[0, 5] += torch.tensor([30.0, 0.0, 0.0])
    packed_ids = torch.tensor([[0] * 6 + [1] * 6])  # (b, n), two packed structures of six points

    same = compute_gdt_ts(copy, target)  # (b,)
    worse = compute_gdt_ts(displaced, target, reduction="batch")  # ()
    packed = compute_gdt_ts(copy, target, sequence_id=packed_ids)  # (b_eff,)

    assert same.item() == pytest.approx(1.0, abs=1e-4)
    assert worse.item() < 1.0
    assert packed.shape == (2,) and packed.min() > 0.99


def test_contact_precision_ranks_the_predicted_contacts_against_the_true_ones():
    length = 10
    targets = torch.zeros(length, length)  # (l, l)
    for row, column in [(0, 7), (1, 8), (2, 9), (0, 9), (3, 9)][:5]:
        targets[row, column] = 1.0
    scores = targets + 0.01 * torch.rand(length, length, generator=torch.Generator().manual_seed(0))  # (l, l)

    perfect = contact_precision(scores, targets)  # each metric (1,)
    batched = contact_precision(scores[None].repeat(2, 1, 1), targets[None].repeat(2, 1, 1), src_lengths=torch.tensor([10, 10]))
    short_source = contact_precision(scores, targets, src_lengths=torch.tensor([6]))
    longer = contact_precision(scores, targets, override_length=20)
    bounded = contact_precision(scores, targets, maxsep=8)

    assert perfect["P@L"].item() == pytest.approx(0.5)  # 5 contacts among the 10 highest-ranked pairs
    assert perfect["P@L5"].item() == pytest.approx(1.0)
    assert set(perfect) == {"AUC", "P@L", "P@L5"}
    assert batched["P@L"].shape == (2,)
    assert short_source["P@L"].shape == (1,)
    assert longer["P@L"].item() == pytest.approx(0.25)
    assert bounded["AUC"].shape == (1,)
    with pytest.raises(ValueError, match="Size mismatch"):
        contact_precision(scores, targets[:5, :5])
