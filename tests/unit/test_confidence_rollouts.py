"""Online rollout labels: atom layout, rigid alignment, chain permutation, symmetric atoms, TM scores."""

import math
import numpy as np
import pytest
import torch

from types import SimpleNamespace

from tools.confidence.rollouts import (
    AtomLayout,
    atom_layout,
    chain_assignment,
    chain_label,
    kabsch_transform,
    resolve_ambiguous_atoms,
    true_tm_scores,
)


def _rotation(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def test_chain_labels_continue_past_z():
    assert [chain_label(index) for index in (0, 25, 26, 27, 51, 52)] == ["A", "Z", "AA", "AB", "AZ", "BA"]


def test_atom_layout_maps_native_atom_spans_to_atom14_slots():
    # Chain A is "GA" (4 and 5 heavy atoms) and chain B is "G"; the last two native atoms are padding.
    chain_a = SimpleNamespace(
        tokens=[
            SimpleNamespace(residue_index=0, atom_start=0, atom_count=4, token_index=0),
            SimpleNamespace(residue_index=1, atom_start=4, atom_count=5, token_index=1),
        ]
    )
    chain_b = SimpleNamespace(tokens=[SimpleNamespace(residue_index=0, atom_start=9, atom_count=4, token_index=2)])
    layout = atom_layout([chain_a, chain_b], ["GA", "G"], num_atoms=15, num_tokens=4)
    assert layout.true_index.tolist() == [0, 1, 2, 3, 14, 15, 16, 17, 18, 28, 29, 30, 31, -1, -1]
    assert layout.backbone_indices.tolist() == [[0, 1, 2], [4, 5, 6], [9, 10, 11], [-1, -1, -1]]
    assert [ca.tolist() for ca in layout.chain_ca] == [[1, 5], [10]]
    assert layout.entity_chains == ((0,), (1,))


def test_atom_layout_rejects_an_unexpected_atom_count():
    chain = SimpleNamespace(tokens=[SimpleNamespace(residue_index=0, atom_start=0, atom_count=5, token_index=0)])
    with pytest.raises(ValueError, match="native atoms"):
        atom_layout([chain], ["G"], num_atoms=5, num_tokens=1)


def test_kabsch_transform_recovers_a_rigid_motion_of_row_vectors():
    mobile = np.random.default_rng(0).normal(size=(20, 3))
    fixed = mobile @ _rotation(0.7) + np.array([1.0, -2.0, 3.0])
    rotation, translation = kabsch_transform(mobile, fixed)
    np.testing.assert_allclose(mobile @ rotation + translation, fixed, atol=1e-8)


def test_chain_assignment_recovers_a_homotrimer_permutation_with_unresolved_atoms():
    rng = np.random.default_rng(1)
    true_ca = [rng.normal(size=(12, 3)) + 10.0 * offset for offset in np.eye(3)]
    permutation = [2, 0, 1]  # native chain i models true chain permutation[i]
    predicted_ca = [true_ca[permutation[index]] @ _rotation(1.1) + np.array([5.0, 0.0, -2.0]) for index in range(3)]
    true_ca[1][0] = np.nan
    assert chain_assignment(predicted_ca, true_ca, [(0, 1, 2)]) == permutation


def test_chain_assignment_permutes_only_within_an_entity():
    rng = np.random.default_rng(2)
    true_ca = [rng.normal(size=(10, 3)) + 12.0 * offset for offset in np.eye(3)]
    permutation = [1, 0, 2]  # the two copies of the first entity are swapped
    predicted_ca = [true_ca[permutation[index]] @ _rotation(-0.4) for index in range(3)]
    assert chain_assignment(predicted_ca, true_ca, [(0, 1), (2,)]) == permutation
    assert chain_assignment(predicted_ca, true_ca, [(0,), (1,), (2,)]) == [0, 1, 2]


def test_symmetric_atom_labels_follow_the_prediction():
    true = torch.tensor(np.random.default_rng(3).normal(size=(8, 3)) * 3.0, dtype=torch.float32)  # one aspartate
    swapped = true.clone()
    swapped[[6, 7]] = true[[7, 6]]  # OD1 and OD2 exchanged
    predicted = swapped @ torch.tensor(_rotation(0.4), dtype=torch.float32) + 1.0
    layout = AtomLayout(
        true_index=torch.arange(8),
        chain_atoms=(torch.arange(8),),
        chain_ca=(torch.tensor([1]),),
        entity_chains=((0,),),
        ambiguous_left=torch.tensor([6]),
        ambiguous_right=torch.tensor([7]),
        ambiguous_group=torch.tensor([0]),
        backbone_indices=torch.tensor([[0, 1, 2]]),
    )
    torch.testing.assert_close(resolve_ambiguous_atoms(predicted, true, layout), swapped)
    # Labels that already match the prediction stay as they are.
    torch.testing.assert_close(resolve_ambiguous_atoms(predicted, swapped, layout), swapped)


def test_true_tm_scores_of_an_exact_prediction():
    tokens = 30
    errors = torch.zeros(tokens, tokens)
    pair_mask = torch.ones(tokens, tokens, dtype=torch.bool)
    token_mask = torch.ones(tokens, dtype=torch.bool)
    ptm, iptm = true_tm_scores(errors, pair_mask, torch.tensor([0] * 15 + [1] * 15), token_mask)
    assert ptm == pytest.approx(1.0) and iptm == pytest.approx(1.0)
    _, monomer_iptm = true_tm_scores(errors, pair_mask, torch.zeros(tokens, dtype=torch.long), token_mask)
    assert math.isnan(monomer_iptm)
