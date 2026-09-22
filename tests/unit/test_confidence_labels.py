"""CPU checks for confidence-head geometry targets and losses."""

import pytest
import torch

from tools.confidence.labels import confidence_loss, compute_targets


def _example() -> tuple[torch.Tensor, ...]:
    true = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )  # (6, 3)
    predicted = true.clone()  # (6, 3)
    resolved = torch.ones(6, dtype=torch.bool)  # (6,)
    atom_to_token = torch.tensor([0, 0, 0, 1, 1, 1])  # (6,)
    backbone = torch.tensor([[0, 1, 2], [3, 4, 5]])  # (2, 3)
    token_mask = torch.ones(2, dtype=torch.bool)  # (2,)
    return predicted, true, resolved, atom_to_token, backbone, token_mask


def test_identity_has_perfect_scores_and_zero_pae() -> None:
    targets = compute_targets(*_example())
    assert torch.equal(targets["plddt_target"][targets["plddt_mask"]], torch.full((6,), 49))
    assert torch.equal(targets["pae_target"][targets["pae_mask"]], torch.zeros(4, dtype=torch.long))
    assert torch.allclose(targets["pae_error"], torch.zeros((2, 2)))


def test_rigid_transform_preserves_targets() -> None:
    predicted, true, resolved, atom_to_token, backbone, token_mask = _example()
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # (3, 3)
    shift = torch.tensor([3.0, -2.0, 4.0])  # (3,)
    transformed = true @ rotation.T + shift
    original = compute_targets(predicted, true, resolved, atom_to_token, backbone, token_mask)
    changed = compute_targets(
        transformed, transformed, resolved, atom_to_token, backbone, token_mask
    )
    for name in ("plddt_score", "pae_error", "lddt_ca"):
        assert torch.allclose(original[name], changed[name], atol=1e-5)


def test_pae_uses_source_frame_direction() -> None:
    predicted, true, resolved, atom_to_token, backbone, token_mask = _example()
    predicted = predicted.clone()
    predicted[3:] += torch.tensor([0.0, 0.0, 2.0])
    targets = compute_targets(predicted, true, resolved, atom_to_token, backbone, token_mask)
    assert targets["pae_error"][0, 1] == pytest.approx(2.0)
    assert targets["pae_error"][1, 0] == pytest.approx(2.0)


def test_pae_matches_atlasfold_frame_projection_for_unequal_bond_angles() -> None:
    true = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 2.0, 1.0],
            [4.0, 2.0, 1.0],
            [3.0, 3.0, 1.0],
        ]
    )  # (6, 3)
    predicted = true.clone()  # (6, 3)
    predicted[2] = torch.tensor([0.0, 1.0, 1.0])  # (3,)
    resolved = torch.ones(6, dtype=torch.bool)  # (6,)
    atom_to_token = torch.tensor([0, 0, 0, 1, 1, 1])  # (6,)
    backbone = torch.tensor([[0, 1, 2], [3, 4, 5]])  # (2, 3)
    targets = compute_targets(
        predicted, true, resolved, atom_to_token, backbone, torch.ones(2, dtype=torch.bool)
    )

    def atlas_basis(n: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        n = torch.nn.functional.normalize(n, dim=0)
        c = torch.nn.functional.normalize(c, dim=0)
        e1 = torch.nn.functional.normalize(n + c, dim=0)
        e2 = torch.nn.functional.normalize(c - n, dim=0)
        e3 = torch.linalg.cross(e1, e2, dim=0)
        return torch.stack((e1, e2, e3))

    predicted_basis = atlas_basis(predicted[0] - predicted[1], predicted[2] - predicted[1])
    true_basis = atlas_basis(true[0] - true[1], true[2] - true[1])
    displacement = true[4] - true[1]
    expected = torch.linalg.vector_norm(predicted_basis @ displacement - true_basis @ displacement)
    assert targets["pae_error"][0, 1] == pytest.approx(float(expected), abs=1e-6)
    assert targets["pae_error"][1, 0] == pytest.approx(0.0, abs=1e-6)


def test_missing_atoms_mask_labels() -> None:
    predicted, true, resolved, atom_to_token, backbone, token_mask = _example()
    resolved[2] = False
    targets = compute_targets(predicted, true, resolved, atom_to_token, backbone, token_mask)
    assert not targets["plddt_mask"][2]
    assert not targets["pae_mask"][0].any()


def test_bin_endpoints_are_in_range() -> None:
    predicted, true, resolved, atom_to_token, backbone, token_mask = _example()
    predicted = predicted.clone()
    predicted[0, 0] += 32.0
    targets = compute_targets(predicted, true, resolved, atom_to_token, backbone, token_mask)
    assert int(targets["plddt_target"].min()) >= 0
    assert int(targets["plddt_target"].max()) <= 49
    assert int(targets["pae_target"].min()) >= 0
    assert int(targets["pae_target"].max()) <= 63


def test_empty_labels_raise() -> None:
    predicted, true, resolved, atom_to_token, backbone, token_mask = _example()
    with pytest.raises(ValueError, match="at least one atom"):
        compute_targets(
            predicted[:0], true[:0], resolved[:0], atom_to_token[:0], backbone, token_mask
        )


def test_loss_backpropagates_only_through_logits() -> None:
    targets = compute_targets(*_example())
    plddt_logits = torch.zeros((1, 6, 50), requires_grad=True)  # (1, 6, 50)
    pae_logits = torch.zeros((1, 2, 2, 64), requires_grad=True)  # (1, 2, 2, 64)
    loss = confidence_loss({"plddt_logits": plddt_logits, "pae_logits": pae_logits}, targets)
    loss["total"].backward()
    assert plddt_logits.grad is not None
    assert pae_logits.grad is not None
    assert torch.isfinite(plddt_logits.grad).all()
    assert torch.isfinite(pae_logits.grad).all()


def test_uneven_token_mask_excludes_pae_rows_and_columns() -> None:
    example = list(_example())
    example[-1] = torch.tensor([True, False])  # (2,)
    targets = compute_targets(*example)
    assert targets["pae_mask"].tolist() == [[True, False], [False, False]]


def test_pae_target_needs_only_ca_but_source_needs_full_frame() -> None:
    predicted, true, resolved, atom_to_token, backbone, token_mask = _example()
    resolved[3] = False
    targets = compute_targets(predicted, true, resolved, atom_to_token, backbone, token_mask)
    assert targets["pae_mask"].tolist() == [[True, True], [False, False]]
    assert targets["lddt_ca_mask"].tolist() == [True, True]


def test_invalid_token_mask_excludes_atom_labels() -> None:
    example = list(_example())
    example[-1] = torch.tensor([False, True])  # (2,)
    targets = compute_targets(*example)
    assert not targets["plddt_mask"][:3].any()
    assert targets["plddt_mask"][3:].all()


def test_strict_cutoff_excludes_fifteen_angstrom_pairs() -> None:
    predicted = torch.tensor([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]])  # (2, 3)
    resolved = torch.ones(2, dtype=torch.bool)  # (2,)
    targets = compute_targets(
        predicted,
        predicted,
        resolved,
        torch.tensor([0, 1]),
        torch.tensor([[0, 0, 0], [1, 1, 1]]),
        torch.ones(2, dtype=torch.bool),
    )
    assert not targets["plddt_mask"].any()


def test_bfloat16_inputs_produce_float32_detached_targets() -> None:
    example = [
        value.to(torch.bfloat16) if value.is_floating_point() else value for value in _example()
    ]
    targets = compute_targets(*example)
    assert targets["plddt_score"].dtype == torch.float32
    assert targets["pae_error"].dtype == torch.float32
    assert all(not value.requires_grad for value in targets.values())
