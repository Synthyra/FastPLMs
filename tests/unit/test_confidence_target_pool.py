"""Confidence target pool: eligibility, whole-chain subsets, and token-budget variants."""

import numpy as np

from tools.confidence.target_pool import ChainSubset, _eligible, _targets_for_row, spatial_chain_subset


def _positions(lengths: list[int]) -> np.ndarray:
    """Atom14 positions (l, 14, 3) with every C-alpha resolved and chains placed 10 A apart."""
    positions = np.full((sum(lengths), 14, 3), np.nan, dtype=np.float32)
    start = 0
    for chain, length in enumerate(lengths):
        positions[start : start + length, 1] = [10.0 * chain, 0.0, 0.0]
        start += length
    return positions


def test_spatial_subset_grows_by_nearest_contact_and_skips_chains_that_do_not_fit():
    chain_ca = [np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]), np.array([[50.0, 0.0, 0.0]]), np.array([[49.0, 0.0, 0.0]])]
    lengths = [10, 10, 10, 30]  # the fourth chain never fits a 25-token budget
    for seed in range(10):
        start = int(np.random.default_rng(seed).choice([0, 1, 2]))
        expected = ChainSubset((1, 2), 20) if start == 2 else ChainSubset((0, 1), 20)
        assert spatial_chain_subset(chain_ca, lengths, budget=25, seed=seed) == expected


def test_spatial_subset_is_empty_when_no_chain_fits():
    assert spatial_chain_subset([np.zeros((1, 3))], [30], budget=25, seed=0) == ChainSubset((), 0)


def test_monomers_split_into_standard_and_long_variants():
    positions = np.zeros((1, 14, 3), dtype=np.float32)
    assert _targets_for_row("rcsb", "a", ["A" * 100], positions) == [("standard", ChainSubset((0,), 100))]
    assert _targets_for_row("rcsb", "a", ["A" * 1500], positions) == [("long", ChainSubset((0,), 1500))]
    assert _targets_for_row("rcsb", "a", ["A" * 3000], positions) == []


def test_complexes_keep_at_least_two_chains_in_each_variant():
    small = _targets_for_row("rcsb_multimer", "x", ["A" * 300, "C" * 300], _positions([300, 300]))
    assert small == [("standard", ChainSubset((0, 1), 600))]
    # Two 600-residue chains exceed the standard budget with one chain, so only the long variant remains.
    large = _targets_for_row("rcsb_multimer", "y", ["A" * 600, "C" * 600], _positions([600, 600]))
    assert large == [("long", ChainSubset((0, 1), 1200))]
    assert _targets_for_row("rcsb_multimer", "z", ["A" * 50], _positions([50])) == []


def test_eligibility_requires_resolution_standard_residues_and_resolved_chains():
    positions = _positions([6, 6])
    sequences = ["ACDEFG", "HIKLMN"]
    assert _eligible(sequences, positions, 2.5)
    assert not _eligible(sequences, positions, None)
    assert not _eligible(sequences, positions, 4.5)
    assert not _eligible(["ACDEFX", "HIKLMN"], positions, 2.5)
    sparse = positions.copy()
    sparse[6:9, 1] = np.nan  # the second chain keeps only three resolved C-alpha atoms
    assert not _eligible(sequences, sparse, 2.5)
