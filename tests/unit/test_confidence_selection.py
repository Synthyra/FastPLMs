"""Checkpoint selection must not hide one head's regression in combined loss."""

from tools.confidence.selection import improves_checkpoint


def test_both_heads_must_improve():
    donor = {"plddt_ce": 3.0, "pae_ce": 4.0, "target_plddt_spearman": 0.6}
    assert improves_checkpoint(dict(donor, plddt_ce=2.9, pae_ce=3.9), donor, donor)
    assert not improves_checkpoint(dict(donor, plddt_ce=2.0, pae_ce=4.1), donor, donor)
    assert not improves_checkpoint(dict(donor, plddt_ce=3.1, pae_ce=1.0), donor, donor)


def test_ranking_guard_and_nonfinite_values():
    donor = {"plddt_ce": 3.0, "pae_ce": 4.0, "target_plddt_spearman": 0.6}
    candidate = dict(donor, plddt_ce=2.9, pae_ce=3.9, target_plddt_spearman=0.57)
    assert not improves_checkpoint(candidate, donor, donor)
    assert improves_checkpoint(candidate, donor, donor, overfit=True)
    assert not improves_checkpoint(dict(candidate, pae_ce=float("nan")), donor, donor)
