"""Seeded feature smoke test for the FastPLMs binder-design workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from examples import binder_design_fastplms as binder

APPROVED_CRITIC = "ESMFold2-Experimental-Fast-Cutoff2025"


class FakeInputBuilder:
    def decode(
        self,
        output: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        chain_infos: list[Any],
        num_diffusion_samples: int,
        complex_id: str,
    ) -> dict[str, Any]:
        del output, inputs, chain_infos, num_diffusion_samples
        return {"complex_id": complex_id}


@dataclass
class FakeCritic:
    input_builder = FakeInputBuilder()

    def result_to_cif(self, complex_result: dict[str, Any]) -> str:
        return f"data_{complex_result['complex_id']}\n"

    def result_to_pdb(self, complex_result: dict[str, Any]) -> str:
        del complex_result
        return "HEADER FASTPLMS TEST\nEND\n"


def _fake_fold(
    model: Any,
    target_seq: str,
    target_one_hot: torch.Tensor,
    design: torch.Tensor,
    num_loops: int = 0,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    del model, num_loops, num_sampling_steps, calculate_confidence, seed
    b, binder_length, d = design.shape
    target_length = target_one_hot.size(1)
    aa_weight = torch.linspace(-1.0, 1.0, d, device=design.device)
    binder_signal = (design * aa_weight).sum(dim=-1)
    token_signal = torch.cat(
        (torch.zeros(b, target_length, device=design.device), binder_signal),
        dim=1,
    )
    pair_signal = token_signal[:, :, None] + token_signal[:, None, :]
    bin_basis = torch.linspace(-1.0, 1.0, 128, device=design.device)
    distogram_logits = pair_signal.unsqueeze(-1) * bin_basis
    sequences = [f"{target_seq}|{'A' * binder_length}" for _ in range(b)]
    return {
        "distogram_logits": distogram_logits,
        "inputs": {},
        "chain_info_list": [[] for _ in range(b)],
        "output": {"distogram_logits": distogram_logits},
        "seq_list": sequences,
        "iptm": torch.ones(b, device=design.device),
        "ptm": torch.ones(b, device=design.device),
        "plddt": torch.ones(b, 1, device=design.device),
    }


def _fake_pseudoperplexity(
    lm_model: Any,
    binder_design: torch.Tensor,
    score_mask: torch.Tensor,
    batch_size: int = 4,
    n_passes: int = 4,
    mask_fraction: float = binder.DEFAULT_ESMC_MASK_FRACTION,
) -> torch.Tensor:
    del lm_model, score_mask, batch_size, n_passes, mask_fraction
    return binder_design.square().mean(dim=(1, 2))


def _run_seeded_workflow() -> tuple[
    list[str], dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]
]:
    return binder.design_binder(
        inversion_models={APPROVED_CRITIC: FakeCritic()},
        critic_models={APPROVED_CRITIC: FakeCritic()},
        lm_model=object(),
        target_name=None,
        target_sequence="ACD",
        binder_name=None,
        binder_sequence="###",
        is_antibody=False,
        seed=17,
        batch_size=1,
        steps=1,
    )


@pytest.mark.feature
@pytest.mark.gpu
def test_seeded_binder_workflow_is_short_and_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(binder, "fold_and_get_distogram", _fake_fold)
    monkeypatch.setattr(
        binder,
        "compute_fastplms_pseudoperplexity_nll",
        _fake_pseudoperplexity,
    )

    first_sequences, first_trajectory, first_rows = _run_seeded_workflow()
    second_sequences, second_trajectory, second_rows = _run_seeded_workflow()

    assert first_sequences == second_sequences == ["ACD|AAA"]
    assert list(first_trajectory) == list(second_trajectory) == [0]
    torch.testing.assert_close(
        first_trajectory[0]["total_loss"],
        second_trajectory[0]["total_loss"],
        rtol=0.0,
        atol=0.0,
    )
    assert len(first_rows) == len(second_rows) == 1
    assert first_rows[0]["critic_name"] == APPROVED_CRITIC
    assert first_rows[0]["designed_sequence"] == "ACD|AAA"
    assert first_rows[0]["binder_length"] == 3
    assert first_rows[0]["target_length"] == 3
    assert first_rows[0]["iptm"] == second_rows[0]["iptm"] == 1.0
