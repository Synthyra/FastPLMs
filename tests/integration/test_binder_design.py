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
        device="cpu",
    )


def test_public_binder_runner_propagates_loaded_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LoadedModel:
        def to(self, **kwargs: Any) -> "LoadedModel":
            del kwargs
            return self

        def eval(self) -> "LoadedModel":
            return self

        def requires_grad_(self, value: bool) -> "LoadedModel":
            del value
            return self

    monkeypatch.setattr(
        binder,
        "_load_fold_model",
        lambda *args, **kwargs: LoadedModel(),
    )
    monkeypatch.setattr(
        binder.AutoModelForMaskedLM,
        "from_pretrained",
        lambda *args, **kwargs: LoadedModel(),
    )
    observed: dict[str, Any] = {}

    def fake_design_binder(*args: Any, **kwargs: Any) -> tuple[list, dict, list]:
        del args
        observed.update(kwargs)
        return [], {}, []

    monkeypatch.setattr(binder, "design_binder", fake_design_binder)
    runner = binder.FastPLMsBinderDesign()
    runner.load(device="cpu")
    runner.design()

    assert runner.device == torch.device("cpu")
    assert observed["device"] == torch.device("cpu")


@pytest.mark.feature
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


@pytest.mark.feature
def test_selected_sequence_loss_and_logits_share_the_same_optimization_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    optimization_designs: list[torch.Tensor] = []

    def ranked_fake_fold(
        model: Any,
        target_seq: str,
        target_one_hot: torch.Tensor,
        design: torch.Tensor,
        num_loops: int = 0,
        num_sampling_steps: int = 1,
        calculate_confidence: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        result = _fake_fold(
            model,
            target_seq,
            target_one_hot,
            design,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            calculate_confidence=calculate_confidence,
            seed=seed,
        )
        if num_sampling_steps != 200:
            optimization_step = len(optimization_designs)
            optimization_designs.append(design.detach().clone())
            residue = "A" if optimization_step == 0 else "D"
            result["seq_list"] = [f"{target_seq}|{residue * design.size(1)}"]
            result["iptm"] = torch.full(
                (design.size(0),),
                0.95 if optimization_step == 0 else 0.20,
                device=design.device,
            )
        return result

    monkeypatch.setattr(binder, "fold_and_get_distogram", ranked_fake_fold)
    monkeypatch.setattr(
        binder,
        "compute_fastplms_pseudoperplexity_nll",
        _fake_pseudoperplexity,
    )
    monkeypatch.setattr(binder, "_write_results_table", lambda path, rows: None)
    monkeypatch.setattr(
        binder,
        "_write_official_selection_table",
        lambda path, rows, required_hero_critics=None: None,
    )

    sequences, trajectory, rows = binder.design_binder(
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
        steps=2,
        output_dir=tmp_path,
        device="cpu",
    )

    assert sequences == ["ACD|AAA"]
    assert rows[0]["designed_sequence"] == "ACD|AAA"
    assert rows[0]["selected_step"] == 0
    assert rows[0]["final_loss"] == float(trajectory[0]["total_loss"][0].item())

    selected_logits = torch.load(rows[0]["logits_path"], weights_only=True)
    first_step_temperature = binder.DEFAULT_TEMPERATURE_MIN + (
        1 - binder.DEFAULT_TEMPERATURE_MIN
    ) * 0.5
    torch.testing.assert_close(
        torch.softmax(selected_logits / first_step_temperature, dim=-1),
        optimization_designs[0][0],
        rtol=0.0,
        atol=0.0,
    )


def test_consensus_requires_every_named_hero_critic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pandas")
    monkeypatch.setattr(
        binder,
        "_compute_isoelectric_points",
        lambda sequences: [5.0] * len(sequences),
    )
    critic_a = "critic-a"
    critic_b = "critic-b"
    rows = [
        {
            "critic_name": critic_a,
            "designed_sequence": "ACD|AAA",
            "is_antibody": False,
            "iptm": 0.95,
        },
        {
            "critic_name": critic_b,
            "designed_sequence": "ACD|AAA",
            "is_antibody": False,
            "iptm": 0.96,
        },
        {
            "critic_name": critic_a,
            "designed_sequence": "ACD|DDD",
            "is_antibody": False,
            "iptm": 0.99,
        },
    ]

    selected = binder.select_official_designs(
        rows,
        top_k=2,
        required_hero_critics=(critic_a, critic_b),
    ).set_index("designed_sequence")

    complete = selected.loc["ACD|AAA"]
    assert complete["hero_critic_count"] == 2
    assert complete["required_hero_critic_count"] == 2
    assert bool(complete["all_hero_critics_pass"])

    incomplete = selected.loc["ACD|DDD"]
    assert incomplete["hero_critic_count"] == 1
    assert incomplete["required_hero_critic_count"] == 2
    assert not bool(incomplete["all_hero_critics_pass"])
