"""Tests for the FastPLMs binder design tutorial."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

import cookbook.tutorials.binder_design_fastplms as binder


@dataclass
class TinyTransformerOutput:
    last_hidden_state: torch.Tensor


class TinyTransformer(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> TinyTransformerOutput:
        del attention_mask, output_hidden_states, output_attentions
        return TinyTransformerOutput(last_hidden_state=x)


class TinyTokenizer:
    cls_token_id = 0
    eos_token_id = 1
    mask_token_id = 2


class TinyConfig:
    vocab_size = 24


class TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyConfig()
        self.tokenizer = TinyTokenizer()
        self.embed = torch.nn.Embedding(self.config.vocab_size, 8)
        self.transformer = TinyTransformer()
        self.sequence_head = torch.nn.Linear(8, self.config.vocab_size)


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


class FakeCritic:
    input_builder = FakeInputBuilder()

    def result_to_cif(self, complex_result: dict[str, Any]) -> str:
        return f"data_{complex_result['complex_id']}\n"

    def result_to_pdb(self, complex_result: dict[str, Any]) -> str:
        return "HEADER FASTPLMS TEST\nEND\n"


class FakeScalingCritic(FakeCritic):
    def __init__(self) -> None:
        self.device_moves: list[str] = []

    def to(self, device: torch.device | str) -> "FakeScalingCritic":
        self.device_moves.append(str(device))
        return self


@dataclass
class FakeProteinInput:
    id: str
    sequence: str
    msa: Any


@dataclass
class FakeStructurePredictionInput:
    sequences: list[FakeProteinInput]


class FakeInputTypes:
    ProteinInput = FakeProteinInput
    StructurePredictionInput = FakeStructurePredictionInput


class FakeFoldModel(torch.nn.Module):
    input_types = FakeInputTypes()
    device = torch.device("cpu")

    def __init__(self) -> None:
        super().__init__()
        self.saw_model_input_type = False

    def prepare_structure_input(
        self, input_data: FakeStructurePredictionInput, seed: int | None = None
    ) -> tuple[dict[str, torch.Tensor], list[Any]]:
        del seed
        assert isinstance(input_data, FakeStructurePredictionInput)
        self.saw_model_input_type = True
        features = {
            "dummy": torch.zeros(1, 1),
            "pocket_feature": torch.ones(1, 1),
        }
        return features, []

    def forward(
        self, dummy: torch.Tensor, res_type_soft: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        del dummy
        batch, total_length, _ = res_type_soft.shape
        distogram_logits = torch.zeros(batch, total_length, total_length, 128)
        return {"distogram_logits": distogram_logits}


def test_prompt_sampling_is_reproducible() -> None:
    factory = binder.BINDER_PROMPT_FACTORIES["trastuzumab_framework_vhvl"]

    first = factory.sample(seed=17)
    second = factory.sample(seed=17)
    third = factory.sample(seed=18)

    assert first == second
    assert first != third
    assert binder.MUTABLE_TOKEN in first
    assert 160 <= len(first) <= 260


def test_fold_uses_model_input_types() -> None:
    model = FakeFoldModel()
    target_one_hot = binder.sequence_to_one_hot("ACD", device="cpu")
    logits = torch.randn(1, 3, 20)
    design = torch.softmax(logits, dim=-1)

    result = binder.fold_and_get_distogram(
        model=model,
        target_seq="ACD",
        target_one_hot=target_one_hot,
        design=design,
        num_sampling_steps=1,
    )

    assert model.saw_model_input_type
    assert result["distogram_logits"].shape == (1, 6, 6, 128)


def test_design_input_validation_rejects_ambiguous_targets() -> None:
    with pytest.raises(AssertionError, match="Provide either target name"):
        binder.design_binder(
            inversion_models={},
            critic_models={},
            lm_model=object(),
            target_name="pd-l1",
            target_sequence="ACDE",
            binder_name="minibinder",
            binder_sequence=None,
            is_antibody=None,
            seed=0,
            steps=1,
        )


def test_design_input_validation_rejects_ambiguous_binders() -> None:
    with pytest.raises(AssertionError, match="Provide either binder name"):
        binder.design_binder(
            inversion_models={},
            critic_models={},
            lm_model=object(),
            target_name="pd-l1",
            target_sequence=None,
            binder_name="minibinder",
            binder_sequence="####",
            is_antibody=None,
            seed=0,
            steps=1,
        )


def test_fastplms_pseudoperplexity_nll_is_differentiable() -> None:
    torch.manual_seed(0)
    lm_model = TinyLM()
    logits = torch.randn(1, 5, 20, requires_grad=True)
    design = torch.softmax(logits, dim=-1)
    score_mask = torch.ones(1, 5, dtype=torch.bool)

    loss = binder.compute_fastplms_pseudoperplexity_nll(
        lm_model=lm_model,
        binder_design=design,
        score_mask=score_mask,
        batch_size=1,
        n_passes=2,
        mask_fraction=0.5,
    )
    loss.sum().backward()

    assert loss.shape == (1,)
    assert torch.isfinite(loss).all()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0


def test_antibody_proxy_uses_supplied_cdr_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cdr_lookup(binder_sequence: str) -> list[int]:
        del binder_sequence
        raise AssertionError("CDR lookup should not be called")

    monkeypatch.setattr(binder, "_cdr_indices", fail_cdr_lookup)
    distogram_logits = torch.zeros(1, 6, 6, 128)

    scores = binder.compute_distogram_iptm_proxy(
        distogram_logits=distogram_logits,
        target_length=3,
        binder_sequence="AAA",
        is_antibody=True,
        cdr_indices=[0, 2],
    )

    assert 0.0 <= scores["distogram_iptm_proxy"] <= 1.0
    assert 0.0 <= scores["cdr_distogram_iptm_proxy"] <= 1.0


def test_official_selection_uses_mean_hero_iptm_and_scaling_proxy() -> None:
    rows = [
        {
            "critic_name": "ESMFold2-Experimental-Fast",
            "designed_sequence": "AAA|DEDEDE",
            "is_antibody": False,
            "iptm": 0.9,
            "distogram_iptm_proxy": 0.1,
            "cdr_distogram_iptm_proxy": float("nan"),
            "batch_idx": 0,
        },
        {
            "critic_name": "ESMFold2-Experimental-Cutoff2025",
            "designed_sequence": "AAA|DEDEDE",
            "is_antibody": False,
            "iptm": 0.5,
            "distogram_iptm_proxy": 0.2,
            "cdr_distogram_iptm_proxy": float("nan"),
            "batch_idx": 0,
        },
        {
            "critic_name": "ESMFold2-Experimental-Fast-base300M-step250k",
            "designed_sequence": "AAA|DEDEDE",
            "is_antibody": False,
            "iptm": None,
            "distogram_iptm_proxy": 0.8,
            "cdr_distogram_iptm_proxy": float("nan"),
            "batch_idx": 0,
        },
        {
            "critic_name": "ESMFold2-Experimental-Fast",
            "designed_sequence": "AAA|EEEEEE",
            "is_antibody": False,
            "iptm": 0.95,
            "distogram_iptm_proxy": 0.1,
            "cdr_distogram_iptm_proxy": float("nan"),
            "batch_idx": 1,
        },
        {
            "critic_name": "ESMFold2-Experimental-Cutoff2025",
            "designed_sequence": "AAA|EEEEEE",
            "is_antibody": False,
            "iptm": 0.95,
            "distogram_iptm_proxy": 0.2,
            "cdr_distogram_iptm_proxy": float("nan"),
            "batch_idx": 1,
        },
        {
            "critic_name": "ESMFold2-Experimental-Fast-base300M-step250k",
            "designed_sequence": "AAA|EEEEEE",
            "is_antibody": False,
            "iptm": None,
            "distogram_iptm_proxy": 0.2,
            "cdr_distogram_iptm_proxy": float("nan"),
            "batch_idx": 1,
        },
    ]

    selection = binder.select_official_designs(rows)

    assert selection.iloc[0]["designed_sequence"] == "AAA|DEDEDE"
    assert selection.iloc[0]["iptm_score"] == pytest.approx(0.7)
    assert selection.iloc[0]["iptm_proxy_score"] == pytest.approx(0.8)
    assert selection.iloc[0]["selection_score"] == pytest.approx(0.75)
    assert selection.iloc[0]["hero_iptm_min"] == pytest.approx(0.5)
    assert not bool(selection.iloc[0]["all_hero_critics_pass"])


def test_official_selection_filters_high_pi_minibinders() -> None:
    rows = [
        {
            "critic_name": "ESMFold2-Experimental-Fast",
            "designed_sequence": "AAA|KKKKKK",
            "is_antibody": False,
            "iptm": 0.99,
            "distogram_iptm_proxy": 0.99,
            "cdr_distogram_iptm_proxy": float("nan"),
            "batch_idx": 0,
        },
        {
            "critic_name": "ESMFold2-Experimental-Fast",
            "designed_sequence": "AAA|DEDEDE",
            "is_antibody": False,
            "iptm": 0.5,
            "distogram_iptm_proxy": 0.5,
            "cdr_distogram_iptm_proxy": float("nan"),
            "batch_idx": 1,
        },
    ]

    selection = binder.select_official_designs(rows)

    assert selection["designed_sequence"].tolist() == ["AAA|DEDEDE"]


def test_official_selection_empty_after_pi_filter_keeps_schema() -> None:
    rows = [
        {
            "critic_name": "ESMFold2-Experimental-Fast",
            "designed_sequence": "AAA|KKKKKK",
            "is_antibody": False,
            "iptm": 0.99,
            "distogram_iptm_proxy": 0.99,
            "cdr_distogram_iptm_proxy": float("nan"),
        }
    ]

    selection = binder.select_official_designs(rows)

    assert selection.empty
    assert "selection_score" in selection.columns
    assert "hero_iptm_min" in selection.columns
    assert "all_hero_critics_pass" in selection.columns


def test_official_selection_uses_cdr_proxy_for_scaling_antibodies() -> None:
    rows = [
        {
            "critic_name": "ESMFold2-Experimental-Fast",
            "designed_sequence": "AAA|EVQLVESGGG",
            "is_antibody": True,
            "iptm": 0.6,
            "distogram_iptm_proxy": 0.1,
            "cdr_distogram_iptm_proxy": 0.2,
            "batch_idx": 0,
        },
        {
            "critic_name": "ESMFold2-Experimental-Fast-base300M-step250k",
            "designed_sequence": "AAA|EVQLVESGGG",
            "is_antibody": True,
            "iptm": None,
            "distogram_iptm_proxy": 0.1,
            "cdr_distogram_iptm_proxy": 0.9,
            "batch_idx": 0,
        },
    ]

    selection = binder.select_official_designs(rows)

    assert selection.iloc[0]["iptm_score"] == pytest.approx(0.6)
    assert selection.iloc[0]["iptm_proxy_score"] == pytest.approx(0.9)
    assert selection.iloc[0]["selection_score"] == pytest.approx(0.75)


def test_official_selection_flags_all_hero_critics_pass() -> None:
    rows = [
        {
            "critic_name": "ESMFold2-Experimental-Fast",
            "designed_sequence": "AAA|DEDEDE",
            "is_antibody": False,
            "iptm": 0.91,
            "distogram_iptm_proxy": 0.4,
            "cdr_distogram_iptm_proxy": float("nan"),
            "batch_idx": 0,
        },
        {
            "critic_name": "ESMFold2-Experimental-Fast-Cutoff2025",
            "designed_sequence": "AAA|DEDEDE",
            "is_antibody": False,
            "iptm": 0.92,
            "distogram_iptm_proxy": 0.4,
            "cdr_distogram_iptm_proxy": float("nan"),
            "batch_idx": 0,
        },
        {
            "critic_name": "ESMFold2-Experimental",
            "designed_sequence": "AAA|DEDEDE",
            "is_antibody": False,
            "iptm": 0.93,
            "distogram_iptm_proxy": 0.4,
            "cdr_distogram_iptm_proxy": float("nan"),
            "batch_idx": 0,
        },
        {
            "critic_name": "ESMFold2-Experimental-Cutoff2025",
            "designed_sequence": "AAA|DEDEDE",
            "is_antibody": False,
            "iptm": 0.94,
            "distogram_iptm_proxy": 0.4,
            "cdr_distogram_iptm_proxy": float("nan"),
            "batch_idx": 0,
        },
    ]

    selection = binder.select_official_designs(rows)

    assert bool(selection.iloc[0]["all_hero_critics_pass"])
    assert selection.iloc[0]["consensus_iptm_threshold"] == pytest.approx(0.9)


@pytest.mark.gpu
@pytest.mark.slow
def test_fastplms_esmplusplus_small_pseudoperplexity_smoke() -> None:
    from transformers import AutoModelForMaskedLM

    model = (
        AutoModelForMaskedLM.from_pretrained(
            "Synthyra/ESMplusplus_small",
            trust_remote_code=True,
            dtype=torch.float32,
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    logits = torch.randn(1, 4, 20, device="cuda", requires_grad=True)
    design = torch.softmax(logits, dim=-1)
    score_mask = torch.ones(1, 4, dtype=torch.bool, device="cuda")

    loss = binder.compute_fastplms_pseudoperplexity_nll(
        lm_model=model,
        binder_design=design,
        score_mask=score_mask,
        batch_size=1,
        n_passes=1,
        mask_fraction=0.5,
    )
    loss.sum().backward()

    assert loss.shape == (1,)
    assert torch.isfinite(loss).all()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0

    del model
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_tiny_design_dry_run_writes_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_fold_and_get_distogram(
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
        batch, binder_length, aa_dim = design.shape
        target_length = target_one_hot.size(1)
        total_length = target_length + binder_length
        aa_weight = torch.linspace(-1.0, 1.0, aa_dim, device=design.device)
        binder_signal = (design * aa_weight).sum(dim=-1)
        token_signal = torch.cat(
            (torch.zeros(batch, target_length, device=design.device), binder_signal),
            dim=1,
        )
        pair_signal = token_signal[:, :, None] + token_signal[:, None, :]
        bin_basis = torch.linspace(-1.0, 1.0, 128, device=design.device)
        distogram_logits = pair_signal.unsqueeze(-1) * bin_basis
        seq_list = [target_seq + "|" + "A" * binder_length for _ in range(batch)]
        return {
            "distogram_logits": distogram_logits,
            "inputs": {},
            "chain_info_list": [[] for _ in range(batch)],
            "output": {"distogram_logits": distogram_logits},
            "seq_list": seq_list,
            "iptm": torch.ones(batch, device=design.device),
            "plddt": torch.ones(batch, 1, device=design.device),
        }

    def fake_pseudoperplexity_nll(
        lm_model: Any,
        binder_design: torch.Tensor,
        score_mask: torch.Tensor,
        batch_size: int = 4,
        n_passes: int = 4,
        mask_fraction: float = binder.DEFAULT_ESMC_MASK_FRACTION,
    ) -> torch.Tensor:
        del lm_model, score_mask, batch_size, n_passes, mask_fraction
        return binder_design.square().mean(dim=(1, 2))

    monkeypatch.setattr(binder, "fold_and_get_distogram", fake_fold_and_get_distogram)
    monkeypatch.setattr(
        binder, "compute_fastplms_pseudoperplexity_nll", fake_pseudoperplexity_nll
    )

    scaling_critic = FakeScalingCritic()
    best_sequences, trajectory, rows = binder.design_binder(
        inversion_models={"fake_inversion": object()},
        critic_models={
            "fake_critic": FakeCritic(),
            "ESMFold2-Experimental-Fast-base300M-step250k": scaling_critic,
        },
        lm_model=object(),
        target_name=None,
        target_sequence="ACD",
        binder_name=None,
        binder_sequence="###",
        is_antibody=False,
        seed=0,
        batch_size=1,
        steps=1,
        output_dir=tmp_path,
    )

    assert best_sequences == ["ACD|AAA"]
    assert list(trajectory) == [0]
    assert len(rows) == 2
    assert rows[0]["binder_sequence"] == "AAA"
    assert rows[0]["binder_length"] == 3
    assert rows[0]["target_length"] == 3
    assert rows[0]["mean_plddt"] == 1.0
    assert scaling_critic.device_moves == ["cuda", "cpu"]
    assert (tmp_path / "trajectory.jsonl").exists()
    assert (tmp_path / "best_sequences.fasta").exists()
    assert (tmp_path / "results.parquet").exists()
    assert (tmp_path / "selection.parquet").exists()
