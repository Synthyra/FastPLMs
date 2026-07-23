"""Seeded feature smoke test for the FastPLMs binder-design workflow."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from types import SimpleNamespace
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
        def to(self, **kwargs: Any) -> LoadedModel:
            del kwargs
            return self

        def eval(self) -> LoadedModel:
            return self

        def requires_grad_(self, value: bool) -> LoadedModel:
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


def test_fold_loader_pins_revision_and_propagates_local_files_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    class LoadedModel:
        config = SimpleNamespace(esmc_id="Synthyra/ESMplusplus_6B")
        _esmc = None
        _esmc_fp8 = False
        _esmc_fp8_module_paths: tuple[str, ...] = ()
        _esmc_source = "Synthyra/ESMplusplus_6B"
        _esmc_source_revision = "b" * 40
        _esmc_local_files_only = False
        _esmc_precision_policy = "auto"
        _esmc_precision_status = SimpleNamespace(resolved="bf16")

        def __init__(self) -> None:
            self._esmc_source_files: dict[str, str] = {}

        def load_esmc(self, source: str, **kwargs: Any) -> None:
            observed["esmc_load"] = (source, kwargs)
            self._esmc = object()
            self._esmc_local_files_only = bool(kwargs.get("local_files_only", False))

        def configure_lm_dropout(self, *args: Any, **kwargs: Any) -> None:
            observed["dropout"] = (args, kwargs)

        def to(self, **kwargs: Any) -> LoadedModel:
            observed["to"] = kwargs
            return self

        def eval(self) -> LoadedModel:
            return self

        def requires_grad_(self, value: bool) -> LoadedModel:
            observed["requires_grad"] = value
            return self

    def fake_from_pretrained(source: str, **kwargs: Any) -> LoadedModel:
        observed["source"] = source
        observed["load_kwargs"] = kwargs
        return LoadedModel()

    monkeypatch.setattr(binder.AutoModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(binder, "_ESMC_CACHE", None)
    monkeypatch.setattr(binder, "_ESMC_CACHE_KEY", None)
    monkeypatch.setattr(binder, "_ESMC_CACHE_CONTEXT", {})
    revision = "a" * 40
    model = binder._load_fold_model(
        "Custom/Fold",
        revision=revision,
        lm_dropout=0.5,
        cache_esmc=True,
        device="cpu",
        kernel_backend=None,
        compile_model=False,
        local_files_only=True,
    )

    assert observed["source"] == "Custom/Fold"
    assert observed["load_kwargs"]["revision"] == revision
    assert observed["load_kwargs"]["local_files_only"] is True
    assert observed["load_kwargs"]["trust_remote_code"] is True
    assert observed["esmc_load"] == (
        "Synthyra/ESMplusplus_6B",
        {"device": "cpu", "local_files_only": True},
    )
    assert model._fastplms_binder_load_identity == {
        "repo_id": "Custom/Fold",
        "requested_revision": revision,
        "local_files_only": True,
    }


def test_binder_runner_uses_registered_commits_and_owns_offline_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fold_calls: list[tuple[str, dict[str, Any]]] = []
    lm_calls: list[tuple[str, dict[str, Any]]] = []

    class LoadedModel:
        def to(self, **_kwargs: Any) -> LoadedModel:
            return self

        def eval(self) -> LoadedModel:
            return self

        def requires_grad_(self, _value: bool) -> LoadedModel:
            return self

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(
        binder,
        "_load_fold_model",
        lambda model_name, **kwargs: fold_calls.append((model_name, kwargs)) or LoadedModel(),
    )
    monkeypatch.setattr(
        binder.AutoModelForMaskedLM,
        "from_pretrained",
        lambda source, **kwargs: lm_calls.append((source, kwargs)) or LoadedModel(),
    )

    runner = binder.FastPLMsBinderDesign()
    runner.load(device="cpu", local_files_only=True)

    registered = binder._registered_fast_revisions()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert len(fold_calls) == 3
    for model_name, kwargs in fold_calls:
        repo_id = binder._repo_name(model_name)
        assert kwargs["revision"] == registered[repo_id]
        assert kwargs["local_files_only"] is True
    assert lm_calls == [
        (
            "Synthyra/ESMplusplus_6B",
            {
                "revision": registered["Synthyra/ESMplusplus_6B"],
                "local_files_only": True,
                "trust_remote_code": True,
                "dtype": torch.float32,
            },
        )
    ]
    assert runner.lm_model._fastplms_binder_load_identity == {
        "repo_id": "Synthyra/ESMplusplus_6B",
        "requested_revision": registered["Synthyra/ESMplusplus_6B"],
        "local_files_only": True,
    }


def test_custom_binder_model_requires_immutable_revision() -> None:
    runner = binder.FastPLMsBinderDesign()
    with pytest.raises(ValueError, match="requires an immutable revision"):
        runner.load(
            device="cpu",
            inversion_model_names=("Custom/unpinned",),
        )


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


def test_prompt_sampling_preserves_the_callers_random_stream() -> None:
    factory = binder.BINDER_PROMPT_FACTORIES["minibinder"]
    random.seed(1234)
    expected = random.random()
    random.seed(1234)

    first = factory.sample(seed=17)
    observed = random.random()

    assert first == factory.sample(seed=17)
    assert observed == expected


def test_atom_batch_padding_uses_the_largest_prepared_table_without_truncation() -> None:
    small = {
        "ref_pos": torch.arange(32 * 3, dtype=torch.float32).reshape(1, 32, 3),
        "atom_attention_mask": torch.ones((1, 32), dtype=torch.bool),
    }
    large = {
        "ref_pos": torch.arange(65 * 3, dtype=torch.float32).reshape(1, 65, 3),
        "atom_attention_mask": torch.ones((1, 65), dtype=torch.bool),
    }

    padded = binder._pad_prepared_atom_features([(small, ["small"]), (large, ["large"])])

    assert padded[0][0]["ref_pos"].shape == (1, 96, 3)
    assert padded[1][0]["ref_pos"].shape == (1, 96, 3)
    torch.testing.assert_close(padded[1][0]["ref_pos"][:, :65], large["ref_pos"])
    assert not padded[1][0]["atom_attention_mask"][:, 65:].any()
    with pytest.raises(ValueError, match="Refusing to truncate atom features"):
        binder._resize_tensor(large["ref_pos"], dim=1, size=64)


def test_binder_output_directory_is_exclusive_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_dir = tmp_path / "binder-run"
    output_dir.mkdir()
    (output_dir / "stale-result.txt").write_text("old run\n", encoding="utf-8")
    fold_called = False

    def forbidden_fold(*args: Any, **kwargs: Any) -> Any:
        nonlocal fold_called
        del args, kwargs
        fold_called = True
        raise AssertionError("stale output reached model execution")

    monkeypatch.setattr(binder, "fold_and_get_distogram", forbidden_fold)
    with pytest.raises(FileExistsError, match="never reused"):
        binder.design_binder(
            inversion_models={APPROVED_CRITIC: FakeCritic()},
            critic_models={APPROVED_CRITIC: FakeCritic()},
            lm_model=object(),
            target_name=None,
            target_sequence="ACD",
            binder_name=None,
            binder_sequence="###",
            is_antibody=False,
            seed=17,
            steps=1,
            output_dir=output_dir,
            device="cpu",
        )

    assert not fold_called
    assert (output_dir / "stale-result.txt").read_text(encoding="utf-8") == "old run\n"


def test_interrupted_binder_run_cannot_be_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_dir = tmp_path / "interrupted-run"

    def failing_fold(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("injected fold failure")

    monkeypatch.setattr(binder, "fold_and_get_distogram", failing_fold)
    kwargs = {
        "inversion_models": {APPROVED_CRITIC: FakeCritic()},
        "critic_models": {APPROVED_CRITIC: FakeCritic()},
        "lm_model": object(),
        "target_name": None,
        "target_sequence": "ACD",
        "binder_name": None,
        "binder_sequence": "###",
        "is_antibody": False,
        "seed": 17,
        "steps": 1,
        "output_dir": output_dir,
        "device": "cpu",
    }

    with pytest.raises(RuntimeError, match="injected fold failure"):
        binder.design_binder(**kwargs)
    assert output_dir.is_dir()
    assert not (output_dir / "run_manifest.json").exists()

    with pytest.raises(FileExistsError, match="never reused"):
        binder.design_binder(**kwargs)


def test_cli_rejects_stale_output_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_dir = tmp_path / "existing-run"
    output_dir.mkdir()
    loaded = False

    class ForbiddenRunner:
        def load(self, **kwargs: Any) -> None:
            nonlocal loaded
            del kwargs
            loaded = True

    monkeypatch.setattr(binder, "FastPLMsBinderDesign", ForbiddenRunner)
    args = binder.parse_args(["--steps", "1", "--output-dir", str(output_dir)])
    with pytest.raises(FileExistsError, match="never reused"):
        binder.run_local(args)

    assert not loaded


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

    output_dir = tmp_path / "binder-run"
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
        output_dir=output_dir,
        device="cpu",
    )

    assert sequences == ["ACD|AAA"]
    assert rows[0]["designed_sequence"] == "ACD|AAA"
    assert rows[0]["selected_step"] == 0
    assert rows[0]["final_loss"] == float(trajectory[0]["total_loss"][0].item())

    selected_logits = torch.load(rows[0]["logits_path"], weights_only=True)
    first_step_temperature = (
        binder.DEFAULT_TEMPERATURE_MIN + (1 - binder.DEFAULT_TEMPERATURE_MIN) * 0.5
    )
    torch.testing.assert_close(
        torch.softmax(selected_logits / first_step_temperature, dim=-1),
        optimization_designs[0][0],
        rtol=0.0,
        atol=0.0,
    )
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["seed"] == 17
    assert manifest["steps"] == 2
    assert len(manifest["target_sequence_sha256"]) == 64
    assert len(manifest["binder_prompt_sha256"]) == 64
    assert manifest["configuration"]["is_antibody"] is False
    assert manifest["configuration"]["loss_weights"] == binder.LOSS_WEIGHTS
    assert manifest["command"]
    inversion_identity = manifest["models"]["inversion"][0]
    assert inversion_identity["requested"] == APPROVED_CRITIC
    assert {
        "requested_revision",
        "hub_revision",
        "weights_revision",
        "runtime_revision",
        "local_files_only",
    }.issubset(inversion_identity)
    assert {
        "hf_hub_offline",
        "transformers_offline",
    }.issubset(manifest["environment"])
    assert manifest["tokenizer"] is None


def test_binder_model_identity_records_selected_kernel_and_mixed_parameter_dtypes() -> None:
    class Config:
        _name_or_path = "Synthyra/ESMFold2-test"
        _commit_hash = "0123456789abcdef"
        fastplms_weights_revision = "1" * 40
        fastplms_runtime_revision = "source-tree-sha256:" + "2" * 64
        esmc_attn_backend = "sdpa"

        def to_dict(self) -> dict[str, object]:
            return {"model_type": "esmfold2", "d_pair": 8}

    class PrecisionStatus:
        def as_dict(self) -> dict[str, object]:
            return {
                "requested": "auto",
                "resolved": "bf16",
                "reason": "test contract",
                "device": "cuda:0",
                "transformer_engine_version": None,
            }

    class MixedDtypeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.float_weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
            self.bfloat_weight = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
            self.config = Config()
            self._kernel_backend = "cuequivariance"
            self.esmc_precision_status = PrecisionStatus()

    model = MixedDtypeModel()
    binder._record_model_load_identity(
        model,
        repo_id="Synthyra/ESMFold2-test",
        revision="3" * 40,
        local_files_only=True,
    )
    identity = binder._model_identity("requested-model", model)

    assert identity["kernel_backend"] == "cuequivariance"
    assert identity["repo_id"] == "Synthyra/ESMFold2-test"
    assert identity["requested_revision"] == "3" * 40
    assert identity["hub_revision"] == "0123456789abcdef"
    assert identity["weights_revision"] == "1" * 40
    assert identity["runtime_revision"] == "source-tree-sha256:" + "2" * 64
    assert identity["local_files_only"] is True
    assert identity["parameter_dtype"] == "mixed[torch.bfloat16,torch.float32]"
    assert identity["parameter_dtypes"] == ["torch.bfloat16", "torch.float32"]
    assert identity["parameter_dtype_numel"] == {
        "torch.bfloat16": 3,
        "torch.float32": 2,
    }
    assert identity["effective_precision"]["resolved"] == "bf16"

    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"<pad>": 0, "A": 1},
        init_kwargs={},
        name_or_path=None,
    )
    tokenizer_identity = binder._tokenizer_identity(tokenizer, model=model)
    assert tokenizer_identity is not None
    assert tokenizer_identity["repo_id"] == "Synthyra/ESMFold2-test"
    assert tokenizer_identity["requested_revision"] == "3" * 40
    assert tokenizer_identity["weights_revision"] == "1" * 40
    assert tokenizer_identity["runtime_revision"] == "source-tree-sha256:" + "2" * 64
    assert tokenizer_identity["local_files_only"] is True


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
            "distogram_iptm_proxy": 0.80,
        },
        {
            "critic_name": critic_b,
            "designed_sequence": "ACD|AAA",
            "is_antibody": False,
            "iptm": 0.96,
            "distogram_iptm_proxy": 0.90,
        },
        {
            "critic_name": critic_a,
            "designed_sequence": "ACD|DDD",
            "is_antibody": False,
            "iptm": 0.99,
            "distogram_iptm_proxy": 0.70,
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
    assert complete["iptm_proxy_score"] == pytest.approx(0.85)
    assert bool(complete["all_hero_critics_pass"])

    incomplete = selected.loc["ACD|DDD"]
    assert incomplete["hero_critic_count"] == 1
    assert incomplete["required_hero_critic_count"] == 2
    assert not bool(incomplete["all_hero_critics_pass"])


def test_successful_nonempty_binder_run_reaches_selection_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    pytest.importorskip("pandas")
    monkeypatch.setattr(
        binder,
        "_compute_isoelectric_points",
        lambda sequences: [5.0] * len(sequences),
    )
    rows = [
        {
            "critic_name": "critic-a",
            "designed_sequence": "ACD|AAA",
            "is_antibody": False,
            "iptm": 0.95,
            "distogram_iptm_proxy": 0.85,
        }
    ]

    class FakeRunner:
        def load(self, **kwargs: Any) -> None:
            del kwargs

        def design(self, **kwargs: Any) -> tuple[list[str], None, list[dict[str, Any]]]:
            del kwargs
            return ["AAA"], None, rows

    messages: list[tuple[str, tuple[Any, ...]]] = []
    monkeypatch.setattr(binder, "FastPLMsBinderDesign", FakeRunner)
    monkeypatch.setattr(
        binder.logger,
        "info",
        lambda message, *args: messages.append((message, args)),
    )

    binder.run_local(
        binder.parse_args(["--steps", "1", "--output-dir", str(tmp_path / "binder-summary")])
    )

    summary = [entry for entry in messages if entry[0].startswith("Top official selection")]
    assert len(summary) == 1
    assert summary[0][1][2] == pytest.approx(0.85)
