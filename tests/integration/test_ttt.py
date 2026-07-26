from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from transformers import PretrainedConfig, PreTrainedModel

from fastplms.models.ankh.modeling_ankh import FastAnkhForMaskedLMExtension
from fastplms.models.dplm.modeling_dplm import DPLMForMaskedLM
from fastplms.models.dplm2.modeling_dplm2 import DPLM2ForMaskedLM
from fastplms.models.e1.modeling_e1 import E1ForMaskedLM
from fastplms.models.esm2.modeling_fastesm import FastEsmForMaskedLM
from fastplms.models.esm3.modeling_esm3 import FastESM3Model
from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusForMaskedLM
from fastplms.models.esmfold.modeling_fast_esmfold import FastEsmForProteinFolding
from fastplms.models.ttt import (
    FastPLMTestTimeTrainingMixin,
    LoraInjectedLinear,
    TTTConfig,
)
from tests.conftest import MODEL_REGISTRY, STRUCTURE_MODEL_REGISTRY


TEST_SEQUENCE = "MSTNPKPQRKTKRNT"
LOCAL_MODEL_CLASSES = {
    "esm2": FastEsmForMaskedLM,
    "esmc": ESMplusplusForMaskedLM,
    "esm3": FastESM3Model,
    "e1": E1ForMaskedLM,
    "dplm": DPLMForMaskedLM,
    "dplm2": DPLM2ForMaskedLM,
    "ankh": FastAnkhForMaskedLMExtension,
}


@pytest.mark.parametrize("method", ["ttt", "ttt_reset", "fold_protein_ttt"])
def test_esmfold_ttt_entry_points_reject_without_an_untrained_head(method: str) -> None:
    model = object.__new__(FastEsmForProteinFolding)
    kwargs = {"seq": "ACDE"} if method == "ttt" else {}
    if method == "fold_protein_ttt":
        kwargs = {"sequence": "ACDE"}

    with pytest.raises(RuntimeError, match="does not contain a trained masked-language-model head"):
        getattr(model, method)(**kwargs)


def test_esmfold_fold_protein_ttt_flag_rejects_before_folding() -> None:
    model = object.__new__(FastEsmForProteinFolding)

    with pytest.raises(RuntimeError, match="does not contain a trained masked-language-model head"):
        model.fold_protein("ACDE", ttt=True)


class DummyConfig:
    vocab_size = 8


class DummyTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    eos_token_id = 2
    mask_token_id = 3
    all_special_ids: ClassVar[list[int]] = [0, 1, 2, 3]

    def __init__(self) -> None:
        self.vocab = {
            "A": 4,
            "C": 5,
            "D": 6,
            "E": 7,
        }

    def __call__(
        self,
        seq: str | list[str],
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, padding
        sequences = [seq] if isinstance(seq, str) else seq
        encoded = []
        for sequence in sequences:
            encoded.append(
                [self.cls_token_id] + [self.vocab[aa] for aa in sequence] + [self.eos_token_id]
            )
        max_len = max(len(ids) for ids in encoded)
        # input_ids: (len(encoded), max_len)
        input_ids = torch.full((len(encoded), max_len), self.pad_token_id)
        for row, ids in enumerate(encoded):
            # input_ids[row, :len(ids)]: (...)
            input_ids[row, : len(ids)] = torch.tensor(ids)
        return {"input_ids": input_ids.long()}


class DummyTTTModel(FastPLMTestTimeTrainingMixin, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.config = DummyConfig()
        self.tokenizer = DummyTokenizer()
        self.embed = nn.Embedding(self.config.vocab_size, 8)
        self.backbone = nn.Sequential(
            nn.Linear(8, 8),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, 8),
        )
        self.lm_head = nn.Linear(8, self.config.vocab_size)
        self.init_ttt(
            {
                "steps": 1,
                "ags": 1,
                "batch_size": 1,
                "mask_ratio": 1.0,
                "bert_leave_prob": 0.0,
                "bert_replace_prob": 0.0,
                "lora_rank": 2,
                "lora_alpha": 1.0,
            }
        )

    def _ttt_get_trainable_modules(self) -> list[nn.Module]:
        return [self.backbone]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        # input_ids: (b, l)
        del attention_mask
        hidden = self.backbone(self.embed(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


class FamilyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(8, 8)
        self.value = nn.Linear(8, 8)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (..., d)
        return self.query(hidden_states) + self.value(hidden_states)


class DummyFamilyTargetTTTModel(FastPLMTestTimeTrainingMixin, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.config = DummyConfig()
        self.tokenizer = DummyTokenizer()
        self.embed = nn.Embedding(self.config.vocab_size, 8)
        self.backbone = nn.ModuleDict(
            {
                "attention": FamilyAttention(),
                "feed_forward": nn.Linear(8, 8),
            }
        )
        self.lm_head = nn.Linear(8, self.config.vocab_size)
        self.init_ttt({"lora_target_replace_module": "FamilyAttention"})

    def _ttt_get_trainable_modules(self) -> list[nn.Module]:
        return [self.backbone]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        # input_ids: (b, l)
        del attention_mask
        hidden = self.embed(input_ids)
        hidden = self.backbone["attention"](hidden)
        hidden = self.backbone["feed_forward"](hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


class DummyPretrainedTTTConfig(PretrainedConfig):
    model_type = "dummy_pretrained_ttt"

    def __init__(self, vocab_size: int = 8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size


class DummyPretrainedTTTModel(FastPLMTestTimeTrainingMixin, PreTrainedModel):
    config_class = DummyPretrainedTTTConfig

    def __init__(self, config: DummyPretrainedTTTConfig) -> None:
        PreTrainedModel.__init__(self, config)
        self.tokenizer = DummyTokenizer()
        self.embed = nn.Embedding(config.vocab_size, 8)
        self.backbone = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
        self.lm_head = nn.Linear(8, config.vocab_size)
        self.post_init()
        self.init_ttt(
            {
                "seed": 17,
                "lora_rank": 2,
                "lora_alpha": 1.0,
                "lora_target_modules": ("0", "2"),
            }
        )

    def _ttt_get_trainable_modules(self) -> list[nn.Module]:
        return [self.backbone]

    def forward(self, input_ids: torch.Tensor, **kwargs):
        # input_ids: (b, l)
        del kwargs
        hidden = self.backbone(self.embed(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_ttt_masking_masks_only_residue_tokens() -> None:
    model = DummyTTTModel()
    tokenized = model._ttt_tokenize(seq="ACDE")
    generator = torch.Generator()
    generator.manual_seed(0)

    batch, labels = model._ttt_sample_batch(tokenized, generator)

    assert isinstance(batch, torch.Tensor)
    assert labels[0, 0].item() == -100
    assert labels[0, -1].item() == -100
    assert torch.all(batch[labels != -100] == model.tokenizer.mask_token_id)


def test_ttt_lora_injection_is_lazy_and_backbone_scoped() -> None:
    model = DummyTTTModel()

    assert all("lora_" not in name for name in model.state_dict())

    model._ttt_ensure_initialized()

    assert any(isinstance(module, LoraInjectedLinear) for module in model.backbone.modules())
    assert not any(isinstance(module, LoraInjectedLinear) for module in model.lm_head.modules())


def test_ttt_lora_alpha_is_the_proteinttt_direct_multiplier() -> None:
    adapter = LoraInjectedLinear(
        nn.Linear(3, 2, bias=False),
        rank=2,
        alpha=6.0,
    )
    # inputs: (1, 3)
    inputs = torch.tensor([[1.0, 2.0, 3.0]])
    with torch.no_grad():
        adapter.linear.weight.zero_()
        adapter.lora_down.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            )
        )
        adapter.lora_up.weight.copy_(
            torch.tensor(
                [
                    [1.0, 1.0],
                    [2.0, -1.0],
                ]
            )
        )

    unscaled_delta = adapter.lora_up(adapter.lora_down(inputs))
    torch.testing.assert_close(adapter(inputs), unscaled_delta * 6.0)
    assert not torch.equal(adapter(inputs), unscaled_delta * (6.0 / adapter.rank))


def test_ttt_first_call_mapping_preserves_family_target_class() -> None:
    model = DummyFamilyTargetTTTModel()
    base_weights = {
        "query": model.backbone["attention"].query.weight.detach().clone(),
        "value": model.backbone["attention"].value.weight.detach().clone(),
        "feed_forward": model.backbone["feed_forward"].weight.detach().clone(),
        "lm_head": model.lm_head.weight.detach().clone(),
    }

    metrics = model.ttt(
        seq="ACDE",
        ttt_config={
            "steps": 1,
            "ags": 1,
            "batch_size": 1,
            "mask_ratio": 1.0,
            "bert_leave_prob": 0.0,
            "bert_replace_prob": 0.0,
            "lora_rank": 2,
            "lora_alpha": 1.0,
            "seed": 23,
        },
    )

    assert model.ttt_config.lora_target_replace_module == "FamilyAttention"
    adapter_names = [
        name for name, module in model.named_modules() if isinstance(module, LoraInjectedLinear)
    ]
    assert adapter_names == [
        "backbone.attention.query",
        "backbone.attention.value",
    ]
    assert len(metrics["losses"]) == 1
    torch.testing.assert_close(
        model.backbone["attention"].query.linear.weight,
        base_weights["query"],
    )
    torch.testing.assert_close(
        model.backbone["attention"].value.linear.weight,
        base_weights["value"],
    )
    torch.testing.assert_close(
        model.backbone["feed_forward"].weight,
        base_weights["feed_forward"],
    )
    torch.testing.assert_close(model.lm_head.weight, base_weights["lm_head"])
    assert any(
        not torch.equal(module.lora_up.weight, module._ttt_initial_lora_up)
        for module in model._ttt_lora_modules()
    )


def test_ttt_direct_init_mapping_preserves_family_target_class() -> None:
    model = DummyFamilyTargetTTTModel()

    model.init_ttt({"steps": 2, "seed": 29})
    model._ttt_ensure_initialized()

    assert model.ttt_config.steps == 2
    assert model.ttt_config.seed == 29
    assert model.ttt_config.lora_target_replace_module == "FamilyAttention"
    assert [
        name for name, module in model.named_modules() if isinstance(module, LoraInjectedLinear)
    ] == [
        "backbone.attention.query",
        "backbone.attention.value",
    ]


def test_ttt_first_call_mapping_preserves_explicit_target_override() -> None:
    model = DummyFamilyTargetTTTModel()

    model.ttt(
        seq="ACDE",
        ttt_config={
            "steps": 1,
            "ags": 1,
            "batch_size": 1,
            "mask_ratio": 1.0,
            "bert_leave_prob": 0.0,
            "bert_replace_prob": 0.0,
            "lora_rank": 2,
            "lora_alpha": 1.0,
            "lora_target_replace_module": None,
            "lora_target_modules": ("feed_forward",),
        },
    )

    assert model.ttt_config.lora_target_replace_module is None
    assert model.ttt_config.lora_target_modules == ("feed_forward",)
    assert [
        name for name, module in model.named_modules() if isinstance(module, LoraInjectedLinear)
    ] == ["backbone.feed_forward"]


@pytest.mark.parametrize(
    ("values", "exception"),
    (
        ({"lr": float("nan")}, ValueError),
        ({"lora_alpha": float("inf")}, ValueError),
        ({"momentum": "0.9"}, TypeError),
        ({"momentum": -0.1}, ValueError),
        ({"weight_decay": float("-inf")}, ValueError),
        ({"weight_decay": -0.1}, ValueError),
        ({"seed": True}, TypeError),
        ({"seed": 1.5}, TypeError),
        ({"initial_state_reset": 1}, TypeError),
        ({"automatic_best_state_reset": None}, TypeError),
        ({"eval_each_step": 0}, TypeError),
        ({"gradient_clip": "false"}, TypeError),
        ({"lora_target_replace_module": ""}, ValueError),
        ({"lora_target_replace_module": 7}, TypeError),
        ({"lora_target_modules": []}, TypeError),
        ({"lora_target_modules": ()}, ValueError),
        ({"lora_target_modules": ("query", 7)}, TypeError),
        ({"lora_target_modules": ("query", "")}, ValueError),
        ({"lora_target_modules": ("query", "query")}, ValueError),
    ),
)
def test_ttt_config_rejects_invalid_optimizer_and_target_contracts(
    values: dict[str, object],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        TTTConfig(**values)


def test_ttt_adapter_initialization_is_seeded_and_preserves_ambient_rng() -> None:
    first = DummyTTTModel()
    torch.manual_seed(11)
    # expected_next: (4,)
    expected_next = torch.rand(4)
    torch.manual_seed(11)
    first._ttt_ensure_initialized()
    # actual_next: (4,)
    actual_next = torch.rand(4)

    torch.manual_seed(987654)
    second = DummyTTTModel()
    second._ttt_ensure_initialized()

    torch.testing.assert_close(actual_next, expected_next)
    first_state = first._ttt_snapshot_lora_state()
    second_state = second._ttt_snapshot_lora_state()
    for first_module, second_module in zip(first_state, second_state, strict=True):
        for name in first_module:
            assert torch.equal(first_module[name], second_module[name]), name


def test_ttt_generic_replacements_exclude_reserved_vocabulary_ids() -> None:
    model = DummyTTTModel()
    canonical = {aa: idx + 4 for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    tokenizer = DummyTokenizer()
    tokenizer.vocab = {
        **canonical,
        "<reserved_0>": 24,
        "<reserved_1>": 25,
        "<function>": 26,
    }
    model.tokenizer = tokenizer
    model.config = SimpleNamespace(vocab_size=27, model_type="esm3")
    # input_ids: (1, 6)
    input_ids = torch.tensor([[1, canonical["A"], 24, 25, 26, 2]])

    replacements = model._ttt_replacement_tokens(input_ids)
    trainable = model._ttt_non_special_mask(input_ids)

    assert replacements.tolist() == list(canonical.values())
    assert trainable.tolist() == [[False, True, False, False, False, False]]


def test_ttt_uneven_batch_samples_only_rows_with_residue_targets() -> None:
    model = DummyTTTModel()
    model._ttt_cfg.batch_size = 4
    # tokenized: (2, 3)
    tokenized = torch.tensor(
        [
            [model.tokenizer.cls_token_id, model.tokenizer.vocab["A"], 2],
            [model.tokenizer.cls_token_id, model.tokenizer.eos_token_id, 0],
        ]
    )
    generator = torch.Generator().manual_seed(3)

    _, labels = model._ttt_sample_batch(tokenized, generator)

    assert labels.ne(-100).any(dim=1).all()
    assert torch.equal(labels[labels.ne(-100)], torch.full((4,), model.tokenizer.vocab["A"]))


def test_ttt_rejects_all_ignored_inputs_before_adapter_injection() -> None:
    model = DummyTTTModel()
    # input_ids: (1, 3)
    input_ids = torch.tensor(
        [[model.tokenizer.cls_token_id, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id]]
    )

    with pytest.raises(ValueError, match="no trainable biological residue"):
        model.ttt(input_ids=input_ids)

    assert model._ttt_initialized is False


def test_ttt_rejects_dplm2_structure_tokens_before_adapter_injection() -> None:
    model = DummyTTTModel()
    tokenizer = DummyTokenizer()
    tokenizer.struct_cls_token = "<struct_cls>"
    tokenizer._token_to_id = {"<struct_cls>": 33}
    model.tokenizer = tokenizer
    model.config = SimpleNamespace(vocab_size=64, model_type="dplm2", struct_type=0)
    # input_ids: (1, 5)
    input_ids = torch.tensor([[1, tokenizer.vocab["A"], 33, 40, 2]])

    with pytest.raises(ValueError, match="amino-acid-only"):
        model.ttt(input_ids=input_ids)

    assert model._ttt_initialized is False


def test_ttt_only_lora_params_change_and_reset_restores_adapter() -> None:
    model = DummyTTTModel()
    model._ttt_ensure_initialized()
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    metrics = model.ttt(seq="ACDE")

    changed = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), initial[name])
    ]
    assert len(metrics["losses"]) == 1
    assert len(changed) > 0
    assert all("lora_" in name for name in changed)

    model.ttt_reset()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), initial[name])


def test_seed_and_initial_state_reset_reproduce_losses_and_updates() -> None:
    """Repeat one seeded adaptation from the same initial adapter state."""

    model = DummyTTTModel()
    config = {
        "steps": 2,
        "ags": 2,
        "batch_size": 2,
        "mask_ratio": 0.5,
        "bert_leave_prob": 0.1,
        "bert_replace_prob": 0.2,
        "seed": 7,
        "initial_state_reset": True,
    }

    first = model.ttt(seq="ACDE", ttt_config=config)
    first_state = model._ttt_snapshot_lora_state()
    second = model.ttt(seq="ACDE", ttt_config=config)
    second_state = model._ttt_snapshot_lora_state()

    assert first == second
    for first_module, second_module in zip(first_state, second_state, strict=True):
        assert first_module.keys() == second_module.keys()
        for name in first_module:
            assert torch.equal(first_module[name], second_module[name]), name


def test_ttt_save_pretrained_round_trip_preserves_adapter_and_reset_state(
    tmp_path: Path,
) -> None:
    model = DummyPretrainedTTTModel(DummyPretrainedTTTConfig()).eval()
    model._ttt_ensure_initialized()
    with torch.no_grad():
        for index, module in enumerate(model._ttt_lora_modules(), start=1):
            module.lora_up.weight.fill_(index / 10)
    adapted_state = model._ttt_snapshot_lora_state()

    model.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = DummyPretrainedTTTModel.from_pretrained(tmp_path, local_files_only=True).eval()

    assert reloaded._ttt_initialized is True
    assert reloaded.ttt_config.seed == 17
    assert reloaded.ttt_config.lora_target_modules == ("0", "2")
    reloaded_state = reloaded._ttt_snapshot_lora_state()
    for expected_module, actual_module in zip(adapted_state, reloaded_state, strict=True):
        for name in expected_module:
            torch.testing.assert_close(actual_module[name], expected_module[name])

    expected_initial = [
        (
            module._ttt_initial_lora_down.detach().clone(),
            module._ttt_initial_lora_up.detach().clone(),
        )
        for module in reloaded._ttt_lora_modules()
    ]
    reloaded.ttt_reset()
    for module, (expected_down, expected_up) in zip(
        reloaded._ttt_lora_modules(), expected_initial, strict=True
    ):
        torch.testing.assert_close(module.lora_down.weight, expected_down)
        torch.testing.assert_close(module.lora_up.weight, expected_up)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.network
@pytest.mark.checkpoint
@pytest.mark.parametrize("model_key", list(MODEL_REGISTRY))
def test_sequence_model_ttt_smoke(model_key: str) -> None:
    config = MODEL_REGISTRY[model_key]
    model_cls = LOCAL_MODEL_CLASSES[model_key]
    model = (
        model_cls.from_pretrained(
            config["fast_path"],
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )
    metrics = model.ttt(
        seq=TEST_SEQUENCE,
        ttt_config={
            "steps": 1,
            "ags": 1,
            "batch_size": 1,
            "crop_size": 64,
            "lora_rank": 2,
            "lora_alpha": 1.0,
        },
    )

    assert len(metrics["losses"]) == 1
    assert callable(model.ttt_reset)
    model.ttt_reset()
    del model
    torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
def test_esmfold2_ttt_smoke() -> None:
    from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    config = STRUCTURE_MODEL_REGISTRY["esmfold2_fast"]
    model = (
        ESMFold2Model.from_pretrained(
            config["fast_path"],
            load_esmc=True,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )

    result = model.fold_protein(
        TEST_SEQUENCE,
        num_loops=1,
        num_sampling_steps=1,
        num_diffusion_samples=1,
        seed=0,
        ttt=True,
        ttt_config={
            "steps": 1,
            "ags": 1,
            "batch_size": 1,
            "crop_size": 64,
            "lora_rank": 2,
            "lora_alpha": 1.0,
        },
    )

    assert result.ttt_metrics is not None
    assert len(result.ttt_metrics["losses"]) == 1
    assert len(result.ttt_metrics["step_plddts"]) == 2
    assert result.ttt_metrics["best_step"] in {0, 1}

    del model, result
    torch.cuda.empty_cache()
