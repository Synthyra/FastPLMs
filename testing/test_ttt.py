from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from fastplms.ankh.modeling_ankh import FastAnkhForMaskedLM
from fastplms.dplm.modeling_dplm import DPLMForMaskedLM
from fastplms.dplm2.modeling_dplm2 import DPLM2ForMaskedLM
from fastplms.e1.modeling_e1 import E1ForMaskedLM
from fastplms.esm2.modeling_fastesm import FastEsmForMaskedLM
from fastplms.esm3.modeling_esm3 import FastESM3Model
from fastplms.esm_plusplus.modeling_esm_plusplus import ESMplusplusForMaskedLM
from fastplms.esmfold2.modeling_esmfold2 import ESMFold2Model
from fastplms.test_time_training import (
    FastPLMTestTimeTrainingMixin,
    LoraInjectedLinear,
)
from testing.conftest import MODEL_REGISTRY, STRUCTURE_MODEL_REGISTRY


TEST_SEQUENCE = "MSTNPKPQRKTKRNT"
LOCAL_MODEL_CLASSES = {
    "esm2": FastEsmForMaskedLM,
    "esmc": ESMplusplusForMaskedLM,
    "esm3": FastESM3Model,
    "e1": E1ForMaskedLM,
    "dplm": DPLMForMaskedLM,
    "dplm2": DPLM2ForMaskedLM,
    "ankh": FastAnkhForMaskedLM,
}


class DummyConfig:
    vocab_size = 8


class DummyTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    eos_token_id = 2
    mask_token_id = 3
    all_special_ids = [0, 1, 2, 3]

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
                [self.cls_token_id]
                + [self.vocab[aa] for aa in sequence]
                + [self.eos_token_id]
            )
        max_len = max(len(ids) for ids in encoded)
        input_ids = torch.full((len(encoded), max_len), self.pad_token_id)
        for row, ids in enumerate(encoded):
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
        del attention_mask
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


def test_ttt_only_lora_params_change_and_reset_restores_adapter() -> None:
    model = DummyTTTModel()
    model._ttt_ensure_initialized()
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

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


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", list(MODEL_REGISTRY))
def test_sequence_model_ttt_smoke(model_key: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FastPLMs model smoke tests.")
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
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for ESMFold2 TTT smoke tests.")
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
