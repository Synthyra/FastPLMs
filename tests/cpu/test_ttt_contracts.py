"""Mandatory deterministic test-time-training contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastplms.models.ankh.modeling_ankh import FastAnkhForMaskedLMExtension
from fastplms.models.dplm.modeling_dplm import DPLMConfig, DPLMForMaskedLM
from fastplms.models.dplm2.modeling_dplm2 import DPLM2Config, DPLM2ForMaskedLM
from fastplms.models.e1.modeling_e1 import E1ForMaskedLM
from fastplms.models.esm2.modeling_fastesm import FastEsmForMaskedLM
from fastplms.models.esm3.modeling_esm3 import FastESM3Config, FastESM3Model
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
)
from fastplms.models.ttt import LoraInjectedLinear
from tests.integration import test_ttt as contracts
from tests.unit.test_ankh_cpu_contract import _config as _ankh_config
from tests.unit.test_e1_cache_contract import _tiny_e1_batch, _tiny_e1_config

test_ttt_first_call_mapping_preserves_explicit_target_override = (
    contracts.test_ttt_first_call_mapping_preserves_explicit_target_override
)
test_ttt_direct_init_mapping_preserves_family_target_class = (
    contracts.test_ttt_direct_init_mapping_preserves_family_target_class
)
test_ttt_config_rejects_invalid_optimizer_and_target_contracts = (
    contracts.test_ttt_config_rejects_invalid_optimizer_and_target_contracts
)
test_ttt_first_call_mapping_preserves_family_target_class = (
    contracts.test_ttt_first_call_mapping_preserves_family_target_class
)
test_seed_and_initial_state_reset_reproduce_losses_and_updates = (
    contracts.test_seed_and_initial_state_reset_reproduce_losses_and_updates
)
test_ttt_adapter_initialization_is_seeded_and_preserves_ambient_rng = (
    contracts.test_ttt_adapter_initialization_is_seeded_and_preserves_ambient_rng
)
test_ttt_generic_replacements_exclude_reserved_vocabulary_ids = (
    contracts.test_ttt_generic_replacements_exclude_reserved_vocabulary_ids
)
test_ttt_lora_injection_is_lazy_and_backbone_scoped = (
    contracts.test_ttt_lora_injection_is_lazy_and_backbone_scoped
)
test_ttt_only_lora_params_change_and_reset_restores_adapter = (
    contracts.test_ttt_only_lora_params_change_and_reset_restores_adapter
)
test_ttt_rejects_all_ignored_inputs_before_adapter_injection = (
    contracts.test_ttt_rejects_all_ignored_inputs_before_adapter_injection
)
test_ttt_rejects_dplm2_structure_tokens_before_adapter_injection = (
    contracts.test_ttt_rejects_dplm2_structure_tokens_before_adapter_injection
)
test_ttt_save_pretrained_round_trip_preserves_adapter_and_reset_state = (
    contracts.test_ttt_save_pretrained_round_trip_preserves_adapter_and_reset_state
)
test_ttt_uneven_batch_samples_only_rows_with_residue_targets = (
    contracts.test_ttt_uneven_batch_samples_only_rows_with_residue_targets
)


class _ProteinTokenizer:
    pad_token_id = 1

    def __init__(
        self,
        mask_token_id: int = 3,
        additional_special_ids: tuple[int, ...] = (),
    ) -> None:
        self.mask_token_id = mask_token_id
        self.all_special_ids = sorted(
            {0, 1, 2, mask_token_id, *additional_special_ids}
        )
        available_ids = [
            token_id
            for token_id in range(3, 32)
            if token_id not in self.all_special_ids
        ][:20]
        if len(available_ids) != 20:
            raise ValueError("Tiny tokenizer cannot allocate the canonical amino-acid alphabet")
        self.vocab = {
            amino_acid: token_id
            for amino_acid, token_id in zip(
                "ACDEFGHIKLMNPQRSTVWY",
                available_ids,
                strict=True,
            )
        }

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, 3)


class _DPLM2ProteinTokenizer(_ProteinTokenizer):
    aa_mask_token = "<mask_aa>"
    struct_cls_token = "<cls_struct>"

    def __init__(self) -> None:
        super().__init__(
            mask_token_id=32,
            additional_special_ids=(3, 33, 34, 35, 36),
        )
        self._token_to_id = {
            **self.vocab,
            self.aa_mask_token: 32,
            self.struct_cls_token: 33,
        }


def test_dplm2_ttt_replacements_exclude_ambiguous_and_reserved_tokens() -> None:
    tokenizer = _DPLM2ProteinTokenizer()
    tokenizer._token_to_id.update(
        {"X": 24, "B": 25, "U": 26, "Z": 27, "O": 28, "-": 29}
    )
    replacements = DPLM2ForMaskedLM._ttt_replacement_tokens(
        SimpleNamespace(tokenizer=tokenizer),
        torch.tensor([[0, 4, 2]], dtype=torch.long),
    )

    assert replacements.tolist() == [
        tokenizer._token_to_id[residue]
        for residue in "ACDEFGHIKLMNPQRSTVWY"
    ]


_EXPECTED_ADAPTERS = {
    "esm2": (
        "esm.encoder.layer.0.attention.self.query",
        "esm.encoder.layer.0.attention.self.key",
        "esm.encoder.layer.0.attention.self.value",
        "esm.encoder.layer.0.attention.output.dense",
    ),
    "esm_plusplus": (
        "transformer.blocks.0.attn.layernorm_qkv.1",
        "transformer.blocks.0.attn.out_proj",
    ),
    "esm3": (
        "esm3.transformer.blocks.0.attn.layernorm_qkv.1",
        "esm3.transformer.blocks.0.attn.out_proj",
    ),
    "ankh": (
        "encoder.block.0.layer.0.SelfAttention.q",
        "encoder.block.0.layer.0.SelfAttention.k",
        "encoder.block.0.layer.0.SelfAttention.v",
        "encoder.block.0.layer.0.SelfAttention.o",
    ),
    "dplm": (
        "esm.encoder.layer.0.attention.self.query",
        "esm.encoder.layer.0.attention.self.key",
        "esm.encoder.layer.0.attention.self.value",
        "esm.encoder.layer.0.attention.output.dense",
    ),
    "dplm2": (
        "esm.encoder.layer.0.attention.self.query",
        "esm.encoder.layer.0.attention.self.key",
        "esm.encoder.layer.0.attention.self.value",
        "esm.encoder.layer.0.attention.output.dense",
    ),
    "e1": (
        "model.layers.0.norm_attn_norm.self_attn.q_proj",
        "model.layers.0.norm_attn_norm.self_attn.k_proj",
        "model.layers.0.norm_attn_norm.self_attn.v_proj",
        "model.layers.0.norm_attn_norm.self_attn.o_proj",
    ),
}

_EXPECTED_TARGET_CLASS = {
    "esm2": "EsmAttention",
    "esm_plusplus": "MultiHeadAttention",
    "esm3": "MultiHeadAttention",
    "ankh": "AnkhSelfAttention",
    "dplm": "ModifiedEsmAttention",
    "dplm2": "ModifiedEsmAttention",
    "e1": "Attention",
}


def _family_model_and_inputs(family: str):
    if family == "esm2":
        from tests.cpu.test_sequence_autoclass_contracts import _esm2_config

        config = _esm2_config()
        config.vocab_size = 32
        tokenizer = _ProteinTokenizer(mask_token_id=config.mask_token_id)
        model = FastEsmForMaskedLM(config)
        model.tokenizer = tokenizer
        residues = [tokenizer.convert_tokens_to_ids(value) for value in "AC"]
        return model, {"input_ids": torch.tensor([[0, *residues, 2, 1]])}
    if family == "esm_plusplus":
        tokenizer = _ProteinTokenizer(mask_token_id=3)
        model = ESMplusplusForMaskedLM(
            ESMplusplusConfig(
                vocab_size=32,
                hidden_size=8,
                num_attention_heads=2,
                num_hidden_layers=1,
                dropout=0.0,
                pad_token_id=1,
                mask_token_id=3,
                attn_backend="eager",
            )
        )
        model.tokenizer = tokenizer
        residues = [tokenizer.convert_tokens_to_ids(value) for value in "AC"]
        return model, {"input_ids": torch.tensor([[0, *residues, 2, 1]])}
    if family == "esm3":
        model = FastESM3Model(
            FastESM3Config(
                hidden_size=8,
                num_attention_heads=2,
                num_vector_heads=2,
                num_hidden_layers=1,
                attn_backend="eager",
            )
        )
        return model, {"input_ids": model.encode("AC")["input_ids"]}
    if family == "ankh":
        tokenizer = _ProteinTokenizer(mask_token_id=3)
        model = FastAnkhForMaskedLMExtension(
            _ankh_config(vocab_size=32, num_layers=1, num_decoder_layers=1)
        )
        model.tokenizer = tokenizer
        residues = [tokenizer.convert_tokens_to_ids(value) for value in "AC"]
        return model, {"input_ids": torch.tensor([[*residues, 1, 0]])}
    if family == "dplm":
        from tests.cpu.test_sequence_autoclass_contracts import _dplm_config_values

        config = DPLMConfig(**_dplm_config_values(33))
        tokenizer = _ProteinTokenizer(mask_token_id=config.mask_token_id)
        model = DPLMForMaskedLM(config)
        model.tokenizer = tokenizer
        residues = [tokenizer.convert_tokens_to_ids(value) for value in "AC"]
        return model, {"input_ids": torch.tensor([[0, *residues, 2, 1]])}
    if family == "dplm2":
        from tests.cpu.test_sequence_autoclass_contracts import _dplm2_config_values

        model = DPLM2ForMaskedLM(DPLM2Config(**_dplm2_config_values()))
        model.tokenizer = _DPLM2ProteinTokenizer()
        return model, {"input_ids": torch.tensor([[0, 4, 5, 2, 1]])}
    if family == "e1":
        return E1ForMaskedLM(_tiny_e1_config()), _tiny_e1_batch()
    raise AssertionError(f"Unhandled TTT family: {family}")


@pytest.mark.parametrize("family", tuple(_EXPECTED_ADAPTERS))
def test_each_sequence_family_first_call_ttt_is_scoped_and_reloadable(
    family: str,
    tmp_path: Path,
) -> None:
    model, model_inputs = _family_model_and_inputs(family)
    original_parameters = {
        id(parameter): parameter.detach().clone()
        for parameter in model.parameters()
    }
    metrics = model.ttt(
        **model_inputs,
        ttt_config={
            "steps": 1,
            "ags": 1,
            "batch_size": 1,
            "mask_ratio": 1.0,
            "bert_leave_prob": 0.0,
            "bert_replace_prob": 0.0,
            "lora_rank": 2,
            "lora_alpha": 1.0,
            "seed": 7,
        },
    )

    adapters = tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, LoraInjectedLinear)
    )
    assert model.ttt_config.lora_target_replace_module == _EXPECTED_TARGET_CLASS[family]
    assert adapters == _EXPECTED_ADAPTERS[family]
    assert len(metrics["losses"]) == 1
    assert torch.isfinite(torch.tensor(metrics["losses"])).all()
    for parameter in model.parameters():
        if id(parameter) in original_parameters:
            torch.testing.assert_close(
                parameter.detach(),
                original_parameters[id(parameter)],
                rtol=0.0,
                atol=0.0,
            )
    assert any(
        name.endswith("lora_up.weight")
        and int(torch.count_nonzero(parameter).item()) > 0
        for name, parameter in model.named_parameters()
    )

    save_directory = tmp_path / family
    model.save_pretrained(save_directory, safe_serialization=True)
    reloaded = type(model).from_pretrained(save_directory, local_files_only=True)
    reloaded_adapters = tuple(
        name
        for name, module in reloaded.named_modules()
        if isinstance(module, LoraInjectedLinear)
    )
    assert reloaded_adapters == adapters
    source_state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if ".lora_" in name
    }
    reloaded_state = {
        name: tensor
        for name, tensor in reloaded.state_dict().items()
        if ".lora_" in name
    }
    assert set(reloaded_state) == set(source_state)
    for name, tensor in source_state.items():
        torch.testing.assert_close(reloaded_state[name], tensor, rtol=0.0, atol=0.0)
