"""Mandatory tiny DPLM, DPLM2, and ESM3 generation contracts."""

from __future__ import annotations

import pytest
import torch

from fastplms.models.esm3.modeling_esm3 import FastESM3Config, FastESM3Model
from tests.integration import test_dplm_generation as dplm_contracts
from tests.integration import test_esm3 as esm3_contracts

_integration_dplm_config = dplm_contracts._common_config


def _cpu_dplm_config(vocab_size: int) -> dict[str, object]:
    values = _integration_dplm_config(vocab_size)
    values.update(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "max_position_embeddings": 16,
            # DPLM2 intentionally advertises SDPA only; DPLM retains its
            # eager-path coverage in this shared tiny integration fixture.
            "attn_backend": "sdpa" if vocab_size == 64 else "eager",
        }
    )
    return values


def _cpu_small_esm3_config() -> FastESM3Config:
    return FastESM3Config(
        hidden_size=8,
        num_attention_heads=2,
        num_vector_heads=2,
        num_hidden_layers=1,
        attn_backend="eager",
    )


def _cpu_small_esm3_model() -> FastESM3Model:
    return FastESM3Model(_cpu_small_esm3_config()).eval()


# The integration contracts look up these helpers when each test executes.
# Override them in this positive CPU allowlist so the gate never grows into a
# benchmark-sized model while retaining the exact public behavior assertions.
esm3_contracts._small_config = _cpu_small_esm3_config
esm3_contracts._small_model = _cpu_small_esm3_model
dplm_contracts._common_config = _cpu_dplm_config

test_dplm_argmax_generation_preserves_fixed_positions = (
    dplm_contracts.test_dplm_argmax_generation_preserves_fixed_positions
)
test_dplm_automodel_rejects_decoder_cache_contracts = (
    dplm_contracts.test_dplm_automodel_rejects_decoder_cache_contracts
)
test_dplm_masked_lm_rejects_decoder_and_cross_attention_arguments = (
    dplm_contracts.test_dplm_masked_lm_rejects_decoder_and_cross_attention_arguments
)
test_dplm_task_heads_honor_config_and_explicit_return_dict = (
    dplm_contracts.test_dplm_task_heads_honor_config_and_explicit_return_dict
)
test_dplm2_argmax_generation_preserves_modalities_and_fixed_positions = (
    dplm_contracts.test_dplm2_argmax_generation_preserves_modalities_and_fixed_positions
)
test_dplm2_automodel_infers_official_multimodal_types_and_returns_pooling = (
    dplm_contracts.test_dplm2_automodel_infers_official_multimodal_types_and_returns_pooling
)
test_generation_rejects_invalid_controls = dplm_contracts.test_generation_rejects_invalid_controls
test_masked_lm_resize_updates_input_and_output_projections = (
    dplm_contracts.test_masked_lm_resize_updates_input_and_output_projections
)
test_seeded_stochastic_generation_is_repeatable = (
    dplm_contracts.test_seeded_stochastic_generation_is_repeatable
)

test_esm3_accepts_function_tokens_argument = (
    esm3_contracts.test_esm3_accepts_function_tokens_argument
)
test_esm3_generation_preserves_every_supported_conditioning_track = (
    esm3_contracts.test_esm3_generation_preserves_every_supported_conditioning_track
)
test_esm3_generation_none_num_steps_uses_mask_count = (
    esm3_contracts.test_esm3_generation_none_num_steps_uses_mask_count
)
test_esm3_generation_rejects_noninteger_num_steps = (
    esm3_contracts.test_esm3_generation_rejects_noninteger_num_steps
)
test_esm3_generation_rejects_nonpositive_num_steps = (
    esm3_contracts.test_esm3_generation_rejects_nonpositive_num_steps
)
test_esm3_generation_rejects_unknown_or_ambiguous_inputs = (
    esm3_contracts.test_esm3_generation_rejects_unknown_or_ambiguous_inputs
)
test_esm3_loads_with_automodel = esm3_contracts.test_esm3_loads_with_automodel
test_esm3_rejects_attention_mask_row_without_a_valid_key = (
    esm3_contracts.test_esm3_rejects_attention_mask_row_without_a_valid_key
)
test_esm3_resize_updates_sequence_input_and_output_embeddings = (
    esm3_contracts.test_esm3_resize_updates_sequence_input_and_output_embeddings
)
test_esm3_repeated_save_removes_stale_runtime_outputs = (
    esm3_contracts.test_esm3_repeated_save_removes_stale_runtime_outputs
)
test_esm3_saved_bridge_rejects_poisoned_archive = (
    esm3_contracts.test_esm3_saved_bridge_rejects_poisoned_archive
)
test_esm3_saved_bridge_rejects_preimported_runtime_mismatch = (
    esm3_contracts.test_esm3_saved_bridge_rejects_preimported_runtime_mismatch
)
test_esm3_saved_bridge_reuses_same_runtime_in_process = (
    esm3_contracts.test_esm3_saved_bridge_reuses_same_runtime_in_process
)
test_esm3_saved_model_loads_without_installed_fastplms = (
    esm3_contracts.test_esm3_saved_model_loads_without_installed_fastplms
)
test_esm3_saved_runtime_archive_is_fixed_bounded_and_deterministic = (
    esm3_contracts.test_esm3_saved_runtime_archive_is_fixed_bounded_and_deterministic
)
test_esm3_saved_runtime_rejects_allowlisted_symlink = (
    esm3_contracts.test_esm3_saved_runtime_rejects_allowlisted_symlink
)
test_esm3_saved_runtime_rejects_missing_allowlisted_file = (
    esm3_contracts.test_esm3_saved_runtime_rejects_missing_allowlisted_file
)
test_esm3_saved_runtime_rejects_noncanonical_allowlist_paths = (
    esm3_contracts.test_esm3_saved_runtime_rejects_noncanonical_allowlist_paths
)
test_esm3_saved_runtime_rejects_oversize_allowlisted_file = (
    esm3_contracts.test_esm3_saved_runtime_rejects_oversize_allowlisted_file
)
test_esm3_saved_runtime_rejects_oversize_total = (
    esm3_contracts.test_esm3_saved_runtime_rejects_oversize_total
)
test_esm3_seeded_generation_is_repeatable_and_preserves_context = (
    esm3_contracts.test_esm3_seeded_generation_is_repeatable_and_preserves_context
)
test_esm3_uses_hugging_face_initialization_and_only_retains_requested_states = (
    esm3_contracts.test_esm3_uses_hugging_face_initialization_and_only_retains_requested_states
)


def test_esm3_sequence_only_forward() -> None:
    model = _cpu_small_esm3_model()
    batch = model.tokenize_sequences(["MKTAYIAKQ", "GGGG"], device=model.device)

    with torch.inference_mode():
        output = model(**batch)

    assert output.logits is not None
    assert output.logits.shape == (*batch["input_ids"].shape, model.config.vocab_size)
    assert output.last_hidden_state.shape == (
        *batch["input_ids"].shape,
        model.config.hidden_size,
    )
    assert output.structure_logits.shape[-1] == 4096
    assert output.function_logits.shape[-2:] == (8, 260)
    assert output.residue_logits.shape[-1] == 1478
    assert torch.isfinite(output.logits).all()
    with pytest.raises(TypeError, match="unexpected_cpu_contract"):
        model(**batch, unexpected_cpu_contract=True)


def test_esm3_advertised_model_logits_backward() -> None:
    model = esm3_contracts._small_model().train()
    batch = model.tokenize_sequences(["MKT", "GG"], device=model.device)
    output = model(**batch, return_dict=True)
    loss = output.sequence_logits.float().square().mean()

    assert torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
