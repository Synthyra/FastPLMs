"""Mandatory attention dispatch, masking, fallback, and cache contracts."""

from collections import OrderedDict

import pytest
import torch

import fastplms.attention.interfaces as attention_interfaces
import fastplms.models.esm_plusplus.modeling_esm_plusplus as esmpp_module
from fastplms.attention import _core as attention_core
from fastplms.models.esm2.modeling_fastesm import FastEsmConfig, FastEsmModel
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusModel,
)
from tests.unit import test_attention_interfaces as interface_contracts
from tests.unit import test_attention_regressions as contracts
from tests.unit import test_esmc_diagnostics as esmc_contracts

test_flash_backend_loads_only_its_hugging_face_kernel = (
    interface_contracts.test_flash_backend_loads_only_its_hugging_face_kernel
)
test_causal_masked_flash_uses_varlen_and_zeroes_padding = (
    interface_contracts.test_causal_masked_flash_uses_varlen_and_zeroes_padding
)
test_masked_flash_validates_padding_mask_shape_before_kernel_loading = (
    interface_contracts.test_masked_flash_validates_padding_mask_shape_before_kernel_loading
)
test_public_attention_setter_matches_transformers_513_kernel_policy = (
    interface_contracts.test_public_attention_setter_matches_transformers_513_kernel_policy
)
test_model_flash_flags_match_the_manifest = (
    interface_contracts.test_model_flash_flags_match_the_manifest
)
test_ankh_sdpa_never_mutates_process_global_reduction_policy = (
    interface_contracts.test_ankh_sdpa_never_mutates_process_global_reduction_policy
)
test_ankh_concurrent_fallback_and_sdpa_keep_backend_and_global_policy = (
    interface_contracts.test_ankh_concurrent_fallback_and_sdpa_keep_backend_and_global_policy
)
test_compiled_flex_cache_key_covers_execution_not_batch_contents = (
    interface_contracts.test_compiled_flex_cache_key_covers_execution_not_batch_contents
)
test_flex_block_mask_supports_disjoint_valid_spans_and_exact_cache_keys = (
    interface_contracts.test_flex_block_mask_supports_disjoint_valid_spans_and_exact_cache_keys
)
test_flex_block_mask_key_separates_equal_bytes_with_different_pattern_dtypes = (
    interface_contracts.test_flex_block_mask_key_separates_equal_bytes_with_different_pattern_dtypes
)
test_esmplusplus_flex_sequence_masks_share_exact_bounded_cache = (
    interface_contracts.test_esmplusplus_flex_sequence_masks_share_exact_bounded_cache
)
test_flash_kernel_variant_mismatch_fails_closed = (
    interface_contracts.test_flash_kernel_variant_mismatch_fails_closed
)
test_locked_kernel_is_hash_validated_before_import = (
    interface_contracts.test_locked_kernel_is_hash_validated_before_import
)
test_locked_kernel_offline_resolves_sparse_snapshot_without_hub_api = (
    interface_contracts.test_locked_kernel_offline_resolves_sparse_snapshot_without_hub_api
)
test_locked_kernel_offline_rejects_unlocked_cached_variant = (
    interface_contracts.test_locked_kernel_offline_rejects_unlocked_cached_variant
)

test_attention_masks_require_exact_batch_sequence_shape = (
    contracts.test_attention_masks_require_exact_batch_sequence_shape
)
test_attention_masks_reject_rows_without_valid_keys_before_dispatch = (
    contracts.test_attention_masks_reject_rows_without_valid_keys_before_dispatch
)
test_ankh_output_attentions_eager_fallback_honors_attention_dropout = (
    contracts.test_ankh_output_attentions_eager_fallback_honors_attention_dropout
)
test_ankh_sdpa_receives_training_attention_dropout = (
    contracts.test_ankh_sdpa_receives_training_attention_dropout
)
test_ankh_ttt_missing_input_uses_optimization_safe_validation = (
    contracts.test_ankh_ttt_missing_input_uses_optimization_safe_validation
)
test_bool_to_additive_mask_rejects_non_boolean_input_explicitly = (
    contracts.test_bool_to_additive_mask_rejects_non_boolean_input_explicitly
)
test_dplm_output_attentions_fallback_preserves_padding_mask = (
    contracts.test_dplm_output_attentions_fallback_preserves_padding_mask
)
test_dplm2_output_attentions_fallback_is_call_scoped = (
    contracts.test_dplm2_output_attentions_fallback_is_call_scoped
)
test_dplm_sdpa_output_attentions_fallback_preserves_cross_attention_mask_and_backend = (
    contracts.test_dplm_sdpa_output_attentions_fallback_preserves_cross_attention_mask_and_backend
)
test_e1_output_attentions_fallback_preserves_block_causal_mask_and_backend = (
    contracts.test_e1_output_attentions_fallback_preserves_block_causal_mask_and_backend
)
test_e1_public_fallback_warns_once_masks_padding_and_preserves_flex = (
    contracts.test_e1_public_fallback_warns_once_masks_padding_and_preserves_flex
)
test_esm3_sdpa_fallback_is_call_scoped_and_preserves_padding_mask = (
    contracts.test_esm3_sdpa_fallback_is_call_scoped_and_preserves_padding_mask
)
test_esm3_sequence_id_grouping_combines_with_public_padding_mask = (
    contracts.test_esm3_sequence_id_grouping_combines_with_public_padding_mask
)
test_esm3_rejects_malformed_padding_and_sequence_masks = (
    contracts.test_esm3_rejects_malformed_padding_and_sequence_masks
)
test_esm2_output_attentions_fallback_preserves_padding_mask = (
    contracts.test_esm2_output_attentions_fallback_preserves_padding_mask
)
test_esm2_attention_validates_head_divisibility_without_assertions = (
    contracts.test_esm2_attention_validates_head_divisibility_without_assertions
)
test_esm2_flex_unavailability_raises_explicit_runtime_error = (
    contracts.test_esm2_flex_unavailability_raises_explicit_runtime_error
)
test_esm2_legacy_backend_setter_uses_explicit_validation = (
    contracts.test_esm2_legacy_backend_setter_uses_explicit_validation
)
test_esmfold_flex_training_rejects_unimplemented_attention_dropout = (
    contracts.test_esmfold_flex_training_rejects_unimplemented_attention_dropout
)
test_esmfold_output_attentions_fallback_preserves_padding_mask = (
    contracts.test_esmfold_output_attentions_fallback_preserves_padding_mask
)
test_esmplusplus_rejects_empty_attention_rows_without_fallback_or_mutation = (
    contracts.test_esmplusplus_rejects_empty_attention_rows_without_fallback_or_mutation
)
test_esmplusplus_chain_masks_fail_closed_without_assertions = (
    contracts.test_esmplusplus_chain_masks_fail_closed_without_assertions
)
test_esmplusplus_flex_unavailability_raises_explicit_runtime_error = (
    contracts.test_esmplusplus_flex_unavailability_raises_explicit_runtime_error
)
test_flash_attention_2_dense_and_varlen_use_autograd_wrappers = (
    contracts.test_flash_attention_2_dense_and_varlen_use_autograd_wrappers
)
test_flash_attention_2_dense_and_varlen_preserve_lora_and_input_gradients = (
    contracts.test_flash_attention_2_dense_and_varlen_preserve_lora_and_input_gradients
)
test_flash_attention_2_rejects_low_level_only_kernel_artifact = (
    contracts.test_flash_attention_2_rejects_low_level_only_kernel_artifact
)
test_flash_attention_3_preserves_internal_type_errors = (
    contracts.test_flash_attention_3_preserves_internal_type_errors
)
test_kernel_loader_honors_all_offline_environment_variables = (
    contracts.test_kernel_loader_honors_all_offline_environment_variables
)
test_public_flex_cache_cleanup_is_scoped_and_complete = (
    contracts.test_public_flex_cache_cleanup_is_scoped_and_complete
)

test_esmc_calibration_contains_no_expected_failures = (
    esmc_contracts.test_esmc_calibration_contains_no_expected_failures
)
test_esmc_catastrophic_disagreement_remains_a_hard_failure = (
    esmc_contracts.test_esmc_catastrophic_disagreement_remains_a_hard_failure
)
test_esmc_supported_backend_deviation_warns_and_writes_complete_metrics = (
    esmc_contracts.test_esmc_supported_backend_deviation_warns_and_writes_complete_metrics
)


def _tiny_esm2(attn_backend: str) -> FastEsmModel:
    return FastEsmModel(
        FastEsmConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=16,
            pad_token_id=1,
            mask_token_id=5,
            position_embedding_type="absolute",
            attn_backend=attn_backend,
        )
    )


def _tiny_esmc(attn_backend: str) -> ESMplusplusModel:
    return ESMplusplusModel(
        ESMplusplusConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            dropout=0.0,
            pad_token_id=1,
            mask_token_id=5,
            attn_backend=attn_backend,
        )
    ).eval()


def test_esmc_flex_dispatch_is_compiled_once_and_receives_exact_padding_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supported ESMC Flex path must dispatch without hiding mask semantics."""

    sentinel_block_mask = object()
    observed: dict[str, object] = {
        "block_masks_created": 0,
        "compiled": 0,
        "compile_requests": [],
        "dispatches": [],
        "mask_mod": None,
    }
    compiled = None

    def fake_create_block_mask(mask_mod, *shape, **kwargs):
        observed["block_masks_created"] = int(observed["block_masks_created"]) + 1
        observed["mask_mod"] = mask_mod
        observed["block_mask_shape"] = shape
        observed["block_mask_device"] = kwargs["device"]
        return sentinel_block_mask

    def fake_get_flex_attention_fn(**kwargs):
        nonlocal compiled
        cast_requests = observed["compile_requests"]
        assert isinstance(cast_requests, list)
        cast_requests.append(kwargs)
        if compiled is None:
            observed["compiled"] = int(observed["compiled"]) + 1

            def run_flex(query, key, value, **call_kwargs):
                cast_dispatches = observed["dispatches"]
                assert isinstance(cast_dispatches, list)
                cast_dispatches.append(
                    {
                        "query_shape": tuple(query.shape),
                        "key_shape": tuple(key.shape),
                        "value_shape": tuple(value.shape),
                        **call_kwargs,
                    }
                )
                return value

            compiled = run_flex
        return compiled

    monkeypatch.setattr(attention_core, "_flex_block_masks", OrderedDict())
    monkeypatch.setattr(attention_core, "create_block_mask", fake_create_block_mask)
    monkeypatch.setattr(esmpp_module, "flex_attention", object())
    monkeypatch.setattr(
        esmpp_module,
        "_get_flex_attention_fn",
        fake_get_flex_attention_fn,
    )

    model = _tiny_esmc("flex_attention")
    input_ids = torch.tensor(((0, 3, 4, 2, 1), (0, 6, 2, 1, 1)))
    attention_mask = input_ids.ne(1)
    first = model(input_ids=input_ids, attention_mask=attention_mask)
    second = model(input_ids=input_ids, attention_mask=attention_mask)

    assert first.last_hidden_state.shape == (2, 5, 8)
    assert second.last_hidden_state.shape == first.last_hidden_state.shape
    assert torch.isfinite(first.last_hidden_state).all()
    assert torch.isfinite(second.last_hidden_state).all()
    assert model.attn_backend == "flex_attention"
    assert model.config.attn_backend == "flex_attention"
    assert model.config._attn_implementation == "flex_attention"

    compile_requests = observed["compile_requests"]
    dispatches = observed["dispatches"]
    assert isinstance(compile_requests, list) and len(compile_requests) == 2
    assert observed["compiled"] == 1
    assert compile_requests[0] == compile_requests[1]
    assert compile_requests[0] == {
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "shape": (2, 2, 5, 4),
        "mask_semantics": "padding",
    }
    assert observed["block_masks_created"] == 1
    assert len(attention_core._flex_block_masks) == 1
    assert isinstance(dispatches, list) and len(dispatches) == 2
    for dispatch in dispatches:
        assert dispatch["query_shape"] == (2, 2, 5, 4)
        assert dispatch["key_shape"] == (2, 2, 5, 4)
        assert dispatch["value_shape"] == (2, 2, 5, 4)
        assert dispatch["block_mask"] is sentinel_block_mask
        assert dispatch["scale"] == 0.5
        assert dispatch["kernel_options"] == {
            "PRESCALE_QK": True,
            "BLOCK_N": 32,
        }

    assert observed["block_mask_shape"] == (2, 1, 5, 5)
    assert observed["block_mask_device"] == torch.device("cpu")
    mask_mod = observed["mask_mod"]
    assert callable(mask_mod)
    for batch_index in range(2):
        for query_index in range(5):
            for key_index in range(5):
                expected = bool(
                    attention_mask[batch_index, query_index]
                    == attention_mask[batch_index, key_index]
                )
                assert bool(mask_mod(batch_index, 0, query_index, key_index)) is expected


def test_esmc_fake_fa3_dispatch_receives_exact_mask_and_returns_finite_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supported ESMC FA3 path must preserve public shape and mask contracts."""

    calls: list[dict[str, object]] = []

    def fake_flash_attention(**kwargs):
        query = kwargs["query_states"]
        key = kwargs["key_states"]
        value = kwargs["value_states"]
        attention_mask = kwargs["attention_mask_2d"]
        masked_output = value.masked_fill(
            attention_mask[:, :, None, None].logical_not(),
            0.0,
        )
        calls.append(
            {
                "query_shape": tuple(query.shape),
                "key_shape": tuple(key.shape),
                "value_shape": tuple(value.shape),
                "attention_mask": attention_mask.clone(),
                "causal": kwargs["causal"],
                "implementation": kwargs["implementation"],
                "masked_output": masked_output,
            }
        )
        return masked_output

    monkeypatch.setattr(
        esmpp_module,
        "kernels_flash_attention_func",
        fake_flash_attention,
    )
    monkeypatch.setattr(attention_interfaces, "require_kernels_package", lambda: None)

    model = _tiny_esmc("flash_attention_3")
    input_ids = torch.tensor(((0, 3, 4, 2, 1), (0, 6, 2, 1, 1)))
    attention_mask = input_ids.ne(1)
    output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.last_hidden_state.shape == (2, 5, 8)
    assert torch.isfinite(output.last_hidden_state).all()
    assert model.attn_backend == "flash_attention_3"
    assert model.config.attn_backend == "flash_attention_3"
    assert model.config._attn_implementation == "flash_attention_3"
    assert len(calls) == 1
    call = calls[0]
    assert call["query_shape"] == (2, 5, 2, 4)
    assert call["key_shape"] == (2, 5, 2, 4)
    assert call["value_shape"] == (2, 5, 2, 4)
    assert call["causal"] is False
    assert call["implementation"] == "flash_attention_3"
    assert torch.equal(call["attention_mask"], attention_mask)
    masked_output = call["masked_output"]
    assert isinstance(masked_output, torch.Tensor)
    assert torch.count_nonzero(masked_output[attention_mask.logical_not()]) == 0


def test_eager_and_sdpa_match_for_dense_and_mixed_padding_with_gradients() -> None:
    torch.manual_seed(17)
    eager = _tiny_esm2("eager").eval()
    sdpa = _tiny_esm2("sdpa").eval()
    sdpa.load_state_dict(eager.state_dict())
    input_ids = torch.tensor([[0, 3, 4, 2, 1], [0, 6, 2, 1, 1]])
    masks = (torch.ones_like(input_ids), input_ids.ne(1))

    for attention_mask in masks:
        eager.zero_grad(set_to_none=True)
        sdpa.zero_grad(set_to_none=True)
        eager_output = eager(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        sdpa_output = sdpa(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        torch.testing.assert_close(
            sdpa_output.last_hidden_state,
            eager_output.last_hidden_state,
            rtol=2e-5,
            atol=2e-6,
        )
        assert eager_output.hidden_states is not None
        assert sdpa_output.hidden_states is not None
        for sdpa_hidden, eager_hidden in zip(
            sdpa_output.hidden_states,
            eager_output.hidden_states,
            strict=True,
        ):
            torch.testing.assert_close(
                sdpa_hidden,
                eager_hidden,
                rtol=2e-5,
                atol=2e-6,
            )

        valid = attention_mask.bool().unsqueeze(-1)
        # A squared norm after the final LayerNorm is nearly constant and
        # produces cancellation-dominated gradients.  Project each hidden
        # coordinate onto a distinct fixed coefficient so this parity gate
        # compares a well-conditioned, nonzero backward signal.
        gradient_probe = torch.linspace(
            -1.0,
            1.0,
            eager_output.last_hidden_state.shape[-1],
            dtype=eager_output.last_hidden_state.dtype,
            device=eager_output.last_hidden_state.device,
        ).view(1, 1, -1)
        (eager_output.last_hidden_state * valid * gradient_probe).sum().backward()
        (sdpa_output.last_hidden_state * valid * gradient_probe).sum().backward()
        eager_gradients = {
            name: parameter.grad
            for name, parameter in eager.named_parameters()
            if parameter.grad is not None
        }
        sdpa_gradients = {
            name: parameter.grad
            for name, parameter in sdpa.named_parameters()
            if parameter.grad is not None
        }
        assert eager_gradients
        assert sdpa_gradients.keys() == eager_gradients.keys()
        expected_attention_projections = (
            "query",
            "key",
            "value",
            "attention.output.dense",
        )
        assert all(
            any(projection in name for name in eager_gradients)
            for projection in expected_attention_projections
        )
        for name, eager_gradient in eager_gradients.items():
            torch.testing.assert_close(
                sdpa_gradients[name],
                eager_gradient,
                rtol=3e-5,
                atol=3e-6,
            )
