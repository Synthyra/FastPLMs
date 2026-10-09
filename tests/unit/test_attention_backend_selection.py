"""Backend names, configuration fields, fallback warnings and the pure parts of ``auto`` selection."""

from __future__ import annotations

import pytest
import torch

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from fastplms.attention import (
    AUTO_ATTENTION,
    LEGACY_CHECKPOINT_ATTENTION_BACKENDS,
    VALID_ATTENTION_BACKENDS,
    AttentionBackend,
    AttentionCandidate,
    AttentionExecutionContext,
    AttentionResolution,
    _core,
    bool_to_additive_mask,
    canonical_checkpoint_attention_backend,
    get_attn_implementation,
    resolve_attention_backend,
    resolve_attention_backend_for_call,
    set_config_attn_implementation,
    warn_attention_backend_fallback,
)
from fastplms.attention._auto import (
    attention_execution_context,
    deferred_resolution,
    needs_execution_context,
    provisional_implementation,
    resolve_auto_attention,
)


class TestAttentionBackend:
    @pytest.mark.parametrize(
        ("backend", "is_flash"),
        [
            (AttentionBackend.EAGER, False),
            (AttentionBackend.SDPA, False),
            (AttentionBackend.FLEX_ATTENTION, False),
            (AttentionBackend.FLASH_ATTENTION_2, True),
            (AttentionBackend.FLASH_ATTENTION_3, True),
        ],
    )
    def test_is_flash_names_exactly_the_two_flash_backends(self, backend: AttentionBackend, is_flash: bool) -> None:
        assert backend.is_flash is is_flash

    def test_values_are_the_transformers_names(self) -> None:
        assert VALID_ATTENTION_BACKENDS == (
            "eager",
            "sdpa",
            "flex_attention",
            "flash_attention_2",
            "flash_attention_3",
        )
        assert AttentionBackend.FLEX is AttentionBackend.FLEX_ATTENTION
        assert AttentionBackend("sdpa") == "sdpa"


class TestResolveAttentionBackend:
    def test_none_means_sdpa(self) -> None:
        assert resolve_attention_backend(None) is AttentionBackend.SDPA

    def test_names_and_members_resolve_to_members(self) -> None:
        assert resolve_attention_backend("eager") is AttentionBackend.EAGER
        assert resolve_attention_backend(AttentionBackend.FLASH_ATTENTION_2) is AttentionBackend.FLASH_ATTENTION_2

    @pytest.mark.parametrize("name", ["flex", "flash", "auto", "SDPA", ""])
    def test_unknown_names_are_rejected_not_substituted(self, name: str) -> None:
        with pytest.raises(ValueError, match="Unsupported attention implementation"):
            resolve_attention_backend(name)

    def test_flex_without_pytorch_support_is_a_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_core, "flex_attention", None)
        with pytest.raises(RuntimeError, match="does not provide it"):
            resolve_attention_backend("flex_attention")
        assert resolve_attention_backend("sdpa") is AttentionBackend.SDPA


class TestCheckpointBackendNames:
    def test_the_legacy_flex_spelling_is_renamed(self) -> None:
        assert dict(LEGACY_CHECKPOINT_ATTENTION_BACKENDS) == {"flex": "flex_attention"}
        assert canonical_checkpoint_attention_backend("flex") == "flex_attention"

    def test_other_names_and_none_pass_through(self) -> None:
        assert canonical_checkpoint_attention_backend(None) is None
        assert canonical_checkpoint_attention_backend("flash") == "flash"
        assert canonical_checkpoint_attention_backend("sdpa") == "sdpa"


class TestConfigurationFields:
    def test_get_reads_the_transformers_field_first(self) -> None:
        config = SimpleNamespace(_attn_implementation="eager", attn_backend="sdpa")
        assert get_attn_implementation(config) == "eager"

    def test_get_falls_back_to_the_legacy_field_then_to_sdpa(self) -> None:
        assert get_attn_implementation(SimpleNamespace(attn_backend="eager")) == "eager"
        assert get_attn_implementation(SimpleNamespace()) == "sdpa"
        assert get_attn_implementation(SimpleNamespace(_attn_implementation=None, attn_backend=None)) == "sdpa"

    def test_get_rejects_an_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="Unsupported attention implementation"):
            get_attn_implementation(SimpleNamespace(_attn_implementation="flash"))

    def test_set_writes_both_fields_and_returns_the_name(self) -> None:
        config = SimpleNamespace()
        assert set_config_attn_implementation(config, "eager") == "eager"
        assert config._attn_implementation == "eager"
        assert config.attn_backend == "eager"
        assert not hasattr(config, "_attn_implementation_internal")

    def test_set_prefers_the_internal_field_when_the_config_has_one(self) -> None:
        config = SimpleNamespace(_attn_implementation_internal="sdpa", _attn_implementation="sdpa")
        set_config_attn_implementation(config, "eager")
        assert config._attn_implementation_internal == "eager"
        assert config._attn_implementation == "sdpa"
        assert config.attn_backend == "eager"

    def test_set_rejects_an_unknown_name_and_leaves_the_config_alone(self) -> None:
        config = SimpleNamespace(_attn_implementation="sdpa")
        with pytest.raises(ValueError):
            set_config_attn_implementation(config, "auto")
        assert vars(config) == {"_attn_implementation": "sdpa"}


class TestFallbackWarnings:
    def test_equal_backends_do_not_warn(self, recwarn: pytest.WarningsRecorder) -> None:
        warn_attention_backend_fallback("sdpa", effective_backend=AttentionBackend.SDPA, reason="Not needed.")
        assert len(recwarn) == 0

    def test_a_substitution_warns_once_naming_both_backends(self) -> None:
        with pytest.warns(RuntimeWarning, match="'sdpa'.*'eager'") as caught:
            warn_attention_backend_fallback("sdpa", effective_backend="eager", reason="Attentions were requested.")
        assert len(caught) == 1
        assert "Attentions were requested." in str(caught[0].message)
        assert "configured backend remains unchanged" in str(caught[0].message)

    def test_per_call_resolution_keeps_the_backend_unless_attentions_are_requested(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        assert resolve_attention_backend_for_call("sdpa", output_attentions=False) is AttentionBackend.SDPA
        assert resolve_attention_backend_for_call("eager", output_attentions=True) is AttentionBackend.EAGER
        assert len(recwarn) == 0
        with pytest.warns(RuntimeWarning, match="output_attentions=True"):
            assert resolve_attention_backend_for_call("sdpa", output_attentions=True) is AttentionBackend.EAGER


class TestBoolToAdditiveMask:
    def test_valid_positions_are_zero_and_invalid_are_negative_infinity(self) -> None:
        valid = torch.tensor([[True, False], [False, True]])  # (2, 2)
        additive = bool_to_additive_mask(valid, torch.float32)  # (2, 2)
        assert additive.dtype == torch.float32
        assert additive.tolist() == [[0.0, float("-inf")], [float("-inf"), 0.0]]

    def test_a_non_bool_mask_is_refused(self) -> None:
        with pytest.raises(TypeError, match="requires a bool tensor"):
            bool_to_additive_mask(torch.ones(2, dtype=torch.int64), torch.float32)


class TestAutomaticOrder:
    def test_flash_in_the_order_needs_an_execution_context(self) -> None:
        assert needs_execution_context(("flash_attention_3", "sdpa")) is True
        assert needs_execution_context(("flash_attention_2",)) is True
        assert needs_execution_context(("sdpa", "eager")) is False
        assert needs_execution_context(()) is False

    def test_provisional_implementation_is_the_first_context_free_one(self) -> None:
        assert provisional_implementation(("flash_attention_3", "flash_attention_2", "sdpa", "eager")) == "sdpa"
        assert provisional_implementation(("eager", "sdpa")) == "eager"

    def test_an_order_of_only_flash_has_no_provisional_implementation(self) -> None:
        with pytest.raises(ValueError, match="no context-free implementation"):
            provisional_implementation(("flash_attention_2", "flash_attention_3"))

    def test_deferred_resolution_records_a_waiting_request(self) -> None:
        resolution = deferred_resolution(("flash_attention_3", "sdpa"))
        assert resolution == AttentionResolution(
            requested=AUTO_ATTENTION, resolved="sdpa", candidates=(), context=None, deferred=True
        )

    def test_resolution_takes_the_first_usable_implementation_and_records_why(self) -> None:
        context = AttentionExecutionContext(torch.device("cpu"), torch.float32)
        resolution = resolve_auto_attention(("flash_attention_3", "sdpa", "eager"), context)
        assert resolution.resolved == "sdpa"
        assert resolution.requested == AUTO_ATTENTION
        assert resolution.deferred is False
        assert resolution.context is context
        flash, sdpa = resolution.candidates
        assert (flash.implementation, flash.usable) == ("flash_attention_3", False)
        assert "CUDA device" in flash.reason
        assert (sdpa.implementation, sdpa.usable) == ("sdpa", True)

    def test_resolution_without_a_context_skips_flash(self) -> None:
        resolution = resolve_auto_attention(("flash_attention_2", "eager"), None)
        assert resolution.resolved == "eager"
        assert resolution.candidates[0].reason == "It needs a device and dtype, which are not known."

    def test_resolution_fails_naming_every_reason_when_nothing_is_usable(self) -> None:
        with pytest.raises(RuntimeError, match="No implementation in the automatic attention order is usable") as caught:
            resolve_auto_attention(("flash_attention_2", "flash_attention_3"), None)
        assert "flash_attention_2:" in str(caught.value)
        assert "flash_attention_3:" in str(caught.value)

    def test_candidates_are_immutable_records(self) -> None:
        candidate = AttentionCandidate("sdpa", True, "ok")
        assert (candidate.implementation, candidate.usable, candidate.reason) == ("sdpa", True, "ok")
        with pytest.raises(FrozenInstanceError):
            candidate.usable = False  # type: ignore[misc]


class TestExecutionContext:
    def test_it_describes_the_parameters_device_and_dtype(self) -> None:
        module = torch.nn.Linear(2, 2).to(torch.float64)
        context = attention_execution_context(module)
        assert context == AttentionExecutionContext(torch.device("cpu"), torch.float64)

    def test_an_explicit_dtype_wins(self) -> None:
        module = torch.nn.Linear(2, 2)
        context = attention_execution_context(module, device="cpu", dtype=torch.bfloat16)
        assert context == AttentionExecutionContext(torch.device("cpu"), torch.bfloat16)

    def test_a_module_without_parameters_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="requires a model with parameters"):
            attention_execution_context(torch.nn.Identity())
