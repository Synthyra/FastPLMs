"""Small contracts for isolated official-reference adapters."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tests.parity.support.native_reference import _validated_generation_limitation
from tests.parity.support.reference_adapters import OfficialGenerationUnavailable
from tests.parity.support.reference_adapters.dplm2 import (
    DPLM2_3B_GENERATION_LIMITATION,
    _accepts_type_ids,
    _call_checkpoint_forward,
    _call_checkpoint_generate,
)


class _AcceptsTypeIds(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del type_ids
        return input_ids


class _AcceptsKeywordArguments(nn.Module):
    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return input_ids


class _RejectsTypeIds(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids


class _OfficialWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generation_calls: list[dict[str, object]] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids + 1

    def generate(self, input_tokens: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.generation_calls.append({"input_tokens": input_tokens, **kwargs})
        return input_tokens + 2


class _CheckpointNetwork(_RejectsTypeIds):
    def __init__(self) -> None:
        super().__init__()
        self.generation_calls: list[dict[str, object]] = []

    def generate(
        self,
        batch: dict[str, torch.Tensor],
        max_iter: int,
        sampling_strategy: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.generation_calls.append(
            {
                "batch": batch,
                "max_iter": max_iter,
                "sampling_strategy": sampling_strategy,
            }
        )
        return batch["input_ids"] + 3, torch.zeros_like(batch["input_ids"])


class EsmForDPLM(_RejectsTypeIds):
    """Minimal reproduction of the pinned 3B sampler's missing BOS token."""

    bos_id = None

    def generate(
        self,
        batch: dict[str, torch.Tensor],
        max_iter: int,
        sampling_strategy: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del max_iter, sampling_strategy
        tokens = batch["input_ids"]
        tokens.ne(self.bos_id)
        raise AssertionError("unreachable")


def test_dplm2_checkpoint_forward_selection_is_signature_gated() -> None:
    """Unsupported modality keywords select the public checkpoint network."""

    assert _accepts_type_ids(_AcceptsTypeIds())
    assert _accepts_type_ids(_AcceptsKeywordArguments())

    model = _RejectsTypeIds()
    input_tensor = torch.tensor([1])
    type_ids = torch.tensor([0])
    assert not _accepts_type_ids(model)
    with pytest.raises(TypeError, match="type_ids"):
        model(input_tensor, type_ids=type_ids)
    assert torch.equal(
        _call_checkpoint_forward(_OfficialWrapper(), model, input_tensor, {}),
        input_tensor,
    )
    assert torch.equal(
        _call_checkpoint_forward(
            _OfficialWrapper(),
            _AcceptsTypeIds(),
            input_tensor,
            {},
        ),
        input_tensor + 1,
    )


def test_dplm2_checkpoint_generation_uses_public_selected_network() -> None:
    """The 3B architecture bypasses only the broken multimodal generator."""

    oracle = _OfficialWrapper()
    network = _CheckpointNetwork()
    input_tokens = torch.tensor([[1, 2]])
    generated = _call_checkpoint_generate(
        oracle,
        network,
        input_tokens,
        {
            "max_iter": 4,
            "sampling_strategy": "argmax",
            "unmasking_strategy": "deterministic",
        },
    )

    assert torch.equal(generated, input_tokens + 3)
    assert not oracle.generation_calls
    assert network.generation_calls == [
        {
            "batch": {"input_ids": input_tokens},
            "max_iter": 4,
            "sampling_strategy": "argmax",
        }
    ]


def test_dplm2_multimodal_generation_is_retained_when_supported() -> None:
    """Native DPLM2 networks continue through the official outer sampler."""

    oracle = _OfficialWrapper()
    input_tokens = torch.tensor([[1, 2]])
    generated = _call_checkpoint_generate(
        oracle,
        _AcceptsTypeIds(),
        input_tokens,
        {"max_iter": 4},
    )

    assert torch.equal(generated, input_tokens + 2)
    assert oracle.generation_calls == [
        {"input_tokens": input_tokens, "max_iter": 4}
    ]


def test_dplm2_checkpoint_generation_normalizes_exact_public_failure() -> None:
    """The unusable 3B sampler records evidence without patching the oracle."""

    with pytest.raises(OfficialGenerationUnavailable) as captured:
        _call_checkpoint_generate(
            _OfficialWrapper(),
            EsmForDPLM(),
            torch.tensor([[1, 2]]),
            {"max_iter": 4, "sampling_strategy": "argmax"},
        )

    assert captured.value.as_record() == DPLM2_3B_GENERATION_LIMITATION
    assert isinstance(captured.value.__cause__, TypeError)


def test_native_generation_limitation_policy_is_fail_closed() -> None:
    """Only an exact official_unavailable request may publish the limitation."""

    error = OfficialGenerationUnavailable(
        public_method=DPLM2_3B_GENERATION_LIMITATION["public_method"],
        exception_type=DPLM2_3B_GENERATION_LIMITATION["exception_type"],
        reason=DPLM2_3B_GENERATION_LIMITATION["reason"],
    )
    request = {
        "model_id": "dplm2_3b",
        "generation_policy": "official_unavailable",
        "official_generation_limitation": DPLM2_3B_GENERATION_LIMITATION,
    }
    assert _validated_generation_limitation(request, error) == (
        DPLM2_3B_GENERATION_LIMITATION
    )

    with pytest.raises(RuntimeError, match="official generation is required"):
        _validated_generation_limitation(
            {"model_id": "dplm2_3b", "generation_policy": "required"},
            error,
        )
    mutated = dict(request)
    mutated["official_generation_limitation"] = {
        **DPLM2_3B_GENERATION_LIMITATION,
        "reason": "different",
    }
    with pytest.raises(RuntimeError, match="differs from the manifest-derived request"):
        _validated_generation_limitation(mutated, error)
