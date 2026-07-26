"""Small contracts for isolated official-reference adapters."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from tests.parity.support.native_reference import (
    _adapter_reference_sources,
    _generation_contract,
    _record_generation_contract,
    _validated_generation_limitation,
)
from tests.parity.support.reference_adapters import OfficialGenerationUnavailable
from tests.parity.support.reference_adapters.dplm2 import (
    DPLM2_3B_GENERATION_LIMITATION,
    _accepts_type_ids,
    _call_checkpoint_forward,
    _call_checkpoint_generate,
)
from tools.remote.reference_source_attestation import ReferenceSourceAttestationError


class _AcceptsTypeIds(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # input_ids: (b, l); type_ids: (b, l) or None
        del type_ids
        return input_ids  # (b, l)


class _AcceptsKeywordArguments(nn.Module):
    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        # input_ids: (b, l)
        del kwargs
        return input_ids  # (b, l)


class _RejectsTypeIds(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids  # (b, l)


class _OfficialWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generation_calls: list[dict[str, object]] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids + 1  # (b, l)

    def generate(self, input_tokens: torch.Tensor, **kwargs: object) -> torch.Tensor:
        # input_tokens: (b, l)
        self.generation_calls.append({"input_tokens": input_tokens, **kwargs})
        return input_tokens + 2  # (b, l)


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
        # batch["input_ids"]: (b, l)
        self.generation_calls.append(
            {
                "batch": batch,
                "max_iter": max_iter,
                "sampling_strategy": sampling_strategy,
            }
        )
        return (  # each: (b, l)
            batch["input_ids"] + 3,
            torch.zeros_like(batch["input_ids"]),
        )


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
        tokens = batch["input_ids"]  # (b, l)
        tokens.ne(self.bos_id)
        raise AssertionError("unreachable")


class _AnkhGenerationTokenizer:
    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        if add_special_tokens:
            assert text == "M S T N P K"
            input_ids = torch.tensor([[4, 5, 1]])  # (b=1, l=3)
        else:
            assert text == "A C"
            input_ids = torch.tensor([[2, 3]])  # (b=1, l=2)
        return {
            "input_ids": input_ids,  # (b=1, l)
            "attention_mask": torch.ones_like(input_ids),  # (b=1, l)
        }


class _AnkhGenerationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(decoder_start_token_id=0)
        self.generation_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.generation_calls.append(kwargs)
        decoder_input_ids = kwargs["decoder_input_ids"]
        assert torch.is_tensor(decoder_input_ids)
        return torch.cat(  # (b=1, l + 1)
            (decoder_input_ids, decoder_input_ids.new_tensor([[9]])),  # (1, l); (1, 1)
            dim=1,
        )


class _AnkhGenerationAdapter:
    def __init__(self) -> None:
        self.model = _AnkhGenerationModel()
        self.loads: list[dict[str, object]] = []

    def load_official_seq2seq(
        self,
        **kwargs: object,
    ) -> tuple[_AnkhGenerationModel, _AnkhGenerationTokenizer]:
        self.loads.append(kwargs)
        return self.model, _AnkhGenerationTokenizer()


_VALID_SOURCE_ATTESTATION: dict[str, object] = {
    "attestation_sha256": "b" * 64,
    "file_count": 5218,
    "import_file": "src/transformers/__init__.py",
    "import_name": "transformers",
    "import_root": "src/transformers",
    "package_version": "4.57.6",
    "schema_version": 1,
    "source_revision": "a" * 40,
    "tree_sha256": "c" * 64,
}
_VALID_REFERENCE_SOURCES = {
    "biohub-esm": {
        **_VALID_SOURCE_ATTESTATION,
        "file_count": 157,
        "import_file": "esm/__init__.py",
        "import_name": "esm",
        "import_root": "esm",
        "package_version": "3.3.0",
        "source_revision": "d" * 40,
        "tree_sha256": "e" * 64,
    },
    "biohub-transformers": _VALID_SOURCE_ATTESTATION,
}


@pytest.mark.parametrize("family", ("esm_plusplus", "esm3", "esmfold2"))
def test_biohub_native_adapter_requires_both_stable_source_attestations(family: str) -> None:
    adapter = SimpleNamespace(
        reference_sources=lambda: {
            name: dict(evidence) for name, evidence in _VALID_REFERENCE_SOURCES.items()
        }
    )
    request = {"model_id": "biohub-probe", "family": family}

    assert _adapter_reference_sources(adapter, request) == _VALID_REFERENCE_SOURCES
    with pytest.raises(RuntimeError, match="omits source attestations"):
        _adapter_reference_sources(object(), request)

    incomplete = SimpleNamespace(
        reference_sources=lambda: {"biohub-transformers": _VALID_SOURCE_ATTESTATION}
    )
    with pytest.raises(ReferenceSourceAttestationError, match="names differ"):
        _adapter_reference_sources(incomplete, request)


def test_native_adapter_rejects_malformed_source_attestation() -> None:
    malformed = {
        **_VALID_SOURCE_ATTESTATION,
        "import_file": "/tmp/untrusted/transformers/__init__.py",
    }
    adapter = SimpleNamespace(
        reference_sources=lambda: {
            **_VALID_REFERENCE_SOURCES,
            "biohub-transformers": malformed,
        }
    )

    with pytest.raises(ReferenceSourceAttestationError, match="portable relative"):
        _adapter_reference_sources(
            adapter,
            {"model_id": "esmc_small", "family": "esm_plusplus"},
        )


def test_dplm2_checkpoint_forward_selection_is_signature_gated() -> None:
    """Unsupported modality keywords select the public checkpoint network."""

    assert _accepts_type_ids(_AcceptsTypeIds())
    assert _accepts_type_ids(_AcceptsKeywordArguments())

    model = _RejectsTypeIds()
    input_tensor = torch.tensor([1])  # (l=1,)
    type_ids = torch.tensor([0])  # (l=1,)
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
    input_tokens = torch.tensor([[1, 2]])  # (b=1, l=2)
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
    input_tokens = torch.tensor([[1, 2]])  # (b=1, l=2)
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


def test_native_ankh_generation_uses_an_explicit_decoder_prompt() -> None:
    adapter = _AnkhGenerationAdapter()
    request = {
        "model_id": "ankh_base",
        "family": "ankh",
        "generation_policy": "required",
        "reference_repo_id": "ElnaggarLab/ankh-base",
        "reference_revision": "immutable-revision",
        "seed": 42,
    }

    contract = _generation_contract(
        None,
        object(),
        request,
        torch.device("cpu"),
        adapter=adapter,
    )

    assert contract is not None
    assert contract["decoder_prompt_contract"] == "explicit-task-prompt"
    assert contract["decoder_input_ids"] == [[0, 2, 3]]
    assert contract["decoder_attention_mask"] == [[1, 1, 1]]
    assert contract["output_tokens"] == [[0, 2, 3, 9]]
    assert len(contract["decoder_input_fingerprint"]) == 64
    assert adapter.loads == [
        {
            "reference_repo_id": "ElnaggarLab/ankh-base",
            "reference_revision": "immutable-revision",
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
    ]
    call = adapter.model.generation_calls[0]
    assert torch.equal(call["input_ids"], torch.tensor([[4, 5, 1]]))
    assert torch.equal(call["decoder_input_ids"], torch.tensor([[0, 2, 3]]))
    assert call["do_sample"] is False


def test_native_generation_policy_rejects_missing_required_evidence() -> None:
    with pytest.raises(RuntimeError, match="has no generation contract"):
        _record_generation_contract(
            {},
            None,
            object(),
            {
                "model_id": "unsupported",
                "family": "esm2",
                "generation_policy": "required",
            },
            torch.device("cpu"),
            adapter=object(),
        )

    metadata: dict[str, object] = {}
    _record_generation_contract(
        metadata,
        None,
        object(),
        {
            "model_id": "esm2_8m",
            "family": "esm2",
            "generation_policy": "not_applicable",
        },
        torch.device("cpu"),
        adapter=object(),
    )
    assert metadata == {}
