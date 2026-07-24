"""Short H100 execution smoke for the external benchmark harness."""

from __future__ import annotations

import pytest
import torch
import transformers
from pathlib import Path
from types import SimpleNamespace

from benchmarks.run import (
    _load_model,
    _prepare_esmfold2_inputs,
    _run_esmfold2_esmc_projection,
    cuda_sample_ms,
    measure_blocks,
    prepare_inputs,
    warm_until_stable,
)
from fastplms.registry import get_model_registry


def test_prepare_inputs_counts_residues_not_special_tokens() -> None:
    """Keep logical throughput independent of tokenizer control tokens."""

    class FakeTokenizer:
        def __call__(
            self,
            sequences: list[str],
            *,
            return_tensors: str,
            padding: str,
            max_length: int,
            truncation: bool,
        ) -> dict[str, torch.Tensor]:
            assert return_tensors == "pt"
            assert padding == "max_length"
            assert max_length == 8
            assert truncation
            assert [len(sequence) for sequence in sequences] == [6, 3]

            input_ids = torch.zeros((2, max_length), dtype=torch.long)  # (b=2, l=8)
            attention_mask = torch.zeros_like(input_ids)  # (b=2, l=8)
            # Both sequences receive BOS and EOS control tokens.
            attention_mask[0, :8] = 1  # (l=8,)
            attention_mask[1, :5] = 1  # (l=5,)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class FakeModel:
        tokenizer = FakeTokenizer()

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            # input_ids: (b, l); attention_mask: (b, l)
            del attention_mask
            return input_ids  # (b, l)

    model_inputs, logical_tokens, padded_tokens, sequences = prepare_inputs(
        FakeModel(),
        "unused-local-model",
        (8, 5),
        torch.device("cpu"),
        revision=None,
        local_files_only=True,
    )

    assert [len(sequence) for sequence in sequences] == [6, 3]
    assert model_inputs["attention_mask"].sum().item() == 13
    assert logical_tokens == 9
    assert padded_tokens == 16


def test_local_artifact_model_load_omits_hub_revision_and_keeps_registry_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = tmp_path / "ESM2-8M"
    artifact.mkdir()
    calls: list[tuple[object, dict[str, object]]] = []

    class FakeModel:
        def eval(self) -> FakeModel:
            return self

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, source: object, **kwargs: object) -> FakeModel:
            calls.append((source, kwargs))
            return FakeModel()

    monkeypatch.setattr(transformers, "AutoModelForMaskedLM", FakeAutoModel)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    arguments = SimpleNamespace(
        model=spec.fast.repo_id,
        revision=spec.fast.revision,
        load_model=artifact,
        load_revision=None,
        auto_class="AutoModelForMaskedLM",
        backend="sdpa",
        precision="bf16",
        bf16_execution=spec.family.bf16_execution,
        mode="steady",
        local_files_only=True,
        esmc_load_model=None,
    )

    _load_model(arguments, torch)

    assert calls[0][0] == artifact
    assert "revision" not in calls[0][1]
    assert calls[0][1]["local_files_only"] is True
    expected_dtype = (
        torch.float32
        if spec.family.bf16_execution == "fp32_parameters_autocast"
        else torch.bfloat16
    )
    assert calls[0][1]["dtype"] == expected_dtype


def test_local_artifact_tokenizer_load_omits_hub_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "ESM2-8M"
    artifact.mkdir()
    calls: list[tuple[object, dict[str, object]]] = []

    class FakeTokenizer:
        def __call__(self, sequences: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
            del sequences, kwargs
            return {
                "input_ids": torch.zeros((1, 4), dtype=torch.long),  # (b=1, l=4)
                "attention_mask": torch.ones((1, 4), dtype=torch.long),  # (b=1, l=4)
            }

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, source: object, **kwargs: object) -> FakeTokenizer:
            calls.append((source, kwargs))
            return FakeTokenizer()

    class FakeModel:
        tokenizer = None

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            # input_ids: (b, l); attention_mask: (b, l)
            del attention_mask
            return input_ids  # (b, l)

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    prepare_inputs(
        FakeModel(),
        artifact,
        (4,),
        torch.device("cpu"),
        revision=None,
        local_files_only=True,
    )

    assert calls == [
        (
            artifact,
            {"trust_remote_code": True, "local_files_only": True},
        )
    ]


def test_local_esmfold2_load_uses_validated_local_esmc_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_model_registry()["esmfold2"]
    artifact = tmp_path / "ESMFold2"
    esmc_artifact = tmp_path / "ESMC-6B"
    artifact.mkdir()
    esmc_artifact.mkdir()
    top_level_calls: list[tuple[object, dict[str, object]]] = []
    esmc_calls: list[tuple[str, dict[str, object]]] = []

    class FakeModel:
        def eval(self) -> FakeModel:
            return self

        def load_esmc(self, source: str, **kwargs: object) -> None:
            esmc_calls.append((source, kwargs))

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, source: object, **kwargs: object) -> FakeModel:
            top_level_calls.append((source, kwargs))
            return FakeModel()

    monkeypatch.setattr(transformers, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    arguments = SimpleNamespace(
        model=spec.fast.repo_id,
        revision=spec.fast.revision,
        load_model=artifact,
        load_revision=None,
        auto_class="AutoModel",
        backend="sdpa",
        precision="bf16",
        bf16_execution=spec.family.bf16_execution,
        mode="esmfold2_embed",
        local_files_only=True,
        esmc_load_model=esmc_artifact,
    )

    _load_model(arguments, torch)

    assert top_level_calls[0][0] == artifact
    assert top_level_calls[0][1]["load_esmc"] is False
    assert "revision" not in top_level_calls[0][1]
    assert esmc_calls == [
        (
            str(esmc_artifact),
            {
                "precision": "bf16",
                "device": torch.device("cuda"),
                "local_files_only": True,
            },
        )
    ]


@pytest.mark.benchmark
@pytest.mark.gpu
def test_cuda_event_benchmark_smoke() -> None:
    """Exercise event timing and block accounting without benchmarking a model."""

    assert torch.cuda.is_available()
    X = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)  # (n=64, d=64)
    W = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)  # (d=64, d=64)

    def operation() -> torch.Tensor:
        return X @ W  # (n=64, d=64)

    warmup = warm_until_stable(
        torch,
        operation,
        window=2,
        tolerance=1.0,
        minimum_samples=4,
        maximum_samples=20,
    )
    blocks = measure_blocks(
        torch,
        operation,
        logical_tokens_per_forward=64,
        padded_tokens_per_forward=64,
        blocks=1,
        minimum_block_ms=1.0,
        minimum_forwards=2,
    )

    assert len(warmup) >= 4
    assert len(blocks) == 1
    assert blocks[0].forwards >= 2
    assert blocks[0].logical_tokens_per_second > 0
    assert blocks[0].padded_tokens_per_second > 0
    assert all(sample > 0 for sample in blocks[0].samples_ms)


@pytest.mark.benchmark
@pytest.mark.gpu
def test_esmfold2_esmc_projection_path_smoke() -> None:
    """Exercise residue preparation and the complete representation operation."""

    assert torch.cuda.is_available()

    class FakeESMFold2:
        def _compute_lm_hidden_states(
            self,
            input_ids: torch.Tensor,
            asym_id: torch.Tensor,
            residue_index: torch.Tensor,
            mol_type: torch.Tensor,
            residue_mask: torch.Tensor,
        ) -> torch.Tensor:
            # Each input tensor has shape (b, l).
            del asym_id, residue_index, mol_type
            H = input_ids.to(torch.bfloat16)[..., None, None].expand(  # (b, l, 81, d=4)
                -1, -1, 81, 4
            )
            return H * residue_mask[..., None, None]  # (b, l, 81, d=4)

        def project_esmc_hidden_states(
            self,
            hidden_states: torch.Tensor,
            residue_mask: torch.Tensor,
        ) -> torch.Tensor:
            # hidden_states: (b, l, 81, d); residue_mask: (b, l)
            Z = hidden_states.mean(dim=2)  # (b, l, d)
            return Z * residue_mask[..., None]  # (b, l, d)

    model_inputs, logical_tokens, padded_tokens = _prepare_esmfold2_inputs(
        torch, (7, 3)
    )  # tensor values: (b=2, l=7)
    assert logical_tokens == 10
    assert padded_tokens == 14
    assert model_inputs["residue_mask"].sum().item() == 10
    assert not model_inputs["residue_mask"][1, 3:].any()

    model = FakeESMFold2()

    def operation() -> torch.Tensor:
        with torch.inference_mode():
            return _run_esmfold2_esmc_projection(model, model_inputs)  # (b=2, l=7, d=4)

    elapsed_ms = cuda_sample_ms(torch, operation)
    Z = operation()  # (b=2, l=7, d=4)
    assert elapsed_ms >= 0.0
    assert Z.shape == (2, 7, 4)
    assert torch.count_nonzero(Z[1, 3:]) == 0
