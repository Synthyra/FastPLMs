"""Short H100 execution smoke for the external benchmark harness."""

from __future__ import annotations

import pytest
import torch

from benchmarks.run import (
    _prepare_esmfold2_inputs,
    _run_esmfold2_esmc_projection,
    cuda_sample_ms,
    measure_blocks,
    prepare_inputs,
    warm_until_stable,
)


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

            input_ids = torch.zeros((2, max_length), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            # Both sequences receive BOS and EOS control tokens.
            attention_mask[0, :8] = 1
            attention_mask[1, :5] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class FakeModel:
        tokenizer = FakeTokenizer()

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            del attention_mask
            return input_ids

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


@pytest.mark.benchmark
@pytest.mark.gpu
def test_cuda_event_benchmark_smoke() -> None:
    """Exercise event timing and block accounting without benchmarking a model."""

    assert torch.cuda.is_available()
    X = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
    W = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)

    def operation() -> torch.Tensor:
        return X @ W

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
            del asym_id, residue_index, mol_type
            # H represents ordered ESMC states with shape (b, l, 81, d_model).
            H = input_ids.to(torch.bfloat16)[..., None, None].expand(-1, -1, 81, 4)
            return H * residue_mask[..., None, None]

        def project_esmc_hidden_states(
            self,
            hidden_states: torch.Tensor,
            residue_mask: torch.Tensor,
        ) -> torch.Tensor:
            # Z is the learned-summary stand-in with shape (b, l, d_projection).
            Z = hidden_states.mean(dim=2)
            return Z * residue_mask[..., None]

    model_inputs, logical_tokens, padded_tokens = _prepare_esmfold2_inputs(torch, (7, 3))
    assert logical_tokens == 10
    assert padded_tokens == 14
    assert model_inputs["residue_mask"].sum().item() == 10
    assert not model_inputs["residue_mask"][1, 3:].any()

    model = FakeESMFold2()

    def operation() -> torch.Tensor:
        with torch.inference_mode():
            return _run_esmfold2_esmc_projection(model, model_inputs)

    elapsed_ms = cuda_sample_ms(torch, operation)
    Z = operation()
    assert elapsed_ms >= 0.0
    assert Z.shape == (2, 7, 4)
    assert torch.count_nonzero(Z[1, 3:]) == 0
