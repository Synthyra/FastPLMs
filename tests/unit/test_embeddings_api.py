from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import pytest
import torch
from pathlib import Path
from types import SimpleNamespace
from torch import nn

from fastplms.embeddings import (
    EmbeddingBatch,
    EmbeddingInput,
    EmbeddingRecord,
    EmbeddingResult,
    LazyTensorReference,
    Pooler,
    convert_legacy_sqlite,
    embed_dataset,
    garbage_collect_safetensors_generations,
    iter_fasta,
    load_legacy_pth,
    load_safetensors_result,
    load_sqlite_result,
    pagerank_weights,
    parse_fasta,
    save_safetensors_result,
    save_sqlite_result,
)
from fastplms.embeddings.storage import SafetensorsStreamWriter


class SyntheticEmbeddingModel(nn.Module):
    def __init__(self, backend: str = "eager") -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            model_type="synthetic",
            _name_or_path="synthetic/checkpoint",
            _commit_hash="abc123",
            _attn_implementation=backend,
        )

    def _embedding_batch(self, sequences: list[str]) -> EmbeddingBatch:
        b = len(sequences)
        sequence_length = max(map(len, sequences)) + 2
        X = torch.zeros(b, sequence_length, 2)
        M = torch.zeros(b, sequence_length, dtype=torch.bool)
        for batch_index, sequence in enumerate(sequences):
            for residue_index, residue in enumerate(sequence, start=1):
                X[batch_index, residue_index] = torch.tensor(
                    [float(ord(residue)), float(residue_index)]
                )
                M[batch_index, residue_index] = True
        # A uses two heads and includes BOS/EOS rows that M removes.
        A = torch.ones(b, 2, sequence_length, sequence_length)
        return EmbeddingBatch(X=X, residue_mask=M, attentions=(A, A * 2))


class InterruptibleEmbeddingModel(SyntheticEmbeddingModel):
    def __init__(self, fail_on_call: int | None) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call
        self.calls = 0

    def _embedding_batch(self, sequences: list[str]) -> EmbeddingBatch:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        return super()._embedding_batch(sequences)


class TrainingAwareEmbeddingModel(SyntheticEmbeddingModel):
    def __init__(self) -> None:
        super().__init__()
        self.observed_training: list[bool] = []

    def _embedding_batch(self, sequences: list[str]) -> EmbeddingBatch:
        self.observed_training.append(self.training)
        return super()._embedding_batch(sequences)


class SyntheticAllStatesModel(SyntheticEmbeddingModel):
    def _embedding_batch(
        self,
        sequences: list[str],
        *,
        store_all_hidden_states: bool = False,
    ) -> EmbeddingBatch:
        assert store_all_hidden_states is True
        batch = super()._embedding_batch(sequences)
        return EmbeddingBatch(
            X=torch.stack((batch.X, batch.X + 100), dim=1),
            residue_mask=batch.residue_mask,
        )


class SyntheticE1Preparer:
    boundary_token_ids = torch.tensor([0, 1, 2, 3])

    def get_batch_kwargs(self, sequences, device):
        assert sequences == ["AC"]
        return {"input_ids": torch.tensor([[0, 1, 5, 6, 2, 3]], device=device)}


class SyntheticE1Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(model_type="e1")
        self.prep_tokens = SyntheticE1Preparer()

    def _embed(self, sequences, return_attention_mask, **kwargs):
        del kwargs
        prepared = self.prep_tokens.get_batch_kwargs(sequences, self.anchor.device)
        X = prepared["input_ids"].float().unsqueeze(-1)
        M = torch.ones(X.shape[:2], dtype=torch.bool)
        assert return_attention_mask is True
        return X, M


class SyntheticDecoderEmbeddingModel(SyntheticEmbeddingModel):
    def __init__(self) -> None:
        super().__init__()
        self.config.model_type = "fast_ankh"
        self.observed_decoder_rows: list[list[int]] = []

    def _embedding_metadata(self, **context):
        return {
            "hidden_state_stack": context["hidden_state_source"],
            "source": context["hidden_state_source"],
        }

    def _embedding_batch(
        self,
        sequences: list[str],
        *,
        tokenizer,
        max_length,
        truncate,
        need_attentions,
        hidden_state_source,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        **kwargs,
    ) -> EmbeddingBatch:
        del tokenizer, max_length, truncate, need_attentions, kwargs
        assert hidden_state_source == "decoder"
        assert decoder_input_ids is not None
        assert decoder_attention_mask is not None
        self.observed_decoder_rows.append(decoder_input_ids[:, 0].tolist())
        X = decoder_input_ids.to(dtype=torch.float32).unsqueeze(-1)
        return EmbeddingBatch(X=X, residue_mask=decoder_attention_mask.to(dtype=torch.bool))


def test_result_preserves_order_and_duplicates() -> None:
    model = SyntheticEmbeddingModel()
    inputs = [
        EmbeddingInput("first", "ACD"),
        EmbeddingInput("second", "GG"),
        EmbeddingInput("first", "ACD"),
    ]
    result = embed_dataset(model, inputs, batch_size=2, pooling="mean")

    assert [record.id for record in result] == ["first", "second", "first"]
    assert [record.sequence for record in result] == ["ACD", "GG", "ACD"]
    assert torch.equal(result[0].load_tensor(), result[2].load_tensor())
    assert len(result.metadata["tensor_hashes"]) == 3
    assert result.metadata["outputs"][0]["sha256"] == result.metadata["tensor_hashes"][0]
    assert result.metadata["token_policy"]["unit"] == "residue"
    assert result.metadata["layer"] == -1
    with pytest.raises(ValueError, match="Duplicate id"):
        result.as_dict()
    assert list(result.as_dict(duplicates="first")) == ["first", "second"]


def test_bounded_length_bucketing_restores_input_order() -> None:
    model = SyntheticEmbeddingModel()
    observed: list[list[str]] = []
    original = model._embedding_batch

    def recording_batch(sequences: list[str]) -> EmbeddingBatch:
        observed.append(list(sequences))
        return original(sequences)

    model._embedding_batch = recording_batch  # type: ignore[method-assign]
    inputs = ["A", "BBBB", "CC", "DDD"]
    result = embed_dataset(
        model,
        inputs,
        batch_size=2,
        max_tokens_per_batch=8,
    )

    assert observed == [["BBBB", "DDD"], ["CC", "A"]]
    assert [record.sequence for record in result] == inputs
    assert result.metadata["batching"]["batch_window_size"] == 32
    assert result.metadata["batching"]["ordering"] == ("bounded-length-bucketed-stable-output")


def test_invalid_storage_and_pooling_fail_before_input_consumption(tmp_path: Path) -> None:
    consumed = False

    def inputs():
        nonlocal consumed
        consumed = True
        yield "ACD"

    with pytest.raises(ValueError, match="format must"):
        embed_dataset(
            SyntheticEmbeddingModel(),
            inputs(),
            output=tmp_path / "output",
            format="unknown",
        )
    assert consumed is False

    with pytest.raises(ValueError, match="cannot be combined"):
        embed_dataset(
            SyntheticEmbeddingModel(),
            ["ACD"],
            full_embeddings=True,
            pooling="mean",
        )


def test_decoder_companions_are_fingerprinted_and_bucket_aligned() -> None:
    model = SyntheticDecoderEmbeddingModel()
    inputs = ["A", "BBBB", "CC"]
    decoder_input_ids = torch.tensor([[11, 11], [22, 22], [33, 33]])
    decoder_attention_mask = torch.ones_like(decoder_input_ids)

    result = embed_dataset(
        model,
        inputs,
        hidden_state_source="decoder",
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        batch_size=2,
        batch_window_size=3,
    )

    assert model.observed_decoder_rows == [[22, 33], [11]]
    assert [record.load_tensor().item() for record in result] == [11.0, 22.0, 33.0]
    assert result.metadata["hidden_state_source"] == "decoder"
    assert result.metadata["decoder_input_fingerprint"]
    assert result.metadata["decoder_attention_mask_fingerprint"]
    assert result.metadata["decoder_alignment"] == "input-position"
    assert result.metadata["model_embedding"] == {
        "hidden_state_stack": "decoder",
        "source": "decoder",
    }

    with pytest.raises(ValueError, match="exactly one"):
        embed_dataset(model, inputs, hidden_state_source="decoder")


def test_decoder_embeddings_require_an_explicit_model_capability() -> None:
    with pytest.raises(ValueError, match="does not declare decoder embedding support"):
        embed_dataset(
            SyntheticEmbeddingModel(),
            ["AC"],
            hidden_state_source="decoder",
            decoder_input_ids=torch.tensor([[3, 4]]),
        )


def test_mapping_inputs_embed_values_with_mapping_keys_as_ids() -> None:
    inputs = {
        "protein-a": "ACD",
        "protein-b": "GG",
    }

    result = embed_dataset(SyntheticEmbeddingModel(), inputs, pooling="mean")

    assert [record.id for record in result] == ["protein-a", "protein-b"]
    assert [record.sequence for record in result] == ["ACD", "GG"]
    assert result[0].load_tensor()[0].item() == pytest.approx(sum(map(ord, "ACD")) / len("ACD"))
    with pytest.raises(ValueError, match="at least one sequence"):
        embed_dataset(SyntheticEmbeddingModel(), {}, pooling="mean")


def test_embedding_temporarily_uses_eval_and_restores_training_state() -> None:
    model = TrainingAwareEmbeddingModel()
    model.train()

    embed_dataset(model, ["ACD"])

    assert model.observed_training == [False]
    assert model.training is True

    interrupted = InterruptibleEmbeddingModel(fail_on_call=1)
    interrupted.train()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        embed_dataset(interrupted, ["ACD"])
    assert interrupted.training is True


def test_full_embeddings_contain_biological_residues_only() -> None:
    result = embed_dataset(
        SyntheticEmbeddingModel(),
        ["ACD", "GG"],
        full_embeddings=True,
    )
    assert tuple(result[0].load_tensor().shape) == (3, 2)
    assert tuple(result[1].load_tensor().shape) == (2, 2)
    assert result.metadata["residue_mask_policy"] == "biological-residues-only"


@pytest.mark.parametrize("format", ("safetensors", "sqlite"))
def test_all_hidden_state_embeddings_trim_token_axis_and_round_trip(
    tmp_path: Path,
    format: str,
) -> None:
    output = tmp_path / ("all-states.sqlite" if format == "sqlite" else "all-states")
    result = embed_dataset(
        SyntheticAllStatesModel(),
        ["ACD", "GG"],
        full_embeddings=True,
        store_all_hidden_states=True,
        output=output,
        format=format,
    )

    assert tuple(result[0].load_tensor().shape) == (2, 3, 2)
    assert tuple(result[1].load_tensor().shape) == (2, 2, 2)
    assert torch.equal(result[0].load_tensor()[1], result[0].load_tensor()[0] + 100)
    loaded = load_sqlite_result(output) if format == "sqlite" else load_safetensors_result(output)
    assert torch.equal(loaded[0].load_tensor(), result[0].load_tensor())
    assert loaded.metadata["record_count"] == 2
    assert loaded.metadata["descriptor_index"] in {
        "sqlite-records",
        "safetensors-generation-index",
    }
    assert "outputs" not in loaded.metadata
    assert "tensor_hashes" not in loaded.metadata


def test_all_hidden_states_require_full_embeddings() -> None:
    with pytest.raises(
        ValueError,
        match="store_all_hidden_states=True requires full_embeddings=True",
    ):
        embed_dataset(
            SyntheticAllStatesModel(),
            ["ACD"],
            store_all_hidden_states=True,
        )


def test_embedding_fingerprint_records_loaded_esmc_identity() -> None:
    model = SyntheticEmbeddingModel()
    model._esmc_source = "Synthyra/ESMplusplus_6B"
    model._esmc_source_revision = "a" * 40
    model._esmc_source_files = {"model.safetensors": "sha256:" + "b" * 64}

    result = embed_dataset(model, ["ACD"], pooling="mean")

    assert result.metadata["esmc_source"] == model._esmc_source
    assert result.metadata["esmc_revision"] == model._esmc_source_revision
    assert result.metadata["esmc_files"] == model._esmc_source_files

    model._esmc_source_revision = "c" * 40
    changed = embed_dataset(model, ["ACD"], pooling="mean")
    assert changed.metadata["run_fingerprint"] != result.metadata["run_fingerprint"]


def test_embedding_fingerprint_binds_persisted_parameters_and_buffers(
    tmp_path: Path,
) -> None:
    model = SyntheticEmbeddingModel()
    model.register_buffer("running_value", torch.tensor([3.0]))
    initial = embed_dataset(model, ["ACD"], output=tmp_path / "initial")

    with torch.no_grad():
        model.anchor.fill_(1)
    changed_parameter = embed_dataset(model, ["ACD"], output=tmp_path / "parameter")
    assert (
        changed_parameter.metadata["model_state_fingerprint"]
        != (initial.metadata["model_state_fingerprint"])
    )
    assert changed_parameter.metadata["run_fingerprint"] != initial.metadata["run_fingerprint"]

    model.running_value.add_(1)
    changed_buffer = embed_dataset(model, ["ACD"], output=tmp_path / "buffer")
    assert (
        changed_buffer.metadata["model_state_fingerprint"]
        != (changed_parameter.metadata["model_state_fingerprint"])
    )
    assert initial.metadata["fingerprint_schema_version"] == 3
    assert initial.metadata["model_state_fingerprint_source"] == "computed"


def test_model_state_fingerprint_rehashes_data_and_storage_alias_mutations(
    tmp_path: Path,
) -> None:
    model = SyntheticEmbeddingModel()
    original_path = tmp_path / "original"
    original = embed_dataset(model, ["ACD"], output=original_path)

    # ``Parameter.data`` mutation bypasses autograd's version counter. A state
    # fingerprint must still derive from current bytes rather than object/version
    # metadata retained by a cache.
    model.anchor.data.fill_(1)
    data_mutated = embed_dataset(model, ["ACD"], output=tmp_path / "data-mutated")
    assert (
        data_mutated.metadata["model_state_fingerprint"]
        != (original.metadata["model_state_fingerprint"])
    )
    assert data_mutated.metadata["run_fingerprint"] != original.metadata["run_fingerprint"]
    with pytest.raises(ValueError, match="different run fingerprint"):
        embed_dataset(model, ["ACD"], output=original_path)

    storage_alias = model.anchor.data
    storage_alias.fill_(2)
    alias_mutated = embed_dataset(model, ["ACD"], output=tmp_path / "alias-mutated")
    assert (
        alias_mutated.metadata["model_state_fingerprint"]
        != (data_mutated.metadata["model_state_fingerprint"])
    )
    assert alias_mutated.metadata["run_fingerprint"] != (data_mutated.metadata["run_fingerprint"])


def test_in_memory_embedding_skips_model_state_hash() -> None:
    class NoStateDictEmbeddingModel(SyntheticEmbeddingModel):
        def state_dict(
            self,
            *args: object,
            **kwargs: object,
        ) -> dict[str, torch.Tensor]:
            del args, kwargs
            raise AssertionError("in-memory embeddings must not hash the full model state")

    result = embed_dataset(NoStateDictEmbeddingModel(), ["ACD"])

    assert result.metadata["model_state_fingerprint"] is None
    assert result.metadata["model_state_fingerprint_source"] == "not-computed"


def test_caller_owned_model_state_fingerprint_overrides_state_hash() -> None:
    model = SyntheticEmbeddingModel()
    first = embed_dataset(model, ["ACD"], model_state_fingerprint="external-state-v1")
    with torch.no_grad():
        model.anchor.fill_(9)
    second = embed_dataset(model, ["ACD"], model_state_fingerprint="external-state-v1")

    assert second.metadata["run_fingerprint"] == first.metadata["run_fingerprint"]
    assert second.metadata["model_state_fingerprint"] == "external-state-v1"
    assert second.metadata["model_state_fingerprint_source"] == "caller"
    with pytest.raises(ValueError, match="must not be empty"):
        embed_dataset(model, ["ACD"], model_state_fingerprint="  ")


def test_runtime_versions_are_part_of_resume_identity(monkeypatch) -> None:
    import fastplms.embeddings.runner as runner

    model = SyntheticEmbeddingModel()
    first = embed_dataset(model, ["ACD"])
    versions = runner._software_versions()
    monkeypatch.setattr(
        runner,
        "_software_versions",
        lambda: {**versions, "torch": "different-runtime"},
    )
    changed = embed_dataset(model, ["ACD"])

    assert changed.metadata["run_fingerprint"] != first.metadata["run_fingerprint"]


def test_tokenizer_content_changes_run_fingerprint() -> None:
    class Tokenizer:
        all_special_ids = (0, 2)
        name_or_path = "synthetic/tokenizer"
        vocab_size = 4
        model_max_length = 32
        padding_side = "right"
        truncation_side = "right"

        def __init__(self, vocab, *, mode: str = "first") -> None:
            self._vocab = vocab
            self.init_kwargs = {"mode": mode}
            self.special_tokens_map = {"bos_token": "<bos>", "eos_token": "<eos>"}

        def get_vocab(self):
            return self._vocab

    model = SyntheticEmbeddingModel()
    first = embed_dataset(
        model,
        ["ACD"],
        tokenizer=Tokenizer({"<bos>": 0, "A": 1, "<eos>": 2, "D": 3}),
    )
    changed_vocab = embed_dataset(
        model,
        ["ACD"],
        tokenizer=Tokenizer({"<bos>": 0, "D": 1, "<eos>": 2, "A": 3}),
    )
    changed_config = embed_dataset(
        model,
        ["ACD"],
        tokenizer=Tokenizer(
            {"<bos>": 0, "A": 1, "<eos>": 2, "D": 3},
            mode="second",
        ),
    )

    assert (
        first.metadata["tokenizer"]["content_sha256"]
        != (changed_vocab.metadata["tokenizer"]["content_sha256"])
    )
    assert first.metadata["run_fingerprint"] != changed_vocab.metadata["run_fingerprint"]
    assert first.metadata["run_fingerprint"] != changed_config.metadata["run_fingerprint"]


def test_native_sequence_tokenizer_loader_context_is_bound_without_secret_values() -> None:
    model = SyntheticEmbeddingModel()
    model.__dict__["_fastplms_tokenizer_kwargs"] = {
        "tokenizer_source": "Synthyra/Profluent-E1-150M",
        "revision": "tokenizer-revision-a",
        "cache_dir": "/immutable/cache",
        "local_files_only": True,
        "token": "do-not-persist-this-token",
    }
    first = embed_dataset(model, ["ACD"])
    model.__dict__["_fastplms_tokenizer_kwargs"]["revision"] = "tokenizer-revision-b"
    changed = embed_dataset(model, ["ACD"])

    assert first.metadata["tokenizer"] == {
        "mode": "native-sequence",
        "source": "Synthyra/Profluent-E1-150M",
        "revision": "tokenizer-revision-a",
        "cache_dir": "/immutable/cache",
        "local_files_only": True,
        "token_policy": "provided",
    }
    assert "do-not-persist-this-token" not in json.dumps(first.metadata)
    assert first.metadata["run_fingerprint"] != changed.metadata["run_fingerprint"]


def test_local_artifact_identity_fills_embedding_provenance() -> None:
    model = SyntheticEmbeddingModel()
    model.config._name_or_path = "dist/hub/ESM2-8M"
    model.config._commit_hash = None
    model.config.fastplms_model_id = "esm2_8m"
    model.config.fastplms_checkpoint_repo_id = "Synthyra/ESM2-8M"
    model.config.fastplms_checkpoint_revision = "d" * 40
    model.config.fastplms_checkpoint_hash = "e" * 64

    result = embed_dataset(model, ["ACD"], pooling="mean")

    assert result.metadata["model_id"] == "esm2_8m"
    assert result.metadata["model_revision"] == "d" * 40
    assert result.metadata["checkpoint_repo_id"] == "Synthyra/ESM2-8M"
    assert result.metadata["checkpoint_revision"] == "d" * 40
    assert result.metadata["checkpoint_hash"] == "e" * 64

    model.config.fastplms_checkpoint_hash = "f" * 64
    changed = embed_dataset(model, ["ACD"], pooling="mean")
    assert changed.metadata["run_fingerprint"] != result.metadata["run_fingerprint"]


def test_e1_native_path_removes_all_boundary_tokens() -> None:
    result = embed_dataset(SyntheticE1Model(), ["AC"], pooling="mean")
    assert torch.equal(result[0].load_tensor(), torch.tensor([5.5]))


def test_generic_embedding_uses_a_model_sequence_tokenizer_adapter() -> None:
    class Tokenizer:
        all_special_ids = (0, 2)
        name_or_path = "synthetic/multimodal"
        vocab_size = 8

        def __call__(self, *args, **kwargs):
            raise AssertionError("The generic tokenizer path must not run.")

    class AdaptedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(model_type="synthetic-multimodal")
            self.tokenizer = Tokenizer()
            self.sequences: list[str] | None = None

        def _tokenize_sequence_batch(self, sequences, *, tokenizer, **kwargs):
            self.sequences = list(sequences)
            assert tokenizer is self.tokenizer
            assert kwargs == {"return_tensors": "pt", "padding": True, "truncation": True}
            return {
                "input_ids": torch.tensor([[0, 4, 5, 2]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }

        def _embed(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            return input_ids.float().unsqueeze(-1)

    model = AdaptedModel()
    result = embed_dataset(model, ["AC"], full_embeddings=True)

    assert model.sequences == ["AC"]
    assert torch.equal(result[0].load_tensor(), torch.tensor([[4.0], [5.0]]))


def test_max_length_counts_biological_residues_not_special_tokens() -> None:
    class Tokenizer:
        all_special_ids = (0, 2)
        name_or_path = "synthetic/residue-limit"
        vocab_size = 8

        def num_special_tokens_to_add(self, *, pair: bool) -> int:
            assert pair is False
            return 2

        def __call__(self, sequences, **kwargs):
            assert sequences == ["ACD"]
            assert kwargs["max_length"] == 5
            return {
                "input_ids": torch.tensor([[0, 3, 4, 5, 2]]),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            }

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(model_type="synthetic")

        def _embed(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            return input_ids.float().unsqueeze(-1)

    result = embed_dataset(
        Model(),
        ["ACDE"],
        tokenizer=Tokenizer(),
        max_length=3,
        full_embeddings=True,
    )

    assert result[0].load_tensor().shape == (3, 1)


@pytest.mark.parametrize("input_kind", ("list", "mapping", "generator", "fasta"))
def test_truncate_false_rejects_overlength_inputs_before_custom_adapter_inference(
    tmp_path: Path,
    input_kind: str,
) -> None:
    model = SyntheticEmbeddingModel()
    inference_called = False

    def fail_if_called(sequences: list[str]) -> EmbeddingBatch:
        del sequences
        nonlocal inference_called
        inference_called = True
        raise AssertionError("over-length input reached inference")

    model._embedding_batch = fail_if_called  # type: ignore[method-assign]
    records = [
        EmbeddingInput("short", "AC"),
        EmbeddingInput("too-long", "ACDE"),
    ]
    if input_kind == "list":
        inputs = records
    elif input_kind == "mapping":
        inputs = {record.id: record.sequence for record in records}
    elif input_kind == "generator":
        inputs = (record for record in records)
    else:
        inputs = tmp_path / "proteins.fasta"
        inputs.write_text(">short\nAC\n>too-long\nACDE\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        embed_dataset(
            model,
            inputs,
            max_length=3,
            truncate=False,
        )

    message = str(error.value)
    assert "position 1" in message
    assert "id 'too-long'" in message
    assert "4 biological residues" in message
    assert "max_length=3" in message
    assert inference_called is False


def test_truncate_false_rejects_overlength_inputs_before_raw_adapter_inference() -> None:
    model = SyntheticE1Model()

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("over-length input reached raw adapter inference")

    model._embed = fail_if_called  # type: ignore[method-assign]
    with pytest.raises(ValueError, match=r"position 0.*id '0'.*max_length=3"):
        embed_dataset(
            model,
            ["ACDE"],
            max_length=3,
            truncate=False,
        )


def test_all_poolers_and_output_slices() -> None:
    names = ("mean", "max", "norm", "median", "std", "var", "cls", "parti")
    result = embed_dataset(SyntheticEmbeddingModel(), ["ACD", "GG"], pooling=names)
    assert tuple(result[0].load_tensor().shape) == (16,)
    assert torch.isfinite(result[0].load_tensor()).all()
    assert result.metadata["pool_slices"]["mean"] == (0, 2)
    assert result.metadata["pool_slices"]["parti"] == (14, 16)


def test_poolers_ignore_nonfinite_excluded_positions_and_reject_nonfinite_output() -> None:
    X = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [torch.nan, torch.inf]]])
    M = torch.tensor([[True, True, False]])

    pooled = Pooler(("mean", "norm", "std", "var"))(X, M)

    assert torch.isfinite(pooled).all()
    torch.testing.assert_close(
        pooled,
        torch.tensor(
            [
                [
                    2.0,
                    3.0,
                    10.0**0.5,
                    20.0**0.5,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ]
        ),
    )
    with pytest.raises(ValueError, match="produced non-finite output"):
        Pooler("mean")(torch.tensor([[[torch.nan], [1.0]]]), torch.tensor([[True, False]]))


def test_parti_requires_eager_attention() -> None:
    with pytest.raises(ValueError, match="requires attn_implementation='eager'"):
        embed_dataset(SyntheticEmbeddingModel("sdpa"), ["ACD"], pooling="parti")


def test_parti_rejects_overlength_input_before_model_inference() -> None:
    class Tokenizer:
        all_special_ids = (0, 2)
        vocab_size = 3
        name_or_path = "synthetic/tokenizer"

        def __call__(self, sequences, **kwargs):
            del kwargs
            assert sequences == ["A" * 2_049]
            input_ids = torch.tensor([[0, *([1] * 2_049), 2]])
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                model_type="synthetic",
                _attn_implementation="eager",
            )

        def _embed(self, *args, **kwargs):
            raise AssertionError("parti length validation must run before inference")

    with pytest.raises(ValueError, match="at most 2,048 biological residues"):
        embed_dataset(
            Model(),
            ["A" * 2_049],
            pooling="parti",
            tokenizer=Tokenizer(),
        )


def test_torch_pagerank_handles_dangling_rows() -> None:
    A = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    w = pagerank_weights(A)
    assert torch.isclose(w.sum(), torch.tensor(1.0))
    assert bool((w > 0).all())


def test_fasta_preserves_headers_order_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "proteins.fasta"
    path.write_text(">a description\nACD\n>a\nGG\n", encoding="utf-8")
    records = parse_fasta(path)
    assert records == [EmbeddingInput("a", "ACD"), EmbeddingInput("a", "GG")]


def test_fasta_parser_streams_without_path_read_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "stream.fasta"
    path.write_text(">p1\nAC\nD\n>p2\nGG\n", encoding="utf-8")

    def reject_read_text(*args, **kwargs):
        del args, kwargs
        raise AssertionError("FASTA parsing must stream from the file handle")

    monkeypatch.setattr(Path, "read_text", reject_read_text)
    assert [(record.id, record.sequence) for record in iter_fasta(path)] == [
        ("p1", "ACD"),
        ("p2", "GG"),
    ]


@pytest.mark.parametrize("source_kind", ("generator", "fasta"))
def test_large_streaming_inputs_use_bounded_disk_windows(
    monkeypatch,
    tmp_path,
    source_kind: str,
) -> None:
    import fastplms.embeddings.runner as runner

    count = 2_050
    iterations = 0
    expected_input_digest = hashlib.sha256()
    source_liveness = {"live": 0, "peak": 0}

    class TrackedText:
        def __init__(self, value: str) -> None:
            self.value = value
            source_liveness["live"] += 1
            source_liveness["peak"] = max(source_liveness["peak"], source_liveness["live"])
            if source_liveness["live"] > 32:
                raise AssertionError(
                    "The source generator was fully materialized before disk spooling."
                )

        def __str__(self) -> str:
            return self.value

        def __del__(self) -> None:
            source_liveness["live"] -= 1

    def update_expected_fingerprint(input_id: str, sequence: str) -> None:
        for value in (input_id, sequence):
            encoded = value.encode("utf-8")
            expected_input_digest.update(len(encoded).to_bytes(8, "big"))
            expected_input_digest.update(encoded)

    class SinglePassInputs:
        def __iter__(self):
            nonlocal iterations
            iterations += 1
            if iterations > 1:
                raise AssertionError("The source iterable was consumed more than once.")
            for position in range(count):
                input_id = f"protein-{position}"
                sequence = "A" * (position % 11 + 1)
                update_expected_fingerprint(input_id, sequence)
                yield (TrackedText(input_id), TrackedText(sequence))

    if source_kind == "generator":
        inputs = SinglePassInputs()
    else:
        fasta = tmp_path / "large.fasta"
        with fasta.open("w", encoding="utf-8") as handle:
            for position in range(count):
                input_id = f"protein-{position}"
                sequence = "A" * (position % 11 + 1)
                update_expected_fingerprint(input_id, sequence)
                handle.write(f">{input_id}\n{sequence}\n")
        inputs = fasta

    observed_slice_widths: list[int] = []
    original_getitem = runner._InputSpool.__getitem__

    def tracking_getitem(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                observed_slice_widths.append(stop - start)
        return original_getitem(self, index)

    def reject_full_spool_iteration(self):
        del self
        raise AssertionError("The disk spool must be consumed through bounded slices.")

    monkeypatch.setattr(runner._InputSpool, "__getitem__", tracking_getitem)
    monkeypatch.setattr(runner._InputSpool, "__iter__", reject_full_spool_iteration)
    output = tmp_path / f"{source_kind}.sqlite"
    result = embed_dataset(
        SyntheticEmbeddingModel(),
        inputs,
        batch_size=64,
        batch_window_size=127,
        output=output,
        format="sqlite",
    )

    assert len(result) == count
    for position, record in enumerate(result):
        assert record.id == f"protein-{position}"
        assert record.sequence == "A" * (position % 11 + 1)
    assert observed_slice_widths
    assert max(observed_slice_widths) <= 127
    assert result.metadata["batching"]["input_storage"] == "disk-spool"
    expected_input_digest.update(count.to_bytes(8, "big"))
    assert result.metadata["input_fingerprint"] == expected_input_digest.hexdigest()
    reopened = load_sqlite_result(output)
    assert reopened.metadata["input_fingerprint"] == result.metadata["input_fingerprint"]
    assert reopened.metadata["run_fingerprint"] == result.metadata["run_fingerprint"]
    if source_kind == "generator":
        assert iterations == 1
        assert source_liveness["peak"] <= 32
        assert source_liveness["live"] == 0


def test_sqlite_round_trip_is_lazy_and_bf16_lossless(tmp_path: Path) -> None:
    source = embed_dataset(SyntheticEmbeddingModel(), ["ACD", "GG"])
    source = EmbeddingResult(
        [
            type(record)(record.id, record.sequence, record.load_tensor().to(torch.bfloat16))
            for record in source
        ],
        source.metadata,
    )
    path = tmp_path / "embeddings.sqlite"
    saved = save_sqlite_result(source, path)
    loaded = load_sqlite_result(path)
    assert isinstance(saved[0].tensor, LazyTensorReference)
    assert torch.equal(loaded[0].load_tensor(), source[0].load_tensor())
    assert loaded[0].load_tensor().dtype == torch.bfloat16


def test_sqlite_tensor_corruption_is_detected_on_materialization(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.sqlite"
    source = EmbeddingResult(
        [EmbeddingRecord("protein", "AC", torch.tensor([1.0, 2.0]))],
        {"run_fingerprint": "corrupt-sqlite", "complete": True},
    )
    save_sqlite_result(source, path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE tensors SET data = ?",
            (sqlite3.Binary(torch.tensor([3.0, 4.0]).numpy().tobytes()),),
        )
        connection.commit()

    with pytest.raises(ValueError, match="failed SHA-256 verification"):
        load_sqlite_result(path)[0].load_tensor()


def test_sqlite_loading_and_lazy_tensor_reads_use_read_only_connections(
    monkeypatch,
    tmp_path,
) -> None:
    import fastplms.embeddings.storage as storage

    path = tmp_path / "readonly.sqlite"
    save_sqlite_result(embed_dataset(SyntheticEmbeddingModel(), ["ACD"]), path)
    original_connect = storage.sqlite3.connect
    observed: list[tuple[object, dict[str, object]]] = []

    def tracking_connect(database, *args, **kwargs):
        observed.append((database, dict(kwargs)))
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(storage.sqlite3, "connect", tracking_connect)
    loaded = load_sqlite_result(path)
    loaded[0].load_tensor()

    # Loading metadata, resolving the lazy descriptor, and loading its tensor
    # are independent read-only operations.
    assert len(observed) == 3
    assert all("mode=ro" in str(database) for database, _ in observed)
    assert all(kwargs.get("uri") is True for _, kwargs in observed)


def test_sqlite_filtered_retrieval_preserves_selector_order_and_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection.sqlite"
    result = EmbeddingResult(
        [
            EmbeddingRecord("x", "AA", torch.tensor([0.0])),
            EmbeddingRecord("y", "BB", torch.tensor([1.0])),
            EmbeddingRecord("x", "AA", torch.tensor([2.0])),
        ],
        {"run_fingerprint": "selection-run", "complete": True},
    )
    save_sqlite_result(result, path)

    by_position = load_sqlite_result(path, positions=[2, 0, 2])
    assert [record.load_tensor().item() for record in by_position] == [2.0, 0.0, 2.0]
    by_id = load_sqlite_result(path, record_ids=["x", "y", "x"])
    assert [record.load_tensor().item() for record in by_id] == [
        0.0,
        2.0,
        1.0,
        0.0,
        2.0,
    ]
    by_sequence = load_sqlite_result(path, sequences=["BB", "AA"])
    assert [record.id for record in by_sequence] == ["y", "x", "x"]


def test_legacy_sqlite_converter_accepts_compact_blobs_without_pickle(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.sqlite"
    output = tmp_path / "converted.sqlite"
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    shape = tuple(tensor.shape)
    blob = (
        struct.pack(
            f"<BBi{len(shape)}i",
            1,
            2,
            len(shape),
            *shape,
        )
        + tensor.numpy().tobytes()
    )
    with sqlite3.connect(source) as connection:
        connection.execute(
            "CREATE TABLE embeddings (sequence TEXT PRIMARY KEY, embedding BLOB NOT NULL)"
        )
        connection.execute("INSERT INTO embeddings VALUES (?, ?)", ("AC", blob))
        connection.commit()

    converted = convert_legacy_sqlite(source, output)

    assert [(record.id, record.sequence) for record in converted] == [("0", "AC")]
    assert torch.equal(converted[0].load_tensor(), tensor)
    assert converted.metadata["source_format"] == "legacy-fastplms-sqlite-v0"
    with pytest.raises(ValueError, match="different output path"):
        convert_legacy_sqlite(source, source)


def test_sqlite_streaming_resumes_an_ordered_prefix(tmp_path: Path) -> None:
    path = tmp_path / "stream.sqlite"
    inputs = ["ACD", "GG", "M"]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        embed_dataset(
            InterruptibleEmbeddingModel(fail_on_call=2),
            inputs,
            batch_size=1,
            batch_window_size=1,
            output=path,
            format="sqlite",
        )
    partial = load_sqlite_result(path)
    assert len(partial) == 1
    assert partial.metadata["complete"] is False

    resumed = embed_dataset(
        InterruptibleEmbeddingModel(fail_on_call=None),
        inputs,
        batch_size=1,
        batch_window_size=1,
        output=path,
        format="sqlite",
    )
    assert [record.sequence for record in resumed] == inputs
    assert resumed.metadata["complete"] is True
    assert resumed.metadata["record_count"] == 3
    assert resumed.metadata["descriptor_index"] == "sqlite-records"
    assert "outputs" not in resumed.metadata


def test_sqlite_successful_overwrite_becomes_default_and_retains_prior_run(
    tmp_path,
) -> None:
    path = tmp_path / "overwrite.sqlite"
    model = SyntheticEmbeddingModel()
    original = embed_dataset(
        model,
        ["AC"],
        output=path,
        format="sqlite",
    )
    original_run_id = original.metadata["run_fingerprint"]
    original_tensor = original[0].load_tensor().clone()

    replacement = embed_dataset(
        model,
        ["GG", "M"],
        batch_size=1,
        batch_window_size=1,
        output=path,
        format="sqlite",
        resume=False,
    )

    current = load_sqlite_result(path)
    retained = load_sqlite_result(path, run_id=original_run_id)
    assert current.metadata["run_fingerprint"] == replacement.metadata["run_fingerprint"]
    assert current.metadata["run_fingerprint"] != original_run_id
    assert [record.sequence for record in current] == ["GG", "M"]
    assert current.metadata["complete"] is True
    assert [record.sequence for record in retained] == ["AC"]
    torch.testing.assert_close(retained[0].load_tensor(), original_tensor, rtol=0.0, atol=0.0)
    # Readers opened before the replacement remain bound to their explicit run.
    torch.testing.assert_close(original[0].load_tensor(), original_tensor, rtol=0.0, atol=0.0)


def test_interrupted_sqlite_overwrite_retains_prior_run_and_resumable_prefix(
    tmp_path,
) -> None:
    path = tmp_path / "interrupted-overwrite.sqlite"
    original = embed_dataset(
        InterruptibleEmbeddingModel(fail_on_call=None),
        ["AC"],
        output=path,
        format="sqlite",
    )
    original_run_id = original.metadata["run_fingerprint"]
    original_tensor = original[0].load_tensor().clone()
    replacement_inputs = ["GG", "M"]

    with pytest.raises(RuntimeError, match="simulated interruption"):
        embed_dataset(
            InterruptibleEmbeddingModel(fail_on_call=2),
            replacement_inputs,
            batch_size=1,
            batch_window_size=1,
            output=path,
            format="sqlite",
            resume=False,
        )

    partial = load_sqlite_result(path)
    retained = load_sqlite_result(path, run_id=original_run_id)
    assert partial.metadata["run_fingerprint"] != original_run_id
    assert partial.metadata["complete"] is False
    assert [record.sequence for record in partial] == replacement_inputs[:1]
    assert [record.sequence for record in retained] == ["AC"]
    torch.testing.assert_close(retained[0].load_tensor(), original_tensor, rtol=0.0, atol=0.0)

    resumed = embed_dataset(
        InterruptibleEmbeddingModel(fail_on_call=None),
        replacement_inputs,
        batch_size=1,
        batch_window_size=1,
        output=path,
        format="sqlite",
    )
    assert resumed.metadata["run_fingerprint"] == partial.metadata["run_fingerprint"]
    assert resumed.metadata["complete"] is True
    assert [record.sequence for record in resumed] == replacement_inputs
    assert [record.sequence for record in load_sqlite_result(path, run_id=original_run_id)] == [
        "AC"
    ]


def test_sqlite_first_batch_publication_is_atomic_and_hidden_run_resumes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "first-batch-atomic.sqlite"
    original = embed_dataset(
        InterruptibleEmbeddingModel(fail_on_call=None),
        ["AC"],
        output=path,
        format="sqlite",
    )
    original_run_id = original.metadata["run_fingerprint"]
    replacement_inputs = ["GG", "M"]

    with pytest.raises(RuntimeError, match="simulated interruption"):
        embed_dataset(
            InterruptibleEmbeddingModel(fail_on_call=1),
            replacement_inputs,
            batch_size=1,
            batch_window_size=1,
            output=path,
            format="sqlite",
            resume=False,
        )

    current = load_sqlite_result(path)
    assert current.metadata["run_fingerprint"] == original_run_id
    assert [record.sequence for record in current] == ["AC"]

    resumed = embed_dataset(
        InterruptibleEmbeddingModel(fail_on_call=None),
        replacement_inputs,
        batch_size=1,
        batch_window_size=1,
        output=path,
        format="sqlite",
    )
    assert resumed.metadata["run_fingerprint"] != original_run_id
    assert resumed.metadata["complete"] is True
    assert [record.sequence for record in resumed] == replacement_inputs
    assert (
        load_sqlite_result(path).metadata["run_fingerprint"] == resumed.metadata["run_fingerprint"]
    )


def test_sqlite_same_run_replacement_is_deferred_until_first_batch_commit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "same-run-atomic.sqlite"
    inputs = ["AC", "GG"]
    original = embed_dataset(
        InterruptibleEmbeddingModel(fail_on_call=None),
        inputs,
        batch_size=1,
        batch_window_size=1,
        output=path,
        format="sqlite",
    )
    original_run_id = original.metadata["run_fingerprint"]
    original_tensors = [record.load_tensor().clone() for record in original]

    with pytest.raises(RuntimeError, match="simulated interruption"):
        embed_dataset(
            InterruptibleEmbeddingModel(fail_on_call=1),
            inputs,
            batch_size=1,
            batch_window_size=1,
            output=path,
            format="sqlite",
            resume=False,
        )

    retained = load_sqlite_result(path)
    assert retained.metadata["run_fingerprint"] == original_run_id
    assert retained.metadata["complete"] is True
    for observed, expected in zip(retained, original_tensors, strict=True):
        torch.testing.assert_close(observed.load_tensor(), expected, rtol=0.0, atol=0.0)

    replacement = embed_dataset(
        InterruptibleEmbeddingModel(fail_on_call=None),
        inputs,
        batch_size=1,
        batch_window_size=1,
        output=path,
        format="sqlite",
        resume=False,
    )
    assert replacement.metadata["run_fingerprint"] == original_run_id
    assert replacement.metadata["complete"] is True
    assert [record.sequence for record in replacement] == inputs


def test_sqlite_prepublication_schema_remains_readable_and_migrates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pre-publication-schema.sqlite"
    original = embed_dataset(
        SyntheticEmbeddingModel(),
        ["AC"],
        output=path,
        format="sqlite",
    )
    original_run_id = original.metadata["run_fingerprint"]
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX runs_published_order_idx")
        connection.execute("ALTER TABLE runs DROP COLUMN published_order")
        connection.commit()

    legacy_view = load_sqlite_result(path)
    assert legacy_view.metadata["run_fingerprint"] == original_run_id
    assert [record.sequence for record in legacy_view] == ["AC"]

    replacement = embed_dataset(
        SyntheticEmbeddingModel(),
        ["GG"],
        output=path,
        format="sqlite",
        resume=False,
    )
    with sqlite3.connect(path) as connection:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
    assert "published_order" in columns
    assert (
        load_sqlite_result(path).metadata["run_fingerprint"]
        == replacement.metadata["run_fingerprint"]
    )
    assert [record.sequence for record in load_sqlite_result(path, run_id=original_run_id)] == [
        "AC"
    ]


def test_safetensors_round_trip_is_lazy(tmp_path: Path) -> None:
    source = embed_dataset(SyntheticEmbeddingModel(), ["ACD", "GG"])
    output = tmp_path / "safe"
    with pytest.raises(ValueError, match="cannot fit"):
        save_safetensors_result(source, output, shard_size=4)
    saved = save_safetensors_result(source, output, shard_size=8)
    loaded = load_safetensors_result(output)
    assert isinstance(saved[0].tensor, LazyTensorReference)
    assert len(list(output.glob("*.safetensors"))) >= 2
    index_path = output / "index.json"
    run_manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    generation_path = output / run_manifest["index"]["file"]
    assert run_manifest["index"] == {
        "file": generation_path.name,
        "sha256": hashlib.sha256(generation_path.read_bytes()).hexdigest(),
    }
    assert run_manifest["version"] == 2
    assert run_manifest["record_count"] == len(source)
    pointer = json.loads(index_path.read_text(encoding="utf-8"))
    assert pointer["index"] == run_manifest["index"]
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    assert "records" not in generation
    assert generation["record_count"] == len(source)
    assert generation["metadata"]["descriptor_index"] == "safetensors-generation-index"
    assert "outputs" not in generation["metadata"]
    descriptors = []
    for shard in generation["descriptor_shards"]:
        descriptors.extend(
            json.loads(line)
            for line in (output / shard["file"]).read_text(encoding="utf-8").splitlines()
            if line
        )
    assert len(descriptors) == len(source)
    assert len({item["tensor"]["key"] for item in descriptors}) == len(source)
    assert torch.equal(loaded[1].load_tensor(), source[1].load_tensor())


def test_safetensors_descriptor_shards_have_a_bounded_record_count(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bounded-descriptors"
    source = EmbeddingResult(
        [
            EmbeddingRecord(str(position), "A", torch.tensor([float(position)]))
            for position in range(1_025)
        ],
        {"run_fingerprint": "bounded-descriptors", "complete": True},
    )

    save_safetensors_result(source, output, shard_size=1024**2)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    generation = json.loads((output / run["index"]["file"]).read_text(encoding="utf-8"))

    assert [item["count"] for item in generation["descriptor_shards"]] == [1_024, 1]
    assert sum(item["count"] for item in generation["descriptor_shards"]) == len(source)
    assert "records" not in generation
    assert "outputs" not in generation["metadata"]


def test_safetensors_tensor_corruption_is_detected_on_materialization(
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file

    output = tmp_path / "corrupt-safe"
    source = EmbeddingResult(
        [EmbeddingRecord("protein", "AC", torch.tensor([1.0, 2.0]))],
        {"run_fingerprint": "corrupt-safe", "complete": True},
    )
    save_safetensors_result(source, output)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    generation = json.loads((output / run["index"]["file"]).read_text(encoding="utf-8"))
    descriptor_path = output / generation["descriptor_shards"][0]["file"]
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8").splitlines()[0])
    tensor_path = output / descriptor["tensor"]["file"]
    save_file({descriptor["tensor"]["key"]: torch.tensor([3.0, 4.0])}, tensor_path)

    with pytest.raises(ValueError, match="failed SHA-256 verification"):
        load_safetensors_result(output)[0].load_tensor()


def test_safetensors_streaming_resumes_an_ordered_prefix(tmp_path: Path) -> None:
    path = tmp_path / "stream-safe"
    inputs = ["ACD", "GG", "M"]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        embed_dataset(
            InterruptibleEmbeddingModel(fail_on_call=3),
            inputs,
            batch_size=1,
            batch_window_size=1,
            output=path,
            format="safetensors",
            shard_size=8,
        )
    partial = load_safetensors_result(path)
    assert len(partial) == 1
    assert partial.metadata["complete"] is False

    resumed = embed_dataset(
        InterruptibleEmbeddingModel(fail_on_call=None),
        inputs,
        batch_size=1,
        batch_window_size=1,
        output=path,
        format="safetensors",
        shard_size=8,
    )
    assert [record.sequence for record in resumed] == inputs
    assert resumed.metadata["complete"] is True
    assert resumed.metadata["record_count"] == 3
    assert resumed.metadata["descriptor_index"] == "safetensors-generation-index"
    assert "outputs" not in resumed.metadata


@pytest.mark.parametrize(
    ("format", "expected_granularity"),
    (("sqlite", "batch-window"), ("safetensors", "shard-flush")),
)
def test_persistent_resume_metadata_records_true_commit_granularity(
    tmp_path: Path,
    format: str,
    expected_granularity: str,
) -> None:
    output = tmp_path / ("embeddings.sqlite" if format == "sqlite" else "embeddings")

    result = embed_dataset(
        SyntheticEmbeddingModel(),
        ["AC", "GG"],
        batch_size=1,
        batch_window_size=1,
        output=output,
        format=format,
    )

    assert result.metadata["batching"]["resume_commit_granularity"] == (expected_granularity)


def test_safetensors_streaming_packs_batches_into_shards(tmp_path: Path) -> None:
    output = tmp_path / "packed"
    embed_dataset(
        SyntheticEmbeddingModel(),
        ["AC", "GG", "MM"],
        batch_size=1,
        output=output,
        format="safetensors",
        shard_size=24,
    )

    assert len(list(output.glob("*.safetensors"))) == 1


def test_safetensors_manifest_rejects_shard_path_traversal(tmp_path: Path) -> None:
    output = tmp_path / "safe"
    save_safetensors_result(
        EmbeddingResult(
            [EmbeddingRecord("protein", "AC", torch.tensor([1.0, 2.0]))],
            {"complete": True},
        ),
        output,
    )
    run_path = output / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    generation_path = output / run["index"]["file"]
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    descriptor_reference = generation["descriptor_shards"][0]
    descriptor_path = output / descriptor_reference["file"]
    descriptors = [
        json.loads(line)
        for line in descriptor_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    descriptors[0]["tensor"]["file"] = "../outside.safetensors"
    descriptor_bytes = b"".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for item in descriptors
    )
    descriptor_path.write_bytes(descriptor_bytes)
    descriptor_reference["sha256"] = hashlib.sha256(descriptor_bytes).hexdigest()
    generation_bytes = (json.dumps(generation, indent=2, sort_keys=True) + "\n").encode()
    generation_path.write_bytes(generation_bytes)
    run["index"]["sha256"] = hashlib.sha256(generation_bytes).hexdigest()
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside its output directory"):
        load_safetensors_result(output)[0]


def test_pooler_rejects_duplicate_operations() -> None:
    with pytest.raises(ValueError, match="Duplicate pooling operations"):
        Pooler(("mean", "mean"))


def test_failed_safetensors_overwrite_preserves_previous_valid_generation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "safe"
    original = EmbeddingResult(
        [EmbeddingRecord("old", "AC", torch.tensor([1.0, 2.0]))],
        {"complete": True},
    )
    save_safetensors_result(original, output, shard_size=8)
    replacement = EmbeddingResult(
        [
            EmbeddingRecord("new-1", "GG", torch.tensor([3.0, 4.0])),
            EmbeddingRecord("new-2", "MM", torch.tensor([5.0, 6.0])),
            EmbeddingRecord("too-large", "M", torch.arange(3, dtype=torch.float32)),
        ],
        {"complete": True},
    )

    original_manifest = (output / "run.json").read_bytes()
    with pytest.raises(ValueError, match="cannot fit"):
        save_safetensors_result(replacement, output, shard_size=8)

    assert (output / "run.json").read_bytes() == original_manifest
    loaded = load_safetensors_result(output)
    assert [(record.id, record.sequence) for record in loaded] == [("old", "AC")]
    assert torch.equal(loaded[0].load_tensor(), torch.tensor([1.0, 2.0]))


def test_open_safetensors_reader_survives_successful_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "retained"
    save_safetensors_result(
        EmbeddingResult(
            [EmbeddingRecord("old", "AC", torch.tensor([1.0, 2.0]))],
            {"complete": True},
        ),
        output,
    )
    old_reader = load_safetensors_result(output)
    old_manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    old_index_path = output / old_manifest["index"]["file"]
    old_index = json.loads(old_index_path.read_text(encoding="utf-8"))
    old_descriptor_path = output / old_index["descriptor_shards"][0]["file"]
    old_tensor_path = output / old_index["descriptor_shards"][0]["tensor_file"]

    save_safetensors_result(
        EmbeddingResult(
            [EmbeddingRecord("middle", "GG", torch.tensor([3.0, 4.0]))],
            {"complete": True},
        ),
        output,
    )
    # A third writer starts while the first generation is already
    # non-authoritative. Writer initialization must not treat it as an orphan.
    save_safetensors_result(
        EmbeddingResult(
            [EmbeddingRecord("new", "MM", torch.tensor([5.0, 6.0]))],
            {"complete": True},
        ),
        output,
    )
    current = load_safetensors_result(output)

    assert [(record.id, record.sequence) for record in current] == [("new", "MM")]
    assert torch.equal(current[0].load_tensor(), torch.tensor([5.0, 6.0]))
    assert old_index_path.is_file()
    assert old_descriptor_path.is_file()
    assert old_tensor_path.is_file()
    assert [(record.id, record.sequence) for record in old_reader] == [("old", "AC")]
    assert torch.equal(old_reader[0].load_tensor(), torch.tensor([1.0, 2.0]))


def test_safetensors_generation_gc_is_dry_run_and_explicitly_exclusive(
    tmp_path: Path,
) -> None:
    output = tmp_path / "retained"
    save_safetensors_result(
        EmbeddingResult(
            [EmbeddingRecord("old", "AC", torch.tensor([1.0, 2.0]))],
            {"complete": True},
        ),
        output,
    )
    old_manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    old_index_path = output / old_manifest["index"]["file"]
    old_index = json.loads(old_index_path.read_text(encoding="utf-8"))
    old_generation_paths = {
        old_index_path,
        output / old_index["descriptor_shards"][0]["file"],
        output / old_index["descriptor_shards"][0]["tensor_file"],
    }
    save_safetensors_result(
        EmbeddingResult(
            [EmbeddingRecord("new", "GG", torch.tensor([3.0, 4.0]))],
            {"complete": True},
        ),
        output,
    )

    preview = set(garbage_collect_safetensors_generations(output))
    assert old_generation_paths.issubset(preview)
    assert all(path.is_file() for path in old_generation_paths)
    with pytest.raises(ValueError, match="confirm_no_active_readers_or_writers"):
        garbage_collect_safetensors_generations(output, dry_run=False)
    assert all(path.is_file() for path in old_generation_paths)

    deleted = set(
        garbage_collect_safetensors_generations(
            output,
            dry_run=False,
            confirm_no_active_readers_or_writers=True,
        )
    )
    assert deleted == preview
    assert not any(path.exists() for path in old_generation_paths)
    current = load_safetensors_result(output)
    assert [(record.id, record.sequence) for record in current] == [("new", "GG")]
    assert torch.equal(current[0].load_tensor(), torch.tensor([3.0, 4.0]))


def test_interrupted_embedding_overwrite_preserves_previous_generation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "safe"
    original = embed_dataset(
        SyntheticEmbeddingModel(),
        ["AC"],
        output=output,
        format="safetensors",
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        embed_dataset(
            InterruptibleEmbeddingModel(fail_on_call=2),
            ["GG", "M"],
            batch_size=1,
            output=output,
            format="safetensors",
            resume=False,
        )

    loaded = load_safetensors_result(output)
    assert [(record.id, record.sequence) for record in loaded] == [
        (record.id, record.sequence) for record in original
    ]
    assert torch.equal(loaded[0].load_tensor(), original[0].load_tensor())


def test_interrupted_metadata_publish_recovers_last_committed_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "safe"
    original = EmbeddingResult(
        [EmbeddingRecord("old", "AC", torch.tensor([1.0, 2.0]))],
        {"complete": True},
    )
    save_safetensors_result(original, output, shard_size=8)

    writer = SafetensorsStreamWriter(
        output,
        {"complete": False},
        shard_size=8,
        publish_initial=False,
        publish_incremental=False,
    )
    writer.append(
        [EmbeddingRecord("new", "GG", torch.tensor([3.0, 4.0]))],
        publish=False,
    )
    run_manifest_path = output / "run.json"
    original_replace = Path.replace

    def interrupt_manifest_replace(path: Path, target: Path) -> Path:
        if Path(target) == run_manifest_path:
            raise OSError("simulated metadata publication interruption")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupt_manifest_replace)
    with pytest.raises(OSError, match="metadata publication interruption"):
        writer.publish(complete=True)

    loaded = load_safetensors_result(output)
    assert [(record.id, record.sequence) for record in loaded] == [("old", "AC")]
    assert torch.equal(loaded[0].load_tensor(), torch.tensor([1.0, 2.0]))


def test_named_safetensors_outputs_do_not_share_shards(tmp_path: Path) -> None:
    first_source = embed_dataset(SyntheticEmbeddingModel(), ["ACD"])
    second_source = embed_dataset(SyntheticEmbeddingModel(), ["GGG"])
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"

    save_safetensors_result(first_source, first_path)
    save_safetensors_result(second_source, second_path)

    first_loaded = load_safetensors_result(first_path)
    second_loaded = load_safetensors_result(second_path)
    assert torch.equal(first_loaded[0].load_tensor(), first_source[0].load_tensor())
    assert torch.equal(second_loaded[0].load_tensor(), second_source[0].load_tensor())
    assert list(tmp_path.glob("first-embeddings-*.safetensors"))
    assert list(tmp_path.glob("second-embeddings-*.safetensors"))


def test_safetensors_manifest_snapshot_recovers_mismatched_standalone_index(
    tmp_path: Path,
) -> None:
    output = tmp_path / "safe"
    source = embed_dataset(SyntheticEmbeddingModel(), ["ACD"])
    save_safetensors_result(source, output)
    index_path = output / "index.json"
    index_path.write_text(index_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    recovered = load_safetensors_result(output)
    assert len(recovered) == 1
    assert recovered[0].sequence == "ACD"

    run_path = output / "run.json"
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    generation_path = output / manifest["index"]["file"]
    generation_path.write_text(
        generation_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match its index"):
        load_safetensors_result(output)


def test_resume_recovers_from_authoritative_manifest_when_index_is_missing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "safe"
    model = SyntheticEmbeddingModel()
    original = embed_dataset(model, ["ACD"], output=output)
    (output / "index.json").unlink()

    recovered = load_safetensors_result(output)
    resumed = embed_dataset(model, ["ACD"], output=output)

    assert not (output / "index.json").exists()
    assert torch.equal(recovered[0].load_tensor(), original[0].load_tensor())
    assert resumed.metadata["run_fingerprint"] == original.metadata["run_fingerprint"]


def test_resume_requires_matching_fingerprint(tmp_path: Path) -> None:
    output = tmp_path / "resume"
    first = embed_dataset(SyntheticEmbeddingModel(), ["ACD"], output=output)
    resumed = embed_dataset(SyntheticEmbeddingModel(), ["ACD"], output=output)
    assert isinstance(first[0].tensor, LazyTensorReference)
    assert resumed.metadata["run_fingerprint"] == first.metadata["run_fingerprint"]
    with pytest.raises(ValueError, match="different run fingerprint"):
        embed_dataset(SyntheticEmbeddingModel(), ["GG"], output=output)


def test_resume_rejects_legacy_fingerprint_schema(tmp_path: Path) -> None:
    output = tmp_path / "legacy-schema.sqlite"
    embed_dataset(
        SyntheticEmbeddingModel(),
        ["ACD"],
        output=output,
        format="sqlite",
    )
    with sqlite3.connect(output) as connection:
        run_id, metadata_json = connection.execute(
            "SELECT run_id, metadata_json FROM runs"
        ).fetchone()
        metadata = json.loads(metadata_json)
        metadata["fingerprint_schema_version"] = 1
        connection.execute(
            "UPDATE runs SET metadata_json = ? WHERE run_id = ?",
            (json.dumps(metadata, sort_keys=True), run_id),
        )

    with pytest.raises(ValueError, match="incompatible run fingerprint schema"):
        embed_dataset(
            SyntheticEmbeddingModel(),
            ["ACD"],
            output=output,
            format="sqlite",
        )


def test_legacy_pth_import_requires_explicit_unsafe_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pth"
    torch.save({"ACD": torch.ones(3)}, path)
    with pytest.raises(ValueError, match="allow_unsafe_pickle=True"):
        load_legacy_pth(path)
    loaded = load_legacy_pth(path, allow_unsafe_pickle=True)
    assert torch.equal(loaded[0].load_tensor(), torch.ones(3))


def test_new_api_never_writes_pth(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not supported"):
        embed_dataset(
            SyntheticEmbeddingModel(),
            ["ACD"],
            output=tmp_path / "embeddings.pth",
            format="pth",
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("batch_size", True),
        ("batch_size", 1.5),
        ("max_length", True),
        ("max_tokens_per_batch", True),
        ("batch_window_size", True),
        ("shard_size", True),
        ("full_embeddings", 1),
        ("resume", 1),
        ("truncate", 1),
        ("model_state_fingerprint", ""),
    ],
)
def test_strict_embedding_controls_fail_before_consuming_inputs(name, value) -> None:
    consumed = False

    def source():
        nonlocal consumed
        consumed = True
        yield "AC"

    with pytest.raises((TypeError, ValueError)):
        embed_dataset(SyntheticEmbeddingModel(), source(), **{name: value})
    assert consumed is False


@pytest.mark.parametrize(
    "decoder_input_ids",
    [
        torch.tensor([1], dtype=torch.int64),
        torch.empty((1, 0), dtype=torch.int64),
        torch.tensor([[1]], dtype=torch.int16),
        torch.tensor([[1.0]], dtype=torch.float32),
    ],
)
def test_decoder_input_ids_require_nonempty_2d_int32_or_int64(decoder_input_ids) -> None:
    with pytest.raises((TypeError, ValueError)):
        embed_dataset(
            SyntheticDecoderEmbeddingModel(),
            ["AC"],
            hidden_state_source="decoder",
            decoder_input_ids=decoder_input_ids,
        )


@pytest.mark.parametrize(
    "decoder_attention_mask",
    [
        torch.tensor([[0, 2]], dtype=torch.int64),
        torch.tensor([[0.0, float("nan")]]),
        torch.tensor([[1]], dtype=torch.int64),
    ],
)
def test_decoder_attention_masks_are_exact_finite_binary_shapes(
    decoder_attention_mask,
) -> None:
    with pytest.raises(ValueError):
        embed_dataset(
            SyntheticDecoderEmbeddingModel(),
            ["AC"],
            hidden_state_source="decoder",
            decoder_input_ids=torch.tensor([[1, 2]], dtype=torch.int64),
            decoder_attention_mask=decoder_attention_mask,
        )


class InvalidEmbeddingBatchModel(SyntheticEmbeddingModel):
    def __init__(self, case: str) -> None:
        super().__init__()
        self.case = case

    def _embedding_batch(self, sequences: list[str]) -> EmbeddingBatch:
        batch = super()._embedding_batch(sequences)
        if self.case == "x_type":
            return EmbeddingBatch(X="bad", residue_mask=batch.residue_mask)  # type: ignore[arg-type]
        if self.case == "mask_type":
            return EmbeddingBatch(X=batch.X, residue_mask="bad")  # type: ignore[arg-type]
        if self.case == "x_integer":
            return EmbeddingBatch(X=batch.X.to(torch.int64), residue_mask=batch.residue_mask)
        if self.case == "mask_nonbinary":
            return EmbeddingBatch(X=batch.X, residue_mask=batch.residue_mask.float() * 0.5)
        if self.case == "wrong_batch":
            return EmbeddingBatch(X=batch.X[:1], residue_mask=batch.residue_mask[:1])
        X = batch.X.clone()
        X[0, 1, 0] = torch.inf
        return EmbeddingBatch(X=X, residue_mask=batch.residue_mask)


@pytest.mark.parametrize(
    "case",
    ["x_type", "mask_type", "x_integer", "mask_nonbinary", "wrong_batch", "nonfinite"],
)
def test_embedding_batch_adapter_outputs_are_validated(case) -> None:
    with pytest.raises((TypeError, ValueError)):
        embed_dataset(InvalidEmbeddingBatchModel(case), ["AC", "GG"])


@pytest.mark.parametrize(
    "case",
    ["loader_type", "dtype", "verify", "record_tensor"],
)
def test_embedding_value_types_fail_closed(case) -> None:
    if case == "record_tensor":
        with pytest.raises(TypeError):
            EmbeddingRecord("id", "AC", object())  # type: ignore[arg-type]
        return
    reference = LazyTensorReference(
        source="memory",
        key="x",
        dtype="float32",
        shape=(1,),
        sha256="0" * 64,
        _loader=(lambda: object()) if case == "loader_type" else (lambda: torch.ones(1)),
    )
    if case == "loader_type":
        with pytest.raises(TypeError):
            reference.load(verify=False)
    elif case == "dtype":
        wrong_dtype = LazyTensorReference(
            source="memory",
            key="x",
            dtype="float64",
            shape=(1,),
            sha256="0" * 64,
            _loader=lambda: torch.ones(1),
        )
        with pytest.raises(ValueError, match="dtype"):
            wrong_dtype.load(verify=False)
    else:
        with pytest.raises(TypeError, match="verify"):
            reference.load(verify=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"damping": float("nan")},
        {"tolerance": 0.0},
        {"max_iterations": True},
    ],
)
def test_pagerank_controls_require_finite_valid_values(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        pagerank_weights(torch.ones(2, 2), **kwargs)


def test_tensor_sha256_uses_bounded_chunks_and_preserves_legacy_digest(monkeypatch) -> None:
    from fastplms.embeddings import storage

    X = torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
    expected = hashlib.sha256()
    expected.update(b"float32")
    expected.update(json.dumps(tuple(X.shape)).encode())
    expected.update(X.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())

    monkeypatch.setattr(storage, "_TENSOR_HASH_CHUNK_BYTES", 7)
    monkeypatch.setattr(
        storage,
        "_tensor_bytes",
        lambda _: (_ for _ in ()).throw(AssertionError("full byte copy used")),
    )
    assert storage.tensor_sha256(X) == expected.hexdigest()


def test_sqlite_lazy_references_are_absolute_after_cwd_change(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "embeddings.sqlite"
    saved = save_sqlite_result(
        EmbeddingResult(
            [EmbeddingRecord("id", "AC", torch.arange(4, dtype=torch.float32))],
            {"run_fingerprint": "absolute-path"},
        ),
        output,
    )
    monkeypatch.chdir(tmp_path.parent)
    assert Path(saved[0].tensor.source).is_absolute()
    assert torch.equal(saved[0].load_tensor(), torch.arange(4, dtype=torch.float32))
