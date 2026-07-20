from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fastplms.embeddings import (
    EmbeddingBatch,
    EmbeddingInput,
    EmbeddingRecord,
    EmbeddingResult,
    LazyTensorReference,
    Pooler,
    embed_dataset,
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


def test_mapping_inputs_embed_values_with_mapping_keys_as_ids() -> None:
    inputs = {
        "protein-a": "ACD",
        "protein-b": "GG",
    }

    result = embed_dataset(SyntheticEmbeddingModel(), inputs, pooling="mean")

    assert [record.id for record in result] == ["protein-a", "protein-b"]
    assert [record.sequence for record in result] == ["ACD", "GG"]
    assert result[0].load_tensor()[0].item() == pytest.approx(
        sum(map(ord, "ACD")) / len("ACD")
    )
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
    tmp_path, format: str
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
    loaded = (
        load_sqlite_result(output)
        if format == "sqlite"
        else load_safetensors_result(output)
    )
    assert torch.equal(loaded[0].load_tensor(), result[0].load_tensor())
    assert loaded.metadata["outputs"][0]["shape"] == [2, 3, 2]


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


def test_embedding_fingerprint_binds_persisted_parameters_and_buffers(tmp_path) -> None:
    model = SyntheticEmbeddingModel()
    model.register_buffer("running_value", torch.tensor([3.0]))
    initial = embed_dataset(model, ["ACD"], output=tmp_path / "initial")

    with torch.no_grad():
        model.anchor.fill_(1)
    changed_parameter = embed_dataset(model, ["ACD"], output=tmp_path / "parameter")
    assert changed_parameter.metadata["model_state_fingerprint"] != (
        initial.metadata["model_state_fingerprint"]
    )
    assert changed_parameter.metadata["run_fingerprint"] != initial.metadata["run_fingerprint"]

    model.running_value.add_(1)
    changed_buffer = embed_dataset(model, ["ACD"], output=tmp_path / "buffer")
    assert changed_buffer.metadata["model_state_fingerprint"] != (
        changed_parameter.metadata["model_state_fingerprint"]
    )
    assert initial.metadata["fingerprint_schema_version"] == 2
    assert initial.metadata["model_state_fingerprint_source"] == "computed"


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


def test_tokenizer_content_changes_run_fingerprint() -> None:
    class Tokenizer:
        all_special_ids = (0, 2)
        name_or_path = "synthetic/tokenizer"
        vocab_size = 4
        special_tokens_map = {"bos_token": "<bos>", "eos_token": "<eos>"}
        model_max_length = 32
        padding_side = "right"
        truncation_side = "right"

        def __init__(self, vocab, *, mode: str = "first") -> None:
            self._vocab = vocab
            self.init_kwargs = {"mode": mode}

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

    assert first.metadata["tokenizer"]["content_sha256"] != (
        changed_vocab.metadata["tokenizer"]["content_sha256"]
    )
    assert first.metadata["run_fingerprint"] != changed_vocab.metadata["run_fingerprint"]
    assert first.metadata["run_fingerprint"] != changed_config.metadata["run_fingerprint"]


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


def test_fasta_preserves_headers_order_and_duplicates(tmp_path) -> None:
    path = tmp_path / "proteins.fasta"
    path.write_text(">a description\nACD\n>a\nGG\n", encoding="utf-8")
    records = parse_fasta(path)
    assert records == [EmbeddingInput("a", "ACD"), EmbeddingInput("a", "GG")]


def test_sqlite_round_trip_is_lazy_and_bf16_lossless(tmp_path) -> None:
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


def test_sqlite_streaming_resumes_an_ordered_prefix(tmp_path) -> None:
    path = tmp_path / "stream.sqlite"
    inputs = ["ACD", "GG", "M"]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        embed_dataset(
            InterruptibleEmbeddingModel(fail_on_call=2),
            inputs,
            batch_size=1,
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
        output=path,
        format="sqlite",
    )
    assert [record.sequence for record in resumed] == inputs
    assert resumed.metadata["complete"] is True
    assert len(resumed.metadata["outputs"]) == 3


def test_safetensors_round_trip_is_lazy(tmp_path) -> None:
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
    assert run_manifest["index"] == {
        "file": "index.json",
        "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    assert run_manifest["record_count"] == len(source)
    assert torch.equal(loaded[1].load_tensor(), source[1].load_tensor())


def test_safetensors_streaming_resumes_an_ordered_prefix(tmp_path) -> None:
    path = tmp_path / "stream-safe"
    inputs = ["ACD", "GG", "M"]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        embed_dataset(
            InterruptibleEmbeddingModel(fail_on_call=3),
            inputs,
            batch_size=1,
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
        output=path,
        format="safetensors",
        shard_size=8,
    )
    assert [record.sequence for record in resumed] == inputs
    assert resumed.metadata["complete"] is True
    assert len(resumed.metadata["outputs"]) == 3


def test_safetensors_streaming_packs_batches_into_shards(tmp_path) -> None:
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


def test_safetensors_manifest_rejects_shard_path_traversal(tmp_path) -> None:
    output = tmp_path / "safe"
    save_safetensors_result(
        EmbeddingResult(
            [EmbeddingRecord("protein", "AC", torch.tensor([1.0, 2.0]))],
            {"complete": True},
        ),
        output,
    )
    index_path = output / "index.json"
    run_path = output / "run.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["records"][0]["tensor"]["file"] = "../outside.safetensors"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    index_path.write_bytes(encoded)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["index_payload"] = payload
    run["index"]["sha256"] = hashlib.sha256(encoded).hexdigest()
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside its output directory"):
        load_safetensors_result(output)


def test_pooler_rejects_duplicate_operations() -> None:
    with pytest.raises(ValueError, match="Duplicate pooling operations"):
        Pooler(("mean", "mean"))


def test_failed_safetensors_overwrite_preserves_previous_valid_generation(tmp_path) -> None:
    output = tmp_path / "safe"
    original = EmbeddingResult(
        [EmbeddingRecord("old", "AC", torch.tensor([1.0, 2.0]))],
        {"complete": True},
    )
    save_safetensors_result(original, output, shard_size=8)
    replacement = EmbeddingResult(
        [
            EmbeddingRecord("new", "GG", torch.tensor([3.0, 4.0])),
            EmbeddingRecord("too-large", "M", torch.arange(3, dtype=torch.float32)),
        ],
        {"complete": True},
    )

    with pytest.raises(ValueError, match="cannot fit"):
        save_safetensors_result(replacement, output, shard_size=8)

    loaded = load_safetensors_result(output)
    assert [(record.id, record.sequence) for record in loaded] == [("old", "AC")]
    assert torch.equal(loaded[0].load_tensor(), torch.tensor([1.0, 2.0]))


def test_interrupted_embedding_overwrite_preserves_previous_generation(tmp_path) -> None:
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


def test_named_safetensors_outputs_do_not_share_shards(tmp_path) -> None:
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
    legacy_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    legacy_manifest.pop("index_payload")
    run_path.write_text(
        json.dumps(legacy_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match its index"):
        load_safetensors_result(output)


def test_resume_requires_matching_fingerprint(tmp_path) -> None:
    output = tmp_path / "resume"
    first = embed_dataset(SyntheticEmbeddingModel(), ["ACD"], output=output)
    resumed = embed_dataset(SyntheticEmbeddingModel(), ["ACD"], output=output)
    assert isinstance(first[0].tensor, LazyTensorReference)
    assert resumed.metadata["run_fingerprint"] == first.metadata["run_fingerprint"]
    with pytest.raises(ValueError, match="different run fingerprint"):
        embed_dataset(SyntheticEmbeddingModel(), ["GG"], output=output)


def test_resume_rejects_legacy_fingerprint_schema(tmp_path) -> None:
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


def test_legacy_pth_import_requires_explicit_unsafe_opt_in(tmp_path) -> None:
    path = tmp_path / "legacy.pth"
    torch.save({"ACD": torch.ones(3)}, path)
    with pytest.raises(ValueError, match="allow_unsafe_pickle=True"):
        load_legacy_pth(path)
    loaded = load_legacy_pth(path, allow_unsafe_pickle=True)
    assert torch.equal(loaded[0].load_tensor(), torch.ones(3))


def test_new_api_never_writes_pth(tmp_path) -> None:
    with pytest.raises(ValueError, match="not supported"):
        embed_dataset(
            SyntheticEmbeddingModel(),
            ["ACD"],
            output=tmp_path / "embeddings.pth",
            format="pth",
        )
