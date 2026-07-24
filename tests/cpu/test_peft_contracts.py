"""Mandatory shipped fine-tuning, PEFT, collator, and persistence contracts."""

from __future__ import annotations

import pytest
import torch
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from examples import fine_tuning
from fastplms.models.esm2.modeling_fastesm import (
    FastEsmConfig,
    FastEsmForSequenceClassification,
)
from tests.unit import test_fine_tuning_example as fine_tuning_contracts


test_pair_collator_enforces_longest_first_tokenizer_limit = (
    fine_tuning_contracts.test_pair_collator_enforces_longest_first_tokenizer_limit
)
test_pair_token_budget_includes_special_tokens_at_the_exact_boundary = (
    fine_tuning_contracts.test_pair_token_budget_includes_special_tokens_at_the_exact_boundary
)
test_reporting_is_opt_in_for_the_minimal_training_install = (
    fine_tuning_contracts.test_reporting_is_opt_in_for_the_minimal_training_install
)
test_max_length_contract_is_an_encoded_budget_including_added_tokens = (
    fine_tuning_contracts.test_max_length_contract_is_an_encoded_budget_including_added_tokens
)
test_ordered_training_row_hash_is_content_and_order_sensitive = (
    fine_tuning_contracts.test_ordered_training_row_hash_is_content_and_order_sensitive
)
test_persisted_hash_scope_covers_full_state_or_only_lora_payload = (
    fine_tuning_contracts.test_persisted_hash_scope_covers_full_state_or_only_lora_payload
)
test_atomic_final_artifact_reload_preserves_trainer_and_held_out_logits = (
    fine_tuning_contracts.test_atomic_final_artifact_reload_preserves_trainer_and_held_out_logits
)


def test_plot_contract_uses_task_output_paths_and_has_no_interactive_or_overwrite_path() -> None:
    contract = fine_tuning_contracts
    contract.test_plot_contract_uses_task_output_paths_and_has_no_interactive_or_overwrite_path()


def test_training_manifest_records_reproducible_model_data_and_tokenizer_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = fine_tuning_contracts
    contract.test_training_manifest_records_reproducible_model_data_and_tokenizer_identity(
        monkeypatch,
        tmp_path,
    )


def test_immutable_sources_reject_moving_refs_pin_only_shipped_defaults_and_detect_drift(
    tmp_path: Path,
) -> None:
    contract = fine_tuning_contracts
    contract.test_immutable_sources_reject_moving_refs_pin_only_shipped_defaults_and_detect_drift(
        tmp_path
    )


@pytest.mark.parametrize("backend", ("flash_attention_2", "flash_attention_3"))
def test_fine_tuning_example_rejects_flash_without_explicit_bf16_policy(
    backend: str,
) -> None:
    with pytest.raises(SystemExit):
        fine_tuning.build_parser().parse_args(["--attn-backend", backend])
    with pytest.raises(ValueError, match="explicit BF16 CUDA"):
        fine_tuning.initialize_model("unused", 2, attn_backend=backend)


def test_fine_tuning_main_wires_both_tasks_without_external_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the shipped CLI entry point without loading models or datasets."""

    regression_calls: list[dict[str, Any]] = []
    classification_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        fine_tuning,
        "train_regression_model",
        lambda **kwargs: regression_calls.append(kwargs),
    )
    monkeypatch.setattr(
        fine_tuning,
        "train_classification_model",
        lambda **kwargs: classification_calls.append(kwargs),
    )

    result = fine_tuning.main(
        [
            "--task",
            "both",
            "--model_path",
            "offline/tiny-model",
            "--model-revision",
            "a" * 40,
            "--classification-dataset-source",
            "offline/classification",
            "--classification-dataset-revision",
            "b" * 40,
            "--regression-train-dataset-source",
            "offline/regression-train",
            "--regression-train-dataset-revision",
            "c" * 40,
            "--regression-validation-dataset-source",
            "offline/regression-validation",
            "--regression-validation-dataset-revision",
            "d" * 40,
            "--regression-test-dataset-source",
            "offline/regression-test",
            "--regression-test-dataset-revision",
            "e" * 40,
            "--no-use-lora",
            "--batch_size",
            "3",
            "--lr",
            "0.001",
            "--epochs",
            "2.5",
            "--max_length",
            "64",
            "--attn-backend",
            "eager",
            "--output-dir",
            str(tmp_path / "fine-tuning"),
            "--grad_accum",
            "2",
            "--patience",
            "1",
            "--seed",
            "17",
            "--full-determinism",
            "--no-plot-results",
        ]
    )

    shared = {
        "model_name": "offline/tiny-model",
        "model_revision": "a" * 40,
        "use_lora": False,
        "batch_size": 3,
        "learning_rate": 0.001,
        "num_epochs": 2.5,
        "max_length": 64,
        "gradient_accumulation_steps": 2,
        "patience": 1,
        "seed": 17,
        "full_determinism": True,
        "plot_results": False,
        "attn_backend": "eager",
    }
    assert result == 0
    assert regression_calls == [
        {
            **shared,
            "train_dataset_source": "offline/regression-train",
            "train_dataset_revision": "c" * 40,
            "validation_dataset_source": "offline/regression-validation",
            "validation_dataset_revision": "d" * 40,
            "test_dataset_source": "offline/regression-test",
            "test_dataset_revision": "e" * 40,
            "output_dir": tmp_path / "fine-tuning" / "regression",
        }
    ]
    assert classification_calls == [
        {
            **shared,
            "dataset_source": "offline/classification",
            "dataset_revision": "b" * 40,
            "output_dir": tmp_path / "fine-tuning" / "classification",
        }
    ]


def test_multi_task_preflight_rejects_any_existing_child_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "fine-tuning"
    existing = output_root / "classification_lora"
    existing.mkdir(parents=True)
    calls: list[str] = []
    monkeypatch.setattr(
        fine_tuning,
        "train_regression_model",
        lambda **kwargs: calls.append("regression"),
    )
    monkeypatch.setattr(
        fine_tuning,
        "train_classification_model",
        lambda **kwargs: calls.append("classification"),
    )

    with pytest.raises(FileExistsError, match="could mix prior state"):
        fine_tuning.main(["--task", "both", "--output-dir", str(output_root)])
    assert calls == []


def test_atomic_output_reservation_rejects_reuse_and_cleans_failed_runs(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with (
        pytest.raises(FileExistsError, match="already exists"),
        fine_tuning._reserved_output_directory(existing),
    ):
        pytest.fail("an existing output must never be entered")

    failed = tmp_path / "failed"
    with (
        pytest.raises(RuntimeError, match="training failed"),
        fine_tuning._reserved_output_directory(failed) as reserved,
    ):
        (reserved / "partial-checkpoint.bin").write_bytes(b"partial")
        raise RuntimeError("training failed")
    assert not failed.exists()

    completed = tmp_path / "completed"
    with fine_tuning._reserved_output_directory(completed) as reserved:
        (reserved / "result.txt").write_text("complete\n", encoding="utf-8")
    assert (completed / "result.txt").is_file()
    assert not (completed / fine_tuning._OUTPUT_RESERVATION_FILE).exists()


class _ColumnDataset:
    def __init__(self, **columns: list[Any]) -> None:
        self.columns = columns
        self.column_names = list(columns)

    def __getitem__(self, column: str) -> list[Any]:
        return self.columns[column]

    def __len__(self) -> int:
        return len(next(iter(self.columns.values()), ()))


def _classification_data(
    *,
    train_labels: list[Any] | None = None,
    valid_labels: list[Any] | None = None,
    test_labels: list[Any] | None = None,
) -> dict[str, _ColumnDataset]:
    return {
        "train": _ColumnDataset(
            seqs=["AC", "DE"],
            labels=[0, 1] if train_labels is None else train_labels,
        ),
        "valid": _ColumnDataset(
            seqs=["FG"],
            labels=[0] if valid_labels is None else valid_labels,
        ),
        "test": _ColumnDataset(
            seqs=["HI"],
            labels=[1] if test_labels is None else test_labels,
        ),
    }


def test_dataset_contracts_reject_noncontiguous_unseen_and_nonfinite_labels() -> None:
    assert fine_tuning._validate_classification_dataset_dict(_classification_data()) == 2

    with pytest.raises(ValueError, match="contiguous zero-based"):
        fine_tuning._validate_classification_dataset_dict(_classification_data(train_labels=[0, 2]))
    with pytest.raises(ValueError, match="absent from train"):
        fine_tuning._validate_classification_dataset_dict(_classification_data(valid_labels=[2]))
    with pytest.raises(ValueError, match="must be an integer"):
        fine_tuning._validate_classification_dataset_dict(_classification_data(test_labels=[1.0]))

    valid_regression = _ColumnDataset(SeqA=["AC"], SeqB=["DE"], labels=[-8.2])
    fine_tuning._validate_regression_dataset(valid_regression, split="train")
    invalid_regression = _ColumnDataset(SeqA=["AC"], SeqB=["DE"], labels=[float("nan")])
    with pytest.raises(ValueError, match="finite real number"):
        fine_tuning._validate_regression_dataset(invalid_regression, split="test")


def test_invalid_classification_contract_precedes_model_initialization_and_cleans_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        fine_tuning,
        "_load_dataset_immutable",
        lambda *args, **kwargs: (_classification_data(valid_labels=[2]), {}),
    )
    monkeypatch.setattr(
        fine_tuning,
        "initialize_model",
        lambda *args, **kwargs: pytest.fail("model initialized before data validation"),
    )
    output_dir = tmp_path / "classification"

    with pytest.raises(ValueError, match="absent from train"):
        fine_tuning.train_classification_model(
            dataset_source="offline/classification",
            dataset_revision="a" * 40,
            output_dir=output_dir,
        )
    assert not output_dir.exists()


def test_invalid_regression_contract_precedes_model_initialization_and_cleans_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    datasets = iter(
        (
            _ColumnDataset(SeqA=["AC"], SeqB=["DE"], labels=[1.0]),
            _ColumnDataset(SeqA=["FG"], SeqB=["HI"], labels=[2.0]),
            _ColumnDataset(SeqA=["KL"], SeqB=["MN"], labels=[float("inf")]),
        )
    )
    monkeypatch.setattr(
        fine_tuning,
        "_load_dataset_immutable",
        lambda *args, **kwargs: (next(datasets), {}),
    )
    monkeypatch.setattr(
        fine_tuning,
        "initialize_model",
        lambda *args, **kwargs: pytest.fail("model initialized before data validation"),
    )
    output_dir = tmp_path / "regression"

    with pytest.raises(ValueError, match="finite real number"):
        fine_tuning.train_regression_model(
            train_dataset_source="offline/train",
            train_dataset_revision="a" * 40,
            validation_dataset_source="offline/validation",
            validation_dataset_revision="b" * 40,
            test_dataset_source="offline/test",
            test_dataset_revision="c" * 40,
            output_dir=output_dir,
        )
    assert not output_dir.exists()


def test_plot_writes_are_exclusive_and_preserve_existing_bytes(tmp_path: Path) -> None:
    class FakeFigure:
        def savefig(self, handle: Any, **kwargs: Any) -> None:
            assert kwargs == {"format": "png", "dpi": 300}
            handle.write(b"new-png")

    target = tmp_path / "classification_results.png"
    assert fine_tuning._save_figure_exclusive(FakeFigure(), target) == target
    assert target.read_bytes() == b"new-png"
    target.write_bytes(b"existing-png")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        fine_tuning._save_figure_exclusive(FakeFigure(), target)
    assert target.read_bytes() == b"existing-png"


class _OfflineProteinTokenizer:
    """Small deterministic tokenizer sufficient for both shipped collators."""

    pad_token_id = 1
    name_or_path = "offline-cpu-contract"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _encode(sequence: str) -> list[int]:
        alphabet = "ACDEFGHIKLMN"
        available_ids = (3, 4, *range(6, 16))
        token_ids = dict(zip(alphabet, available_ids, strict=True))
        return [token_ids[residue] for residue in sequence]

    def __call__(
        self,
        sequences: str | tuple[str, ...] | list[str],
        pairs: str | tuple[str, ...] | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        scalar = isinstance(sequences, str)
        sequence_rows = [sequences] if scalar else list(sequences)
        if pairs is None:
            pair_rows: list[str | None] = [None] * len(sequence_rows)
        elif isinstance(pairs, str):
            pair_rows = [pairs]
        else:
            pair_rows = list(pairs)
        if len(sequence_rows) != len(pair_rows):
            raise ValueError("sequence and pair batches must align")

        rows: list[list[int]] = []
        for sequence, pair in zip(sequence_rows, pair_rows, strict=True):
            row = [0, *self._encode(sequence), 2]
            if pair is not None:
                row.extend((*self._encode(pair), 2))
            max_length = kwargs.get("max_length")
            if kwargs.get("truncation") and max_length is not None:
                row = row[: int(max_length)]
            rows.append(row)

        if kwargs.get("return_tensors") != "pt":
            return {"input_ids": rows[0] if scalar else rows}

        width = max(map(len, rows))
        multiple = kwargs.get("pad_to_multiple_of")
        if multiple is not None:
            width = min(
                ((width + int(multiple) - 1) // int(multiple)) * int(multiple),
                int(kwargs.get("max_length", width)),
            )
        input_ids = torch.tensor(
            [row + [self.pad_token_id] * (width - len(row)) for row in rows],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id).long(),
        }

    def save_pretrained(self, save_directory: str | Path) -> tuple[str]:
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "tokenizer.json"
        path.write_text("{}\n", encoding="utf-8")
        return (str(path),)


def _tiny_config(attn_backend: str = "eager") -> FastEsmConfig:
    return FastEsmConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=2,
        pad_token_id=1,
        mask_token_id=5,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=16,
        position_embedding_type="absolute",
        attn_backend=attn_backend,
    )


def test_shipped_collators_create_tokenizer_aware_sequence_and_pair_batches() -> None:
    tokenizer = _OfflineProteinTokenizer()
    sequence_batch = fine_tuning.SequenceCollator(
        tokenizer,
        regression=False,
        max_length=8,
    )([("ACD", 0), ("EF", 1)])
    pair_batch = fine_tuning.PairCollator(
        tokenizer,
        regression=True,
        max_length=8,
    )([("ACD", "EF", 1.5), ("GH", "IK", 2.5)])

    assert sequence_batch["input_ids"].shape == (2, 8)
    assert sequence_batch["attention_mask"].shape == (2, 8)
    assert sequence_batch["labels"].dtype == torch.long
    assert pair_batch["input_ids"].shape == (2, 8)
    assert pair_batch["attention_mask"].shape == (2, 8)
    assert pair_batch["labels"].dtype == torch.float32
    assert tokenizer.calls[0] == {
        "padding": "longest",
        "return_tensors": "pt",
        "truncation": True,
        "max_length": 8,
        "pad_to_multiple_of": 8,
    }
    assert tokenizer.calls[1] == {
        "padding": "longest",
        "return_tensors": "pt",
        "truncation": "longest_first",
        "max_length": 8,
        "pad_to_multiple_of": 8,
    }


def test_shipped_initializer_drives_one_peft_step_and_atomic_final_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    template = FastEsmForSequenceClassification(_tiny_config())
    base_state = {  # parameter name -> checkpoint-shaped tensor
        name: tensor.detach().clone()
        for name, tensor in template.state_dict().items()
    }
    tokenizer = _OfflineProteinTokenizer()
    loader_calls: list[tuple[str, dict[str, Any]]] = []

    def load_tiny_model(
        model_name: str,
        **kwargs: Any,
    ) -> FastEsmForSequenceClassification:
        loader_calls.append((model_name, dict(kwargs)))
        model = FastEsmForSequenceClassification(
            _tiny_config(attn_backend=kwargs["attn_implementation"])
        )
        model.load_state_dict(base_state)
        model.tokenizer = tokenizer
        return model

    monkeypatch.setattr(
        fine_tuning.AutoModelForSequenceClassification,
        "from_pretrained",
        staticmethod(load_tiny_model),
    )
    adapter, observed_tokenizer = fine_tuning.initialize_model(
        "offline/tiny-esm2",
        num_labels=2,
        use_lora=True,
        lora_config=None,
        model_revision="a" * 40,
    )

    assert observed_tokenizer is tokenizer
    assert adapter.config.attn_backend == "sdpa"
    assert loader_calls == [
        (
            "offline/tiny-esm2",
            {
                "trust_remote_code": True,
                "num_labels": 2,
                "attn_implementation": "sdpa",
                "revision": "a" * 40,
            },
        )
    ]
    assert "classifier" in adapter.peft_config["default"].modules_to_save

    batch = fine_tuning.SequenceCollator(  # token tensors: (b=2, l)
        tokenizer,
        regression=False,
        max_length=8,
    )([("ACD", 0), ("EFG", 1)])
    before = {  # parameter name -> parameter-shaped tensor
        name: parameter.detach().clone()
        for name, parameter in adapter.named_parameters()
    }
    trainable = {name for name, parameter in adapter.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all("lora_" in name or "classifier" in name for name in trainable)

    adapter.train()
    sdpa_calls: list[dict[str, Any]] = []
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def instrumented_sdpa(*args: Any, **kwargs: Any) -> torch.Tensor:
        sdpa_calls.append(dict(kwargs))
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        instrumented_sdpa,
    )
    optimizer = torch.optim.SGD(
        (parameter for parameter in adapter.parameters() if parameter.requires_grad),
        lr=0.5,
    )
    output = adapter(**batch)  # logits: (b=2, c=2); loss: ()
    assert len(sdpa_calls) == 1
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    expected_changed = {
        name
        for name, parameter in adapter.named_parameters()
        if parameter.grad is not None and int(torch.count_nonzero(parameter.grad).item()) > 0
    }
    assert expected_changed
    optimizer.step()
    changed = {
        name
        for name, parameter in adapter.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    }
    assert changed == expected_changed
    assert changed <= trainable
    assert any("lora_" in name for name in changed)
    assert any("classifier" in name for name in changed)

    inference_inputs = {  # token tensors: (b, l)
        key: value
        for key, value in batch.items()
        if key != "labels"
    }
    adapter.eval()
    with torch.inference_mode():
        expected = adapter(**inference_inputs).logits  # (b, c)

    verification_rows = [("ACD", 0), ("EFG", 1)]
    verification_collator = fine_tuning.SequenceCollator(
        tokenizer,
        regression=False,
        max_length=8,
    )

    class _TinyTrainer:
        def __init__(self, model: torch.nn.Module) -> None:
            self.model = model
            self.model_wrapped = model
            self.args = SimpleNamespace(
                device=torch.device("cpu"),
                bf16=False,
                fp16=False,
            )

        def predict(self, rows: Any) -> Any:
            prediction_batch = verification_collator(rows)
            prediction_inputs = {
                key: value for key, value in prediction_batch.items() if key != "labels"
            }
            self.model.eval()
            with torch.inference_mode():
                logits = (  # (b, c)
                    self.model(**prediction_inputs).logits.detach().cpu().numpy()
                )
            return SimpleNamespace(predictions=logits)

        def save_model(self, save_directory: str | Path) -> None:
            self.model.save_pretrained(save_directory, safe_serialization=True)

    trainer = _TinyTrainer(adapter)
    original_model = trainer.model
    original_wrapped = trainer.model_wrapped
    artifact = fine_tuning._save_reload_verify_final_artifact(
        trainer,
        tokenizer,
        output_dir=str(tmp_path / "training-output"),
        model_name="offline/tiny-esm2",
        model_revision="a" * 40,
        num_labels=2,
        use_lora=True,
        verification_dataset=verification_rows,
        data_collator=verification_collator,
    )
    adapter_directory = tmp_path / "training-output" / "final_model"
    assert trainer.model is original_model
    assert trainer.model_wrapped is original_wrapped
    assert (adapter_directory / "adapter_config.json").is_file()
    assert (adapter_directory / "adapter_model.safetensors").is_file()
    assert (adapter_directory / "artifact_metadata.json").is_file()
    assert (adapter_directory / "tokenizer.json").is_file()
    assert artifact["path"] == str(adapter_directory.resolve())
    assert len(artifact["tree_sha256"]) == 64
    assert artifact["reload_verified"] is True
    assert artifact["held_out_inference"]["rows"] == 2
    assert artifact["held_out_inference"]["max_absolute_error"] <= 1e-6
    assert artifact["verified_parameter_sha256"] == (
        fine_tuning._persisted_parameter_hashes(adapter, use_lora=True)
    )
    assert loader_calls[-1] == (
        "offline/tiny-esm2",
        {
            "trust_remote_code": True,
            "num_labels": 2,
            "attn_implementation": "sdpa",
            "revision": "a" * 40,
        },
    )
    with torch.inference_mode():
        observed = adapter(**inference_inputs).logits  # (b, c)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
