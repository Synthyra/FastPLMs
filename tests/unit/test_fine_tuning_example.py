"""Lightweight source-contract tests for the optional fine-tuning example."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import numpy as np
import pytest
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "fine_tuning.py"


def _tree() -> ast.Module:
    return ast.parse(EXAMPLE.read_text(encoding="utf-8"), filename=str(EXAMPLE))


def _assignment_value(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment for {name}.")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing function {name}.")


def _definition(tree: ast.Module, name: str) -> ast.FunctionDef | ast.ClassDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(f"Missing definition {name}.")


def test_lora_configuration_persists_classifier_and_preserves_custom_modules() -> None:
    tree = _tree()
    classifier_name = _assignment_value(tree, "CLASSIFIER_MODULE_NAME")
    helper = _function(tree, "_ensure_classifier_persistence")
    namespace = {
        "Any": Any,
        "CLASSIFIER_MODULE_NAME": classifier_name,
    }
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(EXAMPLE), "exec"), namespace)
    ensure_persistence = namespace["_ensure_classifier_persistence"]

    config = SimpleNamespace(modules_to_save=["contact_head"])
    assert ensure_persistence(config) is config
    assert config.modules_to_save == ["contact_head", "classifier"]

    ensure_persistence(config)
    assert config.modules_to_save == ["contact_head", "classifier"]

    empty_config = SimpleNamespace(modules_to_save=None)
    ensure_persistence(empty_config)
    assert empty_config.modules_to_save == ["classifier"]


def test_defaults_only_advertise_sequence_classification_artifacts() -> None:
    tree = _tree()
    default_model = _assignment_value(tree, "DEFAULT_MODEL")
    assert default_model == "Synthyra/ESM2-8M"

    for function_name in ("train_regression_model", "train_classification_model"):
        function = _function(tree, function_name)
        model_default = function.args.defaults[0]
        assert isinstance(model_default, ast.Name)
        assert model_default.id == "DEFAULT_MODEL"

    source = EXAMPLE.read_text(encoding="utf-8")
    assert "Synthyra/ESMplusplus_small" not in source
    assert "Synthyra/ESMplusplus_large" not in source


def test_cli_exposes_symmetric_lora_switch() -> None:
    tree = _tree()
    lora_argument: ast.Call | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        options = [
            ast.literal_eval(argument)
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        ]
        if "--use-lora" in options:
            lora_argument = node
            assert {"--use-lora", "--use_lora"}.issubset(options)
            break

    assert lora_argument is not None
    keywords = {keyword.arg: keyword.value for keyword in lora_argument.keywords}
    action = keywords["action"]
    assert isinstance(action, ast.Attribute)
    assert isinstance(action.value, ast.Name)
    assert (action.value.id, action.attr) == ("argparse", "BooleanOptionalAction")
    assert ast.literal_eval(keywords["default"]) is True


def test_reporting_is_opt_in_for_the_minimal_training_install() -> None:
    tree = _tree()
    plot_argument: ast.Call | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        options = [
            ast.literal_eval(argument)
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        ]
        if "--plot-results" in options:
            plot_argument = node
            break

    assert plot_argument is not None
    keywords = {keyword.arg: keyword.value for keyword in plot_argument.keywords}
    assert ast.literal_eval(keywords["default"]) is False
    for function_name in ("train_regression_model", "train_classification_model"):
        function = _function(tree, function_name)
        defaults = dict(
            zip(
                (argument.arg for argument in function.args.args[-len(function.args.defaults) :]),
                function.args.defaults,
                strict=True,
            )
        )
        assert ast.literal_eval(defaults["plot_results"]) is False


def test_max_length_contract_is_an_encoded_budget_including_added_tokens() -> None:
    tree = _tree()
    for function_name in (
        "train_regression_model",
        "train_classification_model",
        "PairDatasetHF",
        "SequenceDatasetHF",
    ):
        definition = _definition(tree, function_name)
        docstring = ast.get_docstring(definition) or ""
        assert "Encoded token budget" in docstring
        assert "special tokens" in docstring

    max_length_argument: ast.Call | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        options = [
            ast.literal_eval(argument)
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        ]
        if "--max_length" in options:
            max_length_argument = node
            break
    assert max_length_argument is not None
    keywords = {keyword.arg: keyword.value for keyword in max_length_argument.keywords}
    help_text = ast.literal_eval(keywords["help"])
    assert "encoded token count" in help_text
    assert "special and pair separator tokens" in help_text


def test_plot_contract_uses_task_output_paths_and_has_no_interactive_or_overwrite_path() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")
    assert 'Path(output_dir) / "regression_results.png"' in source
    assert 'Path(output_dir) / "classification_results.png"' in source
    assert "plt.show(" not in source

    for function_name in ("plot_regression_results", "plot_classification_results"):
        function = _function(_tree(), function_name)
        assert "output_path" in [argument.arg for argument in function.args.args]
        called_names = {
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_save_figure_exclusive" in called_names


def test_pair_token_budget_includes_special_tokens_at_the_exact_boundary() -> None:
    torch = pytest.importorskip("torch")
    tree = _tree()
    namespace = {"Any": Any, "torch": torch}
    nodes = [
        _definition(tree, "_encoded_length"),
        _definition(tree, "_fits_token_budget"),
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EXAMPLE), "exec"), namespace)

    class FakeTokenizer:
        def __call__(
            self,
            sequence: str,
            pair: str | None,
            **kwargs: Any,
        ) -> dict[str, list[int]]:
            assert kwargs["add_special_tokens"] is True
            assert kwargs["truncation"] is False
            special_tokens = 3 if pair is not None else 2
            return {"input_ids": [0] * (len(sequence) + len(pair or "") + special_tokens)}

    fits = namespace["_fits_token_budget"]
    assert fits(FakeTokenizer(), "AC", "DEF", 8)
    assert not fits(FakeTokenizer(), "AC", "DEF", 7)


def test_pair_collator_enforces_longest_first_tokenizer_limit() -> None:
    torch = pytest.importorskip("torch")
    tree = _tree()
    namespace = {"Any": Any, "torch": torch}
    nodes = [
        _definition(tree, "_tokenization_kwargs"),
        _definition(tree, "PairCollator"),
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EXAMPLE), "exec"), namespace)

    class FakeTokenizer:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] | None = None

        def __call__(self, seqs_a: Any, seqs_b: Any, **kwargs: Any) -> dict[str, Any]:
            del seqs_a, seqs_b
            self.kwargs = kwargs
            return {
                "input_ids": torch.zeros(  # (b=2, l=max_length)
                    (2, kwargs["max_length"]),
                    dtype=torch.long,
                ),
                "attention_mask": torch.ones((2, kwargs["max_length"]), dtype=torch.long),
            }

    tokenizer = FakeTokenizer()
    collator = namespace["PairCollator"](tokenizer, regression=True, max_length=16)
    batch = collator([("AC", "DE", 1.0), ("FG", "HI", 2.0)])

    assert tokenizer.kwargs is not None
    assert tokenizer.kwargs["truncation"] == "longest_first"
    assert tokenizer.kwargs["max_length"] == 16
    assert tokenizer.kwargs["pad_to_multiple_of"] == 8
    assert batch["input_ids"].shape == (2, 16)
    assert batch["labels"].dtype == torch.float32


def test_training_manifest_records_reproducible_model_data_and_tokenizer_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    tree = _tree()
    namespace = {
        "Any": Any,
        "Mapping": Mapping,
        "Path": Path,
        "TrainingArguments": Any,
        "hashlib": hashlib,
        "json": json,
        "metadata": metadata,
        "platform": platform,
        "_IMMUTABLE_REVISION": re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE),
        "_PINNED_DEFAULT_REVISIONS": {},
        "sys": sys,
        "torch": torch,
    }
    nodes = [
        _definition(tree, name)
        for name in (
            "_package_version",
            "_json_safe",
            "_sha256_json",
            "_tokenizer_identity",
            "_tree_sha256",
            "_immutable_source_identity",
            "_verify_training_source_unchanged",
            "_effective_attention_backend",
            "_ordered_rows_sha256",
            "_dataset_identity",
            "_write_training_manifest",
        )
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EXAMPLE), "exec"), namespace)

    class FakeConfig:
        _name_or_path = "Synthyra/tiny"
        _commit_hash = "a" * 40
        attn_backend = "sdpa"

        def to_dict(self) -> dict[str, Any]:
            return {"hidden_size": 8, "attn_backend": self.attn_backend}

    class FakeAdapterConfig:
        def to_dict(self) -> dict[str, Any]:
            return {"r": 2, "target_modules": {"query", "value"}}

    class FakeModel:
        config = FakeConfig()

        def __init__(self) -> None:
            self.peft_config = {"default": FakeAdapterConfig()}
            self.parameter = torch.nn.Parameter(torch.ones(1))
            self._fastplms_training_source_identity = {
                "kind": "hub",
                "source_kind": "model",
                "repo_id": "Synthyra/tiny",
                "revision": "a" * 40,
            }

        def parameters(self) -> Any:
            yield self.parameter

    class FakeTokenizer:
        name_or_path = "Synthyra/tiny"
        cls_token_id = 0
        eos_token_id = 2
        pad_token_id = 1

        def __init__(self) -> None:
            self.init_kwargs = {"revision": "a" * 40}

        def get_vocab(self) -> dict[str, int]:
            return {"<cls>": 0, "<pad>": 1, "<eos>": 2, "A": 3}

    class FakeDataset:
        _fingerprint = "dataset-fingerprint"
        info = SimpleNamespace(builder_name="builder", config_name="config", version="1.0")
        rows: ClassVar[list[dict[str, object]]] = [
            {"sequence": "AC", "label": 0},
            {"sequence": "DEF", "label": 1},
            {"sequence": "GHIK", "label": 0},
        ]

        def __len__(self) -> int:
            return len(self.rows)

        def __iter__(self) -> Any:
            return iter(self.rows)

    class FakeTrainingArguments:
        device = torch.device("cpu")
        bf16 = False
        fp16 = False
        optim = "adamw_torch"
        lr_scheduler_type = "linear"
        warmup_steps = 5
        weight_decay = 0.01
        eval_strategy = "steps"
        eval_steps = 4
        save_strategy = "steps"
        save_steps = 4
        logging_strategy = "steps"
        logging_steps = 2
        load_best_model_at_end = True
        metric_for_best_model = "eval_loss"
        greater_is_better = False
        label_names: ClassVar[list[str]] = ["labels"]
        report_to: ClassVar[list[str]] = []

    monkeypatch.setattr(sys, "argv", ["fine_tuning.py", "--task", "regression"])
    namespace["_write_training_manifest"](
        str(tmp_path),
        task="test",
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        model_name="Synthyra/tiny",
        model_revision="a" * 40,
        seed=7,
        max_length=16,
        use_lora=True,
        batch_size=2,
        gradient_accumulation_steps=3,
        learning_rate=1e-4,
        num_epochs=1,
        full_determinism=True,
        datasets={"train": FakeDataset()},
        dataset_contracts={
            "train": {
                "source": {
                    "kind": "hub",
                    "source_kind": "dataset",
                    "repo_id": "Synthyra/tiny-data",
                    "revision": "b" * 40,
                },
                "split": "train",
                "columns": ("sequence", "label"),
            }
        },
        training_arguments=FakeTrainingArguments(),
        patience=2,
        final_artifact={"reload_verified": True, "tree_sha256": "c" * 64},
    )

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == ["fine_tuning.py", "--task", "regression"]
    assert manifest["model"]["revision"] == "a" * 40
    assert manifest["model"]["attention_backend"] == "sdpa"
    assert manifest["model"]["requested_attention_backend"] == "sdpa"
    assert manifest["model"]["effective_attention_backend"] == "sdpa"
    assert manifest["model"]["parameter_dtype"] == "torch.float32"
    assert len(manifest["model"]["configuration_sha256"]) == 64
    assert manifest["model"]["adapters"]["default"]["target_modules"] == [
        "query",
        "value",
    ]
    assert manifest["tokenizer"]["revision"] == "a" * 40
    assert len(manifest["tokenizer"]["vocab_sha256"]) == 64
    assert manifest["datasets"]["train"]["library_fingerprint_advisory"] == "dataset-fingerprint"
    assert len(manifest["datasets"]["train"]["ordered_rows_sha256"]) == 64
    assert manifest["datasets"]["train"]["columns"] == ["sequence", "label"]
    assert manifest["datasets"]["train"]["source"]["revision"] == "b" * 40
    assert manifest["datasets"]["train"]["rows"] == 3
    assert manifest["training"]["compute_dtype"] == "torch.float32"
    assert manifest["training"]["optimizer"] == "adamw_torch"
    assert manifest["training"]["early_stopping_patience"] == 2
    assert manifest["training"]["eval_steps"] == 4
    assert manifest["training"]["save_steps"] == 4
    assert manifest["training"]["logging_steps"] == 2
    assert manifest["training"]["max_length_semantics"] == (
        "encoded token budget including tokenizer-added special and pair tokens"
    )
    assert manifest["final_artifact"]["reload_verified"] is True


def test_immutable_sources_reject_moving_refs_pin_only_shipped_defaults_and_detect_drift(
    tmp_path: Path,
) -> None:
    tree = _tree()
    default_model = _assignment_value(tree, "DEFAULT_MODEL")
    default_revision = _assignment_value(tree, "DEFAULT_MODEL_REVISION")
    namespace = {
        "Any": Any,
        "Mapping": Mapping,
        "Path": Path,
        "hashlib": hashlib,
        "_IMMUTABLE_REVISION": re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE),
        "_PINNED_DEFAULT_REVISIONS": {("model", default_model): default_revision},
    }
    nodes = [
        _definition(tree, name)
        for name in (
            "_tree_sha256",
            "_immutable_source_identity",
            "_verify_training_source_unchanged",
        )
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EXAMPLE), "exec"), namespace)
    identity = namespace["_immutable_source_identity"]

    pinned = identity(default_model, None, source_kind="model")
    assert pinned["revision"] == default_revision
    with pytest.raises(ValueError, match="40-character"):
        identity("Custom/model", None, source_kind="model")
    with pytest.raises(ValueError, match="branches, tags"):
        identity("Custom/model", "main", source_kind="model")

    local_model = tmp_path / "local-model"
    local_model.mkdir()
    (local_model / "config.json").write_text('{"hidden_size": 8}\n', encoding="utf-8")
    loaded_identity = identity(str(local_model), None, source_kind="model")
    model = SimpleNamespace(_fastplms_training_source_identity=loaded_identity)
    assert (
        namespace["_verify_training_source_unchanged"](
            model,
            str(local_model),
            None,
        )
        == loaded_identity
    )
    (local_model / "config.json").write_text('{"hidden_size": 16}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after initialization"):
        namespace["_verify_training_source_unchanged"](
            model,
            str(local_model),
            None,
        )


def test_ordered_training_row_hash_is_content_and_order_sensitive() -> None:
    tree = _tree()
    namespace = {"Any": Any, "Mapping": Mapping, "hashlib": hashlib, "json": json}
    nodes = [
        _definition(tree, name) for name in ("_json_safe", "_sha256_json", "_ordered_rows_sha256")
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EXAMPLE), "exec"), namespace)
    hash_rows = namespace["_ordered_rows_sha256"]
    rows = [
        {"sequence": "AC", "label": 0, "ignored": "x"},
        {"sequence": "DEF", "label": 1, "ignored": "y"},
    ]
    columns = ("sequence", "label")

    baseline = hash_rows(rows, columns)
    assert hash_rows([dict(row) for row in rows], columns) == baseline
    assert hash_rows(list(reversed(rows)), columns) != baseline
    changed = [dict(row) for row in rows]
    changed[1]["sequence"] = "DEG"
    assert hash_rows(changed, columns) != baseline
    ignored_change = [dict(row) for row in rows]
    ignored_change[0]["ignored"] = "changed"
    assert hash_rows(ignored_change, columns) == baseline


def test_persisted_hash_scope_covers_full_state_or_only_lora_payload() -> None:
    torch = pytest.importorskip("torch")
    tree = _tree()
    namespace = {"Any": Any, "hashlib": hashlib, "torch": torch}
    nodes = [_definition(tree, name) for name in ("_tensor_sha256", "_persisted_parameter_hashes")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EXAMPLE), "exec"), namespace)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.classifier = torch.nn.Linear(2, 1)
            self.lora_A = torch.nn.Parameter(torch.ones(1, 2))
            self.modules_to_save = torch.nn.Linear(2, 1)
            self.register_buffer("running_state", torch.tensor([3.0]))  # (1,)

    model = FakeModel()
    hashes = namespace["_persisted_parameter_hashes"]
    full_hashes = hashes(model, use_lora=False)
    lora_hashes = hashes(model, use_lora=True)
    assert set(full_hashes) == set(model.state_dict())
    assert "running_state" in full_hashes
    assert lora_hashes
    assert all("lora_" in name or "modules_to_save" in name for name in lora_hashes)
    assert not any("backbone" in name for name in lora_hashes)
    assert not any(name.startswith("classifier") for name in lora_hashes)


def test_atomic_final_artifact_reload_preserves_trainer_and_held_out_logits(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    tree = _tree()
    namespace = {
        "Any": Any,
        "Mapping": Mapping,
        "Path": Path,
        "Trainer": Any,
        "hashlib": hashlib,
        "json": json,
        "np": np,
        "os": os,
        "shutil": shutil,
        "tempfile": tempfile,
        "torch": torch,
    }
    nodes = [
        _definition(tree, name)
        for name in (
            "_tree_sha256",
            "_tensor_sha256",
            "_persisted_parameter_hashes",
            "_primary_prediction_tensor",
            "_held_out_reload_verification",
            "_save_reload_verify_final_artifact",
        )
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EXAMPLE), "exec"), namespace)

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(2, 1, bias=False)
            self._fastplms_training_source_identity = {
                "kind": "hub",
                "source_kind": "model",
                "repo_id": "offline/tiny",
                "revision": "a" * 40,
            }

        def forward(self, input_ids: Any, labels: Any = None) -> Any:
            del labels
            return SimpleNamespace(logits=self.projection(input_ids.float()))

    model = TinyModel().eval()
    with torch.no_grad():
        model.projection.weight.copy_(torch.tensor([[0.25, -0.5]]))  # (c=1, d=2)

    def collate(rows: list[tuple[torch.Tensor, float]]) -> dict[str, torch.Tensor]:
        inputs, labels = zip(*rows, strict=True)
        return {
            "input_ids": torch.stack(inputs),
            "labels": torch.tensor(labels),  # (b,)
        }

    class TinyTrainer:
        def __init__(self, trained_model: TinyModel) -> None:
            self.model = trained_model
            self.model_wrapped = trained_model
            self.args = SimpleNamespace(device=torch.device("cpu"), bf16=False, fp16=False)

        def predict(self, rows: Any) -> Any:
            batch = collate(rows)
            with torch.inference_mode():
                logits = self.model(**batch).logits.numpy()  # (b, c)
            return SimpleNamespace(predictions=logits)

        def save_model(self, directory: str | Path) -> None:
            Path(directory, "model.safetensors").write_bytes(b"safe-test-weights")

    class TinyTokenizer:
        def save_pretrained(self, directory: str | Path) -> None:
            Path(directory, "tokenizer.json").write_text("{}\n", encoding="utf-8")

    trainer = TinyTrainer(model)
    original_model = trainer.model
    reloaded = TinyModel()
    reloaded.load_state_dict(model.state_dict())
    namespace["_verify_training_source_unchanged"] = (
        lambda current_model, model_name, model_revision: dict(
            current_model._fastplms_training_source_identity
        )
    )
    namespace["_reload_final_model"] = lambda *args, **kwargs: reloaded
    verification_rows = [
        (torch.tensor([1.0, 2.0]), 0.0),
        (torch.tensor([3.0, 4.0]), 1.0),
    ]

    artifact = namespace["_save_reload_verify_final_artifact"](
        trainer,
        TinyTokenizer(),
        output_dir=str(tmp_path / "output"),
        model_name="offline/tiny",
        model_revision="a" * 40,
        num_labels=1,
        use_lora=False,
        verification_dataset=verification_rows,
        data_collator=collate,
    )
    final_dir = tmp_path / "output" / "final_model"
    assert trainer.model is original_model
    assert trainer.model_wrapped is original_model
    assert final_dir.is_dir()
    assert not list((tmp_path / "output").glob(".final-model-*"))
    assert artifact["reload_verified"] is True
    assert artifact["held_out_inference"]["rows"] == 2
    assert artifact["held_out_inference"]["max_absolute_error"] == 0.0
    metadata_payload = json.loads(
        (final_dir / "artifact_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata_payload["held_out_inference"] == artifact["held_out_inference"]


def test_lora_adapter_save_reload_preserves_trained_classifier_logits(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from fastplms.models.esm2.modeling_fastesm import (
        FastEsmConfig,
        FastEsmForSequenceClassification,
    )

    torch.manual_seed(7)
    config = FastEsmConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=2,
        pad_token_id=1,
        mask_token_id=5,
        position_embedding_type="absolute",
        attn_backend="eager",
    )
    base = FastEsmForSequenceClassification(config)
    base_state = {name: tensor.detach().clone() for name, tensor in base.state_dict().items()}
    adapter = peft.get_peft_model(
        base,
        peft.LoraConfig(
            task_type=peft.TaskType.SEQ_CLS,
            r=2,
            lora_alpha=4,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
        ),
    )
    with torch.no_grad():
        for parameter in adapter.base_model.model.classifier.parameters():
            parameter.add_(0.25)

    inputs = {
        "input_ids": torch.tensor([[0, 3, 4, 2]], dtype=torch.long),  # (b=1, l=4)
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    adapter.eval()
    with torch.inference_mode():
        expected = adapter(**inputs).logits  # (b, c)
    adapter.save_pretrained(tmp_path)

    restored_base = FastEsmForSequenceClassification(config)
    restored_base.load_state_dict(base_state)
    restored = peft.PeftModel.from_pretrained(restored_base, tmp_path).eval()
    with torch.inference_mode():
        observed = restored(**inputs).logits  # (b, c)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
