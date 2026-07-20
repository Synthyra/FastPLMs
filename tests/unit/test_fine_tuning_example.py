"""Lightweight source-contract tests for the optional fine-tuning example."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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
    base_state = {
        name: tensor.detach().clone()
        for name, tensor in base.state_dict().items()
    }
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
        "input_ids": torch.tensor([[0, 3, 4, 2]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    adapter.eval()
    with torch.inference_mode():
        expected = adapter(**inputs).logits
    adapter.save_pretrained(tmp_path)

    restored_base = FastEsmForSequenceClassification(config)
    restored_base.load_state_dict(base_state)
    restored = peft.PeftModel.from_pretrained(restored_base, tmp_path).eval()
    with torch.inference_mode():
        observed = restored(**inputs).logits

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
