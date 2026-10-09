"""Weights survive ``save_pretrained`` and ``from_pretrained`` bit for bit, in every model family.

A tiny configuration of each family is built with random parameters, saved as safetensors and loaded again.
The loaded state dictionary must hold every key of the original, no extra key, and each tensor with the same
dtype, shape and bytes. This is the weight-loading contract that a published checkpoint relies on, checked on
a CPU with no download.
"""

from __future__ import annotations

import importlib
import pytest
import torch

from pathlib import Path
from safetensors import safe_open
from tests.unit.test_boltz_checkpoint_io import _TinyCore
from tests.unit.tiny_families import tiny_config

from fastplms.models.boltz import modeling_boltz2
from fastplms.registry import get_model_registry
from tools.tensor_digests import tensor_bytes


def model_classes() -> list[tuple[str, str]]:
    """Every (family, auto class) pair that builds a model from a configuration."""
    pairs = []
    for family_id, family in get_model_registry().families.items():
        for auto_class in sorted(family.auto_map):
            if auto_class != "AutoConfig":
                pairs.append((family_id, auto_class))
    return pairs


@pytest.fixture(autouse=True)
def stand_in_boltz_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real Boltz2 core is the 2 GB inference network; its weights are loaded by ``from_boltz_checkpoint``.

    The Hugging Face wrapper around it saves and loads whatever core it holds, so one parameter is enough to
    prove the wrapper keeps every key and byte.
    """
    monkeypatch.setattr(modeling_boltz2, "Boltz2InferenceCore", _TinyCore)


def load_arguments(family_id: str) -> dict[str, object]:
    """Never reach the network or the Hub cache: ESMFold2 would otherwise also fetch its ESMC backbone."""
    arguments: dict[str, object] = {"local_files_only": True}
    if family_id == "esmfold2":
        arguments["load_esmc"] = False
    return arguments


def resolve(path: str) -> type:
    module_name, _, class_name = path.rpartition(".")
    return getattr(importlib.import_module(module_name), class_name)


def randomized(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Give every floating parameter and buffer distinct random values so that zeros cannot hide a lost key."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for tensor in [*model.parameters(), *model.buffers()]:
            if tensor.is_floating_point() and tensor.numel() > 1:
                tensor.copy_(torch.randn(tensor.shape, generator=generator).to(tensor.dtype))
    return model.eval()


def assert_same_state(original: dict[str, torch.Tensor], loaded: dict[str, torch.Tensor]) -> None:
    # original, loaded: (...) one tensor per parameter name, checkpoint-defined shapes
    assert sorted(loaded) == sorted(original), (
        f"missing {sorted(set(original) - set(loaded))[:5]}, unexpected {sorted(set(loaded) - set(original))[:5]}"
    )
    for key, tensor in original.items():
        other = loaded[key]
        assert other.dtype == tensor.dtype and other.shape == tensor.shape, key
        assert tensor_bytes(other) == tensor_bytes(tensor), f"{key} changed bytes"


@pytest.mark.parametrize(("family_id", "auto_class"), model_classes(), ids=lambda value: str(value))
def test_every_state_dict_key_and_tensor_survives_a_save_and_load(
    tmp_path: Path, family_id: str, auto_class: str,
) -> None:
    family = get_model_registry().families[family_id]
    model_class = resolve(family.auto_map[auto_class])
    model = randomized(model_class(tiny_config(family_id)), seed=len(family_id) + len(auto_class))
    original = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    assert original, "the tiny model has no parameters, so the round trip proves nothing"

    model.save_pretrained(tmp_path, safe_serialization=True)
    saved = sorted(tmp_path.glob("*.safetensors"))
    assert saved, "save_pretrained wrote no safetensors file"
    with safe_open(str(saved[0]), framework="pt") as handle:
        assert set(handle.keys()) <= set(original), "the file holds a key the model does not have"

    loaded, report = model_class.from_pretrained(tmp_path, output_loading_info=True, **load_arguments(family_id))
    assert not report["missing_keys"], report["missing_keys"][:5]
    assert not report["unexpected_keys"], report["unexpected_keys"][:5]
    assert not report["mismatched_keys"], report["mismatched_keys"][:5]
    assert_same_state(original, {key: tensor for key, tensor in loaded.state_dict().items()})


@pytest.mark.parametrize("family_id", sorted(get_model_registry().families))
def test_a_second_save_of_the_loaded_model_is_byte_identical(tmp_path: Path, family_id: str) -> None:
    family = get_model_registry().families[family_id]
    model_class = resolve(family.auto_map["AutoModel"])
    model = randomized(model_class(tiny_config(family_id)), seed=11)
    first, second = tmp_path / "first", tmp_path / "second"
    model.save_pretrained(first, safe_serialization=True)
    model_class.from_pretrained(first, **load_arguments(family_id)).save_pretrained(second, safe_serialization=True)
    names = sorted(path.name for path in first.glob("*.safetensors"))
    assert names == sorted(path.name for path in second.glob("*.safetensors"))
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
