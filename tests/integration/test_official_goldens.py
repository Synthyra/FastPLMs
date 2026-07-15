"""Fast candidate regression against manifest-declared official goldens."""

from __future__ import annotations

import contextlib
import gc
import importlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from fastplms.registry import ModelSpec, get_model_registry
from tests.parity.test_model_parity import (
    BF16_CONTRACT,
    _assert_logits_contract,
    _assert_tensor_contract,
    _last_hidden,
)
from tools.goldens import validate_golden_bundle

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = get_model_registry()
SEQUENCE_GOLDENS = tuple(
    spec
    for spec in REGISTRY.values()
    if spec.official_golden is not None and spec.family.tokenizer_mode != "structure"
)


def _parameter(spec: ModelSpec) -> object:
    marks = [pytest.mark.large] if spec.size_category == "xlarge" else []
    return pytest.param(spec, id=spec.id, marks=marks)


def _model_class(spec: ModelSpec) -> type[torch.nn.Module]:
    """Resolve the current package implementation declared by the manifest."""

    if spec.family.id == "ankh":
        auto_class = "AutoModel"
    elif "AutoModelForMaskedLM" in spec.auto_map:
        auto_class = "AutoModelForMaskedLM"
    else:
        auto_class = "AutoModel"
    qualified_name = spec.auto_map[auto_class]
    module_name, class_name = qualified_name.rsplit(".", maxsplit=1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    assert issubclass(model_class, torch.nn.Module)
    return model_class


@pytest.mark.gpu
@pytest.mark.parametrize("spec", [_parameter(spec) for spec in SEQUENCE_GOLDENS])
def test_declared_sequence_golden_matches_candidate(spec: ModelSpec) -> None:
    """Run one compact BF16 regression without importing an official package."""

    declaration = spec.official_golden
    assert declaration is not None
    metadata_path = ROOT / declaration.metadata.path
    tensors_path = ROOT / declaration.tensors.path
    validate_golden_bundle(
        spec,
        REGISTRY,
        metadata_path=metadata_path,
        tensors_path=tensors_path,
        declaration=declaration,
    )
    tensors = load_file(tensors_path, device="cpu")
    device = torch.device("cuda")
    use_bf16_autocast = spec.family.bf16_execution == "fp32_parameters_autocast"
    load_dtype = torch.float32 if use_bf16_autocast else torch.bfloat16
    model = (
        _model_class(spec)
        .from_pretrained(
            spec.fast.repo_id,
            revision=spec.fast.revision,
            dtype=load_dtype,
            device_map=device,
        )
        .eval()
    )
    inputs = {
        name.removeprefix("input__"): T.to(device)
        for name, T in tensors.items()
        if name.startswith("input__")
    }
    residue_mask = tensors["residue_mask"].to(device).bool()
    numeric_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16_autocast
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), numeric_context:
        output = model(**inputs, output_hidden_states=True)
    _assert_tensor_contract(
        _last_hidden(output),
        tensors["output__last_hidden_state"].to(device),
        residue_mask,
        BF16_CONTRACT,
        f"{spec.id}:bf16:golden:last_hidden_state",
    )
    official_logits = tensors.get("output__logits")
    candidate_logits = getattr(output, "logits", None)
    assert (candidate_logits is None) == (official_logits is None), (
        f"{spec.id}: golden and candidate output-head contracts differ"
    )
    if official_logits is not None:
        _assert_logits_contract(
            candidate_logits,
            official_logits.to(device),
            residue_mask,
            BF16_CONTRACT,
            f"{spec.id}:bf16:golden:logits",
        )
    del model, output, tensors
    gc.collect()
    torch.cuda.empty_cache()
