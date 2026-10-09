"""Candidate regression against manifest-declared official goldens.

Each sequence golden holds the official model's strict-FP32 outputs, which agree
across devices to rounding, and the official BF16 outputs from the same run.
FastPLMs in FP32 must match the FP32 outputs under the cross-device FP32 contract.
FastPLMs in BF16 must be no farther from them than official BF16 was, within a
fixed ratio, so neither check depends on the GPU that recorded the golden.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import pytest
import torch

from pathlib import Path
from safetensors.torch import load_file
from tests.conftest import strict_fp32_matmul
from tests.parity.test_model_parity import (
    BF16_GOLDEN_ERROR_RATIO_HARD,
    BF16_GOLDEN_ERROR_RATIO_TARGET,
    FP32_GOLDEN_CONTRACT,
    _assert_logits_contract,
    _assert_tensor_contract,
    _assert_upper,
    _last_hidden,
    _logits_metrics,
    tensor_metrics,
)

from fastplms.registry import ModelSpec, get_model_registry
from tools.goldens import OFFICIAL_BF16_PREFIX, validate_golden_bundle


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


def _golden_tensors(spec: ModelSpec) -> dict[str, torch.Tensor]:
    """Validate the declared bundle against its pinned digests and load it."""

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
    return load_file(tensors_path, device="cpu")  # (...) one tensor per golden name


def _run_candidate(
    spec: ModelSpec,
    tensors: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Run FastPLMs on the golden inputs the way the family executes ``dtype``."""

    device = torch.device("cuda")
    use_bf16_autocast = (
        dtype == torch.bfloat16 and spec.family.bf16_execution == "fp32_parameters_autocast"
    )
    load_dtype = torch.float32 if use_bf16_autocast else dtype
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
    if dtype == torch.float32:
        numeric_context = strict_fp32_matmul()
    elif use_bf16_autocast:
        numeric_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        numeric_context = contextlib.nullcontext()
    with torch.inference_mode(), numeric_context:
        output = model(**inputs, output_hidden_states=True)
    candidate = {"last_hidden_state": _last_hidden(output).float().cpu()}  # (b, l, d)
    logits = getattr(output, "logits", None)
    if logits is not None:
        candidate["logits"] = logits.float().cpu()  # (b, l, c)
    del model, output
    gc.collect()
    torch.cuda.empty_cache()
    return candidate


def _assert_same_output_heads(
    spec: ModelSpec,
    tensors: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> None:
    golden_heads = {name.removeprefix("output__") for name in tensors if name.startswith("output__")}
    assert golden_heads == set(candidate), (
        f"{spec.id}: golden and candidate output-head contracts differ "
        f"({sorted(golden_heads)} != {sorted(candidate)})"
    )


@pytest.mark.gpu
@pytest.mark.checkpoint
@pytest.mark.parametrize("spec", [_parameter(spec) for spec in SEQUENCE_GOLDENS])
def test_declared_sequence_golden_matches_candidate(spec: ModelSpec) -> None:
    """FastPLMs in strict FP32 reproduces the official FP32 outputs."""

    tensors = _golden_tensors(spec)
    # residue_mask: (b, l)
    residue_mask = tensors["residue_mask"].bool()
    candidate = _run_candidate(spec, tensors, torch.float32)
    _assert_same_output_heads(spec, tensors, candidate)
    _assert_tensor_contract(
        candidate["last_hidden_state"],
        tensors["output__last_hidden_state"],
        residue_mask,
        FP32_GOLDEN_CONTRACT,
        f"{spec.id}:fp32:golden:last_hidden_state",
    )
    if "logits" in candidate:
        _assert_logits_contract(
            candidate["logits"],
            tensors["output__logits"],
            residue_mask,
            FP32_GOLDEN_CONTRACT,
            f"{spec.id}:fp32:golden:logits",
        )


@pytest.mark.gpu
@pytest.mark.checkpoint
@pytest.mark.parametrize("spec", [_parameter(spec) for spec in SEQUENCE_GOLDENS])
def test_declared_sequence_golden_bounds_candidate_bf16_error(spec: ModelSpec) -> None:
    """FastPLMs BF16 is about as close to the official FP32 outputs as official BF16."""

    tensors = _golden_tensors(spec)
    # residue_mask: (b, l)
    residue_mask = tensors["residue_mask"].bool()
    candidate = _run_candidate(spec, tensors, torch.bfloat16)
    _assert_same_output_heads(spec, tensors, candidate)
    for head, value in candidate.items():
        truth = tensors[f"output__{head}"]  # (b, l, d)
        official = tensors[OFFICIAL_BF16_PREFIX + head]  # (b, l, d)
        context = f"{spec.id}:bf16:golden:{head}"
        candidate_error = tensor_metrics(value, truth, residue_mask)
        official_error = tensor_metrics(official, truth, residue_mask)
        for name in ("relative_l2", "relative_q999"):
            _assert_upper(
                f"{name} ratio to official BF16",
                getattr(candidate_error, name) / getattr(official_error, name),
                BF16_GOLDEN_ERROR_RATIO_TARGET,
                BF16_GOLDEN_ERROR_RATIO_HARD,
                context,
            )
        if head == "logits":
            # Jensen-Shannon divergence grows with the square of a small logit error.
            candidate_jsd = _logits_metrics(value, truth, residue_mask, context).mean_jsd
            official_jsd = _logits_metrics(official, truth, residue_mask, context).mean_jsd
            _assert_upper(
                "mean_jsd ratio to official BF16",
                candidate_jsd / official_jsd,
                BF16_GOLDEN_ERROR_RATIO_TARGET**2,
                BF16_GOLDEN_ERROR_RATIO_HARD**2,
                context,
            )
