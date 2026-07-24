from __future__ import annotations

import torch
from collections.abc import Mapping


def assert_model_parameters_fp32(model: torch.nn.Module, model_name: str) -> None:
    non_fp32: list[dict[str, str]] = []
    parameter_count = 0
    for name, parameter in model.named_parameters():
        # parameter: (...)
        parameter_count += 1
        if parameter.dtype != torch.float32:
            non_fp32.append({"name": name, "dtype": str(parameter.dtype)})

    if parameter_count == 0:
        raise AssertionError(f"{model_name} has no parameters.")
    if non_fp32:
        raise AssertionError(
            f"{model_name} parameters must all be torch.float32. "
            f"non_fp32_count={len(non_fp32)} sample={non_fp32[:5]}"
        )


def assert_state_dict_floating_tensors_fp32(
    state_dict: Mapping[str, torch.Tensor],
    state_dict_name: str,
) -> None:
    non_fp32: list[dict[str, str]] = []
    for tensor_name in sorted(state_dict.keys()):
        tensor = state_dict[tensor_name]  # (...)
        if not torch.is_tensor(tensor):
            raise AssertionError(
                f"{state_dict_name} state_dict entry must be a tensor. "
                f"name={tensor_name} type={type(tensor)}"
            )
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            non_fp32.append({"name": tensor_name, "dtype": str(tensor.dtype)})

    if non_fp32:
        raise AssertionError(
            f"{state_dict_name} floating tensors must be torch.float32. "
            f"non_fp32_count={len(non_fp32)} sample={non_fp32[:5]}"
        )


def assert_state_dict_equal(
    reference_state_dict: Mapping[str, torch.Tensor],
    candidate_state_dict: Mapping[str, torch.Tensor],
    context: str,
    max_report: int = 10,
) -> None:
    reference_keys = set(reference_state_dict)
    candidate_keys = set(candidate_state_dict)
    missing = sorted(reference_keys - candidate_keys)
    unexpected = sorted(candidate_keys - reference_keys)
    errors: list[str] = []
    if missing:
        errors.append(f"missing keys: {missing[:max_report]}")
    if unexpected:
        errors.append(f"unexpected keys: {unexpected[:max_report]}")
    for name in sorted(reference_keys & candidate_keys):
        reference = reference_state_dict[name]  # (...)
        candidate = candidate_state_dict[name]  # (...)
        if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
            errors.append(f"{name}: both entries must be tensors")
            continue
        if reference.shape != candidate.shape:
            errors.append(f"{name}: shape {tuple(reference.shape)} != {tuple(candidate.shape)}")
            continue
        if reference.dtype != candidate.dtype:
            errors.append(f"{name}: dtype {reference.dtype} != {candidate.dtype}")
            continue
        if not torch.equal(reference, candidate):
            errors.append(f"{name}: tensor values differ")
    if errors:
        raise AssertionError(
            f"{context} state_dict parity failed: {' | '.join(errors[:max_report])}"
        )


def assert_models_fp32_and_equal(
    reference_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    context: str,
    max_report: int = 5,
) -> None:
    assert_model_parameters_fp32(model=reference_model, model_name=f"{context} reference model")
    assert_model_parameters_fp32(model=candidate_model, model_name=f"{context} candidate model")
    assert_state_dict_equal(
        reference_state_dict=reference_model.state_dict(),
        candidate_state_dict=candidate_model.state_dict(),
        context=context,
        max_report=max_report,
    )
