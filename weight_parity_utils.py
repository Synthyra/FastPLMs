import torch
from typing import Mapping


def assert_model_parameters_fp32(model: torch.nn.Module, model_name: str) -> None:
    non_fp32: list[dict[str, str]] = []
    parameter_count = 0
    for name, parameter in model.named_parameters():
        parameter_count += 1
        if parameter.dtype != torch.float32:
            non_fp32.append({"name": name, "dtype": str(parameter.dtype)})

    assert parameter_count > 0, f"{model_name} has no parameters."
    assert len(non_fp32) == 0, (
        f"{model_name} parameters must all be torch.float32. "
        f"non_fp32_count={len(non_fp32)} sample={non_fp32[:5]}"
    )


def assert_state_dict_floating_tensors_fp32(
    state_dict: Mapping[str, torch.Tensor],
    state_dict_name: str,
) -> None:
    non_fp32: list[dict[str, str]] = []
    for tensor_name in sorted(state_dict.keys()):
        tensor = state_dict[tensor_name]
        assert torch.is_tensor(tensor), (
            f"{state_dict_name} state_dict entry must be a tensor. "
            f"name={tensor_name} type={type(tensor)}"
        )
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            non_fp32.append({"name": tensor_name, "dtype": str(tensor.dtype)})

    assert len(non_fp32) == 0, (
        f"{state_dict_name} floating tensors must be torch.float32. "
        f"non_fp32_count={len(non_fp32)} sample={non_fp32[:5]}"
    )


def assert_state_dict_equal(
    reference_state_dict: Mapping[str, torch.Tensor],
    candidate_state_dict: Mapping[str, torch.Tensor],
    context: str,
    max_report: int = 5,
) -> None:
    reference_keys = set(reference_state_dict.keys())
    candidate_keys = set(candidate_state_dict.keys())
    only_in_reference = sorted(reference_keys - candidate_keys)
    only_in_candidate = sorted(candidate_keys - reference_keys)
    shape_mismatches: list[dict[str, object]] = []
    differing_tensors: list[dict[str, object]] = []
    max_abs_diff = 0.0
    max_abs_diff_param = ""

    common_keys = sorted(reference_keys & candidate_keys)
    for name in common_keys:
        reference_tensor = reference_state_dict[name].detach().cpu()
        candidate_tensor = candidate_state_dict[name].detach().cpu()
        if reference_tensor.shape != candidate_tensor.shape:
            shape_mismatches.append(
                {
                    "name": name,
                    "reference_shape": list(reference_tensor.shape),
                    "candidate_shape": list(candidate_tensor.shape),
                }
            )
            continue
        if torch.equal(reference_tensor, candidate_tensor):
            continue

        if reference_tensor.is_floating_point() and candidate_tensor.is_floating_point():
            abs_diff = torch.abs(reference_tensor - candidate_tensor)
            param_max_abs_diff = float(torch.max(abs_diff).item())
            param_mean_abs_diff = float(torch.mean(abs_diff).item())
        else:
            param_max_abs_diff = float("nan")
            param_mean_abs_diff = float("nan")

        differing_tensors.append(
            {
                "name": name,
                "max_abs_diff": param_max_abs_diff,
                "mean_abs_diff": param_mean_abs_diff,
            }
        )

        if reference_tensor.is_floating_point() and candidate_tensor.is_floating_point():
            if param_max_abs_diff > max_abs_diff:
                max_abs_diff = param_max_abs_diff
                max_abs_diff_param = name

    assert len(only_in_reference) == 0 and len(only_in_candidate) == 0 and len(shape_mismatches) == 0 and len(differing_tensors) == 0, (
        f"{context} requires exact state_dict parity in torch.float32. "
        f"only_in_reference={only_in_reference[:max_report]} "
        f"only_in_candidate={only_in_candidate[:max_report]} "
        f"shape_mismatches={shape_mismatches[:max_report]} "
        f"diff_param_count={len(differing_tensors)} "
        f"max_abs_diff={max_abs_diff} "
        f"max_abs_diff_param={max_abs_diff_param} "
        f"diff_params_sample={differing_tensors[:max_report]}"
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
