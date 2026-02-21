import torch
from torch.nn.functional import mse_loss
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
    max_report: int = 10,
) -> None:
    error_msgs = []
    for (ref_name, ref_tensor), (cand_name, cand_tensor) in zip(reference_state_dict.items(), candidate_state_dict.items()):
        if ref_name != cand_name:
            msg = f"Name mismatch: {ref_name} != {cand_name}"
            print(msg)
            error_msgs.append(msg)
        else:
            diff = mse_loss(ref_tensor, cand_tensor).item()
            if diff > 0.0:
                msg = f"{ref_name}: {diff}"
                print(msg)
                error_msgs.append(msg)
    assert not error_msgs, (
        f"{context} state_dict parity failed:{' | '.join(error_msgs[:max_report])}"
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
