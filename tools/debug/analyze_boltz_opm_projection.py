"""Compare bounded runtime variants for Boltz2's first OPM output projection."""

from __future__ import annotations

import argparse
import json
import torch
import torch.nn.functional as F
from collections.abc import Callable, Sequence
from pathlib import Path
from safetensors.torch import load_file


_PREFIX = "msa_module__layers__0__outer_product_mean__proj_o"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device for bounded projection variants.",
    )
    return parser


def _comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    # actual, expected: (...); both tensors have the same shape.
    difference = actual.float() - expected.float()  # (...)
    scale = torch.linalg.vector_norm(expected.float()).clamp_min(  # ()
        torch.finfo(torch.float32).tiny
    )
    unequal = torch.ne(actual, expected)  # (...)
    return {
        "exact": bool(torch.equal(actual, expected)),
        "unequal_values": int(unequal.sum().item()),
        "max_absolute_error": float(difference.abs().max().item()),
        "relative_l2": float((torch.linalg.vector_norm(difference) / scale).item()),
    }


def _chunked_linear(
    X: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    # X: (..., d_in); W: (d_out, d_in); bias: (d_out)
    output = torch.zeros((*X.shape[:-1], W.shape[0]), device=X.device)  # (..., d_out)
    for start in range(0, X.shape[-1], chunk_size):
        stop = min(start + chunk_size, X.shape[-1])
        # X[..., start:stop]: (..., d_chunk); W[:, start:stop].T: (d_chunk, d_out)
        output.add_(X[..., start:stop] @ W[:, start:stop].T)  # (..., d_out)
    return output + bias  # (..., d_out)


def _autocast_linear(
    X: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor,
    *,
    allow_reduced_bf16_reduction: bool,
) -> torch.Tensor:
    # X: (..., d_in); W: (d_out, d_in); bias: (d_out)
    if not X.is_cuda:
        raise RuntimeError("The BF16 reduction-policy probe requires CUDA.")
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            allow_reduced_bf16_reduction
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.linear(X, W, bias)  # (..., d_out)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


def main(argv: Sequence[str] | None = None) -> int:
    """Report isolated projection errors against the native BF16 oracle."""

    arguments = _parser().parse_args(argv)
    candidate = load_file(arguments.candidate, device="cpu")  # values: (...)
    reference = load_file(arguments.reference, device="cpu")  # values: (...)
    input_key = f"{_PREFIX}__call_000__args__0"
    output_key = f"{_PREFIX}__call_000__output"
    weight_key = f"{_PREFIX}__parameter__weight"
    bias_key = f"{_PREFIX}__parameter__bias"
    contract_equal = {
        "input": torch.equal(candidate[input_key], reference[input_key]),
        "weight": torch.equal(candidate[weight_key], reference[weight_key]),
        "bias": torch.equal(candidate[bias_key], reference[bias_key]),
    }
    for name, equal in contract_equal.items():
        if not equal:
            raise RuntimeError(f"Projection trace {name} differs.")

    if arguments.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    # d_in and d_out are the projection input and output widths.
    X = candidate[input_key].to(device)  # (..., d_in)
    W = candidate[weight_key].to(device)  # (d_out, d_in)
    bias = candidate[bias_key].to(device)  # (d_out)
    expected = reference[output_key]  # (..., d_out)
    output_dtype = expected.dtype
    X_bf16 = X.to(torch.bfloat16).float()  # (..., d_in)
    W_bf16 = W.to(torch.bfloat16).float()  # (d_out, d_in)
    bias_bf16 = bias.to(torch.bfloat16).float()  # (d_out)

    # Every variant returns (..., d_out).
    variants: dict[str, Callable[[], torch.Tensor]] = {
        "recorded_autocast_bf16": lambda: candidate[output_key].to(device),
        "fp32_linear": lambda: F.linear(X.float(), W.float(), bias.float()).to(output_dtype),
        "bf16_operands_fp32_linear": lambda: F.linear(X_bf16, W_bf16, bias_bf16).to(output_dtype),
        "bf16_operands_fp32_chunk32": lambda: _chunked_linear(X_bf16, W_bf16, bias_bf16, 32).to(
            output_dtype
        ),
        "bf16_operands_fp32_chunk64": lambda: _chunked_linear(X_bf16, W_bf16, bias_bf16, 64).to(
            output_dtype
        ),
    }
    if device.type == "cuda":
        variants.update(
            {
                "autocast_bf16_reduced_reduction_on": lambda: _autocast_linear(
                    X,
                    W,
                    bias,
                    allow_reduced_bf16_reduction=True,
                ),
                "autocast_bf16_reduced_reduction_off": lambda: _autocast_linear(
                    X,
                    W,
                    bias,
                    allow_reduced_bf16_reduction=False,
                ),
            }
        )
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        results = {
            name: _comparison(compute().cpu(), expected) for name, compute in variants.items()
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    payload = {
        "candidate_torch": torch.__version__,
        "device": str(device),
        "localization": {
            "operation": "msa_module.layers.0.outer_product_mean.proj_o",
            "first_differing_kernel": "autocast_bf16_linear_output",
            "input_equal": contract_equal["input"],
            "weight_equal": contract_equal["weight"],
            "bias_equal": contract_equal["bias"],
            "recorded_output_equal": torch.equal(candidate[output_key], reference[output_key]),
        },
        "output_dtype": str(output_dtype),
        "variants": results,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
