"""Run one minimal Transformer Engine FP8 linear layer on CUDA."""

from __future__ import annotations

import json
import torch
from importlib.metadata import version


def main() -> None:
    import transformer_engine.pytorch as te

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Transformer Engine FP8 probe.")
    torch.manual_seed(17)
    device = torch.device("cuda")
    X = torch.randn(32, 16, device=device, dtype=torch.bfloat16)  # (n=32, d=16)
    linear = te.Linear(
        16,
        32,
        bias=False,
        params_dtype=torch.bfloat16,
        device=device,
    ).eval()
    autocast = getattr(te, "autocast", None)
    if autocast is None:
        autocast = te.fp8_autocast
    with torch.inference_mode(), autocast(enabled=True):
        Z = linear(X)  # (n=32, d_out=32)
    torch.cuda.synchronize()
    if Z.shape != (32, 32) or not torch.isfinite(Z).all():
        raise RuntimeError("Transformer Engine FP8 linear output is invalid.")
    print(
        json.dumps(
            {
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(device),
                "input_dtype": str(X.dtype),
                "output_dtype": str(Z.dtype),
                "output_shape": list(Z.shape),
                "torch": torch.__version__,
                "transformer_engine": version("transformer-engine"),
                "transformer_engine_torch": version("transformer-engine-torch"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
