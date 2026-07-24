"""Contracts for the bounded Boltz2 OPM runtime diagnostic."""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file

from tools.debug.analyze_boltz_opm_projection import _PREFIX, main


def test_opm_report_localizes_an_output_kernel_difference(tmp_path: Path) -> None:
    X = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) / 10  # (b=1, l=2, d=4)
    W = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 20  # (d_out=3, d=4)
    bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)  # (d_out=3,)
    candidate_output = F.linear(X, W, bias).to(torch.bfloat16)  # (b=1, l=2, d_out=3)
    reference_output = candidate_output.clone()  # (b=1, l=2, d_out=3)
    reference_output.reshape(-1)[0] += torch.tensor(0.125, dtype=torch.bfloat16)  # ()

    input_key = f"{_PREFIX}__call_000__args__0"
    output_key = f"{_PREFIX}__call_000__output"
    weight_key = f"{_PREFIX}__parameter__weight"
    bias_key = f"{_PREFIX}__parameter__bias"
    common = {input_key: X, weight_key: W, bias_key: bias}
    candidate_path = tmp_path / "candidate.safetensors"
    reference_path = tmp_path / "reference.safetensors"
    report_path = tmp_path / "report.json"
    save_file({**common, output_key: candidate_output}, candidate_path)
    save_file({**common, output_key: reference_output}, reference_path)

    assert (
        main(
            [
                str(candidate_path),
                str(reference_path),
                "--output",
                str(report_path),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    localization = report["localization"]
    assert localization == {
        "operation": "msa_module.layers.0.outer_product_mean.proj_o",
        "first_differing_kernel": "autocast_bf16_linear_output",
        "input_equal": True,
        "weight_equal": True,
        "bias_equal": True,
        "recorded_output_equal": False,
    }
    recorded = report["variants"]["recorded_autocast_bf16"]
    assert recorded["exact"] is False
    assert recorded["unequal_values"] == 1
