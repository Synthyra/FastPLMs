"""Release checks for the E1 agreement's runtime attribution."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    inherited_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_path, inherited_path)) if inherited_path else source_path
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_importing_e1_does_not_display_runtime_attribution() -> None:
    result = _run_isolated("import fastplms.models.e1.modeling_e1")
    assert result.returncode == 0, result.stderr
    assert "Profluent-E1" not in result.stdout
    assert "Profluent-E1" not in result.stderr


def test_constructing_public_e1_model_displays_attribution_once() -> None:
    result = _run_isolated(
        """
from fastplms.models.e1.modeling_e1 import E1Config, E1Model

config = E1Config(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_num_sequences=8,
    max_num_positions_within_seq=64,
    max_num_positions_global=256,
    dtype="float32",
)
E1Model(config)
"""
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert output.count("Profluent-E1") == 1, output
