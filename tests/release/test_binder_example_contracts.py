"""Release contracts for the binder example's validation and environment."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "binder_design_fastplms.py"
GUIDE = ROOT / "docs" / "binder_design.md"


def test_binder_example_has_no_optimized_away_or_private_validation() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EXAMPLE))
    runtime_asserts = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]

    assert not runtime_asserts, "binder validation must survive python -O"
    assert "abnumber.common" not in source
    assert "_anarci_align" not in source
    assert "Chain.multiple_domains" in source
    assert "use_anarcii=True" in source


def test_binder_example_uses_project_python_and_dependency_metadata() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    guide = GUIDE.read_text(encoding="utf-8")

    assert "# /// script" not in source
    assert "requires-python" not in source
    assert project["requires-python"] == ">=3.11,<3.15"
    assert "Python 3.11-3.14" in guide
    assert "no standalone PEP 723 dependency block" in guide
    for fragment in (
        "uv run",
        "--extra structure",
        "--with abnumber",
        "--with pandas",
        "--with pyarrow",
        "python examples/binder_design_fastplms.py",
    ):
        assert fragment in guide


def test_binder_validation_survives_python_optimized_mode() -> None:
    program = textwrap.dedent(
        """
        import torch
        from examples import binder_design_fastplms as binder

        checks = (
            lambda: binder.build_initial_soft_sequence_logits("A?", batch_size=1),
            lambda: binder.compute_distogram_iptm_proxy(
                torch.zeros(3, 3, 128),
                target_length=2,
                binder_sequence="AA",
                is_antibody=False,
            ),
            lambda: binder._binder_sequence_from_designed_sequence("missing-separator"),
        )
        for check in checks:
            try:
                check()
            except ValueError:
                continue
            raise SystemExit("validation disappeared under python -O")
        """
    )
    environment = os.environ.copy()
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))),
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", program],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_binder_output_contract_is_documented_fail_closed() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")

    assert "_require_fresh_output_directory(args.output_dir)" in source
    assert source.index("_require_fresh_output_directory(args.output_dir)") < source.index(
        "runner.load("
    )
    assert source.index("_write_official_selection_table(") < source.rindex("_write_run_manifest(")
    assert "must not already exist, including as an empty directory" in guide
    assert "written atomically and last" in guide
    assert "treat the\ndirectory as an incomplete run" in guide
