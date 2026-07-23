"""Fail closed when CI references a repository input that is not present."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = ROOT / ".github" / "workflows"
BASELINE_ARGUMENT = re.compile(r"--baseline\s+([A-Za-z0-9_./-]+)")


def test_workflow_declared_baselines_are_present_portable_files() -> None:
    declared: set[str] = set()
    for workflow in sorted(WORKFLOW_ROOT.glob("*.yml")):
        declared.update(BASELINE_ARGUMENT.findall(workflow.read_text(encoding="utf-8")))

    assert declared, "No workflow-declared baseline inputs were discovered."
    for relative_name in sorted(declared):
        relative = PurePosixPath(relative_name)
        assert relative.as_posix() == relative_name
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        path = ROOT.joinpath(*relative.parts)
        assert path.is_file(), f"Workflow-declared baseline is absent: {relative_name}"
        assert path.stat().st_size > 0, f"Workflow-declared baseline is empty: {relative_name}"
