"""Fail closed when CI references a repository input that is not present."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = ROOT / ".github" / "workflows"
BASELINE_ARGUMENT = re.compile(r"--baseline\s+([A-Za-z0-9_./-]+)")
ACTION_REFERENCE = re.compile(
    r"^\s*(?:-\s*)?uses:\s*([^@\s]+)@([^\s#]+)",
    flags=re.MULTILINE,
)
FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


def test_workflow_declared_baselines_are_present_portable_files() -> None:
    declared: set[str] = set()
    for workflow in sorted(WORKFLOW_ROOT.glob("*.yml")):
        declared.update(BASELINE_ARGUMENT.findall(workflow.read_text(encoding="utf-8")))

    for relative_name in sorted(declared):
        relative = PurePosixPath(relative_name)
        assert relative.as_posix() == relative_name
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        path = ROOT.joinpath(*relative.parts)
        assert path.is_file(), f"Workflow-declared baseline is absent: {relative_name}"
        assert path.stat().st_size > 0, f"Workflow-declared baseline is empty: {relative_name}"


def test_remote_workflow_actions_are_immutable_and_consistent() -> None:
    revisions_by_action: dict[str, set[str]] = {}
    for workflow in sorted((*WORKFLOW_ROOT.glob("*.yml"), *WORKFLOW_ROOT.glob("*.yaml"))):
        for action, revision in ACTION_REFERENCE.findall(
            workflow.read_text(encoding="utf-8")
        ):
            if action.startswith("./"):
                continue
            revisions_by_action.setdefault(action, set()).add(revision)

    assert revisions_by_action, "No remote workflow actions were discovered."
    for action, revisions in sorted(revisions_by_action.items()):
        assert len(revisions) == 1, (
            f"{action} uses inconsistent revisions: {sorted(revisions)}"
        )
        revision = next(iter(revisions))
        assert FULL_COMMIT.fullmatch(revision), (
            f"{action} must be pinned to one immutable 40-character commit."
        )
