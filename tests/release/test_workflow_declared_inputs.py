"""Keep repository validation independent of hosted GitHub Actions."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_repository_has_no_github_actions_workflows() -> None:
    workflow_root = ROOT / ".github" / "workflows"
    workflows = (
        ()
        if not workflow_root.exists()
        else tuple((*workflow_root.glob("*.yml"), *workflow_root.glob("*.yaml")))
    )

    assert workflows == ()
    assert not (ROOT / ".github" / "dependabot.yml").exists()
