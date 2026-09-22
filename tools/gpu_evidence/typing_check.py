"""Fail on mypy errors that a working tree adds relative to its Git baseline.

The broad FastPLMs surface carries a recorded backlog of mypy errors, so an
absolute check cannot judge a change. This runs the documented mypy command on
each changed source file in both trees and reports only the errors the change
introduced. The evidence tool itself must be clean outright.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

from collections import Counter
from pathlib import Path


# The flags documented in docs/testing.md for the critical typing scope.
MYPY_FLAGS = (
    "--python-version",
    "3.12",
    "--ignore-missing-imports",
    "--explicit-package-bases",
    "--follow-imports=silent",
    "--no-error-summary",
    "--no-color-output",
    "--show-error-codes",
)
_ERROR_LINE = re.compile(r"^(?P<path>[^:\n]+):\d+(?::\d+)?: error: (?P<message>.*)$")


def changed_python_files(candidate_root: Path, baseline_root: Path, scope: str) -> list[str]:
    """Return scope-relative Python paths that are new or differ from the baseline."""
    changed = []
    for path in sorted((candidate_root / scope).rglob("*.py")):
        relative = path.relative_to(candidate_root).as_posix()
        baseline = baseline_root / relative
        # Line endings vary with the checkout, not with the change under review.
        if not baseline.is_file() or _normalized(path) != _normalized(baseline):
            changed.append(relative)
    return changed


def _normalized(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def parse_errors(output: str) -> dict[str, Counter[str]]:
    """Group mypy error messages by file, ignoring line numbers that edits shift."""
    errors: dict[str, Counter[str]] = {}
    for line in output.splitlines():
        match = _ERROR_LINE.match(line)
        if match is not None:
            path = match["path"].replace("\\", "/")
            errors.setdefault(path, Counter())[match["message"]] += 1
    return errors


def introduced_errors(
    candidate: dict[str, Counter[str]], baseline: dict[str, Counter[str]]
) -> dict[str, list[str]]:
    """Return, per file, the error messages present more often than in the baseline."""
    introduced = {}
    for path, messages in candidate.items():
        extra = messages - baseline.get(path, Counter())
        if extra:
            introduced[path] = sorted(extra.elements())
    return introduced


def _mypy(root: Path, config_file: Path, targets: list[str]) -> dict[str, Counter[str]]:
    if not targets:
        return {}
    completed = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(config_file), *MYPY_FLAGS, *targets],
        cwd=root,
        capture_output=True,
        text=True,
    )
    # mypy exits 1 when it reports errors and 2 when it could not run at all.
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"mypy failed to run in {root}:\n{completed.stdout}{completed.stderr}")
    return parse_errors(completed.stdout)


def main() -> int:
    candidate_root, baseline_root = Path("/workspace"), Path("/baseline")
    config_file = candidate_root / "mypy.ini"
    changed = changed_python_files(candidate_root, baseline_root, "src")
    shared = [path for path in changed if (baseline_root / path).is_file()]
    introduced = introduced_errors(
        _mypy(candidate_root, config_file, changed), _mypy(baseline_root, config_file, shared)
    )
    tool_errors = _mypy(candidate_root, config_file, ["tools/gpu_evidence"])
    report = {
        "baseline_revision": (baseline_root / "REVISION").read_text(encoding="utf-8").strip(),
        "changed_source_files": changed,
        "introduced_errors": introduced,
        "evidence_tool_errors": {
            path: sorted(msgs.elements()) for path, msgs in tool_errors.items()
        },
    }
    print(json.dumps(report, indent=2))
    return 1 if introduced or tool_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
