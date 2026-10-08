"""Stand-ins for the checks that bind a benchmark artifact to its runtime snapshot, for tests that fabricate artifacts."""

from __future__ import annotations

import benchmarks.suite as benchmark_suite
import pytest


def stub_artifact_validation(
    monkeypatch: pytest.MonkeyPatch,
    runtime_revision: str,
    source_sha256: str,
) -> None:
    """Accept every built artifact and report ``runtime_revision`` and ``source_sha256`` as the frozen runtime identity."""

    monkeypatch.setattr(benchmark_suite, "_validate_built_artifact", lambda *_args: None)
    monkeypatch.setattr(
        benchmark_suite,
        "_frozen_runtime_identity",
        lambda *_args: (runtime_revision, source_sha256),
    )
