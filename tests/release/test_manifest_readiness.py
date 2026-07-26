"""Fail-closed release readiness checks for immutable source identities."""

from __future__ import annotations

import pytest

from fastplms.registry import get_model_registry


@pytest.mark.artifact
def test_release_manifest_has_no_unresolved_files() -> None:
    """Require independent hashes for every checkpoint and tokenizer asset."""

    get_model_registry().require_resolved()
