"""Shared runtime provenance gate for Biohub-backed official adapters."""

from __future__ import annotations

import os
from pathlib import Path

from tools.remote.biohub_reference_environment import (
    capture_biohub_reference_environment,
)
from tools.remote.reference_source_attestation import (
    validate_reference_sources_evidence,
    verify_reference_source,
)

BIOHUB_ESM_REVISION = "82ee35553d39169d678f784c8d3f8712ffd7d2c4"
BIOHUB_ESM_TREE_SHA256 = (
    "c5489f1fc58de200978803de2c38e1a78f769cb183a2ee90be833f0f4a0212e8"
)
BIOHUB_TRANSFORMERS_REVISION = "3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf"
BIOHUB_TRANSFORMERS_TREE_SHA256 = (
    "28b910cc18b821870db2fb6d1c50376c2d14287ae18485080699e03fa4ba4f43"
)
BIOHUB_REFERENCE_SOURCE_NAMES = ("biohub-esm", "biohub-transformers")


def _reference_source(
    *,
    name: str,
    environment_prefix: str,
    expected_revision: str,
    expected_tree_sha256: str,
) -> dict[str, object]:
    """Verify one exact Biohub source tree and its runtime import origin."""

    configured_revision = os.environ.get(f"{environment_prefix}_REVISION")
    if configured_revision != expected_revision:
        raise RuntimeError(
            f"Biohub reference container does not declare the pinned {name} revision: "
            f"expected {expected_revision}, received {configured_revision!r}."
        )
    required_environment = (
        f"{environment_prefix}_SOURCE",
        f"{environment_prefix}_ATTESTATION",
        f"{environment_prefix}_CONTRACT",
    )
    try:
        source_root, attestation, contract = (
            Path(os.environ[name]) for name in required_environment
        )
    except KeyError as error:
        raise RuntimeError(
            "Biohub reference source attestation environment is incomplete."
        ) from error
    evidence = verify_reference_source(
        source_root,
        attestation,
        contract,
        expected_revision=expected_revision,
    )
    if evidence["tree_sha256"] != expected_tree_sha256:
        raise RuntimeError(f"{name} source digest differs from the adapter pin.")
    return evidence


def reference_sources() -> dict[str, dict[str, object]]:
    """Verify both source trees that together define the Biohub oracle."""

    evidence = {
        "biohub-esm": _reference_source(
            name="Biohub ESM",
            environment_prefix="FASTPLMS_BIOHUB_ESM",
            expected_revision=BIOHUB_ESM_REVISION,
            expected_tree_sha256=BIOHUB_ESM_TREE_SHA256,
        ),
        "biohub-transformers": _reference_source(
            name="Biohub Transformers",
            environment_prefix="FASTPLMS_BIOHUB_TRANSFORMERS",
            expected_revision=BIOHUB_TRANSFORMERS_REVISION,
            expected_tree_sha256=BIOHUB_TRANSFORMERS_TREE_SHA256,
        ),
    }
    return validate_reference_sources_evidence(
        evidence,
        required_sources=BIOHUB_REFERENCE_SOURCE_NAMES,
    )


def reference_environment() -> dict[str, object]:
    """Verify the exact GH200 lock, installed inventory, and image identities."""

    required_environment = (
        "FASTPLMS_BIOHUB_LOCK_ROOT",
        "FASTPLMS_BIOHUB_LOCK_CONTRACT",
        "FASTPLMS_REFERENCE_CONTAINER_IDENTITIES",
        "FASTPLMS_REFERENCE_CONTAINER_TARGET",
    )
    try:
        lock_root, contract, container_identities = (
            Path(os.environ[name]) for name in required_environment[:3]
        )
        reference_target = os.environ[required_environment[3]]
    except KeyError as error:
        raise RuntimeError(
            "Biohub reference environment identity is incomplete."
        ) from error
    return capture_biohub_reference_environment(
        lock_root,
        contract,
        container_identities,
        reference_target=reference_target,
    )


__all__ = [
    "BIOHUB_ESM_REVISION",
    "BIOHUB_ESM_TREE_SHA256",
    "BIOHUB_REFERENCE_SOURCE_NAMES",
    "BIOHUB_TRANSFORMERS_REVISION",
    "BIOHUB_TRANSFORMERS_TREE_SHA256",
    "reference_environment",
    "reference_sources",
]
