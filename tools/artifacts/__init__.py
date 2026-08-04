"""Deterministic local Hugging Face artifact construction."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "ArtifactError",
    "build_artifact",
    "build_local_artifact",
    "canonicalize_checkpoint_weights",
    "hash_file",
    "render_model_card",
    "validate_artifact",
    "validate_repository_legal_inventory",
    "validate_weight_artifact",
    "verify_checkpoint",
]

_BUILD_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name not in _BUILD_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("tools.artifacts.build"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
