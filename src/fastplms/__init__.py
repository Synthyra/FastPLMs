"""FastPLMs public package interface.

The module uses lazy exports so importing :mod:`fastplms` does not initialize
Torch, download checkpoints, construct tokenizers, or compile kernels.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.0.0"

_LAZY_EXPORTS = {
    "CheckpointSource": ("fastplms.registry", "CheckpointSource"),
    "EmbeddingInput": ("fastplms.embeddings", "EmbeddingInput"),
    "EmbeddingRecord": ("fastplms.embeddings", "EmbeddingRecord"),
    "EmbeddingResult": ("fastplms.embeddings", "EmbeddingResult"),
    "FileDigest": ("fastplms.registry", "FileDigest"),
    "ModelFamily": ("fastplms.registry", "ModelFamily"),
    "ModelRegistry": ("fastplms.registry", "ModelRegistry"),
    "ModelSpec": ("fastplms.registry", "ModelSpec"),
    "OracleAsset": ("fastplms.registry", "OracleAsset"),
    "RegistryError": ("fastplms.registry", "RegistryError"),
    "RuntimeProfile": ("fastplms.runtime", "RuntimeProfile"),
    "UpstreamSource": ("fastplms.registry", "UpstreamSource"),
    "embed_dataset": ("fastplms.embeddings", "embed_dataset"),
    "get_model_registry": ("fastplms.registry", "get_model_registry"),
    "get_model_spec": ("fastplms.registry", "get_model_spec"),
    "load_model_registry": ("fastplms.registry", "load_model_registry"),
    "runtime_profile": ("fastplms.runtime", "runtime_profile"),
}

__all__ = ["__version__", *_LAZY_EXPORTS]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
