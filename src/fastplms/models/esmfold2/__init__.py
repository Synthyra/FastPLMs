"""ESMFold2 public classes, imported lazily to keep optional extras isolated."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .configuration_esmfold2 import ESMFold2Config as ESMFold2Config
    from .modeling_esmfold2 import ESMFold2Model as ESMFold2Model
    from .modeling_esmfold2 import ESMFold2Output as ESMFold2Output
    from .modeling_esmfold2_experimental import (
        ESMFold2ExperimentalModel as ESMFold2ExperimentalModel,
    )
    from .reproducibility import seed_context as seed_context

_EXPORT_MODULES = {
    "ESMFold2Config": ".configuration_esmfold2",
    "ESMFold2ExperimentalModel": ".modeling_esmfold2_experimental",
    "ESMFold2Model": ".modeling_esmfold2",
    "ESMFold2Output": ".modeling_esmfold2",
    "seed_context": ".reproducibility",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES))


__all__ = list(_EXPORT_MODULES)
