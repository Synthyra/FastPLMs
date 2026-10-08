"""Helpers the parity tests share: pinned upstream source loaded under private names, and parameter alias groups.

None of these files is copied into an oracle image, so this module may import anything the host tests import.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import torch

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path


_MISSING = object()


def namespace_package(name: str) -> types.ModuleType:
    """Return an empty package module called ``name``, so that dotted imports below it resolve."""

    package = types.ModuleType(name)
    package.__path__ = []  # type: ignore[attr-defined]
    return package


def namespace_packages(*names: str) -> dict[str, types.ModuleType]:
    """Return one empty package module per dotted name, keyed by name."""

    return {name: namespace_package(name) for name in names}


@contextmanager
def temporary_modules(modules: Mapping[str, types.ModuleType]) -> Iterator[None]:
    """Put ``modules`` into ``sys.modules`` for the block and restore the previous entries afterwards."""

    previous = {name: sys.modules.get(name, _MISSING) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module  # type: ignore[assignment]


def load_pinned_source(
    module_name: str,
    path: Path,
    aliases: Mapping[str, types.ModuleType] | None = None,
) -> types.ModuleType:
    """Execute the pinned source file ``path`` as ``module_name`` while ``aliases`` stand in for the modules it imports."""

    assert path.is_file(), f"pinned source is missing: {path}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    with temporary_modules({**(aliases or {}), module_name: module}):
        specification.loader.exec_module(module)
    return module


def alias_groups(model: torch.nn.Module) -> set[frozenset[str]]:
    """Return the sets of parameter names that share one parameter object."""

    names_by_parameter: dict[int, set[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        names_by_parameter.setdefault(id(parameter), set()).add(name)
    return {frozenset(names) for names in names_by_parameter.values() if len(names) > 1}
