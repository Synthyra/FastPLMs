"""Cheap manifest-wide coverage that never loads checkpoint weights."""

from __future__ import annotations

import importlib

from fastplms.registry import get_model_registry

_CPU_FAMILIES = {
    "ankh",
    "boltz2",
    "dplm",
    "dplm2",
    "e1",
    "esm2",
    "esm3",
    "esm_plusplus",
    "esmfold",
    "esmfold2",
}
_CPU_CHECKPOINTS = frozenset(
    {
        "ankh2_large",
        "ankh3_large",
        "ankh3_xl",
        "ankh_base",
        "ankh_large",
        "boltz2",
        "dplm2_150m",
        "dplm2_3b",
        "dplm2_650m",
        "dplm_150m",
        "dplm_3b",
        "dplm_650m",
        "e1_150m",
        "e1_300m",
        "e1_600m",
        "esm2_150m",
        "esm2_35m",
        "esm2_3b",
        "esm2_650m",
        "esm2_8m",
        "esm3_small",
        "esmc_6b",
        "esmc_large",
        "esmc_small",
        "esmfold",
        "esmfold2",
        "esmfold2_experimental_cutoff2025",
        "esmfold2_experimental_fast_cutoff2025",
        "esmfold2_fast",
    }
)


def _load_symbol(path: str) -> type:
    module_name, _, symbol_name = path.rpartition(".")
    assert module_name and symbol_name, path
    return getattr(importlib.import_module(module_name), symbol_name)


def test_every_checkpoint_and_family_is_owned_by_the_cpu_matrix() -> None:
    registry = get_model_registry()
    assert frozenset(registry) == _CPU_CHECKPOINTS
    assert {spec.family.id for spec in registry.values()} == _CPU_FAMILIES
    for spec in registry.values():
        assert len(spec.fast.revision) == 40
        assert len(spec.official.revision) == 40
        assert not spec.fast.unresolved_files
        assert not spec.official.unresolved_files


def test_every_advertised_automap_symbol_imports_without_optional_runtime_work() -> None:
    registry = get_model_registry()
    family_maps = {
        family_id: family.auto_map
        for family_id, family in registry.families.items()
    }
    assert sum(len(auto_map) for auto_map in family_maps.values()) == 37
    for family_id, auto_map in sorted(family_maps.items()):
        for auto_class, symbol_path in sorted(auto_map.items()):
            symbol = _load_symbol(symbol_path)
            assert isinstance(symbol, type), (family_id, auto_class, symbol_path)

    for model_id, spec in sorted(registry.items()):
        for auto_class, symbol_path in sorted(spec.auto_map.items()):
            symbol = _load_symbol(symbol_path)
            assert isinstance(symbol, type), (model_id, auto_class, symbol_path)
