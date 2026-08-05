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

_ANKH_OFFICIAL_FILES = {
    "ankh_base": {
        "config.json": "git-sha1:abd44a36b5469e9a7cb019e4059b5ac1392d8422",
        "pytorch_model.bin": (
            "sha256:9b2a886374f0ff4a893f4e7a989deed76bb2458c8998bd5202ea8e97d92ddcc3"
        ),
        "special_tokens_map.json": "git-sha1:55b145827029ae9672e50d4bb368540daacce791",
        "tokenizer.json": "git-sha1:212c5ef08819fa2463c6289ba4ef7db30e715c0a",
        "tokenizer_config.json": "git-sha1:a8a872ae3441e7cc85ce19210dff1e4c5d2d7bd0",
    },
    "ankh_large": {
        "config.json": "git-sha1:1abf33e52ee3d6be67d780ec57d32ac2b27b5306",
        "pytorch_model.bin": (
            "sha256:517b6e8b279dedcb477af240b35c46bd6eb3307723eb281e60d4b2c8a87b889b"
        ),
        "special_tokens_map.json": "git-sha1:55b145827029ae9672e50d4bb368540daacce791",
        "tokenizer.json": "git-sha1:212c5ef08819fa2463c6289ba4ef7db30e715c0a",
        "tokenizer_config.json": "git-sha1:d7fe02ba6f2b18d9ccfa19ac129c9fdc9ec24d09",
    },
    "ankh2_large": {
        "config.json": "git-sha1:9286bed4ecbc4f7113024919d16ec9719b0c0748",
        "generation_config.json": (
            "git-sha1:91f792e452403d46e170e206f9e50be5ddef9b9a"
        ),
        "pytorch_model.bin": (
            "sha256:2df583f28f111276ee22a7b76007f4297e9a69766d60bccd9c8d7169c06ac606"
        ),
        "special_tokens_map.json": "git-sha1:55b145827029ae9672e50d4bb368540daacce791",
        "tokenizer.json": "git-sha1:212c5ef08819fa2463c6289ba4ef7db30e715c0a",
        "tokenizer_config.json": "git-sha1:854e5db75dae8b1e9dd39c5bae80dae5508b3e25",
    },
    "ankh3_large": {
        "config.json": "git-sha1:f5278f77d158cdd8a173df888e3ed365e84a80a3",
        "generation_config.json": (
            "git-sha1:5767cc0cacebfd06884eb27ae1c796d3ca829fd2"
        ),
        "pytorch_model.bin": (
            "sha256:26321a345e07a25b21c6c41b651c4db91b420892e52c0dcbc55bd7a8f510f95b"
        ),
        "special_tokens_map.json": "git-sha1:d596919b7fa2a197edd441ec3ec4685ecacd2de4",
        "spiece.model": (
            "sha256:f2b5e1bbd110b71ca9b2878e1fcd3265610076ecc97bd696e8a745c9bacc54e0"
        ),
        "tokenizer.json": "git-sha1:90f0c94b43c81496b3ca81e3ec1c092ef2dd7fca",
        "tokenizer_config.json": "git-sha1:0e699eebfa778698473b4faf1e66ef363b93fb21",
    },
    "ankh3_xl": {
        "config.json": "git-sha1:f8997040e8913df75fd2eebe71a2a8eb750ed0d0",
        "generation_config.json": (
            "git-sha1:91f792e452403d46e170e206f9e50be5ddef9b9a"
        ),
        "pytorch_model-00001-of-00003.bin": (
            "sha256:2c9793cbee16697cd4149debe07d3a27143e280f6e970fa46042aae820fea981"
        ),
        "pytorch_model-00002-of-00003.bin": (
            "sha256:31c5a860e414513c829ae52affb0970d7cef2c0545df2d6e1338b6806ab7174b"
        ),
        "pytorch_model-00003-of-00003.bin": (
            "sha256:055a853bdd3623db95a637935aa299427e837cd8ea69fc04708b0262508bec75"
        ),
        "special_tokens_map.json": "git-sha1:d596919b7fa2a197edd441ec3ec4685ecacd2de4",
        "spiece.model": (
            "sha256:f2b5e1bbd110b71ca9b2878e1fcd3265610076ecc97bd696e8a745c9bacc54e0"
        ),
        "tokenizer.json": "git-sha1:90f0c94b43c81496b3ca81e3ec1c092ef2dd7fca",
        "tokenizer_config.json": "git-sha1:0e699eebfa778698473b4faf1e66ef363b93fb21",
    },
}


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
    assert sum(len(auto_map) for auto_map in family_maps.values()) == 45
    for family_id, auto_map in sorted(family_maps.items()):
        for auto_class, symbol_path in sorted(auto_map.items()):
            symbol = _load_symbol(symbol_path)
            assert isinstance(symbol, type), (family_id, auto_class, symbol_path)

    for model_id, spec in sorted(registry.items()):
        for auto_class, symbol_path in sorted(spec.auto_map.items()):
            symbol = _load_symbol(symbol_path)
            assert isinstance(symbol, type), (model_id, auto_class, symbol_path)


def test_ankh_official_asset_inventory_is_exact_and_complete() -> None:
    """Pin every official runtime asset while excluding the obsolete PyTorch index."""

    registry = get_model_registry()
    for model_id, expected in _ANKH_OFFICIAL_FILES.items():
        source = registry[model_id].official
        assert {path: item.encoded for path, item in source.file_map.items()} == expected

    for model_id in ("ankh_base", "ankh_large"):
        assert "generation_config.json" not in registry[model_id].official.file_map
        assert "spiece.model" not in registry[model_id].official.file_map
    assert "generation_config.json" in registry["ankh2_large"].official.file_map
    for model_id in ("ankh3_large", "ankh3_xl"):
        assert "generation_config.json" in registry[model_id].official.file_map
        assert "spiece.model" in registry[model_id].official.file_map

    xl = registry["ankh3_xl"]
    assert "pytorch_model.bin.index.json" not in xl.official.file_map
    assert "official PyTorch shard index is deliberately excluded" in xl.notes
