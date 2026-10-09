"""Canonical SAE pins are immutable, typed, and separate from the base-model mapping."""

from __future__ import annotations

import copy
import tomllib
import pytest

from importlib import resources
from types import MappingProxyType

from fastplms.registry import (
    RegistryError, SparseAutoencoderSpec, _load_manifest_bytes,
    _parse_sparse_autoencoders, load_model_registry,
)


def manifest_bytes() -> bytes:
    return resources.files("fastplms").joinpath("models.toml").read_bytes()


def rows() -> list[dict[str, object]]:
    return tomllib.loads(manifest_bytes().decode())["sparse_autoencoders"]


def test_canonical_sae_registry_preserves_model_inventory():
    registry = load_model_registry()
    assert len(registry) == 31
    assert isinstance(registry.sparse_autoencoders, MappingProxyType)
    expected = (("esmc_small", 23, 960), ("esmc_large", 27, 1152), ("esmc_6b", 60, 2560))
    for base, layer, width in expected:
        spec = registry.sae_for_base(base, layer=layer, k=64, codebook_dim=16384)
        assert isinstance(spec, SparseAutoencoderSpec)
        assert spec.input_width == width
        assert spec.input_kind == "hidden_state"
        assert spec.id not in registry
        assert len(spec.checkpoint.revision) == 40
        assert set(spec.checkpoint.file_map) == {"config.json", f"layer_{layer}.safetensors"}
        assert all(item.algorithm == "sha256" for item in spec.checkpoint.files)


def test_historical_manifests_remain_readable_without_sae_entries():
    original = manifest_bytes()
    historical = original.split(b"[[sparse_autoencoders]]", 1)[0]
    historical += b"[[attention_kernels]]" + original.split(b"[[attention_kernels]]", 1)[1]
    registry = _load_manifest_bytes(historical)
    assert len(registry) == 31 and not registry.sparse_autoencoders
    assert dict(registry) == dict(load_model_registry())
    with pytest.raises(RegistryError, match="Expected one pinned SAE"):
        registry.sae_for_base("esmc_small", layer=23, k=64, codebook_dim=16384)


@pytest.mark.parametrize(("field", "value"), [
    ("base_model", "missing"), ("base_model", "esm2_8m"),
    ("layer", True), ("layer", -1), ("input_width", 0), ("input_width", "960"),
    ("k", 0), ("k", 16385), ("codebook_dim", 0),
    ("input_kind", "residual_update"), ("checkpoint_revision", "main"),
    ("checkpoint_repo", "not-a-repository"), ("extra", "unsupported"),
])
def test_invalid_sae_records_fail(field, value):
    raw = rows()
    raw[0][field] = value
    with pytest.raises(RegistryError):
        _parse_sparse_autoencoders(raw, load_model_registry())


@pytest.mark.parametrize("mutation", ["id", "selection", "missing_config", "missing_weights", "wrong_layer", "weak_digest", "extra_file"])
def test_ambiguous_or_incomplete_sae_identity_fails(mutation):
    raw = rows()
    if mutation in {"id", "selection"}:
        extra = copy.deepcopy(raw[0])
        if mutation == "selection":
            extra["id"] = "another_selection"
        raw.append(extra)
    elif mutation == "missing_config":
        raw[0]["checkpoint_files"] = raw[0]["checkpoint_files"][1:]
    elif mutation == "missing_weights":
        raw[0]["checkpoint_files"] = raw[0]["checkpoint_files"][:1]
    elif mutation == "wrong_layer":
        raw[0]["layer"] = 22
    elif mutation == "weak_digest":
        raw[0]["checkpoint_files"][0] = "config.json=git-sha1:" + "a" * 40
    else:
        raw[0]["checkpoint_files"].append("model.py=sha256:" + "a" * 64)
    with pytest.raises(RegistryError):
        _parse_sparse_autoencoders(raw, load_model_registry())


def test_unregistered_selection_does_not_fall_back_to_main():
    with pytest.raises(RegistryError, match="found 0"):
        load_model_registry().sae_for_base("esmc_small", layer=23, k=64, codebook_dim=8192)
