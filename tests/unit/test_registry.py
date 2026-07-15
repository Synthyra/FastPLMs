from __future__ import annotations

import configparser
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from fastplms.registry import FileDigest, RegistryError, load_model_registry

ROOT = Path(__file__).resolve().parents[2]


def test_package_manifest_is_complete_and_typed() -> None:
    registry = load_model_registry()

    assert registry.schema_version == 1
    assert {item.path for item in registry.legal_files} == {
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    }
    assert len(registry) == 29
    assert set(registry.upstreams) == {
        "ankh",
        "biohub-esm",
        "biohub-transformers",
        "boltz",
        "dplm",
        "e1",
        "fair-esm",
        "openfold",
        "protein-ttt",
    }
    assert {source_id: source.path for source_id, source in registry.upstreams.items()} == {
        "ankh": "vendor/upstream/ankh",
        "biohub-esm": "vendor/upstream/biohub-esm",
        "biohub-transformers": "vendor/upstream/biohub-transformers",
        "boltz": "vendor/upstream/boltz",
        "dplm": "vendor/upstream/dplm",
        "e1": "vendor/upstream/e1",
        "fair-esm": "vendor/upstream/fair-esm",
        "openfold": "vendor/upstream/openfold",
        "protein-ttt": "vendor/upstream/protein-ttt",
    }
    assert registry["esm2_8m"].fast.repo_id == "Synthyra/ESM2-8M"
    assert registry["esm2_8m"].artifact_source == "fast"
    assert registry["esm2_8m"].artifact_checkpoint is registry["esm2_8m"].fast
    assert registry["esm2_8m"].is_deep_reference
    assert registry["esm2_650m"].size_category == "large"
    assert registry["esmc_small"].family.reference_adapter.endswith(".esm_plusplus")
    assert registry.families["esmfold2"].backbone_model == "esmc_6b"
    assert registry[registry.families["esmfold2"].backbone_model].fast.repo_id == (
        "Synthyra/ESMplusplus_6B"
    )
    assert registry["e1_150m"].family.conversion_provenance.startswith("Input:")
    assert registry["dplm2_150m"].family.tokenizer_class == (
        "fastplms.models.dplm2.tokenization_dplm2.DPLM2Tokenizer"
    )
    assert all(
        family.tokenizer_class is None
        for family_id, family in registry.families.items()
        if family_id != "dplm2"
    )
    assert {
        family.id: family.bf16_execution
        for family in registry.families.values()
        if family.bf16_execution == "fp32_parameters_autocast"
    } == {
        "boltz2": "fp32_parameters_autocast",
        "dplm": "fp32_parameters_autocast",
        "dplm2": "fp32_parameters_autocast",
        "esmfold": "fp32_parameters_autocast",
        "esmfold2": "fp32_parameters_autocast",
    }
    assert all(
        family.bf16_execution in {"static_parameters", "fp32_parameters_autocast"}
        for family in registry.families.values()
    )
    for model in registry.by_family("ankh"):
        assert model.artifact_source == "official"
        assert model.artifact_checkpoint is model.official
    for model in registry.by_family("dplm2"):
        assert model.artifact_source == "official"
        assert model.artifact_checkpoint is model.official
    for source in registry.upstreams.values():
        assert tuple(item.path for item in source.license_digests) == source.license_files
        assert source.distribution_files


def test_generation_contracts_are_explicit_and_exact() -> None:
    registry = load_model_registry()
    required = {"dplm_150m", "dplm_650m", "dplm_3b", "dplm2_150m", "dplm2_650m"}
    unavailable = {"dplm2_3b"}

    assert {model.id for model in registry.values() if model.generation_contract == "required"} == (
        required
    )
    assert {
        model.id
        for model in registry.values()
        if model.generation_contract == "official_unavailable"
    } == unavailable
    assert {
        model.id for model in registry.values() if model.generation_contract == "not_applicable"
    } == set(registry).difference(required, unavailable)


def test_esmfold2_ccd_runtime_asset_is_typed_and_immutable() -> None:
    registry = load_model_registry()
    assert set(registry.runtime_assets) == {"esmfold2_ccd"}
    asset = registry.runtime_assets["esmfold2_ccd"]
    assert asset.repository == "biohub/ESMFold2"
    assert asset.revision == "1ebf0e3481a5184eb6171d40615c79e384b48796"
    assert asset.path == "ccd.pkl"
    assert asset.sha256 == "9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5"
    assert asset.size == 417306584
    assert asset.consumer_family == "esmfold2"
    assert asset.trust_kind == "hash_pinned_pickle"


def test_attention_kernel_revisions_are_typed_and_immutable() -> None:
    registry = load_model_registry()
    assert {
        implementation: kernel.revision
        for implementation, kernel in registry.attention_kernels.items()
    } == {
        "flash_attention_2": "db6b51744f0cd7061386442c09df890fc6d9f47e",
        "flash_attention_3": "43f0bd269777115d94ff826e0d113ce9c1c9087b",
    }
    assert {
        implementation: kernel.dtypes
        for implementation, kernel in registry.attention_kernels.items()
    } == {
        "flash_attention_2": ("bfloat16",),
        "flash_attention_3": ("bfloat16",),
    }
    assert registry.supported_attention_dtypes("esm2", "sdpa") == (
        "float32",
        "bfloat16",
    )
    assert registry.supported_attention_dtypes("esm2", "flash_attention_3") == (
        "bfloat16",
    )


def test_attention_kernel_lock_matches_manifest_and_h100_variants() -> None:
    registry = load_model_registry()
    entries = json.loads((ROOT / "kernels.lock").read_text(encoding="utf-8"))
    locked = {entry["repo_id"]: entry for entry in entries}
    assert set(locked) == {
        "kernels-community/flash-attn2",
        "kernels-community/flash-attn3",
    }
    for kernel in registry.attention_kernels.values():
        assert locked[kernel.repository]["sha"] == kernel.revision
        assert locked[kernel.repository]["variants"]

    expected_h100 = {
        "kernels-community/flash-attn2": (
            "torch213-cxx11-cu130-x86_64-linux",
            "sha256-238cdad1945962331ad685a07119bb9e893ed976f11ecbf257e03d36682f95e4",
        ),
        "kernels-community/flash-attn3": (
            "torch-stable-abi29-cu130-x86_64-linux",
            "sha256-8dc3c4645b8ed2c5ce27873f8c6deb4ecf60060b5f08f538389b3d79e8842a2f",
        ),
    }
    for repository, (variant, digest) in expected_h100.items():
        variant_lock = locked[repository]["variants"][variant]
        assert variant_lock == {"hash": digest, "hash_type": "git_lfs_concat"}

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["kernels"]["dependencies"] == {
        "kernels-community/flash-attn2": 2,
        "kernels-community/flash-attn3": 1,
    }


def test_manifest_rejects_mutable_attention_kernel_revision(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = manifest.replace(
        'revision = "db6b51744f0cd7061386442c09df890fc6d9f47e"',
        'revision = "main"',
        1,
    )
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(
        RegistryError,
        match=r"attention_kernels\[0\]\.revision must be an immutable",
    ):
        load_model_registry(path)


@pytest.mark.parametrize(
    ("anchor", "unknown_field", "context"),
    (
        ("schema_version = 1\n", 'unknown_root = "value"\n', "manifest"),
        ("[[attention_kernels]]\n", 'unknown_kernel = "value"\n', "attention_kernels[0]"),
        ("[[runtime_assets]]\n", 'unknown_asset = "value"\n', "runtime_assets[0]"),
        ("[[upstreams]]\n", 'unknown_upstream = "value"\n', "upstreams[0]"),
        ("[families.esm2]\n", 'unknown_family = "value"\n', "families.esm2"),
        ("[[models]]\n", 'unknown_model = "value"\n', "models[0]"),
    ),
)
def test_manifest_rejects_unknown_table_fields(
    tmp_path: Path,
    anchor: str,
    unknown_field: str,
    context: str,
) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = manifest.replace(anchor, anchor + unknown_field, 1)
    assert invalid != manifest
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RegistryError, match=rf"{re.escape(context)} contains unknown fields"):
        load_model_registry(path)


@pytest.mark.parametrize("path_value", ("official/ankh", "vendor/upstream/ankh/nested", "../ankh"))
def test_manifest_rejects_noncanonical_upstream_paths(
    tmp_path: Path,
    path_value: str,
) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = manifest.replace(
        'path = "vendor/upstream/ankh"',
        f'path = "{path_value}"',
        1,
    )
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RegistryError, match="normalized directory directly under"):
        load_model_registry(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ('extra = "core"', 'extra = "gpu"', r"families\.esm2\.extra must be one of"),
        (
            'test_tiers = ["check", "compliance", "feature", "artifact", "benchmark"]',
            'test_tiers = ["check", "compliance", "feature", "artifact", "nightly"]',
            r"families\.esm2\.test_tiers contains unsupported tiers",
        ),
        (
            'vram_tier = "sequence"',
            'vram_tier = "host-specific"',
            r"families\.esm2\.vram_tier must be one of",
        ),
        (
            'bf16_execution = "static_parameters"',
            'bf16_execution = "implicit"',
            r"families\.esm2\.bf16_execution must be one of",
        ),
        (
            'reference_container = "reference-esm2"',
            'reference_container = "../reference-esm2"',
            r"families\.esm2\.reference_container must be a portable",
        ),
        (
            'reference_adapter = "tests.parity.support.reference_adapters.esm2"',
            'reference_adapter = "fastplms.reference_adapters.esm2"',
            r"families\.esm2\.reference_adapter must name one module",
        ),
        (
            'documentation = "docs/models.md#esm2"',
            'documentation = "../models.md#esm2"',
            r"families\.esm2\.documentation must reference a normalized Markdown file",
        ),
        (
            'documentation = "docs/models.md#esm2"',
            'documentation = "docs/models.md#ESM2"',
            r"families\.esm2\.documentation has an invalid heading fragment",
        ),
    ),
)
def test_manifest_rejects_invalid_family_enums_and_paths(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = manifest.replace(old, new, 1)
    assert invalid != manifest
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RegistryError, match=message):
        load_model_registry(path)


def test_manifest_rejects_unknown_backbone_model(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = manifest.replace(
        'backbone_model = "esmc_6b"',
        'backbone_model = "missing_esmc"',
        1,
    )
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RegistryError, match="references unknown backbone model 'missing_esmc'"):
        load_model_registry(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            'repository = "biohub/ESMFold2"',
            'repository = "not-a-repository"',
            r"runtime_assets\[0\]\.repository must be a Hugging Face repository ID",
        ),
        (
            'revision = "1ebf0e3481a5184eb6171d40615c79e384b48796"',
            'revision = "main"',
            r"runtime_assets\[0\]\.revision must be an immutable",
        ),
        ('path = "ccd.pkl"', 'path = "../ccd.pkl"', "Runtime asset path is not portable"),
        (
            'sha256 = "9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5"',
            'sha256 = "unresolved"',
            "Invalid runtime asset SHA-256",
        ),
        ("size = 417306584", "size = 0", r"runtime_assets\[0\]\.size must be a positive"),
        (
            'consumer_family = "esmfold2"',
            'consumer_family = "unknown"',
            r"runtime_assets\[0\]\.consumer_family references unknown family",
        ),
        (
            'trust_kind = "hash_pinned_pickle"',
            'trust_kind = "pickle"',
            r"runtime_assets\[0\]\.trust_kind must be one of",
        ),
        (
            'path = "ccd.pkl"',
            'path = "ccd.bin"',
            r"runtime_assets\[0\]\.path must end in '.pkl'",
        ),
    ),
)
def test_manifest_rejects_invalid_runtime_asset_fields(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = manifest.replace(old, new, 1)
    assert invalid != manifest
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RegistryError, match=message):
        load_model_registry(path)


def test_manifest_accepts_an_alternative_hash_pinned_runtime_asset(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    alternative = manifest.replace(
        "9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5",
        "aff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5",
        1,
    )
    path = tmp_path / "models.toml"
    path.write_text(alternative, encoding="utf-8")

    registry = load_model_registry(path)
    assert registry.runtime_assets["esmfold2_ccd"].sha256.startswith("aff44b")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            'generation_contract = "not_applicable"',
            'generation_contract = "best_effort"',
            r"models\[0\]\.generation_contract must be one of",
        ),
        (
            'generation_contract = "not_applicable"\n',
            "",
            r"models\[0\]\.generation_contract must be a non-empty string",
        ),
    ),
)
def test_manifest_rejects_invalid_generation_contracts(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = manifest.replace(old, new, 1)
    assert invalid != manifest
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RegistryError, match=message):
        load_model_registry(path)


def test_hub_license_metadata_is_typed_and_complete() -> None:
    registry = load_model_registry()
    assert {family_id: family.hub_license for family_id, family in registry.families.items()} == {
        "ankh": "cc-by-nc-sa-4.0",
        "boltz2": "mit",
        "dplm": "apache-2.0",
        "dplm2": "apache-2.0",
        "e1": "other",
        "esm2": "mit",
        "esm3": "mit",
        "esm_plusplus": "mit",
        "esmfold": "mit",
        "esmfold2": "mit",
    }
    e1 = registry.families["e1"]
    assert dict(e1.hub_license_metadata) == {
        "license": "other",
        "license_name": "Profluent-E1 Clickthrough License Agreement",
        "license_link": (
            "https://github.com/Profluent-AI/E1/blob/"
            "bfd2620a602248499f3d2583d85a7ecddf0b6e02/LICENSE"
        ),
    }
    for family_id, family in registry.families.items():
        if family_id != "e1":
            assert dict(family.hub_license_metadata) == {"license": family.hub_license}


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ('hub_license = "mit"\n', "", "families.esm2.hub_license"),
        (
            'hub_license = "mit"\n',
            'hub_license = "proprietary"\n',
            "supported Hugging Face license identifier",
        ),
        (
            'hub_license = "mit"\n',
            'hub_license = "apache-2.0"\n',
            "must be 'mit' for checkpoint terms",
        ),
        (
            'hub_license_name = "Profluent-E1 Clickthrough License Agreement"\n',
            "",
            "must define hub_license_name and hub_license_link",
        ),
        (
            "https://github.com/Profluent-AI/E1/blob/",
            "http://github.com/Profluent-AI/E1/blob/",
            "hub_license_link must be an absolute HTTPS URL",
        ),
        (
            'hub_license = "mit"\n',
            'hub_license = "mit"\n'
            'hub_license_name = "Unexpected custom terms"\n'
            'hub_license_link = "https://example.invalid/LICENSE"\n',
            "may define hub_license_name and hub_license_link only",
        ),
        (
            'hub_license = "mit"\n',
            'hub_license = "mit"\nhub_license_nam = "misspelled"\n',
            "contains unsupported Hub license fields",
        ),
    ),
)
def test_manifest_rejects_invalid_hub_license_metadata(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = manifest.replace(old, new, 1)
    assert invalid != manifest
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RegistryError, match=message):
        load_model_registry(path)


def test_esmfold2_support_is_exactly_the_approved_four() -> None:
    registry = load_model_registry()
    assert {model.official.repo_id for model in registry.by_family("esmfold2")} == {
        "biohub/ESMFold2",
        "biohub/ESMFold2-Fast",
        "biohub/ESMFold2-Experimental-Cutoff2025",
        "biohub/ESMFold2-Experimental-Fast-Cutoff2025",
    }
    assert set(registry.by_family("esmfold2")[0].family.precisions) == {
        "auto",
        "fp32",
        "bf16",
        "fp8",
    }


def test_boltz2_is_explicitly_provisional() -> None:
    registry = load_model_registry()
    spec = registry["boltz2"]
    assert set(spec.family.test_tiers) == {"structure", "artifact", "benchmark"}
    assert "check" not in spec.family.test_tiers
    assert "compliance" not in spec.family.test_tiers
    assert "provisional in FastPLMs 1.0" in spec.notes
    assert "does not claim official inference equivalence" in spec.notes


def test_esm2_native_oracle_assets_are_hash_pinned() -> None:
    registry = load_model_registry()
    for model in registry.by_family("esm2"):
        assert set(model.oracle_asset_map) == {"weights", "contact_regression"}
        official_name = model.official.repo_id.split("/", maxsplit=1)[1]
        assert model.oracle_asset_map["weights"].path == f"models/{official_name}.pt"
        assert model.oracle_asset_map["contact_regression"].path == (
            f"regression/{official_name}-contact-regression.pt"
        )
        for asset in model.oracle_assets:
            assert asset.url == f"https://dl.fbaipublicfiles.com/fair-esm/{asset.path}"
            assert len(asset.sha256) == 64
            assert asset.size > 0

    esmfold = registry["esmfold"]
    assert set(esmfold.oracle_asset_map) == {"weights"}
    assert esmfold.oracle_asset_map["weights"].path == "models/esmfold_3B_v1.pt"
    assert esmfold.oracle_asset_map["weights"].sha256 == (
        "e9a52579027e77d2d2e0a18218e755821f395730e86624cab9413dc117f5ca62"
    )
    assert esmfold.oracle_asset_map["weights"].size == 2771653574

    for model in registry.values():
        if model.family.id not in {"esm2", "esmfold"}:
            assert not model.oracle_assets


def test_checkpoint_provenance_is_explicit_and_release_gated() -> None:
    registry = load_model_registry()
    unresolved_count = 0
    for model in registry.values():
        for checkpoint in (model.fast, model.official):
            assert len(checkpoint.revision) == 40
            assert checkpoint.files
            assert all(item.algorithm in {"git-sha1", "sha256"} for item in checkpoint.files)
            assert set(checkpoint.file_map).isdisjoint(checkpoint.unresolved_files)
            unresolved_count += len(checkpoint.unresolved_files)
        if model.family.tokenizer_mode == "tokenizer":
            assert any("tokenizer" in path or "vocab" in path for path in model.fast.file_map)

    assert unresolved_count == 0
    assert (
        registry["esm2_3b"].fast.file_map["model-00003-of-00003.safetensors"].digest
        == "a6b3a55b9e3b2e1778de34c665c3dd17bdfdf6da9d6d5c97730c57168709ccae"
    )
    registry.require_resolved("esm2_8m")
    registry.require_resolved("esm2_35m")
    registry.require_resolved()


def test_runtime_paths_cannot_include_official_sources() -> None:
    registry = load_model_registry()
    for family in registry.families.values():
        assert all(not path.startswith("vendor/") for path in family.runtime_paths)
        assert "models/__init__.py" in family.runtime_paths
    assert (ROOT / "src" / "fastplms" / "models" / "__init__.py").is_file()


def test_gitmodules_matches_manifest_paths_and_urls() -> None:
    registry = load_model_registry()
    parser = configparser.ConfigParser()
    parser.read(ROOT / ".gitmodules", encoding="utf-8")

    configured = {}
    for section in parser.sections():
        configured[parser[section]["path"]] = parser[section]["url"]
    assert configured == {source.path: source.url for source in registry.upstreams.values()}


def test_manifest_rejects_an_alternative_pinned_esmfold2_repository(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    alternative = manifest.replace(
        'official_repo = "biohub/ESMFold2-Experimental-Cutoff2025"',
        'official_repo = "biohub/ESMFold2-Experimental"',
        1,
    )
    path = tmp_path / "models.toml"
    path.write_text(alternative, encoding="utf-8")

    with pytest.raises(RegistryError, match="exactly the four approved"):
        load_model_registry(path)


def test_manifest_rejects_a_fifth_esmfold2_checkpoint(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = manifest.replace(
        'id = "esmfold"\nfamily = "esmfold"',
        'id = "esmfold_legacy_fifth"\nfamily = "esmfold2"',
        1,
    )
    assert invalid != manifest
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RegistryError, match="exactly the four approved"):
        load_model_registry(path)


def test_manifest_rejects_missing_e1_legal_notice(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = re.sub(
        r'^  "MODIFICATIONS\.md=sha256:[0-9a-f]{64}",\n',
        "",
        manifest,
        count=1,
        flags=re.MULTILINE,
    )
    assert invalid != manifest
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RegistryError, match="missing E1 legal files"):
        load_model_registry(path)


def test_manifest_rejects_missing_conversion_record(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = re.sub(
        r"^conversion_provenance = .*\n",
        "",
        manifest,
        count=1,
        flags=re.MULTILINE,
    )
    assert invalid != manifest
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RegistryError, match="conversion_provenance"):
        load_model_registry(path)


def test_manifest_rejects_unpinned_oracle_asset(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    invalid = manifest.replace(
        "46f002a9870c9bdecd0ea887acb1f9a38a6b561e8f8bf8a6990b679b9d31b928",
        "unresolved",
        1,
    )
    path = tmp_path / "models.toml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RegistryError, match="Invalid oracle asset SHA-256"):
        load_model_registry(path)


def test_manifest_parses_optional_hash_pinned_official_golden(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    declaration = (
        'official_golden = { metadata = "tests/goldens/esm2_8m.json=sha256:'
        + "a" * 64
        + '", tensors = "tests/goldens/esm2_8m.safetensors=sha256:'
        + "b" * 64
        + '" }\n'
    )
    modified = manifest.replace('id = "esm2_8m"\n', 'id = "esm2_8m"\n' + declaration, 1)
    path = tmp_path / "models.toml"
    path.write_text(modified, encoding="utf-8")

    golden = load_model_registry(path)["esm2_8m"].official_golden
    assert golden is not None
    assert golden.metadata.path == "tests/goldens/esm2_8m.json"
    assert golden.metadata.digest == "a" * 64
    assert golden.tensors.path == "tests/goldens/esm2_8m.safetensors"
    assert golden.tensors.digest == "b" * 64


def test_manifest_rejects_an_unsafe_official_golden_path(tmp_path: Path) -> None:
    manifest = (ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8")
    declaration = (
        'official_golden = { metadata = "../esm2_8m.json=sha256:'
        + "a" * 64
        + '", tensors = "tests/goldens/esm2_8m.safetensors=sha256:'
        + "b" * 64
        + '" }\n'
    )
    modified = manifest.replace('id = "esm2_8m"\n', 'id = "esm2_8m"\n' + declaration, 1)
    path = tmp_path / "models.toml"
    path.write_text(modified, encoding="utf-8")

    with pytest.raises(RegistryError, match="Checkpoint file path is not portable"):
        load_model_registry(path)


@pytest.mark.parametrize(
    "value",
    [
        "../model.safetensors=sha256:" + "a" * 64,
        "model.safetensors=md5:" + "a" * 32,
        "model.safetensors=sha256:short",
    ],
)
def test_file_digest_rejects_unsafe_or_unverifiable_values(value: str) -> None:
    with pytest.raises(RegistryError):
        FileDigest.parse(value)


def test_top_level_import_does_not_import_torch() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, fastplms; assert fastplms.__version__ == '1.0.0'; "
            "assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
