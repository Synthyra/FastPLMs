"""The immutable records the manifest produces, and the reversible Torch runtime profile."""

from __future__ import annotations

import re
import pytest
import torch

import fastplms

from dataclasses import FrozenInstanceError, replace
from pathlib import PurePosixPath
from types import MappingProxyType

from fastplms.registry import (
    AttentionKernelSpec,
    CheckpointSource,
    FileDigest,
    GoldenArtifact,
    ModelRegistry,
    OracleAsset,
    RegistryError,
    RuntimeAsset,
    get_model_registry,
)
from fastplms.runtime import RuntimeProfile, runtime_profile


SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")


class TestAttentionKernelSpec:
    def test_every_kernel_is_an_immutable_pin_for_its_own_backend(self) -> None:
        kernels = get_model_registry().attention_kernels
        assert kernels, "the manifest pins at least one FlashAttention kernel"
        for implementation, kernel in kernels.items():
            assert isinstance(kernel, AttentionKernelSpec)
            assert kernel.implementation == implementation
            assert REVISION.fullmatch(kernel.revision)
            assert kernel.version >= 1
            assert kernel.dtypes and set(kernel.dtypes) <= {"float32", "bfloat16"}
            major, minor = kernel.min_cuda_capability
            assert major >= 1 and minor >= 0

    def test_a_record_cannot_be_edited(self) -> None:
        kernel = next(iter(get_model_registry().attention_kernels.values()))
        with pytest.raises(FrozenInstanceError):
            kernel.revision = "0" * 40  # type: ignore[misc]

    def test_the_default_capability_is_ampere(self) -> None:
        kernel = AttentionKernelSpec(
            implementation="flash_attention_2",
            repository="org/kernel",
            revision="a" * 40,
            version=1,
            expected_variant="variant",
            dtypes=("bfloat16",),
        )
        assert kernel.min_cuda_capability == (8, 0)

    def test_supported_dtypes_are_the_family_dtypes_the_kernel_also_supports(self) -> None:
        registry = get_model_registry()
        for family_id, family in registry.families.items():
            for implementation in family.attention:
                dtypes = registry.supported_attention_dtypes(family_id, implementation)
                kernel = registry.attention_kernels.get(implementation)
                expected = family.dtypes if kernel is None else tuple(d for d in family.dtypes if d in kernel.dtypes)
                assert dtypes == expected

    def test_an_unadvertised_backend_is_a_key_error(self) -> None:
        registry = get_model_registry()
        family_id = next(iter(registry.families))
        with pytest.raises(KeyError, match="does not advertise"):
            registry.supported_attention_dtypes(family_id, "not_a_backend")


class TestOracleAsset:
    def test_every_asset_is_a_hash_pinned_relative_file(self) -> None:
        seen = 0
        for model_id, model in get_model_registry().items():
            roles = [asset.role for asset in model.oracle_assets]
            assert len(roles) == len(set(roles)), f"{model_id} repeats an oracle role"
            for asset in model.oracle_assets:
                seen += 1
                assert isinstance(asset, OracleAsset)
                assert SHA256.fullmatch(asset.sha256)
                assert asset.size > 0
                assert asset.url.startswith("https://")
                path = PurePosixPath(asset.path)
                assert not path.is_absolute() and ".." not in path.parts
        assert seen > 0

    def test_the_asset_map_is_a_read_only_view_keyed_by_role(self) -> None:
        model = next(model for model in get_model_registry().values() if model.oracle_assets)
        mapping = model.oracle_asset_map
        assert isinstance(mapping, MappingProxyType)
        assert set(mapping) == {asset.role for asset in model.oracle_assets}
        for asset in model.oracle_assets:
            assert mapping[asset.role] is asset
        with pytest.raises(TypeError):
            mapping["new"] = model.oracle_assets[0]  # type: ignore[index]

    def test_a_model_without_oracle_assets_has_an_empty_map(self) -> None:
        model = next(model for model in get_model_registry().values() if not model.oracle_assets)
        assert dict(model.oracle_asset_map) == {}


class TestRuntimeAsset:
    def test_every_asset_is_pinned_to_a_revision_and_digest(self) -> None:
        assets = get_model_registry().runtime_assets
        assert assets, "the manifest pins at least one runtime asset"
        for asset_id, asset in assets.items():
            assert isinstance(asset, RuntimeAsset)
            assert asset.id == asset_id
            assert REVISION.fullmatch(asset.revision)
            assert SHA256.fullmatch(asset.sha256)
            assert asset.size > 0
            assert asset.trust_kind == "hash_pinned_pickle"
            assert asset.consumer_family in get_model_registry().families
            assert asset.license_expression and asset.offline_behavior


class TestGoldenArtifact:
    def test_every_artifact_names_a_registered_model_and_an_immutable_file(self) -> None:
        registry = get_model_registry()
        assert registry.golden_artifacts
        for key, artifact in registry.golden_artifacts.items():
            assert isinstance(artifact, GoldenArtifact)
            assert artifact.model_id in registry
            assert REVISION.fullmatch(artifact.revision)
            assert SHA256.fullmatch(artifact.sha256)
            assert artifact.size > 0
            assert artifact.path and artifact.offline_behavior
            assert key == artifact.model_id or key.startswith(artifact.model_id)


class TestModelFamilyPrecisions:
    def test_stable_precisions_drop_the_experimental_ones_and_keep_order(self) -> None:
        families = get_model_registry().families
        assert families["esm_plusplus"].precisions == ("default", "fp8")
        assert families["esm_plusplus"].stable_precisions == ("default",)
        for family in families.values():
            assert family.stable_precisions == tuple(
                precision for precision in family.precisions if precision not in family.experimental_precisions
            )


class TestSaeSelection:
    def test_every_registered_sae_is_found_by_its_own_selection(self) -> None:
        registry = get_model_registry()
        assert registry.sparse_autoencoders
        for spec in registry.sparse_autoencoders.values():
            found = registry.sae_for_base(
                spec.base_model, layer=spec.layer, k=spec.k, codebook_dim=spec.codebook_dim
            )
            assert found is spec

    def test_an_unregistered_selection_fails_instead_of_falling_back(self) -> None:
        registry = get_model_registry()
        spec = next(iter(registry.sparse_autoencoders.values()))
        with pytest.raises(RegistryError, match="Expected one pinned SAE"):
            registry.sae_for_base(spec.base_model, layer=spec.layer + 1, k=spec.k, codebook_dim=spec.codebook_dim)


class TestRequireResolved:
    def test_a_resolved_registry_passes_for_one_model_or_all(self) -> None:
        registry = get_model_registry()
        model_id = next(iter(registry))
        registry.require_resolved(model_id)

    def test_unresolved_files_are_listed_by_model_and_side(self) -> None:
        registry = get_model_registry()
        model = next(iter(registry.values()))
        broken = replace(model, fast=replace(model.fast, unresolved_files=("weights.bin",)))
        patched = ModelRegistry(
            schema_version=registry.schema_version,
            upstreams=registry.upstreams,
            families=registry.families,
            models={broken.id: broken},
        )
        with pytest.raises(RegistryError, match=rf"{broken.id}\.fast:weights\.bin"):
            patched.require_resolved()
        with pytest.raises(RegistryError, match=rf"{broken.id}\.fast:weights\.bin"):
            patched.require_resolved(broken.id)

    def test_a_pending_publication_is_unresolved(self) -> None:
        registry = get_model_registry()
        model = next(iter(registry.values()))
        pending = replace(model, publication_status="pending")
        patched = ModelRegistry(
            schema_version=registry.schema_version,
            upstreams=registry.upstreams,
            families=registry.families,
            models={pending.id: pending},
        )
        with pytest.raises(RegistryError, match=rf"{pending.id}\.fast:unpublished"):
            patched.require_resolved()


class TestPackageExports:
    def test_lazy_exports_resolve_to_their_defining_modules(self) -> None:
        assert fastplms.get_model_registry is get_model_registry
        assert fastplms.RuntimeProfile is RuntimeProfile
        assert fastplms.runtime_profile is runtime_profile
        assert fastplms.OracleAsset is OracleAsset
        assert fastplms.CheckpointSource is CheckpointSource
        assert fastplms.FileDigest is FileDigest
        assert {"embed_dataset", "ModelSpec", "get_model_spec", "load_model_registry"} <= set(dir(fastplms))

    def test_an_unknown_export_is_an_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match="has no attribute 'nothing_here'"):
            fastplms.nothing_here  # noqa: B018

    def test_the_version_is_a_dotted_string(self) -> None:
        assert re.fullmatch(r"\d+\.\d+\.\d+", fastplms.__version__)


class TestRuntimeProfile:
    def test_defaults_ask_for_the_highest_precision_and_keep_tf32(self) -> None:
        profile = RuntimeProfile()
        assert (profile.float32_matmul_precision, profile.allow_tf32) == ("highest", None)
        with pytest.raises(FrozenInstanceError):
            profile.allow_tf32 = True  # type: ignore[misc]

    def test_the_context_sets_the_precision_and_restores_it(self) -> None:
        before = torch.get_float32_matmul_precision()
        other = "medium" if before != "medium" else "high"
        with runtime_profile(RuntimeProfile(float32_matmul_precision=other)):
            assert torch.get_float32_matmul_precision() == other
        assert torch.get_float32_matmul_precision() == before

    def test_the_default_profile_requests_highest_precision(self) -> None:
        before = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("medium")
        try:
            with runtime_profile():
                assert torch.get_float32_matmul_precision() == "highest"
            assert torch.get_float32_matmul_precision() == "medium"
        finally:
            torch.set_float32_matmul_precision(before)

    def test_the_previous_settings_come_back_after_an_exception(self) -> None:
        before = torch.get_float32_matmul_precision()
        other = "medium" if before != "medium" else "high"
        with (
            pytest.raises(RuntimeError, match="boom"),
            runtime_profile(RuntimeProfile(float32_matmul_precision=other)),
        ):
            raise RuntimeError("boom")
        assert torch.get_float32_matmul_precision() == before

    def test_tf32_is_set_for_the_block_and_restored(self) -> None:
        matmul = torch.backends.cuda.matmul
        cudnn = torch.backends.cudnn
        before = (matmul.allow_tf32, cudnn.allow_tf32)
        wanted = not before[0]
        with runtime_profile(RuntimeProfile(allow_tf32=wanted)):
            assert matmul.allow_tf32 is wanted
            assert cudnn.allow_tf32 is wanted
        assert (matmul.allow_tf32, cudnn.allow_tf32) == before

    def test_tf32_follows_the_matmul_precision_and_is_restored_when_the_profile_does_not_name_it(self) -> None:
        matmul = torch.backends.cuda.matmul
        before = matmul.allow_tf32
        with runtime_profile(RuntimeProfile(float32_matmul_precision="high")):
            # Torch derives the legacy flag from the precision setting; the profile sets only the precision.
            assert matmul.allow_tf32 is True
        assert matmul.allow_tf32 == before
