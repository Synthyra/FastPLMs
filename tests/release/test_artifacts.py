from __future__ import annotations

import configparser
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import pytest
import torch
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from safetensors.torch import load_file, save_file

from fastplms.registry import (
    CheckpointSource,
    FileDigest,
    ModelFamily,
    ModelRegistry,
    ModelSpec,
    UpstreamSource,
    get_model_registry,
    load_model_registry,
)
from tools.artifacts import (
    ArtifactError,
    build_artifact,
    canonicalize_checkpoint_weights,
    hash_file,
    validate_artifact,
    validate_repository_legal_inventory,
    validate_weight_artifact,
    verify_checkpoint,
)
from tools.artifacts.build import (
    _RELEASE_TOOL_SCOPE_PATHS,
    _canonical_state_sha256,
    _checkpoint_identity_hash,
    _conversion_equality_attestation,
    _copy_attention_kernel_lock,
    _copy_official_tokenizer_assets,
    _git_runtime_revision,
    _is_weight_file,
    _materialize_model_card,
    _provenance,
    _render_artifact_requirements,
    _tokenizer_checkpoint,
    _validate_vendor_revisions,
    _validated_release_tool_snapshot,
    render_model_card,
)


ROOT = Path(__file__).resolve().parents[2]


def _canonical_text_sha256(path: Path) -> str:
    raw = path.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def _initialize_release_tool_repository(root: Path) -> None:
    for relative_name in _RELEASE_TOOL_SCOPE_PATHS:
        path = root.joinpath(*relative_name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# release tool {relative_name}\n", encoding="utf-8")
    git = ["git", "-c", f"safe.directory={root.as_posix()}"]
    subprocess.run(
        [*git, "init", "--initial-branch=main"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run([*git, "config", "user.email", "tests@example.invalid"], cwd=root, check=True)
    subprocess.run([*git, "config", "user.name", "FastPLMs Tests"], cwd=root, check=True)
    subprocess.run([*git, "add", "."], cwd=root, check=True)
    subprocess.run(
        [*git, "commit", "-m", "immutable release tools"],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "relative_name",
    (
        "tools/artifacts/build.py",
        "tools/artifacts/publish.py",
        "tools/artifacts/offline_probe.py",
        "tools/conversion/state_transforms.py",
    ),
)
def test_release_tool_snapshot_rejects_dirty_critical_tool(
    tmp_path: Path,
    relative_name: str,
) -> None:
    _initialize_release_tool_repository(tmp_path)
    path = tmp_path.joinpath(*relative_name.split("/"))
    path.write_text(path.read_text(encoding="utf-8") + "# dirty\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="release tools must be tracked and clean"):
        _validated_release_tool_snapshot(tmp_path)


def test_release_tool_snapshot_rejects_untracked_scope_growth(tmp_path: Path) -> None:
    _initialize_release_tool_repository(tmp_path)
    (tmp_path / "tools" / "artifacts" / "new_validator.py").write_text(
        "# untracked validation bypass\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="release tools must be tracked and clean"):
        _validated_release_tool_snapshot(tmp_path)


def test_materialized_model_card_keeps_runtime_identity_out_of_user_facing_text() -> None:
    template = render_model_card(get_model_registry()["esm2_8m"])
    runtime_revision = "a" * 40
    source_tree_sha256 = "b" * 64
    runtime_bundle_sha256 = "c" * 64

    card = _materialize_model_card(
        template,
        runtime_revision=runtime_revision,
        source_tree_sha256=source_tree_sha256,
        runtime_bundle_sha256=runtime_bundle_sha256,
    )

    assert "<runtime-revision>" not in card
    assert "FastPLMs.git@" not in card
    assert runtime_revision not in card
    assert source_tree_sha256 not in card
    assert runtime_bundle_sha256 not in card
    assert "- Runtime revision: recorded separately" in card
    assert "- Runtime source identities: recorded in `source-record.json`" in card
    assert "SHA-256" not in card


def test_shared_sources_are_in_runtime_artifacts() -> None:
    """Keep remote-code artifacts closed over their package-source imports."""

    registry = get_model_registry()
    required = {
        "esm2": {"models/_esm_rotary.py"},
        "dplm": {"models/_diffusion_generation.py", "models/_esm_rotary.py"},
        "dplm2": {"models/_diffusion_generation.py", "models/_esm_rotary.py"},
        "esmfold": {"models/_esm_rotary.py", "models/classification_probe.py"},
        "esmfold2": {
            "models/_esm_rotary.py",
            "models/classification_probe.py",
            "models/esm_plusplus",
        },
    }
    package_root = ROOT / "src" / "fastplms"
    for family_id, paths in required.items():
        family = registry.families[family_id]
        assert paths.issubset(family.runtime_paths)
        for relative_path in paths:
            assert (package_root / relative_path).exists()


@pytest.mark.parametrize(
    ("model_id", "required", "excluded"),
    (
        ("esm2_8m", ("torch>=2.13,<2.14", "kernels>=0.15,<0.16"), ("biotite",)),
        ("ankh_base", ("torch>=2.13,<2.14",), ("kernels", "biotite")),
        ("boltz2", ("torch>=2.13,<2.14", "biotite>=1.4,<2"), ("kernels",)),
    ),
)
def test_artifact_requirements_match_advertised_runtime(
    model_id: str,
    required: tuple[str, ...],
    excluded: tuple[str, ...],
) -> None:
    payloads = {
        relative_name: (ROOT / relative_name).read_bytes()
        for relative_name in (
            "requirements/core.in",
            "requirements/features/flash.in",
            "requirements/features/structure.in",
        )
    }
    rendered = _render_artifact_requirements(get_model_registry()[model_id], payloads)

    for requirement in required:
        assert requirement in rendered
    for requirement in excluded:
        assert requirement not in rendered
    assert "fastplms" not in "\n".join(
        line for line in rendered.splitlines() if not line.startswith("#")
    ).lower()


def test_esmfold2_runtime_asset_provenance_records_trust_and_offline_boundary() -> None:
    registry = get_model_registry()
    provenance = _provenance(
        registry,
        registry["esmfold2"],
        {},
        runtime_revision="a" * 40,
        source_tree_sha256="b" * 64,
        runtime_bundle_sha256="c" * 64,
        release_tool_revision="d" * 40,
        release_tool_sha256="e" * 64,
    )

    assert provenance["runtime_assets"] == [
        {
            "id": "esmfold2_ccd",
            "repository": "biohub/ESMFold2",
            "revision": "1ebf0e3481a5184eb6171d40615c79e384b48796",
            "path": "ccd.pkl",
            "sha256": "9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5",
            "size": 417306584,
            "license": "MIT",
            "consumer_family": "esmfold2",
            "trust_kind": "hash_pinned_pickle",
            "offline_behavior": "requires_cached_verified_file",
            "cache_identity": hashlib.sha256(
                b"biohub/ESMFold2@1ebf0e3481a5184eb6171d40615c79e384b48796:"
                b"ccd.pkl:9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5:"
                b"417306584"
            ).hexdigest(),
        }
    ]
    assert len(provenance["runtime_assets"][0]["cache_identity"]) == 64


def test_e1_runtime_artifact_closes_over_split_modules() -> None:
    """Keep every E1 responsibility module inside the bundled runtime tree."""

    family = get_model_registry().families["e1"]
    assert "models/e1" in family.runtime_paths
    source_root = ROOT / "src" / "fastplms" / "models" / "e1"
    for source_name in (
        "attention.py",
        "cache.py",
        "modeling_e1.py",
        "preparation.py",
        "retrieval.py",
    ):
        assert (source_root / source_name).is_file()


@pytest.mark.parametrize("model_id", ("esm2_8m", "esmc_small", "dplm_150m"))
def test_flash_artifact_build_embeds_the_kernel_lock(
    model_id: str,
    tmp_path: Path,
) -> None:
    """Exercise the kernel-lock build step without requiring checkpoint weights."""

    registry = get_model_registry()
    _copy_attention_kernel_lock(
            ROOT,
            tmp_path,
            registry,
            registry[model_id],
        )
    assert (tmp_path / "kernels.lock").read_bytes() == (ROOT / "kernels.lock").read_bytes()


def _synthetic_registry(source_root: Path, checkpoint: Path) -> tuple[ModelRegistry, ModelSpec]:
    requirements = source_root / "requirements"
    (requirements / "features").mkdir(parents=True)
    (requirements / "core.in").write_text(
        "torch>=2.13,<2.14\ntransformers>=5.13,<5.14\n",
        encoding="utf-8",
    )
    (requirements / "features" / "flash.in").write_text(
        "kernels>=0.15,<0.16\n",
        encoding="utf-8",
    )
    (requirements / "features" / "structure.in").write_text(
        "biotite>=1.4,<2\n",
        encoding="utf-8",
    )
    package = source_root / "src" / "fastplms"
    (package / "models" / "toy").mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = '1.0.0'\n", encoding="utf-8")
    (package / "models" / "toy" / "modeling_toy.py").write_text(
        "class ToyConfig: pass\nclass ToyModel: pass\n", encoding="utf-8"
    )
    upstream_root = source_root / "vendor" / "upstream" / "toy"
    upstream_root.mkdir(parents=True)
    canonical_license = upstream_root / "LICENSE"
    canonical_license.write_text("Synthetic test license\n", encoding="utf-8")
    inventory_root = source_root / "LICENSES" / "toy"
    inventory_root.mkdir(parents=True)
    distribution_license = inventory_root / "LICENSE"
    distribution_license.write_text("Synthetic test license\n", encoding="utf-8")
    project_license = source_root / "LICENSE"
    project_license.write_text("FastPLMs test license\n", encoding="utf-8")
    third_party_notices = source_root / "THIRD_PARTY_NOTICES.md"
    third_party_notices.write_text("Synthetic test notice\n", encoding="utf-8")

    config = checkpoint / "config.json"
    weight = checkpoint / "model.safetensors"
    config.write_text('{"model_type": "toy"}\n', encoding="utf-8")
    save_file(
        {
            "linear.bias": torch.arange(4, dtype=torch.float32),
            "linear.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        },
        weight,
        metadata={"format": "pt"},
    )
    fast = CheckpointSource(
        repo_id="Synthyra/ToyModel",
        revision="1" * 40,
        files=(
            FileDigest("config.json", "git-sha1", hash_file(config, "git-sha1")),
            FileDigest("model.safetensors", "sha256", hash_file(weight)),
        ),
    )
    official = CheckpointSource(
        repo_id="upstream/ToyModel",
        revision="2" * 40,
        files=(FileDigest("model.safetensors", "sha256", "3" * 64),),
    )
    upstream = UpstreamSource(
        id="toy",
        path="vendor/upstream/toy",
        url="https://github.com/example/toy.git",
        revision="4" * 40,
        license_expression="MIT",
        license_files=("LICENSE",),
        license_digests=(
            FileDigest("LICENSE", "sha256", _canonical_text_sha256(canonical_license)),
        ),
        distribution_files=(
            FileDigest("LICENSE", "sha256", _canonical_text_sha256(distribution_license)),
        ),
    )
    family = ModelFamily(
        id="toy",
        architecture="Toy",
        upstreams=("toy",),
        tokenizer_mode="sequence",
        public_input="Synthetic token IDs",
        extra="core",
        reference_container="reference-toy",
        reference_adapter="tests.parity.support.reference_adapters.toy",
        attention=("eager",),
        dtypes=("float32",),
        bf16_execution="static_parameters",
        precisions=("default",),
        vram_tier="sequence",
        checkpoint_license="MIT",
        hub_license="mit",
        state_transform="identity",
        representative="toy",
        documentation="docs/toy.md",
        test_tiers=("artifact",),
        runtime_paths=("__init__.py", "models/toy"),
        auto_map_items=(
            ("AutoConfig", "fastplms.models.toy.modeling_toy.ToyConfig"),
            ("AutoModel", "fastplms.models.toy.modeling_toy.ToyModel"),
        ),
        weights_publication_allowed=True,
        conversion_provenance=(
            "Input: synthetic official state. Transformation: identity. "
            "Output: synthetic FastPLMs state. Validation: exact hash equality. "
            "Limitation: synthetic test only."
        ),
    )
    spec = ModelSpec(
        id="toy",
        family=family,
        fast=fast,
        official=official,
        size_category="small",
    )
    registry = ModelRegistry(
        schema_version=1,
        upstreams={"toy": upstream},
        families={"toy": family},
        models={"toy": spec},
        legal_files=(
            FileDigest("LICENSE", "sha256", _canonical_text_sha256(project_license)),
            FileDigest(
                "THIRD_PARTY_NOTICES.md",
                "sha256",
                _canonical_text_sha256(third_party_notices),
            ),
        ),
    )
    return registry, spec


def _inject_checkpoint_race(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    *,
    replacement: bytes,
    in_place: bool = False,
) -> None:
    """Mutate a pinned source after selection but before the builder copies it."""

    from tools.artifacts import build as build_module

    original_copy = build_module._copy_file
    fired = False

    def racing_copy(source: Path, destination: Path) -> None:
        nonlocal fired
        if not fired and source.resolve() == target.resolve():
            fired = True
            if in_place:
                target.write_bytes(replacement)
            else:
                staged = target.with_name(target.name + ".concurrent-replacement")
                staged.write_bytes(replacement)
                staged.replace(target)
        original_copy(source, destination)

    monkeypatch.setattr(build_module, "_copy_file", racing_copy)


def test_artifact_build_rejects_concurrent_config_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    _inject_checkpoint_race(
        monkeypatch,
        checkpoint / "config.json",
        replacement=b'{"model_type":"forged"}\n',
        in_place=True,
    )

    with pytest.raises(ArtifactError, match="Preserved checkpoint bytes differ"):
        build_artifact(
            spec,
            registry,
            checkpoint,
            tmp_path / "artifact",
            source_root,
            _allow_untracked_runtime_for_tests=True,
        )


def test_official_tokenizer_copy_rejects_concurrent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    destination = tmp_path / "artifact"
    snapshot.mkdir()
    tokenizer = snapshot / "tokenizer_config.json"
    tokenizer.write_bytes(b'{"tokenizer_class":"Pinned"}\n')
    source = CheckpointSource(
        repo_id="upstream/tokenizer",
        revision="a" * 40,
        files=(
            FileDigest(
                tokenizer.name,
                "git-sha1",
                hash_file(tokenizer, "git-sha1"),
            ),
        ),
    )
    _inject_checkpoint_race(
        monkeypatch,
        tokenizer,
        replacement=b'{"tokenizer_class":"Forged"}\n',
    )

    with pytest.raises(ArtifactError, match="Preserved checkpoint bytes differ"):
        _copy_official_tokenizer_assets(snapshot, destination, source)


def test_weight_snapshot_rejects_concurrent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    # weights: (...)
    weights = snapshot / "model.safetensors"
    save_file({"weight": torch.arange(8, dtype=torch.float32)}, weights)
    source = CheckpointSource(
        repo_id="upstream/weights",
        revision="b" * 40,
        files=(FileDigest(weights.name, "sha256", hash_file(weights)),),
    )
    _inject_checkpoint_race(
        monkeypatch,
        weights,
        replacement=b"not-the-pinned-safetensors-bytes",
    )

    with pytest.raises(ArtifactError, match="Preserved checkpoint bytes differ"):
        canonicalize_checkpoint_weights(snapshot, source, tmp_path / "artifact")


def _build_synthetic_for_replacement(
    spec: ModelSpec,
    registry: ModelRegistry,
    checkpoint: Path,
    output_root: Path,
    source_root: Path,
    *,
    replace_existing: bool = False,
) -> Path:
    return build_artifact(
        spec,
        registry,
        checkpoint,
        output_root,
        source_root,
        replace=replace_existing,
        _allow_untracked_runtime_for_tests=True,
    )


def _change_synthetic_runtime(source_root: Path, marker: str) -> None:
    runtime = source_root / "src" / "fastplms" / "models" / "toy" / "modeling_toy.py"
    runtime.write_text(
        runtime.read_text(encoding="utf-8") + f"\n# {marker}\n",
        encoding="utf-8",
        newline="\n",
    )


def test_artifact_replace_restores_prior_version_after_backup_rename_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    output_root = tmp_path / "artifacts"
    artifact = _build_synthetic_for_replacement(
        spec,
        registry,
        checkpoint,
        output_root,
        source_root,
    )
    original_manifest = (artifact / "artifact-manifest.json").read_bytes()
    _change_synthetic_runtime(source_root, "replacement generation")

    from tools.artifacts import build as build_module

    original_rename = build_module._atomic_artifact_rename
    backup = output_root / ".ToyModel.backup"
    fired = False

    def fail_after_backup_rename(source: Path, destination: Path, root: Path) -> None:
        nonlocal fired
        original_rename(source, destination, root)
        if destination == backup and not fired:
            fired = True
            raise OSError("injected failure after old-to-backup rename")

    monkeypatch.setattr(build_module, "_atomic_artifact_rename", fail_after_backup_rename)
    with pytest.raises(ArtifactError, match="prior artifact was restored"):
        _build_synthetic_for_replacement(
            spec,
            registry,
            checkpoint,
            output_root,
            source_root,
            replace_existing=True,
        )

    assert fired
    assert (artifact / "artifact-manifest.json").read_bytes() == original_manifest
    validate_artifact(artifact, spec=spec, registry=registry)
    assert not (output_root / ".ToyModel.tmp").exists()
    assert not backup.exists()


def test_artifact_replace_restores_prior_version_when_new_install_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    output_root = tmp_path / "artifacts"
    artifact = _build_synthetic_for_replacement(
        spec,
        registry,
        checkpoint,
        output_root,
        source_root,
    )
    original_manifest = (artifact / "artifact-manifest.json").read_bytes()
    _change_synthetic_runtime(source_root, "replacement generation")

    from tools.artifacts import build as build_module

    original_rename = build_module._atomic_artifact_rename
    temporary = output_root / ".ToyModel.tmp"
    fired = False

    def fail_new_install(source: Path, destination: Path, root: Path) -> None:
        nonlocal fired
        if source == temporary and destination == artifact and not fired:
            fired = True
            raise OSError("injected failure during temporary-to-destination rename")
        original_rename(source, destination, root)

    monkeypatch.setattr(build_module, "_atomic_artifact_rename", fail_new_install)
    with pytest.raises(ArtifactError, match="prior validated artifact was restored"):
        _build_synthetic_for_replacement(
            spec,
            registry,
            checkpoint,
            output_root,
            source_root,
            replace_existing=True,
        )

    assert fired
    assert (artifact / "artifact-manifest.json").read_bytes() == original_manifest
    validate_artifact(artifact, spec=spec, registry=registry)
    assert not temporary.exists()
    assert not (output_root / ".ToyModel.backup").exists()


def test_artifact_replace_recovers_backup_on_next_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    output_root = tmp_path / "artifacts"
    artifact = _build_synthetic_for_replacement(
        spec,
        registry,
        checkpoint,
        output_root,
        source_root,
    )
    original_manifest = (artifact / "artifact-manifest.json").read_bytes()
    backup = output_root / ".ToyModel.backup"
    temporary = output_root / ".ToyModel.tmp"
    artifact.rename(backup)
    temporary.mkdir()
    (temporary / "partial-write").write_text("incomplete", encoding="utf-8")

    from tools.artifacts import build as build_module

    def stop_after_recovery(*args: object, **kwargs: object) -> None:
        raise ArtifactError("injected build stop after transaction recovery")

    monkeypatch.setattr(build_module, "_copy_checkpoint_assets", stop_after_recovery)
    with pytest.raises(ArtifactError, match="injected build stop after transaction recovery"):
        _build_synthetic_for_replacement(
            spec,
            registry,
            checkpoint,
            output_root,
            source_root,
            replace_existing=True,
        )

    assert artifact.is_dir()
    assert (artifact / "artifact-manifest.json").read_bytes() == original_manifest
    validate_artifact(artifact, spec=spec, registry=registry)
    assert not temporary.exists()
    assert not backup.exists()


def test_artifact_replace_success_cleans_transaction_slots(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    output_root = tmp_path / "artifacts"
    artifact = _build_synthetic_for_replacement(
        spec,
        registry,
        checkpoint,
        output_root,
        source_root,
    )
    original_manifest = (artifact / "artifact-manifest.json").read_bytes()
    _change_synthetic_runtime(source_root, "successful replacement")

    replaced = _build_synthetic_for_replacement(
        spec,
        registry,
        checkpoint,
        output_root,
        source_root,
        replace_existing=True,
    )

    assert replaced == artifact
    assert (artifact / "artifact-manifest.json").read_bytes() != original_manifest
    assert "successful replacement" in (
        artifact / "fastplms" / "models" / "toy" / "modeling_toy.py"
    ).read_text(encoding="utf-8")
    validate_artifact(artifact, spec=spec, registry=registry)
    assert not (output_root / ".ToyModel.tmp").exists()
    assert not (output_root / ".ToyModel.backup").exists()


def test_repeated_artifact_replace_does_not_retain_stale_files(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    output_root = tmp_path / "artifacts"
    artifact = _build_synthetic_for_replacement(
        spec,
        registry,
        checkpoint,
        output_root,
        source_root,
    )

    for generation in range(2):
        stale = artifact / f"stale-generation-{generation}.bin"
        stale.write_bytes(b"must not survive replacement")
        _change_synthetic_runtime(source_root, f"replacement {generation}")
        artifact = _build_synthetic_for_replacement(
            spec,
            registry,
            checkpoint,
            output_root,
            source_root,
            replace_existing=True,
        )
        assert not stale.exists()
        assert not any(artifact.glob("stale-generation-*.bin"))
        validate_artifact(artifact, spec=spec, registry=registry)

    assert not (output_root / ".ToyModel.tmp").exists()
    assert not (output_root / ".ToyModel.backup").exists()


@pytest.mark.parametrize("slot_suffix", (".tmp", ".backup"))
def test_artifact_replace_rejects_symlink_transaction_slots(
    tmp_path: Path,
    slot_suffix: str,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    output_root = tmp_path / "artifacts"
    artifact = _build_synthetic_for_replacement(
        spec,
        registry,
        checkpoint,
        output_root,
        source_root,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    os.symlink(outside, output_root / f".ToyModel{slot_suffix}", target_is_directory=True)

    with pytest.raises(ArtifactError, match="must not be a symlink"):
        _build_synthetic_for_replacement(
            spec,
            registry,
            checkpoint,
            output_root,
            source_root,
            replace_existing=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert artifact.is_dir()


def test_artifact_build_rejects_symlink_output_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    outside = tmp_path / "outside-output"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    output_root = tmp_path / "artifact-link"
    os.symlink(outside, output_root, target_is_directory=True)

    with pytest.raises(ArtifactError, match="output root must not be a symlink"):
        _build_synthetic_for_replacement(
            spec,
            registry,
            checkpoint,
            output_root,
            source_root,
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (outside / "ToyModel").exists()


def test_artifact_build_rejects_repository_name_path_escape(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    escaped_fast = replace(spec.fast, repo_id="Synthyra/../../escaped-artifact")
    escaped_spec = replace(spec, fast=escaped_fast)
    escaped_registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families=registry.families,
        models={spec.id: escaped_spec},
        runtime_assets=registry.runtime_assets,
        attention_kernels=registry.attention_kernels,
        legal_files=registry.legal_files,
    )
    output_root = tmp_path / "artifacts"

    with pytest.raises(ArtifactError, match="Invalid artifact repository name"):
        _build_synthetic_for_replacement(
            escaped_spec,
            escaped_registry,
            checkpoint,
            output_root,
            source_root,
        )
    assert not (tmp_path / "escaped-artifact").exists()


def test_artifact_build_is_deterministic_and_self_verifying(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)

    first = build_artifact(
        spec,
        registry,
        checkpoint,
        tmp_path / "first",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )
    second = build_artifact(
        spec,
        registry,
        checkpoint,
        tmp_path / "second",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )
    validate_artifact(first)
    validate_artifact(second)

    first_manifest = json.loads((first / "artifact-manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert first_manifest == second_manifest
    config = json.loads((first / "config.json").read_text(encoding="utf-8"))
    assert config["auto_map"] == {
        "AutoConfig": "modeling_fastplms.ToyConfig",
        "AutoModel": "modeling_fastplms.ToyModel",
    }
    assert config["fastplms_model_id"] == spec.id
    assert config["fastplms_checkpoint_repo_id"] == spec.artifact_checkpoint.repo_id
    assert config["fastplms_checkpoint_revision"] == spec.artifact_checkpoint.revision
    assert config["fastplms_weights_revision"] == spec.artifact_checkpoint.revision
    assert config["fastplms_runtime_revision"].startswith("source-tree-sha256:")
    assert len(config["fastplms_source_tree_sha256"]) == 64
    assert len(config["fastplms_runtime_bundle_sha256"]) == 64
    assert config["fastplms_checkpoint_hash"] == _checkpoint_identity_hash(
        spec.artifact_checkpoint
    )
    assert (first / "fastplms" / "models" / "toy" / "modeling_toy.py").is_file()
    assert (first / "fastplms_bundle.py").is_file()
    assert (first / "requirements.txt").read_text(encoding="utf-8") == (
        "# Direct runtime dependencies for Synthyra/ToyModel.\n"
        "# FastPLMs source is embedded in this model repository.\n"
        "torch>=2.13,<2.14\n"
        "transformers>=5.13,<5.14\n"
    )
    bridge = (first / "modeling_fastplms.py").read_text(encoding="utf-8")
    assert "from .fastplms_bundle import RUNTIME_DATA, RUNTIME_HASH" in bridge
    assert "from .fastplms." not in bridge
    assert not (first / "vendor").exists()
    assert (first / "LICENSES" / "toy" / "LICENSE").is_file()
    assert (first / "THIRD_PARTY_NOTICES.md").is_file()
    assert (first / "model.safetensors.index.json").is_file()
    assert not (first / "model.safetensors").exists()
    assert len(list(first.glob("model-*.safetensors"))) == 1
    provenance = json.loads((first / "source-record.json").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == 4
    assert provenance["generator"] == {
        "name": "tools.artifacts.build",
        "version": 4,
    }
    assert provenance["weights_license_status"] == "resolved"
    assert provenance["redistributable"] is True
    assert provenance["weights_revision"] == spec.artifact_checkpoint.revision
    assert provenance["runtime_revision"] == config["fastplms_runtime_revision"]
    assert provenance["source_tree_sha256"] == config["fastplms_source_tree_sha256"]
    assert provenance["runtime_bundle_sha256"] == config["fastplms_runtime_bundle_sha256"]
    assert provenance["release_tool_revision"] == config[
        "fastplms_release_tool_revision"
    ]
    assert provenance["release_tool_sha256"] == config["fastplms_release_tool_sha256"]
    assert "<runtime-revision>" not in (first / "README.md").read_text(encoding="utf-8")
    assert provenance["attestations"]["complete_artifact"]["scope"] == "weights+runtime"
    runtime_attestation = json.loads(
        (first / "runtime-attestation.json").read_text(encoding="utf-8")
    )
    assert runtime_attestation["scope"] == "runtime-only"
    assert runtime_attestation["weights_license_status"] == "resolved"
    assert runtime_attestation["redistributable"] is True
    assert runtime_attestation["weights"] == {
        "repo_id": spec.fast.repo_id,
        "revision": spec.fast.revision,
    }
    assert "source-record.json" not in runtime_attestation["files"]
    assert "requirements.txt" in runtime_attestation["files"]
    assert not any(_is_weight_file(path) for path in runtime_attestation["files"])
    assert provenance["bf16_execution"] == "static_parameters"
    assert provenance["canonical_weights"]["source_schema"] == "canonical"
    assert provenance["canonical_weights"]["state_transform"] == "identity"
    assert provenance["hub_license_metadata"] == {"license": "mit"}
    assert "`static_parameters`" in (first / "README.md").read_text(encoding="utf-8")

    second_config_path = second / "config.json"
    second_config = json.loads(second_config_path.read_text(encoding="utf-8"))
    second_config["fastplms_model_id"] = "wrong-model"
    second_config_path.write_text(
        json.dumps(second_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    second_manifest["config.json"] = f"sha256:{hash_file(second_config_path)}"
    (second / "artifact-manifest.json").write_text(
        json.dumps(second_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="packaging identity differs"):
        validate_artifact(second)

    readme_path = first / "README.md"
    manifest_path = first / "artifact-manifest.json"
    original_readme = readme_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    tampered_readme = original_readme.decode("utf-8").replace(
        'license: "mit"',
        'license: "apache-2.0"',
        1,
    )
    assert tampered_readme.encode("utf-8") != original_readme
    readme_path.write_text(tampered_readme, encoding="utf-8", newline="\n")
    tampered_manifest = json.loads(original_manifest)
    tampered_manifest["README.md"] = f"sha256:{hash_file(readme_path)}"
    manifest_path.write_text(
        json.dumps(tampered_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ArtifactError, match="differs from provenance"):
        validate_artifact(first)
    readme_path.write_bytes(original_readme)
    manifest_path.write_bytes(original_manifest)

    next(first.glob("model-*.safetensors")).write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="digest mismatch"):
        validate_artifact(first)


def test_complete_artifact_rejects_self_attested_card_runtime_placeholder(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    artifact = build_artifact(
        spec,
        registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )
    readme = artifact / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            "requirements.txt",
            "requirements.txt@<runtime-revision>",
            1,
        ),
        encoding="utf-8",
        newline="\n",
    )
    runtime_attestation_path = artifact / "runtime-attestation.json"
    runtime_attestation = json.loads(runtime_attestation_path.read_text(encoding="utf-8"))
    runtime_attestation["files"]["README.md"] = f"sha256:{hash_file(readme)}"
    runtime_attestation_path.write_text(
        json.dumps(runtime_attestation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = artifact / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["README.md"] = f"sha256:{hash_file(readme)}"
    manifest["runtime-attestation.json"] = (
        f"sha256:{hash_file(runtime_attestation_path)}"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ArtifactError, match="unresolved runtime-revision placeholder"):
        validate_artifact(artifact, spec=spec, registry=registry)


def test_generated_bridge_uses_private_verified_runtime(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    artifact = build_artifact(
        spec,
        registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    runtime_hash = config["fastplms_runtime_bundle_sha256"]
    poisoned_cache = tmp_path / f"_fastplms_runtime_{runtime_hash}" / "fastplms"
    poisoned_cache.mkdir(parents=True)
    (poisoned_cache / "__init__.py").write_text(
        "raise RuntimeError('shared cache was trusted')\n",
        encoding="utf-8",
    )
    probe = tmp_path / "private_runtime_probe.py"
    probe.write_text(
        textwrap.dedent(
            """\
            import importlib.util
            import sys
            import types
            from pathlib import Path


            artifact = Path(sys.argv[1])
            poisoned_cache = Path(sys.argv[2]).resolve()
            package = types.ModuleType("artifact_private")
            package.__package__ = "artifact_private"
            package.__path__ = [str(artifact)]
            sys.modules["artifact_private"] = package
            module_name = "artifact_private.modeling_fastplms"
            spec = importlib.util.spec_from_file_location(
                module_name,
                artifact / "modeling_fastplms.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Unable to load generated bridge")
            bridge = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = bridge
            spec.loader.exec_module(bridge)

            assert len(bridge._RUNTIME_TEMPORARIES) == 1
            private_root = Path(bridge._RUNTIME_TEMPORARIES[0].name).resolve()
            package_root = private_root / "fastplms"
            runtime = sys.modules["fastplms"]
            assert runtime.__fastplms_artifact_runtime_temporaries__ == tuple(
                bridge._RUNTIME_TEMPORARIES
            )
            assert private_root.name.startswith("fastplms-artifact-runtime-")
            assert poisoned_cache != package_root
            assert poisoned_cache not in package_root.parents
            assert not any(path.name == "__pycache__" for path in package_root.rglob("*"))

            before = bridge._runtime_file_hashes(package_root)
            source = next(path for path in package_root.rglob("*.py") if path.is_file())
            relative = source.relative_to(package_root).as_posix()
            original = source.read_bytes()
            source.write_bytes(original + b"\\n# in-place mutation\\n")
            after = bridge._runtime_file_hashes(package_root)
            assert after[relative] != before[relative]
            source.write_bytes(original)

            bytecode = package_root / "injected.pyc"
            bytecode.write_bytes(b"not trusted bytecode")
            try:
                bridge._runtime_file_hashes(package_root)
            except RuntimeError as error:
                assert "contains bytecode" in str(error)
            else:
                raise AssertionError("Private runtime bytecode was accepted")
            bytecode.unlink()

            link_probe = package_root / "pretend_symlink.py"
            link_probe.write_text("# symlink stand-in\\n", encoding="utf-8")
            original_is_symlink = Path.is_symlink
            Path.is_symlink = lambda path: path == link_probe or original_is_symlink(path)
            try:
                try:
                    bridge._runtime_file_hashes(package_root)
                except RuntimeError as error:
                    assert "contains a symlink" in str(error)
                else:
                    raise AssertionError("Private runtime symlink was accepted")
            finally:
                Path.is_symlink = original_is_symlink
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "TMPDIR": str(tmp_path),
            "TMP": str(tmp_path),
            "TEMP": str(tmp_path),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(probe), str(artifact), str(poisoned_cache)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "relative_name",
    (
        "../outside.txt",
        "/absolute.txt",
        "C:/absolute.txt",
        "nested//non-normalized.txt",
        r"nested\windows-path.txt",
        "NUL",
        "nested/con.py",
        "nested/bad:name.py",
        "nested/trailing.",
    ),
)
def test_artifact_validation_rejects_unsafe_manifest_paths(
    tmp_path: Path,
    relative_name: str,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    artifact = build_artifact(
        spec,
        registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )
    manifest_path = artifact / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[relative_name] = "sha256:" + "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ArtifactError, match="invalid artifact manifest path"):
        validate_artifact(artifact)


def test_different_runtime_bundles_fail_without_replacing_loaded_runtime(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)

    first = build_artifact(
        spec,
        registry,
        checkpoint,
        tmp_path / "first",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )
    runtime_source = source_root / "src" / "fastplms" / "models" / "toy" / "modeling_toy.py"
    runtime_source.write_text(
        runtime_source.read_text(encoding="utf-8") + "\n# Distinct runtime identity.\n",
        encoding="utf-8",
        newline="\n",
    )
    second = build_artifact(
        spec,
        registry,
        checkpoint,
        tmp_path / "second",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )

    probe = tmp_path / "mixed_runtime_probe.py"
    probe.write_text(
        textwrap.dedent(
            """\
            import importlib.util
            import sys
            import types
            from pathlib import Path


            def load_bridge(root, package_name):
                package = types.ModuleType(package_name)
                package.__package__ = package_name
                package.__path__ = [str(root)]
                sys.modules[package_name] = package
                module_name = f"{package_name}.modeling_fastplms"
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    root / "modeling_fastplms.py",
                )
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"Unable to load {root}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module


            first = load_bridge(Path(sys.argv[1]), "artifact_first")
            runtime = sys.modules["fastplms"]
            runtime_hash = runtime.__fastplms_artifact_runtime_hash__
            try:
                load_bridge(Path(sys.argv[2]), "artifact_second")
            except RuntimeError as error:
                if "incompatible runtime sources" not in str(error):
                    raise
            else:
                raise AssertionError("A different runtime bundle loaded silently")
            assert sys.modules["fastplms"] is runtime
            assert runtime.__fastplms_artifact_runtime_hash__ == runtime_hash
            assert first.ToyConfig().__class__ is first.ToyConfig
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(probe), str(first), str(second)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_complementary_fastplms_artifacts_load_in_one_process(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, first_spec = _synthetic_registry(source_root, checkpoint)

    second_module = source_root / "src" / "fastplms" / "models" / "toy_second"
    second_module.mkdir(parents=True)
    (second_module / "modeling_toy_second.py").write_text(
        "class ToySecondConfig: pass\nclass ToySecondModel: pass\n",
        encoding="utf-8",
    )
    second_family = replace(
        first_spec.family,
        id="toy_second",
        runtime_paths=("__init__.py", "models/toy_second"),
        auto_map_items=(
            (
                "AutoConfig",
                "fastplms.models.toy_second.modeling_toy_second.ToySecondConfig",
            ),
            (
                "AutoModel",
                "fastplms.models.toy_second.modeling_toy_second.ToySecondModel",
            ),
        ),
    )
    second_spec = replace(
        first_spec,
        id="toy_second",
        family=second_family,
        fast=replace(first_spec.fast, repo_id="Synthyra/ToyModelSecond"),
    )
    registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families={
            first_spec.family.id: first_spec.family,
            second_family.id: second_family,
        },
        models={
            first_spec.id: first_spec,
            second_spec.id: second_spec,
        },
        runtime_assets=registry.runtime_assets,
        attention_kernels=registry.attention_kernels,
        legal_files=registry.legal_files,
    )
    first = build_artifact(
        first_spec,
        registry,
        checkpoint,
        tmp_path / "first",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )
    second = build_artifact(
        second_spec,
        registry,
        checkpoint,
        tmp_path / "second",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )

    probe = tmp_path / "compatible_runtime_probe.py"
    probe.write_text(
        textwrap.dedent(
            """\
            import importlib.util
            import sys
            import types
            from pathlib import Path


            def load_bridge(root, package_name):
                package = types.ModuleType(package_name)
                package.__package__ = package_name
                package.__path__ = [str(root)]
                sys.modules[package_name] = package
                module_name = f"{package_name}.modeling_fastplms"
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    root / "modeling_fastplms.py",
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module


            first = load_bridge(Path(sys.argv[1]), "artifact_first")
            runtime = sys.modules["fastplms"]
            second = load_bridge(Path(sys.argv[2]), "artifact_second")
            assert sys.modules["fastplms"] is runtime
            assert len(runtime.__fastplms_artifact_runtime_hashes__) == 2
            assert first.ToyConfig().__class__ is first.ToyConfig
            assert second.ToySecondConfig().__class__ is second.ToySecondConfig
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(probe), str(first), str(second)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_preloaded_non_artifact_fastplms_runtime_is_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    source_package = source_root / "src" / "fastplms"
    artifact = build_artifact(
        spec,
        registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )
    probe = tmp_path / "external_runtime_probe.py"
    probe.write_text(
        textwrap.dedent(
            """\
            import importlib.util
            import sys
            import types
            from pathlib import Path

            source_package = Path(sys.argv[1])
            artifact = Path(sys.argv[2])
            external_spec = importlib.util.spec_from_file_location(
                "fastplms",
                source_package / "__init__.py",
                submodule_search_locations=[str(source_package)],
            )
            external = importlib.util.module_from_spec(external_spec)
            sys.modules["fastplms"] = external
            external_spec.loader.exec_module(external)

            artifact_package = types.ModuleType("artifact")
            artifact_package.__package__ = "artifact"
            artifact_package.__path__ = [str(artifact)]
            sys.modules["artifact"] = artifact_package
            bridge_spec = importlib.util.spec_from_file_location(
                "artifact.modeling_fastplms",
                artifact / "modeling_fastplms.py",
            )
            bridge = importlib.util.module_from_spec(bridge_spec)
            sys.modules["artifact.modeling_fastplms"] = bridge
            try:
                bridge_spec.loader.exec_module(bridge)
            except RuntimeError as error:
                if "non-artifact fastplms module" not in str(error):
                    raise
            else:
                raise AssertionError("External FastPLMs source loaded into an artifact runtime")
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(probe),
            str(source_root / "src" / "fastplms"),
            str(artifact),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_legal_texts_use_canonical_lf_across_checkouts(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    legal_paths = (
        source_root / "LICENSE",
        source_root / "THIRD_PARTY_NOTICES.md",
        source_root / "vendor" / "upstream" / "toy" / "LICENSE",
        source_root / "LICENSES" / "toy" / "LICENSE",
    )
    for path in legal_paths:
        raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        path.write_bytes(raw.replace(b"\n", b"\r\n"))

    validate_repository_legal_inventory(source_root, registry)
    artifact = build_artifact(
        spec,
        registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )
    distributed = (
        artifact / "LICENSES" / "toy" / "LICENSE",
        artifact / "LICENSES" / "FastPLMs-Apache-2.0.txt",
        artifact / "THIRD_PARTY_NOTICES.md",
    )
    for path in distributed:
        assert b"\r" not in path.read_bytes()

    canonical = legal_paths[2]
    canonical.write_text("Changed license content\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="canonical LF normalization"):
        validate_repository_legal_inventory(source_root, registry)


def test_manifest_distributes_required_modified_file_notices() -> None:
    registry = load_model_registry()
    distribution = {
        source_id: {item.path for item in source.distribution_files}
        for source_id, source in registry.upstreams.items()
    }
    assert {"Apache-2.0.txt", "BSD-3-Clause.txt", "MODIFICATIONS.md"}.issubset(distribution["e1"])
    assert {"LICENSE", "SOURCE_RECORD.md"}.issubset(distribution["dplm"])
    assert {"LICENSE", "MODIFICATIONS.md", "SOURCE_RECORD.md"}.issubset(distribution["openfold"])


def test_artifact_rejects_stale_checked_in_hub_license_metadata(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    card_path = source_root / "model_cards" / "toy.md"
    card_path.parent.mkdir()
    card_path.write_text(
        '---\nlibrary_name: transformers\nlicense: "apache-2.0"\n---\n\n# Toy\n',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match=r"license metadata differs from models\.toml"):
        build_artifact(
            spec,
            registry,
            checkpoint,
            tmp_path / "artifact",
            source_root,
            _allow_untracked_runtime_for_tests=True,
        )


def test_artifact_copies_official_tokenizer_bytes_exactly(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    official_snapshot = tmp_path / "official"
    checkpoint.mkdir()
    official_snapshot.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)

    candidate_tokenizer = checkpoint / "tokenizer.json"
    official_tokenizer = official_snapshot / "tokenizer.json"
    candidate_tokenizer_config = checkpoint / "tokenizer_config.json"
    official_tokenizer_config = official_snapshot / "tokenizer_config.json"
    candidate_tokenizer.write_bytes(b'{"source":"candidate"}\n')
    official_tokenizer.write_bytes(b'{"source":"official"}\n')
    candidate_tokenizer_config.write_bytes(b'{"tokenizer_class":"CandidateTokenizer"}\n')
    official_tokenizer_config.write_bytes(b'{"tokenizer_class":"BuiltInTokenizer"}\n')
    fast = replace(
        spec.fast,
        files=(
            *spec.fast.files,
            FileDigest("tokenizer.json", "sha256", hash_file(candidate_tokenizer)),
            FileDigest(
                "tokenizer_config.json",
                "sha256",
                hash_file(candidate_tokenizer_config),
            ),
        ),
    )
    official = replace(
        spec.official,
        files=(
            *spec.official.files,
            FileDigest("tokenizer.json", "sha256", hash_file(official_tokenizer)),
            FileDigest(
                "tokenizer_config.json",
                "sha256",
                hash_file(official_tokenizer_config),
            ),
        ),
    )
    family = replace(spec.family, tokenizer_mode="tokenizer")
    tokenizer_spec = replace(spec, family=family, fast=fast, official=official)
    tokenizer_registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families={family.id: family},
        models={tokenizer_spec.id: tokenizer_spec},
        legal_files=registry.legal_files,
    )

    artifact = build_artifact(
        tokenizer_spec,
        tokenizer_registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        tokenizer_dir=official_snapshot,
        _allow_untracked_runtime_for_tests=True,
    )
    assert (artifact / "tokenizer.json").read_bytes() == official_tokenizer.read_bytes()
    assert (
        artifact / "tokenizer_config.json"
    ).read_bytes() == official_tokenizer_config.read_bytes()
    provenance = json.loads((artifact / "source-record.json").read_text(encoding="utf-8"))
    assert provenance["tokenizer_checkpoint"]["repo_id"] == official.repo_id
    assert provenance["tokenizer_checkpoint"]["revision"] == official.revision
    assert provenance["tokenizer_checkpoint"]["files"] == {
        "tokenizer.json": official.files[-2].encoded,
        "tokenizer_config.json": official.files[-1].encoded,
    }


def test_artifact_rewrites_custom_tokenizer_auto_map_to_local_bridge(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    official_snapshot = tmp_path / "official"
    checkpoint.mkdir()
    official_snapshot.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)

    runtime_path = source_root / "src" / "fastplms" / "models" / "toy" / "modeling_toy.py"
    runtime_path.write_text(
        runtime_path.read_text(encoding="utf-8") + "\nclass ToyTokenizer: pass\n",
        encoding="utf-8",
        newline="\n",
    )
    candidate_tokenizer = checkpoint / "tokenizer.json"
    official_tokenizer = official_snapshot / "tokenizer.json"
    official_tokenizer_config = official_snapshot / "tokenizer_config.json"
    candidate_tokenizer.write_text('{"source":"candidate"}\n', encoding="utf-8")
    official_tokenizer.write_text('{"source":"official"}\n', encoding="utf-8")
    official_tokenizer_config.write_text(
        json.dumps(
            {
                "auto_map": {
                    "AutoProcessor": "upstream_processing.UpstreamProcessor",
                    "AutoTokenizer": [
                        "upstream_tokenization.UpstreamTokenizer",
                        None,
                    ],
                },
                "preserved": True,
                "tokenizer_class": "UpstreamTokenizer",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fast = replace(
        spec.fast,
        files=(
            *spec.fast.files,
            FileDigest("tokenizer.json", "sha256", hash_file(candidate_tokenizer)),
        ),
    )
    official = replace(
        spec.official,
        files=(
            *spec.official.files,
            FileDigest("tokenizer.json", "sha256", hash_file(official_tokenizer)),
            FileDigest(
                "tokenizer_config.json",
                "sha256",
                hash_file(official_tokenizer_config),
            ),
        ),
    )
    family = replace(
        spec.family,
        tokenizer_mode="tokenizer",
        tokenizer_class="fastplms.models.toy.modeling_toy.ToyTokenizer",
    )
    tokenizer_spec = replace(spec, family=family, fast=fast, official=official)
    tokenizer_registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families={family.id: family},
        models={tokenizer_spec.id: tokenizer_spec},
        legal_files=registry.legal_files,
    )

    artifact = build_artifact(
        tokenizer_spec,
        tokenizer_registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        tokenizer_dir=official_snapshot,
        _allow_untracked_runtime_for_tests=True,
    )

    tokenizer_config = json.loads(
        (artifact / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert tokenizer_config["auto_map"] == {
        "AutoProcessor": "upstream_processing.UpstreamProcessor",
        "AutoTokenizer": ["modeling_fastplms.ToyTokenizer", None],
    }
    assert tokenizer_config["preserved"] is True
    assert "ToyTokenizer =" in (artifact / "modeling_fastplms.py").read_text(encoding="utf-8")
    validate_artifact(artifact)


def test_esm3_uses_the_manifest_pinned_official_esmc_tokenizer() -> None:
    registry = load_model_registry()
    spec = registry["esm3_small"]
    tokenizer_checkpoint = _tokenizer_checkpoint(registry, spec)

    assert spec.tokenizer_source_id == "esmc_small"
    assert tokenizer_checkpoint == registry["esmc_small"].official
    assert tokenizer_checkpoint.repo_id == "biohub/ESMC-300M"
    assert {Path(item.path).name for item in tokenizer_checkpoint.files}.issuperset(
        {"special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"}
    )


def test_fast_checkpoint_artifact_requires_explicit_official_tokenizer_snapshot(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    candidate_tokenizer = checkpoint / "tokenizer.json"
    candidate_tokenizer.write_bytes(b'{"source":"candidate"}\n')
    fast = replace(
        spec.fast,
        files=(
            *spec.fast.files,
            FileDigest("tokenizer.json", "sha256", hash_file(candidate_tokenizer)),
        ),
    )
    official = replace(
        spec.official,
        files=(
            *spec.official.files,
            FileDigest("tokenizer.json", "sha256", "0" * 64),
        ),
    )
    family = replace(spec.family, tokenizer_mode="tokenizer")
    tokenizer_spec = replace(spec, family=family, fast=fast, official=official)
    tokenizer_registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families={family.id: family},
        models={tokenizer_spec.id: tokenizer_spec},
        legal_files=registry.legal_files,
    )

    with pytest.raises(ArtifactError, match="requires the pinned official tokenizer snapshot"):
        build_artifact(
            tokenizer_spec,
            tokenizer_registry,
            checkpoint,
            tmp_path / "artifact",
            source_root,
        )


def test_checkpoint_verification_reports_hash_mismatch(tmp_path: Path) -> None:
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"content")
    source = CheckpointSource(
        repo_id="Synthyra/ToyModel",
        revision="1" * 40,
        files=(FileDigest("model.safetensors", "sha256", "0" * 64),),
    )
    with pytest.raises(ArtifactError, match="Checkpoint verification failed"):
        verify_checkpoint(tmp_path, source)


def test_artifact_build_rejects_unresolved_provenance(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    unresolved_fast = replace(spec.fast, unresolved_files=("tokenizer.json",))
    unresolved_spec = replace(spec, fast=unresolved_fast)
    unresolved_registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families=registry.families,
        models={spec.id: unresolved_spec},
        legal_files=registry.legal_files,
    )

    with pytest.raises(ArtifactError, match="Release provenance is unresolved"):
        build_artifact(
            unresolved_spec,
            unresolved_registry,
            checkpoint,
            tmp_path / "artifact",
            source_root,
        )


def test_artifact_build_rejects_missing_legal_inventory(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    (source_root / "LICENSES" / "toy" / "LICENSE").unlink()

    with pytest.raises(ArtifactError, match="Missing required toy distribution legal file"):
        build_artifact(
            spec,
            registry,
            checkpoint,
            tmp_path / "artifact",
            source_root,
        )


def test_artifact_build_rejects_missing_conversion_record(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    family = replace(spec.family, conversion_provenance="")
    invalid_spec = replace(spec, family=family)
    invalid_registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families={family.id: family},
        models={invalid_spec.id: invalid_spec},
        legal_files=registry.legal_files,
    )

    with pytest.raises(ArtifactError, match="missing conversion provenance"):
        build_artifact(
            invalid_spec,
            invalid_registry,
            checkpoint,
            tmp_path / "artifact",
            source_root,
        )


def test_artifact_build_stamps_unresolved_checkpoint_as_nonredistributable(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    family = replace(spec.family, weights_publication_allowed=False)
    invalid_spec = replace(spec, family=family)
    invalid_registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families={family.id: family},
        models={invalid_spec.id: invalid_spec},
        legal_files=registry.legal_files,
    )

    artifact = build_artifact(
        invalid_spec,
        invalid_registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )

    provenance = json.loads((artifact / "source-record.json").read_text(encoding="utf-8"))
    attestation = json.loads(
        (artifact / "runtime-attestation.json").read_text(encoding="utf-8")
    )
    assert provenance["weights_license_status"] == "unresolved"
    assert provenance["redistributable"] is False
    assert attestation["weights_license_status"] == "unresolved"
    assert attestation["redistributable"] is False


@pytest.mark.parametrize("relative_name", ("credentials.pem", "secrets.py"))
def test_artifact_build_rejects_unknown_runtime_source_extension(
    tmp_path: Path,
    relative_name: str,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    secret = source_root / "src" / "fastplms" / "models" / "toy" / relative_name
    secret.write_text("private material", encoding="utf-8")

    with pytest.raises(ArtifactError, match="sensitive path"):
        build_artifact(
            spec,
            registry,
            checkpoint,
            tmp_path / "artifact",
            source_root,
        )


def test_artifact_build_rejects_untracked_runtime_source(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    subprocess.run(["git", "init", "--initial-branch=main"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "FastPLMs Tests"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "src/fastplms"], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "-m", "runtime fixture"], cwd=source_root, check=True)
    untracked = source_root / "src" / "fastplms" / "models" / "toy" / "injected.py"
    untracked.write_text("TOKEN = 'unsafe'\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="tracked and clean"):
        build_artifact(
            spec,
            registry,
            checkpoint,
            tmp_path / "artifact",
            source_root,
        )


def test_artifact_build_rejects_runtime_source_without_git_provenance(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    injected = source_root / "src" / "fastplms" / "models" / "toy" / "injected.py"
    injected.write_text("APPROVED_EXTENSION_BUT_UNTRACKED = True\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="verifiable Git worktree"):
        build_artifact(
            spec,
            registry,
            checkpoint,
            tmp_path / "artifact",
            source_root,
        )


def test_artifact_build_retains_validated_names_when_worktree_file_is_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    subprocess.run(["git", "init", "--initial-branch=main"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "FastPLMs Tests"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(["git", "add", "src/fastplms"], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "-m", "runtime fixture"], cwd=source_root, check=True)
    runtime_source = (
        source_root / "src" / "fastplms" / "models" / "toy" / "modeling_toy.py"
    )
    committed = runtime_source.read_bytes()

    def delete_after_validation(*args: object, **kwargs: object) -> str | None:
        revision = _git_runtime_revision(*args, **kwargs)  # type: ignore[arg-type]
        runtime_source.unlink()
        return revision

    monkeypatch.setattr(
        "tools.artifacts.build._git_runtime_revision",
        delete_after_validation,
    )

    artifact = build_artifact(
        spec,
        registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )

    assert (artifact / "fastplms" / "models" / "toy" / "modeling_toy.py").read_bytes() == (
        committed
    )


def test_artifact_build_uses_validated_git_blobs_after_worktree_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    subprocess.run(["git", "init", "--initial-branch=main"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "FastPLMs Tests"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "src/fastplms"], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "-m", "runtime fixture"], cwd=source_root, check=True)
    runtime_source = (
        source_root / "src" / "fastplms" / "models" / "toy" / "modeling_toy.py"
    )
    committed = runtime_source.read_bytes()

    def mutate_during_checkpoint_build(*args: object, **kwargs: object) -> object:
        runtime_source.write_bytes(b"MUTATED_DURING_BUILD = True\n")
        try:
            return canonicalize_checkpoint_weights(*args, **kwargs)
        finally:
            runtime_source.write_bytes(committed)

    monkeypatch.setattr(
        "tools.artifacts.build.canonicalize_checkpoint_weights",
        mutate_during_checkpoint_build,
    )

    artifact = build_artifact(
        spec,
        registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )

    assert (artifact / "fastplms" / "models" / "toy" / "modeling_toy.py").read_bytes() == (
        committed
    )


@pytest.mark.parametrize(
    ("field", "forged"),
    (
        ("checkpoint_license", "Forged-License"),
        ("legal_files", {}),
        ("upstreams", []),
    ),
)
def test_artifact_validation_rejects_self_attested_forged_legal_provenance(
    tmp_path: Path,
    field: str,
    forged: object,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    artifact = build_artifact(
        spec,
        registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )
    provenance_path = artifact / "source-record.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance[field] = forged
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = artifact / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source-record.json"] = f"sha256:{hash_file(provenance_path)}"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ArtifactError, match="differs from the current registry"):
        validate_artifact(artifact, spec=spec, registry=registry)


def test_dplm2_artifact_materializes_non_decoder_cache_config(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    source_config_path = checkpoint / "config.json"
    source_config_path.write_text(
        json.dumps(
            {
                "model_type": "toy",
                "is_decoder": True,
                "add_cross_attention": True,
                "use_cache": True,
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    source_files = tuple(
        FileDigest(item.path, item.algorithm, hash_file(source_config_path, item.algorithm))
        if item.path == "config.json"
        else item
        for item in spec.fast.files
    )
    source_config_digest = next(item for item in source_files if item.path == "config.json")
    official = replace(
        spec.official,
        files=source_files,
        repo_id="upstream/DPLM2Official",
        revision="5" * 40,
    )
    dplm2_family = replace(spec.family, id="dplm2", architecture="DPLM2")
    selected_spec = replace(
        spec,
        family=dplm2_family,
        official=official,
        artifact_source="official",
        canonical_state_sha256=_canonical_state_sha256(
            load_file(checkpoint / "model.safetensors")
        ),
    )
    selected_registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families={dplm2_family.id: dplm2_family},
        models={selected_spec.id: selected_spec},
        legal_files=registry.legal_files,
    )

    artifact = build_artifact(
        selected_spec,
        selected_registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )

    raw_config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    assert raw_config["is_decoder"] is False
    assert raw_config["add_cross_attention"] is False
    assert raw_config["use_cache"] is False
    provenance = json.loads((artifact / "source-record.json").read_text(encoding="utf-8"))
    assert provenance["artifact_checkpoint"]["files"]["config.json"] == (
        source_config_digest.encoded
    )
    assert provenance["official_checkpoint"]["files"]["config.json"] == (
        source_config_digest.encoded
    )
    assert hash_file(artifact / "config.json", source_config_digest.algorithm) != (
        source_config_digest.digest
    )


def test_non_dplm2_artifact_preserves_source_cache_fields(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    source_config_path = checkpoint / "config.json"
    source_values = {
        "model_type": "toy",
        "is_decoder": True,
        "add_cross_attention": True,
        "use_cache": True,
    }
    source_config_path.write_text(
        json.dumps(source_values) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fast = replace(
        spec.fast,
        files=tuple(
            FileDigest(item.path, item.algorithm, hash_file(source_config_path, item.algorithm))
            if item.path == "config.json"
            else item
            for item in spec.fast.files
        ),
    )
    selected_spec = replace(spec, fast=fast)
    selected_registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families=registry.families,
        models={selected_spec.id: selected_spec},
        legal_files=registry.legal_files,
    )

    artifact = build_artifact(
        selected_spec,
        selected_registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )

    raw_config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    assert {key: raw_config[key] for key in source_values} == source_values


def test_artifact_uses_manifest_selected_official_checkpoint(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    official = replace(
        spec.official,
        files=spec.fast.files,
        repo_id="upstream/ToyOfficial",
        revision="5" * 40,
    )
    selected_spec = replace(
        spec,
        official=official,
        artifact_source="official",
        canonical_state_sha256=_canonical_state_sha256(
            load_file(checkpoint / "model.safetensors")
        ),
    )
    selected_registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families=registry.families,
        models={selected_spec.id: selected_spec},
        legal_files=registry.legal_files,
    )

    artifact = build_artifact(
        selected_spec,
        selected_registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )
    provenance = json.loads((artifact / "source-record.json").read_text(encoding="utf-8"))
    assert provenance["artifact_source"] == "official"
    assert provenance["artifact_checkpoint"]["repo_id"] == "upstream/ToyOfficial"
    assert provenance["artifact_checkpoint"]["revision"] == "5" * 40
    assert provenance["canonical_weights"]["source_schema"] == "official"
    assert provenance["canonical_weights"]["state_digest"] == (
        provenance["conversion_equality_attestation"]["canonical_state"]
    )


def test_artifact_rejects_self_attested_forged_canonical_weight(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    official = replace(
        spec.official,
        files=spec.fast.files,
        repo_id="upstream/ToyOfficial",
        revision="5" * 40,
    )
    selected_spec = replace(
        spec,
        official=official,
        artifact_source="official",
        canonical_state_sha256=_canonical_state_sha256(
            load_file(checkpoint / "model.safetensors")
        ),
    )
    selected_registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families=registry.families,
        models={selected_spec.id: selected_spec},
        legal_files=registry.legal_files,
    )
    artifact = build_artifact(
        selected_spec,
        selected_registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )

    shard = next(artifact.glob("model-*.safetensors"))
    forged_state = load_file(shard)
    # forged_state['linear.bias']: (...)
    forged_state["linear.bias"] = forged_state["linear.bias"].clone()
    forged_state["linear.bias"][0] += 1
    save_file(forged_state, shard, metadata={"format": "pt"})
    forged_state_sha256 = _canonical_state_sha256(forged_state)

    provenance_path = artifact / "source-record.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["canonical_weights"]["shards"][shard.name] = (
        f"sha256:{hash_file(shard)}"
    )
    provenance["canonical_weights"]["state_digest"]["sha256"] = forged_state_sha256
    forged_spec = replace(selected_spec, canonical_state_sha256=forged_state_sha256)
    provenance["conversion_equality_attestation"] = _conversion_equality_attestation(
        forged_spec
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = artifact / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[shard.name] = f"sha256:{hash_file(shard)}"
    manifest["source-record.json"] = f"sha256:{hash_file(provenance_path)}"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactError,
        match="conversion equality attestation differs from the current registry",
    ):
        validate_artifact(
            artifact,
            spec=selected_spec,
            registry=selected_registry,
        )


def test_hash_pinned_bin_is_canonicalized_with_safe_loading(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    (checkpoint / "model.safetensors").unlink()
    bin_path = checkpoint / "pytorch_model.bin"
    torch.save(
        {
            "linear.bias": torch.arange(4, dtype=torch.float32),
            "linear.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        },
        bin_path,
    )
    config_digest = spec.fast.file_map["config.json"]
    bin_source = replace(
        spec.fast,
        files=(
            config_digest,
            FileDigest("pytorch_model.bin", "sha256", hash_file(bin_path)),
        ),
    )
    bin_spec = replace(spec, fast=bin_source)
    bin_registry = ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        families=registry.families,
        models={bin_spec.id: bin_spec},
        legal_files=registry.legal_files,
    )

    artifact = build_artifact(
        bin_spec,
        bin_registry,
        checkpoint,
        tmp_path / "artifact",
        source_root,
        _allow_untracked_runtime_for_tests=True,
    )
    validate_weight_artifact(artifact)
    assert not (artifact / "pytorch_model.bin").exists()
    assert list(artifact.glob("model-*.safetensors"))


def test_canonical_weight_sharding_and_index_validation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    checkpoint.mkdir()
    weight = checkpoint / "model.safetensors"
    save_file(
        {f"layer.{index}.weight": torch.arange(40, dtype=torch.float32) for index in range(4)},
        weight,
        metadata={"format": "pt"},
    )
    source = CheckpointSource(
        repo_id="Synthyra/ShardedToy",
        revision="6" * 40,
        files=(FileDigest("model.safetensors", "sha256", hash_file(weight)),),
    )

    record = canonicalize_checkpoint_weights(
        checkpoint,
        source,
        output,
        max_shard_bytes=512,
    )
    index = validate_weight_artifact(output, max_shard_bytes=512)
    assert len(record["shards"]) == 2
    assert len(set(index["weight_map"].values())) == 2
    assert all(path.stat().st_size <= 512 for path in output.glob("*.safetensors"))

    index["weight_map"].pop("layer.0.weight")
    (output / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ArtifactError, match="keys differ from the weight index"):
        validate_weight_artifact(output, max_shard_bytes=512)


def test_canonicalization_applies_declared_esm2_transform_before_sharding(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    checkpoint.mkdir()
    weight = checkpoint / "model.safetensors"
    source_state = {
        "embed_tokens.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "layers.0.self_attn.q_proj.weight": torch.tensor([[1.0, 2.0]]),
        "lm_head.weight": torch.tensor([[3.0, 4.0]]),
        "lm_head.bias": torch.tensor([5.0, 6.0]),
    }
    save_file(source_state, weight, metadata={"format": "pt"})
    source = CheckpointSource(
        repo_id="facebook/esm2-synthetic",
        revision="7" * 40,
        files=(FileDigest("model.safetensors", "sha256", hash_file(weight)),),
    )

    record = canonicalize_checkpoint_weights(
        checkpoint,
        source,
        output,
        state_transform="esm2_hf_to_fastplms_v1",
    )
    converted: dict[str, torch.Tensor] = {}
    for shard in sorted(output.glob("model-*.safetensors")):
        converted.update(load_file(shard, device="cpu"))

    assert set(converted) == {
        "esm.embeddings.word_embeddings.weight",
        "esm.encoder.layer.0.attention.self.query.weight",
        "lm_head.bias",
        "lm_head.decoder.weight",
    }
    assert torch.equal(converted["lm_head.bias"], source_state["lm_head.bias"])
    assert record["state_transform"] == "esm2_hf_to_fastplms_v1"


def test_canonical_esmfold_artifact_drops_only_declared_unused_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    checkpoint.mkdir()
    weight = checkpoint / "model.safetensors"
    source_state = {
        "esm.encoder.layer.0.weight": torch.tensor([1.0]),
        "esm.contact_head.regression.bias": torch.tensor([2.0]),
        "mlm_head.bias": torch.tensor([3.0]),
        "positional_encoding._float_tensor": torch.tensor([4.0]),
    }
    save_file(source_state, weight, metadata={"format": "pt"})
    source = CheckpointSource(
        repo_id="Synthyra/FastESMFold-synthetic",
        revision="8" * 40,
        files=(FileDigest("model.safetensors", "sha256", hash_file(weight)),),
    )

    canonicalize_checkpoint_weights(
        checkpoint,
        source,
        output,
        state_transform="esmfold_meta_to_fastplms_v1",
        source_is_canonical=True,
    )
    converted: dict[str, torch.Tensor] = {}
    for shard in sorted(output.glob("model-*.safetensors")):
        converted.update(load_file(shard, device="cpu"))

    assert set(converted) == {"esm.encoder.layer.0.weight"}


def test_official_submodule_worktrees_match_manifest_revisions() -> None:
    registry = load_model_registry()
    parser = configparser.ConfigParser(interpolation=None)
    assert parser.read(ROOT / ".gitmodules", encoding="utf-8")
    declared = {
        parser.get(section, "path"): parser.get(section, "url") for section in parser.sections()
    }
    expected = {source.path: source.url for source in registry.upstreams.values()}
    assert declared == expected

    # The portable remote runner deliberately strips every .git entry from its
    # source archive. Archive validation can still require the manifest-selected
    # source directories and exact .gitmodules declarations; a full checkout
    # additionally verifies the Git-link objects and worktree revisions below.
    if not (ROOT / ".git").exists():
        for source in registry.upstreams.values():
            checkout = ROOT / source.path
            assert checkout.is_dir()
            assert not (checkout / ".git").exists()
        return

    for source in registry.upstreams.values():
        checkout = ROOT / source.path
        gitlink = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={ROOT.as_posix()}",
                "-C",
                str(ROOT),
                "ls-files",
                "--stage",
                "--",
                source.path,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert gitlink.returncode == 0, gitlink.stderr
        mode, revision, stage_and_path = gitlink.stdout.strip().split(maxsplit=2)
        assert mode == "160000"
        assert revision == source.revision
        assert stage_and_path == f"0\t{source.path}"
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={checkout.as_posix()}",
                "-C",
                str(checkout),
                "rev-parse",
                "HEAD",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == source.revision


def test_artifact_build_rejects_dirty_official_source(tmp_path: Path) -> None:
    """A matching HEAD is insufficient when tracked oracle bytes were modified."""

    source_root = tmp_path / "source"
    checkout = source_root / "vendor" / "upstream" / "toy"
    checkout.mkdir(parents=True)
    subprocess.run(["git", "init", "--initial-branch=main"], cwd=source_root, check=True)
    subprocess.run(["git", "init", "--initial-branch=main"], cwd=checkout, check=True)
    tracked = checkout / "oracle.py"
    tracked.write_text("scale = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "oracle.py"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=FastPLMs Tests",
            "-c",
            "user.email=fastplms-tests@example.invalid",
            "commit",
            "-m",
            "Pin oracle",
        ],
        cwd=checkout,
        check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = SimpleNamespace(path="vendor/upstream/toy", revision=revision)
    registry = SimpleNamespace(upstreams={"toy": source})
    spec = SimpleNamespace(family=SimpleNamespace(upstreams=("toy",)))

    _validate_vendor_revisions(source_root, registry, spec)
    tracked.write_text("scale = 2\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="must have a clean worktree"):
        _validate_vendor_revisions(source_root, registry, spec)


def test_repository_legal_inventory_matches_manifest_digests() -> None:
    validate_repository_legal_inventory(ROOT, load_model_registry())
