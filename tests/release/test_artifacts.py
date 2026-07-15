from __future__ import annotations

import configparser
import hashlib
import json
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
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
from tools.artifacts.build import _checkpoint_identity_hash, _validate_vendor_revisions

ROOT = Path(__file__).resolve().parents[2]


def _canonical_text_sha256(path: Path) -> str:
    raw = path.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def test_shared_sources_are_in_runtime_artifacts() -> None:
    """Keep remote-code artifacts closed over their package-source imports."""

    registry = get_model_registry()
    required = {
        "esm2": {"models/_esm_rotary.py"},
        "dplm": {"models/_diffusion_generation.py", "models/_esm_rotary.py"},
        "dplm2": {"models/_diffusion_generation.py", "models/_esm_rotary.py"},
        "esmfold": {"models/_esm_rotary.py"},
        "esmfold2": {"models/esm_plusplus"},
    }
    package_root = ROOT / "src" / "fastplms"
    for family_id, paths in required.items():
        family = registry.families[family_id]
        assert paths.issubset(family.runtime_paths)
        for relative_path in paths:
            assert (package_root / relative_path).exists()


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


def _synthetic_registry(source_root: Path, checkpoint: Path) -> tuple[ModelRegistry, ModelSpec]:
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


def test_artifact_build_is_deterministic_and_self_verifying(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)

    first = build_artifact(spec, registry, checkpoint, tmp_path / "first", source_root)
    second = build_artifact(spec, registry, checkpoint, tmp_path / "second", source_root)
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
    assert config["fastplms_checkpoint_hash"] == _checkpoint_identity_hash(
        spec.artifact_checkpoint
    )
    assert (first / "fastplms" / "models" / "toy" / "modeling_toy.py").is_file()
    assert (first / "fastplms_bundle.py").is_file()
    bridge = (first / "modeling_fastplms.py").read_text(encoding="utf-8")
    assert "from .fastplms_bundle import RUNTIME_DATA, RUNTIME_HASH" in bridge
    assert "from .fastplms." not in bridge
    assert not (first / "vendor").exists()
    assert (first / "LICENSES" / "toy" / "LICENSE").is_file()
    assert (first / "THIRD_PARTY_NOTICES.md").is_file()
    assert (first / "model.safetensors.index.json").is_file()
    assert not (first / "model.safetensors").exists()
    assert len(list(first.glob("model-*.safetensors"))) == 1
    provenance = json.loads((first / "provenance.json").read_text(encoding="utf-8"))
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


def test_different_runtime_bundles_fail_without_replacing_loaded_runtime(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)

    first = build_artifact(spec, registry, checkpoint, tmp_path / "first", source_root)
    runtime_source = source_root / "src" / "fastplms" / "models" / "toy" / "modeling_toy.py"
    runtime_source.write_text(
        runtime_source.read_text(encoding="utf-8") + "\n# Distinct runtime identity.\n",
        encoding="utf-8",
        newline="\n",
    )
    second = build_artifact(spec, registry, checkpoint, tmp_path / "second", source_root)

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
                if "different FastPLMs runtime" not in str(error):
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
    assert {"LICENSE", "MODIFICATIONS.md", "PROVENANCE.md"}.issubset(distribution["openfold"])


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
        build_artifact(spec, registry, checkpoint, tmp_path / "artifact", source_root)


def test_artifact_copies_official_tokenizer_bytes_exactly(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    official_snapshot = tmp_path / "official"
    checkpoint.mkdir()
    official_snapshot.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)

    candidate_tokenizer = checkpoint / "tokenizer.json"
    official_tokenizer = official_snapshot / "tokenizer.json"
    candidate_tokenizer.write_bytes(b'{"source":"candidate"}\n')
    official_tokenizer.write_bytes(b'{"source":"official"}\n')
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
    )
    assert (artifact / "tokenizer.json").read_bytes() == official_tokenizer.read_bytes()
    provenance = json.loads((artifact / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["tokenizer_checkpoint"]["repo_id"] == official.repo_id
    assert provenance["tokenizer_checkpoint"]["revision"] == official.revision


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
    )
    provenance = json.loads((artifact / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["artifact_source"] == "official"
    assert provenance["artifact_checkpoint"]["repo_id"] == "upstream/ToyOfficial"
    assert provenance["artifact_checkpoint"]["revision"] == "5" * 40
    assert provenance["canonical_weights"]["source_schema"] == "official"


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
        "lm_head.decoder.bias",
        "lm_head.decoder.weight",
    }
    assert torch.equal(converted["lm_head.bias"], source_state["lm_head.bias"])
    assert torch.equal(converted["lm_head.decoder.bias"], source_state["lm_head.bias"])
    assert record["state_transform"] == "esm2_hf_to_fastplms_v1"


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
