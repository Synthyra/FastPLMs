"""Mandatory fail-closed artifact and publication security contracts."""

import hashlib
import json
import pytest
import torch
from dataclasses import replace
from pathlib import Path
from typing import Any
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from safetensors.torch import save_file

from fastplms.registry import (
    CheckpointSource,
    FileDigest,
    ModelRegistry,
    ModelSpec,
    get_model_registry,
)
from tests.release import test_artifacts as artifact_contracts
from tests.release import test_publish_files_only as publish_contracts
from tools.artifacts import ArtifactError, hash_file
from tools.artifacts import publish as publish_module
from tools.artifacts.build import (
    _canonical_state_sha256,
    _content_manifest,
    _provenance,
    _runtime_attestation,
    validate_artifact,
    validate_weight_artifact,
)


# Register the release module's immutable source fixture when these contracts are
# collected through this CPU-only allowlist module.
_clean_publication_source = publish_contracts._clean_publication_source

test_artifact_build_rejects_unknown_runtime_source_extension = (
    artifact_contracts.test_artifact_build_rejects_unknown_runtime_source_extension
)
test_artifact_build_rejects_untracked_runtime_source = (
    artifact_contracts.test_artifact_build_rejects_untracked_runtime_source
)
test_artifact_validation_rejects_unsafe_manifest_paths = (
    artifact_contracts.test_artifact_validation_rejects_unsafe_manifest_paths
)
test_artifact_validation_rejects_self_attested_forged_legal_provenance = (
    artifact_contracts.test_artifact_validation_rejects_self_attested_forged_legal_provenance
)
test_artifact_build_uses_validated_git_blobs_after_worktree_mutation = (
    artifact_contracts.test_artifact_build_uses_validated_git_blobs_after_worktree_mutation
)
test_artifact_build_rejects_concurrent_config_replacement = (
    artifact_contracts.test_artifact_build_rejects_concurrent_config_replacement
)
test_official_tokenizer_copy_rejects_concurrent_replacement = (
    artifact_contracts.test_official_tokenizer_copy_rejects_concurrent_replacement
)
test_weight_snapshot_rejects_concurrent_replacement = (
    artifact_contracts.test_weight_snapshot_rejects_concurrent_replacement
)
test_artifact_rejects_self_attested_forged_canonical_weight = (
    artifact_contracts.test_artifact_rejects_self_attested_forged_canonical_weight
)
test_dplm2_artifact_materializes_non_decoder_cache_config = (
    artifact_contracts.test_dplm2_artifact_materializes_non_decoder_cache_config
)
test_complete_publication_rejects_synthetic_unresolved_checkpoint_license = (
    publish_contracts.test_complete_publication_rejects_synthetic_unresolved_checkpoint_license
)
test_complete_ankh_publish_rejects_missing_probe_binding = (
    publish_contracts.test_complete_ankh_publish_rejects_missing_probe_binding
)
test_complete_plan_rejects_unpinned_competing_remote_weight = (
    publish_contracts.test_complete_plan_rejects_unpinned_competing_remote_weight
)
test_complete_publish_rejects_hand_built_unknown_plan = (
    publish_contracts.test_complete_publish_rejects_hand_built_unknown_plan
)
test_complete_publish_rehashes_every_file_before_atomic_commit = (
    publish_contracts.test_complete_publish_rehashes_every_file_before_atomic_commit
)
test_files_only_plan_rejects_forged_registry_provenance = (
    publish_contracts.test_files_only_plan_rejects_forged_registry_provenance
)
test_files_only_plan_rejects_self_attested_invalid_bundle_data = (
    publish_contracts.test_files_only_plan_rejects_self_attested_invalid_bundle_data
)
test_files_only_plan_rejects_self_attested_stale_release_text = (
    publish_contracts.test_files_only_plan_rejects_self_attested_stale_release_text
)
test_files_only_plan_rejects_self_attested_substituted_bundle_member = (
    publish_contracts.test_files_only_plan_rejects_self_attested_substituted_bundle_member
)
test_files_only_plan_rejects_stale_runtime_revision = (
    publish_contracts.test_files_only_plan_rejects_stale_runtime_revision
)
test_files_only_plan_rejects_unknown_manifest_path = (
    publish_contracts.test_files_only_plan_rejects_unknown_manifest_path
)
test_files_only_plan_rejects_unlisted_and_sensitive_files = (
    publish_contracts.test_files_only_plan_rejects_unlisted_and_sensitive_files
)
test_files_only_plan_supports_every_manifest_model = (
    publish_contracts.test_files_only_plan_supports_every_manifest_model
)
test_files_only_publish_uses_preflighted_bytes_after_local_mutation = (
    publish_contracts.test_files_only_publish_uses_preflighted_bytes_after_local_mutation
)
test_files_only_publish_rejects_release_text_change_after_preflight = (
    publish_contracts.test_files_only_publish_rejects_release_text_change_after_preflight
)
test_files_only_publish_rejects_source_change_after_preflight = (
    publish_contracts.test_files_only_publish_rejects_source_change_after_preflight
)
test_release_snapshot_rejects_untracked_card_symlink_to_tracked_file = (
    publish_contracts.test_release_snapshot_rejects_untracked_card_symlink_to_tracked_file
)
test_required_complete_probe_groups_ankh_views = (
    publish_contracts.test_required_complete_probe_groups_ankh_views
)


def _synthetic_complete_ankh_registry(
    artifact: Path,
    shard_states: tuple[dict[str, torch.Tensor], ...],
    tokenizer_payloads: dict[str, bytes],
) -> tuple[ModelRegistry, ModelSpec, dict[str, Any]]:
    current_registry = get_model_registry()
    current_spec = current_registry["ankh_base"]
    shards = tuple(
        artifact / f"model-{index:05d}-of-{len(shard_states):05d}.safetensors"
        for index in range(1, len(shard_states) + 1)
    )
    for path, state in zip(shards, shard_states, strict=True):
        save_file(state, path, metadata={"format": "pt"})

    weight_map = {
        key: path.name for path, state in zip(shards, shard_states, strict=True) for key in state
    }
    total_size = sum(
        tensor.numel() * tensor.element_size()
        for state in shard_states
        for tensor in state.values()
    )
    index = artifact / "model.safetensors.index.json"
    publish_contracts._write_json(
        index,
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    combined_state = {key: tensor for state in shard_states for key, tensor in state.items()}
    canonical_state_sha256 = _canonical_state_sha256(combined_state)
    canonical_weights: dict[str, Any] = {
        "format": "safetensors",
        "index": index.name,
        "index_digest": f"sha256:{hash_file(index)}",
        "max_shard_bytes": 1024,
        "shards": {path.name: f"sha256:{hash_file(path)}" for path in shards},
        "source_schema": "official",
        "state_transform": current_spec.family.state_transform,
        "state_digest": {
            "schema_version": 1,
            "algorithm": "sha256",
            "sha256": canonical_state_sha256,
        },
        "tensor_count": len(weight_map),
        "total_size": total_size,
    }
    official_files = [
        FileDigest(path=index.name, algorithm="sha256", digest=hash_file(index)),
        *(
            FileDigest(path=path.name, algorithm="sha256", digest=hash_file(path))
            for path in shards
        ),
        *(
            FileDigest(
                path=relative_name,
                algorithm="sha256",
                digest=hashlib.sha256(payload).hexdigest(),
            )
            for relative_name, payload in sorted(tokenizer_payloads.items())
        ),
    ]
    synthetic_spec = replace(
        current_spec,
        family=replace(
            current_spec.family,
            requires_complete_weight_publication=True,
        ),
        official=CheckpointSource(
            repo_id=current_spec.official.repo_id,
            revision=current_spec.official.revision,
            files=tuple(official_files),
        ),
        canonical_state_sha256=canonical_state_sha256,
    )
    models = dict(current_registry)
    models[synthetic_spec.id] = synthetic_spec
    synthetic_registry = ModelRegistry(
        schema_version=current_registry.schema_version,
        upstreams=current_registry.upstreams,
        families=current_registry.families,
        models=models,
        runtime_assets=current_registry.runtime_assets,
        attention_kernels=current_registry.attention_kernels,
        legal_files=current_registry.legal_files,
    )
    return synthetic_registry, synthetic_spec, canonical_weights


def _complete_ankh_artifact(
    root: Path,
    registry: ModelRegistry,
    spec: ModelSpec,
    canonical_weights: dict[str, Any],
    prepared_weights: Path,
    tokenizer_payloads: dict[str, bytes],
) -> Path:
    artifact = publish_contracts._files_only_artifact(root, spec)
    stale_shard = artifact / "model-00001-of-00001.safetensors"
    stale_shard.unlink()
    for relative_name in (
        "model.safetensors.index.json",
        *canonical_weights["shards"],
    ):
        source = prepared_weights / relative_name
        (artifact / relative_name).write_bytes(source.read_bytes())
    for relative_name, payload in tokenizer_payloads.items():
        (artifact / relative_name).write_bytes(payload)

    seed_provenance = json.loads((artifact / "source-record.json").read_text(encoding="utf-8"))
    runtime_revision = seed_provenance["runtime_revision"]
    source_tree_sha256 = seed_provenance["source_tree_sha256"]
    runtime_bundle_sha256 = seed_provenance["runtime_bundle_sha256"]
    release_tool_revision = seed_provenance["release_tool_revision"]
    release_tool_sha256 = seed_provenance["release_tool_sha256"]
    provenance = _provenance(
        registry,
        spec,
        canonical_weights,
        runtime_revision=runtime_revision,
        source_tree_sha256=source_tree_sha256,
        runtime_bundle_sha256=runtime_bundle_sha256,
        release_tool_revision=release_tool_revision,
        release_tool_sha256=release_tool_sha256,
    )
    publish_contracts._write_json(artifact / "source-record.json", provenance)
    runtime_attestation = _runtime_attestation(
        artifact,
        spec,
        weights_revision=spec.fast.revision,
        runtime_revision=runtime_revision,
        source_tree_sha256=source_tree_sha256,
        runtime_bundle_sha256=runtime_bundle_sha256,
        release_tool_revision=release_tool_revision,
        release_tool_sha256=release_tool_sha256,
    )
    publish_contracts._write_json(
        artifact / "runtime-attestation.json",
        runtime_attestation,
    )
    publish_contracts._write_json(
        artifact / "artifact-manifest.json",
        _content_manifest(artifact),
    )
    return artifact


def test_complete_ankh_multishard_inventory_is_published_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_weights = tmp_path / "prepared-weights"
    prepared_weights.mkdir()
    shard_states = (
        {
            "shared.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),  # (vocab, d)
            "encoder.block.0.layer.0.SelfAttention.q.weight": torch.ones(4, 4),  # (d, d)
        },
        {
            "decoder.block.0.layer.1.EncDecAttention.q.weight": torch.ones(4, 4),  # (d, d)
            "lm_head.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),  # (vocab, d)
        },
    )
    tokenizer_payloads = {
        "special_tokens_map.json": b'{"eos_token":"</s>","pad_token":"<pad>"}\n',
        "tokenizer.json": b'{"version":"1.0"}\n',
        "tokenizer_config.json": b'{"eos_token":"</s>","pad_token":"<pad>"}\n',
    }
    registry, spec, canonical_weights = _synthetic_complete_ankh_registry(
        prepared_weights,
        shard_states,
        tokenizer_payloads,
    )
    assert spec.family.requires_complete_weight_publication is True
    assert spec.artifact_source == "official"
    monkeypatch.setattr(publish_module, "get_model_registry", lambda: registry)
    monkeypatch.setattr(publish_contracts, "get_model_registry", lambda: registry)
    artifact = _complete_ankh_artifact(
        tmp_path,
        registry,
        spec,
        canonical_weights,
        prepared_weights,
        tokenizer_payloads,
    )
    validate_artifact(artifact, spec=spec, registry=registry)
    validated_index = validate_weight_artifact(artifact)
    replacement_weights = {
        "model.safetensors.index.json",
        *canonical_weights["shards"],
    }
    assert set(validated_index["weight_map"].values()) == (
        replacement_weights - {"model.safetensors.index.json"}
    )

    api = publish_contracts.FakeApi(spec)
    probe_calls: list[tuple[str, Path, str]] = []

    def required_autoclass_probe(probe_spec: ModelSpec, probe_artifact: Path) -> tuple[str, ...]:
        config = json.loads((probe_artifact / "config.json").read_text(encoding="utf-8"))
        auto_map = config["auto_map"]
        assert auto_map["AutoModel"].endswith(".FastAnkhModel")
        assert auto_map["AutoModelForSeq2SeqLM"].endswith(".FastAnkhForConditionalGeneration")
        probe_calls.append(
            (
                probe_spec.id,
                probe_artifact,
                hash_file(probe_artifact / "artifact-manifest.json"),
            )
        )
        return ("AutoModel", "AutoModelForSeq2SeqLM")

    monkeypatch.setattr(
        publish_module,
        "_run_required_complete_autoclass_probe",
        required_autoclass_probe,
    )
    real_prepare_complete_plan = publish_module.prepare_complete_plan
    prepared_plans = []

    def recording_prepare_complete_plan(*args: Any, **kwargs: Any) -> Any:
        prepared_plan = real_prepare_complete_plan(*args, **kwargs)
        prepared_plans.append(prepared_plan)
        return prepared_plan

    monkeypatch.setattr(
        publish_module,
        "prepare_complete_plan",
        recording_prepare_complete_plan,
    )
    plan = publish_module.prepare_complete_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,
    )
    assert prepared_plans == [plan]
    assert plan.validated_auto_classes == ("AutoModel", "AutoModelForSeq2SeqLM")
    assert set(plan.replacement_weight_paths) == replacement_weights
    assert replacement_weights.issubset(plan.files)
    assert plan.deletes == ("model-00001-of-00001.safetensors",)

    publish_module.publish_complete(
        (plan,),
        api=api,  # type: ignore[arg-type]
        commit_message="Atomic ANKH multi-shard publication",
    )

    assert prepared_plans == [plan, plan]
    assert len(probe_calls) == 2
    assert probe_calls[0] == probe_calls[1]
    assert len(api.create_commit_calls) == 1
    call = api.create_commit_calls[0]
    assert call["parent_commit"] == "a" * 40
    operations = call["operations"]
    additions = {
        operation.path_in_repo
        for operation in operations
        if isinstance(operation, CommitOperationAdd)
    }
    deletions = {
        operation.path_in_repo
        for operation in operations
        if isinstance(operation, CommitOperationDelete)
    }
    assert additions == set(plan.files)
    assert replacement_weights.issubset(additions)
    assert deletions == {"model-00001-of-00001.safetensors"}

    (artifact / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ArtifactError, match="index and artifact shard files differ"):
        validate_weight_artifact(artifact)
