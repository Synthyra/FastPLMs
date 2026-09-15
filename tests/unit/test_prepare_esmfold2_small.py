"""Focused contracts for local ESMFold2 small artifact preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastplms.registry import CheckpointSource, FileDigest, get_model_registry
from tools.artifacts.build import ArtifactError, hash_file, verify_checkpoint
from tools.artifacts.prepare_esmfold2_small import _materialize_config


def test_preparation_verifies_every_pinned_source_file(tmp_path: Path) -> None:
    source = tmp_path / "config.json"
    source.write_bytes(b"original")
    checkpoint = CheckpointSource(
        repo_id="biohub/example",
        revision="a" * 40,
        files=(FileDigest("config.json", "sha256", hash_file(source)),),
    )
    verify_checkpoint(tmp_path, checkpoint)
    source.write_bytes(b"altered")

    with pytest.raises(ArtifactError, match="Checkpoint verification failed"):
        verify_checkpoint(tmp_path, checkpoint)


@pytest.mark.parametrize(
    ("model_id", "expected_backbone"),
    (
        ("esmfold2_300", "Synthyra/ESMplusplus_small"),
        ("esmfold2_600", "Synthyra/ESMplusplus_large"),
    ),
)
def test_materialized_config_preserves_biological_fields_and_normalizes_backbone(
    model_id: str,
    expected_backbone: str,
) -> None:
    registry = get_model_registry()
    spec = registry[model_id]
    snapshot_name = "fold300" if model_id == "esmfold2_300" else "fold600"
    source = json.loads(
        (
            Path(__file__).parents[2]
            / "artifacts"
            / "esmfold2-small"
            / snapshot_name
            / "config.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    source["sentinel_biological_field"] = {"preserve": True}

    materialized = _materialize_config(source, spec, "b" * 64)

    assert materialized["sentinel_biological_field"] == {"preserve": True}
    assert materialized["esmc_id"] == expected_backbone
    assert materialized["msa_conditioning"] is False
    assert materialized["msa_encoder"]["enabled"] is False
    assert materialized["fastplms_runtime_bundle_sha256"] == "b" * 64
    assert materialized["auto_map"]["AutoModel"].startswith("modeling_fastplms.")
