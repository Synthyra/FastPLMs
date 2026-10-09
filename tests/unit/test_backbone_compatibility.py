"""Content audits distinguish storage precision from effective execution weights."""

from __future__ import annotations

import json
import pytest
import torch

from pathlib import Path
from safetensors.torch import save_file

from fastplms.registry import CheckpointSource, FileDigest
from tools.artifacts import backbone_compatibility as audit
from tools.artifacts.build import ArtifactError, hash_file


def state() -> dict[str, torch.Tensor]:
    return {  # (...) embed.weight (2, 3), transformer.norm.weight (3,), sequence_head.0.weight (3, 3)
        "embed.weight": torch.ones(2, 3),
        "transformer.norm.weight": torch.ones(3),
        "sequence_head.0.weight": torch.zeros(3, 3),
    }


def source(snapshot: Path) -> CheckpointSource:
    save_file(state(), snapshot / "model.safetensors")
    (snapshot / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
    (snapshot / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    return CheckpointSource(
        "fixture/model",
        "a" * 40,
        tuple(
            FileDigest(name, "sha256", hash_file(snapshot / name))
            for name in ("config.json", "tokenizer.json", "model.safetensors")
        ),
    )


def test_equal_logical_states_have_equal_digest_despite_dictionary_order():
    first = state()
    second = dict(reversed(list(state().items())))
    comparison = audit.compare_states(first, second)
    assert comparison["exactly_equal"]
    assert comparison["left_state_sha256"] == comparison["right_state_sha256"]


def test_head_difference_does_not_masquerade_as_encoder_difference():
    first, second = state(), state()
    second["sequence_head.0.weight"][0, 0] = 1
    assert not audit.compare_states(first, second)["exactly_equal"]
    assert audit.compare_states(audit.encoder_state(first), audit.encoder_state(second))[
        "exactly_equal"
    ]


def test_encoder_difference_is_measured_over_actual_tensors():
    first, second = state(), state()
    second["embed.weight"][0, 0] += 0.25
    compared = audit.compare_states(first, second)
    assert not compared["exactly_equal"]
    assert compared["changed_tensor_count"] == 1
    assert compared["changed_tensor_examples"] == ["embed.weight"]
    assert compared["maximum_absolute_value_difference"] == 0.25
    assert compared["left_state_sha256"] != compared["right_state_sha256"]
    assert not compared["bf16_execution"]["exactly_equal"]


def test_bf16_rounding_is_not_a_different_execution_backbone():
    first = {name: tensor + 0.001 for name, tensor in state().items()}
    second = {name: tensor.bfloat16().float() for name, tensor in first.items()}
    compared = audit.compare_states(first, second)
    assert not compared["exactly_equal"]
    assert compared["left_state_sha256"] != compared["right_state_sha256"]
    assert compared["bf16_execution"] == {
        "dtype": "bfloat16",
        "exactly_equal": True,
        "changed_tensor_count": 0,
        "maximum_absolute_value_difference": 0.0,
        "left_exact_bf16_roundtrip": False,
        "right_exact_bf16_roundtrip": True,
    }


def test_bf16_execution_may_agree_across_storage_dtypes():
    first = state()
    second = {name: tensor.bfloat16() for name, tensor in first.items()}
    compared = audit.compare_states(first, second)
    assert not compared["exactly_equal"]
    assert compared["metadata_changed"]
    assert compared["bf16_execution"]["exactly_equal"]


@pytest.mark.parametrize("change", ["dtype", "shape", "missing", "extra"])
def test_metadata_and_inventory_changes_cannot_pass(change):
    first, second = state(), state()
    if change == "dtype":
        second["embed.weight"] = second["embed.weight"].to(torch.bfloat16)
    elif change == "shape":
        second["embed.weight"] = torch.ones(1, 6)
    elif change == "missing":
        del second["embed.weight"]
    else:
        second["extra"] = torch.zeros(1)
    comparison = audit.compare_states(first, second)
    assert not comparison["exactly_equal"]
    assert comparison["bf16_execution"]["exactly_equal"] == (change == "dtype")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_state_fails(invalid):
    bad = state()
    bad["embed.weight"][0, 0] = invalid
    with pytest.raises(ValueError, match="nonfinite"):
        audit.compare_states(state(), bad)


def test_cast_overflow_cannot_pass_as_equal_bf16_weights():
    bad = {"embed.weight": torch.tensor([1e100], dtype=torch.float64)}
    with pytest.raises(ValueError, match="nonfinite BF16"):
        audit.compare_states(bad, bad)


def test_empty_and_unknown_encoder_states_fail():
    with pytest.raises(ValueError, match="empty"):
        audit.compare_states({}, {})
    with pytest.raises(ValueError, match="incomplete"):
        audit.encoder_state({"sequence_head.0.weight": torch.ones(1)})
    with pytest.raises(ValueError, match="namespace"):
        audit.encoder_state({**state(), "surprise": torch.ones(1)})


def test_acquisition_is_pinned_allowlisted_and_verified(monkeypatch, tmp_path):
    pinned = source(tmp_path)
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(audit, "snapshot_download", download)
    snapshot, record = audit.acquire(pinned, tmp_path / "cache")
    assert snapshot == tmp_path
    assert calls == [
        (
            pinned.repo_id,
            {
                "revision": pinned.revision,
                "cache_dir": tmp_path / "cache",
                "allow_patterns": [file.path for file in pinned.files],
            },
        )
    ]
    assert record["revision"] == pinned.revision
    loaded, conversion = audit.canonical_state(snapshot, pinned)
    assert conversion == "esmc_to_fastplms_v1"
    assert audit.compare_states(loaded, state())["exactly_equal"]
    (snapshot / "config.json").write_text("{}")
    with pytest.raises(ArtifactError, match="verification failed"):
        audit.acquire(pinned, tmp_path / "cache")


def test_transport_failure_is_not_a_negative_compatibility_verdict(monkeypatch, tmp_path):
    pinned = source(tmp_path)

    def fail(*args, **kwargs):
        raise ConnectionError("fixture transport failure")

    monkeypatch.setattr(audit, "snapshot_download", fail)
    with pytest.raises(ConnectionError, match="transport"):
        audit.acquire(pinned, tmp_path / "cache")


def test_executable_or_unresolved_sources_fail_before_download(tmp_path):
    python_source = CheckpointSource(
        "fixture/model",
        "a" * 40,
        (
            FileDigest("model.safetensors", "sha256", "b" * 64),
            FileDigest("model.py", "sha256", "c" * 64),
        ),
    )
    with pytest.raises(ValueError, match="only safetensors and JSON"):
        audit.acquire(python_source, tmp_path)
    with pytest.raises(ValueError, match="fully pinned"):
        audit.acquire(
            CheckpointSource("fixture/model", "a" * 40, (), ("model.safetensors",)), tmp_path
        )


def test_existing_receipt_is_preserved(tmp_path):
    output = tmp_path / "receipt.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError):
        audit.audit("esmc_small", tmp_path / "cache", output)
    assert output.read_text() == "preserve"


@pytest.mark.parametrize("changed", [False, True])
def test_receipt_separates_storage_from_execution_and_keeps_live_parity_unqualified(monkeypatch, tmp_path, changed):
    source(tmp_path)
    first = {name: tensor + 0.001 for name, tensor in state().items()}
    rounded = {name: tensor.bfloat16().float() for name, tensor in first.items()}
    if changed:
        rounded["embed.weight"][0, 0] += 0.125

    def acquire(pinned, cache):
        return tmp_path, {"repo": pinned.repo_id, "revision": pinned.revision}

    def canonical(snapshot, pinned):
        return (rounded if pinned.repo_id.endswith("-1500000") else first), "fixture"

    monkeypatch.setattr(audit, "acquire", acquire)
    monkeypatch.setattr(audit, "canonical_state", canonical)
    output = tmp_path / "audit.json"
    receipt = audit.audit("esmc_small", tmp_path, output)
    assert receipt["schema_version"] == 2
    assert receipt["status"] == "complete"
    assert receipt["identical_encoder_weights"] is False
    assert receipt["identical_encoder_weights_bf16"] is (not changed)
    assert receipt["qualified_for_shared_inference"] is False
    assert json.loads(output.read_text()) == receipt
