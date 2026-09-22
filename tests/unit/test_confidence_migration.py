"""CPU-only cache and receipt checks for migrating confidence training."""

import hashlib
import json
import struct
import sys

import numpy as np
import pytest

from pathlib import Path
from types import SimpleNamespace

from safetensors.numpy import save_file

from tools.confidence import migration, target_splits


def cache_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[dict[str, object]]]:
    targets = [
        {"target_id": "validation-a", "split": "validation", "num_chains": 1},
        {"target_id": "validation-b", "split": "validation", "num_chains": 2},
    ]
    targets.sort(key=lambda target: hashlib.sha256(target["target_id"].encode()).hexdigest())
    monkeypatch.setattr(target_splits, "load_split", lambda _: list(reversed(targets)))
    directory = tmp_path / "runs/esmfold2_300/validation-cache-3-loops-50-steps-4-samples"
    directory.mkdir(parents=True)
    return directory, targets


def write_cache(path: Path, target: dict[str, object], samples: int = 4) -> None:
    save_file(
        {"x_pred": np.zeros((samples, 2, 3), dtype=np.float32)},
        str(path),
        metadata={"target_id": target["target_id"], "num_chains": str(target["num_chains"])},
    )


def test_partial_cache_uses_host_order_and_ignores_incomplete_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, targets = cache_fixture(tmp_path, monkeypatch)
    write_cache(directory / "0001.safetensors", targets[1])
    (directory / "0000.tmp").write_bytes(b"incomplete")
    report = migration.verify_validation_cache(tmp_path, "esmfold2_300")
    assert report["cached_targets"] == 1
    assert report["missing_targets"] == 1
    assert report["ignored_temporary_files"] == ["0000.tmp"]
    assert report["files"]["0001.safetensors"]["target_id"] == targets[1]["target_id"]
    assert len(report["files"]["0001.safetensors"]["header_sha256"]) == 64
    assert report["payload_hashes_verified"] is False


@pytest.mark.parametrize("filename", ["1.safetensors", "0002.safetensors"])
def test_cache_rejects_noncanonical_or_out_of_range_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    directory, targets = cache_fixture(tmp_path, monkeypatch)
    write_cache(directory / filename, targets[0])
    with pytest.raises(ValueError, match="filename|index"):
        migration.verify_validation_cache(tmp_path, "esmfold2_300")


def test_cache_rejects_a_valid_file_for_the_wrong_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, targets = cache_fixture(tmp_path, monkeypatch)
    write_cache(directory / "0000.safetensors", targets[1])
    with pytest.raises(ValueError, match="Cached target differs"):
        migration.verify_validation_cache(tmp_path, "esmfold2_300")


def test_cache_rejects_different_sampling_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, targets = cache_fixture(tmp_path, monkeypatch)
    write_cache(directory / "0000.safetensors", targets[0], samples=1)
    with pytest.raises(ValueError, match="sample shape"):
        migration.verify_validation_cache(tmp_path, "esmfold2_300")


@pytest.mark.parametrize("contents", [b"short", struct.pack("<Q", 64) + b"{}", struct.pack("<Q", migration.MAX_HEADER_BYTES + 1)])
def test_cache_rejects_truncated_or_unbounded_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: bytes
) -> None:
    directory, _ = cache_fixture(tmp_path, monkeypatch)
    (directory / "0000.safetensors").write_bytes(contents)
    with pytest.raises(ValueError, match="header"):
        migration.verify_validation_cache(tmp_path, "esmfold2_300")


def test_cache_rejects_truncated_coordinate_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, targets = cache_fixture(tmp_path, monkeypatch)
    path = directory / "0000.safetensors"
    write_cache(path, targets[0])
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match="offsets exceed"):
        migration.verify_validation_cache(tmp_path, "esmfold2_300")


def test_missing_checkpoint_does_not_import_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    report = migration.checkpoint_identity(tmp_path, "esmfold2_300")
    assert report["checkpoint_present"] is False
    assert report["update"] == 0


def test_existing_checkpoint_requires_trust_and_loads_only_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "runs/esmfold2_300/v2"
    directory.mkdir(parents=True)
    for name in ("last.pt", "best-ema.safetensors", "wandb-id.txt"):
        (directory / name).write_text(name)
    with pytest.raises(ValueError, match="explicit trust"):
        migration.checkpoint_identity(tmp_path, "esmfold2_300")
    loaded = []
    state = {
        "config": {"model_id": "esmfold2_300"}, "update": 23, "elapsed": 500.0,
        "head": {}, "optimizer": {}, "ema": {}, "history": [], "best": {}, "rng": {},
    }

    def load(path, **kwargs):
        loaded.append(kwargs)
        return state

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load))
    report = migration.checkpoint_identity(tmp_path, "esmfold2_300", trusted=True)
    assert loaded == [{"map_location": "cpu", "weights_only": False}]
    assert report["update"] == 23
    assert len(report["file"]["sha256"]) == 64
    assert set(report["companions"]) == {"best-ema.safetensors", "wandb-id.txt"}


def test_migration_receipt_requires_stopped_writer_and_rejects_duplicate(tmp_path: Path) -> None:
    arguments = {
        "old_gpu": "H200", "new_gpu": "B200", "old_call_id": "fc-old123",
        "new_call_id": "fc-new456", "old_call_status": "RUNNING",
        "checkpoint": {"model_id": "esmfold2_300", "checkpoint_present": False, "update": 0},
        "validation_cache": {"model_id": "esmfold2_300", "cached_targets": 20},
        "source_files": {"online_training.py": {"sha256": "a" * 64}},
    }
    with pytest.raises(ValueError, match="confirmed terminal"):
        migration.write_migration_receipt(tmp_path, "esmfold2_300", **arguments)
    arguments["old_call_status"] = "TERMINATED"
    path = migration.write_migration_receipt(tmp_path, "esmfold2_300", **arguments)
    receipt = json.loads(path.read_text())
    assert receipt["old_call_id"] == "fc-old123"
    assert receipt["new_call_id"] == "fc-new456"
    assert receipt["checkpoint"]["update"] == 0
    with pytest.raises(FileExistsError):
        migration.write_migration_receipt(tmp_path, "esmfold2_300", **arguments)
