"""Frozen upload inventories exclude credentials and identify dirty source exactly."""

from __future__ import annotations

import hashlib
import json
import pytest

from pathlib import Path

from tools.execution.budget import BudgetLedger
from tools.execution.source import excluded_from_upload, stage_source_snapshot


def test_source_snapshot_hashes_copied_bytes_and_survives_worktree_edits(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "src").mkdir(parents=True)
    source = repository / "src/model.py"
    source.write_bytes(b"dirty source\n")
    (repository / "src/.env").write_bytes(b"must remain local")
    (repository / "README.md").write_bytes(b"source instructions\n")

    snapshot = stage_source_snapshot(
        repository,
        tmp_path / "snapshot",
        directories=("src",),
        files=("README.md",),
    )
    record = snapshot.to_dict()
    assert record["file_count"] == 2
    assert record["files"] == [
        {
            "path": "README.md",
            "size": 20,
            "sha256": hashlib.sha256(b"source instructions\n").hexdigest(),
        },
        {
            "path": "src/model.py",
            "size": 13,
            "sha256": hashlib.sha256(b"dirty source\n").hexdigest(),
        },
    ]
    assert str(repository) not in json.dumps(record)
    source.write_bytes(b"later worktree edit\n")
    assert (snapshot.root / "src/model.py").read_bytes() == b"dirty source\n"
    assert not (snapshot.root / "src/.env").exists()
    assert (
        snapshot.tree_sha256
        != stage_source_snapshot(
            repository,
            tmp_path / "later",
            directories=("src",),
            files=("README.md",),
        ).tree_sha256
    )


@pytest.mark.parametrize("name", [".env/config.py", "credentials/data.py", "id_rsa", "x.pem/data"])
def test_nested_credential_paths_are_excluded(name: str) -> None:
    assert excluded_from_upload(Path(name))


def test_custom_exclusion_cannot_enable_credential_upload(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/.env").write_bytes(b"excluded")
    snapshot = stage_source_snapshot(
        tmp_path,
        tmp_path / "snapshot",
        directories=("src",),
        files=(),
        exclude=lambda _: False,
    )
    assert snapshot.files == ()


@pytest.mark.parametrize("directory_link", [False, True])
def test_snapshot_refuses_external_symlinks_before_reading(
    tmp_path: Path,
    directory_link: bool,
) -> None:
    repository = tmp_path / "repository"
    (repository / "src").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.py").write_bytes(b"outside source")
    try:
        (repository / "src/link").symlink_to(
            outside if directory_link else outside / "payload.py",
            target_is_directory=directory_link,
        )
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")
    with pytest.raises(RuntimeError, match="symlink"):
        stage_source_snapshot(repository, tmp_path / "snapshot", directories=("src",), files=())
    assert not (tmp_path / "snapshot").exists()


def test_snapshot_refuses_reuse_and_recursive_destination(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    destination = tmp_path / "snapshot"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        stage_source_snapshot(tmp_path, destination, directories=("src",), files=())
    with pytest.raises(ValueError, match="outside selected inputs"):
        stage_source_snapshot(tmp_path, tmp_path / "src/copy", directories=("src",), files=())


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -1.0])
def test_nonfinite_ledger_cannot_bypass_budget_checks(tmp_path: Path, cost: float) -> None:
    path = tmp_path / "budget.json"
    path.write_text(json.dumps([{"stage": "test", "reserved_dollars": cost}]))
    with pytest.raises(ValueError, match="nonnegative and finite"):
        BudgetLedger(path).committed()


def test_generic_ledger_does_not_invent_confidence_stages(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    reservation = ledger.reserve("source-check", "CPU tests", 2.0)
    assert ledger.summary()["stages"] == {"source-check": 2.0}
    assert BudgetLedger(ledger.path).committed() == 2.0
    ledger.complete(reservation, 0.5)
    assert ledger.summary()["stages"] == {"source-check": 0.5}
