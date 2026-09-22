"""CPU checks for release inventory and publication preconditions."""

from __future__ import annotations

import json

import pytest

from tools.confidence.release import _inventory, prepare_release


def test_inventory_is_sorted_and_hashed(tmp_path) -> None:
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    inventory = _inventory(tmp_path)
    assert [item["path"] for item in inventory] == ["a.txt", "z.txt"]
    assert all(item["size"] == 1 and len(item["sha256"]) == 64 for item in inventory)


def test_prepare_release_requires_passed_package_and_eval(tmp_path) -> None:
    model_root = tmp_path / "esmfold2_300"
    (model_root / "package").mkdir(parents=True)
    (model_root / "evaluate").mkdir()
    (model_root / "train").mkdir()
    (model_root / "package" / "result.json").write_text(
        json.dumps({"model_id": "esmfold2_300", "status": "prepared"}), encoding="utf-8"
    )
    (model_root / "evaluate" / "result.json").write_text(
        json.dumps({"model_id": "esmfold2_300", "acceptance": {"accepted": False}}), encoding="utf-8"
    )
    (model_root / "train" / "result.json").write_text(
        json.dumps({"model_id": "esmfold2_300", "status": "complete"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="passed package"):
        prepare_release(tmp_path, "esmfold2_300")
