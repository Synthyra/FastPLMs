"""The shared file writers keep the bytes each tool and test helper used to write for itself."""

from __future__ import annotations

import json
import math
import os
import stat
import pytest

from pathlib import Path

from tools import stored_files
from tools.stored_files import write_stored_bytes, write_stored_json


VALUE = {"b": [1, 2], "a": {"d": None, "c": "caf\N{LATIN SMALL LETTER E WITH ACUTE}"}}


def test_default_form_is_sorted_two_space_ascii_json_with_one_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    write_stored_json(path, VALUE, newline="\n")
    expected = json.dumps(VALUE, indent=2, sort_keys=True) + "\n"
    assert path.read_bytes() == expected.encode("utf-8")
    assert "\\u00e9" in path.read_text(encoding="utf-8")


def test_insertion_order_and_non_ascii_text_can_be_kept(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    write_stored_json(path, VALUE, sort_keys=False, ensure_ascii=False, newline="\n")
    expected = json.dumps(VALUE, indent=2, ensure_ascii=False) + "\n"
    assert path.read_bytes() == expected.encode("utf-8")
    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["b", "a"]


def test_the_platform_separator_is_written_unless_a_bare_line_feed_is_asked_for(tmp_path: Path) -> None:
    platform_path = tmp_path / "platform.json"
    bare_path = tmp_path / "bare.json"
    write_stored_json(platform_path, {"k": 1})
    write_stored_json(bare_path, {"k": 1}, newline="\n")
    assert bare_path.read_bytes() == b'{\n  "k": 1\n}\n'
    assert platform_path.read_bytes().replace(b"\r\n", b"\n") == bare_path.read_bytes()


def test_missing_parent_directories_are_created(tmp_path: Path) -> None:
    json_path = tmp_path / "one" / "two" / "record.json"
    bytes_path = tmp_path / "three" / "report.bin"
    write_stored_json(json_path, [1, 2, 3])
    write_stored_bytes(bytes_path, b"\x00\xff")
    assert json.loads(json_path.read_text(encoding="utf-8")) == [1, 2, 3]
    assert bytes_path.read_bytes() == b"\x00\xff"


def test_an_existing_file_is_replaced_and_no_temporary_file_remains(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_text("old", encoding="utf-8")
    write_stored_json(path, {"new": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert [entry.name for entry in tmp_path.iterdir()] == ["record.json"]


def test_a_value_that_cannot_be_serialized_leaves_the_old_file_untouched(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_text("old", encoding="utf-8")
    with pytest.raises(ValueError):
        write_stored_json(path, {"x": math.nan}, allow_nan=False)
    assert path.read_text(encoding="utf-8") == "old"
    assert [entry.name for entry in tmp_path.iterdir()] == ["record.json"]


def test_the_file_gets_the_mode_a_plain_open_would_give_under_the_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modes: list[int] = []
    monkeypatch.setattr(stored_files.os, "umask", lambda _mask: 0o027)
    monkeypatch.setattr(Path, "chmod", lambda _self, mode: modes.append(mode))
    write_stored_json(tmp_path / "record.json", {"k": 1})
    write_stored_bytes(tmp_path / "report.bin", b"x")
    assert modes == [0o640, 0o640]


def test_a_file_written_under_the_default_umask_is_readable_by_group_and_others(tmp_path: Path) -> None:
    previous = os.umask(0o022)
    try:
        path = tmp_path / "record.json"
        write_stored_json(path, {"k": 1})
    finally:
        os.umask(previous)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode & 0o600 == 0o600
    if os.name == "posix":  # Windows files carry no group or other permission bits
        assert mode == 0o644
