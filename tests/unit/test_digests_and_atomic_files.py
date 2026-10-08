"""Digests, JSON forms and atomic writes that FastPLMs code and tools share."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pytest

from pathlib import Path

from fastplms import atomic_files
from fastplms.atomic_files import write_bytes_atomically, write_text_atomically
from fastplms.checkpoint_files import ArtifactError, hash_file
from fastplms.digests import FILE_READ_BYTES, file_sha256, json_sha256
from fastplms.json_files import compact_json, indented_json


class TestFileSha256:
    @pytest.mark.parametrize("size", [0, 1, FILE_READ_BYTES - 1, FILE_READ_BYTES, FILE_READ_BYTES + 1, 3 * FILE_READ_BYTES + 5])
    def test_matches_hashlib_around_the_block_size(self, tmp_path: Path, size: int) -> None:
        payload = bytes(index % 251 for index in range(size))
        path = tmp_path / "payload.bin"
        path.write_bytes(payload)
        assert file_sha256(path) == hashlib.sha256(payload).hexdigest()

    def test_accepts_a_string_path(self, tmp_path: Path) -> None:
        path = tmp_path / "payload.bin"
        path.write_bytes(b"abc")
        assert file_sha256(str(path)) == hashlib.sha256(b"abc").hexdigest()

    def test_reads_in_blocks_and_never_the_whole_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "payload.bin"
        path.write_bytes(b"x" * (2 * FILE_READ_BYTES + 1))

        def reject_read_bytes(*args: object, **kwargs: object) -> bytes:
            raise AssertionError("file_sha256 must stream the file")

        monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
        assert file_sha256(path) == hashlib.sha256(b"x" * (2 * FILE_READ_BYTES + 1)).hexdigest()

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            file_sha256(tmp_path / "missing.bin")


class TestJsonForms:
    def test_compact_json_sorts_keys_and_drops_whitespace(self) -> None:
        assert compact_json({"b": [1, 2], "a": {"d": None, "c": True}}) == '{"a":{"c":true,"d":null},"b":[1,2]}'

    def test_compact_json_escapes_non_ascii_unless_asked(self) -> None:
        assert compact_json({"k": "\u00e9"}) == '{"k":"\\u00e9"}'
        assert compact_json({"k": "\u00e9"}, ensure_ascii=False) == '{"k":"\u00e9"}'

    def test_compact_json_refuses_nan_when_asked(self) -> None:
        assert compact_json({"x": math.nan}) == '{"x":NaN}'
        with pytest.raises(ValueError):
            compact_json({"x": math.nan}, allow_nan=False)

    def test_indented_json_is_two_space_sorted_with_one_trailing_newline(self) -> None:
        assert indented_json({"b": 1, "a": [1]}) == '{\n  "a": [\n    1\n  ],\n  "b": 1\n}\n'

    def test_indented_json_can_keep_insertion_order(self) -> None:
        assert indented_json({"b": 1, "a": 2}, sort_keys=False) == '{\n  "b": 1,\n  "a": 2\n}\n'

    def test_indented_json_ascii_and_nan_options(self) -> None:
        assert indented_json({"k": "\u00e9"}) == '{\n  "k": "\\u00e9"\n}\n'
        assert indented_json({"k": "\u00e9"}, ensure_ascii=False) == '{\n  "k": "\u00e9"\n}\n'
        with pytest.raises(ValueError):
            indented_json({"x": math.inf}, allow_nan=False)


class TestJsonSha256:
    @pytest.mark.parametrize(
        "value",
        [{}, [], {"a": 1}, {"b": [1, 2.5, None], "a": "text"}, ["\u00e9", {"z": False}]],
    )
    def test_is_the_digest_of_the_compact_form(self, value: object) -> None:
        compact = json.dumps(value, sort_keys=True, separators=(",", ":"))
        assert json_sha256(value) == hashlib.sha256(compact.encode("utf-8")).hexdigest()

    def test_key_order_does_not_matter(self) -> None:
        assert json_sha256({"a": 1, "b": 2}) == json_sha256({"b": 2, "a": 1})

    def test_ascii_escaping_changes_the_digest_of_non_ascii_text(self) -> None:
        raw = hashlib.sha256(json.dumps({"k": "\u00e9"}, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        assert json_sha256({"k": "\u00e9"}, ensure_ascii=False) == raw.hexdigest()
        assert json_sha256({"k": "\u00e9"}) != raw.hexdigest()

    def test_nan_is_refused_when_asked(self) -> None:
        json_sha256({"x": math.nan})
        with pytest.raises(ValueError):
            json_sha256({"x": math.nan}, allow_nan=False)


class TestAtomicWrites:
    def test_bytes_replace_an_existing_file_and_leave_no_temporary(self, tmp_path: Path) -> None:
        path = tmp_path / "state.bin"
        path.write_bytes(b"old")
        write_bytes_atomically(path, b"new bytes")
        assert path.read_bytes() == b"new bytes"
        assert [entry.name for entry in tmp_path.iterdir()] == ["state.bin"]

    def test_text_encodes_utf8_and_keeps_requested_newlines(self, tmp_path: Path) -> None:
        path = tmp_path / "state.txt"
        write_text_atomically(path, "a\n\u00e9\n", newline="\n")
        assert path.read_bytes() == b"a\n\xc3\xa9\n"

    def test_text_default_newline_follows_the_platform(self, tmp_path: Path) -> None:
        path = tmp_path / "state.txt"
        write_text_atomically(path, "a\nb\n")
        assert path.read_bytes() == ("a" + os.linesep + "b" + os.linesep).encode()

    def test_failed_write_keeps_the_old_file_and_removes_the_temporary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "state.bin"
        path.write_bytes(b"old")

        def failing_fsync(descriptor: int) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(atomic_files.os, "fsync", failing_fsync)
        with pytest.raises(OSError, match="disk full"):
            write_bytes_atomically(path, b"new")
        assert path.read_bytes() == b"old"
        assert [entry.name for entry in tmp_path.iterdir()] == ["state.bin"]

    def test_failed_text_write_removes_the_temporary(self, tmp_path: Path) -> None:
        path = tmp_path / "state.txt"
        with pytest.raises(UnicodeEncodeError):
            write_text_atomically(path, "\u00e9", encoding="ascii")
        assert list(tmp_path.iterdir()) == []

    def test_missing_parent_is_an_error_unless_created(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "state.bin"
        with pytest.raises(FileNotFoundError):
            write_bytes_atomically(path, b"x")
        write_bytes_atomically(path, b"x", create_parent=True)
        assert path.read_bytes() == b"x"
        text_path = tmp_path / "other" / "state.txt"
        with pytest.raises(FileNotFoundError):
            write_text_atomically(text_path, "x")
        write_text_atomically(text_path, "x", newline="\n", create_parent=True)
        assert text_path.read_text(encoding="utf-8") == "x"


class TestHashFile:
    def test_sha256_is_the_shared_file_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "weights.bin"
        path.write_bytes(b"weights" * 1000)
        assert hash_file(path) == file_sha256(path)
        assert hash_file(path, "sha256") == file_sha256(path)

    def test_git_sha1_is_the_git_blob_id(self, tmp_path: Path) -> None:
        path = tmp_path / "pointer.txt"
        path.write_bytes(b"hello\n")
        # `printf 'hello\n' | git hash-object --stdin`
        assert hash_file(path, "git-sha1") == "ce013625030ba8dba906f756967f9e9ca394464a"

    def test_git_sha1_streams_large_files(self, tmp_path: Path) -> None:
        payload = b"y" * (FILE_READ_BYTES + 7)
        path = tmp_path / "large.bin"
        path.write_bytes(payload)
        expected = hashlib.sha1(b"blob %d\0" % len(payload) + payload, usedforsecurity=False).hexdigest()
        assert hash_file(path, "git-sha1") == expected

    def test_unknown_algorithm_is_an_artifact_error(self, tmp_path: Path) -> None:
        path = tmp_path / "weights.bin"
        path.write_bytes(b"x")
        with pytest.raises(ArtifactError, match="Unsupported digest algorithm"):
            hash_file(path, "md5")
