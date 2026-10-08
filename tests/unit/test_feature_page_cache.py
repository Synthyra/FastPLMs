"""A written feature part leaves the page cache once it is durable, and a platform without the call keeps the flush."""

from __future__ import annotations

import os
import pytest
import torch

from pathlib import Path
from safetensors.torch import load_file

from fastplms.features import store as storage
from fastplms.features import transactions


DONTNEED = 4  # Patched in as os.POSIX_FADV_DONTNEED, which Windows does not define.


class CallLog:
    """The fsync and posix_fadvise calls in order, each with the descriptor and arguments it received."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, tuple[int, ...]]] = []

    def fsync(self, descriptor: int) -> None:
        self.calls.append(("fsync", descriptor, ()))

    def fadvise(self, descriptor: int, offset: int, length: int, advice: int) -> None:
        self.calls.append(("fadvise", descriptor, (offset, length, advice)))

    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> CallLog:
    log = CallLog()
    monkeypatch.setattr(os, "fsync", log.fsync)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", DONTNEED, raising=False)
    monkeypatch.setattr(os, "posix_fadvise", log.fadvise, raising=False)
    return log


def test_flush_and_evict_drops_the_whole_file_after_the_flush(tmp_path: Path, calls: CallLog) -> None:
    with (tmp_path / "part.safetensors").open("wb") as handle:
        transactions.flush_and_evict(handle)
        descriptor = handle.fileno()

    assert calls.calls == [("fsync", descriptor, ()), ("fadvise", descriptor, (0, 0, DONTNEED))]


def test_a_platform_without_posix_fadvise_keeps_only_the_flush(
    tmp_path: Path, calls: CallLog, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(os, "posix_fadvise")
    monkeypatch.delattr(os, "POSIX_FADV_DONTNEED")  # Windows defines neither name.
    with (tmp_path / "part.safetensors").open("wb") as handle:
        transactions.flush_and_evict(handle)
        descriptor = handle.fileno()

    assert calls.calls == [("fsync", descriptor, ())]


def test_publishing_a_staged_file_evicts_it_before_the_rename(tmp_path: Path, calls: CallLog) -> None:
    staged, published = tmp_path / "sidecar.writing", tmp_path / "sidecar"
    staged.write_bytes(b"rows")

    transactions.publish_file(staged, published, sync_parent=False)

    assert calls.names() == ["fsync", "fadvise"]
    assert published.read_bytes() == b"rows" and not staged.exists()


def test_a_streamed_part_is_evicted_once_and_still_reads_back(tmp_path: Path, calls: CallLog) -> None:
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3)  # (b, w), b = 2 sequences, w = 3 columns
    part = tmp_path / "part-00000.safetensors"

    digest, size = storage._write_safetensors_streaming(part, {"values": values})

    assert calls.names() == ["fsync", "fadvise"]
    assert part.stat().st_size == size and len(digest) == 64
    assert torch.equal(load_file(str(part))["values"], values)
