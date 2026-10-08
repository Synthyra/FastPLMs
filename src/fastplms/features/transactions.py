"""Process ownership and durable publication for the local feature store.

Locks are kernel-owned, not PID files, and release when a process exits. Lock files stay
in place: unlinking one would let another process lock a different inode at the same path.
See https://docs.python.org/3/library/fcntl.html and
https://www.sqlite.org/atomiccommit.html for the locking and flush assumptions.
"""

from __future__ import annotations

import errno
import os
import time

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


@contextmanager
def file_lock(path: Path, *, wait: bool = True) -> Iterator[None]:
    """Hold a process lock on a stable file, optionally refusing a competing owner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    owner_pid = os.getpid()
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if not wait:
                        raise BlockingIOError(
                            "Feature segment already has an active writer."
                        ) from error
                    time.sleep(0.05)
            try:
                yield
            finally:
                if os.getpid() == owner_pid:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            flags = fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB)
            try:
                fcntl.flock(handle.fileno(), flags)
            except BlockingIOError as error:
                raise BlockingIOError("Feature segment already has an active writer.") from error
            try:
                yield
            finally:
                if os.getpid() == owner_pid:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def sync_directory(path: Path) -> None:
    """Flush directory entries on POSIX; Windows has no equivalent directory fsync."""
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def flush_and_evict(handle: BinaryIO) -> None:
    """Make a written file durable, then drop its pages from the page cache.

    A feature store writes far more than it reads back soon, and written pages stay cached. On a GH200 the
    kernel fills the GPU's HBM, which Linux exposes as a NUMA node, with that cache, so ``nvidia-smi`` reads
    the device as full and the next large allocation waits while the kernel evicts. The pages are clean
    after ``fsync``, so ``POSIX_FADV_DONTNEED`` drops all of them. Windows has no ``posix_fadvise`` and
    keeps only the flush. Callers flush Python buffers first.
    """
    descriptor = handle.fileno()
    os.fsync(descriptor)
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)  # offset 0, length 0: the whole file


def publish_file(temporary: Path, destination: Path, *, sync_parent: bool = True) -> None:
    """Flush a complete staged file, rename it, then flush its containing directory.

    A caller that publishes many files into one directory passes ``sync_parent=False`` and flushes the
    directory once, before the commit marker, so a part costs one flush and not two. The staged file
    leaves the page cache once it is durable (``flush_and_evict``).
    """
    with temporary.open("r+b") as handle:
        flush_and_evict(handle)
    temporary.replace(destination)
    if sync_parent:
        sync_directory(destination.parent)
