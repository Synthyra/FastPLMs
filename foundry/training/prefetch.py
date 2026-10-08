"""Grouping a loader's batches into patch groups, and preparing the next group while the step runs.

A *patch* is one item a `DataLoader` yields. A trainer's step takes a *patch group*: a list of
patches, `patch_accum` of them, forwarded together so one loss sees the whole batch. Everything here
turns an iterator of patches into an iterator of `(group, exhausted)` pairs:

- an *accumulator* cuts the groups: `PatchAccumulator` takes `patch_accum` patches at a time, and
  `KeyedPatchAccumulator` also starts a new group whenever the key of the patch changes;
- a *prefetcher* wraps an accumulator and an optional `transform_fn` (for example a cached-embedding
  lookup that fills each patch), and is either synchronous or runs both on a background thread
  behind a bounded queue so the step is not waiting on storage.

`exhausted` is True on the last group, which may be partial or, once the loader is out, empty. A
consumer stops at the first empty group and always calls `shutdown()`.
"""

from __future__ import annotations

import queue
import threading
import traceback

from collections.abc import Callable, Iterator
from typing import Any, Protocol


POLL_SECONDS = 0.1
"""How long a thread waits on the queue before it checks for shutdown and for a crashed worker."""


class PatchGroups(Protocol):
    """What a trainer's loop calls on whatever `build_prefetcher` returned."""

    def next(self) -> tuple[list[Any], bool]:
        """The next `(patches, exhausted)`; `patches` is empty only when the loader is fully consumed."""
        ...

    def shutdown(self) -> None:
        """Stop any background thread and release the loader's iterator."""
        ...


class PatchAccumulator:
    """Cuts the loader's patches into consecutive groups of `patch_accum`; the last may be short."""

    def __init__(self, data_iter: Iterator[Any], patch_accum: int) -> None:
        assert patch_accum > 0
        self.data_iter = data_iter
        self.patch_accum = patch_accum
        self._exhausted = False

    def next(self) -> tuple[list[Any], bool]:
        if self._exhausted:
            return [], True

        patches: list[Any] = []
        for _ in range(self.patch_accum):
            try:
                patch = next(self.data_iter)
            except StopIteration:
                self._exhausted = True
                break
            patches.append(patch)
        return patches, self._exhausted

    def shutdown(self) -> None:
        """Nothing runs in the background."""


class SizedPatchAccumulator:
    """Cuts a loader of known length into consecutive groups of `patch_accum`, and flags the group holding its last patch.

    `PatchAccumulator` learns that the loader is out only by asking for a patch that is not there, so a
    loader whose length is a multiple of `patch_accum` never flags its last group, and the trainer's
    leftover accumulation steps in the next epoch. This one counts instead: the group that takes the
    `total_patches`-th patch is flagged, so the trainer steps on it at the end of the epoch. It reads no
    patch before the group that needs it, which keeps the order in which a pipeline draws from a global
    random generator between a step and the evaluation that follows it.
    """

    def __init__(self, data_iter: Iterator[Any], patch_accum: int, total_patches: int) -> None:
        assert patch_accum > 0
        assert total_patches >= 0
        self.data_iter = data_iter
        self.patch_accum = patch_accum
        self.total_patches = total_patches
        self._taken = 0

    def next(self) -> tuple[list[Any], bool]:
        patches: list[Any] = []
        while len(patches) < self.patch_accum and self._taken < self.total_patches:
            try:
                patches.append(next(self.data_iter))
            except StopIteration:
                self._taken = self.total_patches
                break
            self._taken += 1
        return patches, self._taken >= self.total_patches

    def shutdown(self) -> None:
        """Nothing runs in the background."""


class KeyedPatchAccumulator:
    """Groups up to `patch_accum` consecutive patches whose `key_getter` value is equal.

    A change of key ends the group early and the patch that changed it starts the next one, so a
    run of one key shorter than `patch_accum` is still trained, as a partial group. Atlas groups by
    species this way.
    """

    def __init__(self, data_iter: Iterator[Any], patch_accum: int, key_getter: Callable[[Any], int]) -> None:
        assert patch_accum > 0
        self.data_iter = data_iter
        self.patch_accum = patch_accum
        self.key_getter = key_getter
        self._source_exhausted = False
        self._pending_patch: Any | None = None

    def next(self) -> tuple[list[Any], bool]:
        if self._source_exhausted and self._pending_patch is None:
            return [], True

        patches: list[Any] = []
        current_key: int | None = None

        if self._pending_patch is not None:
            pending_patch = self._pending_patch
            self._pending_patch = None
            current_key = self.key_getter(pending_patch)
            patches.append(pending_patch)

        while len(patches) < self.patch_accum:
            try:
                patch = next(self.data_iter)
            except StopIteration:
                self._source_exhausted = True
                break

            patch_key = self.key_getter(patch)
            if current_key is None:
                current_key = patch_key
                patches.append(patch)
                continue

            if patch_key == current_key:
                patches.append(patch)
                continue
            self._pending_patch = patch
            break

        exhausted = self._source_exhausted and self._pending_patch is None
        return patches, exhausted

    def shutdown(self) -> None:
        """Nothing runs in the background."""


class SynchronousGroupPrefetcher:
    """An accumulator and an optional `transform_fn`, run on the caller's thread, one group per call."""

    def __init__(self, accumulator: PatchGroups, transform_fn: Callable[[list[Any]], list[Any]] | None = None) -> None:
        self.accumulator = accumulator
        self.transform_fn = transform_fn

    def next(self) -> tuple[list[Any], bool]:
        patches, exhausted = self.accumulator.next()
        if len(patches) > 0 and self.transform_fn is not None:
            patches = self.transform_fn(patches)
        return patches, exhausted

    def shutdown(self) -> None:
        self.accumulator.shutdown()


class AsyncGroupPrefetcher:
    """An accumulator and an optional `transform_fn` on a background thread, up to `max_queue_size` groups ahead.

    Groups reach the consumer in the order the accumulator cut them, since one thread produces and a
    FIFO queue carries them. An error in the accumulator or the transform stops the producer and
    is raised from the consumer's next `next()` as a `RuntimeError` that chains it. `shutdown()`
    joins the thread, so it waits for a group the thread is still building.
    """

    def __init__(
        self,
        accumulator: PatchGroups,
        max_queue_size: int = 4,
        transform_fn: Callable[[list[Any]], list[Any]] | None = None,
    ) -> None:
        assert max_queue_size > 0

        self.accumulator = accumulator
        self.transform_fn = transform_fn

        self._queue: queue.Queue[tuple[list[Any], bool]] = queue.Queue(maxsize=max_queue_size)
        self._exhausted = False
        self._shutdown = False
        self._error: BaseException | None = None

        self._worker = threading.Thread(target=self._produce_loop, daemon=True)
        self._worker.start()

    def _produce_loop(self) -> None:
        try:
            while not self._shutdown and not self._exhausted:
                patches, exhausted = self.accumulator.next()
                if len(patches) > 0 and self.transform_fn is not None:
                    patches = self.transform_fn(patches)
                while not self._shutdown:
                    try:
                        self._queue.put((patches, exhausted), timeout=POLL_SECONDS)
                        break
                    except queue.Full:
                        continue
                self._exhausted = exhausted
        except BaseException as error:  # noqa: BLE001 the producer thread reports any failure to the consumer
            print(f"[AsyncGroupPrefetcher] Worker thread crashed: {type(error).__name__}: {error}")
            traceback.print_exc()
            self._error = error
            self._exhausted = True

    def _check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Prefetcher worker crashed: {self._error}") from self._error

    def next(self) -> tuple[list[Any], bool]:
        if self._shutdown:
            return [], True
        self._check_error()

        if self._queue.empty() and self._exhausted:
            self._check_error()
            return [], True

        while True:
            if self._shutdown:
                return [], True
            self._check_error()
            try:
                return self._queue.get(timeout=POLL_SECONDS)
            except queue.Empty:
                if self._exhausted:
                    self._check_error()
                    return [], True

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._worker.join()
        self.accumulator.shutdown()


def build_prefetcher(
    accumulator: PatchGroups,
    *,
    asynchronous: bool,
    queue_size: int = 4,
    transform_fn: Callable[[list[Any]], list[Any]] | None = None,
) -> PatchGroups:
    """The prefetcher that runs `accumulator` and `transform_fn` on a thread, or on the caller's when not `asynchronous`."""
    if asynchronous:
        return AsyncGroupPrefetcher(accumulator, max_queue_size=queue_size, transform_fn=transform_fn)
    return SynchronousGroupPrefetcher(accumulator, transform_fn=transform_fn)
