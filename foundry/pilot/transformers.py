"""Time a Transformers trainer's real input iterator without shortening its learning-rate schedule."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any
from transformers import Trainer, TrainerCallback

from . import Pilot


class TimedLoader:
    """Keep the loader's full length and sampler while measuring its actual next-batch waits."""

    def __init__(self, loader: Any, pilot: Pilot) -> None:
        if len(loader) < pilot.warmup + pilot.steps:
            raise ValueError("the optimizer pilot must fit within one complete training epoch")
        self.loader, self.pilot = loader, pilot
        self.iterator: Iterator[Any] | None = None

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)

    def __iter__(self) -> Iterator[Any]:
        if self.iterator is not None:
            raise RuntimeError("a qualification loader cannot start a second epoch")
        self.iterator = self.pilot.watch(self.loader)
        return self.iterator

    def finish_window(self) -> None:
        if self.iterator is None:
            raise RuntimeError("the optimizer never requested its training inputs")
        sentinel = object()
        if next(self.iterator, sentinel) is not sentinel:
            raise RuntimeError("the trainer stopped before the requested pilot window")

    def close(self) -> None:
        if self.iterator is not None:
            self.iterator.close()


class StopWindow(TrainerCallback):
    """Stop after measured optimizer updates, leaving the original total-step schedule intact."""

    def __init__(self, loader: TimedLoader, pilot: Pilot) -> None:
        self.loader, self.pilot = loader, pilot

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if state.global_step:
            raise ValueError("optimizer qualification must start from fresh scratch state")
        self.pilot.total_steps = state.max_steps
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if state.global_step == self.pilot.warmup + self.pilot.steps:
            self.loader.finish_window()
            control.should_training_stop = True
        return control


@contextmanager
def timed_training(module: ModuleType, pilot: Pilot) -> Iterator[None]:
    """Wrap Protify's existing plain/cluster trainer classes for one fresh, single-device pilot.

    Input waits come from the real loader, rather than a callback's synthetic range.
    Gradient accumulation is refused because one yielded batch must mean one optimizer update.
    Classes and background measurement resources are restored even when fitting raises.
    """
    originals = {name: getattr(module, name) for name in ("Trainer", "ClusterTrainer")}
    loaders: list[TimedLoader] = []

    def measured_type(original: type[Trainer]) -> type[Trainer]:
        class MeasuredTrainer(original):
            def get_train_dataloader(self) -> TimedLoader:
                if self.args.gradient_accumulation_steps != 1 or self.args.world_size != 1:
                    raise ValueError("optimizer timing requires one device and one batch per update")
                if loaders:
                    raise RuntimeError("one pilot measures exactly one trainer")
                loader = TimedLoader(super().get_train_dataloader(), pilot)
                loaders.append(loader)
                self.add_callback(StopWindow(loader, pilot))
                return loader

        return MeasuredTrainer

    try:
        for name, original in originals.items():
            setattr(module, name, measured_type(original))
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)
        for loader in loaders:
            loader.close()
        if not pilot.closed:
            list(pilot.watch(()))
