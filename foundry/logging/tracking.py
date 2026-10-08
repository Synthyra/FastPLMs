"""`@track`: put a training function on W&B by the workspace standard with one line.

    @track(project="atlas-ppi", job_type="train")
    def train(config: Config, *, run: Run) -> dict[str, float]:
        for step, batch in enumerate(loader):
            run.log({"train/loss": loss.item()}, step=step)
        return {"test/auroc": auroc}

    train(config)  # or train(config, directory=output / "wandb")

The decorated function receives an open `Run` as `run` and may return a mapping of final
scalars, which becomes the run's summary. The run's directory is `directory`, else
`config.output_dir / "wandb"`; its revision is the code of the file the function is defined
in; the run finishes `completed`, or `failed` when the function raises, unless the function
finished it itself, as a planned `time_stop` does. A config with an integer `seed` seeds
`random`, NumPy and torch first (`foundry.training.set_seed`). Loop over `run.batches(loader)`
and the job is piloted with `WS_PILOT=1`, and its cost lands in the summary either way.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, TypeVar

from .revision import revision_of
from .run import Run


P = ParamSpec("P")
R = TypeVar("R")


def track(*, project: str, **options: Any) -> Callable[[Callable[Concatenate[Any, P], R]], Callable[..., R]]:
    """Run the decorated function inside a `Run`; `options` are `Run`'s keyword arguments."""

    def decorate(train: Callable[Concatenate[Any, P], R]) -> Callable[..., R]:
        source = Path(inspect.getfile(train))

        @functools.wraps(train)
        def tracked(config: Any, *args: P.args, directory: Path | None = None, **kwargs: P.kwargs) -> R:
            values = config_values(config)
            if directory is None:
                if "output_dir" not in values:
                    raise ValueError(f"{train.__name__} needs directory= or a config with output_dir for its run records.")
                directory = Path(values["output_dir"]) / "wandb"
            seed = values.get("seed")
            if isinstance(seed, int) and not isinstance(seed, bool):
                from foundry.training import set_seed

                set_seed(seed)
            with Run(Path(directory), values, project=project, revision=revision_of(source), **options) as run:
                returned = train(config, *args, run=run, **kwargs)
                if isinstance(returned, Mapping) and run.summary_path is None:
                    run.finish(summary=returned)
            return returned

        return tracked

    return decorate


def config_values(config: Any) -> dict[str, Any]:
    """A config as the mapping W&B records: a mapping, a dataclass, or an argparse namespace."""
    if isinstance(config, Mapping):
        return dict(config)
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError(f"a {type(config).__name__} config cannot be recorded; pass a mapping, dataclass or namespace.")
