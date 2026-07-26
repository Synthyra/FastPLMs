"""Dependency-light RNG scoping for ESMFold2 workflows."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class _RandomState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Tensor
    torch_cuda: list[Tensor] | None


def _capture_random_state() -> _RandomState:
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return _RandomState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=cuda_state,
    )


def _restore_random_state(state: _RandomState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(state.torch_cuda)


@contextmanager
def seed_context(seed: int | None) -> Iterator[None]:
    """Seed Python, NumPy, and Torch temporarily, then restore every stream."""

    if seed is None:
        yield
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be None or an integer (excluding bool).")
    state = _capture_random_state()
    normalized_seed = seed % (2**32)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)
    try:
        yield
    finally:
        _restore_random_state(state)


__all__ = ["seed_context"]
