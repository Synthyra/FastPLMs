"""The fixed set of commands a Modal evidence worker may run.

Stages mirror the repository's own suite commands. A worker only ever receives
a stage name and an optional pytest selection, never a command string, so a
launch cannot make a worker run anything outside this table.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Literal

# Shared with the Docker-host release suite so the two cannot drift apart.
from tools.remote.run import _RELEASE_LOCAL_PARITY_TESTS

from .config import (
    CPU_WORKER_TIMEOUT_SECONDS,
    FOLD_WORKER_TIMEOUT_SECONDS,
    GPU_WORKER_TIMEOUT_SECONDS,
)


Device = Literal["cpu", "gpu"]
# ``fold`` is the candidate image plus an isolated official ESMFold2 environment.
WorkerImage = Literal["candidate", "fold"]
_PYTEST = ("-m", "pytest")
# A pytest ``-k`` expression: identifiers, boolean words, brackets, and spaces.
_SELECTION_PATTERN = re.compile(r"[A-Za-z0-9_ ()\[\]\-.]+")


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One runnable stage: interpreter arguments, where it runs, and its bound."""

    name: str
    device: Device
    arguments: tuple[str, ...]
    timeout_seconds: int
    image: WorkerImage = "candidate"

    @property
    def is_pytest(self) -> bool:
        return self.arguments[: len(_PYTEST)] == _PYTEST


STAGES: dict[str, StageSpec] = {
    spec.name: spec
    for spec in (
        # The required pre-merge gate (docs/testing.md). Its hermetic bootstrap
        # samples memory through procfs, so it runs only on Linux.
        StageSpec(
            "cpu-contract",
            "cpu",
            (
                *_PYTEST,
                "tests/cpu",
                "-m",
                "cpu_contract",
                "-n",
                "auto",
                "--dist=loadscope",
                "--durations=25",
            ),
            CPU_WORKER_TIMEOUT_SECONDS,
        ),
        # The confidence-pilot tests need packages outside the requirement files and
        # are owned by the ``tests`` stage of tools.confidence.launch.
        StageSpec(
            "unit",
            "gpu",
            (*_PYTEST, "tests/unit", "--ignore-glob=tests/unit/test_confidence_*.py"),
            GPU_WORKER_TIMEOUT_SECONDS,
        ),
        # Exact module parity against the pinned official ESMFold2 source.
        StageSpec(
            "parity-local",
            "gpu",
            (*_PYTEST, *_RELEASE_LOCAL_PARITY_TESTS),
            GPU_WORKER_TIMEOUT_SECONDS,
        ),
        # Execution contracts of the locked FlashAttention kernels. The checkpoint cases
        # download weights, so narrow the run with a selection when they are not needed.
        StageSpec(
            "flash-integration",
            "gpu",
            (*_PYTEST, "tests/integration/test_flash_attention_backends.py"),
            GPU_WORKER_TIMEOUT_SECONDS,
        ),
        # Do the manifest-locked FlashAttention kernels still resolve and load?
        StageSpec("probe", "gpu", ("tools/debug/probe_flash_kernels.py",), 600),
        # mypy errors the working tree adds relative to its Git baseline.
        StageSpec(
            "typing", "cpu", ("-m", "tools.gpu_evidence.typing_check"), CPU_WORKER_TIMEOUT_SECONDS
        ),
        # Working tree versus Git baseline latency, interleaved on one GPU.
        StageSpec(
            "lever-bench",
            "gpu",
            ("-m", "tools.gpu_evidence.lever_bench"),
            GPU_WORKER_TIMEOUT_SECONDS,
        ),
        # The same comparison for the padded FlashAttention path, which needs the locked kernels.
        StageSpec(
            "flash-lever-bench",
            "gpu",
            (
                "-m",
                "tools.gpu_evidence.lever_bench",
                "--workloads",
                "esm2-padded-b8-flash_attention_2",
                "esmpp-padded-b8-flash_attention_2",
            ),
            GPU_WORKER_TIMEOUT_SECONDS,
        ),
        # The same comparison for the levers added after the first lever-bench evidence.
        StageSpec(
            "runner-lever-bench",
            "gpu",
            (
                "-m",
                "tools.gpu_evidence.lever_bench",
                "--workloads",
                "runner-full-embeddings-b64",
                "e1-single-sequences-b4",
                "esmpp-single-l256-sdpa",
                "esmpp-padded-b8-sdpa",
            ),
            GPU_WORKER_TIMEOUT_SECONDS,
        ),
        # Working-tree latency of every advertised backend on padded and full batches.
        StageSpec(
            "backend-bench",
            "gpu",
            ("-m", "tools.gpu_evidence.lever_bench", "--mode", "backends"),
            GPU_WORKER_TIMEOUT_SECONDS,
        ),
        # Sizes packed-token execution on a generic encoder before any runtime change.
        StageSpec(
            "packed-probe",
            "gpu",
            ("-m", "tools.gpu_evidence.packed_probe"),
            GPU_WORKER_TIMEOUT_SECONDS,
        ),
        # ESMFold2 folding throughput and peak memory by protein length: the official
        # implementation, the Git baseline tree, and the working tree on one worker.
        StageSpec(
            "fold-bench",
            "gpu",
            ("-m", "tools.gpu_evidence.fold_bench"),
            FOLD_WORKER_TIMEOUT_SECONDS,
            image="fold",
        ),
        # Which model source lines hold the memory at the peak of a fold.
        StageSpec(
            "fold-peak-memory",
            "gpu",
            ("-m", "tools.gpu_evidence.fold_bench", "--peak"),
            GPU_WORKER_TIMEOUT_SECONDS,
            image="fold",
        ),
        # The 2,048-residue extension of the sweep, one instrumented fold per series.
        StageSpec(
            "fold-bench-long",
            "gpu",
            ("-m", "tools.gpu_evidence.fold_bench", "--long"),
            FOLD_WORKER_TIMEOUT_SECONDS,
            image="fold",
        ),
        # The same path on the smallest checkpoint with shortened folds, to catch
        # environment and loading failures before the expensive sweep.
        StageSpec(
            "fold-bench-smoke",
            "gpu",
            ("-m", "tools.gpu_evidence.fold_bench", "--smoke"),
            GPU_WORKER_TIMEOUT_SECONDS,
            image="fold",
        ),
    )
}


def stage_arguments(spec: StageSpec, selection: str | None, junit_path: str) -> tuple[str, ...]:
    """Return interpreter arguments for a stage, narrowed by a pytest selection."""
    if not spec.is_pytest:
        if selection is not None:
            raise ValueError(f"Stage {spec.name!r} is not a pytest stage and takes no selection.")
        return spec.arguments
    arguments = (*spec.arguments, f"--junitxml={junit_path}")
    if selection is None:
        return arguments
    if not _SELECTION_PATTERN.fullmatch(selection):
        raise ValueError(
            "A pytest selection may contain only letters, digits, spaces, and _()[]-. characters."
        )
    return (*arguments, "-k", selection)
