"""Sequence selection for multiple-sequence alignments."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from .esmfold2_system import run_subprocess_with_errorcheck


def _byte_matrix(array: np.ndarray) -> np.ndarray:
    """Return a two-dimensional byte view used for Hamming comparisons."""

    matrix = np.asarray(array).view(np.uint8)
    return matrix.reshape(matrix.shape[0], -1)


def _hamming_to_all(query: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    return np.not_equal(sequences, query).mean(axis=1, dtype=np.float64)


def greedy_select_indices(array: np.ndarray, num_seqs: int, mode: str = "max") -> list[int]:
    """Select MSA rows by greedy mean Hamming distance from the query row.

    Row zero is always retained. At each step the selector chooses the remaining
    row with greatest distance for ``mode="max"`` or least distance for
    ``mode="min"``. Returned indices follow source order.
    """

    if mode not in {"max", "min"}:
        raise AssertionError(f"unsupported selection mode: {mode}")
    depth = array.shape[0]
    if depth <= num_seqs:
        return list(range(depth))

    sequences = _byte_matrix(array)
    selected = [0]
    available = np.ones(depth, dtype=bool)
    available[0] = False
    distance_sum = _hamming_to_all(sequences[0], sequences)
    choose = np.argmax if mode == "max" else np.argmin

    while len(selected) < num_seqs:
        candidates = np.flatnonzero(available)
        candidate_scores = distance_sum[candidates] / len(selected)
        next_index = int(candidates[int(choose(candidate_scores))])
        selected.append(next_index)
        available[next_index] = False
        distance_sum += _hamming_to_all(sequences[next_index], sequences)
    return sorted(selected)


def _temporary_root() -> str | None:
    shared_memory = Path("/dev/shm")
    return os.fspath(shared_memory) if shared_memory.is_dir() else None


def hhfilter(
    sequences: list[str],
    seqid: int = 90,
    diff: int = 0,
    cov: int = 0,
    qid: int = 0,
    qsc: float = -20.0,
    binary: str = "hhfilter",
) -> list[int]:
    """Run HH-suite filtering and return source indices from its FASTA headers."""

    with tempfile.TemporaryDirectory(dir=_temporary_root()) as directory:
        work = Path(directory)
        source_path = work / "input.fasta"
        result_path = work / "output.fasta"
        records = (f">{index}\n{sequence}" for index, sequence in enumerate(sequences))
        source_path.write_text("\n".join(records), encoding="utf-8")
        command = [
            binary,
            "-i",
            os.fspath(source_path),
            "-M",
            "a3m",
            "-o",
            os.fspath(result_path),
            "-id",
            str(seqid),
            "-diff",
            str(diff),
            "-cov",
            str(cov),
            "-qid",
            str(qid),
            "-qsc",
            str(qsc),
        ]
        run_subprocess_with_errorcheck(command, capture_output=True)
        headers = result_path.read_text(encoding="utf-8").splitlines()
        return [int(line[1:].strip()) for line in headers if line.startswith(">")]
