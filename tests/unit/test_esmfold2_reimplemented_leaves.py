"""Behavioral contracts for independently organized ESMFold2 leaf utilities."""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest
import torch

from fastplms.models.esmfold2 import esmfold2_input_builder
from fastplms.models.esmfold2.esmfold2_aligner import Aligner
from fastplms.models.esmfold2.esmfold2_atom_indexer import AtomIndexer
from fastplms.models.esmfold2.esmfold2_msa_filter_sequences import (
    greedy_select_indices,
    hhfilter,
)
from fastplms.models.esmfold2.esmfold2_normalize_coordinates import index_by_atom_name
from fastplms.models.esmfold2.esmfold2_predicted_aligned_error import (
    compute_predicted_aligned_error,
    compute_tm,
)
from fastplms.models.esmfold2.esmfold2_system import run_subprocess_with_errorcheck
from fastplms.models.esmfold2.esmfold2_types import ProteinInput, StructurePredictionInput

pytestmark = pytest.mark.structure


def test_greedy_msa_selection_preserves_official_tie_order() -> None:
    sequences = np.asarray(
        [list(row) for row in ("AAAA", "AAAT", "AATT", "TTTT", "ATAT")],
        dtype="S1",
    )
    assert greedy_select_indices(sequences, 3, mode="max") == [0, 1, 3]
    assert greedy_select_indices(sequences, 3, mode="min") == [0, 1, 2]
    assert greedy_select_indices(sequences, 10) == [0, 1, 2, 3, 4]
    with pytest.raises(AssertionError, match="unsupported selection mode"):
        greedy_select_indices(sequences, 2, mode="median")


def test_hhfilter_passes_paths_as_distinct_arguments(tmp_path) -> None:
    executable = tmp_path / "fake_hhfilter.py"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "output.write_text('>2\\nCCC\\n>0\\nAAA\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    assert hhfilter(["AAA", "BBB", "CCC"], binary=str(executable)) == [2, 0]


def test_subprocess_failure_includes_standard_error() -> None:
    with pytest.raises(RuntimeError, match="intentional failure"):
        run_subprocess_with_errorcheck(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('intentional failure'); sys.exit(4)",
            ],
            capture_output=True,
        )


def test_schema_namespace_preserves_type_identity() -> None:
    assert ProteinInput is esmfold2_input_builder.ProteinInput
    assert StructurePredictionInput is esmfold2_input_builder.StructurePredictionInput


@dataclasses.dataclass
class _Structure:
    atom37_positions: np.ndarray
    atom37_mask: np.ndarray

    def __len__(self) -> int:
        return self.atom37_positions.shape[0]


def test_atom_indexer_selects_the_declared_property_and_axis() -> None:
    positions = np.arange(2 * 37 * 3, dtype=np.float32).reshape(2, 37, 3)
    structure = _Structure(positions, np.ones((2, 37), dtype=bool))
    indexer = AtomIndexer(structure, "atom37_positions", dim=1)
    np.testing.assert_array_equal(indexer["CA"], positions[:, 1])
    np.testing.assert_array_equal(indexer[["N", "C"]], positions[:, [0, 2]])


def test_atom_name_selection_matches_for_numpy_and_torch() -> None:
    positions = np.arange(2 * 37 * 3, dtype=np.float32).reshape(2, 37, 3)
    expected = positions[:, [0, 1, 2]]
    np.testing.assert_array_equal(
        index_by_atom_name(positions, ["N", "CA", "C"]),
        expected,
    )
    assert torch.equal(
        index_by_atom_name(torch.from_numpy(positions), ["N", "CA", "C"]),
        torch.from_numpy(expected),
    )


def test_pae_bin_expectation_and_tm_score_are_exact_for_uniform_logits() -> None:
    logits = torch.zeros((1, 2, 2, 4), dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    pae = compute_predicted_aligned_error(logits, mask)
    assert torch.equal(pae, torch.full((1, 2, 2), 31.0, dtype=torch.float64))

    centers = torch.tensor([7.75, 23.25, 38.75, 54.25], dtype=torch.float64)
    d0 = 1.24 * 4 ** (1 / 3) - 1.8
    expected_tm = (1 / (1 + (centers / d0) ** 2)).mean().reshape(1)
    torch.testing.assert_close(compute_tm(logits, mask), expected_tm, rtol=2e-6, atol=1e-12)


def test_aligner_recovers_a_rigid_translation() -> None:
    mobile_positions = np.full((1, 37, 3), np.nan, dtype=np.float32)
    mobile_positions[0, :3] = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    mask = np.zeros((1, 37), dtype=bool)
    mask[0, :3] = True
    target_positions = mobile_positions.copy()
    target_positions[mask] += np.asarray([2.0, -1.0, 3.0], dtype=np.float32)
    mobile = _Structure(mobile_positions, mask)
    target = _Structure(target_positions, mask)

    aligner = Aligner(mobile, target, only_use_backbone=True)
    aligned = aligner.apply(mobile)
    assert aligner.rmsd < 1e-7
    np.testing.assert_allclose(aligned.atom37_positions[mask], target_positions[mask], atol=1e-6)
