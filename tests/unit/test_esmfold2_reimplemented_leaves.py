"""Behavioral contracts for independently organized ESMFold2 leaf utilities."""

from __future__ import annotations

import dataclasses
import sys
import numpy as np
import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

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
from fastplms.models.esmfold2.esmfold2_protein_chain import ProteinChain
from fastplms.models.esmfold2.esmfold2_system import run_subprocess_with_errorcheck
from fastplms.models.esmfold2.esmfold2_types import (
    PocketConditioning,
    ProteinInput,
    StructurePredictionInput,
)


pytestmark = pytest.mark.structure


def test_greedy_msa_selection_preserves_official_tie_order() -> None:
    sequences = np.asarray(  # (n=5, l=4)
        [list(row) for row in ("AAAA", "AAAT", "AATT", "TTTT", "ATAT")],
        dtype="S1",
    )
    assert greedy_select_indices(sequences, 3, mode="max") == [0, 1, 3]
    assert greedy_select_indices(sequences, 3, mode="min") == [0, 1, 2]
    assert greedy_select_indices(sequences, 10) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="unsupported selection mode"):
        greedy_select_indices(sequences, 2, mode="median")
    with pytest.raises(ValueError, match="greater than zero"):
        greedy_select_indices(sequences, 0)
    with pytest.raises(ValueError, match="non-empty shape"):
        greedy_select_indices(np.empty((0, 4), dtype="S1"), 1)


def test_fast_msa_stack_preserves_headerless_and_mixed_inputs() -> None:
    from fastplms.models.esmfold2.esmfold2_msa import FastMSA

    first = FastMSA(np.asarray([list("AAA"), list("AAT")], dtype="S1"))  # (n=2, l=3)
    second = FastMSA(np.asarray([list("AAA"), list("ATT")], dtype="S1"))  # (n=2, l=3)
    headerless = FastMSA.stack([first, second])

    assert headerless.depth == 3
    assert headerless.headers is None

    with_headers = FastMSA(
        np.asarray([list("AAA"), list("ATA")], dtype="S1"),  # (n=2, l=3)
        ["query", "named"],
    )
    mixed = FastMSA.stack([first, with_headers])
    assert mixed.depth == 3
    assert mixed.headers == ["", "", "named"]


def test_hhfilter_passes_paths_as_distinct_arguments(tmp_path: Path) -> None:
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
    assert PocketConditioning is esmfold2_input_builder.PocketConditioning
    assert StructurePredictionInput is esmfold2_input_builder.StructurePredictionInput


def test_msa_rejects_unequal_biological_rows() -> None:
    from fastplms.models.esmfold2.esmfold2_types import MSA

    with pytest.raises(ValueError, match="MSA row length mismatch"):
        MSA.from_sequences(["ACDE", "ACD"])


def test_a3m_dot_insertions_and_raw_rows_have_consistent_metadata() -> None:
    import io

    from fastplms.models.esmfold2.esmfold2_msa import MSA

    source = ">query\nA.CD\n>hit\nAaCD\n"
    match_columns = MSA.from_a3m(io.StringIO(source))
    assert match_columns.sequences == ["ACD", "ACD"]
    assert match_columns.deletions is not None
    # n=2 aligned sequences, l=3 retained match columns.
    assert match_columns.deletions.shape == (2, 3)

    raw_rows = MSA.from_a3m(io.StringIO(source), remove_insertions=False)
    assert raw_rows.sequences == ["A.CD", "AaCD"]
    assert raw_rows.deletions is None


def test_protein_chain_rejects_misaligned_atom37_tables() -> None:
    with pytest.raises(ValueError, match=r"shape \(length, 37, 3\)"):
        ProteinChain.from_atom37(
            np.zeros((2, 36, 3), dtype=np.float32)  # (l=2, a=36, xyz=3)
        )


def test_protein_chain_contacts_require_retained_mmcif_source() -> None:
    chain = ProteinChain.from_atom37(
        np.zeros((2, 37, 3), dtype=np.float32),  # (l=2, a=37, xyz=3)
        sequence="AC",
    )
    with pytest.raises(ValueError, match="keep_source=True"):
        chain.find_nonpolymer_contacts()


def test_unavailable_structure_kernel_backend_fails_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastplms.models.esmfold2 import modeling_esmfold2_common as common

    with pytest.raises(RuntimeError, match="does not bundle"):
        common.validate_kernel_backend("fused")
    monkeypatch.setattr(common, "CUE_AVAILABLE", False)
    with pytest.raises(
        RuntimeError,
        match=r"requires cuequivariance_torch.*cuequivariance_ops_torch",
    ):
        common.validate_kernel_backend("cuequivariance")
    with pytest.raises(ValueError, match="backend must be one of"):
        common.validate_kernel_backend("silent-fallback")


def test_experimental_top_level_kernel_backend_validates_before_zero_layer_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastplms.models.esmfold2 import modeling_esmfold2_common as common
    from fastplms.models.esmfold2.modeling_esmfold2_experimental import (
        ESMFold2ExperimentalModel,
    )

    class ZeroLayerRecorder:
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        def set_kernel_backend(self, backend: str | None) -> None:
            self.calls.append(backend)

    folding_trunk = ZeroLayerRecorder()
    structure_head = ZeroLayerRecorder()
    model = SimpleNamespace(
        folding_trunk=folding_trunk,
        confidence_head=None,
        structure_head=structure_head,
        _kernel_backend=None,
    )
    monkeypatch.setattr(common, "CUE_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="requires cuequivariance_torch"):
        ESMFold2ExperimentalModel.set_kernel_backend(model, "cuequivariance")
    assert folding_trunk.calls == []
    assert structure_head.calls == []
    assert model._kernel_backend is None

    ESMFold2ExperimentalModel.set_kernel_backend(model, None)
    assert folding_trunk.calls == [None]
    assert structure_head.calls == [None]
    assert model._kernel_backend is None


def test_pocket_conditioning_is_rejected_instead_of_silently_dropped() -> None:
    from fastplms.models.esmfold2.esmfold2_processor import clean_esmfold2_input

    request = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="ACD")],
        pocket=PocketConditioning(binder_chain_id="A", contacts=[("A", 0)]),
    )
    with pytest.raises(NotImplementedError, match="refuses this input"):
        clean_esmfold2_input(request)


def test_multiple_delimited_proteins_keep_their_own_split_identity() -> None:
    from fastplms.models.esmfold2.esmfold2_processor import clean_esmfold2_input

    request = StructurePredictionInput(
        sequences=[
            ProteinInput(id="first", sequence="AA:CC"),
            ProteinInput(id="second", sequence="GG:TT"),
        ]
    )
    cleaned = clean_esmfold2_input(request)

    assert [protein.sequence for protein in cleaned.sequences] == ["AA", "CC", "GG", "TT"]
    assert [protein.id for protein in cleaned.sequences] == [
        ["first_0"],
        ["first_1"],
        ["second_0"],
        ["second_1"],
    ]


@dataclasses.dataclass
class _Structure:
    atom37_positions: np.ndarray
    atom37_mask: np.ndarray

    def __len__(self) -> int:
        return self.atom37_positions.shape[0]


def test_atom_indexer_selects_the_declared_property_and_axis() -> None:
    positions = np.arange(2 * 37 * 3, dtype=np.float32).reshape(  # (l=2, a=37, xyz=3)
        2, 37, 3
    )
    structure = _Structure(positions, np.ones((2, 37), dtype=bool))  # mask: (l=2, a=37)
    indexer = AtomIndexer(structure, "atom37_positions", dim=1)
    np.testing.assert_array_equal(indexer["CA"], positions[:, 1])
    np.testing.assert_array_equal(indexer[["N", "C"]], positions[:, [0, 2]])


def test_atom_name_selection_matches_for_numpy_and_torch() -> None:
    positions = np.arange(2 * 37 * 3, dtype=np.float32).reshape(  # (l=2, a=37, xyz=3)
        2, 37, 3
    )
    expected = positions[:, [0, 1, 2]]  # (l=2, a_selected=3, xyz=3)
    np.testing.assert_array_equal(
        index_by_atom_name(positions, ["N", "CA", "C"]),
        expected,
    )
    assert torch.equal(
        index_by_atom_name(torch.from_numpy(positions), ["N", "CA", "C"]),
        torch.from_numpy(expected),
    )


def test_pae_bin_expectation_and_tm_score_are_exact_for_uniform_logits() -> None:
    logits = torch.zeros((1, 2, 2, 4), dtype=torch.float64)  # (b=1, l=2, l=2, c=4)
    mask = torch.ones((1, 2), dtype=torch.bool)  # (b=1, l=2)
    pae = compute_predicted_aligned_error(logits, mask)  # (b=1, l=2, l=2)
    assert torch.equal(
        pae,
        torch.full((1, 2, 2), 31.0, dtype=torch.float64),  # (b=1, l=2, l=2)
    )

    centers = torch.tensor([7.75, 23.25, 38.75, 54.25], dtype=torch.float64)  # (c=4,)
    d0 = 1.24 * 4 ** (1 / 3) - 1.8
    expected_tm = (1 / (1 + (centers / d0) ** 2)).mean().reshape(1)  # (b=1,)
    torch.testing.assert_close(compute_tm(logits, mask), expected_tm, rtol=2e-6, atol=1e-12)


def test_aligner_recovers_a_rigid_translation() -> None:
    mobile_positions = np.full((1, 37, 3), np.nan, dtype=np.float32)  # (l=1, a=37, xyz=3)
    mobile_positions[0, :3] = np.asarray(  # (a_selected=3, xyz=3)
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    mask = np.zeros((1, 37), dtype=bool)  # (l=1, a=37)
    mask[0, :3] = True  # (a_selected=3,)
    target_positions = mobile_positions.copy()  # (l=1, a=37, xyz=3)
    target_positions[mask] += np.asarray(  # (xyz=3,), broadcast over n_selected=3
        [2.0, -1.0, 3.0], dtype=np.float32
    )
    mobile = _Structure(mobile_positions, mask)
    target = _Structure(target_positions, mask)

    aligner = Aligner(mobile, target, only_use_backbone=True)
    aligned = aligner.apply(mobile)  # atom37_positions: (l=1, a=37, xyz=3)
    assert aligner.rmsd < 1e-7
    np.testing.assert_allclose(aligned.atom37_positions[mask], target_positions[mask], atol=1e-6)
