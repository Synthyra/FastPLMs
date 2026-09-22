"""CPU tests for bounded AtlasFold confidence-data preparation."""

from __future__ import annotations

import zipfile
import io

import numpy as np
import pytest

from tools.confidence.data import (
    SelectionSpec,
    UnsupportedStructureError,
    _normalize_record,
    np_to_npz_bytes,
    safe_extract_archive,
    select_balanced_records,
    validate_structure_npz,
    _normalize_chain,
)


@pytest.mark.parametrize("atom_count", [1, 4, 6])
def test_compact_coordinates_reject_wrong_count_before_assignment(atom_count):
    archive = io.BytesIO()
    np.savez(archive, sequence="A", coordinates=np.zeros((atom_count, 3)))
    with pytest.raises(UnsupportedStructureError, match="coordinate count mismatch"):
        _normalize_chain(archive.getvalue(), chain_index=0)


def _record(index: int, chain_count: int = 1) -> dict[str, object]:
    chains = [
        {"id": str(chain), "sequence": "A" * (64 // chain_count)} for chain in range(chain_count)
    ]
    return {
        "pdb_id": f"pdb{index}",
        "cluster_id": f"cluster{index}",
        "sequence": "".join(str(chain["sequence"]) for chain in chains),
        "chains": chains,
        "resolution": 1.5,
        "structure_path": f"structures/pdb{index}.npz",
    }


@pytest.mark.parametrize("shape", [(), (1,)])
def test_single_sequence_field_accepts_official_one_element_array(shape):
    payload = np_to_npz_bytes(
        {
            "sequence": np.array("A").reshape(shape),
            "coordinates": np.zeros((5, 3), dtype=np.float32),
        }
    )
    sequence, arrays = _normalize_chain(payload, chain_index=0)
    assert sequence == "A"
    assert arrays["coordinates"].shape == (1, 14, 3)


def test_balanced_selection_is_seeded_and_excludes_clusters() -> None:
    records = [_record(index, chain_count) for chain_count in (1, 2) for index in range(8)]
    spec = SelectionSpec("train", count=4, monomers=2, dimers=2)
    first = select_balanced_records(records, spec)
    second = select_balanced_records(records, spec)
    assert first == second
    assert sum(len(record["chains"]) == 1 for record in first) == 2
    assert sum(len(record["chains"]) == 2 for record in first) == 2


def test_selection_fails_closed_without_cluster_ids() -> None:
    record = _record(1)
    del record["cluster_id"]
    with pytest.raises(ValueError, match="cluster_id"):
        select_balanced_records([record], SelectionSpec("train", 1, 1, 0))


def test_structure_schema_is_validated(tmp_path) -> None:
    path = tmp_path / "structure.npz"
    np.savez(
        path,
        coordinates=np.zeros((4, 14, 3), dtype=np.float32),
        atom_names=np.full((4, 14), "CA"),
        chain_index=np.zeros(4, dtype=np.int32),
        residue_index=np.arange(1, 5, dtype=np.int32),
    )
    shapes = validate_structure_npz(path)
    assert shapes["coordinates"] == (4, 14, 3)


def test_structure_indices_and_atom_names_are_checked_against_sequences(tmp_path) -> None:
    path = tmp_path / "structure.npz"
    names = np.full((2, 14), "", dtype="U4")
    names[0, :5] = ["N", "CA", "C", "O", "CB"]
    names[1, :5] = ["N", "CA", "C", "O", "CB"]
    np.savez(
        path,
        coordinates=np.zeros((2, 14, 3), dtype=np.float32),
        atom_names=names,
        chain_index=np.zeros(2, dtype=np.int32),
        residue_index=np.array([1, 2], dtype=np.int32),
    )
    validate_structure_npz(path, sequences=["AA"])

    names[1, 4] = "SG"
    np.savez(
        path,
        coordinates=np.zeros((2, 14, 3), dtype=np.float32),
        atom_names=names,
        chain_index=np.zeros(2, dtype=np.int32),
        residue_index=np.array([1, 2], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="invalid"):
        validate_structure_npz(path, sequences=["AA"])


def test_structure_indices_must_be_contiguous(tmp_path) -> None:
    path = tmp_path / "structure.npz"
    np.savez(
        path,
        coordinates=np.zeros((2, 14, 3), dtype=np.float32),
        atom_names=np.full((2, 14), "", dtype="U4"),
        chain_index=np.array([0, 2], dtype=np.int32),
        residue_index=np.array([1, 2], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="contiguous"):
        validate_structure_npz(path)


def _multimer_payload(*, num_chains: int = 2, extra_chain: bool = False) -> bytes:
    payload: dict[str, np.ndarray] = {
        "name": np.array("complex", dtype="S"),
        "num_chains": np.array(num_chains, dtype=np.int64),
    }
    for index in range(num_chains + int(extra_chain)):
        payload[f"{index}.name"] = np.array(f"chain{index}", dtype="S")
        payload[f"{index}.sequence"] = np.array("A" * 64, dtype="S")
        payload[f"{index}.coordinates"] = np.zeros((64 * 5, 3), dtype=np.float32)
        payload[f"{index}.b_factors"] = np.zeros(64 * 5, dtype=np.float32)
    return np_to_npz_bytes(payload)


def _multimer_metadata() -> dict[str, object]:
    return {
        "id": "complex",
        "chains": [
            {"id": "A", "num_residues": 64},
            {"id": "B", "num_residues": 64},
        ],
        "exp": {"pdb_id": "1abc", "resolution": 1.5},
    }


def test_multimer_payload_uses_official_chain_count_and_prefixes(tmp_path) -> None:
    record = _normalize_record(
        _multimer_metadata(),
        _multimer_payload(),
        source="rcsb_multimer",
        output_path=tmp_path / "complex.npz",
    )
    assert [chain["id"] for chain in record["chains"]] == ["A", "B"]


def test_multimer_payload_chain_count_mismatch_is_rejected(tmp_path) -> None:
    with pytest.raises(UnsupportedStructureError, match="chain count mismatch"):
        _normalize_record(
            _multimer_metadata(),
            _multimer_payload(num_chains=1),
            source="rcsb_multimer",
            output_path=tmp_path / "complex.npz",
        )


def test_multimer_payload_extra_chain_prefix_is_rejected(tmp_path) -> None:
    with pytest.raises(UnsupportedStructureError, match="prefixes"):
        _normalize_record(
            _multimer_metadata(),
            _multimer_payload(extra_chain=True),
            source="rcsb_multimer",
            output_path=tmp_path / "complex.npz",
        )


def test_safe_extraction_rejects_traversal(tmp_path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="escapes"):
        safe_extract_archive(archive, tmp_path / "out")


def test_extraction_can_skip_template_databases(tmp_path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("rcsb_multimer/structure.lmdb/data.mdb", "structures")
        writer.writestr("rcsb_multimer/template.lmdb/data.mdb", "templates")
    destination = tmp_path / "out"
    safe_extract_archive(archive, destination, excluded_directories=frozenset({"template.lmdb"}))
    assert (destination / "rcsb_multimer/structure.lmdb/data.mdb").read_text() == "structures"
    assert not (destination / "rcsb_multimer/template.lmdb").exists()
