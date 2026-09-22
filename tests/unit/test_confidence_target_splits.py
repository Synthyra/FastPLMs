"""Confidence v2 splits: homology-disjoint held-out sets, pilot exclusions, weights, and digests."""

import json
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.confidence import target_splits


def _target(name: str, sequences: list[str], tokens: int = 100) -> dict[str, object]:
    return {
        "target_id": name,
        "variant": "standard",
        "sequences": sequences,
        "num_chains": len(sequences),
        "num_tokens": tokens,
    }


@pytest.fixture
def split_inputs(tmp_path, monkeypatch):
    targets = [_target(f"single-{index}", [f"S{index}"]) for index in range(30)]
    targets += [_target(f"family-{index}", [f"F{index}"]) for index in range(3)]
    targets += [_target("pilot-test-homolog", ["P0"]), _target("pilot-train-homolog", ["T0"])]
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    pq.write_table(pa.Table.from_pylist(targets), pool_dir / "targets.parquet")
    pilot = [
        {"split": "final_test", "chains": [{"sequence": "P1"}]},
        {"split": "train", "chains": [{"sequence": "T1"}]},
    ]
    pilot_path = tmp_path / "records.json"
    pilot_path.write_text(json.dumps(pilot), encoding="utf-8")

    def first_letter_family(sequences, mmseqs, threads):
        # One cluster per singleton; the F family, P pair, and T pair each share a cluster.
        names = sorted({sequence if sequence.startswith("S") else sequence[0] for sequence in sequences})
        return {sequence: names.index(sequence if sequence.startswith("S") else sequence[0]) for sequence in sequences}

    monkeypatch.setattr(target_splits, "cluster_sequences", first_letter_family)
    monkeypatch.setattr(target_splits, "TEST_QUOTAS", {"monomer_short": 5})
    monkeypatch.setattr(target_splits, "VALIDATION_QUOTAS", {"monomer_short": 4})
    monkeypatch.setattr(target_splits, "MAX_HELDOUT_COMPONENT_TARGETS", 2)
    return pool_dir, pilot_path, tmp_path / "splits"


def test_splits_are_homology_disjoint_and_respect_pilot_records(split_inputs):
    pool_dir, pilot_path, output_dir = split_inputs
    report = target_splits.build_splits(pool_dir, pilot_path, output_dir, mmseqs="unused", threads=1)
    targets = {target["target_id"]: target for target in target_splits.load_split(output_dir)}

    assert report["counts"]["test/monomer_short"] == 5
    assert report["counts"]["validation/monomer_short"] == 4
    heldout_clusters = {cluster for target in targets.values() if target["split"] in {"test", "validation"} for cluster in target["clusters"]}
    train_clusters = {cluster for target in targets.values() if target["split"] == "train" for cluster in target["clusters"]}
    assert heldout_clusters.isdisjoint(train_clusters)

    # Homologs of pilot chains never enter held-out sets; only final-test homologs leave training.
    assert targets["pilot-test-homolog"]["split"] == "unused"
    assert targets["pilot-train-homolog"]["split"] == "train"
    # Families larger than the held-out component limit stay in training, weighted by cluster size.
    assert {targets[f"family-{index}"]["split"] for index in range(3)} == {"train"}
    assert targets["family-0"]["weight"] == pytest.approx(1 / 3)
    assert targets["single-0"]["weight"] in (0.0, pytest.approx(1.0))


def test_load_split_rejects_a_changed_table(split_inputs):
    pool_dir, pilot_path, output_dir = split_inputs
    target_splits.build_splits(pool_dir, pilot_path, output_dir, mmseqs="unused", threads=1)
    table = pq.read_table(output_dir / "targets.parquet")
    pq.write_table(table.slice(1), output_dir / "targets.parquet")
    with pytest.raises(ValueError, match="differs"):
        target_splits.load_split(output_dir)


def test_strata_follow_chain_count_length_and_symmetry():
    assert target_splits.stratum("standard", ["A" * 10], 256) == "monomer_short"
    assert target_splits.stratum("standard", ["A" * 10], 257) == "monomer_medium"
    assert target_splits.stratum("standard", ["A" * 10], 513) == "monomer_long"
    assert target_splits.stratum("standard", ["AC", "AC"], 4) == "dimer_homo"
    assert target_splits.stratum("standard", ["AC", "CA"], 4) == "dimer_hetero"
    assert target_splits.stratum("standard", ["A", "C", "D"], 3) == "complex"
    assert target_splits.stratum("long", ["A"], 1500) == "long"
