"""Split confidence targets into train, validation, and an untouched test set.

Chain sequences are clustered once with MMseqs2 at 40% identity and 80% coverage. Targets that
share any chain cluster belong to one component, and held-out sets take whole components so that
no held-out chain has a training homolog under that definition. Held-out targets never come from a
component that contains a pilot record. Training excludes targets sharing a cluster with a pilot
final-test chain, the cluster-level disjointness the pilot's own splits used, so the pilot test stays
usable as a secondary benchmark. Component-level exclusion there would drop the largest component,
which joins about a third of all targets through transitive homology.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from collections import Counter
from pathlib import Path


SEED = 20260917
MAX_HELDOUT_COMPONENT_TARGETS = 50  # large families stay in training instead of leaving with one pick
TEST_QUOTAS = {
    "monomer_short": 64,
    "monomer_medium": 64,
    "monomer_long": 64,
    "dimer_homo": 96,
    "dimer_hetero": 96,
    "complex": 128,
    "long": 64,
}
VALIDATION_QUOTAS = {
    "monomer_short": 32,
    "monomer_medium": 32,
    "monomer_long": 32,
    "dimer_homo": 48,
    "dimer_hetero": 48,
    "complex": 64,
}


def stratum(variant: str, sequences: list[str], num_tokens: int) -> str:
    if variant == "long":
        return "long"
    if len(sequences) == 1:
        return "monomer_short" if num_tokens <= 256 else "monomer_medium" if num_tokens <= 512 else "monomer_long"
    if len(sequences) == 2:
        return "dimer_homo" if sequences[0] == sequences[1] else "dimer_hetero"
    return "complex"


def cluster_sequences(sequences: list[str], mmseqs: str, threads: int) -> dict[str, int]:
    """Map each sequence to a cluster number with `mmseqs easy-cluster`."""
    with tempfile.TemporaryDirectory() as work:
        fasta = Path(work, "sequences.fasta")
        fasta.write_text("".join(f">{index}\n{sequence}\n" for index, sequence in enumerate(sequences)), encoding="ascii")
        subprocess.run(
            [mmseqs, "easy-cluster", str(fasta), str(Path(work, "clusters")), str(Path(work, "tmp")),
             "--min-seq-id", "0.4", "-c", "0.8", "--cov-mode", "0", "--threads", str(threads)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        representatives: dict[str, int] = {}
        cluster_of: dict[str, int] = {}
        for line in Path(work, "clusters_cluster.tsv").read_text(encoding="ascii").splitlines():
            representative, member = line.split("\t")
            cluster_of[sequences[int(member)]] = representatives.setdefault(representative, len(representatives))
    missing = set(sequences) - cluster_of.keys()
    if missing:
        raise ValueError(f"MMseqs2 left {len(missing)} sequences unclustered")
    return cluster_of


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        self.parent[self.find(left)] = self.find(right)


def build_splits(pool_dir: Path, pilot_records: Path, output_dir: Path, mmseqs: str, threads: int) -> dict[str, object]:
    targets = pq.read_table(pool_dir / "targets.parquet").to_pylist()
    pilot = json.loads(pilot_records.read_text(encoding="utf-8"))
    pilot_sequences = {
        split: {chain["sequence"] for record in pilot if record["split"] in splits for chain in record["chains"]}
        for split, splits in (("any", {"train", "validation", "final_test"}), ("final_test", {"final_test"}))
    }
    sequences = sorted({sequence for target in targets for sequence in target["sequences"]} | pilot_sequences["any"])
    cluster_of = cluster_sequences(sequences, mmseqs, threads)

    components = UnionFind(max(cluster_of.values()) + 1)
    for target in targets:
        clusters = [cluster_of[sequence] for sequence in target["sequences"]]
        for cluster in clusters[1:]:
            components.union(clusters[0], cluster)
    pilot_any = {components.find(cluster_of[sequence]) for sequence in pilot_sequences["any"]}
    pilot_final_clusters = {cluster_of[sequence] for sequence in pilot_sequences["final_test"]}
    for target in targets:
        target["clusters"] = [cluster_of[sequence] for sequence in target["sequences"]]
        target["component"] = components.find(target["clusters"][0])
        target["stratum"] = stratum(target["variant"], target["sequences"], target["num_tokens"])
    component_sizes = Counter(target["component"] for target in targets if target["variant"] == "standard")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(targets))
    heldout_components: dict[int, str] = {}
    for split, quotas in (("test", TEST_QUOTAS), ("validation", VALIDATION_QUOTAS)):
        filled = Counter()
        for index in order:
            target = targets[index]
            component = target["component"]
            if (
                filled[target["stratum"]] < quotas.get(target["stratum"], 0)
                and component not in heldout_components
                and component not in pilot_any
                and component_sizes.get(component, 0) <= MAX_HELDOUT_COMPONENT_TARGETS
            ):
                target["split"] = split
                heldout_components[component] = split
                filled[target["stratum"]] += 1
        if filled != Counter(quotas):
            raise ValueError(f"{split} quotas not met: {dict(filled)}")

    train_clusters = Counter()
    for target in targets:
        if "split" in target:
            continue
        if (
            target["variant"] == "standard"
            and target["component"] not in heldout_components
            and not pilot_final_clusters.intersection(target["clusters"])
        ):
            target["split"] = "train"
            train_clusters.update(target["clusters"])
        else:
            target["split"] = "unused"
    for target in targets:
        # Mean inverse cluster size over chains, so large families do not dominate sampling.
        target["weight"] = (
            float(np.mean([1.0 / train_clusters[cluster] for cluster in target["clusters"]]))
            if target["split"] == "train"
            else 0.0
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "targets.parquet"
    pq.write_table(pa.Table.from_pylist(targets), table_path)
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    counts = Counter((target["split"], target["stratum"]) for target in targets)
    largest = component_sizes.most_common(5)
    report = {
        "status": "verified",
        "targets_sha256": digest,
        "sequences": len(sequences),
        "clusters": max(cluster_of.values()) + 1,
        "components": len(component_sizes),
        "largest_components": [{"component": component, "standard_targets": size} for component, size in largest],
        "counts": {f"{split}/{name}": count for (split, name), count in sorted(counts.items())},
        "train_targets": sum(target["split"] == "train" for target in targets),
        "pilot_components_excluded_from_heldout": len(pilot_any),
        "pilot_final_test_clusters_excluded_from_train": len(pilot_final_clusters),
        "mmseqs": {"min_seq_id": 0.4, "coverage": 0.8, "cov_mode": 0},
        "seed": SEED,
    }
    (output_dir / "split-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def load_split(output_dir: Path) -> list[dict[str, object]]:
    """Read split targets after checking the table against its recorded digest."""
    report = json.loads((output_dir / "split-report.json").read_text(encoding="utf-8"))
    table_path = output_dir / "targets.parquet"
    if report.get("status") != "verified" or hashlib.sha256(table_path.read_bytes()).hexdigest() != report["targets_sha256"]:
        raise ValueError("split table differs from its verified report")
    return pq.read_table(table_path).to_pylist()
