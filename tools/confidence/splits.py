"""Exclude PDB and chain-sequence overlap from the confidence adaptation splits."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess

from pathlib import Path


COUNTS = {"train": {1: 512, 2: 512}, "validation": {1: 64, 2: 64}, "final_test": {1: 64, 2: 64}}


def select_disjoint(
    records: list[dict], clusters: dict[tuple[str, int], str], counts: dict = COUNTS
) -> list[dict]:
    """Select complete targets, reserving every component chain's cluster."""
    used_clusters, used_pdbs = set(), set()
    selected = []
    # Reserve the official validation pool, then draw a held-out final test set
    # and training set from separate clusters in the experimental structure pool.
    for split in ("validation", "final_test", "train"):
        for chain_count, required in counts[split].items():
            pool = sorted(
                (
                    record
                    for record in records
                    if (
                        record["split"] == split
                        or (record["split"] == "pool" and split in {"train", "final_test"})
                    )
                    and len(record["chains"]) == chain_count
                ),
                key=lambda record: record["id"],
            )
            random.Random(17).shuffle(pool)
            chosen = 0
            for record in pool:
                pdb = str(record["pdb_id"]).lower()
                cluster_ids = {clusters[record["id"], index] for index in range(chain_count)}
                if pdb in used_pdbs or cluster_ids & used_clusters:
                    continue
                selected.append({**record, "split": split})
                used_pdbs.add(pdb)
                used_clusters.update(cluster_ids)
                chosen += 1
                if chosen == required:
                    break
            if chosen != required:
                raise ValueError(
                    f"Only {chosen}/{required} disjoint {split} targets with {chain_count} chains"
                )
    return selected


def chain_clusters(records: list[dict], directory: Path) -> dict[tuple[str, int], str]:
    directory.mkdir(parents=True, exist_ok=True)
    sequences = {}
    identifiers = {}
    for record in sorted(records, key=lambda item: item["id"]):
        for index, chain in enumerate(record["chains"]):
            name = f"c{len(sequences):07d}"
            identifiers[name] = (record["id"], index)
            sequences[name] = chain["sequence"]
    fasta = directory / "chains.fasta"
    fasta.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in sequences.items()))
    matches = directory / "matches.tsv"
    command = [
        "mmseqs",
        "easy-search",
        str(fasta),
        str(fasta),
        str(matches),
        str(directory / "temporary"),
        "--min-seq-id",
        "0.4",
        "-c",
        "0.8",
        "--cov-mode",
        "1",
        "--format-output",
        "query,target",
        "--threads",
        "4",
        "-s",
        "7.5",
        "--max-seqs",
        str(len(sequences)),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    (directory / "mmseqs.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("MMseqs2 sequence exclusion failed; see the remote mmseqs.log")
    parents = {name: name for name in sequences}

    def find(name: str) -> str:
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    for line in matches.read_text().splitlines():
        query, target = line.split("\t")
        left, right = find(query), find(target)
        parents[max(left, right)] = min(left, right)
    version = subprocess.check_output(["mmseqs", "version"], text=True).strip()
    (directory / "settings.json").write_text(
        json.dumps(
            {
                "command": command,
                "version": version,
                "identity": 0.4,
                "target_coverage": 0.8,
                "all_vs_all": True,
            },
            indent=2,
        )
        + "\n"
    )
    return {identifiers[name]: find(name) for name in sequences}


def finalize_splits(data_root: Path) -> dict:
    records = json.loads((data_root / "candidates.json").read_text())
    clusters = chain_clusters(records, data_root / "sequence_exclusion")
    selected = select_disjoint(records, clusters)
    destination = data_root / "records.json"
    destination.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    report = {
        "status": "verified",
        "records_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "counts": {
            split: {
                str(n): sum(r["split"] == split and len(r["chains"]) == n for r in selected)
                for n in (1, 2)
            }
            for split in COUNTS
        },
        "sequence_identity": 0.4,
        "target_coverage": 0.8,
        "scope": "adaptation splits; original training homologs are not fully known",
    }
    (data_root / "split-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
