"""Build confidence targets from local AtlasFold-Data `rcsb` and `rcsb_multimer` Parquet files.

Monomer targets come from `rcsb` chains and multi-chain targets from `rcsb_multimer` assemblies.
Complexes larger than a token budget become whole-chain spatial subsets: chains are added in order
of their closest C-alpha contact to the growing subset. Residue-level crops are not used because the
inference API folds exactly the chain sequences it receives. Coordinates of every target are
stored in memory-mapped atom14 arrays so training reads them without Parquet decoding.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dataclasses import dataclass
from pathlib import Path


MAX_RESOLUTION_ANGSTROM = 4.0  # production ESMFold2 confidence-loss filter
TOKEN_BUDGET = 1024
LONG_TOKEN_BUDGET = 2048
MIN_RESOLVED_CA = 4
STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
COLUMNS = ["id", "sequence", "exp_resolution", "exp_pdb_id", "exp_release_date", "atom14_positions"]

TARGET_SCHEMA = pa.schema(
    [
        ("target_id", pa.string()),
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("variant", pa.string()),  # "standard" (at most 1,024 tokens) or "long" (1,025 to 2,048)
        ("chain_indices", pa.list_(pa.int32())),
        ("sequences", pa.list_(pa.string())),
        ("num_chains", pa.int32()),
        ("num_tokens", pa.int32()),
        ("resolution", pa.float64()),
        ("pdb_id", pa.string()),
        ("release_date", pa.string()),
        ("positions_file", pa.string()),
        ("residue_offset", pa.int64()),
    ]
)


@dataclass(frozen=True)
class ChainSubset:
    chain_indices: tuple[int, ...]
    num_tokens: int


def _row_positions(column: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Flatten an Array3D batch column into (l_total, 14, 3) values and (n_rows + 1,) residue offsets."""
    # Arrow column of n_rows variable-length residue lists
    storage = column.storage if isinstance(column, pa.ExtensionArray) else column
    offsets = storage.offsets.to_numpy()  # (n_rows + 1,)
    values = storage.flatten()  # flattened Arrow array; list nesting decreases until scalar coordinates
    while pa.types.is_list(values.type):
        values = values.flatten()  # flattened Arrow array; list nesting decreases until scalar coordinates
    positions = values.to_numpy(zero_copy_only=False).reshape(-1, 14, 3)  # (l_total, 14, 3)
    return positions, offsets - offsets[0]  # (l_total, 14, 3), (n_rows + 1,)


def spatial_chain_subset(chain_ca: list[np.ndarray], lengths: list[int], budget: int, seed: int) -> ChainSubset:
    """Grow a whole-chain subset from a seeded chain by nearest resolved C-alpha contact.

    `chain_ca[i]` holds the resolved C-alpha coordinates (n_i, 3) of chain i. Chains that no longer fit
    the token budget are skipped, so a later, shorter chain can still be added.
    """
    rng = np.random.default_rng(seed)
    candidates = [index for index, length in enumerate(lengths) if length <= budget]
    if not candidates:
        return ChainSubset((), 0)
    chosen = [int(rng.choice(candidates))]
    tokens = lengths[chosen[0]]
    remaining = [index for index in range(len(lengths)) if index != chosen[0]]
    distances = {index: np.inf for index in remaining}
    while remaining:
        newest = chain_ca[chosen[-1]]  # (n_newest, 3)
        for index in remaining:
            # () minimum distance between resolved CA coordinates
            gap = np.sqrt(((chain_ca[index][:, None, :] - newest[None, :, :]) ** 2).sum(-1)).min()
            distances[index] = min(distances[index], float(gap))
        fitting = [index for index in remaining if tokens + lengths[index] <= budget]
        if not fitting:
            break
        nearest = min(fitting, key=lambda index: distances[index])
        chosen.append(nearest)
        tokens += lengths[nearest]
        remaining.remove(nearest)
    return ChainSubset(tuple(sorted(chosen)), tokens)


def _row_seed(identifier: str) -> int:
    return int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "little")


def _targets_for_row(source: str, identifier: str, sequences: list[str], positions: np.ndarray) -> list[tuple[str, ChainSubset]]:
    """Return (variant, subset) choices for one eligible row; `positions` is (l, 14, 3)."""
    lengths = [len(sequence) for sequence in sequences]
    total = sum(lengths)
    if source == "rcsb":
        if total <= TOKEN_BUDGET:
            return [("standard", ChainSubset((0,), total))]
        return [("long", ChainSubset((0,), total))] if total <= LONG_TOKEN_BUDGET else []
    if len(sequences) < 2:
        return []
    starts = np.cumsum([0, *lengths])  # (chains + 1,) cumulative residue offsets
    chain_ca = []
    for index in range(len(sequences)):
        ca = positions[starts[index] : starts[index + 1], 1]  # (l_c, 3)
        chain_ca.append(ca[np.isfinite(ca).all(-1)])  # (resolved chain CA atoms, 3)
    choices = []
    standard = ChainSubset(tuple(range(len(sequences))), total) if total <= TOKEN_BUDGET else spatial_chain_subset(
        chain_ca, lengths, TOKEN_BUDGET, _row_seed(identifier)
    )
    if len(standard.chain_indices) >= 2:
        choices.append(("standard", standard))
    if total > TOKEN_BUDGET:
        long = ChainSubset(tuple(range(len(sequences))), total) if total <= LONG_TOKEN_BUDGET else spatial_chain_subset(
            chain_ca, lengths, LONG_TOKEN_BUDGET, _row_seed(identifier + "#long")
        )
        if len(long.chain_indices) >= 2 and long.num_tokens > TOKEN_BUDGET:
            choices.append(("long", long))
    return choices


def _eligible(sequences: list[str], positions: np.ndarray, resolution: float | None) -> bool:
    # positions: (residues across all chains, 14, 3).
    if resolution is None or not 0 < resolution <= MAX_RESOLUTION_ANGSTROM:
        return False
    if any(not set(sequence) <= STANDARD_RESIDUES for sequence in sequences):
        return False
    start = 0
    for sequence in sequences:
        ca = positions[start : start + len(sequence), 1]  # (l_c, 3)
        if int(np.isfinite(ca).all(-1).sum()) < MIN_RESOLVED_CA:
            return False
        start += len(sequence)
    return True


def _pool_file(task: tuple[str, str, str]) -> dict[str, object]:
    """Select targets from one Parquet file and write their atom14 positions."""
    source, parquet_path, output_dir = task
    stem = f"{source}-{Path(parquet_path).stem}"
    positions_path = Path(output_dir) / "positions" / f"{stem}.npy"
    rows: list[dict[str, object]] = []
    chunks: list[np.ndarray] = []
    offset = 0
    counts = {"rows": 0, "eligible": 0}
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=512, columns=COLUMNS):
        # (batch residues, 14, 3), (batch rows + 1,)
        positions, residue_offsets = _row_positions(batch["atom14_positions"])
        for index in range(batch.num_rows):
            counts["rows"] += 1
            identifier = batch["id"][index].as_py()
            sequences = batch["sequence"][index].as_py().split(":")
            row_positions = positions[residue_offsets[index] : residue_offsets[index + 1]]  # (l, 14, 3)
            resolution = batch["exp_resolution"][index].as_py()
            if not _eligible(sequences, row_positions, resolution):
                continue
            counts["eligible"] += 1
            starts = np.cumsum([0, *[len(sequence) for sequence in sequences]])  # (chains + 1,)
            for variant, subset in _targets_for_row(source, identifier, sequences, row_positions):
                # per selected chain (chain residues, 14, 3)
                selected = [row_positions[starts[chain] : starts[chain + 1]] for chain in subset.chain_indices]
                chunks.extend(selected)
                rows.append(
                    {
                        "target_id": f"{source}/{identifier}#{variant}",
                        "source": source,
                        "source_id": identifier,
                        "variant": variant,
                        "chain_indices": list(subset.chain_indices),
                        "sequences": [sequences[chain] for chain in subset.chain_indices],
                        "num_chains": len(subset.chain_indices),
                        "num_tokens": subset.num_tokens,
                        "resolution": resolution,
                        "pdb_id": batch["exp_pdb_id"][index].as_py(),
                        "release_date": batch["exp_release_date"][index].as_py(),
                        "positions_file": positions_path.name,
                        "residue_offset": offset,
                    }
                )
                offset += subset.num_tokens
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.concatenate(chunks) if chunks else np.empty((0, 14, 3), dtype=np.float32)  # (l_file, 14, 3)
    np.save(positions_path, stacked.astype(np.float32, copy=False))
    table_path = Path(output_dir) / "tables" / f"{stem}.parquet"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=TARGET_SCHEMA), table_path)
    return {"file": stem, **counts, "targets": len(rows), "residues": offset}


def build_pool(atlasfold_hub: Path, output_dir: Path, workers: int) -> dict[str, object]:
    """Scan the structure configs and write `targets.parquet` plus per-file position arrays."""
    import datasets  # noqa: F401 - registers the Array3D extension type for pyarrow reads

    tasks = [
        (source, str(path), str(output_dir))
        for source in ("rcsb", "rcsb_multimer")
        for path in sorted((atlasfold_hub / "data" / source).glob("*.parquet"))
    ]
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers) as pool:
        reports = pool.map(_pool_file, tasks)
    table = pa.concat_tables(pq.read_table(path) for path in sorted((output_dir / "tables").glob("*.parquet")))
    pq.write_table(table, output_dir / "targets.parquet")
    variants = table.group_by(["source", "variant"]).aggregate([("target_id", "count")]).to_pylist()
    summary = {
        "rows": sum(int(report["rows"]) for report in reports),
        "eligible_rows": sum(int(report["eligible"]) for report in reports),
        "targets": table.num_rows,
        "targets_by_source_and_variant": variants,
        "max_resolution_angstrom": MAX_RESOLUTION_ANGSTROM,
        "token_budget": TOKEN_BUDGET,
        "long_token_budget": LONG_TOKEN_BUDGET,
    }
    (output_dir / "pool-report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def load_positions(output_dir: Path, target: dict[str, object]) -> np.ndarray:
    """Return the (l, 14, 3) float32 positions of one pool target."""
    # (file residues, 14, 3)
    array = np.load(output_dir / "positions" / str(target["positions_file"]), mmap_mode="r")
    start = int(target["residue_offset"])  # type: ignore[arg-type]
    return np.array(array[start : start + int(target["num_tokens"])])  # type: ignore[arg-type]  # (l, 14, 3)
