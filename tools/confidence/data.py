"""Prepare bounded AtlasFold-derived structure records for confidence training.

The module deliberately does not import AtlasFold or a model runtime.  Modal
CPU workers can use it after downloading the four pinned archives described by
an external manifest.  Structure arrays use the compact atom14 contract:
``coordinates`` has shape ``(residue, 14, 3)``, ``atom_names`` has shape
``(residue, 14)``, and chain and residue indices have shape ``(residue,)``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import zipfile

import numpy as np

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random


ATLASFOLD_REVISION = "444f376d85b9954a5f2f5f3f8b3cbcae1201ebb1"
ATLASFOLD_DATA_FOLDER_ID = "1EiRTKSUL3iD_MQ_0qmj5Sb-2KMh-3kmS"
DEFAULT_SEED = 17
MIN_RESIDUES = 64
MAX_RESIDUES = 384
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class UnsupportedStructureError(ValueError):
    """Raised when an AtlasFold structure cannot represent a protein record."""


@dataclass(frozen=True)
class ArchiveSpec:
    """One immutable AtlasFold archive entry from the download manifest."""

    name: str
    file_id: str
    url: str
    sha256: str | None = None


ATLASFOLD_ARCHIVES = (
    ArchiveSpec(
        "rcsb",
        "1TEH73v9oxA1oYYnsPZntHqES_04vEz8P",
        "https://drive.google.com/uc?export=download&id=1TEH73v9oxA1oYYnsPZntHqES_04vEz8P",
    ),
    ArchiveSpec(
        "rcsb_multimer",
        "1aN9zUL4JokQc0L6AWVUNlsnftQBi8pjr",
        "https://drive.google.com/uc?export=download&id=1aN9zUL4JokQc0L6AWVUNlsnftQBi8pjr",
    ),
    ArchiveSpec(
        "cameo_val",
        "10fhgH7nnVA022nvN-v3bTg1Xor97t2Ne",
        "https://drive.google.com/uc?export=download&id=10fhgH7nnVA022nvN-v3bTg1Xor97t2Ne",
    ),
    ArchiveSpec(
        "rcsb_multimer_val",
        "17meo4uBvvFfB2M-uor17KWwqQDYdSGFI",
        "https://drive.google.com/uc?export=download&id=17meo4uBvvFfB2M-uor17KWwqQDYdSGFI",
    ),
)


@dataclass(frozen=True)
class SelectionSpec:
    """Bounds used for deterministic train, validation, or test selection."""

    split: str
    count: int
    monomers: int
    dimers: int
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.count != self.monomers + self.dimers:
            raise ValueError("count must equal monomers + dimers")
        if min(self.count, self.monomers, self.dimers) < 0:
            raise ValueError("selection counts must be non-negative")


def archive_manifest(path: str | os.PathLike[str]) -> tuple[ArchiveSpec, ...]:
    """Read archive names, file IDs, URLs, and optional hashes without secrets."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    archives = payload.get("archives")
    if not isinstance(archives, list) or not archives:
        raise ValueError("manifest must contain a non-empty archives list")
    specs: list[ArchiveSpec] = []
    for entry in archives:
        if not isinstance(entry, Mapping):
            raise ValueError("each archive entry must be an object")
        name, file_id, url = (entry.get(key) for key in ("name", "file_id", "url"))
        if not all(isinstance(value, str) and value for value in (name, file_id, url)):
            raise ValueError("archive entries require non-empty name, file_id, and url")
        specs.append(ArchiveSpec(name, file_id, url, entry.get("sha256")))
    return tuple(specs)


def read_atlasfold_manifest(
    dataset_root: str | os.PathLike[str], *, filtered: bool = True
) -> list[dict[str, object]]:
    """Read AtlasFold's native msgpack manifest without importing AtlasFold."""

    try:
        import msgpack
    except ImportError as error:
        raise RuntimeError("msgpack is required to read AtlasFold manifests") from error
    root = Path(dataset_root)
    manifest_name = "manifest_confidence.msgpack" if filtered else "manifest.msgpack"
    path = root / manifest_name
    if not path.exists() and filtered:
        path = root / "manifest.msgpack"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        manifest = msgpack.unpackb(handle.read(), raw=False)
    if not isinstance(manifest, list) or not all(isinstance(item, dict) for item in manifest):
        raise ValueError(f"AtlasFold manifest must contain a list of objects: {path}")
    return manifest


def read_lmdb_value(lmdb_path: str | os.PathLike[str], key: str) -> bytes:
    """Read one immutable compressed NPZ value from an AtlasFold LMDB."""

    try:
        import lmdb
    except ImportError as error:
        raise RuntimeError("lmdb is required to read AtlasFold structures") from error
    environment = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    try:
        with environment.begin() as transaction:
            value = transaction.get(key.encode())
    finally:
        environment.close()
    if value is None:
        raise KeyError(f"AtlasFold LMDB key not found: {key}")
    return bytes(value)


def _read_lmdb_values(
    lmdb_path: Path, entries: Sequence[Mapping[str, object]]
) -> list[tuple[Mapping[str, object], bytes]]:
    """Read a bounded metadata sample in one read-only LMDB transaction."""

    try:
        import lmdb
    except ImportError as error:
        raise RuntimeError("lmdb is required to read AtlasFold structures") from error
    environment = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    try:
        with environment.begin() as transaction:
            values: list[tuple[Mapping[str, object], bytes]] = []
            for metadata in entries:
                key = str(metadata["id"])
                value = transaction.get(key.encode())
                if value is None:
                    raise KeyError(f"AtlasFold LMDB key not found: {key}")
                values.append((metadata, bytes(value)))
            return values
    finally:
        environment.close()


def _atom14_names(sequence: str) -> np.ndarray:
    atom14 = {
        "A": ("N", "CA", "C", "O", "CB"),
        "C": ("N", "CA", "C", "O", "CB", "SG"),
        "D": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
        "E": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
        "F": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
        "G": ("N", "CA", "C", "O"),
        "H": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
        "I": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
        "K": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
        "L": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
        "M": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
        "N": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
        "P": ("N", "CA", "C", "O", "CB", "CG", "CD"),
        "Q": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
        "R": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
        "S": ("N", "CA", "C", "O", "CB", "OG"),
        "T": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
        "V": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
        "W": (
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "NE1",
            "CE2",
            "CE3",
            "CZ2",
            "CZ3",
            "CH2",
        ),
        "Y": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    }
    names = np.full((len(sequence), 14), "", dtype="U4")  # (residues, 14)
    for residue_index, residue in enumerate(sequence):
        names[residue_index, : len(atom14.get(residue, ()))] = atom14.get(residue, ())  # (residue atoms,)
    return names  # (residues, 14)


def _decode_npz(value: bytes) -> dict[str, np.ndarray]:
    import io

    with np.load(io.BytesIO(value), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def _normalize_chain(value: bytes, *, chain_index: int) -> tuple[str, dict[str, np.ndarray]]:
    arrays = _decode_npz(value)
    required = {"sequence", "coordinates"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise UnsupportedStructureError(
            f"chain {chain_index} payload is missing arrays: {', '.join(missing)}"
        )
    sequence_value = arrays["sequence"]  # one-element array; rank validated only by size
    if sequence_value.size != 1:
        raise UnsupportedStructureError(f"chain {chain_index} sequence must contain one string")
    sequence = sequence_value.item()
    if isinstance(sequence, bytes):
        sequence = sequence.decode("utf-8")
    if not isinstance(sequence, str) or not sequence:
        raise UnsupportedStructureError(f"chain {chain_index} sequence is not a non-empty string")
    unknown_residues = sorted(set(sequence) - STANDARD_AMINO_ACIDS)
    if unknown_residues:
        raise UnsupportedStructureError(
            f"chain {chain_index} contains unsupported residues {unknown_residues}: "
            f"sequence={sequence[:32]!r}"
        )
    compact_coordinates = arrays["coordinates"]  # (atoms, 3), validated below
    if compact_coordinates.ndim != 2 or compact_coordinates.shape[1] != 3:
        raise UnsupportedStructureError("AtlasFold chain coordinates must have shape (atom, 3)")
    coordinates = np.full((len(sequence), 14, 3), np.nan, dtype=np.float32)  # (residues, 14, 3)
    atom_names = _atom14_names(sequence)  # (residues, 14)
    expected_atoms = int((atom_names != "").sum())
    if len(compact_coordinates) != expected_atoms:
        raise UnsupportedStructureError(
            f"AtlasFold compact coordinate count mismatch: stored={len(compact_coordinates)}, "
            f"expected={expected_atoms}, chain={chain_index}"
        )
    cursor = 0
    for residue_index, names in enumerate(atom_names):
        valid = names != ""  # (14,)
        n_atoms = int(valid.sum())
        coordinates[residue_index, valid] = compact_coordinates[cursor : cursor + n_atoms]  # (n_atoms, 3)
        cursor += n_atoms
    if cursor != len(compact_coordinates):
        raise UnsupportedStructureError(
            "AtlasFold compact coordinate count mismatch: "
            f"chain={chain_index}, sequence_length={len(sequence)}, "
            f"stored_shape={tuple(compact_coordinates.shape)}, "
            f"expected_atoms={cursor}"
        )
    if not np.isfinite(compact_coordinates).all(axis=1).any():
        raise UnsupportedStructureError(f"chain {chain_index} contains no finite coordinates")
    return sequence, {
        "coordinates": coordinates,
        "atom_names": atom_names,
        "chain_index": np.full(len(sequence), chain_index, dtype=np.int32),  # (residues,)
        "residue_index": np.arange(1, len(sequence) + 1, dtype=np.int32),  # (residues,)
    }


def _normalize_record(
    metadata: Mapping[str, object],
    value: bytes,
    *,
    source: str,
    output_path: Path,
) -> dict[str, object]:
    chain_metadata = metadata.get("chains")
    if isinstance(chain_metadata, list):
        if len(chain_metadata) not in (1, 2):
            raise UnsupportedStructureError(
                f"AtlasFold payload has unsupported chain count: {len(chain_metadata)}"
            )
        arrays = _decode_npz(value)
        payload_count = arrays.get("num_chains")  # one-element integer array or None; validated below
        if (
            payload_count is None
            or payload_count.size != 1
            or not np.issubdtype(payload_count.dtype, np.integer)
        ):
            raise UnsupportedStructureError(
                "multimer NPZ must contain a one-element integer num_chains array"
            )
        if int(payload_count.item()) != len(chain_metadata):
            raise UnsupportedStructureError(
                "multimer metadata/payload chain count mismatch: "
                f"metadata={len(chain_metadata)}, payload={int(payload_count.item())}"
            )
        payload_indices = {
            int(key.split(".", 1)[0])
            for key in arrays
            if "." in key and key.split(".", 1)[0].isdigit()
        }
        expected_indices = set(range(len(chain_metadata)))
        if payload_indices != expected_indices:
            raise UnsupportedStructureError(
                "multimer NPZ chain prefixes do not match num_chains: "
                f"expected={sorted(expected_indices)}, observed={sorted(payload_indices)}"
            )
        chain_arrays: list[dict[str, np.ndarray]] = []
        sequences: list[str] = []
        for index, chain in enumerate(chain_metadata):
            if not isinstance(chain, Mapping):
                raise UnsupportedStructureError("multimer chain metadata must be objects")
            prefix = f"{index}."
            chain_value = {
                key.removeprefix(prefix): value
                for key, value in arrays.items()
                if key.startswith(prefix)
            }
            if "sequence" not in chain_value or "coordinates" not in chain_value:
                raise UnsupportedStructureError("multimer NPZ is missing chain arrays")
            sequence, normalized = _normalize_chain(np_to_npz_bytes(chain_value), chain_index=index)
            expected_length = chain.get("num_residues")
            if expected_length is not None and int(expected_length) != len(sequence):
                raise UnsupportedStructureError(
                    f"chain {index} metadata/payload residue count mismatch: "
                    f"metadata={expected_length}, payload={len(sequence)}"
                )
            expected_sequence = chain.get("sequence")
            if expected_sequence is not None:
                if isinstance(expected_sequence, bytes):
                    expected_sequence = expected_sequence.decode("utf-8")
                if expected_sequence != sequence:
                    raise UnsupportedStructureError(
                        f"chain {index} metadata/payload sequence mismatch"
                    )
            sequences.append(sequence)
            chain_arrays.append(normalized)
        # Concatenate the residue axis; retain (14, 3), (14,), or () per-field trailing axes.
        normalized_arrays = {
            key: np.concatenate([item[key] for item in chain_arrays], axis=0)
            for key in ("coordinates", "atom_names", "chain_index", "residue_index")
        }
        chains = [
            {
                "id": chain.get("id", chain.get("label_asym_id", str(index))),
                "sequence": sequence,
            }
            for index, (chain, sequence) in enumerate(zip(chain_metadata, sequences, strict=True))
        ]
    else:
        sequence, normalized_arrays = _normalize_chain(value, chain_index=0)
        chains = [{"id": metadata.get("label_asym_id", "A"), "sequence": sequence}]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **normalized_arrays)
    experiment = metadata.get("exp") or {}
    return {
        "id": metadata["id"],
        "pdb_id": experiment.get("pdb_id", str(metadata["id"]).split("_")[0]),
        "chains": chains,
        "sequence": ":".join(chain["sequence"] for chain in chains),
        "resolution": experiment.get("resolution"),
        "release_date": experiment.get("release_date"),
        "cluster_id": metadata.get("cluster_id")
        or "__".join(str(chain.get("cluster_id")) for chain in chain_metadata or []),
        "structure_path": str(output_path),
        "structure_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "source": source,
        "source_metadata": dict(metadata),
    }


def _biologically_eligible(record: Mapping[str, object]) -> bool:
    """Apply sequence, length, resolution, and resolved-CA checks to a cache."""

    chains = record.get("chains")
    sequence = record.get("sequence")
    resolution = record.get("resolution")
    if (
        not isinstance(chains, list)
        or not isinstance(sequence, str)
        or not isinstance(resolution, int | float)
        or not 0.1 <= resolution <= 3.0
        or not 64 <= sum(len(str(chain.get("sequence", ""))) for chain in chains) <= 384
    ):
        return False
    if len(chains) not in (1, 2) or any(
        not isinstance(chain, Mapping)
        or not isinstance(chain.get("sequence"), str)
        or not chain["sequence"]
        or any(residue not in STANDARD_AMINO_ACIDS for residue in chain["sequence"])
        for chain in chains
    ):
        return False
    validate_structure_npz(
        record["structure_path"], sequences=[chain["sequence"] for chain in chains]
    )
    with np.load(str(record["structure_path"]), allow_pickle=False) as arrays:
        coordinates = arrays["coordinates"]  # (residues, 14, 3)
        chain_index = arrays["chain_index"]  # (residues,)
        for index in range(len(chains)):
            if int(np.isfinite(coordinates[chain_index == index, 1, :]).all(axis=-1).sum()) < 4:
                return False
    return True


def np_to_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize a chain-array mapping for reuse by the normalizer."""

    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def prepare(
    data_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    train: SelectionSpec | None = None,
    validation: SelectionSpec | None = None,
    candidate_multiplier: int = 4,
    return_candidates: bool = False,
) -> dict[str, list[dict[str, object]]]:
    """Materialize normalized records from extracted AtlasFold archives.

    The input root must contain ``rcsb`` and ``rcsb_multimer`` directories
    with AtlasFold's native ``structure.lmdb`` and msgpack manifests.  No
    archive download is performed here, so Modal orchestration can resume
    between download, extraction, and normalization stages.
    """

    root, destination = Path(data_root), Path(output_root)
    sources = (
        ("rcsb", root / "rcsb"),
        ("rcsb_multimer", root / "rcsb_multimer"),
        ("cameo_val", root / "cameo_val"),
        ("rcsb_multimer_val", root / "rcsb_multimer_val"),
    )
    if candidate_multiplier < 1:
        raise ValueError("candidate_multiplier must be positive")
    source_metadata: dict[str, list[dict[str, object]]] = {}
    source_roots = {name: root / name for name, _ in sources}
    for source, source_root in sources:
        if not source_root.exists():
            continue
        manifest = read_atlasfold_manifest(source_root)
        source_metadata[source] = manifest

    def metadata_candidates(source: str, spec: SelectionSpec) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        for metadata in source_metadata.get(source, []):
            experiment = metadata.get("exp") or {}
            resolution = experiment.get("resolution")
            if not isinstance(resolution, int | float) or not 0.1 <= resolution <= 3.0:
                continue
            chains = metadata.get("chains")
            if isinstance(chains, list):
                chain_count = len(chains)
                total_residues = sum(int(chain.get("num_residues", 0)) for chain in chains)
            else:
                chain_count = 1
                total_residues = int(metadata.get("num_residues", 0))
            if chain_count not in (1, 2) or not MIN_RESIDUES <= total_residues <= MAX_RESIDUES:
                continue
            if chain_count == 1 and spec.monomers == 0:
                continue
            if chain_count == 2 and spec.dimers == 0:
                continue
            candidates.append(metadata)
        return candidates

    specs = [spec for spec in (train, validation) if spec is not None]
    if not specs:
        raise ValueError("at least one of train or validation must be specified")
    bounded_metadata: dict[str, list[dict[str, object]]] = {}
    for spec in specs:
        sources_for_spec = (
            ("rcsb", "rcsb_multimer") if spec is train else ("cameo_val", "rcsb_multimer_val")
        )
        for source in sources_for_spec:
            pool = metadata_candidates(source, spec)
            Random(spec.seed).shuffle(pool)
            limit = max(spec.count * candidate_multiplier, spec.count)
            bounded_metadata[source] = pool[:limit]

    all_records: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    for source, entries in bounded_metadata.items():
        if not entries:
            continue
        lmdb_path = source_roots[source] / "structure.lmdb"
        for metadata, value in _read_lmdb_values(lmdb_path, entries):
            key = str(metadata["id"])
            if "\x00" in key:
                raise ValueError("AtlasFold record ID contains a NUL byte")
            safe_key = key.replace("/", "_").replace("\\", "_")
            filename = f"{safe_key}.npz"
            output_path = _safe_member_path(destination / source, filename)
            try:
                all_records.append(
                    _normalize_record(metadata, value, source=source, output_path=output_path)
                )
            except UnsupportedStructureError as error:
                rejections.append({"source": source, "id": key, "reason": str(error)})
                continue
    if rejections:
        destination.mkdir(parents=True, exist_ok=True)
        rejection_path = destination / "rejections.json"
        rejection_path.write_text(
            json.dumps(rejections, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if return_candidates:
        candidates: dict[str, list[dict[str, object]]] = {}
        if train is not None:
            candidates[train.split] = [
                record
                for record in all_records
                if record["source"] in {"rcsb", "rcsb_multimer"} and _biologically_eligible(record)
            ]
        if validation is not None:
            candidates[validation.split] = [
                record
                for record in all_records
                if record["source"] in {"cameo_val", "rcsb_multimer_val"}
                and _biologically_eligible(record)
            ]
        return candidates
    selected: dict[str, list[dict[str, object]]] = {}
    if train is not None:
        train_records = [
            record for record in all_records if record["source"] in {"rcsb", "rcsb_multimer"}
        ]
        selected[train.split] = select_balanced_records(train_records, train)
    if validation is not None:
        validation_records = [
            record
            for record in all_records
            if record["source"] in {"cameo_val", "rcsb_multimer_val"}
        ]
        selected[validation.split] = select_balanced_records(validation_records, validation)
    return selected


def _safe_member_path(root: Path, member_name: str) -> Path:
    root = root.resolve()
    destination = (root / member_name).resolve()
    if destination != root and root not in destination.parents:
        raise ValueError(f"archive member escapes extraction root: {member_name!r}")
    return destination


def safe_extract_archive(
    archive: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    excluded_directories: frozenset[str] = frozenset(),
) -> list[Path]:
    """Extract a zip or tar archive while rejecting traversal and symlinks."""

    archive_path = Path(archive)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    tar_source = None
    if archive_path.name.endswith(".zst"):
        try:
            import zstandard
        except ImportError as error:
            raise RuntimeError(
                "zstandard is required to extract AtlasFold .tar.zst archives"
            ) from error
        tar_source = zstandard.ZstdDecompressor().stream_reader(archive_path.open("rb"))
        source = tarfile.open(fileobj=tar_source, mode="r|")  # noqa: SIM115
    else:
        source = None
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as source:
            for member in source.infolist():
                target = _safe_member_path(root, member.filename)
                if excluded_directories.intersection(target.relative_to(root).parts):
                    continue
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(member))
                extracted.append(target)
        return extracted
    if source is not None or tarfile.is_tarfile(archive_path):
        with source or tarfile.open(archive_path) as source:
            for member in source:
                if member.issym() or member.islnk():
                    raise ValueError(f"links are not allowed in archive: {member.name!r}")
                target = _safe_member_path(root, member.name)
                if excluded_directories.intersection(target.relative_to(root).parts):
                    continue
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"unsupported archive member: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                handle = source.extractfile(member)
                if handle is None:
                    raise ValueError(f"could not read archive member: {member.name!r}")
                with target.open("wb") as output:
                    shutil.copyfileobj(handle, output, length=1024 * 1024)
                extracted.append(target)
        return extracted
    raise ValueError(f"unsupported archive format: {archive_path}")


def validate_structure_npz(
    path: str | os.PathLike[str], *, sequences: Sequence[str] | None = None
) -> dict[str, tuple[int, ...]]:
    """Validate an atom14 structure cache and return its array shapes."""

    with np.load(path, allow_pickle=False) as arrays:
        required = ("coordinates", "atom_names", "chain_index", "residue_index")
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError(f"structure cache is missing arrays: {', '.join(missing)}")
        # (residues, 14, 3), (residues, 14), validated below
        coordinates, atom_names = arrays["coordinates"], arrays["atom_names"]
        # each (residues,), validated below
        chain_index, residue_index = arrays["chain_index"], arrays["residue_index"]
        if coordinates.ndim != 3 or coordinates.shape[1:] != (14, 3):
            raise ValueError("coordinates must have shape (residue, 14, 3)")
        if coordinates.dtype != np.float32:
            raise ValueError("coordinates must be float32")
        if atom_names.shape != coordinates.shape[:2]:
            raise ValueError("atom_names must have shape (residue, 14)")
        if chain_index.shape != residue_index.shape or chain_index.shape != coordinates.shape[:1]:
            raise ValueError("chain_index and residue_index must have shape (residue,)")
        if not np.issubdtype(chain_index.dtype, np.integer) or not np.issubdtype(
            residue_index.dtype, np.integer
        ):
            raise ValueError("chain_index and residue_index must be integer arrays")
        if atom_names.dtype.kind not in {"U", "S"}:
            raise ValueError("atom_names must be a string array")
        if np.any(chain_index < 0) or np.any(residue_index < 1):
            raise ValueError("chain_index and residue_index must be non-negative/one-based")
        observed_chains = sorted(set(int(value) for value in chain_index.tolist()))
        if observed_chains != list(range(len(observed_chains))):
            raise ValueError("chain_index values must be contiguous from zero")
        for chain in observed_chains:
            rows = np.flatnonzero(chain_index == chain)  # (chain residues,)
            expected_residues = np.arange(1, len(rows) + 1, dtype=residue_index.dtype)  # (chain residues,)
            if not np.array_equal(residue_index[rows], expected_residues):
                raise ValueError("residue_index must be contiguous and one-based per chain")
        if sequences is not None:
            if len(sequences) != len(observed_chains):
                raise ValueError("structure chain count does not match record metadata")
            for chain, sequence in enumerate(sequences):
                rows = np.flatnonzero(chain_index == chain)  # (chain residues,)
                expected = _atom14_names(sequence)  # (chain residues, 14)
                observed = atom_names[rows]  # (chain residues, 14)
                if observed.shape != expected.shape:
                    raise ValueError("structure residue count does not match chain sequence")
                for row in range(len(rows)):
                    names = {
                        item.decode("utf-8").strip()
                        if isinstance(item, bytes)
                        else str(item).strip()
                        for item in observed[row]
                        if (
                            item.decode("utf-8").strip()
                            if isinstance(item, bytes)
                            else str(item).strip()
                        )
                    }
                    allowed = set(expected[row][expected[row] != ""].tolist())
                    if not names.issubset(allowed):
                        raise ValueError(
                            f"atom_names contain atoms invalid for chain {chain} residue {row + 1}"
                        )
        return {name: tuple(arrays[name].shape) for name in required}


def _eligible(record: Mapping[str, object]) -> bool:
    sequence = record.get("sequence")
    chains = record.get("chains")
    if (
        not isinstance(sequence, str)
        or not sequence
        or any(residue not in STANDARD_AMINO_ACIDS for residue in sequence.replace(":", ""))
    ):
        return False
    if not isinstance(chains, Sequence) or len(chains) not in (1, 2):
        return False
    length = sum(len(chain.get("sequence", "")) for chain in chains if isinstance(chain, Mapping))
    resolution = record.get("resolution")
    if length < MIN_RESIDUES or length > MAX_RESIDUES or not isinstance(resolution, int | float):
        return False
    return 0.1 <= resolution <= 3.0 and not any(
        len(str(chain.get("sequence", ""))) == 0 for chain in chains if isinstance(chain, Mapping)
    )


def select_balanced_records(
    records: Iterable[Mapping[str, object]], spec: SelectionSpec
) -> list[dict[str, object]]:
    """Select a reproducible monomer/dimer sample with cluster exclusions.

    Records must carry ``cluster_id`` and ``pdb_id`` from a shared clustering
    run. Independent source inventories do not have comparable cluster IDs.
    Missing IDs fail closed because sequence leakage cannot be inferred here.
    """

    candidates: list[Mapping[str, object]] = []
    for record in records:
        if not _eligible(record):
            continue
        for key in ("cluster_id", "pdb_id", "structure_path"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"eligible record is missing {key}")
        if record.get("split") not in (None, spec.split):
            continue
        candidates.append(record)

    selected: list[dict[str, object]] = []
    used_clusters: set[str] = set()
    used_pdbs: set[str] = set()
    random = Random(spec.seed)
    for chain_count, target in ((1, spec.monomers), (2, spec.dimers)):
        pool = [record for record in candidates if len(record["chains"]) == chain_count]
        random.shuffle(pool)
        for record in pool:
            cluster_id, pdb_id = str(record["cluster_id"]), str(record["pdb_id"])
            if cluster_id in used_clusters or pdb_id in used_pdbs:
                continue
            selected.append(dict(record))
            used_clusters.add(cluster_id)
            used_pdbs.add(pdb_id)
            if sum(len(item["chains"]) == chain_count for item in selected) == target:
                break
        if sum(len(item["chains"]) == chain_count for item in selected) != target:
            raise ValueError(f"insufficient eligible {chain_count}-chain records for {spec.split}")
    random.shuffle(selected)
    return selected


def write_records_json(
    records: Sequence[Mapping[str, object]], path: str | os.PathLike[str]
) -> None:
    """Write records with stable formatting for hashing and W&B artifacts."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(list(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")
