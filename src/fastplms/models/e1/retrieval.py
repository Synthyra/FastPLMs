"""FASTA, MSA, context sampling, and homologue-search utilities for E1."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict, namedtuple
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel
from transformers.utils import logging

from fastplms.embeddings import Pooler

from .cache import KVCache
from .preparation import DataPrepConfig, E1BatchPreparer, get_context

if TYPE_CHECKING:
    from .modeling_e1 import E1MaskedLMOutputWithPast


def _get_logger():
    """Resolve the Transformers logger only when a retrieval path emits a message."""

    return logging.get_logger(__name__)


DOCKER_IMAGE = "ghcr.io/soedinglab/mmseqs2"
COLABFOLD_HOST = "https://api.colabfold.com"
LOWERCASE_CHARS = b"abcdefghijklmnopqrstuvwxyz"
DEFAULT_MAX_CONTEXT_TOKENS = [6144, 12288, 24576]
DEFAULT_SIMILARITY_THRESHOLDS = [1.0, 0.95, 0.9, 0.7, 0.5]
DEFAULT_EMBED_MAX_TOKENS = 8192
DEFAULT_EMBED_SIMILARITY = 0.95
E1_MSA_SAMPLING_SOURCE_REVISION = "bfd2620a602248499f3d2583d85a7ecddf0b6e02"

IdSequence = namedtuple("IdSequence", ["id", "sequence"])
IndexedSequence = tuple[int, str]


def _sequence_output_dir(output_dir: str, seq_id: str) -> str:
    """Return the per-sequence directory after enforcing path containment."""

    if not isinstance(seq_id, str) or not seq_id or "\x00" in seq_id:
        raise ValueError("seq_id must be a non-empty string without null bytes")
    if (
        seq_id in {".", ".."}
        or PurePosixPath(seq_id).name != seq_id
        or PureWindowsPath(seq_id).name != seq_id
        or PureWindowsPath(seq_id).is_absolute()
    ):
        raise ValueError(f"seq_id must be a single relative filename component: {seq_id!r}")

    output_root = Path(output_dir).resolve()
    sequence_dir = Path(output_dir) / seq_id
    resolved_sequence_dir = sequence_dir.resolve()
    if resolved_sequence_dir.parent != output_root:
        raise ValueError(f"seq_id resolves outside output_dir: {seq_id!r}")
    return os.fspath(sequence_dir)


@dataclass
class ContextSpecification:
    max_num_samples: int = 511
    max_token_length: int = 32768
    max_query_similarity: float = 1.0
    min_query_similarity: float = 0.0
    neighbor_similarity_lower_bound: float = 0.8


class E1Prediction(TypedDict, total=False):
    id: str | int
    context_id: str | int | None
    logits: torch.Tensor
    token_embeddings: torch.Tensor
    mean_token_embeddings: torch.Tensor


def read_fasta_sequences(path: str) -> dict[str, str]:
    sequences: dict[str, str] = {}
    header: str | None = None
    parts: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequences[header] = "".join(parts)
                header = line[1:].strip()
                parts = []
            else:
                assert header is not None, f"FASTA sequence found before header in {path}"
                parts.append(line)
    if header is not None:
        sequences[header] = "".join(parts)
    return sequences


def write_fasta_sequences(path: str, sequences: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for header, sequence in sequences.items():
            handle.write(f">{header}\n{sequence}\n")


def parse_msa(path: str) -> list[IdSequence]:
    records = read_fasta_sequences(path)
    sequences = []
    for record_id, record_seq in records.items():
        sequence = str(record_seq).replace("\x00", "").replace(".", "-")
        sequences.append(IdSequence(record_id, sequence))
    assert len(sequences) > 0, f"No sequences found in MSA file: {path}"
    return sequences


def convert_to_tensor(
    sequences: list[IdSequence], device: torch.device | None = None
) -> torch.ByteTensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    byte_sequences = [
        sequence.sequence.encode("ascii").translate(None, LOWERCASE_CHARS) for sequence in sequences
    ]
    lengths = {len(byte_sequence) for byte_sequence in byte_sequences}
    assert len(lengths) == 1, (
        f"MSA rows must have equal aligned lengths after removing insertions: {sorted(lengths)}"
    )
    array = np.vstack(
        [np.frombuffer(byte_sequence, dtype=np.uint8) for byte_sequence in byte_sequences]
    )
    return torch.from_numpy(array).to(device)


def get_num_neighbors(byte_seqs: torch.ByteTensor, sim_threshold: float = 0.8) -> list[int]:
    gap_token_id = np.frombuffer(b"-", np.uint8)[0].item()
    seq_lens = (byte_seqs != gap_token_id).sum(dim=1)
    num_neighbors: list[int] = []
    for i in range(byte_seqs.shape[0]):
        query_non_gaps = byte_seqs[i] != gap_token_id
        seqs_sim = (byte_seqs[:, query_non_gaps] == byte_seqs[i, query_non_gaps]).sum(
            dim=1
        ) / seq_lens
        num_neighbors.append(int((seqs_sim >= sim_threshold).sum().item()))
    return num_neighbors


def get_similarity_to_query(byte_seqs: torch.ByteTensor) -> torch.FloatTensor:
    return (byte_seqs == byte_seqs[0, :]).sum(dim=1) / byte_seqs.shape[1]


def sample_context(
    msa_path: str,
    max_num_samples: int,
    max_token_length: int,
    max_query_similarity: float = 1.0,
    min_query_similarity: float = 0.0,
    neighbor_similarity_lower_bound: float = 0.8,
    use_full_sequences_in_context: bool = False,
    full_sequences_path: str | None = None,
    seed: int = 0,
    device: torch.device | None = None,
    cache_num_neighbors_path: str | None = None,
) -> tuple[str, list[str]]:
    msa_sequences = parse_msa(msa_path)
    msa_as_byte_tensor = convert_to_tensor(msa_sequences, device)
    if cache_num_neighbors_path is not None and os.path.exists(cache_num_neighbors_path):
        num_neighbors = np.load(cache_num_neighbors_path)
    else:
        num_neighbors = np.array(
            get_num_neighbors(msa_as_byte_tensor, neighbor_similarity_lower_bound)
        )
        if cache_num_neighbors_path is not None:
            np.save(cache_num_neighbors_path, num_neighbors)

    sampling_weights = 1.0 / num_neighbors
    query_similarity = get_similarity_to_query(msa_as_byte_tensor)
    filtered_mask = (query_similarity <= max_query_similarity) & (
        query_similarity >= min_query_similarity
    )
    assert filtered_mask.sum() >= 1, (
        f"No sequences found with similarity to query within range "
        f"{min_query_similarity} <= query_similarity <= {max_query_similarity}."
    )

    filtered_weights = np.where(filtered_mask.cpu().numpy(), sampling_weights, 0.0)
    sampled_indices = np.random.default_rng(seed).choice(
        len(filtered_weights),
        size=min(max_num_samples, int(filtered_mask.sum())),
        p=filtered_weights / filtered_weights.sum(),
        replace=False,
        shuffle=True,
    )

    if use_full_sequences_in_context:
        assert full_sequences_path is not None, (
            "full_sequences_path is required when use_full_sequences_in_context=True"
        )
        full_sequences = parse_msa(full_sequences_path)
        assert len(full_sequences) == len(msa_sequences), (
            "Number of full sequences must match number of MSA sequences"
        )
        for i, (full_seq, msa_seq) in enumerate(zip(full_sequences, msa_sequences, strict=True)):
            assert full_seq.id == msa_seq.id, (
                "Full sequences and MSA sequences must be in the same order and have the same ids. "
                f"Found differing id for sample {i}: {full_seq.id} != {msa_seq.id}"
            )
        sampled_sequences = [full_sequences[int(i)] for i in sampled_indices]
    else:
        sampled_sequences = [msa_sequences[int(i)] for i in sampled_indices]

    context_sequences: list[str] = []
    context_ids: list[str] = []
    context_length = 0
    for seq in sampled_sequences:
        seq_str = seq.sequence.upper().encode("ascii").translate(None, b"-").decode("ascii")
        if context_length + len(seq_str) > max_token_length:
            break
        context_sequences.append(seq_str)
        context_ids.append(seq.id)
        context_length += len(seq_str)
    return ",".join(context_sequences), context_ids


def sample_multiple_contexts(
    msa_path: str,
    context_specifications: list[ContextSpecification],
    use_full_sequences_in_context: bool = False,
    full_sequences_path: str | None = None,
    seed: int = 0,
    device: torch.device | None = None,
    cache_num_neighbors_path: str | None = None,
) -> tuple[list[str], list[list[str]]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        if cache_num_neighbors_path is None:
            cache_num_neighbors_path = os.path.join(temp_dir, "num_neighbors.npy")

        contexts: list[str] = []
        context_ids: list[list[str]] = []
        for i, context_specification in enumerate(context_specifications):
            context, ids = sample_context(
                msa_path=msa_path,
                max_num_samples=context_specification.max_num_samples,
                max_token_length=context_specification.max_token_length,
                max_query_similarity=context_specification.max_query_similarity,
                min_query_similarity=context_specification.min_query_similarity,
                neighbor_similarity_lower_bound=context_specification.neighbor_similarity_lower_bound,
                use_full_sequences_in_context=use_full_sequences_in_context,
                full_sequences_path=full_sequences_path,
                seed=seed + i,
                device=device,
                cache_num_neighbors_path=cache_num_neighbors_path,
            )
            contexts.append(context)
            context_ids.append(ids)
    return contexts, context_ids


def get_context_id(max_tokens: int, sim_threshold: float) -> str:
    return f"identity_{sim_threshold}_tokens_{max_tokens}"


def build_context_specifications(
    max_context_tokens: list[int] | None = None,
    similarity_thresholds: list[float] | None = None,
    min_query_similarity: float = 0.3,
) -> list[tuple[ContextSpecification, str]]:
    if max_context_tokens is None:
        max_context_tokens = DEFAULT_MAX_CONTEXT_TOKENS
    if similarity_thresholds is None:
        similarity_thresholds = DEFAULT_SIMILARITY_THRESHOLDS

    specs = []
    for max_tokens in max_context_tokens:
        for sim_threshold in similarity_thresholds:
            spec = ContextSpecification(
                max_num_samples=511,
                max_token_length=max_tokens,
                max_query_similarity=sim_threshold,
                min_query_similarity=min_query_similarity,
                neighbor_similarity_lower_bound=0.8,
            )
            specs.append((spec, get_context_id(max_tokens, sim_threshold)))
    return specs


def sample_contexts_for_msa(
    a3m_path: str,
    context_specs: list[tuple[ContextSpecification, str]],
    seed: int = 42,
) -> dict[str, str]:
    specs_only = [spec for spec, _ in context_specs]
    context_ids = [context_id for _, context_id in context_specs]
    contexts, _ = sample_multiple_contexts(
        msa_path=a3m_path,
        context_specifications=specs_only,
        seed=seed,
    )
    return dict(zip(context_ids, contexts, strict=True))


def _strip_a3m_insertions(sequence: str) -> str:
    uppercase_or_gap = [char for char in sequence if char.isupper() or char in "-."]
    return "".join(uppercase_or_gap).replace("-", "").replace(".", "")


def get_query_from_a3m(path: str) -> str:
    header_found = False
    seq_parts: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header_found:
                    break
                header_found = True
                continue
            if header_found:
                seq_parts.append(line)
    assert header_found, f"No FASTA header found in A3M file: {path}"
    return _strip_a3m_insertions("".join(seq_parts))


def load_msa_dir(msa_dir: str) -> dict[str, str]:
    msa_lookup: dict[str, str] = {}
    a3m_files = list(Path(msa_dir).rglob("*.a3m"))
    if not a3m_files:
        raise FileNotFoundError(f"No .a3m files found in {msa_dir}")
    for a3m_path in tqdm(a3m_files, desc="Loading MSAs"):
        query_seq = get_query_from_a3m(str(a3m_path))
        msa_lookup[query_seq] = str(a3m_path)
    _get_logger().info("Loaded %d MSAs from %s", len(msa_lookup), msa_dir)
    return msa_lookup


def _safe_extract_tar(tar: tarfile.TarFile, output_dir: str) -> None:
    output_root = Path(output_dir).resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise ValueError(f"Tar links are not allowed: {member.name}")
        if member.isdev():
            raise ValueError(f"Tar device entries are not allowed: {member.name}")
        target = (output_root / member.name).resolve()
        if output_root != target and output_root not in target.parents:
            raise ValueError(f"Unsafe tar member path: {member.name}")
    tar.extractall(output_root, filter="data")


def load_msa_from_hf(
    hf_path: str,
    cache_dir: str | None = None,
    token: str | None = None,
) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fastplms_msa")
    os.makedirs(cache_dir, exist_ok=True)
    local_dir = os.path.join(cache_dir, hf_path.replace("/", "_"))
    if not os.path.exists(local_dir) or not any(Path(local_dir).rglob("*.a3m")):
        local_dir = snapshot_download(
            repo_id=hf_path,
            repo_type="dataset",
            local_dir=local_dir,
            token=token,
        )
        for tar_path in Path(local_dir).rglob("*.tar.gz"):
            with tarfile.open(tar_path) as tar:
                _safe_extract_tar(tar, str(tar_path.parent))
    return load_msa_dir(local_dir)


def get_msa_for_sequence(
    sequence: str, msa_lookup: dict[str, str], min_identity: float = 0.95
) -> str | None:
    if sequence in msa_lookup:
        return msa_lookup[sequence]

    best_match_path: str | None = None
    best_identity = 0.0
    for query_seq, a3m_path in msa_lookup.items():
        if abs(len(query_seq) - len(sequence)) > 10:
            continue
        min_len = min(len(query_seq), len(sequence))
        if min_len == 0:
            continue
        matches = sum(a == b for a, b in zip(query_seq[:min_len], sequence[:min_len], strict=True))
        identity = matches / min_len
        if identity > best_identity:
            best_identity = identity
            best_match_path = a3m_path

    if best_identity >= min_identity:
        return best_match_path
    return None


class ContextCache:
    """Content-addressed JSON cache for deterministic E1 MSA contexts."""

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        cache_dir: str,
        specs_hash: str,
        seed: int,
        source_revision: str = E1_MSA_SAMPLING_SOURCE_REVISION,
    ) -> None:
        self.cache_dir = cache_dir
        self.specs_hash = specs_hash
        self.seed = seed
        self.source_revision = source_revision
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> str:
        safe_key = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{safe_key}_seed{self.seed}_{self.specs_hash}.json")

    def _input_fingerprint(self, key: str) -> str:
        descriptor: dict[str, Any] = {
            "key": os.path.abspath(key) if os.path.isfile(key) else key,
            "seed": self.seed,
            "source_revision": self.source_revision,
            "specs_hash": self.specs_hash,
        }
        if os.path.isfile(key):
            hasher = hashlib.sha256()
            with open(key, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            descriptor["content_sha256"] = hasher.hexdigest()
        else:
            descriptor["literal_key"] = key
        payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load(self, key: str) -> dict[str, str] | None:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != self._SCHEMA_VERSION:
            return None
        if payload.get("input_fingerprint") != self._input_fingerprint(key):
            return None
        if payload.get("source_revision") != self.source_revision:
            return None
        contexts = payload.get("contexts")
        if not isinstance(contexts, dict) or not all(
            isinstance(name, str) and isinstance(context, str) for name, context in contexts.items()
        ):
            return None
        return contexts

    def store(self, key: str, contexts: dict[str, str]) -> None:
        if not all(
            isinstance(name, str) and isinstance(context, str) for name, context in contexts.items()
        ):
            raise TypeError("contexts must map string identifiers to string contexts")
        path = self._cache_path(key)
        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "source_revision": self.source_revision,
            "input_fingerprint": self._input_fingerprint(key),
            "contexts": contexts,
        }
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.cache_dir,
                prefix=".context-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = handle.name
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)


def compute_ppll(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    assert token_ids.numel() > 0, "Cannot score an empty token sequence"
    if token_ids.device != logits.device:
        token_ids = token_ids.to(logits.device)
    if logits.shape[0] != token_ids.shape[0]:
        raise ValueError(
            f"Logits length {logits.shape[0]} != token_ids length {token_ids.shape[0]}"
        )
    probs = logits.softmax(dim=-1)
    token_probs = probs.gather(dim=1, index=token_ids.unsqueeze(1)).squeeze(1)
    return float(token_probs.mean().item())


class _E1ContextPredictor:
    def __init__(
        self,
        model: PreTrainedModel,
        data_prep_config: DataPrepConfig | None = None,
        max_batch_tokens: int = 65536,
        use_cache: bool = True,
        cache_size: int = 4,
        save_masked_positions_only: bool = False,
        fields_to_save: list[str] | None = None,
        keep_predictions_in_gpu: bool = False,
        progress: bool = True,
    ) -> None:
        self.model = model
        self.max_batch_tokens = max_batch_tokens
        self.batch_preparer = E1BatchPreparer(data_prep_config=data_prep_config)
        self.model.eval()
        self.kv_cache = KVCache(cache_size=cache_size) if use_cache else None
        self.fields_to_save = fields_to_save or [
            "logits",
            "token_embeddings",
            "mean_token_embeddings",
        ]
        self.save_masked_positions_only = save_masked_positions_only
        self.keep_predictions_in_gpu = keep_predictions_in_gpu
        self.progress = progress

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def group_by_length(
        self, indexed_sequences: list[IndexedSequence]
    ) -> list[list[IndexedSequence]]:
        batches: list[list[IndexedSequence]] = [[]]
        for idx, seq in sorted(
            indexed_sequences, key=lambda idx_seq: (len(idx_seq[1]), idx_seq[0])
        ):
            if len(batches[-1]) > 0 and len(seq) * (len(batches[-1]) + 1) > self.max_batch_tokens:
                batches.append([])
            batches[-1].append((idx, seq))
        return batches

    def group_by_context(
        self, indexed_sequences: list[IndexedSequence]
    ) -> list[list[IndexedSequence]]:
        batches: dict[str | None, list[IndexedSequence]] = defaultdict(list)
        for idx, seq in indexed_sequences:
            batches[get_context(seq)].append((idx, seq))
        return list(batches.values())

    def batch_sequences(self, sequences: list[str]) -> list[list[int]]:
        indexed_sequences: list[IndexedSequence] = list(enumerate(sequences))
        indexed_batches = self.group_by_context(indexed_sequences)
        indexed_batches = list(
            itertools.chain.from_iterable(
                [self.group_by_length(batch) for batch in indexed_batches]
            )
        )
        batches = [[item[0] for item in batch] for batch in indexed_batches]
        flattened_indices = list(itertools.chain.from_iterable(batches))
        assert sorted(flattened_indices) == list(range(len(sequences))), (
            "Batches must contain all indices with no repetition"
        )
        return batches

    @torch.no_grad()
    def predict_batch(
        self, sequences: list[str], sequence_metadata: list[dict[str, str | int]]
    ) -> list[E1Prediction]:
        outputs = self.predict_batch_padded(sequences)
        outputs["logits"] = outputs["logits"].float()
        outputs["embeddings"] = outputs["embeddings"].float()

        token_mask = outputs["non_boundary_token_mask"] & outputs["last_sequence_mask"]
        if self.save_masked_positions_only:
            token_mask = token_mask & outputs["mask_positions_mask"]

        predictions: list[E1Prediction] = []
        for i in range(len(sequences)):
            pred: E1Prediction = {"id": sequence_metadata[i]["id"]}
            if "context_id" in sequence_metadata[i]:
                pred["context_id"] = sequence_metadata[i]["context_id"]
            if "logits" in self.fields_to_save:
                pred["logits"] = outputs["logits"][i, token_mask[i]]
                if not self.keep_predictions_in_gpu:
                    pred["logits"] = pred["logits"].to("cpu")
            if "token_embeddings" in self.fields_to_save:
                pred["token_embeddings"] = outputs["embeddings"][i, token_mask[i]]
                if not self.keep_predictions_in_gpu:
                    pred["token_embeddings"] = pred["token_embeddings"].to("cpu")
            if "mean_token_embeddings" in self.fields_to_save:
                pred["mean_token_embeddings"] = outputs["embeddings"][i, token_mask[i]].mean(dim=0)
                if not self.keep_predictions_in_gpu:
                    pred["mean_token_embeddings"] = pred["mean_token_embeddings"].to("cpu")
            predictions.append(pred)
        return predictions

    @torch.no_grad()
    def predict_batch_padded(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        device = self.device
        autocast_enabled = device.type == "cuda"
        with torch.autocast(device.type, torch.bfloat16, enabled=autocast_enabled):
            batch = self.batch_preparer.get_batch_kwargs(sequences, device=device)
            if self.kv_cache is not None:
                self.kv_cache.before_forward(batch)

            past_key_values = batch.get("past_key_values")
            use_cache = bool(batch["use_cache"]) if "use_cache" in batch else False
            output: E1MaskedLMOutputWithPast = self.model(
                input_ids=batch["input_ids"],
                within_seq_position_ids=batch["within_seq_position_ids"],
                global_position_ids=batch["global_position_ids"],
                sequence_ids=batch["sequence_ids"],
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=False,
                output_hidden_states=False,
            )
            if self.kv_cache is not None:
                self.kv_cache.after_forward(batch, output)

            padding_mask = batch["input_ids"] == self.batch_preparer.pad_token_id
            last_sequence_mask = (
                batch["sequence_ids"] == batch["sequence_ids"].max(dim=1).values[:, None]
            )
            boundary_token_mask = self.batch_preparer.get_boundary_token_mask(batch["input_ids"])
            mask_positions_mask = self.batch_preparer.get_mask_positions_mask(batch["input_ids"])
            return {
                "logits": output.logits,
                "embeddings": output.last_hidden_state,
                "last_sequence_mask": last_sequence_mask,
                "non_boundary_token_mask": ~boundary_token_mask,
                "mask_positions_mask": mask_positions_mask,
                "valid_token_mask": ~padding_mask,
            }

    @torch.no_grad()
    def predict(
        self,
        sequences: Sequence[str],
        sequence_ids: Sequence[int | str] | None = None,
        context_seqs: dict[str, str] | None = None,
    ) -> Iterator[E1Prediction]:
        if sequence_ids is None:
            sequence_ids = list(range(len(sequences)))
        if context_seqs:
            sequences_with_context = [
                (ctx + "," + seq, {"context_id": ctx_id, "id": sequence_id})
                for ctx_id, ctx in context_seqs.items()
                for seq, sequence_id in zip(sequences, sequence_ids, strict=True)
            ]
        else:
            sequences_with_context = [
                (seq, {"id": sequence_id})
                for seq, sequence_id in zip(sequences, sequence_ids, strict=True)
            ]

        batched_sequences, sequence_metadata = tuple(zip(*sequences_with_context, strict=True))
        batches = self.batch_sequences(list(batched_sequences))
        iterator = tqdm(batches, desc="Predicting batches", disable=not self.progress)
        for indices in iterator:
            sequence_batch = [batched_sequences[i] for i in indices]
            sequence_batch_metadata = [sequence_metadata[i] for i in indices]
            yield from self.predict_batch(sequence_batch, sequence_batch_metadata)


def _pool_hidden_states(
    hidden_list: list[torch.Tensor],
    pooling_types: list[str],
    device: torch.device,
) -> torch.Tensor:
    pooler = Pooler(pooling_types)
    max_len = max(hidden.shape[0] for hidden in hidden_list)
    hidden_dim = hidden_list[0].shape[1]
    batch_size = len(hidden_list)
    padded = torch.zeros(batch_size, max_len, hidden_dim, device=device)
    attention_mask = torch.zeros(batch_size, max_len, device=device)
    for i, hidden in enumerate(hidden_list):
        seq_len = hidden.shape[0]
        padded[i, :seq_len] = hidden
        attention_mask[i, :seq_len] = 1.0
    return pooler(padded, attention_mask)


def _forward_for_embedding(
    model: PreTrainedModel,
    sequences: list[str],
    context: str | None,
    max_batch_tokens: int,
    progress: bool,
) -> list[torch.Tensor]:
    predictor = _E1ContextPredictor(
        model=model,
        data_prep_config=DataPrepConfig(remove_X_tokens=True),
        max_batch_tokens=max_batch_tokens,
        fields_to_save=["token_embeddings"],
        keep_predictions_in_gpu=True,
        use_cache=False,
        cache_size=1,
        progress=progress,
    )
    context_seqs = {"embed_ctx": context} if context else None
    predictions = list(
        predictor.predict(
            sequences=sequences,
            sequence_ids=list(range(len(sequences))),
            context_seqs=context_seqs,
        )
    )
    predictions.sort(key=lambda prediction: prediction["id"])
    return [prediction["token_embeddings"] for prediction in predictions]


class HomologueSearcher:
    def __init__(
        self,
        target_db: str,
        docker_image: str = DOCKER_IMAGE,
        sensitivity: float = 7.5,
        max_seqs: int = 1000,
        min_seq_id: float = 0.0,
        coverage: float = 0.8,
        split_memory_limit: str | None = None,
        use_gpu: bool = True,
    ) -> None:
        self.target_db = target_db
        self.docker_image = docker_image
        self.sensitivity = sensitivity
        self.max_seqs = max_seqs
        self.min_seq_id = min_seq_id
        self.coverage = coverage
        self.split_memory_limit = split_memory_limit
        self.use_gpu = use_gpu

    @staticmethod
    def _seq_hash(sequence: str) -> str:
        return hashlib.md5(sequence.encode()).hexdigest()[:12]

    def _run_docker_command(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, **kwargs)

    def _validate_paths_under_cwd(self, *paths: str) -> None:
        cwd = os.path.abspath(os.getcwd())
        for path in paths:
            absolute_path = os.path.abspath(path)
            if not (absolute_path == cwd or absolute_path.startswith(cwd + os.sep)):
                raise ValueError(
                    "Path must be under the current working directory for docker volume mount. "
                    f"cwd={cwd!r}, path={absolute_path!r}"
                )

    def _path_in_container(self, local_path: str) -> str:
        self._validate_paths_under_cwd(local_path)
        rel = os.path.relpath(os.path.abspath(local_path), start=os.path.abspath(os.getcwd()))
        return rel.replace(os.sep, "/")

    def _docker_base_cmd(self) -> list[str]:
        cmd = ["docker", "run", "--rm", "-v", f"{os.getcwd()}:/app", "-w", "/app"]
        if self.use_gpu and torch.cuda.is_available():
            cmd.extend(["--gpus", "all"])
        cmd.append(self.docker_image)
        return cmd

    def _ensure_docker_image(self) -> None:
        subprocess.run(["docker", "version"], capture_output=True, text=True, check=True)
        inspect = subprocess.run(
            ["docker", "image", "inspect", self.docker_image], capture_output=True, text=True
        )
        if inspect.returncode == 0:
            return
        self._run_docker_command(["docker", "pull", self.docker_image], check=True, text=True)

    def create_db(self, fasta_path: str, db_path: str) -> str:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        if os.path.exists(f"{db_path}.dbtype"):
            return db_path
        self._ensure_docker_image()
        self._validate_paths_under_cwd(fasta_path, db_path)
        self._run_docker_command(
            [
                *self._docker_base_cmd(),
                "createdb",
                self._path_in_container(fasta_path),
                self._path_in_container(db_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return db_path

    def create_index(self, db_path: str, tmp_dir: str | None = None) -> None:
        if tmp_dir is None:
            tmp_dir = os.path.join(os.path.dirname(db_path), "tmp_index")
        os.makedirs(tmp_dir, exist_ok=True)
        self._ensure_docker_image()
        self._validate_paths_under_cwd(db_path, tmp_dir)
        self._run_docker_command(
            [
                *self._docker_base_cmd(),
                "createindex",
                self._path_in_container(db_path),
                self._path_in_container(tmp_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def search(self, sequence: str, output_dir: str, seq_id: str | None = None) -> str:
        if seq_id is None:
            seq_id = self._seq_hash(sequence)
        seq_output_dir = _sequence_output_dir(output_dir, seq_id)
        a3m_output = os.path.join(seq_output_dir, f"{seq_id}.a3m")
        if os.path.exists(a3m_output):
            return a3m_output

        self._ensure_docker_image()
        os.makedirs(seq_output_dir, exist_ok=True)
        query_fasta = os.path.join(seq_output_dir, "query.fasta")
        write_fasta_sequences(query_fasta, {seq_id: sequence})
        query_db = os.path.join(seq_output_dir, "queryDB")
        result_db = os.path.join(seq_output_dir, "resultDB")
        tmp_dir = os.path.join(seq_output_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        self._validate_paths_under_cwd(
            query_fasta, query_db, self.target_db, seq_output_dir, result_db, tmp_dir
        )

        docker_base = self._docker_base_cmd()
        self._run_docker_command(
            [
                *docker_base,
                "createdb",
                self._path_in_container(query_fasta),
                self._path_in_container(query_db),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        search_cmd = [
            *docker_base,
            "search",
            self._path_in_container(query_db),
            self._path_in_container(self.target_db),
            self._path_in_container(result_db),
            self._path_in_container(tmp_dir),
            "-s",
            str(self.sensitivity),
            "--max-seqs",
            str(self.max_seqs),
            "--min-seq-id",
            str(self.min_seq_id),
            "-c",
            str(self.coverage),
        ]
        if self.split_memory_limit is not None:
            search_cmd.extend(["--split-memory-limit", self.split_memory_limit])
        if self.use_gpu and torch.cuda.is_available():
            search_cmd.extend(["--gpu", "1"])
        self._run_docker_command(search_cmd, check=True, capture_output=True, text=True)
        self._run_docker_command(
            [
                *docker_base,
                "result2msa",
                self._path_in_container(query_db),
                self._path_in_container(self.target_db),
                self._path_in_container(result_db),
                self._path_in_container(a3m_output),
                "--msa-format-mode",
                "6",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for pattern in ["queryDB*", "resultDB*"]:
            for path in Path(seq_output_dir).glob(pattern):
                path.unlink(missing_ok=True)
        tmp_path = Path(tmp_dir)
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        return a3m_output

    def batch_search(
        self,
        sequences: list[str],
        output_dir: str,
        seq_ids: list[str] | None = None,
        continue_on_error: bool = True,
    ) -> dict[str, str]:
        if seq_ids is None:
            seq_ids = [self._seq_hash(seq) for seq in sequences]
        os.makedirs(output_dir, exist_ok=True)
        results: dict[str, str] = {}
        for seq, sid in tqdm(
            list(zip(sequences, seq_ids, strict=True)),
            desc="Searching homologues",
        ):
            try:
                results[seq] = self.search(seq, output_dir, sid)
            except Exception:
                if not continue_on_error:
                    raise
        return results


@dataclass(frozen=True)
class _ColabFoldResponse:
    """Minimal response surface required by the ColabFold API client."""

    status_code: int
    headers: dict[str, str]
    content: bytes

    def json(self) -> dict[str, Any]:
        value = json.loads(self.content.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("ColabFold returned a non-object JSON response")
        return value


class ColabFoldSearcher:
    def __init__(
        self,
        host_url: str = COLABFOLD_HOST,
        user_agent: str = "",
        mode: str = "env",
        timeout: float = 30.0,
        max_retries: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        inter_request_delay: tuple[float, float] = (1.0, 3.0),
        max_wait_time: int = 600,
    ) -> None:
        self.host_url = host_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.inter_request_delay = inter_request_delay
        self.max_wait_time = max_wait_time
        self.headers = {"User-Agent": user_agent} if user_agent else {}

    @staticmethod
    def _seq_hash(sequence: str) -> str:
        return hashlib.md5(sequence.encode()).hexdigest()[:12]

    def _backoff_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (2**attempt), self.max_delay)
        return min(delay + random.uniform(0, delay * 0.5), self.max_delay)

    def _retry_after_delay(self, headers: dict[str, str], attempt: int) -> float:
        raw_value = next(
            (value for name, value in headers.items() if name.lower() == "retry-after"),
            None,
        )
        if raw_value is None:
            return self._backoff_delay(attempt)
        try:
            delay = float(raw_value)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(raw_value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                delay = (retry_at - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return self._backoff_delay(attempt)
        if not math.isfinite(delay):
            return self._backoff_delay(attempt)
        return min(max(0.0, delay), self.max_delay)

    def _remaining_timeout(self, deadline: float | None, context: str) -> float:
        if deadline is None:
            return self.timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{context} exceeded the {self.max_wait_time}s deadline")
        return min(self.timeout, remaining)

    def _sleep_with_deadline(
        self,
        delay: float,
        deadline: float | None,
        context: str,
    ) -> None:
        delay = max(0.0, delay)
        if deadline is None:
            time.sleep(delay)
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{context} exceeded the {self.max_wait_time}s deadline")
        if delay >= remaining:
            time.sleep(remaining)
            raise TimeoutError(f"{context} exceeded the {self.max_wait_time}s deadline")
        time.sleep(delay)

    @staticmethod
    def _http_error(response: _ColabFoldResponse, url: str) -> RuntimeError:
        return RuntimeError(f"ColabFold request to {url} returned HTTP {response.status_code}")

    def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        deadline: float | None = None,
        **kwargs: Any,
    ) -> _ColabFoldResponse:
        payload = kwargs.pop("data", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported HTTP request options: {unexpected}")

        encoded_payload = None
        headers = dict(self.headers)
        if payload is not None:
            encoded_payload = urllib.parse.urlencode(payload).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        last_error: BaseException | None = None
        for attempt in range(self.max_retries):
            try:
                request = urllib.request.Request(
                    url,
                    data=encoded_payload,
                    headers=headers,
                    method=method.upper(),
                )
                try:
                    timeout = self._remaining_timeout(deadline, f"Request to {url}")
                    with urllib.request.urlopen(request, timeout=timeout) as stream:
                        response = _ColabFoldResponse(
                            status_code=int(stream.status),
                            headers={name.lower(): value for name, value in stream.headers.items()},
                            content=stream.read(),
                        )
                except urllib.error.HTTPError as error:
                    response = _ColabFoldResponse(
                        status_code=int(error.code),
                        headers={name.lower(): value for name, value in error.headers.items()},
                        content=error.read(),
                    )
                if response.status_code == 429:
                    last_error = self._http_error(response, url)
                    if attempt + 1 >= self.max_retries:
                        break
                    self._sleep_with_deadline(
                        self._retry_after_delay(response.headers, attempt),
                        deadline,
                        f"Request to {url}",
                    )
                    continue
                if response.status_code >= 500:
                    last_error = self._http_error(response, url)
                    if attempt + 1 >= self.max_retries:
                        break
                    self._sleep_with_deadline(
                        self._backoff_delay(attempt),
                        deadline,
                        f"Request to {url}",
                    )
                    continue
                if not 200 <= response.status_code < 300:
                    raise self._http_error(response, url)
                return response
            except (TimeoutError, urllib.error.URLError) as error:
                last_error = error
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Request to {url} exceeded the {self.max_wait_time}s deadline"
                    ) from error
                if attempt + 1 >= self.max_retries:
                    break
                self._sleep_with_deadline(
                    self._backoff_delay(attempt),
                    deadline,
                    f"Request to {url}",
                )
        raise RuntimeError(
            f"Request to {url} failed after {self.max_retries} attempts"
        ) from last_error

    def _submit(
        self,
        sequence: str,
        mode: str | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        mode = mode or self.mode
        query = f">101\n{sequence}\n"
        for attempt in range(self.max_retries):
            response = self._request_with_retries(
                "POST",
                f"{self.host_url}/ticket/msa",
                data={"q": query, "mode": mode},
                deadline=deadline,
            )
            data = response.json()
            status = data.get("status", "UNKNOWN")
            if status in ("RATELIMIT", "UNKNOWN"):
                if attempt + 1 >= self.max_retries:
                    break
                self._sleep_with_deadline(
                    self._backoff_delay(attempt),
                    deadline,
                    "ColabFold job submission",
                )
                continue
            return data
        raise RuntimeError(f"Failed to submit sequence after {self.max_retries} attempts")

    def _poll(self, ticket_id: str, deadline: float | None = None) -> dict[str, Any]:
        if deadline is None:
            deadline = time.monotonic() + self.max_wait_time
        poll_interval = 1.0
        while True:
            response = self._request_with_retries(
                "GET",
                f"{self.host_url}/ticket/{ticket_id}",
                deadline=deadline,
            )
            data = response.json()
            status = data.get("status", "ERROR")
            if status in ("COMPLETE", "ERROR"):
                return data
            if status not in ("RUNNING", "PENDING", "UNKNOWN"):
                return data
            wait = min(poll_interval + random.uniform(0, 0.5), 5.0)
            self._sleep_with_deadline(wait, deadline, f"Job {ticket_id}")
            poll_interval = min(poll_interval + 1.0, 5.0)

    def _download(
        self,
        ticket_id: str,
        output_path: str,
        deadline: float | None = None,
    ) -> None:
        response = self._request_with_retries(
            "GET",
            f"{self.host_url}/result/download/{ticket_id}",
            deadline=deadline,
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as handle:
            handle.write(response.content)

    def _extract_a3m(self, tar_path: str, output_dir: str, seq_id: str) -> str:
        with tarfile.open(tar_path) as tar:
            _safe_extract_tar(tar, output_dir)

        uniref_a3m = os.path.join(output_dir, "uniref.a3m")
        env_a3m = os.path.join(output_dir, "bfd.mgnify30.metaeuk30.smag30.a3m")
        a3m_files: list[str] = []
        if os.path.exists(uniref_a3m):
            a3m_files.append(uniref_a3m)
        if "env" in self.mode and os.path.exists(env_a3m):
            a3m_files.append(env_a3m)
        combined_path = os.path.join(output_dir, f"{seq_id}.a3m")
        if len(a3m_files) == 1:
            os.replace(a3m_files[0], combined_path)
        elif len(a3m_files) > 1:
            with open(combined_path, "w", encoding="utf-8") as out_handle:
                for a3m_file in a3m_files:
                    with open(a3m_file, encoding="utf-8") as in_handle:
                        out_handle.write(in_handle.read())
        else:
            raise RuntimeError("No .a3m files found in downloaded archive")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        for a3m_file in a3m_files:
            if os.path.exists(a3m_file) and a3m_file != combined_path:
                os.remove(a3m_file)
        return combined_path

    def search(self, sequence: str, output_dir: str, seq_id: str | None = None) -> str:
        if seq_id is None:
            seq_id = self._seq_hash(sequence)
        seq_output_dir = _sequence_output_dir(output_dir, seq_id)
        a3m_output = os.path.join(seq_output_dir, f"{seq_id}.a3m")
        if os.path.exists(a3m_output):
            return a3m_output
        os.makedirs(seq_output_dir, exist_ok=True)
        deadline = time.monotonic() + self.max_wait_time
        result = self._submit(sequence, deadline=deadline)
        status = result.get("status", "UNKNOWN")
        if status == "ERROR":
            raise RuntimeError(f"ColabFold API error for {seq_id}")
        if status == "MAINTENANCE":
            raise RuntimeError("ColabFold API is under maintenance")
        ticket_id = result["id"]
        result = self._poll(ticket_id, deadline=deadline)
        status = result.get("status", "UNKNOWN")
        if status != "COMPLETE":
            raise RuntimeError(f"Job failed for {seq_id}: {status}")
        tar_path = os.path.join(seq_output_dir, f"{seq_id}.tar.gz")
        self._download(ticket_id, tar_path, deadline=deadline)
        return self._extract_a3m(tar_path, seq_output_dir, seq_id)

    def batch_search(
        self,
        sequences: list[str],
        output_dir: str,
        seq_ids: list[str] | None = None,
        continue_on_error: bool = True,
    ) -> dict[str, str]:
        if seq_ids is None:
            seq_ids = [self._seq_hash(seq) for seq in sequences]
        os.makedirs(output_dir, exist_ok=True)
        results: dict[str, str] = {}
        pairs = list(zip(sequences, seq_ids, strict=True))
        for i, (seq, sid) in enumerate(tqdm(pairs, desc="ColabFold search")):
            try:
                results[seq] = self.search(seq, output_dir, sid)
            except Exception:
                if not continue_on_error:
                    raise
            if i < len(pairs) - 1:
                time.sleep(random.uniform(*self.inter_request_delay))
        return results


def _make_homologue_searcher(
    provider: str, target_db: str | None, **kwargs
) -> HomologueSearcher | ColabFoldSearcher:
    if provider == "mmseqs2":
        assert target_db is not None, "target_db is required for MMseqs2 homologue search"
        return HomologueSearcher(target_db=target_db, **kwargs)
    if provider == "colabfold":
        return ColabFoldSearcher(**kwargs)
    raise ValueError(f"Unknown homologue search provider: {provider}")
