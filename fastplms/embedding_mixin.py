import io
import os
import queue
import sqlite3
import struct
import threading
import time

import networkx as nx
import numpy as np
import torch
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase


# Compact blob serialization constants
# Canonical source: core/embed/blob.py. Keep in sync with protify/utils.py.
_COMPACT_VERSION = 0x01
_DTYPE_TO_CODE = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_CODE_TO_DTYPE = {0: torch.float16, 1: torch.bfloat16, 2: torch.float32}
_CODE_TO_NP_DTYPE = {0: np.float16, 1: np.float16, 2: np.float32}


def tensor_to_embedding_blob(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to compact binary format for SQLite blob storage.

    Format: [version:1][dtype_code:1][ndim:4][shape:4*ndim][raw_bytes]
    bfloat16 tensors are stored as float16 bytes (numpy lacks bfloat16)
    but tagged with dtype_code=1 so they can be cast back on read.
    Falls back to torch.save for unsupported dtypes.
    """
    t = tensor.cpu()
    if t.dtype not in _DTYPE_TO_CODE:
        buffer = io.BytesIO()
        torch.save(t, buffer)
        return buffer.getvalue()
    dtype_code = _DTYPE_TO_CODE[t.dtype]

    if t.dtype == torch.bfloat16:
        raw = t.half().numpy().tobytes()
    else:
        raw = t.numpy().tobytes()

    shape = t.shape
    header = struct.pack(f'<BBi{len(shape)}i', _COMPACT_VERSION, dtype_code, len(shape), *shape)
    return header + raw


def _compact_header(dtype: torch.dtype, shape: tuple) -> bytes:
    """Build just the compact header for a given dtype and shape."""
    dtype_code = _DTYPE_TO_CODE[dtype]
    return struct.pack(f'<BBi{len(shape)}i', _COMPACT_VERSION, dtype_code, len(shape), *shape)


def batch_tensor_to_blobs(batch: torch.Tensor) -> List[bytes]:
    """Serialize a batch of same-shape tensors to compact blobs (fast path for vectors).

    Builds the header once and slices raw bytes per row. Much faster than
    per-row tensor_to_embedding_blob calls for uniform-shape batches.
    """
    assert batch.ndim >= 2, f"Expected batch with >= 2 dims, got {batch.ndim}"
    t = batch.cpu()
    store_dtype = t.dtype
    if t.dtype not in _DTYPE_TO_CODE:
        return [tensor_to_embedding_blob(t[i]) for i in range(t.shape[0])]

    if t.dtype == torch.bfloat16:
        arr = t.half().numpy()
        store_dtype = torch.bfloat16
    else:
        arr = t.numpy()

    row_shape = tuple(t.shape[1:])
    header = _compact_header(store_dtype, row_shape)
    raw = arr.tobytes()
    stride = len(raw) // t.shape[0]
    return [header + raw[i * stride:(i + 1) * stride] for i in range(t.shape[0])]


def embedding_blob_to_tensor(blob: bytes, fallback_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """Deserialize a blob back to a tensor. Auto-detects compact vs legacy formats."""
    if len(blob) >= 6 and blob[0] == _COMPACT_VERSION:
        dtype_code = blob[1]
        ndim = struct.unpack_from('<i', blob, 2)[0]
        shape = struct.unpack_from(f'<{ndim}i', blob, 6)
        data_offset = 6 + 4 * ndim
        np_dtype = _CODE_TO_NP_DTYPE[dtype_code]
        arr = np.frombuffer(blob, dtype=np_dtype, offset=data_offset).copy().reshape(shape)
        t = torch.from_numpy(arr)
        target_dtype = _CODE_TO_DTYPE[dtype_code]
        if target_dtype != t.dtype:
            t = t.to(target_dtype)
        return t

    # Fallback: try torch.load (pickle format)
    try:
        buffer = io.BytesIO(blob)
        return torch.load(buffer, map_location='cpu', weights_only=True)
    except Exception:
        pass

    # Legacy fallback: raw float32 bytes with caller-supplied shape
    assert fallback_shape is not None, "Cannot deserialize blob: unknown format and no fallback_shape provided."
    arr = np.frombuffer(blob, dtype=np.float32).copy().reshape(fallback_shape)
    return torch.from_numpy(arr)


def maybe_compile(model: torch.nn.Module, dynamic: bool = False) -> torch.nn.Module:
    """Compile model with torch.compile if possible.

    Skips compilation when dynamic=True (padding='longest') because
    flex attention's create_block_mask is incompatible with dynamic shapes
    under torch.compile, causing CUDA illegal memory access.
    """
    if dynamic:
        print("Skipping torch.compile (dynamic shapes + flex attention incompatible)")
        return model
    try:
        model = torch.compile(model)
        print("Model compiled")
    except Exception as e:
        print(f"Skipping torch.compile: {e}")
    return model


def build_collator(
    tokenizer: PreTrainedTokenizerBase,
    padding: str = 'max_length',
    max_length: int = 512,
) -> Callable[[List[str]], Dict[str, torch.Tensor]]:
    def _collate_fn(sequences: List[str]) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, Any] = dict(
            return_tensors="pt", padding=padding, truncation=True, max_length=max_length,
        )
        if padding != 'max_length':
            kwargs['pad_to_multiple_of'] = 8
        return tokenizer(sequences, **kwargs)
    return _collate_fn


def _make_embedding_progress(
    dataloader: DataLoader,
    padding: str,
    n_warmup: int = 3,
    n_calibration: int = 5,
) -> Iterator[Tuple[int, Any]]:
    """Progress-bar wrapper for embedding loops. Drop-in replacement for enumerate(dataloader).

    When padding='max_length', all batches have uniform cost so plain tqdm works.
    When padding='longest' (sorted longest-first), batch times vary dramatically.
    In that case: yield warmup batches first (compiler warmup + OOM check on longest
    sequences), then time mid-length calibration batches to estimate total ETA.

    Keep in sync with protify/embedder.py and core/atlas/precomputed.py.
    """
    total = len(dataloader)
    if padding == 'max_length' or total <= n_warmup + n_calibration:
        for i, batch in tqdm(enumerate(dataloader), total=total, desc='Embedding batches'):
            yield i, batch
        return

    dl_iter = iter(dataloader)

    # Phase 1: warmup on longest batches (first n_warmup, since sorted longest-first)
    warmup_bar = tqdm(range(n_warmup), desc='Warmup (longest batches)', leave=False)
    for i in warmup_bar:
        batch = next(dl_iter)
        yield i, batch
    warmup_bar.close()

    # Phase 2: skip to middle of dataset for calibration timing
    # We need to yield all intermediate batches too (they contain real data)
    mid_start = total // 2
    intermediate_bar = tqdm(
        range(n_warmup, mid_start), desc='Embedding batches', leave=False,
    )
    for i in intermediate_bar:
        batch = next(dl_iter)
        yield i, batch
    intermediate_bar.close()

    # Phase 3: time calibration batches from the middle
    calibration_times: List[float] = []
    cal_bar = tqdm(range(n_calibration), desc='Calibrating ETA', leave=False)
    for j in cal_bar:
        t0 = time.perf_counter()
        batch = next(dl_iter)
        yield mid_start + j, batch
        calibration_times.append(time.perf_counter() - t0)
    cal_bar.close()

    avg_time = sum(calibration_times) / len(calibration_times)
    remaining_start = mid_start + n_calibration
    remaining_count = total - remaining_start
    estimated_total_seconds = avg_time * remaining_count

    # Phase 4: remaining batches with calibrated ETA
    main_bar = tqdm(
        range(remaining_count),
        desc='Embedding batches',
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    )
    main_bar.set_postfix_str(f'ETA ~{estimated_total_seconds:.0f}s (calibrated)')
    for k in main_bar:
        batch = next(dl_iter)
        yield remaining_start + k, batch
    main_bar.close()


class _SQLWriter:
    """Context manager for async SQL embedding writes. Matches core/embed/storage.SQLEmbeddingWriter."""

    def __init__(self, conn: sqlite3.Connection, queue_maxsize: int = 4) -> None:
        self._conn = conn
        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_SQLWriter":
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        return self

    def write_batch(self, rows: List[Tuple[str, bytes]]) -> None:
        self._queue.put(rows)

    def _writer_loop(self) -> None:
        cursor = self._conn.cursor()
        while True:
            item = self._queue.get()
            if item is None:
                break
            cursor.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", item)
            if self._queue.qsize() == 0:
                self._conn.commit()
        self._conn.commit()

    def __exit__(self, *exc) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None


class Pooler:
    def __init__(self, pooling_types: List[str]) -> None:
        self.pooling_types = pooling_types
        self.pooling_options: Dict[str, Callable] = {
            'mean': self.mean_pooling,
            'max': self.max_pooling,
            'norm': self.norm_pooling,
            'median': self.median_pooling,
            'std': self.std_pooling,
            'var': self.var_pooling,
            'cls': self.cls_pooling,
            'parti': self._pool_parti,
        }

    def _create_pooled_matrices_across_layers(self, attentions: torch.Tensor) -> torch.Tensor:
        assert isinstance(attentions, torch.Tensor)
        maxed_attentions = torch.max(attentions, dim=1)[0]
        return maxed_attentions

    def _page_rank(self, attention_matrix: np.ndarray, personalization: Optional[dict] = None, nstart: Optional[dict] = None, prune_type: str = "top_k_outdegree") -> Dict[int, float]:
        G = self._convert_to_graph(attention_matrix)
        if G.number_of_nodes() != attention_matrix.shape[0]:
            raise Exception(
                f"The number of nodes in the graph should be equal to the number of tokens in sequence! You have {G.number_of_nodes()} nodes for {attention_matrix.shape[0]} tokens.")
        if G.number_of_edges() == 0:
            raise Exception(f"You don't seem to have any attention edges left in the graph.")

        return nx.pagerank(G, alpha=0.85, tol=1e-06, weight='weight', personalization=personalization, nstart=nstart, max_iter=100)

    def _convert_to_graph(self, matrix: np.ndarray) -> nx.DiGraph:
        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        return G

    def _calculate_importance_weights(self, dict_importance: Dict[int, float], attention_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        if attention_mask is not None:
            for k in list(dict_importance.keys()):
                if attention_mask[k] == 0:
                    del dict_importance[k]

        total = sum(dict_importance.values())
        return np.array([v / total for _, v in dict_importance.items()])

    def _pool_parti(self, emb: torch.Tensor, attentions: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        maxed_attentions = self._create_pooled_matrices_across_layers(attentions).numpy()
        emb_pooled = []
        for e, a, mask in zip(emb, maxed_attentions, attention_mask):
            dict_importance = self._page_rank(a)
            importance_weights = self._calculate_importance_weights(dict_importance, mask)
            num_tokens = int(mask.sum().item())
            emb_pooled.append(np.average(e[:num_tokens], weights=importance_weights, axis=0))
        pooled = torch.tensor(np.array(emb_pooled))
        return pooled

    def mean_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if attention_mask is None:
            return emb.mean(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    def max_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if attention_mask is None:
            return emb.max(dim=1).values
        else:
            mask = attention_mask.unsqueeze(-1).bool()
            return emb.masked_fill(~mask, float('-inf')).max(dim=1).values

    def norm_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if attention_mask is None:
            return emb.norm(dim=1, p=2)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).norm(dim=1, p=2)

    def median_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if attention_mask is None:
            return emb.median(dim=1).values
        else:
            mask = attention_mask.unsqueeze(-1).bool()
            return emb.masked_fill(~mask, float('nan')).nanmedian(dim=1).values

    def std_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if attention_mask is None:
            return emb.std(dim=1)
        else:
            var = self.var_pooling(emb, attention_mask, **kwargs)
            return torch.sqrt(var)

    def var_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if attention_mask is None:
            return emb.var(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            mean = (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            mean = mean.unsqueeze(1)
            squared_diff = (emb - mean) ** 2
            var = (squared_diff * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            return var

    def cls_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        return emb[:, 0, :]

    def __call__(
            self,
            emb: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            attentions: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        if attention_mask is not None:
            assert attention_mask.sum(dim=-1).min() > 0, (
                "Pooler received samples with all-zero attention masks. "
                "This causes NaN from division by zero. Filter empty inputs before pooling."
            )
        final_emb: List[torch.Tensor] = []
        for pooling_type in self.pooling_types:
            final_emb.append(self.pooling_options[pooling_type](emb=emb, attention_mask=attention_mask, attentions=attentions))
        return torch.cat(final_emb, dim=-1)


class ProteinDataset(TorchDataset):
    """Simple dataset for protein sequences."""
    def __init__(self, sequences: List[str]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def parse_fasta(fasta_path: str) -> List[str]:
    assert os.path.exists(fasta_path), f"FASTA file does not exist: {fasta_path}"
    sequences = []
    current_seq = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
    if current_seq:
        sequences.append(''.join(current_seq))
    return sequences


class EmbeddingMixin:
    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def _read_sequences_from_db(self, db_path: str) -> Set[str]:
        """Read sequences from SQLite database."""
        with sqlite3.connect(db_path, timeout=30) as conn:
            c = conn.cursor()
            c.execute("SELECT sequence FROM embeddings")
            return {row[0] for row in c.fetchall()}

    def _ensure_embeddings_table(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "sequence TEXT PRIMARY KEY, "
            "embedding BLOB NOT NULL"
            ")"
        )
        conn.commit()

    def load_embeddings_from_pth(self, save_path: str) -> Dict[str, torch.Tensor]:
        assert os.path.exists(save_path), f"Embedding file does not exist: {save_path}"
        payload = torch.load(save_path, map_location="cpu", weights_only=True)
        assert isinstance(payload, dict), "Expected .pth embeddings file to contain a dictionary."
        for sequence, tensor in payload.items():
            assert isinstance(sequence, str), "Expected embedding dictionary keys to be sequences (str)."
            assert isinstance(tensor, torch.Tensor), "Expected embedding dictionary values to be tensors."
        return payload

    def load_embeddings_from_db(self, db_path: str, sequences: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        assert os.path.exists(db_path), f"Embedding database does not exist: {db_path}"
        loaded: Dict[str, torch.Tensor] = {}
        with sqlite3.connect(db_path, timeout=30) as conn:
            self._ensure_embeddings_table(conn)
            cursor = conn.cursor()
            if sequences is None:
                cursor.execute("SELECT sequence, embedding FROM embeddings")
            else:
                if len(sequences) == 0:
                    return loaded
                placeholders = ",".join(["?"] * len(sequences))
                cursor.execute(
                    f"SELECT sequence, embedding FROM embeddings WHERE sequence IN ({placeholders})",
                    tuple(sequences),
                )

            rows = cursor.fetchall()
            for row in rows:
                sequence = row[0]
                embedding_bytes = row[1]
                loaded[sequence] = embedding_blob_to_tensor(embedding_bytes)
        return loaded

    def embed_dataset(
        self,
        sequences: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 2,
        max_len: int = 512,
        truncate: bool = True,
        full_embeddings: bool = False,
        embed_dtype: torch.dtype = torch.float32,
        pooling_types: List[str] = ['mean'],
        num_workers: int = 0,
        sql: bool = False,
        save: bool = True,
        sql_db_path: str = 'embeddings.db',
        save_path: str = 'embeddings.pth',
        fasta_path: Optional[str] = None,
        padding: str = 'max_length',
        **kwargs,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Embed a dataset of protein sequences.

        Supports two modes:
        - Tokenizer mode (ESM2/ESM++): provide `tokenizer`, `_embed(input_ids, attention_mask)` is used.
        - Sequence mode (E1): pass `tokenizer=None`, `_embed(sequences, return_attention_mask=True, **kwargs)` is used.

        Sequences can be supplied as a list via `sequences`, parsed from a FASTA file via
        `fasta_path`, or both (the two sources are combined). At least one must be provided.
        """
        if fasta_path is not None:
            fasta_sequences = parse_fasta(fasta_path)
            sequences = list(sequences or []) + fasta_sequences
        assert sequences is not None and len(sequences) > 0, \
            "Must provide at least one sequence via `sequences` or `fasta_path`."
        sequences = list(set([seq[:max_len] if truncate else seq for seq in sequences]))
        sequences = sorted(sequences, key=len, reverse=True)
        hidden_size = self.config.hidden_size
        pooler = Pooler(pooling_types) if not full_embeddings else None
        tokenizer_mode = tokenizer is not None

        # Resolve padding and compilation
        dynamic = padding == 'longest'
        compiled_model = maybe_compile(self, dynamic=dynamic)

        if tokenizer_mode:
            collate_fn = build_collator(tokenizer, padding=padding, max_length=max_len)
            device = self.device
        else:
            collate_fn = None
            device = None

        def get_embeddings(residue_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            assert isinstance(residue_embeddings, torch.Tensor)
            if full_embeddings or residue_embeddings.ndim == 2:
                return residue_embeddings
            return pooler(residue_embeddings, attention_mask)

        def iter_batches(to_embed: List[str]):
            if tokenizer_mode:
                assert collate_fn is not None
                assert device is not None
                dataset = ProteinDataset(to_embed)
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    prefetch_factor=2 if num_workers > 0 else None,
                    collate_fn=collate_fn,
                    shuffle=False,
                    pin_memory=True,
                )
                for i, batch in _make_embedding_progress(dataloader, padding):
                    seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    residue_embeddings = compiled_model._embed(input_ids, attention_mask)
                    yield seqs, residue_embeddings, attention_mask
            else:
                for batch_start in tqdm(range(0, len(to_embed), batch_size), desc='Embedding batches'):
                    seqs = to_embed[batch_start:batch_start + batch_size]
                    batch_output = compiled_model._embed(seqs, return_attention_mask=True, **kwargs)
                    assert isinstance(batch_output, tuple), "Sequence mode _embed must return (last_hidden_state, attention_mask)."
                    assert len(batch_output) == 2, "Sequence mode _embed must return exactly two values."
                    residue_embeddings, attention_mask = batch_output
                    assert isinstance(attention_mask, torch.Tensor), "Sequence mode _embed must return attention_mask as a torch.Tensor."
                    yield seqs, residue_embeddings, attention_mask

        if sql:
            # Step 1: DEDUPLICATE - check existing embeddings in SQL
            conn = sqlite3.connect(sql_db_path, timeout=30, check_same_thread=False)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA busy_timeout=30000')
            conn.execute('PRAGMA synchronous=OFF')
            conn.execute('PRAGMA cache_size=-64000')
            self._ensure_embeddings_table(conn)
            already_embedded = self._read_sequences_from_db(sql_db_path)
            to_embed = [seq for seq in sequences if seq not in already_embedded]
            print(f"Found {len(already_embedded)} already embedded sequences in {sql_db_path}")
            print(f"Embedding {len(to_embed)} new sequences")
            if len(to_embed) > 0:
                # Steps 4-7: BATCH+EMBED -> POOL/TRIM -> SERIALIZE -> WRITE (async)
                with _SQLWriter(conn) as writer:
                    with torch.inference_mode():
                        for seqs, residue_embeddings, attention_mask in iter_batches(to_embed):
                            embeddings = get_embeddings(residue_embeddings, attention_mask).to(embed_dtype)
                            if full_embeddings:
                                batch_rows = []
                                for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                                    batch_rows.append((seq, tensor_to_embedding_blob(emb[mask.bool()].reshape(-1, hidden_size))))
                            else:
                                blobs = batch_tensor_to_blobs(embeddings)
                                batch_rows = list(zip(seqs, blobs))
                            writer.write_batch(batch_rows)
            conn.close()
            return None

        embeddings_dict = {}
        if os.path.exists(save_path):
            embeddings_dict = self.load_embeddings_from_pth(save_path)
            to_embed = [seq for seq in sequences if seq not in embeddings_dict]
            print(f"Found {len(embeddings_dict)} already embedded sequences in {save_path}")
            print(f"Embedding {len(to_embed)} new sequences")
        else:
            to_embed = sequences
            print(f"Embedding {len(to_embed)} new sequences")

        if len(to_embed) > 0:
            with torch.inference_mode():
                for seqs, residue_embeddings, attention_mask in iter_batches(to_embed):
                    embeddings = get_embeddings(residue_embeddings, attention_mask).to(embed_dtype)
                    for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                        if full_embeddings:
                            emb = emb[mask.bool()].reshape(-1, hidden_size)
                        embeddings_dict[seq] = emb.cpu()

        if save:
            torch.save(embeddings_dict, save_path)

        return embeddings_dict


if __name__ == "__main__":
    # py -m pooler
    pooler = Pooler(pooling_types=['max', 'parti'])
    batch_size = 8
    seq_len = 64
    hidden_size = 128
    num_layers = 12
    emb = torch.randn(batch_size, seq_len, hidden_size)
    attentions = torch.randn(batch_size, num_layers, seq_len, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    y = pooler(emb=emb, attention_mask=attention_mask, attentions=attentions)
    print(y.shape)
