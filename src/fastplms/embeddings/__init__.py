"""Ordered, residue-aware protein embedding utilities."""

from .pooling import POOLING_NAMES, Pooler, pagerank_weights
from .runner import EmbeddingMixin, embed_dataset, parse_fasta, select_hidden_state_embeddings
from .storage import (
    DEFAULT_SHARD_SIZE,
    append_sqlite_records,
    initialize_sqlite_run,
    load_legacy_pth,
    load_result,
    load_safetensors_result,
    load_sqlite_result,
    save_result,
    save_safetensors_result,
    save_sqlite_result,
    tensor_sha256,
    update_sqlite_run_metadata,
)
from .types import (
    EmbeddingBatch,
    EmbeddingInput,
    EmbeddingRecord,
    EmbeddingResult,
    LazyTensorReference,
    TensorValue,
)

__all__ = [
    "DEFAULT_SHARD_SIZE",
    "POOLING_NAMES",
    "EmbeddingBatch",
    "EmbeddingInput",
    "EmbeddingMixin",
    "EmbeddingRecord",
    "EmbeddingResult",
    "LazyTensorReference",
    "Pooler",
    "TensorValue",
    "append_sqlite_records",
    "embed_dataset",
    "initialize_sqlite_run",
    "load_legacy_pth",
    "load_result",
    "load_safetensors_result",
    "load_sqlite_result",
    "pagerank_weights",
    "parse_fasta",
    "save_result",
    "save_safetensors_result",
    "save_sqlite_result",
    "select_hidden_state_embeddings",
    "tensor_sha256",
    "update_sqlite_run_metadata",
]
