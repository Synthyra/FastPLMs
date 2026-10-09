"""Ordered, residue-aware protein embedding utilities."""

from .feature_runs import embed_into_features
from .pooling import (
    POOLING_NAMES, POOLING_SEMANTICS_TOKENS, TOKEN_POOLING_NAMES, Pooler, pagerank_weights, pool_token_rows,
)
from .runner import (
    EmbeddingMixin,
    embed_dataset,
    iter_fasta,
    parse_fasta,
    select_hidden_state_embeddings,
)
from .storage import (
    DEFAULT_SHARD_SIZE,
    append_sqlite_records,
    convert_legacy_sqlite,
    garbage_collect_safetensors_generations,
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
from .taps import (
    HiddenTap, LayerAccumulator, ReducedTap, RowSelection, SparseResidueTap, StreamingTap, TapBatch,
)
from .token_batches import BatchGeometry, TokenTapExecutor, plan_geometry_batches, plan_token_batches
from .token_runs import embed_token_features
from .tokens import ResidueVocabulary
from .types import (
    EmbeddingBatch,
    EmbeddingInput,
    EmbeddingRecord,
    EmbeddingResult,
    LazyTensorReference,
    TapRecord,
    TapResult,
    TapRunReceipt,
    TensorValue,
)


__all__ = [
    "DEFAULT_SHARD_SIZE",
    "POOLING_NAMES",
    "POOLING_SEMANTICS_TOKENS",
    "TOKEN_POOLING_NAMES",
    "BatchGeometry",
    "EmbeddingBatch",
    "EmbeddingInput",
    "EmbeddingMixin",
    "EmbeddingRecord",
    "EmbeddingResult",
    "HiddenTap",
    "LayerAccumulator",
    "LazyTensorReference",
    "Pooler",
    "ReducedTap",
    "ResidueVocabulary",
    "RowSelection",
    "SparseResidueTap",
    "StreamingTap",
    "TapBatch",
    "TapRecord",
    "TapResult",
    "TapRunReceipt",
    "TensorValue",
    "TokenTapExecutor",
    "append_sqlite_records",
    "convert_legacy_sqlite",
    "embed_dataset",
    "embed_into_features",
    "embed_token_features",
    "garbage_collect_safetensors_generations",
    "initialize_sqlite_run",
    "iter_fasta",
    "load_legacy_pth",
    "load_result",
    "load_safetensors_result",
    "load_sqlite_result",
    "pagerank_weights",
    "parse_fasta",
    "plan_geometry_batches",
    "plan_token_batches",
    "pool_token_rows",
    "save_result",
    "save_safetensors_result",
    "save_sqlite_result",
    "select_hidden_state_embeddings",
    "tensor_sha256",
    "update_sqlite_run_metadata",
]
