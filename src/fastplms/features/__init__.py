"""The FastPLMs feature store: one storage format for every embedding this workspace keeps.

A feature is one value per sequence, addressed by the SHA-256 of the sequence and by a key that
names the model, its revision, the sparse autoencoder, the layer, the pooling, the dtype, and the
residue limit. `store` holds the directory format and `layouts` the row layouts: dense vectors,
compressed-sparse pooled rows, ragged hidden states, and ragged top-k residue codes. `reader`
serves rows by random access to loops that read a batch at a time, `writing` commits a window of
rows as one segment, and `conversion` moves a cache another format holds into a store and proves the
rows survived.
"""

from .async_writer import AsyncFeatureWriter, PackedBatch
from .conversion import (
    ConversionMismatch,
    ConversionReceipt,
    conversion_fingerprint,
    convert_rows,
    describe_file,
)
from .layouts import (
    CSR,
    DENSE,
    LAYOUT_NAMES,
    RAGGED,
    RAGGED_TOPK,
    SparseRow,
    TopKRow,
    dtype_name,
    value_dtype,
)
from .reader import CsrRows, FeatureReader
from .store import (
    FORMAT,
    FeatureStore,
    RowAddress,
    SegmentReceipt,
    SegmentWriter,
    StoredFeature,
    features_in,
    open_feature,
    partition_sequences,
    sequence_digest,
)
from .writing import write_rows


__all__ = [
    "CSR",
    "DENSE",
    "FORMAT",
    "LAYOUT_NAMES",
    "RAGGED",
    "RAGGED_TOPK",
    "AsyncFeatureWriter",
    "ConversionMismatch",
    "ConversionReceipt",
    "CsrRows",
    "FeatureReader",
    "FeatureStore",
    "PackedBatch",
    "RowAddress",
    "SegmentReceipt",
    "SegmentWriter",
    "SparseRow",
    "StoredFeature",
    "TopKRow",
    "conversion_fingerprint",
    "convert_rows",
    "describe_file",
    "dtype_name",
    "features_in",
    "open_feature",
    "partition_sequences",
    "sequence_digest",
    "value_dtype",
    "write_rows",
]
