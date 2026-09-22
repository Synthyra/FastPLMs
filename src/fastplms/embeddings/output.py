"""Resume validation and transactional publication of ordered embedding windows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .identity import _RUN_FINGERPRINT_SCHEMA_VERSION
from .pooling import Pooler
from .storage import (
    SafetensorsStreamWriter,
    append_sqlite_records,
    initialize_sqlite_run,
    load_result,
    load_sqlite_result,
    safetensors_result_exists,
    save_result,
    tensor_sha256,
    update_sqlite_run_metadata,
)
from .types import EmbeddingInput, EmbeddingRecord, EmbeddingResult, LazyTensorReference


def _output_exists(path: str | Path, format: str) -> bool:
    path = Path(path)
    if format == "sqlite":
        return path.is_file()
    return safetensors_result_exists(path)


def _output_descriptor(position: int, record: EmbeddingRecord) -> dict[str, Any]:
    tensor = record.tensor
    if isinstance(tensor, LazyTensorReference):
        dtype = tensor.dtype
        shape = tensor.shape
        digest = tensor.sha256
    else:
        dtype = str(tensor.dtype).removeprefix("torch.")
        shape = tuple(tensor.shape)
        digest = tensor_sha256(tensor)
    return {
        "position": position,
        "id": record.id,
        "dtype": dtype,
        "shape": shape,
        "sha256": digest,
    }


class EmbeddingOutput:
    """Own the resumable prefix and the commit state of one output destination."""

    def __init__(
        self,
        records: Sequence[EmbeddingInput],
        *,
        output: str | Path | None,
        format: str,
        resume: bool,
        shard_size: int,
        run_fingerprint: str,
        input_fingerprint: str,
        model_state_fingerprint: str | None,
        model_state_fingerprint_source: str,
        pooler: Pooler | None,
        pooling_names: Sequence[str],
    ) -> None:
        self.output = output
        self.format = format
        self.shard_size = shard_size
        self.completed: EmbeddingResult | None = None
        output_already_exists = output is not None and _output_exists(output, format)
        existing: EmbeddingResult | None = None
        self.start_position = 0
        if output is not None and resume and output_already_exists:
            if format == "sqlite":
                try:
                    existing = load_sqlite_result(output, run_id=run_fingerprint)
                except KeyError:
                    existing = load_result(output, format=format)
            else:
                existing = load_result(output, format=format)
            if existing.metadata.get("fingerprint_schema_version") != (
                _RUN_FINGERPRINT_SCHEMA_VERSION
            ):
                raise ValueError(
                    "Existing embeddings use an incompatible run fingerprint schema; "
                    "choose another output or set resume=False."
                )
            if existing.metadata.get("run_fingerprint") != run_fingerprint:
                raise ValueError(
                    "Existing embeddings were produced by a different run fingerprint; "
                    "choose another output or set resume=False."
                )
            if len(existing) > len(records):
                raise ValueError(
                    "Existing embeddings are not an ordered prefix of the requested inputs."
                )
            prefix_matches = all(
                (observed.id, observed.sequence) == (expected.id, expected.sequence)
                for expected, observed in zip(records, existing, strict=False)
            )
            if not prefix_matches:
                raise ValueError(
                    "Existing embeddings are not an ordered prefix of the requested inputs."
                )
            if len(existing) == len(records) and existing.metadata.get("complete", True):
                self.completed = existing
                return
            self.start_position = len(existing)

        self.sqlite_run_id: str | None = None
        self.sqlite_replace_on_first_commit = False
        self.sqlite_initial_metadata: dict[str, Any] | None = None
        if output is not None and format == "sqlite":
            self.sqlite_initial_metadata = {
                "format_version": 1,
                "fingerprint_schema_version": _RUN_FINGERPRINT_SCHEMA_VERSION,
                "run_fingerprint": run_fingerprint,
                "input_fingerprint": input_fingerprint,
                "model_state_fingerprint": model_state_fingerprint,
                "model_state_fingerprint_source": model_state_fingerprint_source,
                "complete": False,
            }
            self.sqlite_run_id = run_fingerprint
            if not resume and output_already_exists:
                try:
                    load_sqlite_result(output, run_id=run_fingerprint)
                except KeyError:
                    pass
                else:
                    # Keep an exact prior run readable until replacement inference
                    # has produced the first complete commit window.
                    self.sqlite_replace_on_first_commit = True
            if not self.sqlite_replace_on_first_commit:
                initialize_sqlite_run(
                    output,
                    self.sqlite_initial_metadata,
                    resume=resume,
                )

        stream_safetensors = output is not None and format == "safetensors"
        self.output_records: list[EmbeddingRecord] = (
            [] if self.sqlite_run_id is not None or stream_safetensors else list(existing or ())
        )
        self.output_descriptors: list[dict[str, Any]] | None = [] if output is None else None
        self.pool_slices: dict[str, tuple[int, int]] = {}
        if existing and pooler is not None:
            pooled_width = existing[0].load_tensor().shape[-1]
            if pooled_width % len(pooling_names) != 0:
                raise ValueError("Stored pooled width is inconsistent with pooling metadata.")
            self.pool_slices = pooler.output_slices(pooled_width // len(pooling_names))

        self.safetensors_writer: SafetensorsStreamWriter | None = None
        if stream_safetensors:
            if output is None:
                raise RuntimeError(
                    "Safetensors streaming was enabled without an output destination."
                )
            transactional_overwrite = output_already_exists and not resume
            self.safetensors_writer = SafetensorsStreamWriter(
                output,
                {
                    "format_version": 1,
                    "fingerprint_schema_version": _RUN_FINGERPRINT_SCHEMA_VERSION,
                    "run_fingerprint": run_fingerprint,
                    "input_fingerprint": input_fingerprint,
                    "model_state_fingerprint": model_state_fingerprint,
                    "model_state_fingerprint_source": model_state_fingerprint_source,
                    "complete": False,
                },
                shard_size=shard_size,
                existing=existing or (),
                reuse_existing=bool(resume and existing is not None),
                publish_initial=not transactional_overwrite,
                publish_incremental=not transactional_overwrite,
            )

    def append(self, window_start: int, new_records: list[EmbeddingRecord]) -> None:
        """Commit a complete ordered window at the storage format's granularity."""

        if self.output_descriptors is not None:
            self.output_descriptors.extend(
                _output_descriptor(window_start + offset, record)
                for offset, record in enumerate(new_records)
            )
        if self.output is not None and self.sqlite_run_id is not None:
            append_sqlite_records(
                self.output,
                self.sqlite_run_id,
                window_start,
                new_records,
                replace_metadata=(
                    self.sqlite_initial_metadata if self.sqlite_replace_on_first_commit else None
                ),
            )
            self.sqlite_replace_on_first_commit = False
        elif self.safetensors_writer is not None:
            self.safetensors_writer.append(new_records)
        else:
            self.output_records.extend(new_records)

    def finish(self, metadata: dict[str, Any]) -> EmbeddingResult:
        """Publish completion only after every window has committed."""

        if self.output is not None and self.sqlite_run_id is not None:
            update_sqlite_run_metadata(self.output, self.sqlite_run_id, metadata)
            return load_sqlite_result(self.output, run_id=self.sqlite_run_id)
        if self.safetensors_writer is not None:
            return self.safetensors_writer.publish(complete=True, metadata=metadata)
        result = EmbeddingResult(self.output_records, metadata)
        if self.output is not None:
            return save_result(result, self.output, format=self.format, shard_size=self.shard_size)
        return result
