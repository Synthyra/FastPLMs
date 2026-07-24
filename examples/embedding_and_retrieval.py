#!/usr/bin/env python3
"""Stream ordered embeddings and reopen an optional SQLite result read-only."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


if __package__:
    from ._runtime import add_execution_arguments, resolve_execution
else:
    from _runtime import add_execution_arguments, resolve_execution


def configure_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def run_embeddings(
    model: Any,
    tokenizer: Any,
    inputs: Any,
    *,
    output: Path | None,
    output_format: str,
    max_length: int,
) -> Any:
    return model.embed_dataset(
        inputs,
        tokenizer=tokenizer,
        batch_size=8,
        batch_window_size=64,
        max_tokens_per_batch=4096,
        max_length=max_length,
        pooling=("mean", "std"),
        output=output,
        format=output_format,
        resume=output is not None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Local manifest-built artifact")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fasta", type=Path)
    source.add_argument("--sequence", action="append", dest="sequences")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--format",
        choices=("safetensors", "sqlite"),
        default="safetensors",
        dest="output_format",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--select-id", action="append", default=[])
    add_execution_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    artifact = arguments.artifact.expanduser().resolve()
    if not (artifact / "config.json").is_file():
        raise SystemExit(f"Not a local artifact: {artifact}")
    if arguments.select_id and (arguments.output is None or arguments.output_format != "sqlite"):
        raise SystemExit("--select-id requires both --output and --format sqlite")
    inputs: Any = arguments.fasta if arguments.fasta is not None else arguments.sequences
    try:
        device, dtype = resolve_execution(arguments.device, arguments.dtype)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    configure_offline()
    from transformers import AutoModel, AutoTokenizer

    model = (
        AutoModel.from_pretrained(
            artifact,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        artifact,
        trust_remote_code=True,
        local_files_only=True,
    )
    result = run_embeddings(
        model,
        tokenizer,
        inputs,
        output=arguments.output,
        output_format=arguments.output_format,
        max_length=arguments.max_length,
    )
    for record in result:
        print(record.id, record.sequence, tuple(record.load_tensor().shape))

    if arguments.output is not None and arguments.output_format == "sqlite" and arguments.select_id:
        from fastplms.embeddings import load_sqlite_result

        selected = load_sqlite_result(
            arguments.output,
            record_ids=arguments.select_id,
        )
        print("selected", [record.id for record in selected])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
