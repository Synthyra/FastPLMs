#!/usr/bin/env python3
"""Embed duplicate E1 queries with a local A3M and shared persistence."""

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


def embed_local_msa(
    model: Any,
    sequence: str,
    a3m_path: Path,
    *,
    output: Path | None,
    output_format: str,
    seed: int,
) -> Any:
    return model.embed_dataset_with_msa(
        [sequence, sequence],
        msa_lookup={sequence: str(a3m_path)},
        batch_size=2,
        max_len=len(sequence),
        pooling_types=["mean"],
        seed=seed,
        progress=False,
        batch_window_size=2,
        max_tokens_per_batch=2 * len(sequence),
        output=output,
        format=output_format,
        resume=output is not None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Local E1 artifact")
    parser.add_argument("a3m", type=Path, help="Local A3M whose query matches --sequence")
    parser.add_argument("--sequence", default="MSTNPKPQRKTKRNT")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--format",
        choices=("safetensors", "sqlite"),
        default="safetensors",
        dest="output_format",
    )
    parser.add_argument("--seed", type=int, default=7)
    add_execution_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    artifact = arguments.artifact.expanduser().resolve()
    a3m_path = arguments.a3m.expanduser().resolve()
    if not (artifact / "config.json").is_file():
        raise SystemExit(f"Not a local artifact: {artifact}")
    if not a3m_path.is_file():
        raise SystemExit(f"Not a local A3M: {a3m_path}")
    try:
        device, dtype = resolve_execution(arguments.device, arguments.dtype)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    configure_offline()
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(
        artifact,
        trust_remote_code=True,
        local_files_only=True,
        dtype=dtype,
    ).to(device).eval()
    result = embed_local_msa(
        model,
        arguments.sequence,
        a3m_path,
        output=arguments.output,
        output_format=arguments.output_format,
        seed=arguments.seed,
    )
    print([(record.id, record.sequence) for record in result])

    if arguments.output is not None and arguments.output_format == "sqlite":
        from fastplms.embeddings import load_sqlite_result

        selected = load_sqlite_result(arguments.output, positions=[1, 0, 1])
        print("selected", [record.id for record in selected])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
