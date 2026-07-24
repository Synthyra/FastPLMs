#!/usr/bin/env python3
"""Run seeded test-time training, persist it, reset, and reload offline."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


if __package__:
    from ._runtime import add_execution_arguments, resolve_execution
else:
    from _runtime import add_execution_arguments, resolve_execution


def configure_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def adapt_and_save(model: Any, sequence: str, output: Path, seed: int) -> Any:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing TTT artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    adaptation_started = False
    try:
        adaptation_started = True
        metrics = model.ttt(
            seq=sequence,
            ttt_config={
                "steps": 3,
                "ags": 1,
                "batch_size": 1,
                "seed": seed,
                "initial_state_reset": True,
            },
        )
        model.save_pretrained(staging, safe_serialization=True)
        if not (staging / "config.json").is_file() or not any(staging.glob("*.safetensors")):
            raise RuntimeError("TTT staging output is missing config or safetensors weights")
        if output.exists():
            raise FileExistsError(f"TTT output appeared during staging: {output}")
        staging.rename(output)
        return metrics
    finally:
        try:
            if adaptation_started:
                model.ttt_reset()
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence", default="MSTNPKPQRKTKRNT")
    parser.add_argument("--seed", type=int, default=7)
    add_execution_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    artifact = arguments.artifact.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if not (artifact / "config.json").is_file():
        raise SystemExit(f"Not a local artifact: {artifact}")
    if output == artifact or artifact in output.parents:
        raise SystemExit("TTT output must not be the source artifact or a directory inside it")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite an existing TTT artifact: {output}")
    try:
        device, dtype = resolve_execution(arguments.device, arguments.dtype)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    configure_offline()
    from transformers import AutoModelForMaskedLM

    model = (
        AutoModelForMaskedLM.from_pretrained(
            artifact,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
        )
        .to(device)
        .eval()
    )
    metrics = adapt_and_save(model, arguments.sequence, output, arguments.seed)
    reloaded = (
        AutoModelForMaskedLM.from_pretrained(
            output,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
        )
        .to(device)
        .eval()
    )
    reloaded.ttt_reset()
    print(metrics)
    print("reloaded", type(reloaded).__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
