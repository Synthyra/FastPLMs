#!/usr/bin/env python3
"""Load one manifest-built Hugging Face artifact without network access."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


AUTO_CLASS_NAMES = (
    "AutoConfig",
    "AutoModel",
    "AutoModelForMaskedLM",
    "AutoModelForSeq2SeqLM",
    "AutoModelForSequenceClassification",
    "AutoModelForTokenClassification",
)


def require_local_artifact(value: str) -> Path:
    artifact = Path(value).expanduser().resolve()
    if not artifact.is_dir() or not (artifact / "config.json").is_file():
        raise argparse.ArgumentTypeError(
            f"Expected a local artifact directory containing config.json: {artifact}"
        )
    return artifact


def configure_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_local_artifact(artifact: Path, auto_class_name: str) -> Any:
    configure_offline()
    import transformers

    auto_class = getattr(transformers, auto_class_name)
    loaded = auto_class.from_pretrained(
        artifact,
        trust_remote_code=True,
        local_files_only=True,
    )
    if hasattr(loaded, "eval"):
        loaded = loaded.eval()
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=require_local_artifact)
    parser.add_argument("--auto-class", choices=AUTO_CLASS_NAMES, default="AutoModel")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    loaded = load_local_artifact(arguments.artifact, arguments.auto_class)
    print(type(loaded).__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
