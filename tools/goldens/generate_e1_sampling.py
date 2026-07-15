"""Generate deterministic E1 MSA-sampling goldens from the pinned oracle.

This producer is intended for the native E1 reference environment. Candidate
tests consume only its JSON output and never import the upstream package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_fixture(path: Path) -> None:
    path.write_text(
        ">query\nACDEFGHI\n>near\nACDEYGH-\n>gapped\nAC-EFGHI\n>mid\nTCD-FGHI\n>far\nTTTTTTTT\n",
        encoding="utf-8",
    )


def _git_revision(upstream_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _generate(upstream_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(upstream_root / "src"))
    from E1.msa_sampling import (  # type: ignore[import-not-found]
        ContextSpecification,
        sample_context,
        sample_multiple_contexts,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        msa_path = Path(temp_dir) / "parity.a3m"
        _write_fixture(msa_path)
        common = {
            "msa_path": str(msa_path),
            "max_num_samples": 3,
            "max_token_length": 32,
            "max_query_similarity": 0.99,
            "min_query_similarity": 0.0,
            "neighbor_similarity_lower_bound": 0.8,
            "device": "cpu",
        }
        single = {str(seed): sample_context(seed=seed, **common) for seed in (0, 3, 11)}
        multiple = sample_multiple_contexts(
            msa_path=str(msa_path),
            context_specifications=[
                ContextSpecification(
                    max_num_samples=3,
                    max_token_length=16,
                    max_query_similarity=0.99,
                    min_query_similarity=0.0,
                    neighbor_similarity_lower_bound=0.8,
                ),
                ContextSpecification(
                    max_num_samples=4,
                    max_token_length=32,
                    max_query_similarity=1.0,
                    min_query_similarity=0.2,
                    neighbor_similarity_lower_bound=0.8,
                ),
            ],
            seed=7,
            device="cpu",
        )
    return {"single_context": single, "multiple_context": multiple}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    upstream_root = args.upstream_root.resolve()
    source_path = upstream_root / "src" / "E1" / "msa_sampling.py"
    payload = {
        "provenance": {
            "upstream_revision": _git_revision(upstream_root),
            "source_path": "src/E1/msa_sampling.py",
            "source_sha256": _sha256_file(source_path),
            "generation_command": [
                "python",
                "tools/goldens/generate_e1_sampling.py",
                "--upstream-root",
                str(args.upstream_root),
                "--output",
                str(args.output),
            ],
        },
        "goldens": _generate(upstream_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
