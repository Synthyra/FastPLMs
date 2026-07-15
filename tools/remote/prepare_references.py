"""Write immutable native-reference requests from the typed model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from fastplms.registry import get_model_registry
from tests.parity.support.esmc_calibration import (
    ESMC_BOUNDARY_LENGTHS,
    load_esmc_biological_holdout,
)

SCHEMA_VERSION = 1
SEED = 42
CANONICAL_AAS = "ACDEFGHIKLMNPQRSTVWY"
MIXED_LENGTHS = (61, 29, 13)
EDGE_SEQUENCES = (
    "ACDEFGHIKLMNPQRSTVWY",
    "AXBJOUZ",
    "acdefghik",
    "A C\nD\tE",
    "",
)
_TOKENIZER_FILE_NAMES = frozenset(
    {
        "added_tokens.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
)


def _sequence_batch() -> tuple[str, ...]:
    generator = random.Random(SEED)
    return tuple(
        "M" + "".join(generator.choices(CANONICAL_AAS, k=length - 1)) for length in MIXED_LENGTHS
    )


def _generated_sequences(lengths: tuple[int, ...]) -> tuple[str, ...]:
    generator = random.Random(SEED)
    return tuple(
        "M" + "".join(generator.choices(CANONICAL_AAS, k=length - 1)) for length in lengths
    )


def _calibration_batch(kind: str, cases: list[dict[str, str]]) -> dict[str, object]:
    return {
        "kind": kind,
        "seed": SEED,
        "cases": [
            {
                **case,
                "sequence_length": len(case["sequence"]),
                "sequence_sha256": hashlib.sha256(case["sequence"].encode("ascii")).hexdigest(),
            }
            for case in cases
        ],
    }


def _esmc_calibration_batches() -> list[dict[str, object]]:
    boundary = [
        {"case_id": f"generated-boundary-{length}", "sequence": sequence}
        for length, sequence in zip(
            ESMC_BOUNDARY_LENGTHS,
            _generated_sequences(ESMC_BOUNDARY_LENGTHS),
            strict=True,
        )
    ]
    biological = [
        {
            "case_id": str(case["case_id"]),
            "sequence": str(case["sequence"]),
            "source": str(case["source"]),
            "source_sha256": str(case["source_sha256"]),
        }
        for case in load_esmc_biological_holdout()
    ]
    return [
        _calibration_batch("generated_kernel_boundary", boundary),
        _calibration_batch("real_biological_holdout", biological),
    ]


def prepare_reference_requests(output_root: Path) -> tuple[Path, ...]:
    """Write one self-contained request for every sequence checkpoint."""

    registry = get_model_registry()
    paths: list[Path] = []
    for spec in registry.values():
        if spec.family.tokenizer_mode == "structure":
            continue
        request = {
            "schema_version": SCHEMA_VERSION,
            "model_id": spec.id,
            "family": spec.family.id,
            "architecture": spec.family.architecture,
            "adapter": spec.family.reference_adapter,
            "reference_container": spec.family.reference_container,
            "reference_repo_id": spec.official.repo_id,
            "reference_revision": spec.official.revision,
            "reference_files": [asdict(item) for item in spec.official.files],
            "state_transform": spec.family.state_transform,
            "tokenizer_mode": spec.family.tokenizer_mode,
            "attention_implementations": list(spec.family.attention),
            "deep_reference": spec.is_deep_reference,
            "oracle_assets": [asdict(asset) for asset in spec.oracle_assets],
            "tokenizer_files": [
                item.path
                for item in spec.official.files
                if Path(item.path).name in _TOKENIZER_FILE_NAMES
            ],
            "sequences": list(_sequence_batch()),
            "edge_sequences": list(EDGE_SEQUENCES),
            "seed": SEED,
        }
        if spec.family.id == "esm_plusplus":
            request["calibration_batches"] = _esmc_calibration_batches()
        path = output_root / "requests" / spec.family.reference_container / f"{spec.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(request, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Refusing to replace a different reference request: {path}")
        path.write_text(encoded, encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/reference"))
    arguments = parser.parse_args()
    for path in prepare_reference_requests(arguments.output_root):
        print(path)


if __name__ == "__main__":
    main()
