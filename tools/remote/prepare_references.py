"""Write immutable native-reference requests from the typed model manifest."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from fastplms.registry import get_model_registry

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
