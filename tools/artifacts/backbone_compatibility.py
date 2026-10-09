"""Compare storage values and BF16 execution weights of pinned ESMC backbones.

Downloads are manifest-scoped, content-verified, and contain no executable Hub code.
Small ESMFold2 snapshots store BF16-rounded values in FP32 containers. Raw FP32
inequality must not be mistaken for different BF16 execution weights. Live feature
parity remains separate from either content comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import torch

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from huggingface_hub import snapshot_download

from fastplms.atomic_files import write_text_atomically
from fastplms.json_files import indented_json
from fastplms.registry import CheckpointSource, get_model_registry
from tools.artifacts.build import (
    _canonical_state_sha256,
    _load_checkpoint_state,
    hash_file,
    verify_checkpoint,
)
from tools.conversion import apply_state_transform
from tools.conversion.esmc_native import esmc_native_to_fastplms_v1
from tools.conversion.verify_esmfold2_backbones import _compare_states


BASES = {"esmc_small": "esmfold2_300", "esmc_large": "esmfold2_600"}


def canonical_state(
    snapshot: Path, source: CheckpointSource
) -> tuple[dict[str, torch.Tensor], str]:
    """Use existing exact conversions to compare native and fused ESMC layouts."""
    state = _load_checkpoint_state(
        snapshot, source
    )  # Each state value has checkpoint-defined shape (...).
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if "esmc.embed_tokens.weight" in state:
        depth = config.get("num_hidden_layers")
        if type(depth) is not int or depth <= 0:
            raise ValueError("native ESMC config must declare num_hidden_layers")
        return esmc_native_to_fastplms_v1(state, depth), "esmc_native_to_fastplms_v1"  # (...) one tensor per parameter name, checkpoint-defined shapes
    converted = apply_state_transform(
        "esmc_to_fastplms_v1", state
    )  # Values and dimensions (...) are preserved.
    if "embed.weight" not in converted or "transformer.norm.weight" not in converted:
        raise ValueError("checkpoint is not a supported native or fused ESMC state")
    return converted, "esmc_to_fastplms_v1"  # (...) one tensor per parameter name, checkpoint-defined shapes


def compare_states(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    """Report exact storage equality separately from the existing BF16 verifier."""
    # left, right: (...) one tensor per parameter name, checkpoint-defined shapes
    # left/right values: heterogeneous checkpoint-defined shapes (...).
    if not left or not right:
        raise ValueError("empty state dictionaries cannot establish compatibility")
    for state in (left, right):
        if any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError("nonfinite checkpoint tensors cannot establish compatibility")
        if any(
            tensor.is_floating_point() and not torch.isfinite(tensor.bfloat16()).all()
            for tensor in state.values()
        ):
            raise ValueError("nonfinite BF16 execution tensors cannot establish compatibility")
    only_left = sorted(left.keys() - right.keys())
    only_right = sorted(right.keys() - left.keys())
    changed = []
    metadata_changed = []
    maximum_difference = 0.0
    for name in sorted(left.keys() & right.keys()):
        first, second = left[name], right[name]  # Each (...), possibly different dimensions.
        if first.shape != second.shape or first.dtype != second.dtype:
            metadata_changed.append(name)
            continue
        if not torch.equal(first, second):
            changed.append(name)
            difference = (
                first.double() - second.double()
            ).abs()  # (...), matching dimensions, measured in FP64.
            maximum_difference = max(maximum_difference, difference.max().item())
    # Reuse the established folding-backbone verifier; right is its native snapshot.
    precision = _compare_states(right, left)
    bf16_equal = not any(
        precision[key]
        for key in ("missing", "unexpected", "shape_mismatches", "bf16_unequal_names")
    )
    return {
        "exactly_equal": not (only_left or only_right or changed or metadata_changed),
        "left_tensors": len(left),
        "right_tensors": len(right),
        "left_state_sha256": _canonical_state_sha256(left),
        "right_state_sha256": _canonical_state_sha256(right),
        "only_left": only_left,
        "only_right": only_right,
        "changed_tensor_count": len(changed),
        "changed_tensor_examples": changed[:20],
        "metadata_changed": metadata_changed,
        "maximum_absolute_value_difference": maximum_difference,
        "bf16_execution": {
            "dtype": "bfloat16",
            "exactly_equal": bf16_equal,
            "changed_tensor_count": len(precision["bf16_unequal_names"]),
            "maximum_absolute_value_difference": precision["max_bf16_abs_error"],
            "left_exact_bf16_roundtrip": precision["standard_exact_bf16_roundtrip"],
            "right_exact_bf16_roundtrip": precision["native_exact_bf16_roundtrip"],
        },
    }


def encoder_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Exclude the language-model head only; retain every embedding/transformer parameter."""
    # state: (...) one tensor per parameter name, checkpoint-defined shapes
    # Every retained value keeps its checkpoint-defined shape (...).
    if any(not name.startswith(("embed.", "transformer.", "sequence_head.")) for name in state):
        raise ValueError("unexpected canonical ESMC parameter namespace")
    encoder = {
        name: tensor for name, tensor in state.items() if not name.startswith("sequence_head.")
    }
    if not encoder or "embed.weight" not in encoder or "transformer.norm.weight" not in encoder:
        raise ValueError("canonical ESMC encoder is incomplete")
    return encoder  # (...) one tensor per parameter name, checkpoint-defined shapes


def acquire(source: CheckpointSource, cache: Path) -> tuple[Path, dict[str, object]]:
    """Download only files with immutable identities and verify each actual payload."""
    if source.unresolved_files or not source.files:
        raise ValueError("backbone audit requires a fully pinned source")
    if not any(file.path.endswith(".safetensors") for file in source.files):
        raise ValueError("backbone audit requires safetensors weights")
    if any(not file.path.endswith((".safetensors", ".json")) for file in source.files):
        raise ValueError("backbone audit accepts only safetensors and JSON files")
    snapshot = Path(
        snapshot_download(
            source.repo_id,
            revision=source.revision,
            cache_dir=cache,
            allow_patterns=[file.path for file in source.files],
        )
    )
    verify_checkpoint(snapshot, source)
    files = {
        file.path: {
            "declared": file.encoded,
            "sha256": hash_file(snapshot / file.path),
            "bytes": (snapshot / file.path).stat().st_size,
        }
        for file in source.files
    }
    return snapshot, {"repo": source.repo_id, "revision": source.revision, "files": files}


def save(path: Path, receipt: Mapping[str, object]) -> None:
    write_text_atomically(path, indented_json(receipt, allow_nan=False))


def audit(base_id: str, cache: Path, output: Path) -> dict[str, object]:
    """Compare fast/official/structural-backbone pins without altering any model or registry."""
    if output.exists():
        raise FileExistsError("choose a new backbone audit output path")
    registry = get_model_registry()
    base = registry[base_id]
    fold = registry[BASES[base_id]]
    if fold.backbone_model != base_id or fold.backbone is None:
        raise ValueError("folding model does not declare the requested ESMC backbone")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, object] = {
        "schema_version": 2,
        "base": base_id,
        "fold": fold.id,
        "status": "in_progress",
        "started": datetime.now(UTC).isoformat(),
        "source_tree": os.environ.get("WS_TREE_HASH"),
        "torch": torch.__version__,
        "device": "cpu",
        "sources": {},
        "qualified_for_shared_inference": False,
    }
    save(output, receipt)
    sources = {}
    states = {}
    configurations = {}
    tokenizers = {}
    for role, source in (
        ("fast", base.fast),
        ("official", base.official),
        ("structural_backbone", fold.backbone),
    ):
        print(f"Auditing {role}: {source.repo_id} @ {source.revision}", flush=True)
        snapshot, record = acquire(source, cache)
        states[role], conversion = canonical_state(snapshot, source)  # Each state value: (...).
        record["conversion"] = conversion
        sources[role] = record
        configurations[role] = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        tokenizers[role] = json.loads((snapshot / "tokenizer.json").read_text(encoding="utf-8"))
        receipt["sources"] = sources
        save(output, receipt)
    comparisons = {}
    for left, right in (
        ("fast", "official"),
        ("official", "structural_backbone"),
        ("fast", "structural_backbone"),
    ):
        comparisons[f"{left}/{right}"] = {
            "full_state": compare_states(states[left], states[right]),
            "encoder": compare_states(encoder_state(states[left]), encoder_state(states[right])),
            "tokenizer_json_equal": tokenizers[left] == tokenizers[right],
        }
    receipt.update(
        {
            "status": "complete",
            "finished": datetime.now(UTC).isoformat(),
            "configurations": configurations,
            "comparisons": comparisons,
            "identical_encoder_weights": comparisons["fast/structural_backbone"]["encoder"][
                "exactly_equal"
            ],
            "identical_encoder_weights_bf16": comparisons["fast/structural_backbone"]["encoder"][
                "bf16_execution"
            ]["exactly_equal"],
            "limitations": [
                "Checkpoint content only; live inference and shared-tap parity remain untested.",
                "SAE training-backbone compatibility still requires independent evidence.",
                "identical_encoder_weights describes stored values, not execution precision.",
                "FP32 storage differences alone do not imply different BF16 backbones. "
                "BF16 equality does not establish FP32 or FP8 execution equivalence.",
            ],
        }
    )
    save(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", choices=BASES, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = audit(arguments.base, arguments.cache, arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "status": receipt["status"],
                "identical_encoder_weights": receipt["identical_encoder_weights"],
                "identical_encoder_weights_bf16": receipt["identical_encoder_weights_bf16"],
                "qualified_for_shared_inference": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
