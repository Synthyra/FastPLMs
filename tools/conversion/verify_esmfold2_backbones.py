"""Verify ESMFold2 ESMC snapshots against their pinned ESM++ backbones."""

from __future__ import annotations

import argparse
import json

from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import torch
from safetensors.torch import load_file

from fastplms.registry import CheckpointSource, FileDigest, get_model_registry
from tools.artifacts.build import ArtifactError, hash_file
from tools.conversion.esmc_native import esmc_native_to_fastplms_v1


def _weight_digest(source: CheckpointSource) -> FileDigest:
    weights = tuple(
        item for item in source.files if PurePosixPath(item.path).suffix.lower() == ".safetensors"
    )
    if len(weights) != 1:
        raise ArtifactError(
            f"Expected one safetensors weight file for {source.repo_id}, found {len(weights)}."
        )
    return weights[0]


def _required_digests(source: CheckpointSource) -> tuple[FileDigest, FileDigest]:
    config = source.file_map.get("config.json")
    if config is None:
        raise ArtifactError(f"Pinned source {source.repo_id} has no config.json digest.")
    return config, _weight_digest(source)


def _verify_config_and_weights(snapshot: Path, source: CheckpointSource) -> dict[str, str]:
    """Verify the manifest identities needed to compare model tensors."""

    snapshot = snapshot.resolve()
    if not snapshot.is_dir():
        raise ArtifactError(f"Checkpoint snapshot does not exist: {snapshot}")
    records: dict[str, str] = {}
    for expected in _required_digests(source):
        path = snapshot.joinpath(*PurePosixPath(expected.path).parts)
        if not path.is_file():
            raise ArtifactError(f"Missing pinned file {expected.path} in {snapshot}")
        actual = hash_file(path, expected.algorithm)
        if actual != expected.digest:
            raise ArtifactError(
                f"{expected.path}: expected {expected.encoded}, "
                f"received {expected.algorithm}:{actual}"
            )
        records[expected.path] = f"{expected.algorithm}:{actual}"
    return records


def _source_pin(source: CheckpointSource, file_hashes: Mapping[str, str]) -> dict[str, object]:
    return {
        "repo_id": source.repo_id,
        "revision": source.revision,
        "files": dict(sorted(file_hashes.items())),
    }


def _load_snapshot_state(snapshot: Path, source: CheckpointSource) -> dict[str, torch.Tensor]:
    weight = _weight_digest(source)
    path = snapshot.resolve().joinpath(*PurePosixPath(weight.path).parts)
    return dict(load_file(path, device="cpu"))


def _load_config(snapshot: Path) -> dict[str, object]:
    path = snapshot.resolve() / "config.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Unable to read checkpoint config: {path}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"Checkpoint config must be a JSON object: {path}")
    return value


def _required_int(config: Mapping[str, object], key: str, path: Path) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactError(f"Checkpoint config {path} has invalid {key}: {value!r}")
    return value


def _compare_states(
    native_state: Mapping[str, torch.Tensor],
    standard_state: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    native_keys = set(native_state)
    standard_keys = set(standard_state)
    missing = sorted(standard_keys - native_keys)
    unexpected = sorted(native_keys - standard_keys)
    shape_mismatches = sorted(
        name
        for name in native_keys & standard_keys
        if native_state[name].shape != standard_state[name].shape
    )
    fp32_unequal_names: list[str] = []
    bf16_unequal_names: list[str] = []
    max_bf16_error = 0.0
    for name in sorted(native_keys & standard_keys):
        native = native_state[name]
        standard = standard_state[name]
        if native.shape != standard.shape:
            continue
        if not torch.equal(native, standard):
            fp32_unequal_names.append(name)
        native_bf16 = native.bfloat16().float() if native.is_floating_point() else native
        standard_bf16 = standard.bfloat16().float() if standard.is_floating_point() else standard
        if not torch.equal(native_bf16, standard_bf16):
            bf16_unequal_names.append(name)
        if native_bf16.is_floating_point():
            error = torch.max(torch.abs(native_bf16 - standard_bf16)).item()
            max_bf16_error = max(max_bf16_error, float(error))

    def exact_bf16_roundtrip(state: Mapping[str, torch.Tensor]) -> bool:
        return all(
            not value.is_floating_point() or torch.equal(value, value.bfloat16().float())
            for value in state.values()
        )

    return {
        "tensor_count": len(standard_state),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "fp32_unequal_count": len(fp32_unequal_names),
        "fp32_unequal_names_preview": fp32_unequal_names[:5],
        "bf16_unequal_names": bf16_unequal_names,
        "max_bf16_abs_error": max_bf16_error,
        "native_exact_bf16_roundtrip": exact_bf16_roundtrip(native_state),
        "standard_exact_bf16_roundtrip": exact_bf16_roundtrip(standard_state),
    }


def verify_backbone_pair(
    model_id: str,
    native_snapshot: Path,
    standard_snapshot: Path,
) -> dict[str, object]:
    """Verify one native step-1500000 and standard ESM++ snapshot pair."""

    registry = get_model_registry()
    spec = registry[model_id]
    if spec.family.id != "esmfold2" or spec.backbone is None or spec.backbone_model is None:
        raise ArtifactError(f"{model_id} is not an ESMFold2 model with a pinned backbone.")
    standard_spec = registry[spec.backbone_model]
    native_files = _verify_config_and_weights(native_snapshot, spec.backbone)
    standard_files = _verify_config_and_weights(standard_snapshot, standard_spec.fast)
    native_config = _load_config(native_snapshot)
    standard_config = _load_config(standard_snapshot)
    native_layers = _required_int(
        native_config,
        "num_hidden_layers",
        native_snapshot / "config.json",
    )
    standard_layers = _required_int(
        standard_config,
        "num_hidden_layers",
        standard_snapshot / "config.json",
    )
    if native_layers != standard_layers:
        raise ArtifactError(
            f"Layer count mismatch: native={native_layers}, standard={standard_layers}"
        )
    native_state = _load_snapshot_state(native_snapshot, spec.backbone)
    converted_state = esmc_native_to_fastplms_v1(native_state, native_layers)
    standard_state = _load_snapshot_state(standard_snapshot, standard_spec.fast)
    comparison = _compare_states(converted_state, standard_state)
    if comparison["missing"] or comparison["unexpected"] or comparison["shape_mismatches"]:
        raise ArtifactError(
            f"Converted ESMC structure does not match {standard_spec.id}: {comparison}"
        )
    if comparison["bf16_unequal_names"]:
        raise ArtifactError(f"Converted ESMC tensors differ after BF16 cast: {comparison}")
    return {
        "schema_version": 1,
        "status": "passed",
        "model_id": model_id,
        "backbone_model_id": spec.backbone_model,
        "native": _source_pin(spec.backbone, native_files),
        "standard": _source_pin(standard_spec.fast, standard_files),
        "comparison": comparison,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--native-snapshot", type=Path, required=True)
    parser.add_argument("--standard-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = verify_backbone_pair(
        arguments.model_id,
        arguments.native_snapshot,
        arguments.standard_snapshot,
    )
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"))
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
