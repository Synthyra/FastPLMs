"""Validate FastPLMs repository source without checkpoint, network, or GPU access."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path


_OFFLINE_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def _network_blocked(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("Network access is forbidden in the repository-source smoke")


def _compile_sources(package_root: Path) -> int:
    source_files = sorted(package_root.rglob("*.py"))
    for path in source_files:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
    return len(source_files)


def run_smoke(expected_python: str, source_root: Path) -> dict[str, object]:
    """Return evidence for an isolated, CPU-only repository-source environment."""

    for name, expected_value in _OFFLINE_ENVIRONMENT.items():
        if os.environ.get(name) != expected_value:
            raise AssertionError(
                f"The repository-source smoke requires {name}={expected_value!r}."
            )

    socket.create_connection = _network_blocked  # type: ignore[assignment]
    socket.getaddrinfo = _network_blocked  # type: ignore[assignment]
    socket.socket.connect = _network_blocked  # type: ignore[method-assign]

    expected = tuple(int(part) for part in expected_python.split("."))
    if sys.version_info[:2] != expected:
        raise AssertionError(
            f"Expected Python {expected_python}, found "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )
    if not (sys.version_info[:2] >= (3, 11) and sys.version_info[:2] < (3, 15)):
        raise AssertionError("Interpreter is outside FastPLMs' supported Python range.")

    source_root = source_root.resolve()
    expected_package_root = source_root / "fastplms"
    if not (expected_package_root / "models.toml").is_file():
        raise AssertionError(f"FastPLMs source is incomplete: {expected_package_root}")
    sys.path.insert(0, str(source_root))

    import torch

    import fastplms
    from fastplms.models.esm2.modeling_fastesm import FastEsmConfig, FastEsmModel
    from fastplms.registry import get_model_registry

    package_root = Path(fastplms.__file__).resolve().parent
    if package_root != expected_package_root:
        raise AssertionError(
            f"FastPLMs did not load from the requested source root: {package_root}"
        )
    if fastplms.__version__ != "1.0.0":
        raise AssertionError(f"Unexpected FastPLMs source version: {fastplms.__version__!r}")
    if importlib.metadata.version("torch").split("+", maxsplit=1)[0] != "2.13.0":
        raise AssertionError(
            f"Expected Torch 2.13.0, found {importlib.metadata.version('torch')}."
        )
    if importlib.metadata.version("transformers") != "5.13.0":
        raise AssertionError(
            "Expected Transformers 5.13.0, found "
            f"{importlib.metadata.version('transformers')}."
        )
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise AssertionError("The repository-source smoke must use the CPU-only Torch build.")
    if importlib.util.find_spec("flash_attn") is not None:
        raise AssertionError("FlashAttention is present in the core source environment.")

    registry = get_model_registry()
    if len(registry.families) != 10 or len(tuple(registry)) != 29:
        raise AssertionError("The source registry must contain 10 families and 29 checkpoints.")
    family_maps = {spec.family.id: spec.auto_map for spec in registry.values()}
    advertised_entries = sum(len(auto_map) for auto_map in family_maps.values())
    if advertised_entries != 37:
        raise AssertionError(f"Expected 37 advertised Auto entries, found {advertised_entries}.")

    imported_entries: list[str] = []
    for family_id, auto_map in sorted(family_maps.items()):
        for auto_class, class_path in sorted(auto_map.items()):
            module_name, separator, class_name = class_path.rpartition(".")
            if not separator:
                raise AssertionError(f"Invalid AutoMap path: {class_path!r}")
            auto_class_type = getattr(importlib.import_module(module_name), class_name)
            if not isinstance(auto_class_type, type):
                raise AssertionError(f"AutoMap target is not a class: {class_path}")
            imported_entries.append(f"{family_id}:{auto_class}")

    source_files = _compile_sources(package_root)
    config = FastEsmConfig(
        vocab_size=33,
        mask_token_id=32,
        pad_token_id=1,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=16,
        position_embedding_type="rotary",
        token_dropout=False,
        attn_backend="eager",
    )
    model = FastEsmModel(config, add_pooling_layer=False).eval()
    input_ids = torch.tensor(((0, 5, 6, 2, 1),), dtype=torch.long, device="cpu")
    attention_mask = input_ids.ne(1)
    with torch.inference_mode():
        # H is the hidden-state tensor with shape (b, l, d).
        hidden_states = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
    if tuple(hidden_states.shape) != (1, 5, 16):
        raise AssertionError(f"Unexpected hidden-state shape: {tuple(hidden_states.shape)}")
    if hidden_states.device.type != "cpu" or not torch.isfinite(hidden_states).all():
        raise AssertionError("The CPU construction smoke produced an invalid tensor.")
    if torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        raise AssertionError("The CPU support smoke initialized CUDA.")

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "fastplms": fastplms.__version__,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "package_root": str(package_root),
        "model_families": len(registry.families),
        "checkpoints": len(tuple(registry)),
        "advertised_auto_entries": len(imported_entries),
        "source_files": source_files,
        "hidden_state_shape": list(hidden_states.shape),
        "device": hidden_states.device.type,
        "cpu_only_torch": torch.version.cuda is None,
        "cuda_initialized": torch.cuda.is_initialized(),  # type: ignore[no-untyped-call]
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-python", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    print(
        json.dumps(
            run_smoke(arguments.expected_python, arguments.source_root),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
