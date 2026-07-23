"""Validate one installed FastPLMs wheel without checkpoint or GPU access."""

from __future__ import annotations

import argparse
import compileall
import importlib
import importlib.util
import json
import os
import socket
import sys
from importlib.metadata import files, metadata, version
from pathlib import Path, PurePosixPath

_OFFLINE_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
_SENSITIVE_NAMES = {".env", ".git", "credentials", "credentials.json", "id_rsa"}
_SENSITIVE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}


def _contained_by(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _network_blocked(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("Network access is forbidden in the installed-wheel smoke")


def run_smoke(expected_python: str, source_root: Path) -> dict[str, object]:
    """Return evidence for an isolated, CPU-only package installation."""

    for name, expected_value in _OFFLINE_ENVIRONMENT.items():
        if os.environ.get(name) != expected_value:
            raise AssertionError(
                f"The installed-wheel smoke requires {name}={expected_value!r}."
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

    import torch

    import fastplms
    from fastplms.models.esm2.modeling_fastesm import FastEsmConfig, FastEsmModel
    from fastplms.registry import get_model_registry

    package_root = Path(fastplms.__file__).resolve().parent
    environment_root = Path(sys.prefix).resolve()
    source_root = source_root.resolve()
    if not _contained_by(package_root, environment_root):
        raise AssertionError(f"FastPLMs did not load from the isolated environment: {package_root}")
    if _contained_by(package_root, source_root):
        raise AssertionError(f"FastPLMs loaded from repository source: {package_root}")
    if "site-packages" not in package_root.parts:
        raise AssertionError(f"FastPLMs did not load from site-packages: {package_root}")

    package_metadata = metadata("fastplms")
    requires_python = package_metadata["Requires-Python"]
    python_bounds = (
        set() if requires_python is None else set(requires_python.replace(" ", "").split(","))
    )
    if python_bounds != {">=3.11", "<3.15"}:
        raise AssertionError(f"Unexpected Requires-Python metadata: {requires_python!r}")
    if fastplms.__version__ != "1.0.0" or version("fastplms") != "1.0.0":
        raise AssertionError("Package and distribution versions must both be 1.0.0.")
    if version("torch").split("+", maxsplit=1)[0] != "2.13.0":
        raise AssertionError(f"Expected Torch 2.13.0, found {version('torch')}.")
    if version("transformers") != "5.13.0":
        raise AssertionError(f"Expected Transformers 5.13.0, found {version('transformers')}.")
    cuda_is_available = torch.cuda.is_available()
    if torch.version.cuda is not None or cuda_is_available:
        raise AssertionError("The clean-wheel smoke must use the CPU-only Torch distribution.")
    if importlib.util.find_spec("flash_attn") is not None:
        raise AssertionError(
            "A source FlashAttention distribution is present in the core environment."
        )

    distribution_files = files("fastplms")
    if distribution_files is None:
        raise AssertionError("The installed distribution has no RECORD inventory.")
    packaged_paths: set[str] = set()
    for raw_path in distribution_files:
        value = str(raw_path)
        if "\\" in value:
            raise AssertionError(f"Wheel RECORD has a non-portable path: {value!r}")
        path = PurePosixPath(value)
        lowered = tuple(part.lower() for part in path.parts)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise AssertionError(f"Wheel RECORD has an unsafe path: {value!r}")
        if any(part in _SENSITIVE_NAMES for part in lowered):
            raise AssertionError(f"Wheel RECORD includes a sensitive path: {value!r}")
        if path.suffix.lower() in _SENSITIVE_SUFFIXES:
            raise AssertionError(f"Wheel RECORD includes a sensitive suffix: {value!r}")
        if "__pycache__" in lowered or path.suffix.lower() in {".pyc", ".pyo"}:
            raise AssertionError(f"Wheel RECORD includes generated bytecode: {value!r}")
        packaged_paths.add(value)
    for required_suffix in ("fastplms/models.toml", ".dist-info/kernels.lock"):
        if not any(value.endswith(required_suffix) for value in packaged_paths):
            raise AssertionError(f"Wheel RECORD is missing {required_suffix!r}.")

    registry = get_model_registry()
    if len(registry.families) != 10 or len(tuple(registry)) != 29:
        raise AssertionError(
            "The packaged registry must contain 10 families and 29 checkpoints."
        )
    family_maps = {spec.family.id: spec.auto_map for spec in registry.values()}
    advertised_entries = sum(len(auto_map) for auto_map in family_maps.values())
    if advertised_entries != 37:
        raise AssertionError(
            f"Expected 37 advertised Auto entries, found {advertised_entries}."
        )
    imported_entries: list[str] = []
    for family_id, auto_map in sorted(family_maps.items()):
        for auto_class, class_path in sorted(auto_map.items()):
            module_name, separator, class_name = class_path.rpartition(".")
            if not separator:
                raise AssertionError(f"Invalid AutoMap path: {class_path!r}")
            value = getattr(importlib.import_module(module_name), class_name)
            if not isinstance(value, type):
                raise AssertionError(f"AutoMap target is not a class: {class_path}")
            imported_entries.append(f"{family_id}:{auto_class}")
    if not compileall.compile_dir(package_root, force=True, quiet=1):
        raise AssertionError("Installed FastPLMs sources did not compile.")

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
    model = FastEsmModel(config, add_pooling_layer=False).eval()  # type: ignore[no-untyped-call]
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
        "fastplms": version("fastplms"),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "requires_python": requires_python,
        "package_root": str(package_root),
        "model_families": len(registry.families),
        "checkpoints": len(tuple(registry)),
        "advertised_auto_entries": len(imported_entries),
        "distribution_files": len(packaged_paths),
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
