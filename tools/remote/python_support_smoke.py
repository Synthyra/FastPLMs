"""Validate one installed FastPLMs wheel without checkpoint or GPU access."""

from __future__ import annotations

import argparse
import compileall
import importlib.util
import json
import sys
from importlib.metadata import metadata, version
from pathlib import Path


def _contained_by(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def run_smoke(expected_python: str, source_root: Path) -> dict[str, object]:
    """Return evidence for an isolated, CPU-only package installation."""

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
    if importlib.util.find_spec("flash_attn") is not None:
        raise AssertionError(
            "A source FlashAttention distribution is present in the core environment."
        )

    registry = get_model_registry()
    if not registry.families or not tuple(registry):
        raise AssertionError("The packaged model registry is empty.")
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
    if torch.cuda.is_initialized():
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
        "hidden_state_shape": list(hidden_states.shape),
        "device": hidden_states.device.type,
        "cuda_initialized": torch.cuda.is_initialized(),
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
