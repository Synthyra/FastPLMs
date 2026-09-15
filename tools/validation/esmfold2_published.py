"""Load and smoke-test the published ESMFold2-300 artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from tools.validation.esmfold2_small import (
    CHUNK_SIZE,
    NUM_LOOPS,
    NUM_SAMPLES,
    NUM_SAMPLING_STEPS,
    SEED,
    SEQUENCE,
    _environment_metadata,
    _finish_model_load,
)


REPOSITORY = "Synthyra/ESMFold2-300"
SCHEMA_VERSION = 1
EXPECTED_BACKBONE = {"d_model": 960, "n_heads": 15, "n_layers": 30}
CONFIDENCE_OUTPUT_NAMES = {
    "plddt_logits",
    "plddt",
    "plddt_per_atom",
    "plddt_ca",
    "complex_plddt",
    "complex_iplddt",
    "pae_logits",
    "pae",
    "pde_logits",
    "pde",
    "ptm",
    "iptm",
    "pair_chains_iptm",
    "resolved_logits",
}


def _config_int(config: object, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _set_sdpa(model: torch.nn.Module) -> None:
    for owner in (model, getattr(model, "_esmc", None)):
        if owner is None:
            continue
        config = getattr(owner, "config", None)
        if config is not None:
            if hasattr(config, "attn_implementation"):
                config.attn_implementation = "sdpa"
            if hasattr(config, "attn_backend"):
                config.attn_backend = "sdpa"
            if hasattr(config, "esmc_attn_backend"):
                config.esmc_attn_backend = "sdpa"
        setter = getattr(owner, "set_attn_implementation", None)
        if callable(setter):
            setter("sdpa")


def _validate_backbone(model: torch.nn.Module) -> dict[str, int]:
    backbone = getattr(model, "_esmc", None)
    if backbone is None:
        raise RuntimeError("Published ESMFold2 model did not load its ESMC backbone.")
    config = backbone.config
    observed = {
        "d_model": _config_int(config, "d_model", "hidden_size"),
        "n_heads": _config_int(config, "n_heads", "num_attention_heads"),
        "n_layers": _config_int(config, "n_layers", "num_hidden_layers"),
    }
    if observed != EXPECTED_BACKBONE:
        raise RuntimeError(f"Published backbone dimensions differ: {observed}")
    return observed


def _validate_outputs(output: Mapping[str, Any]) -> dict[str, list[int]]:
    coordinate_tensor = output.get("sample_atom_coords")
    if not torch.is_tensor(coordinate_tensor):
        raise RuntimeError("Published ESMFold2 output omitted sample_atom_coords.")
    coordinates = coordinate_tensor.float()  # (b, s, a, 3)
    if coordinates.ndim == 3:
        coordinates = coordinates.unsqueeze(1)
    if coordinates.ndim != 4 or coordinates.shape[-1] != 3:
        raise RuntimeError(f"Unexpected coordinate shape: {tuple(coordinates.shape)}")
    if not torch.isfinite(coordinates).all():
        raise RuntimeError("Published ESMFold2 coordinates contain NaN or infinity.")
    for name, value in output.items():
        if name.endswith("coords") and torch.is_tensor(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"Published ESMFold2 {name} contains NaN or infinity.")
    for name in CONFIDENCE_OUTPUT_NAMES:
        if name in output and output[name] is not None:
            raise RuntimeError(f"Confidence output {name!r} was emitted unexpectedly.")
    shapes = {name: list(value.shape) for name, value in output.items() if torch.is_tensor(value)}
    return shapes


def run_published_check(revision: str, *, local_files_only: bool) -> dict[str, Any]:
    """Run the fixed ESMFold2-300 public-artifact inference case."""

    if not torch.cuda.is_available():
        raise RuntimeError("Published ESMFold2 validation requires CUDA.")
    from transformers import AutoConfig, AutoModel

    torch.cuda.reset_peak_memory_stats()
    config = AutoConfig.from_pretrained(
        REPOSITORY,
        revision=revision,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    config.attn_implementation = "sdpa"
    if hasattr(config, "esmc_attn_backend"):
        config.esmc_attn_backend = "sdpa"
    model = AutoModel.from_pretrained(
        REPOSITORY,
        revision=revision,
        config=config,
        trust_remote_code=True,
        dtype=torch.float32,
        device_map="cuda",
        esmc_precision="bf16",
        local_files_only=local_files_only,
    )
    model = model.eval()
    _set_sdpa(model)
    _finish_model_load(model)
    backbone_dimensions = _validate_backbone(model)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.infer_protein(
            SEQUENCE,
            num_loops=NUM_LOOPS,
            num_diffusion_samples=NUM_SAMPLES,
            num_sampling_steps=NUM_SAMPLING_STEPS,
            seed=SEED,
            calculate_confidence=False,
        )
    shapes = _validate_outputs(output)
    precision_status = getattr(model, "esmc_precision_status", None)
    status = (
        precision_status.as_dict()
        if hasattr(precision_status, "as_dict")
        else str(precision_status)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "repository": REPOSITORY,
        "revision": revision,
        "local_files_only": local_files_only,
        "sequence": SEQUENCE,
        "seed": SEED,
        "num_loops": NUM_LOOPS,
        "num_diffusion_samples": NUM_SAMPLES,
        "num_sampling_steps": NUM_SAMPLING_STEPS,
        "chunk_size": CHUNK_SIZE,
        "attention_backend": "sdpa",
        "folding_parameter_dtype": "float32",
        "backbone_compute_dtype": "bfloat16",
        "backbone_dimensions": backbone_dimensions,
        "precision_status": status,
        "output_shapes": shapes,
        "environment": _environment_metadata(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = run_published_check(
            arguments.revision,
            local_files_only=arguments.local_files_only,
        )
        exit_code = 0
    except Exception as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "repository": REPOSITORY,
            "revision": arguments.revision,
            "failure": f"{type(error).__name__}: {error}",
        }
        exit_code = 1
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report["status"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
