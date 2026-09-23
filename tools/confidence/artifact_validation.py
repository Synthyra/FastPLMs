"""Validate a packaged Hub model in a process without source-checkout imports."""

from __future__ import annotations

import hashlib
import json
import sys

import gemmi
import torch

from pathlib import Path
from typing import Any

from safetensors.torch import load_file
from transformers import AutoModel


SEQUENCE = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


def _validate_outputs(
    output: Any, samples: int, atoms: int, tokens: int
) -> dict[str, list[int]]:
    expected = {
        "plddt_per_atom": ((samples, atoms), 1.0),
        "plddt": ((samples, tokens), 1.0),
        "pae": ((samples, tokens, tokens), 32.0),
        "ptm": ((samples,), 1.0),
        "iptm": ((samples,), 1.0),
    }
    shapes = {}
    for name, (shape, maximum) in expected.items():
        value = output[name]  # expected[name] shape; validated below
        if value is None or tuple(value.shape) != shape:
            raise ValueError(f"Invalid confidence output shape: {name}")
        if (
            not torch.isfinite(value).all()
            or (value < 0).any()
            or (value > maximum).any()
        ):
            raise ValueError(f"Invalid confidence range: {name}")
        shapes[name] = list(value.shape)
    return shapes


def _validate_embedded_head(model: Any, expected_head: Path) -> str:
    with expected_head.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    release = getattr(model.config, "confidence_head_release", None)
    if not isinstance(release, dict) or release.get("head_sha256") != digest:
        raise ValueError(
            "Embedded head release identity differs from the evaluated snapshot"
        )
    if model.config.confidence_head_source is not None or model.confidence_head is None:
        raise ValueError(
            "Released confidence must be embedded without an external source"
        )
    expected = load_file(
        str(expected_head)
    )  # native parameter-specific head tensor shapes
    loaded = model.confidence_head.state_dict()  # same named native head tensor shapes
    if expected.keys() != loaded.keys():
        raise ValueError("Reloaded confidence head has different parameter keys")
    for name, value in expected.items():
        actual = loaded[name].detach().cpu()  # same parameter shape as expected[name]
        if (
            actual.shape != value.shape
            or actual.dtype != value.dtype
            or not torch.equal(
                actual.contiguous().reshape(-1).view(torch.uint8),
                value.contiguous().reshape(-1).view(torch.uint8),
            )
        ):
            raise ValueError(
                f"Reloaded confidence tensor differs from evaluated head: {name}"
            )
    return digest


def _validate_reload(
    artifact: Path, expected_head: Path | None = None
) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    model, loading = AutoModel.from_pretrained(
        str(artifact),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
        esmc_precision="bf16",
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Artifact reload state mismatch: {loading}")
    if not model.config.confidence_head.enabled or model.confidence_head is None:
        raise ValueError("Packaged confidence must be enabled by default")
    head_hash = _validate_embedded_head(model, expected_head) if expected_head else None
    model = model.cuda().eval().requires_grad_(False)
    model.set_chunk_size(32)
    from fastplms.models.esmfold2.esmfold2_types import (
        ProteinInput,
        StructurePredictionInput,
    )

    inputs = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence=SEQUENCE),
            ProteinInput(id="B", sequence=SEQUENCE),
        ]
    )
    features, _ = model.prepare_structure_input(inputs, seed=17)
    features = {name: value.cuda() for name, value in features.items()}
    atoms = features["atom_attention_mask"].shape[-1]
    tokens = features["token_attention_mask"].shape[-1]
    shapes = {}
    for samples in (1, 2):
        settings = dict(
            num_loops=3,
            num_sampling_steps=15,
            num_diffusion_samples=samples,
            seed=17,
            return_dict=True,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            disabled = model(**features, calculate_confidence=False, **settings)
            enabled = model(**features, calculate_confidence=True, **settings)
        if not torch.equal(disabled.sample_atom_coords, enabled.sample_atom_coords):
            raise ValueError(
                f"Confidence changed seeded coordinates for {samples} samples"
            )
        if any(
            disabled.get(name) is not None for name in ("plddt", "pae", "ptm", "iptm")
        ):
            raise ValueError("Disabled confidence still produced scores")
        shapes[str(samples)] = _validate_outputs(enabled, samples, atoms, tokens)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = model.fold(inputs, seed=17, num_loops=3, num_sampling_steps=15)
    cif = model.result_to_cif(result)
    (artifact.parent / "validation-complex.cif").write_text(cif)
    block = gemmi.cif.read_string(cif).sole_block()
    chains = set(block.find_values("_atom_site.label_asym_id"))
    values = list(block.find_values("_atom_site.B_iso_or_equiv"))
    if len(chains) != 2 or not values or any(value in {"?", "."} for value in values):
        raise ValueError("CIF must contain two chains and known atom confidence values")
    confidence = [float(value) for value in values]
    if any(not 0 <= value <= 100 for value in confidence) or not any(
        value > 0 for value in confidence
    ):
        raise ValueError("Invalid CIF confidence values")
    report = {
        "coordinate_equality": True,
        "samples": [1, 2],
        "seed": 17,
        "loops": 3,
        "diffusion_steps": 15,
        "dtype": "fp32 parameters, bf16 autocast",
        "backend": "sdpa",
        "confidence_shapes": shapes,
        "cif_atom_count": len(values),
        "cif_chain_ids": sorted(chains),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
    }
    if expected_head is not None:
        with (artifact / "model.safetensors").open("rb") as stream:
            weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        report.update(
            {
                "status": "passed",
                "head_sha256": head_hash,
                "weight_sha256": weight_hash,
                "checks": {
                    "strict_reload": True,
                    "embedded_head_identity": True,
                    "confidence_default_enabled": True,
                    "confidence_ranges": True,
                    "seeded_coordinate_equality": True,
                    "cif_confidence": True,
                },
            }
        )
    return report


if __name__ == "__main__":
    artifact = Path(sys.argv[1])
    expected_head = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    report = _validate_reload(artifact, expected_head)
    (artifact.parent / "reload-validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
