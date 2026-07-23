#!/usr/bin/env python3
"""Run ESM2 masked-LM, contact, sequence, and token task heads offline."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def configure_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def resolve_execution(device_name: str, dtype_name: str) -> tuple[Any, Any]:
    """Validate the portable CPU/CUDA execution requested by the user."""

    import torch

    try:
        device = torch.device(device_name)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Invalid execution device {device_name!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Only CPU and CUDA devices are supported, got {device.type!r}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")
    dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    return device, dtype


def _biological_mask(tokenizer: Any, batch: dict[str, Any]) -> Any:
    mask = batch["attention_mask"].bool()
    for token_id in getattr(tokenizer, "all_special_ids", ()):
        mask &= batch["input_ids"].ne(int(token_id))
    return mask


def _loading_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)) and value:
        return str(value[0])
    return str(value)


def _require_checkpoint_heads(
    loading_info: dict[str, Any],
    prefixes: tuple[str, ...],
) -> None:
    """Reject a checkpoint load that silently initialized an advertised trained head."""

    problems: dict[str, list[str]] = {}
    for field in ("missing_keys", "mismatched_keys"):
        matching = [
            key
            for item in loading_info.get(field, ())
            if (key := _loading_key(item)).startswith(prefixes)
        ]
        if matching:
            problems[field] = matching
    error_messages = [str(value) for value in loading_info.get("error_msgs", ())]
    if error_messages:
        problems["error_msgs"] = error_messages
    if problems:
        raise RuntimeError(
            "The local artifact does not contain the complete checkpoint-provided "
            f"masked-LM/contact head state: {problems}"
        )


def _require_finite_tensor(name: str, value: Any) -> None:
    import torch

    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} contained non-finite values")


def run_task_heads(
    artifact: Path,
    sequences: list[str],
    *,
    device: Any,
    dtype: Any,
    attn_backend: str,
    num_labels: int,
) -> dict[str, Any]:
    """Run the trained ESM2 heads and smoke separately initialized task heads."""

    import torch
    from transformers import (
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    common = {
        "trust_remote_code": True,
        "local_files_only": True,
        "attn_implementation": attn_backend,
        "dtype": dtype,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        artifact,
        trust_remote_code=True,
        local_files_only=True,
    )
    batch = tokenizer(sequences, padding=True, return_tensors="pt")
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    biological_mask = _biological_mask(tokenizer, batch)
    if not biological_mask.any(dim=1).all():
        raise ValueError("Every input sequence must contain at least one biological residue")

    masked_lm, loading_info = AutoModelForMaskedLM.from_pretrained(
        artifact,
        output_loading_info=True,
        **common,
    )
    _require_checkpoint_heads(loading_info, ("lm_head.", "esm.contact_head."))
    masked_lm = masked_lm.to(device).eval()
    masked_ids = batch["input_ids"].clone()
    labels = torch.full_like(masked_ids, -100)
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("Masked-LM scoring requires a tokenizer mask token")
    scored_positions: list[int] = []
    for row in range(masked_ids.shape[0]):
        position = int(torch.nonzero(biological_mask[row], as_tuple=False)[0, 0])
        scored_positions.append(position)
        labels[row, position] = masked_ids[row, position]
        masked_ids[row, position] = int(mask_token_id)

    with torch.inference_mode():
        mlm_output = masked_lm(
            input_ids=masked_ids,
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        contacts = masked_lm.predict_contacts(
            batch["input_ids"],
            batch["attention_mask"],
        )
    probabilities = mlm_output.logits.float().softmax(dim=-1)
    residue_probabilities = [
        float(probabilities[row, position, labels[row, position]].item())
        for row, position in enumerate(scored_positions)
    ]
    _require_finite_tensor("Masked-LM probabilities", probabilities)
    _require_finite_tensor("Contact predictions", contacts)

    sequence_model = (
        AutoModelForSequenceClassification.from_pretrained(
            artifact,
            num_labels=num_labels,
            **common,
        )
        .to(device)
        .eval()
    )
    sequence_labels = torch.zeros(len(sequences), dtype=torch.long, device=device)
    with torch.inference_mode():
        sequence_output = sequence_model(**batch, labels=sequence_labels)

    token_model = (
        AutoModelForTokenClassification.from_pretrained(
            artifact,
            num_labels=num_labels,
            **common,
        )
        .to(device)
        .eval()
    )
    token_labels = torch.full_like(batch["input_ids"], -100)
    token_labels[biological_mask] = 0
    with torch.inference_mode():
        token_output = token_model(**batch, labels=token_labels)

    losses = {
        "masked_lm": float(mlm_output.loss.item()),
        "sequence_classification": float(sequence_output.loss.item()),
        "token_classification": float(token_output.loss.item()),
    }
    if not all(math.isfinite(value) for value in losses.values()):
        raise RuntimeError(f"A task-head loss was non-finite: {losses}")
    return {
        "sequences": len(sequences),
        "device": str(device),
        "dtype": str(dtype),
        "attention_backend": attn_backend,
        "masked_lm": {
            "status": "checkpoint-provided pretrained head",
            "loss": losses["masked_lm"],
            "scored_positions": scored_positions,
            "residue_probabilities": residue_probabilities,
        },
        "contacts": {
            "status": "checkpoint-provided pretrained head",
            "shape": list(contacts.shape),
            "finite": True,
        },
        "sequence_classification": {
            "status": "base weights + untrained task head",
            "loss": losses["sequence_classification"],
            "logits_shape": list(sequence_output.logits.shape),
        },
        "token_classification": {
            "status": "base weights + untrained task head",
            "loss": losses["token_classification"],
            "logits_shape": list(token_output.logits.shape),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Local manifest-built ESM2 artifact")
    parser.add_argument("--sequence", action="append", dest="sequences")
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--attn-backend",
        choices=("eager", "sdpa", "flex_attention"),
        default="sdpa",
        help="Portable backend used by every loaded head",
    )
    parser.add_argument("--num-labels", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    artifact = arguments.artifact.expanduser().resolve()
    if not (artifact / "config.json").is_file():
        raise SystemExit(f"Not a local ESM2 artifact: {artifact}")
    if arguments.num_labels < 2:
        raise SystemExit("--num-labels must be at least 2")
    try:
        device, dtype = resolve_execution(arguments.device, arguments.dtype)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    configure_offline()
    summary = run_task_heads(
        artifact,
        arguments.sequences or ["MSTNPKPQRKTKRNT", "MKTII"],
        device=device,
        dtype=dtype,
        attn_backend=arguments.attn_backend,
        num_labels=arguments.num_labels,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
