"""Run warm-cache FlashAttention parity on representative checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import torch
from collections.abc import Sequence
from typing import Any
from torch.nn import functional as F
from transformers import EsmTokenizer

from fastplms.models.dplm.modeling_dplm import DPLMModel
from fastplms.models.esm2.modeling_fastesm import FastEsmForMaskedLM
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusForMaskedLM,
    EsmSequenceTokenizer,
)
from fastplms.registry import get_model_registry


BACKENDS = ("flash_attention_2", "flash_attention_3")
CHECKPOINTS = {
    "esm2_8m": FastEsmForMaskedLM,
    "esmc_small": ESMplusplusForMaskedLM,
    "dplm_150m": DPLMModel,
}
BACKENDS_BY_CHECKPOINT = {
    "esm2_8m": BACKENDS,
    "esmc_small": BACKENDS,
    "dplm_150m": ("flash_attention_3",),
}


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    # actual, expected: (n_valid, d)
    actual_float = actual.float()  # (n_valid, d)
    expected_float = expected.float()  # (n_valid, d)
    relative_l2 = (
        torch.linalg.vector_norm(actual_float - expected_float)
        / torch.linalg.vector_norm(expected_float).clamp_min(1e-12)
    ).item()  # scalar
    cosine_values = F.cosine_similarity(  # (n_valid,)
        actual_float,
        expected_float,
        dim=-1,
    )
    cosine = cosine_values.min().item()
    return {"relative_l2": relative_l2, "minimum_cosine": cosine}


def _hidden_state(output: object) -> torch.Tensor:
    value = getattr(output, "last_hidden_state", None)
    if not torch.is_tensor(value):
        raise TypeError("Checkpoint output omitted last_hidden_state.")
    return value  # (b, l, d)


def _run_checkpoint(
    model_id: str,
    model_class: type[torch.nn.Module],
) -> dict[str, Any]:
    spec = get_model_registry()[model_id]
    if spec.family.id == "esm_plusplus":
        tokenizer = EsmSequenceTokenizer()
    else:
        tokenizer = EsmTokenizer.from_pretrained(
            spec.fast.repo_id,
            revision=spec.fast.revision,
        )
    batch = tokenizer(
        ["MSTNPKPQRKTKRNTNR", "ACDEFGHIK"],
        return_tensors="pt",
        padding=True,
    )
    # Each tokenized batch value has shape (b=2, l).
    batch = {name: value.to("cuda") for name, value in batch.items()}  # values: (b, l)
    use_bf16_autocast = spec.family.bf16_execution == "fp32_parameters_autocast"
    load_dtype = torch.float32 if use_bf16_autocast else torch.bfloat16
    model = (
        model_class.from_pretrained(
            spec.fast.repo_id,
            revision=spec.fast.revision,
            dtype=load_dtype,
        )
        .eval()
        .to("cuda")
    )
    outputs: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for backend in ("sdpa", *BACKENDS_BY_CHECKPOINT[model_id]):
            model.set_attn_implementation(backend)
            numeric_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16_autocast
                else contextlib.nullcontext()
            )
            with numeric_context:
                outputs[backend] = _hidden_state(model(**batch)).detach()  # (b, l, d)
    valid = batch["attention_mask"].bool()  # (b, l)
    result = {
        backend: {
            **_metrics(
                outputs[backend][valid],  # (n_valid, d)
                outputs["sdpa"][valid],  # (n_valid, d)
            ),
            "finite": bool(torch.isfinite(outputs[backend]).all()),
        }
        for backend in BACKENDS_BY_CHECKPOINT[model_id]
    }
    if not all(value["finite"] for value in result.values()):
        raise RuntimeError(f"{model_id} produced non-finite checkpoint output.")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "checkpoint": spec.fast.repo_id,
        "revision": spec.fast.revision,
        "backends": result,
    }


def main(argv: Sequence[str] | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint FlashAttention validation.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", action="append", choices=tuple(CHECKPOINTS))
    arguments = parser.parse_args(argv)
    model_ids = tuple(arguments.model_id or CHECKPOINTS)
    result = {
        "device": torch.cuda.get_device_name(0),
        "models": {
            model_id: _run_checkpoint(model_id, CHECKPOINTS[model_id]) for model_id in model_ids
        },
        "torch": torch.__version__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
