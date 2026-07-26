#!/usr/bin/env python3
"""Run deterministic DPLM, DPLM2, or conditioned ESM3 generation offline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


if __package__:
    from ._runtime import add_execution_arguments, resolve_execution
else:
    from _runtime import add_execution_arguments, resolve_execution


def configure_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def generate_dplm(model: Any, tokenizer: Any, length: int, steps: int, seed: int) -> Any:
    import torch

    input_ids = tokenizer("A" * length, return_tensors="pt")["input_ids"].to(
        model.device
    )  # (1, l_t)
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(seed)
        output_tokens = model.generate(
            input_ids,
            max_iter=steps,
            sampling_strategy="argmax",
            disable_resample=True,
        )  # (1, l_t)
    return output_tokens  # (1, l_t)


def generate_dplm2(model: Any, tokenizer: Any, length: int, steps: int, seed: int) -> Any:
    import torch

    vocab = tokenizer.get_vocab()
    structure = [
        vocab["<cls_struct>"],
        *([vocab["<mask_struct>"]] * length),
        vocab["<eos_struct>"],
    ]
    amino_acids = [
        vocab["<cls_aa>"],
        *([vocab["<mask_aa>"]] * length),
        vocab["<eos_aa>"],
    ]
    input_ids = torch.tensor(
        [structure + amino_acids], device=model.device
    )  # (1, 2 * (l + 2))
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(seed)
        output_tokens = model.generate(
            input_ids,
            max_iter=steps,
            sampling_strategy="argmax",
            unmasking_strategy="deterministic",
        )["output_tokens"]  # (1, 2 * (l + 2))
    return output_tokens  # (1, 2 * (l + 2))


def generate_esm3(model: Any, request: str | dict[str, Any], steps: int, seed: int) -> Any:
    from fastplms.models.esm3.modeling_esm3 import FastESM3GenerationConfig

    return model.generate(
        request,
        FastESM3GenerationConfig(num_steps=steps, temperature=1.0, seed=seed),
    )


def build_esm3_multimodal_request(model: Any, prompt: str) -> dict[str, Any]:
    """Build a synthetic request carrying every supported conditioning track.

    Replace these valid placeholder tracks with model-prepared biological
    conditioning in a scientific workflow.
    """
    import torch

    encoded = model.encode(
        prompt, device=model.device
    )  # input_ids/attention_mask: (b, l)
    sequence_tokens = encoded["input_ids"]  # (b, l)
    shape = sequence_tokens.shape
    device = sequence_tokens.device
    return {
        "sequence_tokens": sequence_tokens,  # (b, l)
        "attention_mask": encoded["attention_mask"],  # (b, l)
        "structure_tokens": torch.zeros(
            shape, dtype=torch.long, device=device
        ),  # (b, l)
        "ss8_tokens": torch.zeros(shape, dtype=torch.long, device=device),  # (b, l)
        "sasa_tokens": torch.zeros(shape, dtype=torch.long, device=device),  # (b, l)
        "function_tokens": torch.zeros(
            (*shape, 8), dtype=torch.long, device=device
        ),  # (b, l, 8)
        "residue_annotation_tokens": torch.zeros(
            (*shape, 16), dtype=torch.long, device=device
        ),  # (b, l, 16)
        "average_plddt": torch.ones(shape, device=device),  # (b, l)
        "per_res_plddt": torch.zeros(shape, device=device),  # (b, l)
        "structure_coords": torch.full(
            (*shape, 3, 3), float("nan"), device=device
        ),  # (b, l, 3, 3)
        "chain_id": torch.zeros(shape, dtype=torch.long, device=device),  # (b, l)
        "sequence_id": torch.ones(shape, dtype=torch.bool, device=device),  # (b, l)
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("dplm", "dplm2", "esm3"))
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--esm3-prompt", default="MK____A")
    add_execution_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    artifact = arguments.artifact.expanduser().resolve()
    if not (artifact / "config.json").is_file():
        raise SystemExit(f"Not a local artifact: {artifact}")
    try:
        device, dtype = resolve_execution(arguments.device, arguments.dtype)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    configure_offline()
    if arguments.family == "esm3":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            artifact,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
        ).to(device).eval()
        request = build_esm3_multimodal_request(model, arguments.esm3_prompt)
        output = generate_esm3(model, request, arguments.steps, arguments.seed)
    else:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        model = AutoModelForMaskedLM.from_pretrained(
            artifact,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            artifact,
            trust_remote_code=True,
            local_files_only=True,
        )
        if arguments.family == "dplm":
            output = generate_dplm(
                model,
                tokenizer,
                arguments.length,
                arguments.steps,
                arguments.seed,
            )
        else:
            output = generate_dplm2(
                model,
                tokenizer,
                arguments.length,
                arguments.steps,
                arguments.seed,
            )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
