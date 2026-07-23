#!/usr/bin/env python3
"""Extract ANKH hidden states and run task-prompted seq2seq generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

if __package__:
    from ._runtime import add_execution_arguments, resolve_execution
else:
    from _runtime import add_execution_arguments, resolve_execution

from fastplms.models.ankh.modeling_ankh import (
    tokenize_ankh_decoder_prompts,
    tokenize_ankh_sequences,
)


def configure_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def extract_ankh_layers(
    encoder: Any,
    seq2seq: Any,
    tokenizer: Any,
    sequences: list[str],
    decoder_prompts: list[str],
) -> tuple[Any, Any, Any]:
    encoder_final = encoder.embed_dataset(
        sequences,
        tokenizer=tokenizer,
        hidden_state_source="encoder",
        hidden_state_index=-1,
        full_embeddings=True,
    )
    encoder_all = encoder.embed_dataset(
        sequences,
        tokenizer=tokenizer,
        hidden_state_source="encoder",
        store_all_hidden_states=True,
        full_embeddings=True,
    )
    decoder_final = seq2seq.embed_dataset(
        sequences,
        tokenizer=tokenizer,
        hidden_state_source="decoder",
        hidden_state_index=-1,
        decoder_inputs=decoder_prompts,
        full_embeddings=True,
    )
    return encoder_final, encoder_all, decoder_final


def generate_ankh_task(
    model: Any,
    tokenizer: Any,
    sequence: str,
    decoder_prompt: str,
    *,
    max_new_tokens: int,
) -> Any:
    """Generate after an explicit task prompt without shifting the source."""
    import torch

    encoded = tokenize_ankh_sequences(
        tokenizer,
        sequence,
        return_tensors="pt",
    )
    prompt = tokenize_ankh_decoder_prompts(
        tokenizer,
        decoder_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_ids = prompt["input_ids"]
    decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
    if not isinstance(decoder_start_token_id, int):
        raise RuntimeError("The ANKH artifact does not declare decoder_start_token_id.")
    decoder_input_ids = torch.cat(
        (
            prompt_ids.new_full((prompt_ids.shape[0], 1), decoder_start_token_id),
            prompt_ids,
        ),
        dim=1,
    )
    device = model.device
    generation_inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "decoder_input_ids": decoder_input_ids.to(device),
        "decoder_attention_mask": torch.ones_like(decoder_input_ids, device=device),
    }
    if "attention_mask" in encoded:
        generation_inputs["attention_mask"] = encoded["attention_mask"].to(device)
    with torch.inference_mode():
        return model.generate(
            **generation_inputs,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Full local ANKH 1.0 artifact")
    parser.add_argument("--sequence", default="MSTNPKPQRKTKRNT")
    parser.add_argument("--decoder-prompt", default="M<extra_id_0>")
    parser.add_argument("--max-new-tokens", type=int, default=4)
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
    from transformers import AutoModel, AutoModelForSeq2SeqLM

    common = {
        "trust_remote_code": True,
        "local_files_only": True,
        "dtype": dtype,
    }
    encoder = AutoModel.from_pretrained(artifact, **common).to(device).eval()
    seq2seq = AutoModelForSeq2SeqLM.from_pretrained(artifact, **common).to(device).eval()
    tokenizer = seq2seq.tokenizer
    results = extract_ankh_layers(
        encoder,
        seq2seq,
        tokenizer,
        [arguments.sequence],
        [arguments.decoder_prompt],
    )
    print("encoder-final", tuple(results[0][0].load_tensor().shape))
    print("encoder-all", tuple(results[1][0].load_tensor().shape))
    print("decoder-final", tuple(results[2][0].load_tensor().shape))
    generated = generate_ankh_task(
        seq2seq,
        tokenizer,
        arguments.sequence,
        arguments.decoder_prompt,
        max_new_tokens=arguments.max_new_tokens,
    )
    print("generated", tokenizer.decode(generated[0], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
