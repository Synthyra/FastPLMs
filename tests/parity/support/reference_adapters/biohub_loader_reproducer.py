"""Reproduce the pinned Biohub ESMC public-loader meta-tensor failure."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import torch
from collections.abc import Sequence
from pathlib import Path

from tests.parity.support.reference_adapters import (
    pinned_biohub_snapshot,
    use_esm_submodule,
)
from tests.parity.support.reference_adapters.biohub_source import (
    reference_sources,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-name", default="esmc_300m")
    parser.add_argument(
        "--construction",
        choices=("public", "normal"),
        default="public",
        help="Use the failing public builder or the independent normal constructor probe.",
    )
    return parser


def _load_with_normal_construction(
    repo_id: str,
    revision: str,
    model_name: str,
) -> torch.nn.Module:
    """Construct ESMC normally, then apply the pinned official loader exactly."""

    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers
    from huggingface_hub import load_torch_model
    from safetensors import safe_open

    configurations = {
        "esmc_300m": (960, 15, 30),
        "esmc_600m": (1152, 18, 36),
        "esmc_6b": (2560, 40, 80),
    }
    try:
        d_model, n_heads, n_layers = configurations[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported ESMC model name: {model_name!r}") from error

    with pinned_biohub_snapshot(repo_id, revision) as snapshot:
        model = ESMC(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=False,
        ).eval()
        load_torch_model(model, snapshot)
        checkpoint_path = Path(snapshot) / "model.safetensors"
        state = model.state_dict()
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            checkpoint_keys = set(checkpoint.keys())
            state_keys = set(state)
            if checkpoint_keys != state_keys:
                raise RuntimeError(
                    "Normal construction changed the official state-key set: "
                    f"missing={sorted(checkpoint_keys - state_keys)}, "
                    f"unexpected={sorted(state_keys - checkpoint_keys)}."
                )
            for name in sorted(checkpoint_keys):
                if not torch.equal(state[name], checkpoint.get_tensor(name)):
                    raise RuntimeError(f"Normal construction changed official tensor {name!r}.")
    print(
        f"Normal construction preserved all {len(state)} official state tensors exactly.",
        flush=True,
    )
    return model


def main(argv: Sequence[str] | None = None) -> int:
    """Invoke only the pinned public loader and fail if parameters remain meta."""

    arguments = _parser().parse_args(argv)
    sources = reference_sources()
    use_esm_submodule()
    from esm.models.esmc import ESMC

    environment = {
        "accelerate": importlib.metadata.version("accelerate"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "safetensors": importlib.metadata.version("safetensors"),
        "torch": torch.__version__,
        "reference_sources": sources,
    }
    print(json.dumps(environment, sort_keys=True), flush=True)
    if arguments.construction == "normal":
        model = _load_with_normal_construction(
            arguments.repo_id,
            arguments.revision,
            arguments.model_name,
        )
    else:
        with pinned_biohub_snapshot(arguments.repo_id, arguments.revision):
            model = ESMC.from_pretrained(
                arguments.model_name,
                device=torch.device("cpu"),
                use_flash_attn=False,
            )
    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(
            "Pinned Biohub public loader left meta parameters: " + ", ".join(meta_parameters[:10])
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    token_ids = model.tokenizer.encode("MSTNPKPQ", add_special_tokens=True)
    # sequence_tokens: (1,)
    sequence_tokens = torch.tensor([token_ids], device=device)
    with torch.inference_mode():
        output = model(sequence_tokens=sequence_tokens)
    for name in ("sequence_logits", "embeddings", "hidden_states"):
        value = getattr(output, name)
        if value is None or not torch.isfinite(value).all():
            raise RuntimeError(f"Official ESMC forward returned invalid {name}.")
    print(
        f"Biohub ESMC {arguments.construction} loader and unmodified forward passed.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
