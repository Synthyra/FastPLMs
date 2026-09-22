"""Dispatch v2 Modal stages with credentials from the authenticated service SDKs."""

from __future__ import annotations

import os


def main() -> None:
    import wandb

    from huggingface_hub import get_token

    if not os.environ.get("HF_TOKEN"):
        token = get_token()
        if not token:
            raise RuntimeError("Authenticate with Hugging Face before launching the campaign")
        os.environ["HF_TOKEN"] = token
    if not os.environ.get("WANDB_API_KEY"):
        key = wandb.Api().api_key
        if not key:
            raise RuntimeError("Authenticate with W&B before launching the campaign")
        os.environ["WANDB_API_KEY"] = key

    from .modal_v2 import main as dispatch

    dispatch()


if __name__ == "__main__":
    main()
