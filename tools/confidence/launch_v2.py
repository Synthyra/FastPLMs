"""Dispatch v2 Modal stages with credentials from the authenticated service SDKs."""

from __future__ import annotations

import os
import sys


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

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        from .modal_gpu_benchmark import main as dispatch

        del sys.argv[1]
    elif len(sys.argv) > 1 and sys.argv[1] == "resume":
        from .modal_resume import main as dispatch

        del sys.argv[1]
    elif len(sys.argv) > 1 and sys.argv[1] == "publish-live":
        from .modal_live import main as dispatch

        del sys.argv[1]
    else:
        from .modal_v2 import main as dispatch

    dispatch()


if __name__ == "__main__":
    main()
