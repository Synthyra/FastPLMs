"""Trace where FastAnkh diverges from native HuggingFace T5EncoderModel.

Strategy:
  1. Load both with identical weights, fp32, same input.
  2. Confirm embedding outputs match.
  3. Hook both encoders to capture per-layer hidden states.
  4. Compare layer by layer with both absolute and relative metrics.
  5. Once we find the first diverging layer, dig into that layer's submodules
     (layer_norm output, q/k/v projections, attention output, FFN output)
     by hooking inside the block.

Run inside fastplms-ankh image:
    docker run --rm --gpus all --ipc=host -v $(pwd):/workspace fastplms-ankh \
        python -m testing.parity_debug_ankh
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from testing.conftest import CANONICAL_AAS, SEED


SEQUENCES = ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVK"]


def diff_metrics(fast: torch.Tensor, native: torch.Tensor, tag: str) -> str:
    """Both absolute and relative diff metrics."""
    f = fast.float()
    n = native.float()
    d = f - n
    abs_mse = (d ** 2).mean().item()
    abs_max = d.abs().max().item()
    n_std = n.std().item()
    rel = d.std().item() / max(n_std, 1e-12)
    return f"{tag:50s} abs_mse={abs_mse:.3e} abs_max={abs_max:.3e} rel_std={rel:.3e} | native_mean={n.mean().item():.3e} native_std={n_std:.3e}"


def main() -> int:
    device = torch.device("cuda")
    random.seed(SEED)
    torch.manual_seed(SEED)

    from transformers import AutoModelForMaskedLM, T5EncoderModel, AutoTokenizer
    print("Loading fast model...")
    fast = AutoModelForMaskedLM.from_pretrained(
        "Synthyra/ANKH_base", trust_remote_code=True,
        dtype=torch.float32, device_map=device,
    ).eval()
    print("Loading native T5EncoderModel...")
    native = T5EncoderModel.from_pretrained(
        "ElnaggarLab/ankh-base",
        device_map=device,
        dtype=torch.float32,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")

    # ---- Inputs ----
    enc = tokenizer(SEQUENCES, return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    print(f"\ninput_ids shape={tuple(input_ids.shape)}, mask sum={attention_mask.sum().item()}")

    # ---- Locate fast encoder + native encoder ----
    fast_enc = fast.encoder
    native_enc = native.encoder if hasattr(native, "encoder") else native

    n_blocks_fast = len(fast_enc.block)
    n_blocks_native = len(native_enc.block)
    print(f"fast blocks={n_blocks_fast}, native blocks={n_blocks_native}")
    assert n_blocks_fast == n_blocks_native, "block count mismatch"

    # ---- Hook all hidden states (output of each block + final norm) ----
    fast_hs: List[torch.Tensor] = []
    native_hs: List[torch.Tensor] = []

    def make_block_hook(store: List[torch.Tensor]):
        def hook(_mod, _inp, out):
            # block forward returns (hidden_states, attn_weights, position_bias) for fast
            # and (hidden_states, ...) for native T5Block. Take the first element.
            if isinstance(out, tuple):
                store.append(out[0].detach().clone())
            else:
                store.append(out.detach().clone())
        return hook

    handles = []
    for blk in fast_enc.block:
        handles.append(blk.register_forward_hook(make_block_hook(fast_hs)))
    for blk in native_enc.block:
        handles.append(blk.register_forward_hook(make_block_hook(native_hs)))

    # ---- Forward both ----
    print("\nRunning fast forward...")
    with torch.no_grad():
        fout = fast(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    print(f"fast last_hidden_state shape={tuple(fout.last_hidden_state.shape)}")

    print("Running native forward...")
    with torch.no_grad():
        nout = native(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    print(f"native last_hidden_state shape={tuple(nout.last_hidden_state.shape)}")

    for h in handles:
        h.remove()

    # ---- Per-block diff ----
    print("\n=== Per-block hidden-state diff (output of each block, BEFORE final norm) ===")
    print(f"fast hooked={len(fast_hs)}, native hooked={len(native_hs)}")
    for i in range(min(len(fast_hs), len(native_hs))):
        print(diff_metrics(fast_hs[i], native_hs[i], f"  block {i:02d} output"))

    # ---- Full hidden states tuple from outputs ----
    print("\n=== Per-layer hidden_states from .hidden_states tuple ===")
    fh = fout.hidden_states
    nh = nout.hidden_states
    print(f"fast hs tuple len={len(fh)}, native hs tuple len={len(nh)}")
    for i in range(min(len(fh), len(nh))):
        print(diff_metrics(fh[i], nh[i], f"  hidden_states[{i:02d}]"))

    # ---- Final last_hidden_state ----
    print("\n=== Final last_hidden_state (after final_layer_norm) ===")
    print(diff_metrics(fout.last_hidden_state, nout.last_hidden_state, "  last_hidden_state"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
