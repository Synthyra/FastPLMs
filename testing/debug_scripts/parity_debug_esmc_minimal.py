"""Minimal ESMC parity — single sequence, no padding, fp32, sdpa.

Answers: does the per-layer divergence go away without padding?
"""
from __future__ import annotations

import random
import torch
from torch.nn.functional import mse_loss

from testing.conftest import CANONICAL_AAS, SEED
from testing.official.esm_plusplus import load_official_model as load_native_esmc


def fast_hidden_states(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    return tuple(out.hidden_states), out.last_hidden_state


def native_hidden_states(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return tuple(out.hidden_states), out.last_hidden_state


def main() -> int:
    device = torch.device("cuda")
    random.seed(SEED)
    torch.manual_seed(SEED)

    seq = "M" + "".join(random.choices(CANONICAL_AAS, k=63))

    from transformers import AutoModelForMaskedLM
    fast = AutoModelForMaskedLM.from_pretrained(
        "Synthyra/ESMplusplus_small", trust_remote_code=True,
        dtype=torch.float32, device_map=device,
    ).eval()

    native, _ = load_native_esmc(reference_repo_id="esmc-300", device=device, dtype=torch.float32)

    enc = fast.tokenizer([seq], return_tensors="pt", padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}

    print(f"seq_len={enc['input_ids'].shape[1]} (no padding)")

    with torch.inference_mode():
        fh, flast = fast_hidden_states(fast, enc["input_ids"], enc["attention_mask"])
        nh, nlast = native_hidden_states(native, enc["input_ids"], enc["attention_mask"])

    print(f"len(fast_hs)={len(fh)} len(native_hs)={len(nh)}")
    n = min(len(fh), len(nh))
    for i in range(n):
        diff = (fh[i] - nh[i]).abs()
        mse = mse_loss(fh[i].float(), nh[i].float()).item()
        print(f"  layer {i:2d}: mse={mse:.3e} maxabs={diff.max().item():.3e}")
    diff = (flast - nlast).abs()
    mse = mse_loss(flast.float(), nlast.float()).item()
    print(f"  last: mse={mse:.3e} maxabs={diff.max().item():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
