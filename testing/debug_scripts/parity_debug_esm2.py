"""Sanity-check parity on ESM2 to confirm large absolute per-layer diffs are
architectural (huge activation magnitudes + LN), not an ESMC-specific bug."""
from __future__ import annotations

import random
import torch

from testing.conftest import CANONICAL_AAS, SEED
from testing.official.esm2 import load_official_model as load_native_esm2


def main() -> int:
    device = torch.device("cuda")
    random.seed(SEED)
    torch.manual_seed(SEED)

    seq = "M" + "".join(random.choices(CANONICAL_AAS, k=63))

    from transformers import AutoModelForMaskedLM
    fast = AutoModelForMaskedLM.from_pretrained(
        "Synthyra/ESM2-8M", trust_remote_code=True,
        dtype=torch.float32, device_map=device,
    ).eval()

    native, tokenizer = load_native_esm2(
        reference_repo_id="facebook/esm2_t6_8M_UR50D",
        device=device, dtype=torch.float32,
    )

    enc = tokenizer([seq], return_tensors="pt", padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        fout = fast(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)
        nout = native(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)

    print(f"fast #hiddens={len(fout.hidden_states)}  native #hiddens={len(nout.hidden_states)}")
    n = min(len(fout.hidden_states), len(nout.hidden_states))
    for i in range(n):
        f = fout.hidden_states[i].float()
        nt = nout.hidden_states[i].float()
        diff = f - nt
        mse = (diff ** 2).mean().item()
        maxabs = diff.abs().max().item()
        native_std = nt.std(dim=-1).mean().item()
        diff_std = diff.std(dim=-1).mean().item()
        rel = diff_std / max(native_std, 1e-12)
        print(f"  layer {i:2d}: mse={mse:.3e} maxabs={maxabs:.3e} "
              f"native_std={native_std:8.3f} rel_diff={rel:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
